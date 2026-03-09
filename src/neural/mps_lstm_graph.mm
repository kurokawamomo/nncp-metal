/*
 * mps_lstm_graph.mm
 *
 * Implementation of MPS Graph LSTM Engine
 */

#import "mps_lstm_graph.h"
#import <Foundation/Foundation.h>

@interface MPSLSTMGraphPackage : NSObject
@property (strong) MPSGraph* graph;
@property (strong) MPSGraphExecutable* executable;
@property (strong) MPSGraphTensor* inputTensor;
@property (strong) MPSGraphTensor* outputTensor;
@property (strong) MPSGraphTensor* outputStateH;
@property (strong) MPSGraphTensor* outputStateC;
@end

@implementation MPSLSTMGraphPackage
@end

struct MPSLSTMContext {
    id<MTLDevice> device;
    MPSLSTMConfig config;
    
    // Cache for graphs/executables keyed by "batch_seq"
    NSMutableDictionary<NSString*, MPSLSTMGraphPackage*>* graphCache;
    
    // Weight Tensor Data Cache
    NSMutableDictionary<NSString*, MPSGraphTensorData*>* weightData;
    
    // State Tensor Data Cache
    MPSGraphTensorData* stateHData;
    MPSGraphTensorData* stateCData;
    
    // Weights (MTLBuffers)
    id<MTLBuffer> w_ih;
    id<MTLBuffer> w_hh;
    id<MTLBuffer> bias;
    
    // State (MTLBuffers) - Persistent across calls
    id<MTLBuffer> state_h;
    id<MTLBuffer> state_c;
};

MPSLSTMContext* mps_lstm_create(id<MTLDevice> device, MPSLSTMConfig config) {
    if (!device) return NULL;
    
    MPSLSTMContext* ctx = new MPSLSTMContext();
    ctx->device = device;
    ctx->config = config;
    ctx->graphCache = [NSMutableDictionary dictionary];
    ctx->weightData = [NSMutableDictionary dictionary];
    
    // Allocate state buffers
    // Size: [Layers, Batch, Hidden]
    size_t state_size = config.num_layers * config.batch_size * config.hidden_size * sizeof(float);
    ctx->state_h = [device newBufferWithLength:state_size options:MTLResourceStorageModePrivate];
    ctx->state_c = [device newBufferWithLength:state_size options:MTLResourceStorageModePrivate];
    
    // Pre-create state wrappers (assuming batch_size is constant as per config)
    ctx->stateHData = [[MPSGraphTensorData alloc] initWithMTLBuffer:ctx->state_h 
                                                              shape:@[@(config.num_layers), @(config.batch_size), @(config.hidden_size)] 
                                                           dataType:MPSDataTypeFloat32];
    ctx->stateCData = [[MPSGraphTensorData alloc] initWithMTLBuffer:ctx->state_c 
                                                              shape:@[@(config.num_layers), @(config.batch_size), @(config.hidden_size)] 
                                                           dataType:MPSDataTypeFloat32];
    
    return ctx;
}

bool mps_lstm_set_weights(MPSLSTMContext* ctx,
                         id<MTLBuffer> w_ih,
                         id<MTLBuffer> w_hh,
                         id<MTLBuffer> bias) {
    if (!ctx) return false;
    ctx->w_ih = w_ih;
    ctx->w_hh = w_hh;
    ctx->bias = bias;
    
    // Create and cache MPSGraphTensorData for weights
    uint32_t L = ctx->config.num_layers;
    uint32_t H = ctx->config.hidden_size;
    uint32_t I = ctx->config.input_size;
    
    // w_ih: [L, 4*H, I]
    ctx->weightData[@"w_ih"] = [[MPSGraphTensorData alloc] initWithMTLBuffer:w_ih 
                                                                       shape:@[@(L), @(4*H), @(I)] 
                                                                    dataType:MPSDataTypeFloat32];
    
    // w_hh: [L, 4*H, H]
    ctx->weightData[@"w_hh"] = [[MPSGraphTensorData alloc] initWithMTLBuffer:w_hh 
                                                                       shape:@[@(L), @(4*H), @(H)] 
                                                                    dataType:MPSDataTypeFloat32];
    
    // bias: [L, 4*H]
    ctx->weightData[@"bias"] = [[MPSGraphTensorData alloc] initWithMTLBuffer:bias 
                                                                       shape:@[@(L), @(4*H)] 
                                                                    dataType:MPSDataTypeFloat32];
    
    return true;
}

void mps_lstm_destroy(MPSLSTMContext* ctx) {
    if (ctx) {
        delete ctx;
    }
}

void mps_lstm_reset_state(MPSLSTMContext* ctx) {
    if (!ctx) return;
    id<MTLCommandQueue> queue = [ctx->device newCommandQueue];
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit fillBuffer:ctx->state_h range:NSMakeRange(0, ctx->state_h.length) value:0];
    [blit fillBuffer:ctx->state_c range:NSMakeRange(0, ctx->state_c.length) value:0];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

// Helper to build LSTM Cell Graph
static void build_lstm_cell(MPSGraph* graph,
                           MPSGraphTensor* x_t, // [Batch, Input] (or Pre-computed [Batch, 4*Hidden])
                           MPSGraphTensor* h_prev, // [Batch, Hidden]
                           MPSGraphTensor* c_prev, // [Batch, Hidden]
                           MPSGraphTensor* w_hh,   // [Hidden, 4*Hidden]
                           MPSGraphTensor* bias,   // [4*Hidden]
                           MPSGraphTensor** h_next,
                           MPSGraphTensor** c_next,
                           uint32_t hidden_size,
                           bool pre_computed_input) {
    
    MPSGraphTensor* gates;
    
    if (pre_computed_input) {
        // x_t is already [Batch, 4*Hidden] (W_ih * x + bias_ih)
        // We just add W_hh * h_prev + bias_hh
        // Assuming bias is combined or handled.
        // Let's assume x_t includes W_ih * x.
        // We need W_hh * h_prev.
        
        MPSGraphTensor* h_gates = [graph matrixMultiplicationWithPrimaryTensor:h_prev secondaryTensor:w_hh name:nil];
        
        // Add bias if provided separately, or assume it's in x_t or h_gates.
        // Usually bias is [4*Hidden].
        MPSGraphTensor* total_bias = bias;
        
        gates = [graph additionWithPrimaryTensor:x_t secondaryTensor:h_gates name:nil];
        gates = [graph additionWithPrimaryTensor:gates secondaryTensor:total_bias name:nil];
    } else {
        // Not implemented (we use pre-computation optimization)
        gates = x_t; 
    }
    
    // Split gates: i, f, g, o
    // gates: [Batch, 4*Hidden]
    NSArray<MPSGraphTensor*>* splits = [graph splitTensor:gates splitSizes:@[@(hidden_size), @(hidden_size), @(hidden_size), @(hidden_size)] axis:-1 name:nil];
    
    MPSGraphTensor* i_gate = [graph sigmoidWithTensor:splits[0] name:nil];
    MPSGraphTensor* f_gate = [graph sigmoidWithTensor:splits[1] name:nil];
    MPSGraphTensor* g_gate = [graph tanhWithTensor:splits[2] name:nil];
    MPSGraphTensor* o_gate = [graph sigmoidWithTensor:splits[3] name:nil];
    
    // c_t = f_t * c_{t-1} + i_t * g_t
    MPSGraphTensor* c_forget = [graph multiplicationWithPrimaryTensor:f_gate secondaryTensor:c_prev name:nil];
    MPSGraphTensor* c_update = [graph multiplicationWithPrimaryTensor:i_gate secondaryTensor:g_gate name:nil];
    *c_next = [graph additionWithPrimaryTensor:c_forget secondaryTensor:c_update name:nil];
    
    // h_t = o_t * tanh(c_t)
    *h_next = [graph multiplicationWithPrimaryTensor:o_gate secondaryTensor:[graph tanhWithTensor:*c_next name:nil] name:nil];
}

bool mps_lstm_execute(MPSLSTMContext* ctx,
                     const float* input_data,
                     size_t batch_size,
                     size_t seq_len,
                     float* output_data) {
    if (!ctx) return false;
    
    // Cache Key based on dynamic shapes
    NSString* key = [NSString stringWithFormat:@"%zu_%zu", batch_size, seq_len];
    MPSLSTMGraphPackage* pkg = ctx->graphCache[key];
    
    // Compile if not cached
    if (!pkg) {
        pkg = [[MPSLSTMGraphPackage alloc] init];
        MPSGraph* graph = [[MPSGraph alloc] init];
        pkg.graph = graph;
        
        uint32_t B = (uint32_t)batch_size;
        uint32_t S = (uint32_t)seq_len;
        uint32_t I = ctx->config.input_size;
        uint32_t H = ctx->config.hidden_size;
        uint32_t L = ctx->config.num_layers;
        
        // Placeholders
        MPSGraphTensor* input = [graph placeholderWithShape:@[@(B), @(S), @(I)] dataType:MPSDataTypeFloat32 name:@"input"];
        pkg.inputTensor = input;
        
        // Weights Placeholders
        MPSGraphTensor* w_ih_all = [graph placeholderWithShape:@[@(L), @(4*H), @(I)] dataType:MPSDataTypeFloat32 name:@"w_ih"];
        MPSGraphTensor* w_hh_all = [graph placeholderWithShape:@[@(L), @(4*H), @(H)] dataType:MPSDataTypeFloat32 name:@"w_hh"];
        MPSGraphTensor* bias_all = [graph placeholderWithShape:@[@(L), @(4*H)] dataType:MPSDataTypeFloat32 name:@"bias"];
        
        // Initial State Placeholders
        MPSGraphTensor* h_init_all = [graph placeholderWithShape:@[@(L), @(B), @(H)] dataType:MPSDataTypeFloat32 name:@"h_init"];
        MPSGraphTensor* c_init_all = [graph placeholderWithShape:@[@(L), @(B), @(H)] dataType:MPSDataTypeFloat32 name:@"c_init"];
        
        // Outputs
        NSMutableArray<MPSGraphTensor*>* final_h_states = [NSMutableArray array];
        NSMutableArray<MPSGraphTensor*>* final_c_states = [NSMutableArray array];
        
        MPSGraphTensor* current_input = input; // [B, S, I]
        
        for (uint32_t l = 0; l < L; l++) {
            // Slice weights
            MPSGraphTensor* w_ih = [graph squeezeTensor:[graph sliceTensor:w_ih_all dimension:0 start:l length:1 name:nil] axis:0 name:nil];
            w_ih = [graph transposeTensor:w_ih dimension:0 withDimension:1 name:nil]; // [I, 4H]
            
            MPSGraphTensor* w_hh = [graph squeezeTensor:[graph sliceTensor:w_hh_all dimension:0 start:l length:1 name:nil] axis:0 name:nil];
            w_hh = [graph transposeTensor:w_hh dimension:0 withDimension:1 name:nil]; // [H, 4H]
            
            MPSGraphTensor* bias = [graph squeezeTensor:[graph sliceTensor:bias_all dimension:0 start:l length:1 name:nil] axis:0 name:nil];
            
            // Initial state for this layer
            MPSGraphTensor* h_prev = [graph squeezeTensor:[graph sliceTensor:h_init_all dimension:0 start:l length:1 name:nil] axis:0 name:nil];
            MPSGraphTensor* c_prev = [graph squeezeTensor:[graph sliceTensor:c_init_all dimension:0 start:l length:1 name:nil] axis:0 name:nil];
            
            // Optimization: Pre-compute Input Projection
            uint32_t dim_in = (l == 0) ? I : H;
            MPSGraphTensor* flat_input = [graph reshapeTensor:current_input withShape:@[@(-1), @(dim_in)] name:nil];
            MPSGraphTensor* gates_x = [graph matrixMultiplicationWithPrimaryTensor:flat_input secondaryTensor:w_ih name:nil];
            gates_x = [graph reshapeTensor:gates_x withShape:@[@(B), @(S), @(4*H)] name:nil];
            
            // Unroll loop over sequence
            NSMutableArray* h_outputs = [NSMutableArray arrayWithCapacity:S];
            
            for (uint32_t t = 0; t < S; t++) {
                MPSGraphTensor* x_t = [graph sliceTensor:gates_x dimension:1 start:t length:1 name:nil];
                x_t = [graph squeezeTensor:x_t axis:1 name:nil]; // [B, 4H]
                
                MPSGraphTensor *h_next, *c_next;
                build_lstm_cell(graph, x_t, h_prev, c_prev, w_hh, bias, &h_next, &c_next, H, true);
                
                [h_outputs addObject:h_next];
                h_prev = h_next;
                c_prev = c_next;
            }
            
            // Stack outputs: [B, S, H]
            MPSGraphTensor* layer_out = [graph stackTensors:h_outputs axis:1 name:nil];
            current_input = layer_out;
            
            [final_h_states addObject:h_prev];
            [final_c_states addObject:c_prev];
        }
        
        MPSGraphTensor* final_h_stacked = [graph stackTensors:final_h_states axis:0 name:nil];
        MPSGraphTensor* final_c_stacked = [graph stackTensors:final_c_states axis:0 name:nil];
        
        pkg.outputTensor = current_input;
        pkg.outputStateH = final_h_stacked;
        pkg.outputStateC = final_c_stacked;
        
        // Compile
        pkg.executable = [graph compileWithDevice:ctx->device
                                           feeds:@{
                                               @"input": input,
                                               @"w_ih": w_ih_all,
                                               @"w_hh": w_hh_all,
                                               @"bias": bias_all,
                                               @"h_init": h_init_all,
                                               @"c_init": c_init_all
                                           }
                                   targetTensors:@[current_input, final_h_stacked, final_c_stacked]
                                targetOperations:nil
                           compilationDescriptor:nil];
                           
        ctx->graphCache[key] = pkg;
    }
    
    // Prepare State Data
    // Use cached wrappers if batch size matches config, otherwise create temp wrappers
    MPSGraphTensorData* currentStateH = ctx->stateHData;
    MPSGraphTensorData* currentStateC = ctx->stateCData;
    
    if (batch_size != ctx->config.batch_size) {
        currentStateH = [[MPSGraphTensorData alloc] initWithMTLBuffer:ctx->state_h
                                                                shape:@[@(ctx->config.num_layers), @(batch_size), @(ctx->config.hidden_size)]
                                                             dataType:MPSDataTypeFloat32];
        currentStateC = [[MPSGraphTensorData alloc] initWithMTLBuffer:ctx->state_c
                                                                shape:@[@(ctx->config.num_layers), @(batch_size), @(ctx->config.hidden_size)]
                                                             dataType:MPSDataTypeFloat32];
    }
    
    // Input Data (Always new wrapper as data pointer changes)
    MPSGraphTensorData* inputTD = [[MPSGraphTensorData alloc] initWithDevice:ctx->device
                                                                        data:[NSData dataWithBytes:input_data length:batch_size*seq_len*ctx->config.input_size*sizeof(float)]
                                                                       shape:@[@(batch_size), @(seq_len), @(ctx->config.input_size)]
                                                                    dataType:MPSDataTypeFloat32];
    
    NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
    feeds[@"input"] = inputTD;
    feeds[@"w_ih"] = ctx->weightData[@"w_ih"];
    feeds[@"w_hh"] = ctx->weightData[@"w_hh"];
    feeds[@"bias"] = ctx->weightData[@"bias"];
    feeds[@"h_init"] = currentStateH;
    feeds[@"c_init"] = currentStateC;
    
    // Run
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [pkg.executable runWithFeeds:feeds
                                                            targetTensors:@[pkg.outputTensor, pkg.outputStateH, pkg.outputStateC]
                                                         targetOperations:nil];
    
    // Read Output
    MPSGraphTensorData* resOut = results[pkg.outputTensor];
    [resOut.mpsndarray readBytes:output_data strideBytes:NULL];
    
    // Update State Buffers (Read back to state buffers)
    MPSGraphTensorData* resH = results[pkg.outputStateH];
    [resH.mpsndarray readBytes:[ctx->state_h contents] strideBytes:NULL];
    
    MPSGraphTensorData* resC = results[pkg.outputStateC];
    [resC.mpsndarray readBytes:[ctx->state_c contents] strideBytes:NULL];
    
    return true;
}
