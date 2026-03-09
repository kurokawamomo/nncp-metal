/*
 * gpu_native_lstm.mm
 *
 * Implementation of GPU-Native LSTM Engine
 * Optimized for Apple Silicon
 */

#import "gpu_native_lstm.h"
#import <Foundation/Foundation.h>

struct GPULSTMContext {
    id<MTLDevice> device;
    UnifiedMemoryManager* mem_mgr;
    SyncOptimizer* sync_opt; // Added SyncOptimizer
    GPULSTMConfig config;
    id<MTLCommandQueue> cmd_queue;
    id<MTLLibrary> library;
    
    // Pipelines
    id<MTLComputePipelineState> pipeline_mv; // Matrix-Vector (for recurrent)
    id<MTLComputePipelineState> pipeline_mm; // Matrix-Matrix (for input projection)
    id<MTLComputePipelineState> pipeline_lstm_cell; // Element-wise gates update
    
    // Weights
    id<MTLBuffer> w_ih;
    id<MTLBuffer> w_hh;
    id<MTLBuffer> bias;
    
    // State Buffers (resident on GPU)
    // [layers, batch, hidden]
    id<MTLBuffer> state_h;
    id<MTLBuffer> state_c;
    
    bool is_ready;
};

extern "C" {

// Update create to accept SyncOptimizer via setter or assume internal creation
GPULSTMContext* gpu_lstm_create(id<MTLDevice> device, UnifiedMemoryManager* mem_mgr, GPULSTMConfig config) {
    if (!device || !mem_mgr) return NULL;
    
    GPULSTMContext* ctx = new GPULSTMContext();
    ctx->device = device;
    ctx->mem_mgr = mem_mgr;
    ctx->config = config;
    ctx->cmd_queue = [device newCommandQueue];
    ctx->sync_opt = NULL;
    
    NSError* error = nil;
    // Try default library first
    ctx->library = [device newDefaultLibrary];
    if (!ctx->library) {
        NSURL *libURL = [NSURL fileURLWithPath:@"default.metallib"];
        ctx->library = [device newLibraryWithURL:libURL error:&error];
    }
    
    if (!ctx->library) {
        NSString* execPath = [[NSBundle mainBundle] executablePath];
        if (execPath) {
             NSString* folder = [execPath stringByDeletingLastPathComponent];
             NSURL* libURL = [NSURL fileURLWithPath:[folder stringByAppendingPathComponent:@"default.metallib"]];
             ctx->library = [device newLibraryWithURL:libURL error:&error];
        }
    }

    if (!ctx->library) {
        // Fallback to runtime compilation from source
        NSString* sourcePath = @"/Users/shigerukurokawa/Codes/nncp/src/metal/compute/neural_net.metal";
        NSError* sourceError = nil;
        NSString* source = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&sourceError];
        
        if (source) {
            ctx->library = [device newLibraryWithSource:source options:nil error:&error];
        } else {
            printf("Error reading Metal source: %s\n", [[sourceError localizedDescription] UTF8String]);
        }
    }

    if (!ctx->library) {
        printf("Error loading Metal library: %s\n", error ? [[error localizedDescription] UTF8String] : "Unknown error");
        delete ctx;
        return NULL;
    }
    
    auto createPipeline = [&](const char* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> func = [ctx->library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!func) return nil;
        return [device newComputePipelineStateWithFunction:func error:nil];
    };
    
    ctx->pipeline_mm = createPipeline("matrix_multiply"); // Generic MM
    ctx->pipeline_mv = createPipeline("matrix_vector_multiply"); // Generic MV
    ctx->pipeline_lstm_cell = createPipeline("lstm_cell_update"); // Specific
    
    if (!ctx->pipeline_mm || !ctx->pipeline_lstm_cell) {
        printf("Warning: LSTM pipelines missing. Ensure kernels are compiled.\n");
    }
    
    // Allocate state buffers
    size_t state_size = config.num_layers * config.batch_size * config.hidden_size * sizeof(float);
    ctx->state_h = [device newBufferWithLength:state_size options:MTLResourceStorageModePrivate];
    ctx->state_c = [device newBufferWithLength:state_size options:MTLResourceStorageModePrivate];
    
    ctx->is_ready = true;
    return ctx;
}

void gpu_lstm_set_sync_optimizer(GPULSTMContext* ctx, SyncOptimizer* sync_opt) {
    if (ctx) ctx->sync_opt = sync_opt;
}

bool gpu_lstm_set_weights(GPULSTMContext* ctx,
                         id<MTLBuffer> w_ih,
                         id<MTLBuffer> w_hh,
                         id<MTLBuffer> bias) {
    if (!ctx) return false;
    ctx->w_ih = w_ih;
    ctx->w_hh = w_hh;
    ctx->bias = bias;
    return true;
}

void gpu_lstm_reset_state(GPULSTMContext* ctx) {
    if (!ctx) return;
    id<MTLCommandBuffer> cmd = [ctx->cmd_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit fillBuffer:ctx->state_h range:NSMakeRange(0, ctx->state_h.length) value:0];
    [blit fillBuffer:ctx->state_c range:NSMakeRange(0, ctx->state_c.length) value:0];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void gpu_lstm_destroy(GPULSTMContext* ctx) {
    if (ctx) {
        delete ctx;
    }
}

// Internal encoding function
bool gpu_lstm_encode_sequence(GPULSTMContext* ctx,
                              id<MTLCommandBuffer> cmd,
                              const float* input,
                              size_t batch_size,
                              size_t seq_len,
                              float* output) {
    if (!ctx || !ctx->is_ready || !cmd) return false;
    
    // 1. Copy Input to GPU (Unified Memory)
    size_t input_bytes = batch_size * seq_len * ctx->config.input_size * sizeof(float);
    void* raw_input = unified_memory_alloc(ctx->mem_mgr, input_bytes, UNIFIED_POOL_TEMPORARY, "lstm_input");
    if (!raw_input) return false;
    memcpy(raw_input, input, input_bytes);
    
    id<MTLBuffer> buf_input = unified_memory_get_buffer(ctx->mem_mgr, raw_input);
    size_t off_input = unified_memory_get_offset(ctx->mem_mgr, raw_input);
    
    // 2. Allocate Output Buffer
    size_t output_bytes = batch_size * seq_len * ctx->config.hidden_size * sizeof(float);
    void* raw_output = unified_memory_alloc(ctx->mem_mgr, output_bytes, UNIFIED_POOL_TEMPORARY, "lstm_output");
    id<MTLBuffer> buf_output = unified_memory_get_buffer(ctx->mem_mgr, raw_output);
    size_t off_output = unified_memory_get_offset(ctx->mem_mgr, raw_output);
    
    // Intermediate buffers
    id<MTLBuffer> buf_layer_in = buf_input;
    size_t off_layer_in = off_input;
    
    id<MTLBuffer> buf_layer_out = buf_output; 
    size_t off_layer_out = off_output;
    
    // Temp buffer for intermediate layers
    void* raw_temp = NULL;
    id<MTLBuffer> buf_temp = nil;
    size_t off_temp = 0;
    
    if (ctx->config.num_layers > 1) {
        raw_temp = unified_memory_alloc(ctx->mem_mgr, output_bytes, UNIFIED_POOL_TEMPORARY, "lstm_temp");
        buf_temp = unified_memory_get_buffer(ctx->mem_mgr, raw_temp);
        off_temp = unified_memory_get_offset(ctx->mem_mgr, raw_temp);
    }
    
    // Pre-computed gates buffer
    size_t gates_size = batch_size * seq_len * 4 * ctx->config.hidden_size * sizeof(float);
    void* raw_gates = unified_memory_alloc(ctx->mem_mgr, gates_size, UNIFIED_POOL_TEMPORARY, "lstm_gates");
    id<MTLBuffer> buf_gates_pre = unified_memory_get_buffer(ctx->mem_mgr, raw_gates);
    size_t off_gates_pre = unified_memory_get_offset(ctx->mem_mgr, raw_gates);
    
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    for (uint32_t l = 0; l < ctx->config.num_layers; l++) {
        // Determine input/output buffers
        id<MTLBuffer> curr_in = (l == 0) ? buf_input : buf_layer_out;
        size_t off_curr_in = (l == 0) ? off_input : off_layer_out;
        
        id<MTLBuffer> curr_out = (l == ctx->config.num_layers - 1) ? buf_output : buf_temp;
        size_t off_curr_out = (l == ctx->config.num_layers - 1) ? off_output : off_temp;
        
        // Swap logic for > 2 layers (simplified: just use temp and output)
        if (l > 0 && l < ctx->config.num_layers - 1) {
            if (l % 2 == 1) { curr_out = buf_output; off_curr_out = off_output; }
            else { curr_out = buf_temp; off_curr_out = off_temp; }
        }
        
        uint32_t input_dim = (l == 0) ? ctx->config.input_size : ctx->config.hidden_size;
        uint32_t hidden_dim = ctx->config.hidden_size;
        
        // Step 1: Parallel Input Projection
        [enc setComputePipelineState:ctx->pipeline_mm];
        [enc setBuffer:curr_in offset:off_curr_in atIndex:0];
        [enc setBuffer:ctx->w_ih offset:l * input_dim * 4 * hidden_dim * sizeof(float) atIndex:1];
        [enc setBuffer:buf_gates_pre offset:off_gates_pre atIndex:2];
        
        uint32_t M = (uint32_t)(batch_size * seq_len);
        uint32_t N = 4 * hidden_dim;
        uint32_t K = input_dim;
        
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
        
        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize groupSize = MTLSizeMake(32, 32, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        
        // Step 2: Sequential Recurrence
        for (uint32_t t = 0; t < seq_len; t++) {
            [enc setComputePipelineState:ctx->pipeline_lstm_cell];
            
            // Inputs
            [enc setBuffer:buf_gates_pre offset:off_gates_pre + t * batch_size * 4 * hidden_dim * sizeof(float) atIndex:0];
            [enc setBuffer:ctx->state_h offset:l * batch_size * hidden_dim * sizeof(float) atIndex:1];
            [enc setBuffer:ctx->state_c offset:l * batch_size * hidden_dim * sizeof(float) atIndex:2];
            
            // Weights
            [enc setBuffer:ctx->w_hh offset:l * hidden_dim * 4 * hidden_dim * sizeof(float) atIndex:3];
            [enc setBuffer:ctx->bias offset:l * 4 * hidden_dim * sizeof(float) atIndex:4];
            
            // Outputs
            [enc setBuffer:curr_out offset:off_curr_out + t * batch_size * hidden_dim * sizeof(float) atIndex:5];
            
            [enc setBytes:&hidden_dim length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:7];
            
            MTLSize cellGrid = MTLSizeMake(hidden_dim, batch_size, 1);
            [enc dispatchThreads:cellGrid threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        
        buf_layer_out = curr_out;
        off_layer_out = off_curr_out;
    }
    
    [enc endEncoding];
    
    UnifiedMemoryManager* mem_mgr = ctx->mem_mgr;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
        memcpy(output, raw_output, output_bytes);
        
        unified_memory_free(mem_mgr, raw_input);
        unified_memory_free(mem_mgr, raw_output);
        if (raw_temp) unified_memory_free(mem_mgr, raw_temp);
        unified_memory_free(mem_mgr, raw_gates);
    }];
    
    return true;
}

bool gpu_lstm_predict_sequence(GPULSTMContext* ctx,
                              const float* input,
                              size_t batch_size,
                              size_t seq_len,
                              float* output) {
    if (!ctx) return false;
    
    id<MTLCommandBuffer> cmd = nil;
    if (ctx->sync_opt) {
        cmd = ctx->sync_opt->getCommandBuffer();
    } else {
        cmd = [ctx->cmd_queue commandBuffer];
    }
    
    if (!gpu_lstm_encode_sequence(ctx, cmd, input, batch_size, seq_len, output)) {
        return false;
    }
    
    if (ctx->sync_opt) {
        ctx->sync_opt->commitCommandBuffer(cmd, false);
    } else {
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    
    return true;
}

} // extern "C"
