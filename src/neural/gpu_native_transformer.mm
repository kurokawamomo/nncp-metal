/*
 * gpu_native_transformer.mm
 *
 * Implementation of GPU-Native Transformer Engine
 * Optimized for Apple Silicon (M1/M2/M3) with Unified Memory
 */

#import "gpu_native_transformer.h"
#import <Foundation/Foundation.h>
#include <vector>

struct GPUTransformerContext {
    id<MTLDevice> device;
    UnifiedMemoryManager* mem_mgr;
    SyncOptimizer* sync_opt;
    GPUTransformerConfig config;
    id<MTLLibrary> library;
    
    // Pipelines
    id<MTLComputePipelineState> pipeline_embed;
    id<MTLComputePipelineState> pipeline_norm;
    id<MTLComputePipelineState> pipeline_linear;
    id<MTLComputePipelineState> pipeline_attn_score;
    id<MTLComputePipelineState> pipeline_attn_value;
    id<MTLComputePipelineState> pipeline_geglu;
    id<MTLComputePipelineState> pipeline_add;
    
    // Weights (Monolithic buffers)
    id<MTLBuffer> w_embed;
    id<MTLBuffer> w_pos;
    id<MTLBuffer> w_attn_q;
    id<MTLBuffer> w_attn_k;
    id<MTLBuffer> w_attn_v;
    id<MTLBuffer> w_attn_out;
    id<MTLBuffer> w_ffn_1;
    id<MTLBuffer> w_ffn_2;
    id<MTLBuffer> w_ln;
    id<MTLBuffer> w_final_ln;
    id<MTLBuffer> w_out_proj;
    
    bool is_ready;
    
    // Semaphore set by gpu_transformer_encode_batch; waited on by predict_batch
    // to ensure the completion handler (which copies results) has fully run.
    dispatch_semaphore_t _completion_sem;
};

extern "C" {

GPUTransformerContext* gpu_transformer_create(id<MTLDevice> device, UnifiedMemoryManager* mem_mgr, GPUTransformerConfig config) {
    if (!device || !mem_mgr) return NULL;
    
    GPUTransformerContext* ctx = new GPUTransformerContext();
    ctx->device = device;
    ctx->mem_mgr = mem_mgr;
    ctx->config = config;
    ctx->sync_opt = NULL; 
    
    NSError* error = nil;
    // Try to load default library
    ctx->library = [device newDefaultLibrary];
    if (!ctx->library) {
        NSURL *libURL = [NSURL fileURLWithPath:@"default.metallib"];
        ctx->library = [device newLibraryWithURL:libURL error:&error];
    }
    
    if (!ctx->library) {
        // Fallback to checking executable path
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
        if (!func) {
            printf("Error: Function '%s' not found in library\n", name);
            return nil;
        }
        return [device newComputePipelineStateWithFunction:func error:nil];
    };
    
    ctx->pipeline_embed = createPipeline("transformer_embedding_lookup");
    ctx->pipeline_norm = createPipeline("transformer_layer_norm");
    ctx->pipeline_linear = createPipeline("transformer_linear");
    ctx->pipeline_attn_score = createPipeline("transformer_attention_score");
    ctx->pipeline_attn_value = createPipeline("transformer_attention_value");
    ctx->pipeline_geglu = createPipeline("transformer_geglu");
    ctx->pipeline_add = createPipeline("element_add");
    
    if (!ctx->pipeline_embed || !ctx->pipeline_norm || !ctx->pipeline_linear ||
        !ctx->pipeline_attn_score || !ctx->pipeline_attn_value || !ctx->pipeline_geglu || !ctx->pipeline_add) {
        printf("Error: Failed to create one or more pipelines\n");
        delete ctx;
        return NULL;
    }
    
    ctx->is_ready = true;
    return ctx;
}

// New function to set SyncOptimizer
void gpu_transformer_set_sync_optimizer(GPUTransformerContext* ctx, SyncOptimizer* sync_opt) {
    if (ctx) ctx->sync_opt = sync_opt;
}

bool gpu_transformer_set_weights(GPUTransformerContext* ctx, 
                                id<MTLBuffer> embed,
                                id<MTLBuffer> pos_embed,
                                id<MTLBuffer> attn_q,
                                id<MTLBuffer> attn_k,
                                id<MTLBuffer> attn_v,
                                id<MTLBuffer> attn_out,
                                id<MTLBuffer> ffn_1,
                                id<MTLBuffer> ffn_2,
                                id<MTLBuffer> ln_weights,
                                id<MTLBuffer> final_ln_weights,
                                id<MTLBuffer> out_proj) {
    if (!ctx) return false;
    
    ctx->w_embed = embed;
    ctx->w_pos = pos_embed;
    ctx->w_attn_q = attn_q;
    ctx->w_attn_k = attn_k;
    ctx->w_attn_v = attn_v;
    ctx->w_attn_out = attn_out;
    ctx->w_ffn_1 = ffn_1;
    ctx->w_ffn_2 = ffn_2;
    ctx->w_ln = ln_weights;
    ctx->w_out_proj = out_proj;
    
    return true;
}

void gpu_transformer_destroy(GPUTransformerContext* ctx) {
    if (ctx) {
        delete ctx;
    }
}

// Helper for linear layer dispatch
static void dispatch_linear(id<MTLComputeCommandEncoder> encoder,
                           id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> input, size_t input_offset,
                           id<MTLBuffer> weight, size_t weight_offset,
                           id<MTLBuffer> output, size_t output_offset,
                           uint32_t batch_seq,
                           uint32_t in_dim,
                           uint32_t out_dim) {
    
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:input_offset atIndex:0];
    [encoder setBuffer:weight offset:weight_offset atIndex:1];
    // Bias placeholder (null/zero)
    [encoder setBuffer:input offset:input_offset atIndex:2]; // Dummy bind
    [encoder setBuffer:output offset:output_offset atIndex:3];
    [encoder setBytes:&in_dim length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&out_dim length:sizeof(uint32_t) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(out_dim, batch_seq, 1);
    MTLSize threadGroupSize = MTLSizeMake(MIN(out_dim, 32), MIN(batch_seq, 32), 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
}

// Internal encoding function
bool gpu_transformer_encode_batch(GPUTransformerContext* ctx, 
                                  id<MTLCommandBuffer> cmd,
                                  const int32_t* input_tokens, 
                                  size_t batch_size,
                                  size_t seq_len,
                                  float* output_logits) {
    if (!ctx || !ctx->is_ready || !cmd) return false;
    
    size_t input_count = batch_size * seq_len;
    size_t d_model = ctx->config.hidden_size;
    size_t d_head = d_model / ctx->config.num_heads;
    size_t ffn_size = ctx->config.ffn_size;
    
    // 1. Allocate Temporary Buffers using UnifiedMemoryManager
    // Note: We use UNIFIED_POOL_TEMPORARY which should be fast.
    
    // Input copy (Zero-copy optimization: if input_tokens is already in unified memory, we could skip this)
    // For now, alloc and copy.
    void* raw_input = unified_memory_alloc(ctx->mem_mgr, input_count * sizeof(int32_t), UNIFIED_POOL_TEMPORARY, "input_tokens");
    if (!raw_input) return false;
    memcpy(raw_input, input_tokens, input_count * sizeof(int32_t));
    id<MTLBuffer> buf_input = unified_memory_get_buffer(ctx->mem_mgr, raw_input);
    size_t off_input = unified_memory_get_offset(ctx->mem_mgr, raw_input);
    
    size_t embed_bytes = input_count * d_model * sizeof(float);
    void* raw_embed = unified_memory_alloc(ctx->mem_mgr, embed_bytes, UNIFIED_POOL_TEMPORARY, "embed");
    void* raw_residual = unified_memory_alloc(ctx->mem_mgr, embed_bytes, UNIFIED_POOL_TEMPORARY, "residual");
    
    void* raw_q = unified_memory_alloc(ctx->mem_mgr, embed_bytes, UNIFIED_POOL_TEMPORARY, "q");
    void* raw_k = unified_memory_alloc(ctx->mem_mgr, embed_bytes, UNIFIED_POOL_TEMPORARY, "k");
    void* raw_v = unified_memory_alloc(ctx->mem_mgr, embed_bytes, UNIFIED_POOL_TEMPORARY, "v");
    
    size_t score_bytes = batch_size * ctx->config.num_heads * seq_len * seq_len * sizeof(float);
    void* raw_scores = unified_memory_alloc(ctx->mem_mgr, score_bytes, UNIFIED_POOL_TEMPORARY, "scores");
    
    size_t ffn_inter_bytes = input_count * ffn_size * 2 * sizeof(float);
    void* raw_ffn_inter = unified_memory_alloc(ctx->mem_mgr, ffn_inter_bytes, UNIFIED_POOL_TEMPORARY, "ffn_inter");
    void* raw_ffn_out = unified_memory_alloc(ctx->mem_mgr, embed_bytes, UNIFIED_POOL_TEMPORARY, "ffn_out");
    
    size_t out_bytes = input_count * ctx->config.vocab_size * sizeof(float);
    void* raw_logits = unified_memory_alloc(ctx->mem_mgr, out_bytes, UNIFIED_POOL_TEMPORARY, "logits");
    
    // Get Metal buffers and offsets
    id<MTLBuffer> buf_embed = unified_memory_get_buffer(ctx->mem_mgr, raw_embed);
    size_t off_embed = unified_memory_get_offset(ctx->mem_mgr, raw_embed);
    
    id<MTLBuffer> buf_residual = unified_memory_get_buffer(ctx->mem_mgr, raw_residual);
    size_t off_residual = unified_memory_get_offset(ctx->mem_mgr, raw_residual);
    
    id<MTLBuffer> buf_q = unified_memory_get_buffer(ctx->mem_mgr, raw_q);
    size_t off_q = unified_memory_get_offset(ctx->mem_mgr, raw_q);
    
    id<MTLBuffer> buf_k = unified_memory_get_buffer(ctx->mem_mgr, raw_k);
    size_t off_k = unified_memory_get_offset(ctx->mem_mgr, raw_k);
    
    id<MTLBuffer> buf_v = unified_memory_get_buffer(ctx->mem_mgr, raw_v);
    size_t off_v = unified_memory_get_offset(ctx->mem_mgr, raw_v);
    
    id<MTLBuffer> buf_scores = unified_memory_get_buffer(ctx->mem_mgr, raw_scores);
    size_t off_scores = unified_memory_get_offset(ctx->mem_mgr, raw_scores);
    
    id<MTLBuffer> buf_ffn_inter = unified_memory_get_buffer(ctx->mem_mgr, raw_ffn_inter);
    size_t off_ffn_inter = unified_memory_get_offset(ctx->mem_mgr, raw_ffn_inter);
    
    id<MTLBuffer> buf_ffn_out = unified_memory_get_buffer(ctx->mem_mgr, raw_ffn_out);
    size_t off_ffn_out = unified_memory_get_offset(ctx->mem_mgr, raw_ffn_out);
    
    id<MTLBuffer> buf_logits = unified_memory_get_buffer(ctx->mem_mgr, raw_logits);
    size_t off_logits = unified_memory_get_offset(ctx->mem_mgr, raw_logits);
    
    // 2. Command Encoding
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    // --- Embedding ---
    [enc setComputePipelineState:ctx->pipeline_embed];
    [enc setBuffer:buf_input offset:off_input atIndex:0];
    [enc setBuffer:ctx->w_embed offset:0 atIndex:1];
    [enc setBuffer:buf_embed offset:off_embed atIndex:2];
    [enc setBytes:&d_model length:sizeof(uint32_t) atIndex:3];
    uint32_t vocab = ctx->config.vocab_size;
    [enc setBytes:&vocab length:sizeof(uint32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(input_count, d_model, 1) threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    
    // Positional Embedding
    [enc setComputePipelineState:ctx->pipeline_add];
    [enc setBuffer:buf_embed offset:off_embed atIndex:0];
    [enc setBuffer:ctx->w_pos offset:0 atIndex:1];
    [enc setBuffer:buf_embed offset:off_embed atIndex:2];
    uint32_t total_elems = input_count * d_model;
    [enc setBytes:&total_elems length:sizeof(uint32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(total_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    
    // --- Layers ---
    for (uint32_t l = 0; l < ctx->config.num_layers; l++) {
        // LN 1
        [enc setComputePipelineState:ctx->pipeline_norm];
        [enc setBuffer:buf_embed offset:off_embed atIndex:0];
        [enc setBuffer:buf_residual offset:off_residual atIndex:1]; // Output to residual
        uint32_t ln_offset = l * 2 * d_model * sizeof(float);
        [enc setBuffer:ctx->w_ln offset:ln_offset atIndex:2];
        [enc setBuffer:ctx->w_ln offset:ln_offset + d_model*sizeof(float) atIndex:3];
        [enc setBytes:&d_model length:sizeof(uint32_t) atIndex:4];
        float eps = 1e-5;
        [enc setBytes:&eps length:sizeof(float) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(input_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)32, input_count), 1, 1)];
        
        // QKV
        uint32_t w_stride = d_model * d_model * sizeof(float);
        dispatch_linear(enc, ctx->pipeline_linear, buf_residual, off_residual, ctx->w_attn_q, l * w_stride, buf_q, off_q, input_count, d_model, d_model);
        dispatch_linear(enc, ctx->pipeline_linear, buf_residual, off_residual, ctx->w_attn_k, l * w_stride, buf_k, off_k, input_count, d_model, d_model);
        dispatch_linear(enc, ctx->pipeline_linear, buf_residual, off_residual, ctx->w_attn_v, l * w_stride, buf_v, off_v, input_count, d_model, d_model);
        
        // Attention Score
        [enc setComputePipelineState:ctx->pipeline_attn_score];
        [enc setBuffer:buf_q offset:off_q atIndex:0];
        [enc setBuffer:buf_k offset:off_k atIndex:1];
        [enc setBuffer:buf_scores offset:off_scores atIndex:2];
        uint32_t sl = (uint32_t)seq_len;
        uint32_t nh = ctx->config.num_heads;
        uint32_t hd = (uint32_t)d_head;
        float scale = 1.0f / sqrtf((float)d_head);
        [enc setBytes:&sl length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&nh length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&hd length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&scale length:sizeof(float) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(sl, nh, batch_size) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        
        // Attention Value
        [enc setComputePipelineState:ctx->pipeline_attn_value];
        [enc setBuffer:buf_scores offset:off_scores atIndex:0];
        [enc setBuffer:buf_v offset:off_v atIndex:1];
        [enc setBuffer:buf_residual offset:off_residual atIndex:2]; // Reuse residual as Attn Output
        [enc setBytes:&sl length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&nh length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&hd length:sizeof(uint32_t) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(sl, nh, batch_size) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        
        // Output Projection
        dispatch_linear(enc, ctx->pipeline_linear, buf_residual, off_residual, ctx->w_attn_out, l * w_stride, buf_ffn_out, off_ffn_out, input_count, d_model, d_model);
        
        // Residual Add 1
        [enc setComputePipelineState:ctx->pipeline_add];
        [enc setBuffer:buf_embed offset:off_embed atIndex:0];
        [enc setBuffer:buf_ffn_out offset:off_ffn_out atIndex:1];
        [enc setBuffer:buf_embed offset:off_embed atIndex:2];
        [enc setBytes:&total_elems length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        
        // LN 2
        [enc setComputePipelineState:ctx->pipeline_norm];
        [enc setBuffer:buf_embed offset:off_embed atIndex:0];
        [enc setBuffer:buf_residual offset:off_residual atIndex:1];
        [enc setBuffer:ctx->w_ln offset:ln_offset atIndex:2];
        [enc setBuffer:ctx->w_ln offset:ln_offset + d_model*sizeof(float) atIndex:3];
        [enc setBytes:&d_model length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&eps length:sizeof(float) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(input_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)32, input_count), 1, 1)];
        
        // FFN 1
        uint32_t ffn1_stride = d_model * ffn_size * 2 * sizeof(float);
        dispatch_linear(enc, ctx->pipeline_linear, buf_residual, off_residual, ctx->w_ffn_1, l * ffn1_stride, buf_ffn_inter, off_ffn_inter, input_count, d_model, ffn_size * 2);
        
        // GEGLU
        [enc setComputePipelineState:ctx->pipeline_geglu];
        [enc setBuffer:buf_ffn_inter offset:off_ffn_inter atIndex:0];
        [enc setBuffer:buf_ffn_inter offset:off_ffn_inter atIndex:1];
        uint32_t inter_dim = (uint32_t)ffn_size;
        [enc setBytes:&inter_dim length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(inter_dim, input_count, 1) threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
        
        // FFN 2
        uint32_t ffn2_stride = ffn_size * d_model * sizeof(float);
        dispatch_linear(enc, ctx->pipeline_linear, buf_ffn_inter, off_ffn_inter, ctx->w_ffn_2, l * ffn2_stride, buf_ffn_out, off_ffn_out, input_count, ffn_size, d_model);
        
        // Residual Add 2
        [enc setComputePipelineState:ctx->pipeline_add];
        [enc setBuffer:buf_embed offset:off_embed atIndex:0];
        [enc setBuffer:buf_ffn_out offset:off_ffn_out atIndex:1];
        [enc setBuffer:buf_embed offset:off_embed atIndex:2];
        [enc setBytes:&total_elems length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(total_elems, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    
    // Final LN
    [enc setComputePipelineState:ctx->pipeline_norm];
    [enc setBuffer:buf_embed offset:off_embed atIndex:0];
    [enc setBuffer:buf_embed offset:off_embed atIndex:1];
    [enc setBuffer:ctx->w_final_ln ? ctx->w_final_ln : ctx->w_ln offset:0 atIndex:2];
    [enc setBuffer:ctx->w_final_ln ? ctx->w_final_ln : ctx->w_ln offset:d_model*sizeof(float) atIndex:3];
    [enc setBytes:&d_model length:sizeof(uint32_t) atIndex:4];
    float eps = 1e-5;
    [enc setBytes:&eps length:sizeof(float) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(input_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(MIN((size_t)32, input_count), 1, 1)];
    
    // Output Projection
    dispatch_linear(enc, ctx->pipeline_linear, buf_embed, off_embed, ctx->w_out_proj, 0, buf_logits, off_logits, input_count, d_model, ctx->config.vocab_size);
    
    [enc endEncoding];
    
    // Use a semaphore to ensure the completion handler (which copies raw_logits →
    // output_logits) has fully run before this function returns.
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    UnifiedMemoryManager* mem_mgr = ctx->mem_mgr;
    
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
        // GPU is done; raw_logits now contains valid data. Copy to caller's buffer.
        memcpy(output_logits, raw_logits, out_bytes);
        
        unified_memory_free(mem_mgr, raw_input);
        unified_memory_free(mem_mgr, raw_embed);
        unified_memory_free(mem_mgr, raw_residual);
        unified_memory_free(mem_mgr, raw_q);
        unified_memory_free(mem_mgr, raw_k);
        unified_memory_free(mem_mgr, raw_v);
        unified_memory_free(mem_mgr, raw_scores);
        unified_memory_free(mem_mgr, raw_ffn_inter);
        unified_memory_free(mem_mgr, raw_ffn_out);
        unified_memory_free(mem_mgr, raw_logits);
        
        dispatch_semaphore_signal(sem);
    }];
    
    // Store semaphore for gpu_transformer_predict_batch to wait on after commit.
    ctx->_completion_sem = sem;
    
    return true;
}

bool gpu_transformer_predict_batch(GPUTransformerContext* ctx, 
                                  const int32_t* input_tokens, 
                                  size_t batch_size,
                                  size_t seq_len,
                                  float* output_logits) {
    if (!ctx) return false;
    
    // Use SyncOptimizer if available, else default queue
    id<MTLCommandBuffer> cmd = nil;
    if (ctx->sync_opt) {
        cmd = ctx->sync_opt->getCommandBuffer();
    } else {
        id<MTLCommandQueue> queue = [ctx->device newCommandQueue];
        cmd = [queue commandBuffer];
    }
    
    ctx->_completion_sem = nil;
    
    if (!gpu_transformer_encode_batch(ctx, cmd, input_tokens, batch_size, seq_len, output_logits)) {
        return false;
    }
    
    // Commit the command buffer (async - completion handler runs on a background thread).
    if (ctx->sync_opt) {
        ctx->sync_opt->commitCommandBuffer(cmd, false); // async commit
    } else {
        [cmd commit];
    }
    
    // Wait for the completion handler to signal that output_logits is fully populated.
    // This is necessary because [cmd waitUntilCompleted] only guarantees the GPU is done,
    // NOT that the completion-handler memcpy has finished on its background thread.
    if (ctx->_completion_sem) {
        dispatch_semaphore_wait(ctx->_completion_sem, DISPATCH_TIME_FOREVER);
        dispatch_release(ctx->_completion_sem);
        ctx->_completion_sem = nil;
    }
    
    return true;
}

} // extern "C"
