/*
 * gpu_native_transformer.h
 *
 * GPU-Native Transformer Engine
 * Implements full Transformer forward pass on GPU without CPU roundtrips
 * Supports Batch Processing (Masked Attention) for high throughput
 */

#ifndef GPU_NATIVE_TRANSFORMER_H
#define GPU_NATIVE_TRANSFORMER_H

#import <Metal/Metal.h>
#include "unified_memory_manager.h"
#include "sync_optimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GPUTransformerConfig {
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t num_heads;
    uint32_t ffn_size;
    uint32_t context_length; // Max sequence length (e.g., 64 or 2048 for batch)
    uint32_t vocab_size;
} GPUTransformerConfig;

typedef struct GPUTransformerContext GPUTransformerContext;

/**
 * Create a GPU-Native Transformer context
 */
GPUTransformerContext* gpu_transformer_create(id<MTLDevice> device, UnifiedMemoryManager* mem_mgr, GPUTransformerConfig config);

/**
 * Set SyncOptimizer for async execution
 */
void gpu_transformer_set_sync_optimizer(GPUTransformerContext* ctx, SyncOptimizer* sync_opt);

/**
 * Set model weights (using Metal buffers)
 */
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
                                id<MTLBuffer> out_proj);

/**
 * Destroy the context
 */
void gpu_transformer_destroy(GPUTransformerContext* ctx);

/**
 * Encode batch inference commands into a command buffer (Async)
 * Does NOT commit or wait.
 * @param ctx Context
 * @param cmd Command Buffer to encode into
 * @param input_tokens Input token buffer (int32, length = batch_size * seq_len)
 * @param batch_size Number of streams
 * @param seq_len Length of sequence
 * @param output_logits Output buffer (float, length = batch_size * seq_len * vocab_size)
 *                      NOTE: This buffer will be written to asynchronously. 
 *                      Ensure it remains valid until command buffer completes.
 * @return Success status
 */
bool gpu_transformer_encode_batch(GPUTransformerContext* ctx, 
                                  id<MTLCommandBuffer> cmd,
                                  const int32_t* input_tokens, 
                                  size_t batch_size,
                                  size_t seq_len,
                                  float* output_logits);

/**
 * Perform batch inference (Synchronous wrapper)
 * @param ctx Context
 * @param input_tokens Input token buffer (int32, length = batch_size * seq_len)
 * @param batch_size Number of streams
 * @param seq_len Length of sequence (e.g. 32 or 64)
 * @param output_logits Output buffer (float, length = batch_size * seq_len * vocab_size)
 * @return Success status
 */
bool gpu_transformer_predict_batch(GPUTransformerContext* ctx, 
                                  const int32_t* input_tokens, 
                                  size_t batch_size,
                                  size_t seq_len,
                                  float* output_logits);

#ifdef __cplusplus
}
#endif

#endif // GPU_NATIVE_TRANSFORMER_H
