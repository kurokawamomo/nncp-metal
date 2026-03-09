/*
 * mps_transformer_graph.h
 *
 * MPS Graph Transformer Engine
 * Implements Transformer architecture using Apple's MPSGraph framework
 * Optimized for M1/M2/M3 Neural Engine and GPU execution
 */

#ifndef MPS_TRANSFORMER_GRAPH_H
#define MPS_TRANSFORMER_GRAPH_H

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "../memory/unified_memory_manager.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MPSTransformerConfig {
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t ffn_size;
    uint32_t vocab_size;
    uint32_t max_seq_len;
} MPSTransformerConfig;

typedef struct MPSTransformerContext MPSTransformerContext;

/**
 * Create MPS Graph Transformer Context
 */
MPSTransformerContext* mps_transformer_create(id<MTLDevice> device, MPSTransformerConfig config);

/**
 * Set model weights (using Metal buffers)
 * These buffers will be wrapped as MPSGraphTensorData
 */
bool mps_transformer_set_weights(MPSTransformerContext* ctx, 
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
                                id<MTLBuffer> out_proj);

/**
 * Destroy Context
 */
void mps_transformer_destroy(MPSTransformerContext* ctx);

/**
 * Reset KV cache (call at the start of each new compression/decompression session)
 * Sets kv_cache_pos = 0 and zeroes the cache buffers.
 */
void mps_transformer_reset_kv_cache(MPSTransformerContext* ctx);

/**
 * Weight buffer accessor — returns Metal buffers and config so that other
 * modules (e.g. OnlineTrainer) can read/write the weights without knowing
 * the internals of MPSTransformerContext.
 */
typedef struct {
    id<MTLBuffer> embed;      /* [V, H] */
    id<MTLBuffer> pos_embed;  /* [S, H] */
    id<MTLBuffer> attn_q;     /* [L, H, H] */
    id<MTLBuffer> attn_k;     /* [L, H, H] */
    id<MTLBuffer> attn_v;     /* [L, H, H] */
    id<MTLBuffer> attn_out;   /* [L, H, H] */
    id<MTLBuffer> ffn1;       /* [L, H, F*2] */
    id<MTLBuffer> ffn2;       /* [L, F, H] */
    id<MTLBuffer> ln;         /* [L, 2, H] */
    id<MTLBuffer> final_ln;   /* [2, H] */
    id<MTLBuffer> out_proj;   /* [H, V] */
} MPSTransformerWeightBuffers;

bool mps_transformer_get_weight_buffers(MPSTransformerContext* ctx,
                                        MPSTransformerWeightBuffers* out);

/**
 * Returns the configuration stored inside the context (zeroed on error).
 */
MPSTransformerConfig mps_transformer_get_config(MPSTransformerContext* ctx);

/**
 * Execute Batch Inference using MPSGraph (synchronous)
 *
 * Blocks the CPU until GPU computation and readBytes are complete.
 * Use mps_transformer_execute_async for non-blocking operation.
 */
bool mps_transformer_execute(MPSTransformerContext* ctx,
                            const int32_t* input_data,
                            size_t batch_size,
                            size_t seq_len,
                            float* output_data);

/**
 * Execute Batch Inference using MPSGraph (asynchronous)
 *
 * Returns immediately after encoding. The completion_handler is invoked
 * on an internal queue when GPU results are written to output_data.
 * output_data must remain valid until completion_handler is called.
 *
 * @param ctx           Context
 * @param input_data    Pointer to input token data (int32)
 * @param batch_size    Number of sequences
 * @param seq_len       Length of sequences
 * @param output_data   Caller-owned output logits buffer (must outlive callback)
 * @param user_info     Opaque pointer forwarded to completion_handler
 * @param completion_handler  Called on completion; error==nil on success
 * @return true if encoding was successfully submitted, false on early error
 */
bool mps_transformer_execute_async(MPSTransformerContext* ctx,
                                   const int32_t* input_data,
                                   size_t batch_size,
                                   size_t seq_len,
                                   float* output_data,
                                   void* user_info,
                                   void (*completion_handler)(void* user_info, bool success));

#ifdef __cplusplus
}
#endif

#endif // MPS_TRANSFORMER_GRAPH_H
