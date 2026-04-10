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
#include "unified_memory_manager.h"

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
                                id<MTLBuffer> out_proj,
                                id<MTLBuffer> b_k,
                                id<MTLBuffer> b_v,
                                id<MTLBuffer> b_o,
                                id<MTLBuffer> b_ffn1,
                                id<MTLBuffer> b_ffn2,
                                id<MTLBuffer> b_out,
                                id<MTLBuffer> w_rel_r,
                                id<MTLBuffer> b_rel_r,
                                id<MTLBuffer> ln_final_weights);

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
    id<MTLBuffer> ffn1;       /* [L, H, F] */
    id<MTLBuffer> ffn2;       /* [L, F, H] */
    id<MTLBuffer> ln;         /* [L, 2, H] */
    id<MTLBuffer> out_proj;   /* [H, V] */
    id<MTLBuffer> b_k;        /* [L, H] */
    id<MTLBuffer> b_v;        /* [L, H] */
    id<MTLBuffer> b_o;        /* [L, H] */
    id<MTLBuffer> b_ffn1;     /* [L, F] */
    id<MTLBuffer> b_ffn2;     /* [L, H] */
    id<MTLBuffer> b_out;      /* [V] */
    id<MTLBuffer> w_rel_r;     /* default=[NH,HD,D_POS] tied / enwik8=nil */
    id<MTLBuffer> w_rel_r_all; /* enwik8 only: [L, NH, HD, D_POS] per-layer */
    id<MTLBuffer> b_rel_r;    /* [NH, total_len] tied rel PE bias */
    id<MTLBuffer> ln_final;   /* [2, H]: gamma_f, beta_f for LN_FINAL */
} MPSTransformerWeightBuffers;

bool mps_transformer_get_weight_buffers(MPSTransformerContext* ctx,
                                        MPSTransformerWeightBuffers* out);

/** Set per-layer w_rel_r_all buffer (enwik8 only). Must be called after set_weights. */
void mps_transformer_set_relr_all(MPSTransformerContext* ctx, id<MTLBuffer> w_rel_r_all);

/**
 * Returns the configuration stored inside the context (zeroed on error).
 */
MPSTransformerConfig mps_transformer_get_config(MPSTransformerContext* ctx);

/**
 * KV cache accessor for training (Phase E2.3).
 * Layout: [num_layers, batch, kv_total_len, H]
 * Memory positions: [0 .. out_memory_len-1]; current: [out_memory_len .. out_total_len-1]
 * Returns false when the decode pipeline has not been set up yet.
 */
bool mps_transformer_get_kv_cache_buffers(MPSTransformerContext* ctx,
                                           id<MTLBuffer>* out_k,
                                           id<MTLBuffer>* out_v,
                                           uint32_t*      out_batch,
                                           uint32_t*      out_total_len,
                                           uint32_t*      out_memory_len);

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

/**
 * Segment-level forward pass (unified inference + training)
 *
 * Processes n_streams × seg_len tokens in one MPSGraph call and returns
 * logits for every position.  This is the correct granularity for online
 * learning: the training graph sees the same context as inference.
 *
 * input_tokens : int32 [n_streams × seg_len]  (row-major: stream-0 seg, stream-1 seg, …)
 * n_streams    : batch size (= NUM_STREAMS, e.g. 16)
 * seg_len      : segment length (e.g. 32)
 * logits_out   : float [n_streams × seg_len × vocab_size]  (caller-allocated)
 *
 * After the call, kv_cache_pos is advanced by seg_len.  If the cache is
 * full a Transformer-XL memory shift is performed automatically.
 */
void mps_transformer_execute_segment(
    MPSTransformerContext* ctx,
    const int32_t*         input_tokens,
    int                    n_streams,
    int                    seg_len,
    float*                 logits_out);

/**
 * Transformer-XL memory shift (called at block boundaries).
 * Copies the "current" KV window into the "memory" window and resets
 * kv_cache_pos to kv_memory_len so new tokens overwrite the oldest ones.
 *
 * Safe to call even when the decode pipeline has not been set up yet
 * (becomes a no-op in that case).
 */
void mps_transformer_memory_shift(MPSTransformerContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // MPS_TRANSFORMER_GRAPH_H
