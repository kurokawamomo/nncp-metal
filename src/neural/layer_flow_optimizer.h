/*
 * layer_flow_optimizer.h
 *
 * Layer Flow Optimizer & Execution Engine
 * Orchestrates data flow between Unified Memory, GPU Engines, and CPU Application logic.
 */

#ifndef LAYER_FLOW_OPTIMIZER_H
#define LAYER_FLOW_OPTIMIZER_H

#include "unified_memory_manager.h"
#include "gpu_native_transformer.h"
#include "gpu_native_lstm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FLOW_ENGINE_TRANSFORMER,
    FLOW_ENGINE_LSTM
} FlowEngineType;

typedef struct FlowOptimizerContext FlowOptimizerContext;

/**
 * Create Flow Optimizer Context
 */
FlowOptimizerContext* flow_optimizer_create(id<MTLDevice> device);

/**
 * Destroy Flow Optimizer
 */
void flow_optimizer_destroy(FlowOptimizerContext* ctx);

/**
 * Configure Transformer Engine for the flow
 */
bool flow_optimizer_setup_transformer(FlowOptimizerContext* ctx, GPUTransformerConfig config);

/**
 * Set Transformer weights
 */
bool flow_optimizer_set_transformer_weights(FlowOptimizerContext* ctx,
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
 * Configure LSTM Engine for the flow
 */
bool flow_optimizer_setup_lstm(FlowOptimizerContext* ctx, GPULSTMConfig config);

/**
 * Execute Batch Prediction Flow (Async)
 * 
 * Returns true if command was successfully enqueued.
 * Output pointer is valid but data may not be ready until sync.
 */
bool flow_optimizer_execute_batch(FlowOptimizerContext* ctx,
                                 const int32_t* input_tokens,
                                 size_t batch_size,
                                 size_t seq_len,
                                 FlowEngineType engine_type,
                                 float** output_logits_ptr);

/**
 * Wait for all pending operations to complete
 */
void flow_optimizer_sync(FlowOptimizerContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // LAYER_FLOW_OPTIMIZER_H
