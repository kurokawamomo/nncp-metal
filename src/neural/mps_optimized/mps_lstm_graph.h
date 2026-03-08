/*
 * mps_lstm_graph.h
 *
 * MPS Graph LSTM Engine
 * Implements LSTM architecture using Apple's MPSGraph framework
 * Optimized for M1/M2/M3 Neural Engine and GPU execution
 */

#ifndef MPS_LSTM_GRAPH_H
#define MPS_LSTM_GRAPH_H

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "../memory/unified_memory_manager.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MPSLSTMConfig {
    uint32_t input_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t batch_size;
    uint32_t seq_len; // Max sequence length
} MPSLSTMConfig;

typedef struct MPSLSTMContext MPSLSTMContext;

/**
 * Create MPS Graph LSTM Context
 */
MPSLSTMContext* mps_lstm_create(id<MTLDevice> device, MPSLSTMConfig config);

/**
 * Set model weights (using Metal buffers)
 * These buffers will be wrapped as MPSGraphTensorData
 */
bool mps_lstm_set_weights(MPSLSTMContext* ctx,
                         id<MTLBuffer> w_ih, // [Layers, 4*Hidden, Input]
                         id<MTLBuffer> w_hh, // [Layers, 4*Hidden, Hidden]
                         id<MTLBuffer> bias  // [Layers, 4*Hidden]
                         );

/**
 * Destroy Context
 */
void mps_lstm_destroy(MPSLSTMContext* ctx);

/**
 * Reset State (h, c)
 */
void mps_lstm_reset_state(MPSLSTMContext* ctx);

/**
 * Execute Sequence Inference using MPSGraph
 *
 * @param ctx Context
 * @param input_data Pointer to input buffer (float) [Batch, Seq, Input]
 * @param batch_size Number of sequences
 * @param seq_len Length of sequences
 * @param output_data Pointer to output buffer (float) [Batch, Seq, Hidden]
 * @return Success status
 */
bool mps_lstm_execute(MPSLSTMContext* ctx,
                     const float* input_data,
                     size_t batch_size,
                     size_t seq_len,
                     float* output_data);

#ifdef __cplusplus
}
#endif

#endif // MPS_LSTM_GRAPH_H
