/*
 * gpu_native_lstm.h
 *
 * GPU-Native LSTM Engine
 * Implements optimized LSTM forward pass on GPU
 * Features:
 * - Parallel input projection (pre-computation)
 * - GPU-resident state management
 * - Unified Memory integration
 */

#ifndef GPU_NATIVE_LSTM_H
#define GPU_NATIVE_LSTM_H

#import <Metal/Metal.h>
#include "unified_memory_manager.h"
#include "sync_optimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GPULSTMConfig {
    uint32_t input_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t seq_len;    // Max sequence length
    uint32_t batch_size; // Max batch size
} GPULSTMConfig;

typedef struct GPULSTMContext GPULSTMContext;

/**
 * Create a GPU-Native LSTM context
 */
GPULSTMContext* gpu_lstm_create(id<MTLDevice> device, UnifiedMemoryManager* mem_mgr, GPULSTMConfig config);

/**
 * Set SyncOptimizer for async execution
 */
void gpu_lstm_set_sync_optimizer(GPULSTMContext* ctx, SyncOptimizer* sync_opt);

/**
 * Set model weights
 * Weights are expected to be in [4 * hidden, input] or [4 * hidden, hidden] layout
 * Order: Input Gate (i), Forget Gate (f), Cell Gate (g/c), Output Gate (o)
 */
bool gpu_lstm_set_weights(GPULSTMContext* ctx,
                         id<MTLBuffer> w_ih, // Input-Hidden weights [layers, 4*hidden, input]
                         id<MTLBuffer> w_hh, // Hidden-Hidden weights [layers, 4*hidden, hidden]
                         id<MTLBuffer> bias  // Biases [layers, 4*hidden] (optional)
                         );

/**
 * Destroy the context
 */
void gpu_lstm_destroy(GPULSTMContext* ctx);

/**
 * Reset LSTM state (h and c) to zeros
 */
void gpu_lstm_reset_state(GPULSTMContext* ctx);

/**
 * Perform sequence inference
 * @param ctx Context
 * @param input Input buffer (float, [batch, seq, input_size])
 * @param output Output buffer (float, [batch, seq, hidden_size]) - returns last layer output
 * @return Success status
 */
bool gpu_lstm_predict_sequence(GPULSTMContext* ctx,
                              const float* input,
                              size_t batch_size,
                              size_t seq_len,
                              float* output);

#ifdef __cplusplus
}
#endif

#endif // GPU_NATIVE_LSTM_H
