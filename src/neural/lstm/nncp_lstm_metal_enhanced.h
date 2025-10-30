/*
 * Enhanced Metal LSTM Context for CUDA Compatibility
 * 
 * Extends the current Metal LSTM implementation with CUDA compatibility fields,
 * particularly the critical train_len = seg_len relationship that ensures
 * mathematical equivalence with original CUDA implementation.
 * 
 * Based on original CUDA NNCP implementation for exact behavioral matching.
 */

#ifndef NNCP_LSTM_METAL_ENHANCED_H
#define NNCP_LSTM_METAL_ENHANCED_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../config/cuda_profiles.h"
#include "../compatibility/cuda_math_compat.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
typedef id<MTLDevice> MTLDeviceRef;
typedef id<MTLBuffer> MTLBufferRef;
typedef id<MTLCommandQueue> MTLCommandQueueRef;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoderRef;
#else
typedef void* MTLDeviceRef;
typedef void* MTLBufferRef;
typedef void* MTLCommandQueueRef;
typedef void* MTLComputeCommandEncoderRef;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// CUDA-Compatible LSTM Configuration
typedef struct {
    // Core CUDA compatibility parameters
    int32_t seg_len;                    // Segment length (from CUDA profile)
    int32_t train_len;                  // Training length (MUST equal seg_len for CUDA compatibility)
    int32_t batch_size;                 // Batch size for processing
    int32_t hidden_size;                // LSTM hidden state size
    int32_t num_layers;                 // Number of LSTM layers
    int32_t vocab_size;                 // Vocabulary size (n_symbols)
    
    // CUDA mathematical compatibility
    float learning_rate;                // Learning rate
    float dropout_rate;                 // Dropout rate
    bool use_mixed_precision;           // FP16/FP32 mixed precision
    bool enforce_deterministic;         // Force deterministic computation
    uint64_t random_seed;              // Fixed seed for reproducibility
    
    // Memory management
    size_t memory_budget_bytes;         // Memory budget in bytes
    bool auto_optimize_memory;          // Auto-optimize memory usage
    
    // CUDA profile reference
    const CUDAProfile* cuda_profile;    // Associated CUDA profile
    CUDAMathConfig* math_config;        // Mathematical compatibility config
} EnhancedLSTMConfig;

// Enhanced LSTM State (CUDA-compatible)
typedef struct {
    // LSTM internal states
    CUDACompatTensor* hidden_state;     // h_t (batch_size, hidden_size)
    CUDACompatTensor* cell_state;       // c_t (batch_size, hidden_size)
    CUDACompatTensor* input_gate;       // i_t intermediate state
    CUDACompatTensor* forget_gate;      // f_t intermediate state
    CUDACompatTensor* candidate_gate;   // g_t intermediate state
    CUDACompatTensor* output_gate;      // o_t intermediate state
    
    // Sequence processing state
    int32_t current_seq_pos;            // Current position in sequence
    int32_t processed_segments;         // Number of processed segments
    bool is_sequence_start;             // Whether this is sequence start
    
    // CUDA compatibility tracking
    bool cuda_state_validated;         // Whether state matches CUDA expectations
    float precision_tolerance;          // Current precision tolerance
    size_t total_operations;            // Operation count for debugging
} EnhancedLSTMState;

// Enhanced Metal LSTM Context
typedef struct {
    // Metal framework objects
    MTLDeviceRef device;
    MTLCommandQueueRef command_queue;
    MTLBufferRef weight_buffer;
    MTLBufferRef bias_buffer;
    MTLBufferRef state_buffer;
    MTLBufferRef workspace_buffer;
    
    // CUDA compatibility configuration
    EnhancedLSTMConfig config;
    EnhancedLSTMState* lstm_state;
    
    // Model parameters (CUDA-compatible layout)
    CUDACompatTensor* weight_ih;        // Input-to-hidden weights [4*hidden_size, input_size]
    CUDACompatTensor* weight_hh;        // Hidden-to-hidden weights [4*hidden_size, hidden_size]
    CUDACompatTensor* bias_ih;          // Input-to-hidden bias [4*hidden_size]
    CUDACompatTensor* bias_hh;          // Hidden-to-hidden bias [4*hidden_size]
    
    // Sequence processing buffers
    CUDACompatTensor* input_sequence;   // Current input sequence
    CUDACompatTensor* output_sequence;  // Current output sequence
    CUDACompatTensor* attention_weights; // Attention weights (if used)
    
    // Performance tracking
    double total_inference_time_ms;
    size_t total_sequences_processed;
    size_t total_segments_processed;
    
    // CUDA validation state
    bool cuda_compatibility_verified;
    float max_observed_deviation;
    char last_validation_error[256];
    
    // Memory management
    size_t allocated_memory_bytes;
    size_t peak_memory_usage_bytes;
    bool memory_optimized;
} EnhancedMetalLSTMContext;

// Context Management
EnhancedMetalLSTMContext* enhanced_lstm_context_create(const EnhancedLSTMConfig* config);
void enhanced_lstm_context_free(EnhancedMetalLSTMContext* context);

// Configuration Management
EnhancedLSTMConfig* enhanced_lstm_config_create_from_cuda_profile(const CUDAProfile* profile);
EnhancedLSTMConfig* enhanced_lstm_config_create_default(void);
void enhanced_lstm_config_free(EnhancedLSTMConfig* config);
bool enhanced_lstm_config_validate(const EnhancedLSTMConfig* config);

// CUDA Compatibility Verification
bool enhanced_lstm_verify_cuda_compatibility(EnhancedMetalLSTMContext* context);
bool enhanced_lstm_verify_train_len_relationship(const EnhancedLSTMConfig* config);
bool enhanced_lstm_validate_state_consistency(const EnhancedLSTMState* state, const EnhancedLSTMConfig* config);

// Model Parameter Management
bool enhanced_lstm_initialize_weights(EnhancedMetalLSTMContext* context, uint64_t seed);
bool enhanced_lstm_load_cuda_weights(EnhancedMetalLSTMContext* context, const char* weights_path);
bool enhanced_lstm_save_cuda_weights(const EnhancedMetalLSTMContext* context, const char* weights_path);

// Sequence Processing (CUDA-compatible)
bool enhanced_lstm_process_sequence(EnhancedMetalLSTMContext* context, 
                                   const uint8_t* input_data, 
                                   size_t input_length,
                                   float* output_probabilities);

bool enhanced_lstm_process_segment(EnhancedMetalLSTMContext* context,
                                  const uint8_t* segment_data,
                                  size_t segment_length,
                                  float* segment_output);

// State Management
bool enhanced_lstm_reset_state(EnhancedMetalLSTMContext* context);
bool enhanced_lstm_save_state(const EnhancedMetalLSTMContext* context, void** state_data, size_t* state_size);
bool enhanced_lstm_restore_state(EnhancedMetalLSTMContext* context, const void* state_data, size_t state_size);

// CUDA Mathematical Operations (delegated to compatibility layer)
bool enhanced_lstm_forward_pass(EnhancedMetalLSTMContext* context,
                               const CUDACompatTensor* input,
                               CUDACompatTensor* output);

bool enhanced_lstm_compute_gates(EnhancedMetalLSTMContext* context,
                                const CUDACompatTensor* input,
                                const EnhancedLSTMState* prev_state,
                                EnhancedLSTMState* new_state);

// Performance and Debugging
void enhanced_lstm_print_performance_stats(const EnhancedMetalLSTMContext* context);
void enhanced_lstm_print_cuda_compatibility_report(const EnhancedMetalLSTMContext* context);
bool enhanced_lstm_enable_debugging(EnhancedMetalLSTMContext* context, bool enable);

// Memory Optimization
bool enhanced_lstm_optimize_memory(EnhancedMetalLSTMContext* context);
size_t enhanced_lstm_estimate_memory_usage(const EnhancedLSTMConfig* config);
bool enhanced_lstm_free_unused_buffers(EnhancedMetalLSTMContext* context);

// Error Handling
typedef enum {
    ENHANCED_LSTM_SUCCESS = 0,
    ENHANCED_LSTM_ERROR_INVALID_CONFIG,
    ENHANCED_LSTM_ERROR_CUDA_INCOMPATIBLE,
    ENHANCED_LSTM_ERROR_TRAIN_LEN_MISMATCH,
    ENHANCED_LSTM_ERROR_MEMORY_ALLOCATION,
    ENHANCED_LSTM_ERROR_METAL_FAILURE,
    ENHANCED_LSTM_ERROR_STATE_INVALID,
    ENHANCED_LSTM_ERROR_SEQUENCE_TOO_LONG,
    ENHANCED_LSTM_ERROR_PRECISION_EXCEEDED
} EnhancedLSTMError;

const char* enhanced_lstm_error_string(EnhancedLSTMError error);

// Critical CUDA Compatibility Checks
bool enhanced_lstm_check_seg_len_consistency(const EnhancedLSTMConfig* config);
bool enhanced_lstm_check_parameter_alignment(const EnhancedMetalLSTMContext* context);
bool enhanced_lstm_validate_mathematical_equivalence(EnhancedMetalLSTMContext* context, 
                                                    const CUDACompatTensor* reference_output);

#ifdef __cplusplus
}
#endif

#endif // NNCP_LSTM_METAL_ENHANCED_H