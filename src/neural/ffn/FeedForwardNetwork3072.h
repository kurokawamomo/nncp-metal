/*
 * FeedForwardNetwork3072.h
 * 
 * 3072-dimensional Feed-Forward Network Interface
 * CUDA enwik8 profile compatible FFN with SwiGLU activation
 * Full mathematical accuracy - no dummy implementations
 */

#ifndef FEEDFORWARD_NETWORK_3072_H
#define FEEDFORWARD_NETWORK_3072_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct CUDACompatible3072FFN CUDACompatible3072FFN;

// Error codes
typedef enum {
    FFN_3072_SUCCESS = 0,
    FFN_3072_ERROR_INVALID_PARAM,
    FFN_3072_ERROR_MEMORY_ALLOCATION,
    FFN_3072_ERROR_DEVICE_NOT_FOUND,
    FFN_3072_ERROR_BUFFER_ALLOCATION,
    FFN_3072_ERROR_INVALID_DIMENSIONS,
    FFN_3072_ERROR_EXECUTION_FAILED,
    FFN_3072_ERROR_NOT_INITIALIZED,
    FFN_3072_ERROR_UNSUPPORTED_ACTIVATION
} FFN3072Error;

// Activation function types
typedef enum {
    FFN_ACTIVATION_RELU = 0,
    FFN_ACTIVATION_GELU,
    FFN_ACTIVATION_SWISH,
    FFN_ACTIVATION_SWIGLU,      // Swish-Gated Linear Units (CUDA enwik8 compatible)
    FFN_ACTIVATION_GEGLU       // GELU-Gated Linear Units
} FFNActivationType;

// FFN architecture information
typedef struct {
    uint32_t hidden_size;            // Hidden dimension size (768)
    uint32_t ffn_size;               // FFN intermediate size (3072)
    uint32_t gate_size;              // Gate mechanism size (6144 for SwiGLU)
    uint32_t max_sequence_length;    // Maximum sequence length (64)
    FFNActivationType activation_function;  // Activation function type
    
    size_t input_parameters;         // Input projection parameters
    size_t output_parameters;        // Output projection parameters
    size_t bias_parameters;          // Bias parameters
    size_t total_parameters;         // Total FFN parameters
    
    size_t memory_usage_mb;          // Current memory usage in MB
    bool cuda_enwik8_compatible;     // CUDA enwik8 profile compatibility
} FFNArchitectureInfo;

// Core API Functions

/**
 * Create 3072-dimensional feed-forward network with CUDA enwik8 specifications
 * Architecture: 768 → 6144 (SwiGLU) → 3072 → 768
 * @param ffn Pointer to store created FFN context
 * @return FFN_3072_SUCCESS on success, error code on failure
 */
FFN3072Error ffn_3072_create(CUDACompatible3072FFN** ffn);

/**
 * Execute 3072-dimensional FFN forward pass with authentic SwiGLU activation
 * No dummy implementations - full matrix multiplication and activation computation
 * @param ffn 3072-dimensional FFN context
 * @param input_hidden_states Input hidden states [seq_len, 768]
 * @param output_hidden_states Output hidden states [seq_len, 768]
 * @param sequence_length Length of input sequence (≤ 64)
 * @return FFN_3072_SUCCESS on success, error code on failure
 */
FFN3072Error ffn_3072_forward(CUDACompatible3072FFN* ffn,
                              const float* input_hidden_states,
                              float* output_hidden_states,
                              uint32_t sequence_length);

/**
 * Get detailed FFN architecture information and parameter counts
 * @param ffn 3072-dimensional FFN context
 * @param info Output structure for architecture information
 */
void ffn_3072_get_architecture_info(CUDACompatible3072FFN* ffn,
                                   FFNArchitectureInfo* info);

/**
 * Destroy 3072-dimensional FFN context and free all resources
 * @param ffn 3072-dimensional FFN context to destroy
 */
void ffn_3072_destroy(CUDACompatible3072FFN* ffn);

// CUDA enwik8 Profile Constants

// FFN architecture specifications
#define FFN_3072_HIDDEN_SIZE 768
#define FFN_3072_FFN_SIZE 3072
#define FFN_3072_GATE_SIZE 6144      // 3072 * 2 for SwiGLU
#define FFN_3072_SEQ_LEN 64

// Parameter count constants
#define FFN_3072_INPUT_PARAMS (768 * 6144)      // Input projection including gate
#define FFN_3072_OUTPUT_PARAMS (3072 * 768)     // Output projection
#define FFN_3072_BIAS_PARAMS (6144 + 768)       // Input + Output biases

#define FFN_3072_TOTAL_PARAMS \
    (FFN_3072_INPUT_PARAMS + \
     FFN_3072_OUTPUT_PARAMS + \
     FFN_3072_BIAS_PARAMS)

// Memory requirements (estimated)
#define FFN_3072_MIN_MEMORY_MB 512       // Minimum for basic operation
#define FFN_3072_RECOMMENDED_MEMORY_MB 1024  // Recommended for performance
#define FFN_3072_OPTIMAL_MEMORY_MB 2048      // Optimal for full features

// Utility Functions

/**
 * Get error message string for error code
 * @param error_code FFN3072Error code
 * @return Human-readable error message
 */
const char* ffn_3072_get_error_string(FFN3072Error error_code);

/**
 * Verify CUDA enwik8 profile compatibility
 * @param ffn 3072-dimensional FFN context
 * @return true if fully compatible with CUDA enwik8 specification
 */
bool ffn_3072_verify_cuda_compatibility(CUDACompatible3072FFN* ffn);

/**
 * Estimate memory requirements for given configuration
 * @param hidden_size Hidden dimension size (should be 768)
 * @param ffn_size FFN intermediate size (should be 3072)
 * @param seq_len Maximum sequence length (up to 64)
 * @param estimated_mb Estimated memory requirement in MB
 * @return FFN_3072_SUCCESS on success, error code on failure
 */
FFN3072Error ffn_3072_estimate_memory(uint32_t hidden_size,
                                     uint32_t ffn_size,
                                     uint32_t seq_len,
                                     uint32_t* estimated_mb);

/**
 * Compare FFN with original CUDA enwik8 specification
 * @param ffn 3072-dimensional FFN context
 * @param matches_cuda_spec Output boolean for specification match
 * @param parameter_count Output total parameter count
 * @return FFN_3072_SUCCESS on success, error code on failure
 */
FFN3072Error ffn_3072_compare_with_cuda_spec(CUDACompatible3072FFN* ffn,
                                            bool* matches_cuda_spec,
                                            size_t* parameter_count);

// Advanced API Functions

/**
 * Execute FFN with custom activation function (for experimentation)
 * @param ffn 3072-dimensional FFN context
 * @param input_hidden_states Input hidden states [seq_len, 768]
 * @param output_hidden_states Output hidden states [seq_len, 768]
 * @param sequence_length Length of input sequence
 * @param activation_type Custom activation function to use
 * @return FFN_3072_SUCCESS on success, error code on failure
 */
FFN3072Error ffn_3072_forward_with_activation(CUDACompatible3072FFN* ffn,
                                             const float* input_hidden_states,
                                             float* output_hidden_states,
                                             uint32_t sequence_length,
                                             FFNActivationType activation_type);

/**
 * Get intermediate activations for analysis/visualization
 * @param ffn 3072-dimensional FFN context
 * @param gate_values Output gate values [seq_len, 6144]
 * @param intermediate_values Output intermediate values [seq_len, 3072]
 * @param sequence_length Current sequence length
 * @return FFN_3072_SUCCESS on success, error code on failure
 */
FFN3072Error ffn_3072_get_intermediate_activations(CUDACompatible3072FFN* ffn,
                                                  float* gate_values,
                                                  float* intermediate_values,
                                                  uint32_t sequence_length);

/**
 * Compute activation statistics for analysis
 * @param ffn 3072-dimensional FFN context
 * @param activation_stats Output activation statistics
 * @param sequence_length Current sequence length
 * @return FFN_3072_SUCCESS on success, error code on failure
 */
typedef struct {
    float mean_activation;           // Mean activation value
    float std_activation;            // Standard deviation of activations
    float sparsity_ratio;           // Ratio of near-zero activations
    float max_activation;           // Maximum activation value
    float min_activation;           // Minimum activation value
} FFNActivationStats;

FFN3072Error ffn_3072_compute_activation_stats(CUDACompatible3072FFN* ffn,
                                              FFNActivationStats* activation_stats,
                                              uint32_t sequence_length);

/**
 * Analyze parameter gradient norms (for training analysis)
 * @param ffn 3072-dimensional FFN context
 * @param gradient_norms Output parameter gradient norms
 * @param sequence_length Current sequence length
 * @return FFN_3072_SUCCESS on success, error code on failure
 */
typedef struct {
    float input_weights_norm;        // L2 norm of input weight gradients
    float output_weights_norm;       // L2 norm of output weight gradients
    float input_bias_norm;           // L2 norm of input bias gradients
    float output_bias_norm;          // L2 norm of output bias gradients
    float total_gradient_norm;       // Total gradient norm
} FFNGradientNorms;

FFN3072Error ffn_3072_analyze_gradient_norms(CUDACompatible3072FFN* ffn,
                                            FFNGradientNorms* gradient_norms,
                                            uint32_t sequence_length);

#ifdef __cplusplus
}
#endif

#endif // FEEDFORWARD_NETWORK_3072_H
