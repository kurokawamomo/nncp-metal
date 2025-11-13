/*
 * MetalTransformerModel.h
 * 
 * 768-dimensional Transformer Model interface for Apple Silicon
 * CUDA enwik8 profile compatible implementation
 * Target: 14.9% compression ratio achievement
 */

#ifndef METAL_TRANSFORMER_MODEL_H
#define METAL_TRANSFORMER_MODEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct MetalTransformerModel768 MetalTransformerModel768;
typedef struct MetalTransformerConfig768 MetalTransformerConfig768;

// Error codes
typedef enum {
    METAL_TRANSFORMER_SUCCESS = 0,
    METAL_TRANSFORMER_ERROR_INVALID_PARAM,
    METAL_TRANSFORMER_ERROR_MEMORY_ALLOCATION,
    METAL_TRANSFORMER_ERROR_DEVICE_NOT_FOUND,
    METAL_TRANSFORMER_ERROR_BUFFER_ALLOCATION,
    METAL_TRANSFORMER_ERROR_INVALID_DIMENSIONS,
    METAL_TRANSFORMER_ERROR_EXECUTION_FAILED,
    METAL_TRANSFORMER_ERROR_UNSUPPORTED_OPERATION
} MetalTransformerError;

// Core API Functions

/**
 * Create 768-dimensional MetalTransformerModel with custom configuration
 * @param model Pointer to store created model
 * @param config Transformer configuration (768-dim, 12 layers, 16 heads)
 * @return METAL_TRANSFORMER_SUCCESS on success, error code on failure
 */
MetalTransformerError metal_transformer_768_create(MetalTransformerModel768** model,
                                                   const MetalTransformerConfig768* config);

/**
 * Create MetalTransformerModel with CUDA enwik8 profile configuration
 * Automatically configures for 768-dim, 12 layers, 16 heads, 3072 FFN
 * @param model Pointer to store created model
 * @return METAL_TRANSFORMER_SUCCESS on success, error code on failure
 */
MetalTransformerError metal_transformer_768_create_enwik8_profile(MetalTransformerModel768** model);

/**
 * Perform forward pass through 768-dimensional transformer
 * @param model MetalTransformerModel768 instance
 * @param input_data Input byte sequence (256 byte vocabulary)
 * @param input_length Length of input sequence (max 2048)
 * @param output_probabilities Output probability distribution [length x 256]
 * @return METAL_TRANSFORMER_SUCCESS on success, error code on failure
 */
MetalTransformerError metal_transformer_768_forward_pass(MetalTransformerModel768* model,
                                                        const uint8_t* input_data,
                                                        uint32_t input_length,
                                                        float* output_probabilities);

/**
 * Progressively expand context length (64 -> 512 -> 1024 -> 2048)
 * @param model MetalTransformerModel768 instance
 * @param new_context_length New context length (must be <= 2048)
 * @return METAL_TRANSFORMER_SUCCESS on success, error code on failure
 */
MetalTransformerError metal_transformer_768_expand_context(MetalTransformerModel768* model,
                                                          uint32_t new_context_length);

/**
 * Get detailed memory usage statistics
 * @param model MetalTransformerModel768 instance
 * @param total_memory_mb Total memory usage in MB
 * @param weights_memory_mb Memory used by model weights in MB
 * @param context_memory_mb Memory used by context buffers in MB
 */
void metal_transformer_768_get_memory_usage(MetalTransformerModel768* model,
                                           size_t* total_memory_mb,
                                           size_t* weights_memory_mb,
                                           size_t* context_memory_mb);

/**
 * Destroy MetalTransformerModel768 and free all resources
 * @param model MetalTransformerModel768 instance to destroy
 */
void metal_transformer_768_destroy(MetalTransformerModel768* model);

// Configuration Constants

// CUDA enwik8 profile compatibility
#define METAL_TRANSFORMER_768_HIDDEN_SIZE 768
#define METAL_TRANSFORMER_768_NUM_LAYERS 12
#define METAL_TRANSFORMER_768_NUM_HEADS 16
#define METAL_TRANSFORMER_768_FFN_SIZE 3072
#define METAL_TRANSFORMER_768_MAX_SEQ_LEN 2048
#define METAL_TRANSFORMER_768_VOCAB_SIZE 256

// Progressive context expansion milestones
#define METAL_TRANSFORMER_768_CONTEXT_STAGE_1 64
#define METAL_TRANSFORMER_768_CONTEXT_STAGE_2 512
#define METAL_TRANSFORMER_768_CONTEXT_STAGE_3 1024
#define METAL_TRANSFORMER_768_CONTEXT_STAGE_4 2048

// Memory budget recommendations (MB)
#define METAL_TRANSFORMER_768_MEMORY_BUDGET_MIN 2048   // Minimum for basic operation
#define METAL_TRANSFORMER_768_MEMORY_BUDGET_RECOMMENDED 4096  // CUDA enwik8 compatible
#define METAL_TRANSFORMER_768_MEMORY_BUDGET_OPTIMAL 8192      // For best performance

// Performance modes
typedef enum {
    METAL_TRANSFORMER_768_MODE_FAST = 0,        // 256-dim, 4 layers (current baseline)
    METAL_TRANSFORMER_768_MODE_BALANCED = 1,    // 512-dim, 8 layers (intermediate)
    METAL_TRANSFORMER_768_MODE_HIGH_QUALITY = 2 // 768-dim, 12 layers (target)
} MetalTransformer768PerformanceMode;

// Utility Functions

/**
 * Get error message string for error code
 * @param error_code MetalTransformerError code
 * @return Human-readable error message
 */
const char* metal_transformer_768_get_error_string(MetalTransformerError error_code);

/**
 * Check if Apple Silicon Metal support is available
 * @return true if Metal Transformer is supported on current device
 */
bool metal_transformer_768_is_available(void);

/**
 * Get Apple Silicon device capabilities
 * @param device_name Buffer to store device name (size >= 64)
 * @param max_memory_mb Maximum available memory in MB
 * @param compute_units Number of GPU compute units
 * @return METAL_TRANSFORMER_SUCCESS on success, error code on failure
 */
MetalTransformerError metal_transformer_768_get_device_info(char* device_name,
                                                           uint32_t* max_memory_mb,
                                                           uint32_t* compute_units);

/**
 * Estimate memory requirements for given configuration
 * @param hidden_size Hidden dimension size (should be 768)
 * @param num_layers Number of transformer layers (should be 12)
 * @param max_seq_len Maximum sequence length (up to 2048)
 * @param estimated_mb Estimated memory requirement in MB
 * @return METAL_TRANSFORMER_SUCCESS on success, error code on failure
 */
MetalTransformerError metal_transformer_768_estimate_memory(uint32_t hidden_size,
                                                           uint32_t num_layers,
                                                           uint32_t max_seq_len,
                                                           uint32_t* estimated_mb);

#ifdef __cplusplus
}
#endif

#endif // METAL_TRANSFORMER_MODEL_H
