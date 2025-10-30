/*
 * TransformerLayers.h
 * 
 * 12-layer Transformer Architecture Interface
 * CUDA enwik8 profile compatible implementation
 * Full mathematical accuracy - no dummy implementations
 */

#ifndef TRANSFORMER_LAYERS_H
#define TRANSFORMER_LAYERS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct CUDACompatibleTransformer12Layer CUDACompatibleTransformer12Layer;

// Error codes
typedef enum {
    TRANSFORMER_LAYERS_SUCCESS = 0,
    TRANSFORMER_LAYERS_ERROR_INVALID_PARAM,
    TRANSFORMER_LAYERS_ERROR_MEMORY_ALLOCATION,
    TRANSFORMER_LAYERS_ERROR_DEVICE_NOT_FOUND,
    TRANSFORMER_LAYERS_ERROR_BUFFER_ALLOCATION,
    TRANSFORMER_LAYERS_ERROR_INVALID_DIMENSIONS,
    TRANSFORMER_LAYERS_ERROR_EXECUTION_FAILED,
    TRANSFORMER_LAYERS_ERROR_NOT_INITIALIZED,
    TRANSFORMER_LAYERS_ERROR_LAYER_FAILED
} TransformerLayersError;

// Architecture information structure
typedef struct {
    uint32_t num_layers;              // Number of transformer layers (12)
    uint32_t hidden_size;             // Hidden dimension size (768)  
    uint32_t num_heads;               // Number of attention heads (16)
    uint32_t head_size;               // Size per attention head (48)
    uint32_t ffn_size;                // Feed-forward network size (3072)
    uint32_t max_sequence_length;     // Maximum sequence length (64)
    
    size_t parameters_per_layer;      // Parameters in single layer
    size_t total_parameters;          // Total model parameters
    size_t memory_usage_mb;           // Current memory usage in MB
    
    bool cuda_enwik8_compatible;      // CUDA enwik8 profile compatibility
} TransformerArchitectureInfo;

// Core API Functions

/**
 * Create 12-layer CUDA-compatible Transformer with enwik8 profile specifications
 * Architecture: 12 layers × (768-dim attention + 3072-dim FFN)
 * @param transformer Pointer to store created transformer
 * @return TRANSFORMER_LAYERS_SUCCESS on success, error code on failure
 */
TransformerLayersError transformer_layers_create_12_layer(CUDACompatibleTransformer12Layer** transformer);

/**
 * Execute complete 12-layer forward pass with authentic CUDA mathematics
 * No dummy implementations - full attention + FFN computation
 * @param transformer 12-layer transformer instance
 * @param input_embeddings Input embeddings [seq_len, 768]
 * @param output_activations Output activations [seq_len, 768]
 * @param sequence_length Length of input sequence (≤ 64)
 * @return TRANSFORMER_LAYERS_SUCCESS on success, error code on failure
 */
TransformerLayersError transformer_layers_forward_pass_12_layer(CUDACompatibleTransformer12Layer* transformer,
                                                               const float* input_embeddings,
                                                               float* output_activations,
                                                               uint32_t sequence_length);

/**
 * Get detailed architecture information and parameter counts
 * @param transformer 12-layer transformer instance
 * @param info Output structure for architecture information
 */
void transformer_layers_get_architecture_info(CUDACompatibleTransformer12Layer* transformer,
                                             TransformerArchitectureInfo* info);

/**
 * Destroy 12-layer transformer and free all resources
 * @param transformer 12-layer transformer instance to destroy
 */
void transformer_layers_destroy_12_layer(CUDACompatibleTransformer12Layer* transformer);

// CUDA enwik8 Profile Constants

// Layer architecture specifications
#define TRANSFORMER_LAYERS_NUM_LAYERS 12
#define TRANSFORMER_LAYERS_HIDDEN_SIZE 768
#define TRANSFORMER_LAYERS_NUM_HEADS 16
#define TRANSFORMER_LAYERS_HEAD_SIZE 48      // 768 / 16
#define TRANSFORMER_LAYERS_FFN_SIZE 3072
#define TRANSFORMER_LAYERS_SEQ_LEN 64

// Memory requirements (estimated)
#define TRANSFORMER_LAYERS_MIN_MEMORY_MB 1536    // Minimum for basic operation
#define TRANSFORMER_LAYERS_RECOMMENDED_MEMORY_MB 3072  // Recommended for performance
#define TRANSFORMER_LAYERS_OPTIMAL_MEMORY_MB 6144      // Optimal for full features

// Parameter count estimates
#define TRANSFORMER_LAYERS_ATTENTION_PARAMS_PER_LAYER (4 * 768 * 768)      // Q,K,V,O
#define TRANSFORMER_LAYERS_FFN_PARAMS_PER_LAYER (2 * 768 * 3072)           // Input + Output  
#define TRANSFORMER_LAYERS_NORM_PARAMS_PER_LAYER (4 * 768)                 // 2 LayerNorms
#define TRANSFORMER_LAYERS_BIAS_PARAMS_PER_LAYER (4 * 768 + 2 * 3072)      // All biases

#define TRANSFORMER_LAYERS_TOTAL_PARAMS_PER_LAYER \
    (TRANSFORMER_LAYERS_ATTENTION_PARAMS_PER_LAYER + \
     TRANSFORMER_LAYERS_FFN_PARAMS_PER_LAYER + \
     TRANSFORMER_LAYERS_NORM_PARAMS_PER_LAYER + \
     TRANSFORMER_LAYERS_BIAS_PARAMS_PER_LAYER)

#define TRANSFORMER_LAYERS_TOTAL_PARAMS \
    (TRANSFORMER_LAYERS_TOTAL_PARAMS_PER_LAYER * TRANSFORMER_LAYERS_NUM_LAYERS)

// Utility Functions

/**
 * Get error message string for error code
 * @param error_code TransformerLayersError code
 * @return Human-readable error message
 */
const char* transformer_layers_get_error_string(TransformerLayersError error_code);

/**
 * Verify CUDA enwik8 profile compatibility
 * @param transformer 12-layer transformer instance
 * @return true if fully compatible with CUDA enwik8 specification
 */
bool transformer_layers_verify_cuda_compatibility(CUDACompatibleTransformer12Layer* transformer);

/**
 * Estimate memory requirements for given configuration
 * @param num_layers Number of transformer layers (should be 12)
 * @param hidden_size Hidden dimension size (should be 768)
 * @param seq_len Maximum sequence length (up to 64)
 * @param estimated_mb Estimated memory requirement in MB
 * @return TRANSFORMER_LAYERS_SUCCESS on success, error code on failure
 */
TransformerLayersError transformer_layers_estimate_memory(uint32_t num_layers,
                                                         uint32_t hidden_size,
                                                         uint32_t seq_len,
                                                         uint32_t* estimated_mb);

/**
 * Compare architecture with original CUDA enwik8 specification
 * @param transformer 12-layer transformer instance
 * @param matches_cuda_spec Output boolean for specification match
 * @param parameter_count Output total parameter count
 * @return TRANSFORMER_LAYERS_SUCCESS on success, error code on failure
 */
TransformerLayersError transformer_layers_compare_with_cuda_spec(CUDACompatibleTransformer12Layer* transformer,
                                                                bool* matches_cuda_spec,
                                                                size_t* parameter_count);

// Layer-level API Functions

/**
 * Get information about specific layer
 * @param transformer 12-layer transformer instance
 * @param layer_id Layer identifier (0-11)
 * @param attention_params Output attention parameter count
 * @param ffn_params Output FFN parameter count
 * @param is_initialized Output initialization status
 * @return TRANSFORMER_LAYERS_SUCCESS on success, error code on failure
 */
TransformerLayersError transformer_layers_get_layer_info(CUDACompatibleTransformer12Layer* transformer,
                                                        uint32_t layer_id,
                                                        size_t* attention_params,
                                                        size_t* ffn_params,
                                                        bool* is_initialized);

/**
 * Execute single layer forward pass (for debugging/profiling)
 * @param transformer 12-layer transformer instance
 * @param layer_id Layer identifier (0-11)
 * @param input_activations Input activations [seq_len, 768]
 * @param output_activations Output activations [seq_len, 768]
 * @param sequence_length Length of input sequence
 * @return TRANSFORMER_LAYERS_SUCCESS on success, error code on failure
 */
TransformerLayersError transformer_layers_execute_single_layer(CUDACompatibleTransformer12Layer* transformer,
                                                             uint32_t layer_id,
                                                             const float* input_activations,
                                                             float* output_activations,
                                                             uint32_t sequence_length);

#ifdef __cplusplus
}
#endif

#endif // TRANSFORMER_LAYERS_H
