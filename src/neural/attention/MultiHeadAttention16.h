/*
 * MultiHeadAttention16.h
 * 
 * 16-Head Multi-Head Self-Attention Interface
 * CUDA enwik8 profile compatible attention mechanism
 * Full mathematical accuracy - no dummy implementations
 */

#ifndef MULTIHEAD_ATTENTION_16_H
#define MULTIHEAD_ATTENTION_16_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct CUDA16HeadAttentionContext CUDA16HeadAttentionContext;

// Error codes
typedef enum {
    MULTIHEAD_ATTENTION_16_SUCCESS = 0,
    MULTIHEAD_ATTENTION_16_ERROR_INVALID_PARAM,
    MULTIHEAD_ATTENTION_16_ERROR_MEMORY_ALLOCATION,
    MULTIHEAD_ATTENTION_16_ERROR_DEVICE_NOT_FOUND,
    MULTIHEAD_ATTENTION_16_ERROR_BUFFER_ALLOCATION,
    MULTIHEAD_ATTENTION_16_ERROR_INVALID_DIMENSIONS,
    MULTIHEAD_ATTENTION_16_ERROR_EXECUTION_FAILED,
    MULTIHEAD_ATTENTION_16_ERROR_NOT_INITIALIZED,
    MULTIHEAD_ATTENTION_16_ERROR_UNSUPPORTED_OPERATION
} MultiHeadAttention16Error;

// Attention architecture information
typedef struct {
    uint32_t num_heads;              // Number of attention heads (16)
    uint32_t hidden_size;            // Hidden dimension size (768)
    uint32_t head_size;              // Size per attention head (48)
    uint32_t max_sequence_length;    // Maximum sequence length (64)
    
    size_t query_parameters;         // Query weight parameters
    size_t key_parameters;           // Key weight parameters
    size_t value_parameters;         // Value weight parameters
    size_t output_parameters;        // Output projection parameters
    size_t bias_parameters;          // Bias parameters
    size_t total_parameters;         // Total attention parameters
    
    size_t memory_usage_mb;          // Current memory usage in MB
    bool cuda_enwik8_compatible;     // CUDA enwik8 profile compatibility
} AttentionArchitectureInfo;

// Core API Functions

/**
 * Create 16-head multi-head self-attention with CUDA enwik8 specifications
 * Architecture: 16 heads × 48 dimensions = 768 total dimensions
 * @param context Pointer to store created attention context
 * @return MULTIHEAD_ATTENTION_16_SUCCESS on success, error code on failure
 */
MultiHeadAttention16Error multihead_attention_16_create(CUDA16HeadAttentionContext** context);

/**
 * Execute 16-head self-attention forward pass with authentic mathematics
 * No dummy implementations - full scaled dot-product attention computation
 * @param context 16-head attention context
 * @param input_hidden_states Input hidden states [seq_len, 768]
 * @param output_hidden_states Output hidden states [seq_len, 768]  
 * @param sequence_length Length of input sequence (≤ 64)
 * @param attention_mask Optional attention mask (NULL for causal mask only)
 * @return MULTIHEAD_ATTENTION_16_SUCCESS on success, error code on failure
 */
MultiHeadAttention16Error multihead_attention_16_forward(CUDA16HeadAttentionContext* context,
                                                        const float* input_hidden_states,
                                                        float* output_hidden_states,
                                                        uint32_t sequence_length,
                                                        const bool* attention_mask);

/**
 * Get detailed attention architecture information and parameter counts
 * @param context 16-head attention context
 * @param info Output structure for architecture information
 */
void multihead_attention_16_get_architecture_info(CUDA16HeadAttentionContext* context,
                                                 AttentionArchitectureInfo* info);

/**
 * Destroy 16-head attention context and free all resources
 * @param context 16-head attention context to destroy
 */
void multihead_attention_16_destroy(CUDA16HeadAttentionContext* context);

// CUDA enwik8 Profile Constants

// Attention architecture specifications
#define MULTIHEAD_ATTENTION_16_NUM_HEADS 16
#define MULTIHEAD_ATTENTION_16_HIDDEN_SIZE 768
#define MULTIHEAD_ATTENTION_16_HEAD_SIZE 48      // 768 / 16
#define MULTIHEAD_ATTENTION_16_SEQ_LEN 64

// Parameter count constants
#define MULTIHEAD_ATTENTION_16_QUERY_PARAMS (768 * 768)      // Q projection
#define MULTIHEAD_ATTENTION_16_KEY_PARAMS (768 * 768)        // K projection
#define MULTIHEAD_ATTENTION_16_VALUE_PARAMS (768 * 768)      // V projection
#define MULTIHEAD_ATTENTION_16_OUTPUT_PARAMS (768 * 768)     // Output projection
#define MULTIHEAD_ATTENTION_16_BIAS_PARAMS (4 * 768)         // Q,K,V,O biases

#define MULTIHEAD_ATTENTION_16_TOTAL_PARAMS \
    (MULTIHEAD_ATTENTION_16_QUERY_PARAMS + \
     MULTIHEAD_ATTENTION_16_KEY_PARAMS + \
     MULTIHEAD_ATTENTION_16_VALUE_PARAMS + \
     MULTIHEAD_ATTENTION_16_OUTPUT_PARAMS + \
     MULTIHEAD_ATTENTION_16_BIAS_PARAMS)

// Memory requirements (estimated)
#define MULTIHEAD_ATTENTION_16_MIN_MEMORY_MB 256     // Minimum for basic operation
#define MULTIHEAD_ATTENTION_16_RECOMMENDED_MEMORY_MB 512  // Recommended for performance
#define MULTIHEAD_ATTENTION_16_OPTIMAL_MEMORY_MB 1024     // Optimal for full features

// Utility Functions

/**
 * Get error message string for error code
 * @param error_code MultiHeadAttention16Error code
 * @return Human-readable error message
 */
const char* multihead_attention_16_get_error_string(MultiHeadAttention16Error error_code);

/**
 * Verify CUDA enwik8 profile compatibility
 * @param context 16-head attention context
 * @return true if fully compatible with CUDA enwik8 specification
 */
bool multihead_attention_16_verify_cuda_compatibility(CUDA16HeadAttentionContext* context);

/**
 * Estimate memory requirements for given configuration
 * @param num_heads Number of attention heads (should be 16)
 * @param hidden_size Hidden dimension size (should be 768)
 * @param seq_len Maximum sequence length (up to 64)
 * @param estimated_mb Estimated memory requirement in MB
 * @return MULTIHEAD_ATTENTION_16_SUCCESS on success, error code on failure
 */
MultiHeadAttention16Error multihead_attention_16_estimate_memory(uint32_t num_heads,
                                                                uint32_t hidden_size,
                                                                uint32_t seq_len,
                                                                uint32_t* estimated_mb);

/**
 * Compare attention with original CUDA enwik8 specification
 * @param context 16-head attention context
 * @param matches_cuda_spec Output boolean for specification match
 * @param parameter_count Output total parameter count
 * @return MULTIHEAD_ATTENTION_16_SUCCESS on success, error code on failure
 */
MultiHeadAttention16Error multihead_attention_16_compare_with_cuda_spec(CUDA16HeadAttentionContext* context,
                                                                       bool* matches_cuda_spec,
                                                                       size_t* parameter_count);

// Advanced API Functions

/**
 * Execute attention with custom attention patterns (for experimentation)
 * @param context 16-head attention context
 * @param input_hidden_states Input hidden states [seq_len, 768]
 * @param custom_attention_pattern Custom attention pattern [16, seq_len, seq_len]
 * @param output_hidden_states Output hidden states [seq_len, 768]
 * @param sequence_length Length of input sequence
 * @return MULTIHEAD_ATTENTION_16_SUCCESS on success, error code on failure
 */
MultiHeadAttention16Error multihead_attention_16_forward_with_pattern(CUDA16HeadAttentionContext* context,
                                                                     const float* input_hidden_states,
                                                                     const float* custom_attention_pattern,
                                                                     float* output_hidden_states,
                                                                     uint32_t sequence_length);

/**
 * Get attention weights for analysis/visualization
 * @param context 16-head attention context
 * @param attention_weights Output attention weights [16, seq_len, seq_len]
 * @param sequence_length Current sequence length
 * @return MULTIHEAD_ATTENTION_16_SUCCESS on success, error code on failure
 */
MultiHeadAttention16Error multihead_attention_16_get_attention_weights(CUDA16HeadAttentionContext* context,
                                                                      float* attention_weights,
                                                                      uint32_t sequence_length);

/**
 * Compute attention entropy for each head (attention diversity measure)
 * @param context 16-head attention context
 * @param entropy_per_head Output entropy values [16]
 * @param sequence_length Current sequence length
 * @return MULTIHEAD_ATTENTION_16_SUCCESS on success, error code on failure
 */
MultiHeadAttention16Error multihead_attention_16_compute_attention_entropy(CUDA16HeadAttentionContext* context,
                                                                          float* entropy_per_head,
                                                                          uint32_t sequence_length);

/**
 * Analyze attention head specialization patterns
 * @param context 16-head attention context
 * @param head_analysis Output analysis per head [16]
 * @param sequence_length Current sequence length
 * @return MULTIHEAD_ATTENTION_16_SUCCESS on success, error code on failure
 */
MultiHeadAttention16Error multihead_attention_16_analyze_head_patterns(CUDA16HeadAttentionContext* context,
                                                                      float* head_analysis,
                                                                      uint32_t sequence_length);

#ifdef __cplusplus
}
#endif

#endif // MULTIHEAD_ATTENTION_16_H
