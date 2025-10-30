/*
 * LongContextProcessor.h
 * 
 * 1024-token Long Context Processing System
 * Implements extended-range dependency modeling with Flash Attention optimization
 * Compatible with CUDA enwik8 extended context specifications
 */

#ifndef LONG_CONTEXT_PROCESSOR_H
#define LONG_CONTEXT_PROCESSOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "AdaptiveContextManager.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct LongContextProcessor LongContextProcessor;

// Long context configuration (1024 tokens)
typedef struct {
    uint32_t max_tokens;                    // 1024 tokens (CUDA enwik8 extended)
    uint32_t flash_attention_block_size;   // Flash attention block size
    uint32_t memory_tile_size;              // Memory tiling size for efficiency
    uint32_t gradient_accumulation_steps;   // Gradient accumulation steps
    bool use_flash_attention;               // Enable Flash Attention optimization
    bool use_memory_tiling;                 // Enable memory tiling
    bool use_gradient_checkpointing;        // Enable gradient checkpointing
    float attention_dropout_rate;           // Attention dropout rate
    uint32_t num_attention_heads;           // Number of attention heads (16)
    uint32_t head_dimension;                // Dimension per attention head (48)
} LongContextConfig;

// Flash Attention configuration
typedef struct {
    uint32_t block_size_q;                  // Query block size
    uint32_t block_size_k;                  // Key block size  
    uint32_t block_size_v;                  // Value block size
    uint32_t num_warps;                     // Number of GPU warps
    bool use_causal_mask;                   // Use causal attention mask
    bool use_softmax_scaling;               // Use scaled softmax
    float softmax_scale;                    // Softmax scaling factor
    uint32_t max_sequence_length;           // Maximum sequence length supported
} FlashAttentionConfig;

// Long-range dependency modeling types
typedef enum {
    LONG_DEPENDENCY_LOCAL = 0,      // Local dependencies (0-64 tokens)
    LONG_DEPENDENCY_MEDIUM,         // Medium range (64-256 tokens)
    LONG_DEPENDENCY_SECTION,        // Section level (256-512 tokens)
    LONG_DEPENDENCY_DOCUMENT,       // Document level (512-1024 tokens)
    LONG_DEPENDENCY_GLOBAL          // Global cross-document patterns
} LongDependencyRange;

// Memory efficiency optimization types
typedef enum {
    MEMORY_OPT_NONE = 0,           // No optimization
    MEMORY_OPT_GRADIENT_CHECKPOINTING,  // Gradient checkpointing
    MEMORY_OPT_MEMORY_TILING,      // Memory tiling
    MEMORY_OPT_FLASH_ATTENTION,    // Flash attention
    MEMORY_OPT_MIXED_PRECISION,    // Mixed precision
    MEMORY_OPT_ALL                 // All optimizations
} MemoryOptimizationType;

// Context processing statistics for long range
typedef struct {
    uint32_t total_attention_blocks;        // Total attention blocks processed
    uint64_t flash_attention_time_us;       // Flash attention processing time
    uint64_t memory_tiling_time_us;         // Memory tiling time
    float memory_efficiency_ratio;         // Memory efficiency (0.0-1.0)
    float attention_sparsity_ratio;        // Attention sparsity ratio
    size_t peak_memory_usage_bytes;        // Peak memory usage
    size_t average_memory_usage_bytes;     // Average memory usage
    uint32_t gradient_accumulation_steps;  // Gradient accumulation steps used
    float long_range_coherence_score;      // Long-range coherence score
    float document_structure_score;        // Document structure understanding
    uint32_t cache_hit_ratio;              // Attention cache hit ratio
} LongContextStats;

// Document structure analysis results
typedef struct {
    uint32_t section_boundaries[32];       // Section boundary positions
    uint32_t section_count;                // Number of sections found
    uint32_t chapter_boundaries[16];       // Chapter boundary positions  
    uint32_t chapter_count;                // Number of chapters found
    float hierarchical_structure_score;   // Hierarchical structure score
    LongDependencyRange primary_dependency; // Primary dependency range
    uint32_t global_reference_count;       // Global reference patterns
    float semantic_coherence_score;        // Semantic coherence across document
} DocumentStructureAnalysis;

// Attention pattern analysis
typedef struct {
    float local_attention_ratio;           // Local attention percentage
    float medium_attention_ratio;          // Medium-range attention percentage
    float global_attention_ratio;          // Global attention percentage
    uint32_t attention_hotspots[16];       // Positions with high attention
    uint32_t hotspot_count;                // Number of attention hotspots
    float attention_entropy;               // Attention distribution entropy
    float pattern_regularity_score;       // Attention pattern regularity
} AttentionPatternAnalysis;

// Error codes for long context processing
typedef enum {
    LONG_CONTEXT_SUCCESS = 0,
    LONG_CONTEXT_ERROR_INVALID_PARAM,
    LONG_CONTEXT_ERROR_MEMORY_ALLOCATION,
    LONG_CONTEXT_ERROR_CONTEXT_TOO_LONG,
    LONG_CONTEXT_ERROR_FLASH_ATTENTION_FAILED,
    LONG_CONTEXT_ERROR_MEMORY_TILING_FAILED,
    LONG_CONTEXT_ERROR_GRADIENT_CHECKPOINTING_FAILED,
    LONG_CONTEXT_ERROR_ATTENTION_COMPUTATION_FAILED,
    LONG_CONTEXT_ERROR_INSUFFICIENT_GPU_MEMORY,
    LONG_CONTEXT_ERROR_SEQUENCE_TOO_LONG_FOR_FLASH
} LongContextError;

// Core API Functions

/**
 * Create long context processor for 1024-token processing
 * @param processor Pointer to store created processor
 * @param config Long context configuration
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_create(LongContextProcessor** processor,
                                     const LongContextConfig* config);

/**
 * Initialize long context processor with CUDA enwik8 compatible settings
 * @param processor Long context processor instance
 * @param context_manager Parent adaptive context manager
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_initialize_cuda_compat(LongContextProcessor* processor,
                                                     AdaptiveContextManager* context_manager);

/**
 * Process 1024-token input with extended-range dependency modeling
 * @param processor Long context processor instance
 * @param input_tokens Input token sequence (up to 1024 tokens)
 * @param input_length Number of input tokens
 * @param output_tokens Output token buffer
 * @param output_length Maximum output length, actual length on return
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_process_tokens(LongContextProcessor* processor,
                                             const uint32_t* input_tokens,
                                             uint32_t input_length,
                                             uint32_t* output_tokens,
                                             uint32_t* output_length);

/**
 * Process tokens using Flash Attention optimization
 * @param processor Long context processor instance
 * @param input_tokens Input token sequence
 * @param input_length Number of input tokens
 * @param output_tokens Output token buffer
 * @param output_length Maximum output length, actual length on return
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_flash_attention_process(LongContextProcessor* processor,
                                                     const uint32_t* input_tokens,
                                                     uint32_t input_length,
                                                     uint32_t* output_tokens,
                                                     uint32_t* output_length);

/**
 * Analyze document-level structure in long context
 * @param processor Long context processor instance
 * @param input_tokens Input token sequence
 * @param input_length Number of input tokens
 * @param analysis Output document structure analysis
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_analyze_document_structure(LongContextProcessor* processor,
                                                        const uint32_t* input_tokens,
                                                        uint32_t input_length,
                                                        DocumentStructureAnalysis* analysis);

/**
 * Analyze attention patterns in long context
 * @param processor Long context processor instance
 * @param attention_weights Attention weight matrix (length x length)
 * @param input_length Number of tokens
 * @param pattern_analysis Output attention pattern analysis
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_analyze_attention_patterns(LongContextProcessor* processor,
                                                        const float* attention_weights,
                                                        uint32_t input_length,
                                                        AttentionPatternAnalysis* pattern_analysis);

/**
 * Model long-range dependencies across document
 * @param processor Long context processor instance
 * @param input_tokens Input token sequence
 * @param input_length Number of input tokens
 * @param dependency_matrix Output long-range dependency matrix
 * @param semantic_clusters Output semantic cluster assignments
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_model_long_range_dependencies(LongContextProcessor* processor,
                                                           const uint32_t* input_tokens,
                                                           uint32_t input_length,
                                                           float* dependency_matrix,
                                                           uint32_t* semantic_clusters);

/**
 * Optimize memory usage with tiling and checkpointing
 * @param processor Long context processor instance
 * @param optimization_type Type of memory optimization to apply
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_optimize_memory_usage(LongContextProcessor* processor,
                                                   MemoryOptimizationType optimization_type);

/**
 * Get long context processing statistics
 * @param processor Long context processor instance
 * @param stats Output processing statistics
 */
void long_context_get_stats(LongContextProcessor* processor,
                            LongContextStats* stats);

/**
 * Update long context configuration
 * @param processor Long context processor instance
 * @param config New configuration settings
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_update_config(LongContextProcessor* processor,
                                           const LongContextConfig* config);

/**
 * Destroy long context processor and free resources
 * @param processor Long context processor instance to destroy
 */
void long_context_destroy(LongContextProcessor* processor);

// Flash Attention Functions

/**
 * Initialize Flash Attention with optimal configuration
 * @param processor Long context processor instance
 * @param flash_config Flash attention configuration
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_initialize_flash_attention(LongContextProcessor* processor,
                                                        const FlashAttentionConfig* flash_config);

/**
 * Execute Flash Attention computation
 * @param processor Long context processor instance
 * @param query_tokens Query token sequence
 * @param key_tokens Key token sequence
 * @param value_tokens Value token sequence
 * @param sequence_length Sequence length
 * @param attention_output Output attention results
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_execute_flash_attention(LongContextProcessor* processor,
                                                     const float* query_tokens,
                                                     const float* key_tokens,
                                                     const float* value_tokens,
                                                     uint32_t sequence_length,
                                                     float* attention_output);

/**
 * Optimize Flash Attention block sizes for current hardware
 * @param processor Long context processor instance
 * @param sequence_length Target sequence length
 * @param optimal_config Output optimal Flash Attention configuration
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_optimize_flash_attention_blocks(LongContextProcessor* processor,
                                                             uint32_t sequence_length,
                                                             FlashAttentionConfig* optimal_config);

// Configuration Functions

/**
 * Create default long context configuration
 * @param config Output default configuration
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_create_default_config(LongContextConfig* config);

/**
 * Create CUDA enwik8 compatible long context configuration
 * @param config Output CUDA compatible configuration
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_create_cuda_config(LongContextConfig* config);

/**
 * Create Flash Attention configuration optimized for Apple Silicon
 * @param flash_config Output Flash Attention configuration
 * @param max_sequence_length Maximum sequence length to support
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_create_flash_attention_config(FlashAttentionConfig* flash_config,
                                                           uint32_t max_sequence_length);

/**
 * Validate long context configuration
 * @param config Configuration to validate
 * @param is_valid Output boolean for validity
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_validate_config(const LongContextConfig* config,
                                             bool* is_valid);

// Utility Functions

/**
 * Get error string for long context error code
 * @param error_code LongContextError code
 * @return Human-readable error message
 */
const char* long_context_get_error_string(LongContextError error_code);

/**
 * Convert long dependency range to string
 * @param range Long dependency range
 * @return Human-readable dependency range name
 */
const char* long_context_dependency_range_to_string(LongDependencyRange range);

/**
 * Convert memory optimization type to string
 * @param opt_type Memory optimization type
 * @return Human-readable optimization type name
 */
const char* long_context_memory_optimization_to_string(MemoryOptimizationType opt_type);

/**
 * Estimate memory requirements for long context processing
 * @param input_length Number of input tokens
 * @param use_flash_attention Whether Flash Attention is enabled
 * @param use_memory_tiling Whether memory tiling is enabled
 * @param estimated_memory_mb Output estimated memory requirement in MB
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_estimate_memory_requirements(uint32_t input_length,
                                                          bool use_flash_attention,
                                                          bool use_memory_tiling,
                                                          size_t* estimated_memory_mb);

/**
 * Calculate optimal Flash Attention block size for hardware
 * @param available_memory_mb Available GPU memory in MB
 * @param sequence_length Target sequence length
 * @param num_heads Number of attention heads
 * @param optimal_block_size Output optimal block size
 * @return LONG_CONTEXT_SUCCESS on success, error code on failure
 */
LongContextError long_context_calculate_optimal_block_size(size_t available_memory_mb,
                                                          uint32_t sequence_length,
                                                          uint32_t num_heads,
                                                          uint32_t* optimal_block_size);

// Constants for long context processing
#define LONG_CONTEXT_MAX_TOKENS 1024           // Maximum tokens for long context
#define LONG_CONTEXT_FLASH_BLOCK_SIZE 128      // Default Flash Attention block size
#define LONG_CONTEXT_MEMORY_TILE_SIZE 256      // Default memory tile size
#define LONG_CONTEXT_NUM_ATTENTION_HEADS 16    // Number of attention heads (CUDA enwik8)
#define LONG_CONTEXT_HEAD_DIMENSION 48         // Dimension per attention head (768/16)
#define LONG_CONTEXT_ATTENTION_DROPOUT 0.1f    // Default attention dropout rate

// Flash Attention constants
#define FLASH_ATTENTION_MIN_BLOCK_SIZE 32      // Minimum Flash Attention block size
#define FLASH_ATTENTION_MAX_BLOCK_SIZE 256     // Maximum Flash Attention block size
#define FLASH_ATTENTION_DEFAULT_NUM_WARPS 4    // Default number of GPU warps
#define FLASH_ATTENTION_SOFTMAX_SCALE 0.125f   // Default softmax scaling (1/sqrt(64))

// Memory optimization constants
#define LONG_CONTEXT_BASE_MEMORY_MB 256        // Base memory requirement
#define LONG_CONTEXT_FLASH_ATTENTION_OVERHEAD_MB 128  // Flash Attention memory overhead
#define LONG_CONTEXT_GRADIENT_CHECKPOINTING_SAVINGS 0.5f  // Memory savings with checkpointing

// Performance thresholds
#define LONG_CONTEXT_MIN_EFFICIENCY 0.7f       // Minimum efficiency ratio
#define LONG_CONTEXT_MAX_PROCESSING_TIME_MS 500 // Maximum processing time
#define LONG_CONTEXT_MIN_ATTENTION_SPARSITY 0.8f // Minimum attention sparsity for efficiency

#ifdef __cplusplus
}
#endif

#endif // LONG_CONTEXT_PROCESSOR_H
