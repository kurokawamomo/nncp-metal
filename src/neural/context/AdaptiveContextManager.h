/*
 * AdaptiveContextManager.h
 * 
 * Adaptive Context Management System for CUDA enwik8 compatible architecture
 * 3-level hierarchical context management (64/512/2048 tokens)
 * Dynamic context length selection with memory constraint adaptation
 */

#ifndef ADAPTIVE_CONTEXT_MANAGER_H
#define ADAPTIVE_CONTEXT_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct AdaptiveContextManager AdaptiveContextManager;

// Context levels based on CUDA enwik8 specifications
typedef enum {
    CONTEXT_LEVEL_SHORT = 0,        // 64 tokens (CUDA enwik8 seg_len)
    CONTEXT_LEVEL_MEDIUM,           // 512 tokens (intermediate)
    CONTEXT_LEVEL_LONG,             // 1024 tokens (extended)
    CONTEXT_LEVEL_MAXIMUM,          // 2048 tokens (CUDA enwik8 max_seq_len)
    CONTEXT_LEVEL_COUNT             // Total number of levels
} ContextLevel;

// Context selection strategies
typedef enum {
    CONTEXT_STRATEGY_FIXED = 0,     // Fixed context length
    CONTEXT_STRATEGY_ADAPTIVE,      // Adapt based on content analysis
    CONTEXT_STRATEGY_DYNAMIC,       // Dynamic based on memory constraints
    CONTEXT_STRATEGY_PROGRESSIVE,   // Progressive expansion as needed
    CONTEXT_STRATEGY_CUDA_COMPAT    // CUDA enwik8 compatible mode
} ContextStrategy;

// Content analysis types for context selection
typedef enum {
    CONTENT_TYPE_UNKNOWN = 0,
    CONTENT_TYPE_TEXT,              // Natural language text
    CONTENT_TYPE_CODE,              // Source code
    CONTENT_TYPE_BINARY,            // Binary data
    CONTENT_TYPE_STRUCTURED,        // Structured data (JSON, XML, etc.)
    CONTENT_TYPE_REPETITIVE         // Highly repetitive content
} ContentType;

// Context level configuration
typedef struct {
    ContextLevel level;             // Context level identifier
    uint32_t max_tokens;            // Maximum tokens for this level
    size_t memory_requirement_mb;   // Memory requirement in MB
    uint32_t attention_heads;       // Number of attention heads to use
    uint32_t batch_size;            // Recommended batch size
    float processing_cost_factor;   // Relative processing cost (1.0 = baseline)
    bool supports_long_range;       // Supports long-range dependencies
} ContextLevelConfig;

// Context selection metrics
typedef struct {
    ContentType detected_content_type;     // Auto-detected content type
    float content_complexity_score;       // Content complexity (0.0-1.0)
    float repetition_ratio;               // Repetition in content (0.0-1.0)
    uint32_t estimated_optimal_length;    // Estimated optimal context length
    ContextLevel recommended_level;        // Recommended context level
    float confidence_score;               // Confidence in recommendation (0.0-1.0)
} ContextSelectionMetrics;

// Memory constraint information
typedef struct {
    size_t available_memory_mb;           // Available memory in MB
    size_t current_usage_mb;             // Current memory usage in MB
    float memory_pressure_ratio;         // Memory pressure (0.0-1.0)
    bool low_memory_mode;                // Low memory mode activated
    ContextLevel max_affordable_level;   // Maximum affordable context level
} MemoryConstraints;

// Context processing statistics
typedef struct {
    ContextLevel current_level;          // Currently active level
    uint32_t current_context_length;    // Current context length in tokens
    uint64_t processing_time_microseconds; // Last processing time
    size_t memory_usage_bytes;          // Current memory usage
    float attention_efficiency;         // Attention computation efficiency
    float compression_effectiveness;     // Compression effectiveness ratio
    uint32_t context_switches;          // Number of context level switches
    uint64_t total_tokens_processed;    // Total tokens processed
} ContextProcessingStats;

// Error codes
typedef enum {
    CONTEXT_MANAGER_SUCCESS = 0,
    CONTEXT_MANAGER_ERROR_INVALID_PARAM,
    CONTEXT_MANAGER_ERROR_MEMORY_ALLOCATION,
    CONTEXT_MANAGER_ERROR_UNSUPPORTED_LEVEL,
    CONTEXT_MANAGER_ERROR_INSUFFICIENT_MEMORY,
    CONTEXT_MANAGER_ERROR_DEVICE_NOT_FOUND,
    CONTEXT_MANAGER_ERROR_BUFFER_ALLOCATION,
    CONTEXT_MANAGER_ERROR_CONTEXT_TOO_LONG,
    CONTEXT_MANAGER_ERROR_ANALYSIS_FAILED
} ContextManagerError;

// Core API Functions

/**
 * Create adaptive context manager with CUDA enwik8 compatible configuration
 * @param manager Pointer to store created context manager
 * @param strategy Context selection strategy to use
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_create(AdaptiveContextManager** manager,
                                          ContextStrategy strategy);

/**
 * Initialize context manager with CUDA enwik8 level configurations
 * Sets up 4-level hierarchy: 64 → 512 → 1024 → 2048 tokens
 * @param manager Context manager instance
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_initialize_cuda_levels(AdaptiveContextManager* manager);

/**
 * Analyze input content and recommend optimal context level
 * @param manager Context manager instance
 * @param input_data Input data buffer
 * @param input_size Size of input data in bytes
 * @param metrics Output context selection metrics
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_analyze_content(AdaptiveContextManager* manager,
                                                   const void* input_data,
                                                   size_t input_size,
                                                   ContextSelectionMetrics* metrics);

/**
 * Select optimal context level based on content analysis and memory constraints
 * @param manager Context manager instance
 * @param metrics Content analysis metrics
 * @param memory_constraints Current memory constraints
 * @param selected_level Output selected context level
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_select_context_level(AdaptiveContextManager* manager,
                                                        const ContextSelectionMetrics* metrics,
                                                        const MemoryConstraints* memory_constraints,
                                                        ContextLevel* selected_level);

/**
 * Set current context level and allocate necessary resources
 * @param manager Context manager instance
 * @param level Context level to activate
 * @param force_allocation Force resource allocation even if memory is tight
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_set_context_level(AdaptiveContextManager* manager,
                                                     ContextLevel level,
                                                     bool force_allocation);

/**
 * Process input data with current context configuration
 * @param manager Context manager instance
 * @param input_tokens Input token sequence
 * @param input_length Number of input tokens
 * @param output_tokens Output token buffer
 * @param output_length Maximum output length, actual length on return
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_process_tokens(AdaptiveContextManager* manager,
                                                  const uint32_t* input_tokens,
                                                  uint32_t input_length,
                                                  uint32_t* output_tokens,
                                                  uint32_t* output_length);

/**
 * Get current processing statistics
 * @param manager Context manager instance
 * @param stats Output processing statistics
 */
void context_manager_get_processing_stats(AdaptiveContextManager* manager,
                                         ContextProcessingStats* stats);

/**
 * Get memory constraints for current system state
 * @param manager Context manager instance
 * @param constraints Output memory constraints
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_get_memory_constraints(AdaptiveContextManager* manager,
                                                          MemoryConstraints* constraints);

/**
 * Destroy context manager and free all resources
 * @param manager Context manager instance to destroy
 */
void context_manager_destroy(AdaptiveContextManager* manager);

// Level Configuration Functions

/**
 * Get configuration for specific context level
 * @param manager Context manager instance
 * @param level Context level to query
 * @param config Output level configuration
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_get_level_config(AdaptiveContextManager* manager,
                                                    ContextLevel level,
                                                    ContextLevelConfig* config);

/**
 * Update configuration for specific context level
 * @param manager Context manager instance
 * @param level Context level to update
 * @param config New level configuration
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_set_level_config(AdaptiveContextManager* manager,
                                                    ContextLevel level,
                                                    const ContextLevelConfig* config);

/**
 * Check if context level is supported with current memory constraints
 * @param manager Context manager instance
 * @param level Context level to check
 * @param is_supported Output boolean for support status
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_is_level_supported(AdaptiveContextManager* manager,
                                                      ContextLevel level,
                                                      bool* is_supported);

// Content Analysis Functions

/**
 * Detect content type from input data
 * @param input_data Input data buffer
 * @param input_size Size of input data
 * @param content_type Output detected content type
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_detect_content_type(const void* input_data,
                                                       size_t input_size,
                                                       ContentType* content_type);

/**
 * Calculate content complexity score
 * @param input_data Input data buffer
 * @param input_size Size of input data
 * @param complexity_score Output complexity score (0.0-1.0)
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_calculate_complexity(const void* input_data,
                                                        size_t input_size,
                                                        float* complexity_score);

/**
 * Calculate repetition ratio in content
 * @param input_data Input data buffer
 * @param input_size Size of input data
 * @param repetition_ratio Output repetition ratio (0.0-1.0)
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_calculate_repetition(const void* input_data,
                                                        size_t input_size,
                                                        float* repetition_ratio);

// Utility Functions

/**
 * Get error string for error code
 * @param error_code ContextManagerError code
 * @return Human-readable error message
 */
const char* context_manager_get_error_string(ContextManagerError error_code);

/**
 * Convert context level to string
 * @param level Context level
 * @return Human-readable level name
 */
const char* context_manager_level_to_string(ContextLevel level);

/**
 * Convert context strategy to string
 * @param strategy Context strategy
 * @return Human-readable strategy name
 */
const char* context_manager_strategy_to_string(ContextStrategy strategy);

/**
 * Convert content type to string
 * @param content_type Content type
 * @return Human-readable content type name
 */
const char* context_manager_content_type_to_string(ContentType content_type);

/**
 * Estimate processing cost for given context level and input size
 * @param level Context level
 * @param input_size Size of input data
 * @param estimated_cost Output estimated cost (relative to baseline)
 * @return CONTEXT_MANAGER_SUCCESS on success, error code on failure
 */
ContextManagerError context_manager_estimate_processing_cost(ContextLevel level,
                                                           size_t input_size,
                                                           float* estimated_cost);

// CUDA enwik8 compatible constants
#define CUDA_ENWIK8_SHORT_CONTEXT 64        // Original CUDA seg_len
#define CUDA_ENWIK8_MEDIUM_CONTEXT 512      // Intermediate context
#define CUDA_ENWIK8_LONG_CONTEXT 1024       // Extended context
#define CUDA_ENWIK8_MAX_CONTEXT 2048        // Original CUDA max_seq_len

// Context level memory requirements (estimated in MB)
#define SHORT_CONTEXT_MEMORY_MB 64           // 64 tokens context
#define MEDIUM_CONTEXT_MEMORY_MB 256         // 512 tokens context
#define LONG_CONTEXT_MEMORY_MB 512           // 1024 tokens context
#define MAX_CONTEXT_MEMORY_MB 1024           // 2048 tokens context

// Processing cost factors (relative to short context)
#define SHORT_CONTEXT_COST_FACTOR 1.0f       // Baseline
#define MEDIUM_CONTEXT_COST_FACTOR 8.0f      // 8x cost (quadratic attention)
#define LONG_CONTEXT_COST_FACTOR 16.0f       // 16x cost
#define MAX_CONTEXT_COST_FACTOR 32.0f        // 32x cost

// Content analysis thresholds
#define LOW_COMPLEXITY_THRESHOLD 0.3f        // Low complexity content
#define HIGH_COMPLEXITY_THRESHOLD 0.7f       // High complexity content
#define HIGH_REPETITION_THRESHOLD 0.5f       // High repetition content
#define LOW_MEMORY_PRESSURE_THRESHOLD 0.7f   // Low memory pressure

#ifdef __cplusplus
}
#endif

#endif // ADAPTIVE_CONTEXT_MANAGER_H
