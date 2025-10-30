/*
 * MediumContextProcessor.h
 * 
 * 512-token Medium Context Processing System
 * Implements middle-range dependency modeling with CPU-GPU hybrid processing
 * Compatible with CUDA enwik8 extended context specifications
 */

#ifndef MEDIUM_CONTEXT_PROCESSOR_H
#define MEDIUM_CONTEXT_PROCESSOR_H

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
typedef struct MediumContextProcessor MediumContextProcessor;

// Medium context configuration (512 tokens)
typedef struct {
    uint32_t max_tokens;                    // 512 tokens (CUDA enwik8 medium)
    uint32_t segment_size;                  // Segment processing size
    uint32_t overlap_tokens;                // Overlap between segments
    float attention_threshold;              // Attention weight threshold
    bool use_hybrid_processing;             // CPU-GPU hybrid mode
    uint32_t gpu_batch_size;               // GPU batch processing size
    uint32_t cpu_parallel_threads;         // CPU parallel threads
} MediumContextConfig;

// Dependency modeling types
typedef enum {
    DEPENDENCY_LOCAL = 0,        // Local dependencies (0-64 tokens)
    DEPENDENCY_PARAGRAPH,        // Paragraph-level (64-256 tokens)
    DEPENDENCY_SECTION,          // Section-level (256-512 tokens)
    DEPENDENCY_GLOBAL            // Global cross-section
} DependencyRange;

// Context processing statistics for medium range
typedef struct {
    uint32_t total_segments_processed;      // Total segments processed
    uint32_t dependency_links_found;        // Dependency links discovered
    float paragraph_coherence_score;       // Paragraph coherence (0.0-1.0)
    float section_structure_score;         // Section structure score (0.0-1.0)
    uint64_t gpu_processing_time_us;       // GPU processing time
    uint64_t cpu_processing_time_us;       // CPU processing time
    size_t memory_usage_bytes;             // Memory usage
    float hybrid_efficiency_ratio;        // CPU-GPU efficiency ratio
} MediumContextStats;

// Structure recognition results
typedef struct {
    uint32_t paragraph_boundaries[64];     // Paragraph boundary positions
    uint32_t paragraph_count;              // Number of paragraphs found
    uint32_t sentence_boundaries[256];     // Sentence boundary positions
    uint32_t sentence_count;               // Number of sentences found
    float structural_complexity;          // Structural complexity score
    DependencyRange primary_dependency;    // Primary dependency range
} StructureAnalysis;

// Error codes for medium context processing
typedef enum {
    MEDIUM_CONTEXT_SUCCESS = 0,
    MEDIUM_CONTEXT_ERROR_INVALID_PARAM,
    MEDIUM_CONTEXT_ERROR_MEMORY_ALLOCATION,
    MEDIUM_CONTEXT_ERROR_CONTEXT_TOO_LONG,
    MEDIUM_CONTEXT_ERROR_GPU_FAILURE,
    MEDIUM_CONTEXT_ERROR_CPU_FAILURE,
    MEDIUM_CONTEXT_ERROR_HYBRID_SYNC_FAILED,
    MEDIUM_CONTEXT_ERROR_STRUCTURE_ANALYSIS_FAILED
} MediumContextError;

// Core API Functions

/**
 * Create medium context processor for 512-token processing
 * @param processor Pointer to store created processor
 * @param config Medium context configuration
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_create(MediumContextProcessor** processor,
                                         const MediumContextConfig* config);

/**
 * Initialize medium context processor with CUDA enwik8 compatible settings
 * @param processor Medium context processor instance
 * @param context_manager Parent adaptive context manager
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_initialize_cuda_compat(MediumContextProcessor* processor,
                                                         AdaptiveContextManager* context_manager);

/**
 * Process 512-token input with middle-range dependency modeling
 * @param processor Medium context processor instance
 * @param input_tokens Input token sequence (up to 512 tokens)
 * @param input_length Number of input tokens
 * @param output_tokens Output token buffer
 * @param output_length Maximum output length, actual length on return
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_process_tokens(MediumContextProcessor* processor,
                                                 const uint32_t* input_tokens,
                                                 uint32_t input_length,
                                                 uint32_t* output_tokens,
                                                 uint32_t* output_length);

/**
 * Analyze paragraph-level structure in medium context
 * @param processor Medium context processor instance
 * @param input_tokens Input token sequence
 * @param input_length Number of input tokens
 * @param analysis Output structure analysis results
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_analyze_structure(MediumContextProcessor* processor,
                                                    const uint32_t* input_tokens,
                                                    uint32_t input_length,
                                                    StructureAnalysis* analysis);

/**
 * Perform CPU-GPU hybrid processing for medium context
 * @param processor Medium context processor instance
 * @param input_tokens Input token sequence
 * @param input_length Number of input tokens
 * @param use_gpu_priority Prioritize GPU processing when true
 * @param output_tokens Output token buffer
 * @param output_length Maximum output length, actual length on return
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_hybrid_process(MediumContextProcessor* processor,
                                                 const uint32_t* input_tokens,
                                                 uint32_t input_length,
                                                 bool use_gpu_priority,
                                                 uint32_t* output_tokens,
                                                 uint32_t* output_length);

/**
 * Model middle-range dependencies in token sequence
 * @param processor Medium context processor instance
 * @param input_tokens Input token sequence
 * @param input_length Number of input tokens
 * @param dependency_matrix Output dependency weight matrix (length x length)
 * @param primary_dependencies Output primary dependency relationships
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_model_dependencies(MediumContextProcessor* processor,
                                                     const uint32_t* input_tokens,
                                                     uint32_t input_length,
                                                     float* dependency_matrix,
                                                     uint32_t* primary_dependencies);

/**
 * Get medium context processing statistics
 * @param processor Medium context processor instance
 * @param stats Output processing statistics
 */
void medium_context_get_stats(MediumContextProcessor* processor,
                              MediumContextStats* stats);

/**
 * Update medium context configuration
 * @param processor Medium context processor instance
 * @param config New configuration settings
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_update_config(MediumContextProcessor* processor,
                                                const MediumContextConfig* config);

/**
 * Check if input length is suitable for medium context processing
 * @param processor Medium context processor instance
 * @param input_length Number of input tokens
 * @param is_suitable Output boolean for suitability
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_is_suitable_length(MediumContextProcessor* processor,
                                                     uint32_t input_length,
                                                     bool* is_suitable);

/**
 * Destroy medium context processor and free resources
 * @param processor Medium context processor instance to destroy
 */
void medium_context_destroy(MediumContextProcessor* processor);

// Configuration Functions

/**
 * Create default medium context configuration
 * @param config Output default configuration
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_create_default_config(MediumContextConfig* config);

/**
 * Create CUDA enwik8 compatible medium context configuration
 * @param config Output CUDA compatible configuration
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_create_cuda_config(MediumContextConfig* config);

/**
 * Validate medium context configuration
 * @param config Configuration to validate
 * @param is_valid Output boolean for validity
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_validate_config(const MediumContextConfig* config,
                                                  bool* is_valid);

// Utility Functions

/**
 * Get error string for medium context error code
 * @param error_code MediumContextError code
 * @return Human-readable error message
 */
const char* medium_context_get_error_string(MediumContextError error_code);

/**
 * Convert dependency range to string
 * @param range Dependency range
 * @return Human-readable dependency range name
 */
const char* medium_context_dependency_range_to_string(DependencyRange range);

/**
 * Estimate processing cost for medium context
 * @param input_length Number of input tokens
 * @param use_hybrid Whether to use hybrid processing
 * @param estimated_cost Output estimated cost (relative to short context)
 * @return MEDIUM_CONTEXT_SUCCESS on success, error code on failure
 */
MediumContextError medium_context_estimate_cost(uint32_t input_length,
                                               bool use_hybrid,
                                               float* estimated_cost);

// Constants for medium context processing
#define MEDIUM_CONTEXT_MAX_TOKENS 512          // Maximum tokens for medium context
#define MEDIUM_CONTEXT_SEGMENT_SIZE 128        // Default segment size
#define MEDIUM_CONTEXT_OVERLAP_TOKENS 32       // Default overlap between segments
#define MEDIUM_CONTEXT_GPU_BATCH_SIZE 16       // Default GPU batch size
#define MEDIUM_CONTEXT_CPU_THREADS 4           // Default CPU threads
#define MEDIUM_CONTEXT_ATTENTION_THRESHOLD 0.1f // Attention weight threshold

// Memory requirements for medium context (estimated)
#define MEDIUM_CONTEXT_BASE_MEMORY_MB 128       // Base memory requirement
#define MEDIUM_CONTEXT_GPU_MEMORY_MB 64         // Additional GPU memory
#define MEDIUM_CONTEXT_HYBRID_OVERHEAD_MB 32    // Hybrid processing overhead

// Performance thresholds
#define MEDIUM_CONTEXT_MIN_EFFICIENCY 0.6f     // Minimum efficiency ratio
#define MEDIUM_CONTEXT_MAX_PROCESSING_TIME_MS 100  // Maximum processing time

#ifdef __cplusplus
}
#endif

#endif // MEDIUM_CONTEXT_PROCESSOR_H
