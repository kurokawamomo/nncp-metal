#ifndef SEQUENCE_MANAGER_H
#define SEQUENCE_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../tokenizer/bpe_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for sequence manager
typedef enum {
    SEQ_SUCCESS = 0,
    SEQ_ERROR_INVALID_PARAM,
    SEQ_ERROR_MEMORY_ALLOCATION,
    SEQ_ERROR_BUFFER_OVERFLOW,
    SEQ_ERROR_WINDOW_TOO_SMALL,
    SEQ_ERROR_OVERLAP_INVALID,
    SEQ_ERROR_SEQUENCE_TOO_LARGE,
    SEQ_ERROR_CHECKPOINT_FAILED
} SequenceError;

// Window processing configuration
typedef struct {
    uint32_t min_window_size;      // Minimum window size (tokens)
    uint32_t max_window_size;      // Maximum window size (tokens)  
    uint32_t preferred_window_size; // Preferred window size (tokens)
    uint32_t overlap_size;         // Overlap between windows (tokens)
    float overlap_ratio;           // Overlap as ratio of window size (0.0-0.5)
    uint32_t max_memory_mb;        // Maximum memory usage (MB)
    bool adaptive_sizing;          // Enable adaptive window sizing
    bool preserve_boundaries;      // Preserve word/sentence boundaries
} WindowConfig;

// Processing statistics  
typedef struct {
    uint32_t total_windows;        // Total number of windows processed
    uint32_t total_tokens;         // Total tokens processed
    uint32_t overlap_tokens;       // Total overlap tokens
    uint32_t memory_usage_mb;      // Current memory usage (MB)
    uint32_t max_memory_usage_mb;  // Peak memory usage (MB)
    float overlap_efficiency;      // Overlap efficiency ratio
    uint64_t processing_time_ns;   // Total processing time (nanoseconds)
    uint32_t cache_hits;           // Number of cache hits
    uint32_t cache_misses;         // Number of cache misses
} SequenceStats;

// Window boundary information
typedef struct {
    uint32_t start_position;       // Start position in original sequence
    uint32_t end_position;         // End position in original sequence
    uint32_t window_size;          // Actual window size
    uint32_t overlap_start;        // Overlap start position
    uint32_t overlap_end;          // Overlap end position
    bool is_boundary_aligned;      // Whether aligned to word boundaries
    uint32_t sequence_id;          // Unique sequence identifier
} WindowBoundary;

// Window processing context
typedef struct {
    TokenSequence* tokens;         // Window token sequence
    WindowBoundary boundary;       // Window boundary information
    uint32_t* position_map;        // Map to original positions
    bool is_cached;                // Whether this window is cached
    uint64_t cache_timestamp;      // Cache timestamp
    uint32_t reference_count;      // Reference counting for memory management
} WindowContext;

// Sequence window manager
typedef struct {
    WindowConfig config;           // Window configuration
    SequenceStats stats;           // Processing statistics
    TokenSequence* source_sequence; // Source token sequence
    WindowContext** window_cache;  // Window cache array
    uint32_t cache_size;           // Cache capacity
    uint32_t cache_count;          // Current cache entries
    uint32_t current_position;     // Current processing position
    uint32_t total_length;         // Total sequence length
    bool is_processing;            // Processing state flag
    uint32_t* checkpoint_positions; // Checkpoint positions for resumable processing
    uint32_t checkpoint_count;     // Number of checkpoints
    uint32_t max_checkpoints;      // Maximum checkpoints to keep
} SequenceManager;

// Core API Functions

/**
 * Create a new sequence window manager
 * @param manager Pointer to store created manager
 * @param config Window configuration
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_create(SequenceManager** manager, const WindowConfig* config);

/**
 * Initialize manager with source sequence
 * @param manager Sequence manager instance
 * @param sequence Source token sequence to process
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_set_source(SequenceManager* manager, TokenSequence* sequence);

/**
 * Get next window for processing
 * @param manager Sequence manager instance
 * @param window Pointer to store window context
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_get_next_window(SequenceManager* manager, WindowContext** window);

/**
 * Get window at specific position
 * @param manager Sequence manager instance
 * @param position Start position in sequence
 * @param window Pointer to store window context
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_get_window_at(SequenceManager* manager, uint32_t position, WindowContext** window);

/**
 * Release window context (decrements reference count)
 * @param manager Sequence manager instance
 * @param window Window context to release
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_release_window(SequenceManager* manager, WindowContext* window);

/**
 * Check if there are more windows to process
 * @param manager Sequence manager instance
 * @return true if more windows available, false otherwise
 */
bool sequence_manager_has_next_window(SequenceManager* manager);

/**
 * Reset processing to beginning of sequence
 * @param manager Sequence manager instance
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_reset(SequenceManager* manager);

/**
 * Seek to specific position in sequence
 * @param manager Sequence manager instance
 * @param position Position to seek to
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_seek(SequenceManager* manager, uint32_t position);

// Window Configuration Functions

/**
 * Create default window configuration
 * @param config Pointer to store default configuration
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError window_config_create_default(WindowConfig* config);

/**
 * Update window configuration dynamically
 * @param manager Sequence manager instance
 * @param config New configuration
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_update_config(SequenceManager* manager, const WindowConfig* config);

/**
 * Calculate optimal window size based on available memory
 * @param manager Sequence manager instance
 * @param available_memory_mb Available memory in MB
 * @param optimal_size Pointer to store optimal window size
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_calculate_optimal_window_size(SequenceManager* manager, 
                                                           uint32_t available_memory_mb,
                                                           uint32_t* optimal_size);

// Memory Management Functions

/**
 * Get current memory usage
 * @param manager Sequence manager instance
 * @param memory_usage_mb Pointer to store memory usage in MB
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_get_memory_usage(SequenceManager* manager, uint32_t* memory_usage_mb);

/**
 * Clear window cache to free memory
 * @param manager Sequence manager instance
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_clear_cache(SequenceManager* manager);

/**
 * Optimize memory usage by removing unused windows
 * @param manager Sequence manager instance
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_optimize_memory(SequenceManager* manager);

// Checkpoint and Resume Functions

/**
 * Create checkpoint at current position
 * @param manager Sequence manager instance
 * @param checkpoint_id Pointer to store checkpoint ID
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_create_checkpoint(SequenceManager* manager, uint32_t* checkpoint_id);

/**
 * Resume from checkpoint
 * @param manager Sequence manager instance
 * @param checkpoint_id Checkpoint ID to resume from
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_resume_from_checkpoint(SequenceManager* manager, uint32_t checkpoint_id);

/**
 * List available checkpoints
 * @param manager Sequence manager instance
 * @param checkpoint_ids Array to store checkpoint IDs
 * @param max_checkpoints Maximum number of checkpoints to return
 * @param actual_count Pointer to store actual number of checkpoints
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_list_checkpoints(SequenceManager* manager,
                                               uint32_t* checkpoint_ids,
                                               uint32_t max_checkpoints,
                                               uint32_t* actual_count);

// Statistics and Monitoring Functions

/**
 * Get processing statistics
 * @param manager Sequence manager instance
 * @param stats Pointer to store statistics
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_get_stats(SequenceManager* manager, SequenceStats* stats);

/**
 * Reset statistics counters
 * @param manager Sequence manager instance
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_reset_stats(SequenceManager* manager);

/**
 * Get processing progress as percentage
 * @param manager Sequence manager instance
 * @param progress Pointer to store progress (0.0-1.0)
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_get_progress(SequenceManager* manager, float* progress);

// Utility Functions

/**
 * Get error message string
 * @param error_code Sequence error code
 * @return Human-readable error message
 */
const char* sequence_get_error_string(SequenceError error_code);

/**
 * Validate window configuration
 * @param config Configuration to validate
 * @return SEQ_SUCCESS if valid, error code otherwise
 */
SequenceError window_config_validate(const WindowConfig* config);

/**
 * Calculate memory requirements for configuration
 * @param config Window configuration
 * @param sequence_length Length of sequence to process
 * @param memory_mb Pointer to store memory requirement in MB
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError window_config_calculate_memory_requirements(const WindowConfig* config,
                                                         uint32_t sequence_length,
                                                         uint32_t* memory_mb);

/**
 * Free sequence manager memory
 * @param manager Sequence manager to free
 */
void sequence_manager_destroy(SequenceManager* manager);

/**
 * Free window context memory
 * @param window Window context to free
 */
void window_context_destroy(WindowContext* window);

// Integration with existing memory management
#ifdef USE_METAL
#include "memory_manager.h"

/**
 * Create sequence manager using Metal memory manager
 * @param manager Pointer to store created manager
 * @param config Window configuration
 * @param memory_manager Metal memory manager instance
 * @return SEQ_SUCCESS on success, error code on failure
 */
SequenceError sequence_manager_create_with_metal_memory(SequenceManager** manager,
                                                       const WindowConfig* config,
                                                       MemoryManager* memory_manager);
#endif

#ifdef __cplusplus
}
#endif

#endif // SEQUENCE_MANAGER_H
