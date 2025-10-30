#include "../tokenizer/bpe_tokenizer.h"
#include "sequence_manager.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// Internal helper functions
static SequenceError create_window_context(SequenceManager* manager, uint32_t start_pos, uint32_t size, WindowContext** window);
static SequenceError calculate_window_boundaries(SequenceManager* manager, uint32_t position, WindowBoundary* boundary);
static bool is_cache_valid(WindowContext* window);
static SequenceError update_statistics(SequenceManager* manager, WindowContext* window, uint64_t processing_time);
static uint32_t calculate_adaptive_window_size(SequenceManager* manager, uint32_t position);
static bool find_word_boundary(TokenSequence* sequence, uint32_t position, bool forward);
static uint64_t get_timestamp_ns();

// Error messages
static const char* error_messages[] = {
    "Success",
    "Invalid parameter",
    "Memory allocation failed",
    "Buffer overflow",
    "Window size too small",
    "Invalid overlap configuration",
    "Sequence too large",
    "Checkpoint operation failed"
};

// Core API Implementation

SequenceError sequence_manager_create(SequenceManager** manager, const WindowConfig* config) {
    if (!manager || !config) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    // Validate configuration
    SequenceError error = window_config_validate(config);
    if (error != SEQ_SUCCESS) {
        return error;
    }
    
    *manager = (SequenceManager*)calloc(1, sizeof(SequenceManager));
    if (!*manager) {
        return SEQ_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize configuration
    (*manager)->config = *config;
    
    // Initialize cache
    (*manager)->cache_size = (config->max_memory_mb * 1024 * 1024) / 
                            (config->max_window_size * sizeof(uint32_t) * 2);
    if ((*manager)->cache_size < 2) {
        (*manager)->cache_size = 2; // Minimum cache size
    }
    
    (*manager)->window_cache = (WindowContext**)calloc((*manager)->cache_size, sizeof(WindowContext*));
    if (!(*manager)->window_cache) {
        free(*manager);
        *manager = NULL;
        return SEQ_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize checkpoints
    (*manager)->max_checkpoints = 10;
    (*manager)->checkpoint_positions = (uint32_t*)calloc((*manager)->max_checkpoints, sizeof(uint32_t));
    if (!(*manager)->checkpoint_positions) {
        free((*manager)->window_cache);
        free(*manager);
        *manager = NULL;
        return SEQ_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize state
    (*manager)->current_position = 0;
    (*manager)->is_processing = false;
    (*manager)->cache_count = 0;
    (*manager)->checkpoint_count = 0;
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_set_source(SequenceManager* manager, TokenSequence* sequence) {
    if (!manager || !sequence) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    // Check sequence size limits
    if (sequence->token_count > 10000000) { // 10M token limit
        return SEQ_ERROR_SEQUENCE_TOO_LARGE;
    }
    
    manager->source_sequence = sequence;
    manager->total_length = sequence->token_count;
    manager->current_position = 0;
    manager->is_processing = false;
    
    // Clear cache and reset statistics
    sequence_manager_clear_cache(manager);
    sequence_manager_reset_stats(manager);
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_get_next_window(SequenceManager* manager, WindowContext** window) {
    if (!manager || !window || !manager->source_sequence) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    if (!sequence_manager_has_next_window(manager)) {
        *window = NULL;
        return SEQ_SUCCESS;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    // Calculate window size
    uint32_t window_size = manager->config.preferred_window_size;
    if (manager->config.adaptive_sizing) {
        window_size = calculate_adaptive_window_size(manager, manager->current_position);
    }
    
    // Check cache first
    for (uint32_t i = 0; i < manager->cache_count; i++) {
        WindowContext* cached = manager->window_cache[i];
        if (cached && cached->boundary.start_position == manager->current_position &&
            cached->boundary.window_size == window_size && is_cache_valid(cached)) {
            cached->reference_count++;
            *window = cached;
            manager->stats.cache_hits++;
            return SEQ_SUCCESS;
        }
    }
    
    // Create new window
    SequenceError error = create_window_context(manager, manager->current_position, window_size, window);
    if (error != SEQ_SUCCESS) {
        return error;
    }
    
    // Update position for next window
    uint32_t advance = window_size - manager->config.overlap_size;
    if (manager->config.overlap_ratio > 0.0f) {
        advance = (uint32_t)(window_size * (1.0f - manager->config.overlap_ratio));
    }
    manager->current_position += advance;
    
    manager->stats.cache_misses++;
    
    // Update statistics
    uint64_t processing_time = get_timestamp_ns() - start_time;
    update_statistics(manager, *window, processing_time);
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_get_window_at(SequenceManager* manager, uint32_t position, WindowContext** window) {
    if (!manager || !window || !manager->source_sequence) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    if (position >= manager->total_length) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    // Calculate window size
    uint32_t window_size = manager->config.preferred_window_size;
    if (manager->config.adaptive_sizing) {
        window_size = calculate_adaptive_window_size(manager, position);
    }
    
    // Check cache first
    for (uint32_t i = 0; i < manager->cache_count; i++) {
        WindowContext* cached = manager->window_cache[i];
        if (cached && cached->boundary.start_position == position &&
            cached->boundary.window_size == window_size && is_cache_valid(cached)) {
            cached->reference_count++;
            *window = cached;
            manager->stats.cache_hits++;
            return SEQ_SUCCESS;
        }
    }
    
    // Create new window
    SequenceError error = create_window_context(manager, position, window_size, window);
    if (error != SEQ_SUCCESS) {
        return error;
    }
    
    manager->stats.cache_misses++;
    
    // Update statistics
    uint64_t processing_time = get_timestamp_ns() - start_time;
    update_statistics(manager, *window, processing_time);
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_release_window(SequenceManager* manager, WindowContext* window) {
    if (!manager || !window) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    window->reference_count--;
    
    // If no more references and cache is full, consider removing
    if (window->reference_count == 0 && manager->cache_count >= manager->cache_size) {
        // Simple LRU-like removal (remove oldest cached item)
        for (uint32_t i = 0; i < manager->cache_count; i++) {
            if (manager->window_cache[i] == window) {
                window_context_destroy(window);
                // Shift remaining cache entries
                for (uint32_t j = i; j < manager->cache_count - 1; j++) {
                    manager->window_cache[j] = manager->window_cache[j + 1];
                }
                manager->cache_count--;
                break;
            }
        }
    }
    
    return SEQ_SUCCESS;
}

bool sequence_manager_has_next_window(SequenceManager* manager) {
    if (!manager || !manager->source_sequence) {
        return false;
    }
    
    return manager->current_position < manager->total_length;
}

SequenceError sequence_manager_reset(SequenceManager* manager) {
    if (!manager) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    manager->current_position = 0;
    manager->is_processing = false;
    
    // Clear cache
    sequence_manager_clear_cache(manager);
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_seek(SequenceManager* manager, uint32_t position) {
    if (!manager || !manager->source_sequence) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    if (position >= manager->total_length) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    manager->current_position = position;
    return SEQ_SUCCESS;
}

// Window Configuration Functions

SequenceError window_config_create_default(WindowConfig* config) {
    if (!config) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    config->min_window_size = 512;
    config->max_window_size = 4096;
    config->preferred_window_size = 2048;
    config->overlap_size = 256;
    config->overlap_ratio = 0.125f; // 12.5% overlap
    config->max_memory_mb = 512;
    config->adaptive_sizing = true;
    config->preserve_boundaries = true;
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_update_config(SequenceManager* manager, const WindowConfig* config) {
    if (!manager || !config) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    SequenceError error = window_config_validate(config);
    if (error != SEQ_SUCCESS) {
        return error;
    }
    
    // Clear cache when configuration changes
    sequence_manager_clear_cache(manager);
    
    manager->config = *config;
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_calculate_optimal_window_size(SequenceManager* manager, 
                                                           uint32_t available_memory_mb,
                                                           uint32_t* optimal_size) {
    if (!manager || !optimal_size) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    // Estimate memory per token (token ID + position + metadata)
    uint32_t bytes_per_token = sizeof(uint32_t) * 3;
    uint32_t available_bytes = available_memory_mb * 1024 * 1024;
    
    // Reserve 25% for overhead and other structures
    available_bytes = (uint32_t)(available_bytes * 0.75f);
    
    uint32_t max_tokens = available_bytes / bytes_per_token;
    
    // Constrain to configuration limits
    *optimal_size = max_tokens;
    if (*optimal_size < manager->config.min_window_size) {
        *optimal_size = manager->config.min_window_size;
    }
    if (*optimal_size > manager->config.max_window_size) {
        *optimal_size = manager->config.max_window_size;
    }
    
    return SEQ_SUCCESS;
}

// Memory Management Functions

SequenceError sequence_manager_get_memory_usage(SequenceManager* manager, uint32_t* memory_usage_mb) {
    if (!manager || !memory_usage_mb) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    uint32_t total_bytes = sizeof(SequenceManager);
    
    // Add cache memory
    for (uint32_t i = 0; i < manager->cache_count; i++) {
        WindowContext* window = manager->window_cache[i];
        if (window) {
            total_bytes += sizeof(WindowContext);
            if (window->tokens) {
                total_bytes += window->tokens->token_count * sizeof(uint32_t) * 2; // tokens + positions
            }
            if (window->position_map) {
                total_bytes += window->boundary.window_size * sizeof(uint32_t);
            }
        }
    }
    
    // Add checkpoint memory
    total_bytes += manager->max_checkpoints * sizeof(uint32_t);
    
    *memory_usage_mb = (total_bytes + 1024 * 1024 - 1) / (1024 * 1024); // Round up
    
    manager->stats.memory_usage_mb = *memory_usage_mb;
    if (*memory_usage_mb > manager->stats.max_memory_usage_mb) {
        manager->stats.max_memory_usage_mb = *memory_usage_mb;
    }
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_clear_cache(SequenceManager* manager) {
    if (!manager) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    for (uint32_t i = 0; i < manager->cache_count; i++) {
        if (manager->window_cache[i]) {
            window_context_destroy(manager->window_cache[i]);
            manager->window_cache[i] = NULL;
        }
    }
    
    manager->cache_count = 0;
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_optimize_memory(SequenceManager* manager) {
    if (!manager) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    // Remove unreferenced windows
    uint32_t write_index = 0;
    for (uint32_t read_index = 0; read_index < manager->cache_count; read_index++) {
        WindowContext* window = manager->window_cache[read_index];
        if (window && window->reference_count > 0) {
            manager->window_cache[write_index] = window;
            write_index++;
        } else if (window) {
            window_context_destroy(window);
        }
    }
    
    // Clear remaining slots
    for (uint32_t i = write_index; i < manager->cache_count; i++) {
        manager->window_cache[i] = NULL;
    }
    
    manager->cache_count = write_index;
    
    return SEQ_SUCCESS;
}

// Checkpoint and Resume Functions

SequenceError sequence_manager_create_checkpoint(SequenceManager* manager, uint32_t* checkpoint_id) {
    if (!manager || !checkpoint_id) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    if (manager->checkpoint_count >= manager->max_checkpoints) {
        // Remove oldest checkpoint (FIFO)
        for (uint32_t i = 0; i < manager->checkpoint_count - 1; i++) {
            manager->checkpoint_positions[i] = manager->checkpoint_positions[i + 1];
        }
        manager->checkpoint_count--;
    }
    
    *checkpoint_id = manager->checkpoint_count;
    manager->checkpoint_positions[manager->checkpoint_count] = manager->current_position;
    manager->checkpoint_count++;
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_resume_from_checkpoint(SequenceManager* manager, uint32_t checkpoint_id) {
    if (!manager || checkpoint_id >= manager->checkpoint_count) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    manager->current_position = manager->checkpoint_positions[checkpoint_id];
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_list_checkpoints(SequenceManager* manager,
                                               uint32_t* checkpoint_ids,
                                               uint32_t max_checkpoints,
                                               uint32_t* actual_count) {
    if (!manager || !checkpoint_ids || !actual_count) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    *actual_count = (manager->checkpoint_count < max_checkpoints) ? 
                    manager->checkpoint_count : max_checkpoints;
    
    for (uint32_t i = 0; i < *actual_count; i++) {
        checkpoint_ids[i] = i;
    }
    
    return SEQ_SUCCESS;
}

// Statistics and Monitoring Functions

SequenceError sequence_manager_get_stats(SequenceManager* manager, SequenceStats* stats) {
    if (!manager || !stats) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    *stats = manager->stats;
    
    // Calculate derived statistics
    if (stats->total_tokens > 0) {
        stats->overlap_efficiency = (float)stats->overlap_tokens / stats->total_tokens;
    }
    
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_reset_stats(SequenceManager* manager) {
    if (!manager) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    memset(&manager->stats, 0, sizeof(SequenceStats));
    return SEQ_SUCCESS;
}

SequenceError sequence_manager_get_progress(SequenceManager* manager, float* progress) {
    if (!manager || !progress) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    if (manager->total_length == 0) {
        *progress = 0.0f;
    } else {
        *progress = (float)manager->current_position / manager->total_length;
        if (*progress > 1.0f) {
            *progress = 1.0f;
        }
    }
    
    return SEQ_SUCCESS;
}

// Utility Functions

const char* sequence_get_error_string(SequenceError error_code) {
    if (error_code < 0 || error_code >= sizeof(error_messages) / sizeof(error_messages[0])) {
        return "Unknown error";
    }
    return error_messages[error_code];
}

SequenceError window_config_validate(const WindowConfig* config) {
    if (!config) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    if (config->min_window_size == 0 || config->max_window_size == 0) {
        return SEQ_ERROR_WINDOW_TOO_SMALL;
    }
    
    if (config->min_window_size > config->max_window_size) {
        return SEQ_ERROR_WINDOW_TOO_SMALL;
    }
    
    if (config->preferred_window_size < config->min_window_size ||
        config->preferred_window_size > config->max_window_size) {
        return SEQ_ERROR_WINDOW_TOO_SMALL;
    }
    
    if (config->overlap_size >= config->preferred_window_size) {
        return SEQ_ERROR_OVERLAP_INVALID;
    }
    
    if (config->overlap_ratio < 0.0f || config->overlap_ratio >= 0.5f) {
        return SEQ_ERROR_OVERLAP_INVALID;
    }
    
    if (config->max_memory_mb == 0) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    return SEQ_SUCCESS;
}

SequenceError window_config_calculate_memory_requirements(const WindowConfig* config,
                                                         uint32_t sequence_length,
                                                         uint32_t* memory_mb) {
    if (!config || !memory_mb) {
        return SEQ_ERROR_INVALID_PARAM;
    }
    
    // Calculate worst-case memory usage
    uint32_t num_windows = sequence_length / (config->preferred_window_size - config->overlap_size) + 1;
    uint32_t cache_size = (config->max_memory_mb * 1024 * 1024) / 
                         (config->max_window_size * sizeof(uint32_t) * 2);
    
    if (cache_size > num_windows) {
        cache_size = num_windows;
    }
    
    uint32_t total_bytes = sizeof(SequenceManager);
    total_bytes += cache_size * sizeof(WindowContext*);
    total_bytes += cache_size * (sizeof(WindowContext) + config->max_window_size * sizeof(uint32_t) * 3);
    total_bytes += 10 * sizeof(uint32_t); // Checkpoints
    
    *memory_mb = (total_bytes + 1024 * 1024 - 1) / (1024 * 1024);
    
    return SEQ_SUCCESS;
}

void sequence_manager_destroy(SequenceManager* manager) {
    if (!manager) {
        return;
    }
    
    // Clear cache
    sequence_manager_clear_cache(manager);
    
    // Free cache array
    if (manager->window_cache) {
        free(manager->window_cache);
    }
    
    // Free checkpoints
    if (manager->checkpoint_positions) {
        free(manager->checkpoint_positions);
    }
    
    free(manager);
}

void window_context_destroy(WindowContext* window) {
    if (!window) {
        return;
    }
    
    if (window->tokens) {
        token_sequence_destroy(window->tokens);
    }
    
    if (window->position_map) {
        free(window->position_map);
    }
    
    free(window);
}

// Internal helper function implementations

static SequenceError create_window_context(SequenceManager* manager, uint32_t start_pos, uint32_t size, WindowContext** window) {
    *window = (WindowContext*)calloc(1, sizeof(WindowContext));
    if (!*window) {
        return SEQ_ERROR_MEMORY_ALLOCATION;
    }
    
    // Calculate boundaries
    SequenceError error = calculate_window_boundaries(manager, start_pos, &(*window)->boundary);
    if (error != SEQ_SUCCESS) {
        free(*window);
        *window = NULL;
        return error;
    }
    
    // Create token sequence for this window
    uint32_t actual_size = (*window)->boundary.window_size;
    uint32_t end_pos = start_pos + actual_size;
    if (end_pos > manager->total_length) {
        end_pos = manager->total_length;
        actual_size = end_pos - start_pos;
    }
    
    // Allocate token sequence
    (*window)->tokens = (TokenSequence*)calloc(1, sizeof(TokenSequence));
    if (!(*window)->tokens) {
        free(*window);
        *window = NULL;
        return SEQ_ERROR_MEMORY_ALLOCATION;
    }
    
    (*window)->tokens->tokens = (uint32_t*)malloc(actual_size * sizeof(uint32_t));
    (*window)->tokens->positions = (uint32_t*)malloc(actual_size * sizeof(uint32_t));
    if (!(*window)->tokens->tokens || !(*window)->tokens->positions) {
        window_context_destroy(*window);
        *window = NULL;
        return SEQ_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy tokens from source sequence
    for (uint32_t i = 0; i < actual_size; i++) {
        (*window)->tokens->tokens[i] = manager->source_sequence->tokens[start_pos + i];
        (*window)->tokens->positions[i] = manager->source_sequence->positions[start_pos + i];
    }
    
    (*window)->tokens->token_count = actual_size;
    (*window)->tokens->is_valid = true;
    (*window)->tokens->confidence = manager->source_sequence->confidence;
    
    // Create position map
    (*window)->position_map = (uint32_t*)malloc(actual_size * sizeof(uint32_t));
    if (!(*window)->position_map) {
        window_context_destroy(*window);
        *window = NULL;
        return SEQ_ERROR_MEMORY_ALLOCATION;
    }
    
    for (uint32_t i = 0; i < actual_size; i++) {
        (*window)->position_map[i] = start_pos + i;
    }
    
    // Initialize context
    (*window)->is_cached = false;
    (*window)->cache_timestamp = get_timestamp_ns();
    (*window)->reference_count = 1;
    
    // Add to cache if space available
    if (manager->cache_count < manager->cache_size) {
        manager->window_cache[manager->cache_count] = *window;
        (*window)->is_cached = true;
        manager->cache_count++;
    }
    
    return SEQ_SUCCESS;
}

static SequenceError calculate_window_boundaries(SequenceManager* manager, uint32_t position, WindowBoundary* boundary) {
    boundary->start_position = position;
    boundary->window_size = manager->config.preferred_window_size;
    
    // Adjust for adaptive sizing
    if (manager->config.adaptive_sizing) {
        boundary->window_size = calculate_adaptive_window_size(manager, position);
    }
    
    // Ensure we don't exceed sequence length
    if (position + boundary->window_size > manager->total_length) {
        boundary->window_size = manager->total_length - position;
    }
    
    boundary->end_position = position + boundary->window_size;
    
    // Calculate overlap
    if (position > 0) {
        boundary->overlap_start = position;
        boundary->overlap_end = position + manager->config.overlap_size;
        if (boundary->overlap_end > boundary->end_position) {
            boundary->overlap_end = boundary->end_position;
        }
    } else {
        boundary->overlap_start = 0;
        boundary->overlap_end = 0;
    }
    
    // Check boundary alignment
    boundary->is_boundary_aligned = true;
    if (manager->config.preserve_boundaries && manager->source_sequence) {
        // Simple boundary preservation: ensure we don't break in middle of common words
        if (boundary->end_position < manager->total_length) {
            boundary->is_boundary_aligned = find_word_boundary(manager->source_sequence, 
                                                              boundary->end_position, false);
        }
    }
    
    boundary->sequence_id = (uint32_t)(get_timestamp_ns() & 0xFFFFFFFF);
    
    return SEQ_SUCCESS;
}

static bool is_cache_valid(WindowContext* window) {
    // Simple cache validation - check if window is not too old
    uint64_t current_time = get_timestamp_ns();
    uint64_t max_age_ns = 300 * 1000000000ULL; // 5 minutes
    
    return (current_time - window->cache_timestamp) < max_age_ns;
}

static SequenceError update_statistics(SequenceManager* manager, WindowContext* window, uint64_t processing_time) {
    manager->stats.total_windows++;
    manager->stats.total_tokens += window->boundary.window_size;
    manager->stats.processing_time_ns += processing_time;
    
    if (window->boundary.overlap_end > window->boundary.overlap_start) {
        manager->stats.overlap_tokens += (window->boundary.overlap_end - window->boundary.overlap_start);
    }
    
    return SEQ_SUCCESS;
}

static uint32_t calculate_adaptive_window_size(SequenceManager* manager, uint32_t position) {
    // Simple adaptive sizing based on position and available memory
    uint32_t base_size = manager->config.preferred_window_size;
    
    // Reduce window size near end of sequence
    uint32_t remaining = manager->total_length - position;
    if (remaining < base_size) {
        return remaining;
    }
    
    // Increase window size if we have good cache performance
    float cache_hit_rate = 0.0f;
    if (manager->stats.cache_hits + manager->stats.cache_misses > 0) {
        cache_hit_rate = (float)manager->stats.cache_hits / 
                        (manager->stats.cache_hits + manager->stats.cache_misses);
    }
    
    if (cache_hit_rate > 0.8f && base_size < manager->config.max_window_size) {
        base_size = (uint32_t)(base_size * 1.2f);
        if (base_size > manager->config.max_window_size) {
            base_size = manager->config.max_window_size;
        }
    }
    
    return base_size;
}

static bool find_word_boundary(TokenSequence* sequence, uint32_t position, bool forward) {
    // Simplified word boundary detection
    // In a full implementation, this would use linguistic rules
    (void)sequence; // Suppress unused parameter warning
    (void)position;
    (void)forward;
    
    return true; // For now, assume all positions are valid boundaries
}

static uint64_t get_timestamp_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

#ifdef USE_METAL
// Metal memory manager integration placeholder
SequenceError sequence_manager_create_with_metal_memory(SequenceManager** manager,
                                                       const WindowConfig* config,
                                                       MemoryManager* memory_manager) {
    // For now, use standard memory allocation
    // TODO: Integrate with Metal memory manager for window buffers
    (void)memory_manager; // Suppress unused parameter warning
    return sequence_manager_create(manager, config);
}
#endif
