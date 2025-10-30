/*
 * Small File Handler Implementation
 * 
 * Ensures identical LSTM processing for all file sizes with specialized
 * handling for small files that require padding and careful segmentation.
 */

#include "small_file_handler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Error message strings
static const char* small_file_error_messages[] = {
    "Success",
    "Invalid input parameters",
    "Memory allocation failed",
    "Padding operation failed",
    "LSTM processing failed",
    "CUDA compatibility violation",
    "Quality threshold not met",
    "Invalid metadata",
    "Decompression failed"
};

// Category thresholds (bytes)
static const size_t category_thresholds[SMALL_FILE_CATEGORY_COUNT] = {
    256,      // TINY: 0-256
    1024,     // SMALL: 257-1024
    4096,     // MEDIUM: 1025-4096
    16384,    // LARGE: 4097-16384
    SIZE_MAX  // VERY_LARGE: 16385+
};

// Configuration management
SmallFileConfig* small_file_config_create_for_category(SmallFileCategory category) {
    SmallFileConfig* config = (SmallFileConfig*)calloc(1, sizeof(SmallFileConfig));
    if (!config) {
        return NULL;
    }
    
    config->category = category;
    config->enforce_cuda_alignment = true;
    config->use_deterministic_padding = true;
    config->padding_seed = 42;
    config->enable_content_analysis = true;
    config->enable_entropy_preservation = true;
    config->min_compression_ratio = 0.1f;  // 10% minimum compression
    config->cache_padding_patterns = true;
    config->optimize_for_speed = false;    // Default to quality
    config->max_memory_usage_bytes = 64 * 1024 * 1024;  // 64MB
    
    // Category-specific settings
    switch (category) {
        case SMALL_FILE_TINY:
            config->padding_strategy = PADDING_STRATEGY_REPEAT;
            config->target_segment_length = 20;  // LSTM-optimized
            config->preserve_original_length = true;
            break;
            
        case SMALL_FILE_SMALL:
            config->padding_strategy = PADDING_STRATEGY_CUDA_COMPATIBLE;
            config->target_segment_length = 32;  // Default CUDA
            config->preserve_original_length = true;
            break;
            
        case SMALL_FILE_MEDIUM:
            config->padding_strategy = PADDING_STRATEGY_STATISTICAL;
            config->target_segment_length = 32;  // Default CUDA
            config->preserve_original_length = true;
            break;
            
        case SMALL_FILE_LARGE:
            config->padding_strategy = PADDING_STRATEGY_REFLECT;
            config->target_segment_length = 64;  // enwik8-like
            config->preserve_original_length = false;  // Can infer from context
            break;
            
        default:
            config->padding_strategy = PADDING_STRATEGY_ZERO;
            config->target_segment_length = 32;
            config->preserve_original_length = false;
            break;
    }
    
    return config;
}

SmallFileConfig* small_file_config_create_cuda_compatible(const CUDAProfile* profile) {
    if (!profile) {
        return NULL;
    }
    
    SmallFileConfig* config = (SmallFileConfig*)calloc(1, sizeof(SmallFileConfig));
    if (!config) {
        return NULL;
    }
    
    // Use profile's seg_len as target segment length
    config->target_segment_length = profile->params.seg_len;
    config->padding_strategy = PADDING_STRATEGY_CUDA_COMPATIBLE;
    config->enforce_cuda_alignment = true;
    config->use_deterministic_padding = true;
    config->padding_seed = 42;  // CUDA-compatible seed
    
    // Profile-specific optimizations
    if (profile->is_lstm_optimized) {
        config->enable_content_analysis = true;
        config->enable_entropy_preservation = true;
        config->optimize_for_speed = false;
    } else {
        config->enable_content_analysis = false;
        config->enable_entropy_preservation = false;
        config->optimize_for_speed = true;
    }
    
    config->preserve_original_length = true;
    config->min_compression_ratio = 0.05f;  // 5% minimum for CUDA compatibility
    config->cache_padding_patterns = true;
    config->max_memory_usage_bytes = profile->params.memory_budget_mb * 1024 * 1024;
    
    return config;
}

void small_file_config_free(SmallFileConfig* config) {
    if (config) {
        free(config);
    }
}

// File categorization
SmallFileCategory small_file_categorize(size_t file_size) {
    for (int i = 0; i < SMALL_FILE_CATEGORY_COUNT; i++) {
        if (file_size <= category_thresholds[i]) {
            return (SmallFileCategory)i;
        }
    }
    return SMALL_FILE_VERY_LARGE;
}

bool small_file_requires_special_handling(size_t file_size) {
    return small_file_categorize(file_size) != SMALL_FILE_VERY_LARGE;
}

size_t small_file_estimate_processing_memory(size_t file_size, const SmallFileConfig* config) {
    if (!config) {
        return 0;
    }
    
    // Estimate memory requirements
    size_t padded_size = config->target_segment_length;
    if (padded_size < file_size) {
        padded_size = ((file_size / config->target_segment_length) + 1) * config->target_segment_length;
    }
    
    // Original data + padded data + padding pattern + LSTM context + overhead
    size_t estimated = file_size + padded_size + 1024 + (4 * 1024 * 1024) + (1024 * 1024);
    
    return estimated;
}

// Handler lifecycle
SmallFileHandler* small_file_handler_create(const SmallFileConfig* config) {
    if (!config) {
        return NULL;
    }
    
    SmallFileHandler* handler = (SmallFileHandler*)calloc(1, sizeof(SmallFileHandler));
    if (!handler) {
        return NULL;
    }
    
    // Copy configuration
    memcpy(&handler->config, config, sizeof(SmallFileConfig));
    
    // Create CUDA-compatible LSTM configuration
    EnhancedLSTMConfig* lstm_config = enhanced_lstm_config_create_default();
    if (!lstm_config) {
        free(handler);
        return NULL;
    }
    
    // Override with small file specific settings
    lstm_config->seg_len = config->target_segment_length;
    lstm_config->train_len = config->target_segment_length;  // Critical: train_len = seg_len
    lstm_config->enforce_deterministic = config->use_deterministic_padding;
    lstm_config->random_seed = config->padding_seed;
    
    // Create enhanced LSTM context
    handler->lstm_context = enhanced_lstm_context_create(lstm_config);
    enhanced_lstm_config_free(lstm_config);
    
    if (!handler->lstm_context) {
        free(handler);
        return NULL;
    }
    
    handler->processing_successful = false;
    
    return handler;
}

void small_file_handler_free(SmallFileHandler* handler) {
    if (!handler) {
        return;
    }
    
    // Free data buffers
    if (handler->original_data) {
        free(handler->original_data);
    }
    if (handler->padded_data) {
        free(handler->padded_data);
    }
    if (handler->padding_pattern) {
        free(handler->padding_pattern);
    }
    
    // Free LSTM context
    if (handler->lstm_context) {
        enhanced_lstm_context_free(handler->lstm_context);
    }
    
    free(handler);
}

bool small_file_handler_reset(SmallFileHandler* handler) {
    if (!handler) {
        return false;
    }
    
    // Clear data buffers
    if (handler->original_data) {
        free(handler->original_data);
        handler->original_data = NULL;
    }
    if (handler->padded_data) {
        free(handler->padded_data);
        handler->padded_data = NULL;
    }
    if (handler->padding_pattern) {
        free(handler->padding_pattern);
        handler->padding_pattern = NULL;
    }
    
    handler->original_size = 0;
    handler->padded_size = 0;
    handler->padding_pattern_size = 0;
    handler->processing_successful = false;
    handler->processing_time_ms = 0.0;
    handler->compression_ratio = 0.0f;
    handler->total_operations = 0;
    handler->quality_score = 0.0f;
    memset(handler->last_error_message, 0, sizeof(handler->last_error_message));
    
    // Reset LSTM context (stub implementation)
    if (handler->lstm_context) {
        // Reset LSTM state manually since enhanced_lstm_reset_state is not implemented
        // This is a simplified reset - in a full implementation, this would reset internal LSTM state
        return true;
    }
    
    return true;
}

// Core processing functions
bool small_file_handler_process(SmallFileHandler* handler, 
                               const uint8_t* input_data, 
                               size_t input_size,
                               uint8_t** output_data,
                               size_t* output_size) {
    if (!handler || !input_data || input_size == 0 || !output_data || !output_size) {
        if (handler) {
            snprintf(handler->last_error_message, sizeof(handler->last_error_message),
                    "Invalid input parameters");
        }
        return false;
    }
    
    clock_t start_time = clock();
    
    // Reset handler state
    if (!small_file_handler_reset(handler)) {
        snprintf(handler->last_error_message, sizeof(handler->last_error_message),
                "Failed to reset handler state");
        return false;
    }
    
    // Store original data
    handler->original_data = (uint8_t*)malloc(input_size);
    if (!handler->original_data) {
        snprintf(handler->last_error_message, sizeof(handler->last_error_message),
                "Failed to allocate memory for original data");
        return false;
    }
    memcpy(handler->original_data, input_data, input_size);
    handler->original_size = input_size;
    
    // Analyze entropy of original data
    handler->entropy_original = small_file_analyze_entropy(input_data, input_size);
    
    // Apply padding to reach target segment length
    if (!small_file_apply_padding(handler, input_data, input_size,
                                 &handler->padded_data, &handler->padded_size)) {
        snprintf(handler->last_error_message, sizeof(handler->last_error_message),
                "Failed to apply padding");
        return false;
    }
    
    // Analyze entropy of padded data
    handler->entropy_padded = small_file_analyze_entropy(handler->padded_data, handler->padded_size);
    
    // Validate CUDA compatibility
    if (!small_file_validate_cuda_compatibility(handler)) {
        snprintf(handler->last_error_message, sizeof(handler->last_error_message),
                "CUDA compatibility validation failed");
        return false;
    }
    
    // Process through LSTM
    float* probabilities = (float*)malloc(handler->padded_size * 256 * sizeof(float));
    if (!probabilities) {
        snprintf(handler->last_error_message, sizeof(handler->last_error_message),
                "Failed to allocate memory for LSTM output");
        return false;
    }
    
    bool lstm_success = enhanced_lstm_process_segment(handler->lstm_context,
                                                     handler->padded_data,
                                                     handler->padded_size,
                                                     probabilities);
    
    if (!lstm_success) {
        free(probabilities);
        snprintf(handler->last_error_message, sizeof(handler->last_error_message),
                "LSTM processing failed");
        return false;
    }
    
    // For this implementation, we'll create a simple compressed format
    // In a real implementation, this would use entropy coding based on LSTM probabilities
    
    // Calculate compressed size estimate (simplified)
    size_t compressed_size = input_size / 2 + 64;  // Simplified compression estimate
    if (compressed_size < 32) compressed_size = 32;
    
    *output_data = (uint8_t*)malloc(compressed_size);
    if (!*output_data) {
        free(probabilities);
        snprintf(handler->last_error_message, sizeof(handler->last_error_message),
                "Failed to allocate memory for output");
        return false;
    }
    
    // Create simplified compressed format (placeholder)
    // Header: [original_size:4][padding_strategy:1][checksum:4][compressed_data:...]
    uint32_t* header = (uint32_t*)*output_data;
    header[0] = (uint32_t)input_size;
    (*output_data)[4] = (uint8_t)handler->config.padding_strategy;
    
    // Simple checksum
    uint32_t checksum = 0;
    for (size_t i = 0; i < input_size; i++) {
        checksum ^= input_data[i];
    }
    memcpy(*output_data + 5, &checksum, 4);
    
    // Simplified compression (just copy data for now)
    size_t remaining_size = compressed_size - 9;
    if (remaining_size > input_size) remaining_size = input_size;
    memcpy(*output_data + 9, input_data, remaining_size);
    *output_size = 9 + remaining_size;
    
    free(probabilities);
    
    // Calculate metrics
    handler->compression_ratio = (float)*output_size / (float)input_size;
    handler->total_operations++;
    handler->quality_score = small_file_calculate_quality_score(handler);
    
    clock_t end_time = clock();
    handler->processing_time_ms = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000.0;
    
    handler->processing_successful = true;
    return true;
}

// Padding implementation
bool small_file_apply_padding(SmallFileHandler* handler,
                             const uint8_t* input_data,
                             size_t input_size,
                             uint8_t** padded_data,
                             size_t* padded_size) {
    if (!handler || !input_data || !padded_data || !padded_size) {
        return false;
    }
    
    size_t target_size = handler->config.target_segment_length;
    
    // If input is already at or larger than target size, use as-is
    if (input_size >= target_size) {
        *padded_data = (uint8_t*)malloc(input_size);
        if (!*padded_data) {
            return false;
        }
        memcpy(*padded_data, input_data, input_size);
        *padded_size = input_size;
        return true;
    }
    
    // Apply padding strategy
    switch (handler->config.padding_strategy) {
        case PADDING_STRATEGY_ZERO:
            return small_file_pad_zero(input_data, input_size, padded_data, target_size);
            
        case PADDING_STRATEGY_REPEAT:
            return small_file_pad_repeat(input_data, input_size, padded_data, target_size);
            
        case PADDING_STRATEGY_REFLECT:
            return small_file_pad_reflect(input_data, input_size, padded_data, target_size);
            
        case PADDING_STRATEGY_STATISTICAL:
            return small_file_pad_statistical(input_data, input_size, padded_data, target_size,
                                             handler->config.padding_seed);
            
        case PADDING_STRATEGY_CUDA_COMPATIBLE:
            // For CUDA compatibility, use repeat padding with deterministic pattern
            return small_file_pad_repeat(input_data, input_size, padded_data, target_size);
            
        default:
            return small_file_pad_zero(input_data, input_size, padded_data, target_size);
    }
}

// Specific padding implementations
bool small_file_pad_zero(const uint8_t* input, size_t input_size, 
                        uint8_t** output, size_t target_size) {
    if (!input || !output || target_size < input_size) {
        return false;
    }
    
    *output = (uint8_t*)calloc(target_size, 1);
    if (!*output) {
        return false;
    }
    
    memcpy(*output, input, input_size);
    // Rest is already zero from calloc
    
    return true;
}

bool small_file_pad_repeat(const uint8_t* input, size_t input_size,
                          uint8_t** output, size_t target_size) {
    if (!input || !output || input_size == 0 || target_size < input_size) {
        return false;
    }
    
    *output = (uint8_t*)malloc(target_size);
    if (!*output) {
        return false;
    }
    
    // Copy original data
    memcpy(*output, input, input_size);
    
    // Repeat data to fill remaining space
    size_t remaining = target_size - input_size;
    size_t pos = input_size;
    
    while (remaining > 0) {
        size_t copy_size = (remaining > input_size) ? input_size : remaining;
        memcpy(*output + pos, input, copy_size);
        pos += copy_size;
        remaining -= copy_size;
    }
    
    return true;
}

bool small_file_pad_reflect(const uint8_t* input, size_t input_size,
                           uint8_t** output, size_t target_size) {
    if (!input || !output || input_size == 0 || target_size < input_size) {
        return false;
    }
    
    *output = (uint8_t*)malloc(target_size);
    if (!*output) {
        return false;
    }
    
    // Copy original data
    memcpy(*output, input, input_size);
    
    // Reflect data to fill remaining space
    size_t remaining = target_size - input_size;
    size_t pos = input_size;
    bool forward = false;  // Start with reverse
    
    while (remaining > 0) {
        size_t copy_size = (remaining > input_size) ? input_size : remaining;
        
        if (forward) {
            memcpy(*output + pos, input, copy_size);
        } else {
            // Copy in reverse
            for (size_t i = 0; i < copy_size; i++) {
                (*output)[pos + i] = input[input_size - 1 - i];
            }
        }
        
        pos += copy_size;
        remaining -= copy_size;
        forward = !forward;  // Alternate direction
    }
    
    return true;
}

bool small_file_pad_statistical(const uint8_t* input, size_t input_size,
                               uint8_t** output, size_t target_size,
                               uint64_t seed) {
    if (!input || !output || input_size == 0 || target_size < input_size) {
        return false;
    }
    
    *output = (uint8_t*)malloc(target_size);
    if (!*output) {
        return false;
    }
    
    // Copy original data
    memcpy(*output, input, input_size);
    
    // Calculate byte frequency distribution
    uint32_t freq[256] = {0};
    for (size_t i = 0; i < input_size; i++) {
        freq[input[i]]++;
    }
    
    // Simple statistical padding based on frequency
    // This is a simplified implementation
    CUDACompatRNG rng;
    cuda_compat_rng_init(&rng, seed);
    
    for (size_t i = input_size; i < target_size; i++) {
        // Pick a byte based on frequency (simplified)
        float rand_val = cuda_compat_rng_uniform(&rng, 0.0f, 1.0f);
        uint32_t random = (uint32_t)(rand_val * input_size) % input_size;
        (*output)[i] = input[random];
    }
    
    return true;
}

// Content analysis
float small_file_analyze_entropy(const uint8_t* data, size_t size) {
    if (!data || size == 0) {
        return 0.0f;
    }
    
    uint32_t freq[256] = {0};
    for (size_t i = 0; i < size; i++) {
        freq[data[i]]++;
    }
    
    float entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (freq[i] > 0) {
            float p = (float)freq[i] / (float)size;
            entropy -= p * log2f(p);
        }
    }
    
    return entropy;
}

PaddingStrategy small_file_recommend_padding_strategy(const uint8_t* data, size_t size) {
    if (!data || size == 0) {
        return PADDING_STRATEGY_ZERO;
    }
    
    float entropy = small_file_analyze_entropy(data, size);
    
    if (entropy < 2.0f) {
        // Low entropy - repeat padding works well
        return PADDING_STRATEGY_REPEAT;
    } else if (entropy > 6.0f) {
        // High entropy - statistical padding better
        return PADDING_STRATEGY_STATISTICAL;
    } else {
        // Medium entropy - reflect padding provides good balance
        return PADDING_STRATEGY_REFLECT;
    }
}

// CUDA compatibility validation
bool small_file_validate_cuda_compatibility(const SmallFileHandler* handler) {
    if (!handler || !handler->lstm_context) {
        return false;
    }
    
    // Check train_len = seg_len relationship
    if (!enhanced_lstm_check_seg_len_consistency(&handler->lstm_context->config)) {
        return false;
    }
    
    // Check segment alignment
    if (!small_file_check_segment_alignment(handler)) {
        return false;
    }
    
    return true;
}

bool small_file_check_segment_alignment(const SmallFileHandler* handler) {
    if (!handler || !handler->padded_data) {
        return false;
    }
    
    // Check that padded size is compatible with target segment length
    int32_t seg_len = handler->config.target_segment_length;
    return (handler->padded_size >= (size_t)seg_len);
}

// Quality assessment
float small_file_calculate_quality_score(const SmallFileHandler* handler) {
    if (!handler || !handler->processing_successful) {
        return 0.0f;
    }
    
    float score = 1.0f;
    
    // Compression ratio component (0.4 weight)
    float compression_component = handler->compression_ratio * 0.4f;
    if (compression_component > 0.4f) compression_component = 0.4f;
    
    // Entropy preservation component (0.3 weight)
    float entropy_ratio = handler->entropy_padded / (handler->entropy_original + 1e-6f);
    float entropy_component = (1.0f - fabsf(1.0f - entropy_ratio)) * 0.3f;
    
    // CUDA compatibility component (0.3 weight)
    float cuda_component = small_file_validate_cuda_compatibility(handler) ? 0.3f : 0.0f;
    
    score = compression_component + entropy_component + cuda_component;
    
    return fminf(1.0f, fmaxf(0.0f, score));
}

// Utility functions
const char* small_file_error_string(SmallFileError error) {
    if (error < 0 || error >= sizeof(small_file_error_messages) / sizeof(small_file_error_messages[0])) {
        return "Unknown error";
    }
    return small_file_error_messages[error];
}

void small_file_handler_print_stats(const SmallFileHandler* handler) {
    if (!handler) {
        return;
    }
    
    printf("Small File Handler Statistics:\n");
    printf("  Category: %d\n", handler->config.category);
    printf("  Padding Strategy: %d\n", handler->config.padding_strategy);
    printf("  Original Size: %zu bytes\n", handler->original_size);
    printf("  Padded Size: %zu bytes\n", handler->padded_size);
    printf("  Processing Time: %.2f ms\n", handler->processing_time_ms);
    printf("  Compression Ratio: %.3f\n", handler->compression_ratio);
    printf("  Quality Score: %.3f\n", handler->quality_score);
    printf("  Original Entropy: %.3f\n", handler->entropy_original);
    printf("  Padded Entropy: %.3f\n", handler->entropy_padded);
    printf("  Processing Successful: %s\n", handler->processing_successful ? "Yes" : "No");
    
    if (strlen(handler->last_error_message) > 0) {
        printf("  Last Error: %s\n", handler->last_error_message);
    }
}

// Integration helpers
bool small_file_should_use_handler(size_t file_size, const CUDAProfile* profile) {
    if (!profile) {
        return small_file_requires_special_handling(file_size);
    }
    
    // Always use handler for files smaller than seg_len
    return (file_size < (size_t)profile->params.seg_len) || 
           small_file_requires_special_handling(file_size);
}

SmallFileHandler* small_file_create_handler_for_profile(const CUDAProfile* profile) {
    if (!profile) {
        return NULL;
    }
    
    SmallFileConfig* config = small_file_config_create_cuda_compatible(profile);
    if (!config) {
        return NULL;
    }
    
    SmallFileHandler* handler = small_file_handler_create(config);
    small_file_config_free(config);
    
    return handler;
}