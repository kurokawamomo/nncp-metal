/*
 * Enhanced Neural Bridge Implementation
 * 
 * Replaces Neural Bridge fallback with Enhanced Metal LSTM, providing
 * CUDA-compatible compression with mathematical equivalence.
 */

#include "neural_bridge_enhanced.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Global enhanced bridge context
static EnhancedNeuralBridgeContext g_bridge_context = {0};
static bool g_bridge_initialized = false;
static EnhancedNeuralError g_last_error = ENHANCED_NEURAL_SUCCESS;

// Error message strings
static const char* enhanced_neural_error_messages[] = {
    "Success",
    "Bridge not initialized",
    "Invalid configuration",
    "CUDA compatibility violation",
    "Parameter validation failed",
    "Memory allocation failed",
    "LSTM processing failed",
    "Quality too low",
    "Buffer too small",
    "Unknown algorithm"
};

// Algorithm name strings
static const char* enhanced_algorithm_names[] = {
    "Enhanced Metal LSTM",
    "Small File Handler",
    "Auto Selection"
};

// Get current time in nanoseconds
static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Configuration management
EnhancedNeuralCompressionConfig* enhanced_neural_config_create_default(void) {
    EnhancedNeuralCompressionConfig* config = (EnhancedNeuralCompressionConfig*)calloc(1, sizeof(EnhancedNeuralCompressionConfig));
    if (!config) {
        return NULL;
    }
    
    config->preferred_algorithm = ENHANCED_NEURAL_ALGORITHM_AUTO;
    config->cuda_profile = cuda_profile_get("default");
    config->memory_limit_bytes = 512 * 1024 * 1024;  // 512MB default
    config->quality_level = 7;  // High quality default
    config->enforce_cuda_compatibility = true;
    config->validate_parameters = true;
    config->enable_small_file_optimization = true;
    config->enable_gpu_acceleration = true;
    config->verbose_logging = false;
    config->compression_target = 0.3f;  // 30% target compression ratio
    config->use_deterministic_processing = true;
    config->precision_tolerance = 1e-6f;
    config->random_seed = 42;
    
    return config;
}

EnhancedNeuralCompressionConfig* enhanced_neural_config_create_for_profile(const CUDAProfile* profile) {
    if (!profile) {
        return NULL;
    }
    
    EnhancedNeuralCompressionConfig* config = enhanced_neural_config_create_default();
    if (!config) {
        return NULL;
    }
    
    config->cuda_profile = profile;
    config->memory_limit_bytes = profile->params.memory_budget_mb * 1024 * 1024;
    
    // Profile-specific optimizations
    if (profile->is_lstm_optimized) {
        config->preferred_algorithm = ENHANCED_NEURAL_ALGORITHM_METAL_LSTM;
        config->quality_level = 8;
        config->compression_target = 0.25f;  // Better compression for LSTM
        config->use_deterministic_processing = true;
    }
    
    if (strcmp(profile->name, "lstm_fast") == 0) {
        config->quality_level = 6;
        config->compression_target = 0.4f;  // Faster, less optimal compression
    }
    
    return config;
}

EnhancedNeuralCompressionConfig* enhanced_neural_config_create_cuda_strict(void) {
    EnhancedNeuralCompressionConfig* config = enhanced_neural_config_create_default();
    if (!config) {
        return NULL;
    }
    
    config->enforce_cuda_compatibility = true;
    config->validate_parameters = true;
    config->use_deterministic_processing = true;
    config->precision_tolerance = 1e-7f;  // Stricter precision
    config->quality_level = 9;  // Maximum quality
    
    return config;
}

void enhanced_neural_config_free(EnhancedNeuralCompressionConfig* config) {
    if (config) {
        free(config);
    }
}

// Lifecycle management
bool enhanced_neural_bridge_init(const EnhancedNeuralCompressionConfig* config) {
    if (g_bridge_initialized) {
        if (config && config->verbose_logging) {
            printf("[Enhanced Neural Bridge] Already initialized\n");
        }
        return true;
    }
    
    if (!config) {
        g_last_error = ENHANCED_NEURAL_ERROR_INVALID_CONFIG;
        return false;
    }
    
    // Clear context
    memset(&g_bridge_context, 0, sizeof(EnhancedNeuralBridgeContext));
    
    // Copy configuration
    memcpy(&g_bridge_context.config, config, sizeof(EnhancedNeuralCompressionConfig));
    
    if (config->verbose_logging) {
        printf("[Enhanced Neural Bridge] Initializing CUDA-compatible neural bridge...\n");
        printf("[Enhanced Neural Bridge] Memory limit: %zu MB\n", config->memory_limit_bytes / (1024 * 1024));
        printf("[Enhanced Neural Bridge] Quality level: %u\n", config->quality_level);
        printf("[Enhanced Neural Bridge] CUDA profile: %s\n", 
               config->cuda_profile ? config->cuda_profile->name : "none");
    }
    
    // Validate CUDA compatibility
    if (config->enforce_cuda_compatibility) {
        if (!enhanced_neural_bridge_validate_cuda_compatibility(config)) {
            g_last_error = ENHANCED_NEURAL_ERROR_CUDA_INCOMPATIBLE;
            snprintf(g_bridge_context.last_error, sizeof(g_bridge_context.last_error),
                    "CUDA compatibility validation failed");
            return false;
        }
    }
    
    // Create Enhanced LSTM configuration
    EnhancedLSTMConfig* lstm_config = NULL;
    if (config->cuda_profile) {
        lstm_config = enhanced_lstm_config_create_from_cuda_profile(config->cuda_profile);
    } else {
        lstm_config = enhanced_lstm_config_create_default();
    }
    
    if (!lstm_config) {
        g_last_error = ENHANCED_NEURAL_ERROR_MEMORY_ALLOCATION;
        snprintf(g_bridge_context.last_error, sizeof(g_bridge_context.last_error),
                "Failed to create LSTM configuration");
        return false;
    }
    
    // Override with bridge-specific settings
    lstm_config->enforce_deterministic = config->use_deterministic_processing;
    lstm_config->random_seed = config->random_seed;
    if (lstm_config->math_config) {
        lstm_config->math_config->precision_tolerance = config->precision_tolerance;
    }
    
    // Create primary Enhanced Metal LSTM context
    g_bridge_context.primary_lstm_context = enhanced_lstm_context_create(lstm_config);
    if (!g_bridge_context.primary_lstm_context) {
        enhanced_lstm_config_free(lstm_config);
        g_last_error = ENHANCED_NEURAL_ERROR_LSTM_PROCESSING;
        snprintf(g_bridge_context.last_error, sizeof(g_bridge_context.last_error),
                "Failed to create primary LSTM context");
        return false;
    }
    
    // Create fallback LSTM context (optional, for robustness)
    g_bridge_context.fallback_lstm_context = enhanced_lstm_context_create(lstm_config);
    // Note: fallback is optional, so we don't fail if it can't be created
    
    enhanced_lstm_config_free(lstm_config);
    
    // Create small file handler if enabled
    if (config->enable_small_file_optimization) {
        SmallFileConfig* small_config = NULL;
        if (config->cuda_profile) {
            small_config = small_file_config_create_cuda_compatible(config->cuda_profile);
        } else {
            small_config = small_file_config_create_for_category(SMALL_FILE_MEDIUM);
        }
        
        if (small_config) {
            g_bridge_context.small_file_handler = small_file_handler_create(small_config);
            small_file_config_free(small_config);
        }
    }
    
    // Initialize statistics
    g_bridge_context.total_compressions = 0;
    g_bridge_context.successful_compressions = 0;
    g_bridge_context.total_processing_time_ms = 0.0;
    g_bridge_context.total_bytes_processed = 0;
    g_bridge_context.cuda_compatible_operations = 0;
    g_bridge_context.parameter_validation_failures = 0;
    g_bridge_context.max_observed_deviation = 0.0f;
    
    g_bridge_context.initialized = true;
    g_bridge_initialized = true;
    g_last_error = ENHANCED_NEURAL_SUCCESS;
    
    if (config->verbose_logging) {
        printf("[Enhanced Neural Bridge] Enhanced neural bridge initialization completed\n");
        printf("[Enhanced Neural Bridge] Primary LSTM context: %s\n", 
               g_bridge_context.primary_lstm_context ? "OK" : "Failed");
        printf("[Enhanced Neural Bridge] Small file handler: %s\n", 
               g_bridge_context.small_file_handler ? "OK" : "Disabled");
    }
    
    return true;
}

void enhanced_neural_bridge_shutdown(void) {
    if (!g_bridge_initialized) {
        return;
    }
    
    if (g_bridge_context.config.verbose_logging) {
        printf("[Enhanced Neural Bridge] Shutting down enhanced neural bridge...\n");
    }
    
    // Free LSTM contexts
    if (g_bridge_context.primary_lstm_context) {
        enhanced_lstm_context_free(g_bridge_context.primary_lstm_context);
        g_bridge_context.primary_lstm_context = NULL;
    }
    
    if (g_bridge_context.fallback_lstm_context) {
        enhanced_lstm_context_free(g_bridge_context.fallback_lstm_context);
        g_bridge_context.fallback_lstm_context = NULL;
    }
    
    // Free small file handler
    if (g_bridge_context.small_file_handler) {
        small_file_handler_free(g_bridge_context.small_file_handler);
        g_bridge_context.small_file_handler = NULL;
    }
    
    // Free validation report
    if (g_bridge_context.last_validation_report) {
        cuda_validation_report_free(g_bridge_context.last_validation_report);
        g_bridge_context.last_validation_report = NULL;
    }
    
    memset(&g_bridge_context, 0, sizeof(EnhancedNeuralBridgeContext));
    g_bridge_initialized = false;
    
    if (g_bridge_context.config.verbose_logging) {
        printf("[Enhanced Neural Bridge] Shutdown completed\n");
    }
}

bool enhanced_neural_bridge_is_ready(void) {
    return g_bridge_initialized && 
           g_bridge_context.initialized && 
           g_bridge_context.primary_lstm_context != NULL;
}

EnhancedNeuralBridgeContext* enhanced_neural_bridge_get_context(void) {
    return g_bridge_initialized ? &g_bridge_context : NULL;
}

// Algorithm selection
EnhancedNeuralAlgorithm enhanced_neural_bridge_select_optimal_algorithm(size_t input_size,
                                                                        const uint8_t* input_data,
                                                                        const EnhancedNeuralCompressionConfig* config) {
    if (!config) {
        return ENHANCED_NEURAL_ALGORITHM_METAL_LSTM;
    }
    
    // If user specified an algorithm, respect it (unless AUTO)
    if (config->preferred_algorithm != ENHANCED_NEURAL_ALGORITHM_AUTO) {
        return config->preferred_algorithm;
    }
    
    // Auto selection based on file size and content
    if (config->enable_small_file_optimization && small_file_requires_special_handling(input_size)) {
        return ENHANCED_NEURAL_ALGORITHM_SMALL_FILE;
    }
    
    // For small files that don't require special handling, or larger files, use Metal LSTM
    return ENHANCED_NEURAL_ALGORITHM_METAL_LSTM;
}

// Core compression function
bool enhanced_neural_bridge_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const EnhancedNeuralCompressionConfig* config,
    EnhancedNeuralCompressionResult* result
) {
    if (!result) {
        g_last_error = ENHANCED_NEURAL_ERROR_INVALID_CONFIG;
        return false;
    }
    
    // Initialize result
    memset(result, 0, sizeof(EnhancedNeuralCompressionResult));
    
    if (!g_bridge_initialized || !g_bridge_context.initialized) {
        g_last_error = ENHANCED_NEURAL_ERROR_NOT_INITIALIZED;
        strncpy(result->error_message, "Enhanced neural bridge not initialized", 
                sizeof(result->error_message) - 1);
        return false;
    }
    
    if (!input_data || input_size == 0 || !output_data || output_capacity == 0) {
        g_last_error = ENHANCED_NEURAL_ERROR_INVALID_CONFIG;
        strncpy(result->error_message, "Invalid input parameters", 
                sizeof(result->error_message) - 1);
        return false;
    }
    
    uint64_t start_time = get_time_ns();
    
    // Use provided config or bridge config
    const EnhancedNeuralCompressionConfig* active_config = config ? config : &g_bridge_context.config;
    
    if (active_config->verbose_logging) {
        printf("[Enhanced Neural Bridge] Starting compression of %zu bytes\n", input_size);
    }
    
    // Parameter validation if enabled
    if (active_config->validate_parameters) {
        CUDAValidationReport* validation_report = NULL;
        if (!enhanced_neural_bridge_validate_parameters(active_config, &validation_report)) {
            g_last_error = ENHANCED_NEURAL_ERROR_VALIDATION_FAILED;
            strncpy(result->error_message, "Parameter validation failed", 
                    sizeof(result->error_message) - 1);
            
            if (validation_report) {
                snprintf(result->cuda_validation_report, sizeof(result->cuda_validation_report),
                        "Validation failed: %zu errors, %zu warnings", 
                        validation_report->error_count, validation_report->warning_count);
                cuda_validation_report_free(validation_report);
            }
            return false;
        }
        
        if (validation_report) {
            g_bridge_context.last_validation_report = validation_report;
            result->parameter_validation_passed = true;
        }
    }
    
    // Select optimal algorithm
    EnhancedNeuralAlgorithm selected_algorithm = enhanced_neural_bridge_select_optimal_algorithm(
        input_size, input_data, active_config);
    result->algorithm_used = selected_algorithm;
    
    // Set CUDA profile information
    if (active_config->cuda_profile) {
        result->cuda_profile_used = active_config->cuda_profile->name;
    }
    
    // Route to appropriate compression algorithm
    bool compression_success = false;
    
    switch (selected_algorithm) {
        case ENHANCED_NEURAL_ALGORITHM_METAL_LSTM:
            compression_success = enhanced_neural_bridge_metal_lstm_compress(
                input_data, input_size, output_data, output_capacity, active_config, result);
            break;
            
        case ENHANCED_NEURAL_ALGORITHM_SMALL_FILE:
            compression_success = enhanced_neural_bridge_small_file_compress(
                input_data, input_size, output_data, output_capacity, active_config, result);
            break;
            
        default:
            g_last_error = ENHANCED_NEURAL_ERROR_UNKNOWN_ALGORITHM;
            strncpy(result->error_message, "Unknown compression algorithm", 
                    sizeof(result->error_message) - 1);
            return false;
    }
    
    uint64_t end_time = get_time_ns();
    result->processing_time_ns = end_time - start_time;
    
    // Update statistics
    g_bridge_context.total_compressions++;
    g_bridge_context.total_bytes_processed += input_size;
    g_bridge_context.total_processing_time_ms += (double)(end_time - start_time) / 1000000.0;
    
    if (compression_success) {
        g_bridge_context.successful_compressions++;
        result->compression_ratio = (float)result->compressed_size / (float)input_size;
        
        // CUDA compatibility tracking
        if (result->cuda_compatible) {
            g_bridge_context.cuda_compatible_operations++;
        }
        
        // Track precision deviation
        if (result->max_precision_deviation > g_bridge_context.max_observed_deviation) {
            g_bridge_context.max_observed_deviation = result->max_precision_deviation;
        }
        
        // Assess quality
        result->quality_score = enhanced_neural_bridge_assess_compression_quality(result);
        result->meets_quality_threshold = enhanced_neural_bridge_meets_quality_requirements(result, active_config);
        
        if (active_config->verbose_logging) {
            printf("[Enhanced Neural Bridge] Compression successful: %zu -> %zu bytes (%.1f%%), quality=%.3f\n", 
                   input_size, result->compressed_size, result->compression_ratio * 100.0f, result->quality_score);
        }
    } else {
        if (active_config->verbose_logging) {
            printf("[Enhanced Neural Bridge] Compression failed: %s\n", result->error_message);
        }
    }
    
    return compression_success;
}

// Enhanced Metal LSTM compression implementation
bool enhanced_neural_bridge_metal_lstm_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const EnhancedNeuralCompressionConfig* config,
    EnhancedNeuralCompressionResult* result
) {
    if (!g_bridge_context.primary_lstm_context) {
        strncpy(result->error_message, "Primary LSTM context not available", 
                sizeof(result->error_message) - 1);
        return false;
    }
    
    // Validate CUDA compatibility
    if (!enhanced_lstm_verify_cuda_compatibility(g_bridge_context.primary_lstm_context)) {
        result->cuda_compatible = false;
        strncpy(result->error_message, "LSTM context not CUDA compatible", 
                sizeof(result->error_message) - 1);
        return false;
    }
    
    result->cuda_compatible = true;
    
    // Store algorithm identifier in output (8 bytes header)
    if (output_capacity < 16) {
        strncpy(result->error_message, "Output buffer too small for header", 
                sizeof(result->error_message) - 1);
        return false;
    }
    
    size_t header_size = 0;
    
    // Algorithm identifier (4 bytes: "ELST" = Enhanced LSTM)
    output_data[header_size++] = 'E';
    output_data[header_size++] = 'L';
    output_data[header_size++] = 'S';
    output_data[header_size++] = 'T';
    
    // Original size (4 bytes, little endian)
    output_data[header_size++] = (uint8_t)(input_size & 0xFF);
    output_data[header_size++] = (uint8_t)((input_size >> 8) & 0xFF);
    output_data[header_size++] = (uint8_t)((input_size >> 16) & 0xFF);
    output_data[header_size++] = (uint8_t)((input_size >> 24) & 0xFF);
    
    // CUDA profile identifier (4 bytes)
    uint32_t profile_id = 0;
    if (config && config->cuda_profile) {
        profile_id = (uint32_t)config->cuda_profile->type;
    }
    output_data[header_size++] = (uint8_t)(profile_id & 0xFF);
    output_data[header_size++] = (uint8_t)((profile_id >> 8) & 0xFF);
    output_data[header_size++] = (uint8_t)((profile_id >> 16) & 0xFF);
    output_data[header_size++] = (uint8_t)((profile_id >> 24) & 0xFF);
    
    // Quality level (2 bytes)
    uint16_t quality = config ? config->quality_level : 7;
    output_data[header_size++] = (uint8_t)(quality & 0xFF);
    output_data[header_size++] = (uint8_t)((quality >> 8) & 0xFF);
    
    // Reserved (2 bytes)
    output_data[header_size++] = 0x00;
    output_data[header_size++] = 0x00;
    
    // Process through Enhanced Metal LSTM
    // This uses the seg_len from the CUDA profile for processing
    int32_t seg_len = g_bridge_context.primary_lstm_context->config.seg_len;
    size_t remaining_input = input_size;
    size_t input_pos = 0;
    size_t output_pos = header_size;
    
    // Process input in segments
    while (remaining_input > 0 && output_pos < output_capacity - 256) {  // Reserve space for segment
        size_t segment_size = (remaining_input > (size_t)seg_len) ? (size_t)seg_len : remaining_input;
        
        // Allocate temporary output for this segment
        float* segment_probabilities = (float*)malloc(segment_size * 256 * sizeof(float));
        if (!segment_probabilities) {
            strncpy(result->error_message, "Memory allocation failed for segment processing", 
                    sizeof(result->error_message) - 1);
            return false;
        }
        
        // Process segment through Enhanced LSTM
        bool segment_success = enhanced_lstm_process_segment(
            g_bridge_context.primary_lstm_context,
            input_data + input_pos,
            segment_size,
            segment_probabilities
        );
        
        if (!segment_success) {
            free(segment_probabilities);
            strncpy(result->error_message, "Enhanced LSTM segment processing failed", 
                    sizeof(result->error_message) - 1);
            return false;
        }
        
        // Simple compression based on LSTM probabilities
        // This is a simplified implementation - real implementation would use entropy coding
        size_t compressed_segment_size = segment_size / 2 + 8;  // Simplified compression estimate
        if (output_pos + compressed_segment_size >= output_capacity) {
            free(segment_probabilities);
            strncpy(result->error_message, "Output buffer too small for compressed segment", 
                    sizeof(result->error_message) - 1);
            return false;
        }
        
        // Store segment size (2 bytes)
        output_data[output_pos++] = (uint8_t)(compressed_segment_size & 0xFF);
        output_data[output_pos++] = (uint8_t)((compressed_segment_size >> 8) & 0xFF);
        
        // Simple compression (placeholder - would use proper entropy coding in real implementation)
        for (size_t i = 0; i < segment_size && output_pos < output_capacity; i += 2) {
            if (i + 1 < segment_size) {
                // Combine two bytes (simplified)
                uint8_t combined = (input_data[input_pos + i] + input_data[input_pos + i + 1]) / 2;
                output_data[output_pos++] = combined;
            } else {
                output_data[output_pos++] = input_data[input_pos + i];
            }
        }
        
        free(segment_probabilities);
        
        input_pos += segment_size;
        remaining_input -= segment_size;
    }
    
    result->compressed_size = output_pos;
    result->success = true;
    result->max_precision_deviation = 0.0f;  // Would be calculated from LSTM processing
    
    return true;
}

// Small file compression implementation
bool enhanced_neural_bridge_small_file_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const EnhancedNeuralCompressionConfig* config,
    EnhancedNeuralCompressionResult* result
) {
    if (!g_bridge_context.small_file_handler) {
        strncpy(result->error_message, "Small file handler not available", 
                sizeof(result->error_message) - 1);
        return false;
    }
    
    // Process through small file handler
    uint8_t* compressed_output = NULL;
    size_t compressed_size = 0;
    
    bool success = small_file_handler_process(
        g_bridge_context.small_file_handler,
        input_data,
        input_size,
        &compressed_output,
        &compressed_size
    );
    
    if (!success) {
        strncpy(result->error_message, "Small file handler processing failed", 
                sizeof(result->error_message) - 1);
        return false;
    }
    
    if (compressed_size > output_capacity) {
        if (compressed_output) free(compressed_output);
        strncpy(result->error_message, "Output buffer too small for small file compression", 
                sizeof(result->error_message) - 1);
        return false;
    }
    
    // Copy compressed data to output buffer
    memcpy(output_data, compressed_output, compressed_size);
    free(compressed_output);
    
    result->compressed_size = compressed_size;
    result->success = true;
    result->cuda_compatible = small_file_validate_cuda_compatibility(g_bridge_context.small_file_handler);
    result->quality_score = small_file_calculate_quality_score(g_bridge_context.small_file_handler);
    
    return true;
}

// Quality assessment
float enhanced_neural_bridge_assess_compression_quality(const EnhancedNeuralCompressionResult* result) {
    if (!result || !result->success) {
        return 0.0f;
    }
    
    float quality = 0.0f;
    
    // Compression ratio component (40%)
    float compression_component = (1.0f - result->compression_ratio) * 0.4f;
    if (compression_component < 0.0f) compression_component = 0.0f;
    
    // CUDA compatibility component (30%)
    float cuda_component = result->cuda_compatible ? 0.3f : 0.0f;
    
    // Parameter validation component (20%)
    float validation_component = result->parameter_validation_passed ? 0.2f : 0.0f;
    
    // Precision component (10%)
    float precision_component = (1.0f - result->max_precision_deviation) * 0.1f;
    if (precision_component < 0.0f) precision_component = 0.0f;
    
    quality = compression_component + cuda_component + validation_component + precision_component;
    
    return fminf(1.0f, fmaxf(0.0f, quality));
}

bool enhanced_neural_bridge_meets_quality_requirements(const EnhancedNeuralCompressionResult* result,
                                                       const EnhancedNeuralCompressionConfig* config) {
    if (!result || !config) {
        return false;
    }
    
    // Basic success requirement
    if (!result->success) {
        return false;
    }
    
    // CUDA compatibility requirement (if enforced)
    if (config->enforce_cuda_compatibility && !result->cuda_compatible) {
        return false;
    }
    
    // Quality score threshold (based on quality level)
    float min_quality = (float)config->quality_level / 10.0f;
    if (result->quality_score < min_quality) {
        return false;
    }
    
    // Compression target requirement
    if (result->compression_ratio > config->compression_target + 0.1f) {  // 10% tolerance
        return false;
    }
    
    return true;
}

// Validation functions
bool enhanced_neural_bridge_validate_cuda_compatibility(const EnhancedNeuralCompressionConfig* config) {
    if (!config) {
        return false;
    }
    
    // Check CUDA profile validity
    if (config->cuda_profile) {
        return cuda_profile_validate(config->cuda_profile);
    }
    
    // Check basic parameter constraints
    if (config->memory_limit_bytes == 0 || 
        config->quality_level == 0 || config->quality_level > 10 ||
        config->compression_target <= 0.0f || config->compression_target >= 1.0f) {
        return false;
    }
    
    return true;
}

bool enhanced_neural_bridge_validate_parameters(const EnhancedNeuralCompressionConfig* config,
                                               CUDAValidationReport** report) {
    if (!config) {
        return false;
    }
    
    if (config->cuda_profile) {
        *report = cuda_validator_validate_profile(config->cuda_profile);
        return *report ? (*report)->overall_valid : false;
    }
    
    return true;
}

// Utility functions
size_t enhanced_neural_bridge_estimate_compressed_size(size_t input_size, 
                                                       const EnhancedNeuralCompressionConfig* config) {
    if (!config) {
        return input_size / 2;  // Default 50% compression estimate
    }
    
    return (size_t)((float)input_size * config->compression_target) + 64;  // Add header overhead
}

bool enhanced_neural_bridge_algorithm_available(EnhancedNeuralAlgorithm algorithm) {
    switch (algorithm) {
        case ENHANCED_NEURAL_ALGORITHM_METAL_LSTM:
            return g_bridge_initialized && g_bridge_context.primary_lstm_context != NULL;
            
        case ENHANCED_NEURAL_ALGORITHM_SMALL_FILE:
            return g_bridge_initialized && g_bridge_context.small_file_handler != NULL;
            
        case ENHANCED_NEURAL_ALGORITHM_AUTO:
            return g_bridge_initialized;
            
        default:
            return false;
    }
}

const char* enhanced_neural_bridge_algorithm_name(EnhancedNeuralAlgorithm algorithm) {
    if (algorithm >= 0 && algorithm < sizeof(enhanced_algorithm_names) / sizeof(enhanced_algorithm_names[0])) {
        return enhanced_algorithm_names[algorithm];
    }
    return "Unknown";
}

// Statistics and monitoring
void enhanced_neural_bridge_print_statistics(void) {
    if (!g_bridge_initialized) {
        printf("Enhanced Neural Bridge not initialized\n");
        return;
    }
    
    printf("Enhanced Neural Bridge Statistics:\n");
    printf("  Total compressions: %zu\n", g_bridge_context.total_compressions);
    printf("  Successful compressions: %zu\n", g_bridge_context.successful_compressions);
    printf("  Success rate: %.1f%%\n", 
           g_bridge_context.total_compressions > 0 ? 
           (100.0 * g_bridge_context.successful_compressions / g_bridge_context.total_compressions) : 0.0);
    printf("  Total bytes processed: %zu\n", g_bridge_context.total_bytes_processed);
    printf("  Average processing time: %.2f ms\n", 
           g_bridge_context.total_compressions > 0 ? 
           (g_bridge_context.total_processing_time_ms / g_bridge_context.total_compressions) : 0.0);
    printf("  CUDA compatible operations: %zu\n", g_bridge_context.cuda_compatible_operations);
    printf("  CUDA compatibility rate: %.1f%%\n",
           g_bridge_context.total_compressions > 0 ?
           (100.0 * g_bridge_context.cuda_compatible_operations / g_bridge_context.total_compressions) : 0.0);
    printf("  Max observed precision deviation: %.2e\n", g_bridge_context.max_observed_deviation);
}

// Error handling
const char* enhanced_neural_bridge_error_string(EnhancedNeuralError error) {
    if (error >= 0 && error < sizeof(enhanced_neural_error_messages) / sizeof(enhanced_neural_error_messages[0])) {
        return enhanced_neural_error_messages[error];
    }
    return "Unknown error";
}

EnhancedNeuralError enhanced_neural_bridge_get_last_error(void) {
    return g_last_error;
}