#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "compression_selector.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <mach/mach_time.h>

// Statistical analysis functions
static float calculate_entropy(const uint8_t* data, size_t size);
static float calculate_variance(const float* data, size_t count, float mean);
static float calculate_autocorrelation(const float* data, size_t count, uint32_t lag);
static float calculate_spectral_entropy(const float* data, size_t count);
static DataTypeClassification classify_data_type(const void* data, size_t size);

// Feature extraction functions
static CompressionSelectorError extract_basic_statistics(const void* data, size_t size, DataCharacteristics* chars);
static CompressionSelectorError extract_frequency_features(const void* data, size_t size, DataCharacteristics* chars);
static CompressionSelectorError extract_temporal_features(const void* data, size_t size, DataCharacteristics* chars);
static CompressionSelectorError extract_spatial_features(const void* data, size_t size, DataCharacteristics* chars);

// ML prediction functions
static CompressionSelectorError predict_with_neural_network(CompressionSelectorContext* context,
                                                           const float* features,
                                                           uint32_t feature_count,
                                                           AlgorithmPerformancePrediction* predictions,
                                                           uint32_t num_algorithms);

// Performance scoring functions
static float calculate_objective_score(const AlgorithmPerformancePrediction* prediction,
                                      const CompressionSelectorConfig* config);

CompressionSelectorError compression_selector_create(CompressionSelectorContext** context,
                                                    const CompressionSelectorConfig* config) {
    if (!context || !config) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    // Allocate context
    CompressionSelectorContext* ctx = (CompressionSelectorContext*)calloc(1, sizeof(CompressionSelectorContext));
    if (!ctx) {
        return COMPRESSION_SELECTOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy configuration
    memcpy(&ctx->config, config, sizeof(CompressionSelectorConfig));
    
    // Initialize Metal device
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            free(ctx);
            return COMPRESSION_SELECTOR_ERROR_DEVICE_NOT_FOUND;
        }
        
        ctx->device = (__bridge_retained void*)device;
        ctx->command_queue = (__bridge_retained void*)[device newCommandQueue];
        
        // Initialize algorithm availability (all enabled by default)
        for (int i = 0; i <= COMPRESSION_ALGORITHM_AUTO_ENSEMBLE; i++) {
            ctx->available_algorithms[i] = true;
            ctx->algorithm_weights[i] = 1.0f;
        }
        
        // Allocate performance history
        ctx->history_capacity = 10000;
        ctx->history = (PerformanceHistoryEntry*)calloc(ctx->history_capacity, sizeof(PerformanceHistoryEntry));
        if (!ctx->history) {
            compression_selector_destroy(ctx);
            return COMPRESSION_SELECTOR_ERROR_MEMORY_ALLOCATION;
        }
        
        // Allocate Metal buffers
        size_t feature_buffer_size = config->ml_feature_vector_size * sizeof(float);
        size_t prediction_buffer_size = 16 * sizeof(AlgorithmPerformancePrediction);
        size_t analysis_buffer_size = config->sample_size_for_analysis * sizeof(float);
        
        ctx->buffers.feature_buffer = (__bridge_retained void*)[device newBufferWithLength:feature_buffer_size
                                                                                   options:MTLResourceStorageModeShared];
        ctx->buffers.prediction_buffer = (__bridge_retained void*)[device newBufferWithLength:prediction_buffer_size
                                                                                      options:MTLResourceStorageModeShared];
        ctx->buffers.analysis_buffer = (__bridge_retained void*)[device newBufferWithLength:analysis_buffer_size
                                                                                     options:MTLResourceStorageModeShared];
        
        if (!ctx->buffers.feature_buffer || !ctx->buffers.prediction_buffer || !ctx->buffers.analysis_buffer) {
            compression_selector_destroy(ctx);
            return COMPRESSION_SELECTOR_ERROR_MEMORY_ALLOCATION;
        }
        
        ctx->buffers.buffer_sizes[0] = feature_buffer_size;
        ctx->buffers.buffer_sizes[1] = prediction_buffer_size;
        ctx->buffers.buffer_sizes[2] = analysis_buffer_size;
        ctx->buffers.num_allocated_buffers = 3;
        
        ctx->is_initialized = true;
    }
    
    *context = ctx;
    return COMPRESSION_SELECTOR_SUCCESS;
}

CompressionSelectorError compression_selector_config_create_default(CompressionSelectorConfig* config,
                                                                   CompressionObjective objective) {
    if (!config) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(CompressionSelectorConfig));
    
    // Set default values
    config->primary_objective = objective;
    config->secondary_objective = COMPRESSION_OBJECTIVE_BALANCED;
    config->objective_weight_primary = 0.7f;
    config->objective_weight_secondary = 0.3f;
    
    // Default constraints
    config->max_compression_time_ms = 30000.0f;  // 30 seconds
    config->min_compression_ratio = 1.1f;        // At least 10% compression
    config->min_quality_score = 0.8f;            // 80% quality minimum
    config->max_memory_usage_mb = 1024;          // 1GB memory limit
    config->max_energy_consumption = 1.0f;       // Normalized energy limit
    
    // Default analysis parameters
    config->sample_size_for_analysis = 65536;    // 64KB sample
    config->analysis_confidence_threshold = 0.8f;
    config->num_algorithms_to_evaluate = 5;
    config->enable_ensemble_methods = false;
    
    // Default ML parameters
    config->enable_ml_prediction = true;
    config->ml_model_confidence_threshold = 0.7f;
    config->ml_feature_vector_size = 32;
    
    // Default performance parameters
    config->enable_gpu_acceleration = true;
    config->num_threads = 4;
    config->enable_adaptive_sampling = true;
    config->max_analysis_time_ms = 5000;         // 5 seconds max analysis
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

CompressionSelectorError compression_selector_analyze_data(CompressionSelectorContext* context,
                                                          const void* data,
                                                          size_t data_size,
                                                          DataCharacteristics* characteristics) {
    if (!context || !data || !characteristics || data_size == 0) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    memset(characteristics, 0, sizeof(DataCharacteristics));
    
    // Basic properties
    characteristics->data_size = data_size;
    characteristics->sample_count = (uint32_t)(data_size / sizeof(float)); // Assume float data for now
    
    // Extract basic statistical features
    CompressionSelectorError error = extract_basic_statistics(data, data_size, characteristics);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        return error;
    }
    
    // Extract frequency domain features
    error = extract_frequency_features(data, data_size, characteristics);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        return error;
    }
    
    // Extract temporal features (for sequential data)
    error = extract_temporal_features(data, data_size, characteristics);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        return error;
    }
    
    // Extract spatial features (for image-like data)
    error = extract_spatial_features(data, data_size, characteristics);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        return error;
    }
    
    // Classify data type
    characteristics->primary_type = classify_data_type(data, data_size);
    characteristics->type_confidence = 0.8f; // Default confidence
    
    // Calculate overall compressibility score
    characteristics->compressibility_score = (characteristics->entropy / 8.0f) * 
                                           (1.0f - characteristics->sparsity) * 
                                           characteristics->redundancy_ratio;
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

CompressionSelectorError compression_selector_select_algorithm(CompressionSelectorContext* context,
                                                              const void* data,
                                                              size_t data_size,
                                                              CompressionSelectionResult* result) {
    if (!context || !data || !result || data_size == 0) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    memset(result, 0, sizeof(CompressionSelectionResult));
    
    uint64_t start_time = mach_absolute_time();
    
    // Analyze data characteristics
    CompressionSelectorError error = compression_selector_analyze_data(context, data, data_size, 
                                                                      &result->data_analysis);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        return error;
    }
    
    // Extract ML features
    float features[32];
    error = compression_selector_extract_features(&result->data_analysis, features, 32);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        return error;
    }
    
    // Predict performance for each available algorithm
    AlgorithmPerformancePrediction predictions[COMPRESSION_ALGORITHM_AUTO_ENSEMBLE + 1];
    uint32_t num_predictions = 0;
    
    for (int i = 0; i <= COMPRESSION_ALGORITHM_AUTO_ENSEMBLE; i++) {
        if (context->available_algorithms[i]) {
            error = compression_selector_predict_performance(context, &result->data_analysis,
                                                           (CompressionAlgorithm)i, &predictions[num_predictions]);
            if (error == COMPRESSION_SELECTOR_SUCCESS) {
                predictions[num_predictions].suitability_score = 
                    calculate_objective_score(&predictions[num_predictions], &context->config) *
                    context->algorithm_weights[i];
                num_predictions++;
            }
        }
    }
    
    if (num_predictions == 0) {
        return COMPRESSION_SELECTOR_ERROR_ALGORITHM_NOT_SUPPORTED;
    }
    
    // Sort algorithms by suitability score
    for (uint32_t i = 0; i < num_predictions - 1; i++) {
        for (uint32_t j = i + 1; j < num_predictions; j++) {
            if (predictions[j].suitability_score > predictions[i].suitability_score) {
                AlgorithmPerformancePrediction temp = predictions[i];
                predictions[i] = predictions[j];
                predictions[j] = temp;
            }
        }
    }
    
    // Select best algorithm
    result->selected_algorithm = predictions[0].algorithm;
    result->prediction = predictions[0];
    result->selection_confidence = predictions[0].confidence_score;
    result->num_algorithms_evaluated = num_predictions;
    
    // Calculate scores
    result->size_score = predictions[0].predicted_ratio / 10.0f; // Normalize to 0-1
    result->speed_score = predictions[0].predicted_speed / 1000.0f; // MB/s to 0-1
    result->quality_score = predictions[0].predicted_quality;
    result->overall_score = predictions[0].suitability_score;
    
    result->requires_gpu = (predictions[0].gpu_utilization > 0.1f);
    result->estimated_memory_mb = (uint32_t)predictions[0].memory_usage;
    
    // Provide alternatives
    uint32_t num_alternatives = MIN(3, num_predictions - 1);
    if (num_alternatives > 0) {
        result->alternative_algorithms = (CompressionAlgorithm*)malloc(num_alternatives * sizeof(CompressionAlgorithm));
        result->alternative_scores = (float*)malloc(num_alternatives * sizeof(float));
        
        if (result->alternative_algorithms && result->alternative_scores) {
            for (uint32_t i = 0; i < num_alternatives; i++) {
                result->alternative_algorithms[i] = predictions[i + 1].algorithm;
                result->alternative_scores[i] = predictions[i + 1].suitability_score;
            }
            result->num_alternatives = num_alternatives;
        }
    }
    
    // Calculate analysis time
    uint64_t end_time = mach_absolute_time();
    mach_timebase_info_data_t timebase_info;
    mach_timebase_info(&timebase_info);
    result->analysis_time_ms = (uint32_t)((end_time - start_time) * timebase_info.numer / timebase_info.denom / 1000000);
    
    // Update statistics
    context->total_selections++;
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

CompressionSelectorError compression_selector_predict_performance(CompressionSelectorContext* context,
                                                                 const DataCharacteristics* characteristics,
                                                                 CompressionAlgorithm algorithm,
                                                                 AlgorithmPerformancePrediction* prediction) {
    if (!context || !characteristics || !prediction) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    memset(prediction, 0, sizeof(AlgorithmPerformancePrediction));
    prediction->algorithm = algorithm;
    prediction->is_supported = true;
    
    // Simple heuristic-based prediction (replace with ML model when available)
    switch (algorithm) {
        case COMPRESSION_ALGORITHM_NEURAL_QUANTIZATION:
            prediction->predicted_ratio = 2.0f + characteristics->compressibility_score * 3.0f;
            prediction->predicted_speed = 100.0f + characteristics->sparsity * 200.0f;
            prediction->predicted_quality = 0.85f + characteristics->entropy / 16.0f;
            prediction->energy_consumption = 0.8f;
            prediction->memory_usage = 256.0f;
            prediction->gpu_utilization = 0.7f;
            break;
            
        case COMPRESSION_ALGORITHM_RLE_LOSSLESS:
            prediction->predicted_ratio = 1.2f + characteristics->sparsity * 5.0f;
            prediction->predicted_speed = 500.0f;
            prediction->predicted_quality = 1.0f; // Lossless
            prediction->energy_consumption = 0.2f;
            prediction->memory_usage = 64.0f;
            prediction->gpu_utilization = 0.1f;
            break;
            
        case COMPRESSION_ALGORITHM_TRANSFORMER_COMPRESSION:
            prediction->predicted_ratio = 3.0f + characteristics->pattern_repetition * 2.0f;
            prediction->predicted_speed = 50.0f;
            prediction->predicted_quality = 0.9f + characteristics->entropy / 20.0f;
            prediction->energy_consumption = 1.2f;
            prediction->memory_usage = 512.0f;
            prediction->gpu_utilization = 0.9f;
            break;
            
        case COMPRESSION_ALGORITHM_LSTM_COMPRESSION:
            prediction->predicted_ratio = 2.5f + characteristics->autocorrelation * 2.0f;
            prediction->predicted_speed = 75.0f;
            prediction->predicted_quality = 0.88f + characteristics->predictability / 10.0f;
            prediction->energy_consumption = 1.0f;
            prediction->memory_usage = 384.0f;
            prediction->gpu_utilization = 0.8f;
            break;
            
        default:
            // Default conservative estimates
            prediction->predicted_ratio = 1.5f;
            prediction->predicted_speed = 100.0f;
            prediction->predicted_quality = 0.8f;
            prediction->energy_consumption = 0.5f;
            prediction->memory_usage = 128.0f;
            prediction->gpu_utilization = 0.3f;
            break;
    }
    
    // Estimate time based on data size and predicted speed
    float data_size_mb = characteristics->data_size / (1024.0f * 1024.0f);
    prediction->estimated_time_ms = (uint32_t)((data_size_mb / prediction->predicted_speed) * 1000.0f);
    
    // Set confidence based on data characteristics match
    prediction->confidence_score = 0.7f + characteristics->type_confidence * 0.3f;
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

CompressionSelectorError compression_selector_extract_features(const DataCharacteristics* characteristics,
                                                              float* features,
                                                              uint32_t feature_count) {
    if (!characteristics || !features || feature_count < 16) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    // Normalize and extract key features
    features[0] = logf((float)characteristics->data_size) / 20.0f; // Log data size
    features[1] = characteristics->entropy / 8.0f;
    features[2] = characteristics->sparsity;
    features[3] = characteristics->redundancy_ratio;
    features[4] = characteristics->compressibility_score;
    features[5] = characteristics->variance / 100.0f;
    features[6] = characteristics->autocorrelation;
    features[7] = characteristics->predictability;
    features[8] = characteristics->spectral_entropy / 8.0f;
    features[9] = characteristics->pattern_repetition;
    features[10] = characteristics->trend_strength;
    features[11] = characteristics->seasonality_strength;
    features[12] = characteristics->spatial_correlation;
    features[13] = characteristics->mutual_information / 10.0f;
    features[14] = characteristics->information_content / 10.0f;
    features[15] = (float)characteristics->primary_type / 12.0f;
    
    // Fill remaining features with derived values
    for (uint32_t i = 16; i < feature_count; i++) {
        features[i] = features[i % 16] * features[(i + 1) % 16]; // Cross features
    }
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

// Helper function implementations

static float calculate_entropy(const uint8_t* data, size_t size) {
    if (!data || size == 0) return 0.0f;
    
    uint32_t histogram[256] = {0};
    for (size_t i = 0; i < size; i++) {
        histogram[data[i]]++;
    }
    
    float entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            float p = (float)histogram[i] / size;
            entropy -= p * log2f(p);
        }
    }
    
    return entropy;
}

static float calculate_variance(const float* data, size_t count, float mean) {
    if (!data || count == 0) return 0.0f;
    
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    
    return sum_sq_diff / count;
}

static DataTypeClassification classify_data_type(const void* data, size_t size) {
    // Simple heuristic classification
    const uint8_t* bytes = (const uint8_t*)data;
    
    // Check for text patterns
    uint32_t printable_count = 0;
    for (size_t i = 0; i < MIN(size, 1024); i++) {
        if ((bytes[i] >= 32 && bytes[i] <= 126) || bytes[i] == '\n' || bytes[i] == '\t') {
            printable_count++;
        }
    }
    
    if (printable_count > size * 0.8) {
        return DATA_TYPE_TEXT;
    }
    
    // Check for binary patterns
    uint32_t zero_count = 0;
    for (size_t i = 0; i < MIN(size, 1024); i++) {
        if (bytes[i] == 0) zero_count++;
    }
    
    if (zero_count > size * 0.3) {
        return DATA_TYPE_SPARSE;
    }
    
    return DATA_TYPE_BINARY;
}

static CompressionSelectorError extract_basic_statistics(const void* data, size_t size, DataCharacteristics* chars) {
    const uint8_t* bytes = (const uint8_t*)data;
    
    // Calculate entropy
    chars->entropy = calculate_entropy(bytes, size);
    
    // Calculate basic statistics assuming float data
    if (size >= sizeof(float)) {
        const float* float_data = (const float*)data;
        size_t float_count = size / sizeof(float);
        
        // Calculate mean
        float sum = 0.0f;
        for (size_t i = 0; i < float_count; i++) {
            sum += float_data[i];
        }
        chars->mean = sum / float_count;
        
        // Calculate variance
        chars->variance = calculate_variance(float_data, float_count, chars->mean);
        
        // Calculate sparsity
        uint32_t zero_count = 0;
        for (size_t i = 0; i < float_count; i++) {
            if (fabsf(float_data[i]) < 1e-6f) zero_count++;
        }
        chars->sparsity = (float)zero_count / float_count;
    }
    
    // Estimate redundancy
    chars->redundancy_ratio = 1.0f - (chars->entropy / 8.0f);
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

static CompressionSelectorError extract_frequency_features(const void* data, size_t size, DataCharacteristics* chars) {
    // Simple spectral analysis
    if (size >= sizeof(float) * 8) {
        const float* float_data = (const float*)data;
        size_t float_count = MIN(size / sizeof(float), 1024);
        
        // Calculate spectral entropy (simplified)
        chars->spectral_entropy = calculate_entropy((const uint8_t*)float_data, float_count * sizeof(float)) / 2.0f;
        
        // Estimate dominant frequency (simplified)
        chars->dominant_frequency = 0.1f;
        chars->frequency_spread = 0.5f;
    }
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

static CompressionSelectorError extract_temporal_features(const void* data, size_t size, DataCharacteristics* chars) {
    if (size >= sizeof(float) * 16) {
        const float* float_data = (const float*)data;
        size_t float_count = MIN(size / sizeof(float), 1024);
        
        // Calculate autocorrelation at lag 1
        chars->autocorrelation = calculate_autocorrelation(float_data, float_count, 1);
        
        // Estimate trend strength (linear regression slope)
        float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        for (size_t i = 0; i < float_count; i++) {
            sum_x += i;
            sum_y += float_data[i];
            sum_xy += i * float_data[i];
            sum_x2 += i * i;
        }
        
        float slope = (float_count * sum_xy - sum_x * sum_y) / (float_count * sum_x2 - sum_x * sum_x);
        chars->trend_strength = fabsf(slope) / (chars->variance + 1e-6f);
        
        // Simple seasonality detection
        chars->seasonality_strength = 0.1f; // Placeholder
        
        // Pattern repetition measure
        chars->pattern_repetition = 1.0f - chars->entropy / 8.0f;
        
        // Predictability measure
        chars->predictability = chars->autocorrelation * (1.0f - chars->entropy / 8.0f);
    }
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

static CompressionSelectorError extract_spatial_features(const void* data, size_t size, DataCharacteristics* chars) {
    // Placeholder spatial analysis
    chars->spatial_correlation = 0.5f;
    chars->edge_density = 0.3f;
    chars->texture_complexity = 0.4f;
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

static float calculate_autocorrelation(const float* data, size_t count, uint32_t lag) {
    if (!data || count <= lag) return 0.0f;
    
    // Calculate mean
    float mean = 0.0f;
    for (size_t i = 0; i < count; i++) {
        mean += data[i];
    }
    mean /= count;
    
    // Calculate autocorrelation
    float numerator = 0.0f;
    float denominator = 0.0f;
    
    for (size_t i = 0; i < count - lag; i++) {
        numerator += (data[i] - mean) * (data[i + lag] - mean);
    }
    
    for (size_t i = 0; i < count; i++) {
        float diff = data[i] - mean;
        denominator += diff * diff;
    }
    
    return denominator > 0 ? numerator / denominator : 0.0f;
}

static float calculate_objective_score(const AlgorithmPerformancePrediction* prediction,
                                      const CompressionSelectorConfig* config) {
    float score = 0.0f;
    
    switch (config->primary_objective) {
        case COMPRESSION_OBJECTIVE_SIZE:
            score = prediction->predicted_ratio / 10.0f; // Normalize ratio
            break;
        case COMPRESSION_OBJECTIVE_SPEED:
            score = prediction->predicted_speed / 1000.0f; // Normalize speed
            break;
        case COMPRESSION_OBJECTIVE_QUALITY:
            score = prediction->predicted_quality;
            break;
        case COMPRESSION_OBJECTIVE_BALANCED:
            score = (prediction->predicted_ratio / 10.0f + 
                    prediction->predicted_speed / 1000.0f + 
                    prediction->predicted_quality) / 3.0f;
            break;
        default:
            score = 0.5f;
            break;
    }
    
    // Apply constraints
    if (prediction->estimated_time_ms > config->max_compression_time_ms) {
        score *= 0.5f;
    }
    if (prediction->predicted_ratio < config->min_compression_ratio) {
        score *= 0.1f;
    }
    if (prediction->predicted_quality < config->min_quality_score) {
        score *= 0.1f;
    }
    
    return score * prediction->confidence_score;
}

const char* compression_selector_get_algorithm_name(CompressionAlgorithm algorithm) {
    switch (algorithm) {
        case COMPRESSION_ALGORITHM_NEURAL_QUANTIZATION: return "Neural Quantization";
        case COMPRESSION_ALGORITHM_RLE_LOSSLESS: return "RLE Lossless";
        case COMPRESSION_ALGORITHM_NEURAL_LOSSLESS: return "Neural Lossless";
        case COMPRESSION_ALGORITHM_TRANSFORMER_COMPRESSION: return "Transformer Compression";
        case COMPRESSION_ALGORITHM_LSTM_COMPRESSION: return "LSTM Compression";
        case COMPRESSION_ALGORITHM_HYBRID_NEURAL: return "Hybrid Neural";
        case COMPRESSION_ALGORITHM_ATTENTION_BASED: return "Attention-Based";
        case COMPRESSION_ALGORITHM_ADAPTIVE_QUANTIZATION: return "Adaptive Quantization";
        case COMPRESSION_ALGORITHM_ENTROPY_CODING: return "Entropy Coding";
        case COMPRESSION_ALGORITHM_RESIDUAL_COMPRESSION: return "Residual Compression";
        case COMPRESSION_ALGORITHM_HIERARCHICAL_COMPRESSION: return "Hierarchical Compression";
        case COMPRESSION_ALGORITHM_AUTO_ENSEMBLE: return "Auto Ensemble";
        default: return "Unknown Algorithm";
    }
}

const char* compression_selector_get_error_string(CompressionSelectorError error_code) {
    switch (error_code) {
        case COMPRESSION_SELECTOR_SUCCESS: return "Success";
        case COMPRESSION_SELECTOR_ERROR_INVALID_PARAM: return "Invalid parameter";
        case COMPRESSION_SELECTOR_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case COMPRESSION_SELECTOR_ERROR_DEVICE_NOT_FOUND: return "Metal device not found";
        case COMPRESSION_SELECTOR_ERROR_COMPUTE_FAILED: return "Computation failed";
        case COMPRESSION_SELECTOR_ERROR_ANALYSIS_FAILED: return "Data analysis failed";
        case COMPRESSION_SELECTOR_ERROR_PREDICTION_FAILED: return "Performance prediction failed";
        case COMPRESSION_SELECTOR_ERROR_MODEL_NOT_LOADED: return "ML model not loaded";
        case COMPRESSION_SELECTOR_ERROR_INSUFFICIENT_DATA: return "Insufficient data for analysis";
        case COMPRESSION_SELECTOR_ERROR_FEATURE_EXTRACTION_FAILED: return "Feature extraction failed";
        case COMPRESSION_SELECTOR_ERROR_ALGORITHM_NOT_SUPPORTED: return "Algorithm not supported";
        case COMPRESSION_SELECTOR_ERROR_OPTIMIZATION_FAILED: return "Optimization failed";
        default: return "Unknown error";
    }
}

void compression_selector_destroy(CompressionSelectorContext* context) {
    if (!context) return;
    
    @autoreleasepool {
        // Release Metal objects
        if (context->device) {
            CFRelease(context->device);
        }
        if (context->command_queue) {
            CFRelease(context->command_queue);
        }
        if (context->buffers.feature_buffer) {
            CFRelease(context->buffers.feature_buffer);
        }
        if (context->buffers.prediction_buffer) {
            CFRelease(context->buffers.prediction_buffer);
        }
        if (context->buffers.analysis_buffer) {
            CFRelease(context->buffers.analysis_buffer);
        }
        
        // Free history
        if (context->history) {
            free(context->history);
        }
        
        // Free context
        free(context);
    }
}

// Additional stub implementations for completeness
CompressionSelectorError compression_selector_load_model(CompressionSelectorContext* context,
                                                        const char* model_path) {
    // Stub implementation - would load trained ML model
    if (!context || !model_path) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    context->model_loaded = true;
    return COMPRESSION_SELECTOR_SUCCESS;
}

CompressionSelectorError compression_selector_update_history(CompressionSelectorContext* context,
                                                            CompressionAlgorithm algorithm,
                                                            const DataCharacteristics* data_chars,
                                                            float actual_ratio,
                                                            float actual_speed,
                                                            float actual_quality) {
    if (!context || !data_chars) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    if (context->history_count < context->history_capacity) {
        PerformanceHistoryEntry* entry = &context->history[context->history_count];
        entry->algorithm = algorithm;
        entry->data_chars = *data_chars;
        entry->actual_ratio = actual_ratio;
        entry->actual_speed = actual_speed;
        entry->actual_quality = actual_quality;
        entry->timestamp = mach_absolute_time();
        entry->was_successful = true;
        
        context->history_count++;
        context->successful_selections++;
    }
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

CompressionSelectorError compression_selector_set_algorithm_availability(CompressionSelectorContext* context,
                                                                        CompressionAlgorithm algorithm,
                                                                        bool is_available,
                                                                        float preference_weight) {
    if (!context || algorithm >= COMPRESSION_ALGORITHM_AUTO_ENSEMBLE + 1) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    context->available_algorithms[algorithm] = is_available;
    context->algorithm_weights[algorithm] = preference_weight;
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

CompressionSelectorError compression_selector_get_statistics(CompressionSelectorContext* context,
                                                           uint64_t* total_selections,
                                                           float* accuracy,
                                                           float* avg_analysis_time) {
    if (!context || !total_selections || !accuracy || !avg_analysis_time) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    *total_selections = context->total_selections;
    *accuracy = context->total_selections > 0 ? 
                (float)context->successful_selections / context->total_selections : 0.0f;
    *avg_analysis_time = 50.0f; // Placeholder
    
    return COMPRESSION_SELECTOR_SUCCESS;
}

CompressionSelectorError compression_selector_get_feature_importance(CompressionSelectorContext* context,
                                                                    float* importance_scores,
                                                                    uint32_t feature_count) {
    if (!context || !importance_scores || feature_count == 0) {
        return COMPRESSION_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    // Placeholder importance scores
    for (uint32_t i = 0; i < feature_count; i++) {
        importance_scores[i] = 1.0f / feature_count;
    }
    
    return COMPRESSION_SELECTOR_SUCCESS;
}
