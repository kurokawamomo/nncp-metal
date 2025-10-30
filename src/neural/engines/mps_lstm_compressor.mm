#include "mps_lstm_compressor.h"
#include "../ops/mps_lstm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#define MPS_AVAILABLE 1
#else
#define MPS_AVAILABLE 0
#endif

// Internal helper functions
static MPSLSTMCompressorError create_compressor_device(MPSLSTMCompressorContext* context);
static MPSLSTMCompressorError build_compression_graphs(MPSLSTMCompressorContext* context);
static MPSLSTMCompressorError allocate_compressor_buffers(MPSLSTMCompressorContext* context);
static MPSLSTMCompressorError initialize_lstm_contexts(MPSLSTMCompressorContext* context);
static MPSLSTMCompressorError execute_predictive_compression(MPSLSTMCompressorContext* context,
                                                           const float* input_data,
                                                           uint32_t input_size,
                                                           void* compressed_output,
                                                           uint32_t* compressed_size);
static MPSLSTMCompressorError execute_temporal_compression(MPSLSTMCompressorContext* context,
                                                         const float* input_data,
                                                         uint32_t input_size,
                                                         void* compressed_output,
                                                         uint32_t* compressed_size);
static MPSLSTMCompressorError execute_residual_compression(MPSLSTMCompressorContext* context,
                                                         const float* input_data,
                                                         uint32_t input_size,
                                                         void* compressed_output,
                                                         uint32_t* compressed_size);
static MPSLSTMCompressorError execute_decompression_pipeline(MPSLSTMCompressorContext* context,
                                                           const void* compressed_data,
                                                           uint32_t compressed_size,
                                                           float* output_data,
                                                           uint32_t* output_size);
static MPSLSTMCompressorError quantize_data(const float* input, void* output, uint32_t size,
                                           MPSLSTMDataType target_type, float scale);
static MPSLSTMCompressorError dequantize_data(const void* input, float* output, uint32_t size,
                                             MPSLSTMDataType source_type, float scale);
static uint64_t get_compressor_timestamp_ns(void);
static uint32_t calculate_checksum(const void* data, size_t size);
static float calculate_mse(const float* original, const float* reconstructed, uint32_t size);
static float calculate_psnr(float mse, float max_value);

// Error messages
static const char* compressor_error_messages[] = {
    "Success",
    "Invalid parameter",
    "Memory allocation failed",
    "Metal device not found",
    "Compute operation failed",
    "Invalid tensor dimensions",
    "Buffer allocation failed",
    "Graph compilation failed",
    "Execution failed",
    "Unsupported operation",
    "Compression failed",
    "Decompression failed",
    "Invalid compression ratio"
};

// Core API Implementation

MPSLSTMCompressorError mps_lstm_compressor_create(MPSLSTMCompressorContext** context,
                                                 const MPSLSTMCompressorConfig* config) {
    if (!context || !config) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    // Validate configuration
    MPSLSTMCompressorError error = mps_lstm_compressor_config_validate(config);
    if (error != MPS_LSTM_COMPRESSOR_SUCCESS) {
        return error;
    }
    
    *context = (MPSLSTMCompressorContext*)calloc(1, sizeof(MPSLSTMCompressorContext));
    if (!*context) {
        return MPS_LSTM_COMPRESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy configuration
    (*context)->config = *config;
    
    // Initialize statistics
    memset(&(*context)->stats, 0, sizeof(MPSLSTMCompressorStats));
    
    // Create Metal device
    error = create_compressor_device(*context);
    if (error != MPS_LSTM_COMPRESSOR_SUCCESS) {
        mps_lstm_compressor_destroy(*context);
        *context = NULL;
        return error;
    }
    
    // Initialize LSTM contexts
    error = initialize_lstm_contexts(*context);
    if (error != MPS_LSTM_COMPRESSOR_SUCCESS) {
        mps_lstm_compressor_destroy(*context);
        *context = NULL;
        return error;
    }
    
    // Build computation graphs
    error = build_compression_graphs(*context);
    if (error != MPS_LSTM_COMPRESSOR_SUCCESS) {
        mps_lstm_compressor_destroy(*context);
        *context = NULL;
        return error;
    }
    
    // Allocate buffers
    error = allocate_compressor_buffers(*context);
    if (error != MPS_LSTM_COMPRESSOR_SUCCESS) {
        mps_lstm_compressor_destroy(*context);
        *context = NULL;
        return error;
    }
    
    (*context)->is_initialized = true;
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

MPSLSTMCompressorError mps_lstm_compressor_load_weights(MPSLSTMCompressorContext* context,
                                                       const MPSLSTMCompressorWeights* weights) {
    if (!context || !weights) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    // Copy weights structure
    context->weights = *weights;
    
    // Load weights into LSTM contexts
    if (context->encoder_context && weights->encoder_input_weights) {
        // Create weights structure for encoder
        MPSLSTMWeights encoder_weights = {0};
        encoder_weights.input_weights = weights->encoder_input_weights;
        encoder_weights.hidden_weights = weights->encoder_hidden_weights;
        encoder_weights.cell_weights = weights->encoder_cell_weights;
        encoder_weights.output_weights = weights->encoder_output_weights;
        encoder_weights.biases = weights->encoder_biases;
        
        MPSLSTMError lstm_error = mps_lstm_load_weights(context->encoder_context, &encoder_weights);
        if (lstm_error != MPS_LSTM_SUCCESS) {
            return MPS_LSTM_COMPRESSOR_ERROR_COMPUTE_FAILED;
        }
    }
    
    if (context->decoder_context && weights->decoder_input_weights) {
        // Create weights structure for decoder
        MPSLSTMWeights decoder_weights = {0};
        decoder_weights.input_weights = weights->decoder_input_weights;
        decoder_weights.hidden_weights = weights->decoder_hidden_weights;
        decoder_weights.cell_weights = weights->decoder_cell_weights;
        decoder_weights.output_weights = weights->decoder_output_weights;
        decoder_weights.biases = weights->decoder_biases;
        
        MPSLSTMError lstm_error = mps_lstm_load_weights(context->decoder_context, &decoder_weights);
        if (lstm_error != MPS_LSTM_SUCCESS) {
            return MPS_LSTM_COMPRESSOR_ERROR_COMPUTE_FAILED;
        }
    }
    
    context->is_trained = true;
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

MPSLSTMCompressorError mps_lstm_compressor_compress(MPSLSTMCompressorContext* context,
                                                   MPSLSTMCompressorData* data) {
    if (!context || !data) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized || !context->is_trained) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    // Validate data
    MPSLSTMCompressorError error = mps_lstm_compressor_data_validate(data, &context->config);
    if (error != MPS_LSTM_COMPRESSOR_SUCCESS) {
        return error;
    }
    
    uint64_t start_time = get_compressor_timestamp_ns();
    
    // Execute compression based on mode
    uint32_t compressed_size = 0;
    switch (context->config.compression_mode) {
        case MPS_LSTM_COMPRESSION_PREDICTIVE:
            error = execute_predictive_compression(context, (const float*)data->input_data,
                                                  data->input_size, data->compressed_data,
                                                  &compressed_size);
            break;
            
        case MPS_LSTM_COMPRESSION_TEMPORAL:
            error = execute_temporal_compression(context, (const float*)data->input_data,
                                                data->input_size, data->compressed_data,
                                                &compressed_size);
            break;
            
        case MPS_LSTM_COMPRESSION_RESIDUAL:
            error = execute_residual_compression(context, (const float*)data->input_data,
                                                data->input_size, data->compressed_data,
                                                &compressed_size);
            break;
            
        default:
            return MPS_LSTM_COMPRESSOR_ERROR_UNSUPPORTED_OPERATION;
    }
    
    if (error != MPS_LSTM_COMPRESSOR_SUCCESS) {
        return error;
    }
    
    // Update compression metadata
    uint64_t compression_time = get_compressor_timestamp_ns() - start_time;
    
    data->compressed_size = compressed_size;
    data->metadata.original_size = data->input_size;
    data->metadata.compressed_size = compressed_size;
    data->metadata.compression_ratio = (float)data->input_size / compressed_size;
    data->metadata.compression_time_ns = compression_time;
    data->metadata.compression_mode = context->config.compression_mode;
    data->metadata.data_type = context->config.input_data_type;
    data->metadata.version = 1;
    data->metadata.checksum = calculate_checksum(data->compressed_data, compressed_size);
    
    // Update statistics
    context->stats.total_compressions++;
    context->stats.total_bytes_compressed += data->input_size;
    context->stats.total_compression_time_ns += compression_time;
    
    // Calculate averages
    context->stats.average_compression_ratio = 
        (context->stats.average_compression_ratio * (context->stats.total_compressions - 1) + 
         data->metadata.compression_ratio) / context->stats.total_compressions;
    
    context->stats.average_compression_time_ms = 
        (float)context->stats.total_compression_time_ns / (1000000.0f * context->stats.total_compressions);
    
    if (compression_time > 0) {
        float throughput = (float)data->input_size / (compression_time / 1000000000.0f) / (1024 * 1024);
        context->stats.throughput_mbps_compression = throughput;
    }
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

MPSLSTMCompressorError mps_lstm_compressor_decompress(MPSLSTMCompressorContext* context,
                                                     MPSLSTMCompressorData* data) {
    if (!context || !data) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized || !context->is_trained) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    if (!data->compressed_data || data->compressed_size == 0) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_compressor_timestamp_ns();
    
    // Verify checksum
    uint32_t calculated_checksum = calculate_checksum(data->compressed_data, data->compressed_size);
    if (calculated_checksum != data->metadata.checksum) {
        return MPS_LSTM_COMPRESSOR_ERROR_DECOMPRESSION_FAILED;
    }
    
    // Execute decompression
    uint32_t output_size = 0;
    MPSLSTMCompressorError error = execute_decompression_pipeline(context,
                                                                data->compressed_data,
                                                                data->compressed_size,
                                                                (float*)data->decompressed_data,
                                                                &output_size);
    
    if (error != MPS_LSTM_COMPRESSOR_SUCCESS) {
        return error;
    }
    
    // Update decompression metadata
    uint64_t decompression_time = get_compressor_timestamp_ns() - start_time;
    
    data->decompressed_size = output_size;
    data->metadata.decompression_time_ns = decompression_time;
    
    // Calculate quality metrics if original data is available
    if (data->input_data && data->decompressed_data) {
        data->metadata.mse = calculate_mse((const float*)data->input_data,
                                          (const float*)data->decompressed_data,
                                          output_size / sizeof(float));
        data->metadata.psnr = calculate_psnr(data->metadata.mse, 1.0f);
    }
    
    // Update statistics
    context->stats.total_decompressions++;
    context->stats.total_bytes_decompressed += output_size;
    context->stats.total_decompression_time_ns += decompression_time;
    
    context->stats.average_decompression_time_ms = 
        (float)context->stats.total_decompression_time_ns / (1000000.0f * context->stats.total_decompressions);
    
    if (decompression_time > 0) {
        float throughput = (float)output_size / (decompression_time / 1000000000.0f) / (1024 * 1024);
        context->stats.throughput_mbps_decompression = throughput;
    }
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

MPSLSTMCompressorError mps_lstm_compressor_compress_time_series(MPSLSTMCompressorContext* context,
                                                               const float* time_series,
                                                               void* compressed_output,
                                                               MPSLSTMCompressionMetadata* metadata) {
    if (!context || !time_series || !compressed_output || !metadata) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    // Create data structure for time series
    MPSLSTMCompressorData* data = NULL;
    uint32_t data_size = context->config.batch_size * context->config.sequence_length * 
                        context->config.input_size * sizeof(float);
    
    MPSLSTMCompressorError error = mps_lstm_compressor_data_create(&data, data_size,
                                                                  MPS_LSTM_DATA_TYPE_FLOAT32);
    if (error != MPS_LSTM_COMPRESSOR_SUCCESS) {
        return error;
    }
    
    // Set up data structure
    data->input_data = (void*)time_series;
    data->input_size = data_size;
    data->compressed_data = compressed_output;
    data->owns_memory = false;
    
    // Perform compression
    error = mps_lstm_compressor_compress(context, data);
    if (error == MPS_LSTM_COMPRESSOR_SUCCESS) {
        *metadata = data->metadata;
    }
    
    mps_lstm_compressor_data_destroy(data);
    return error;
}

// Configuration Functions

MPSLSTMCompressorError mps_lstm_compressor_config_create_default(MPSLSTMCompressorConfig* config,
                                                                MPSLSTMCompressionMode compression_mode,
                                                                uint32_t input_size,
                                                                uint32_t sequence_length,
                                                                MPSLSTMQualityLevel quality_level) {
    if (!config || input_size == 0 || sequence_length == 0) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    // Initialize configuration with defaults
    memset(config, 0, sizeof(MPSLSTMCompressorConfig));
    
    config->input_size = input_size;
    config->sequence_length = sequence_length;
    config->compression_mode = compression_mode;
    config->quality_level = quality_level;
    
    // Set defaults based on quality level
    switch (quality_level) {
        case MPS_LSTM_QUALITY_LOSSLESS:
            config->hidden_size = input_size * 2;
            config->num_layers = 3;
            config->compressed_data_type = MPS_LSTM_DATA_TYPE_FLOAT32;
            config->compression_ratio_target = 2.0f;
            config->residual_bits = 32;
            break;
            
        case MPS_LSTM_QUALITY_HIGH:
            config->hidden_size = input_size;
            config->num_layers = 2;
            config->compressed_data_type = MPS_LSTM_DATA_TYPE_FLOAT16;
            config->compression_ratio_target = 4.0f;
            config->residual_bits = 16;
            break;
            
        case MPS_LSTM_QUALITY_MEDIUM:
            config->hidden_size = input_size / 2;
            config->num_layers = 2;
            config->compressed_data_type = MPS_LSTM_DATA_TYPE_INT16;
            config->compression_ratio_target = 8.0f;
            config->residual_bits = 8;
            break;
            
        case MPS_LSTM_QUALITY_LOW:
            config->hidden_size = input_size / 4;
            config->num_layers = 1;
            config->compressed_data_type = MPS_LSTM_DATA_TYPE_INT8;
            config->compression_ratio_target = 16.0f;
            config->residual_bits = 4;
            break;
            
        default:
            config->hidden_size = input_size;
            config->num_layers = 2;
            config->compressed_data_type = MPS_LSTM_DATA_TYPE_FLOAT16;
            config->compression_ratio_target = 4.0f;
            config->residual_bits = 16;
            break;
    }
    
    // Set common defaults
    config->batch_size = 1;
    config->prediction_horizon = sequence_length / 4;
    config->input_data_type = MPS_LSTM_DATA_TYPE_FLOAT32;
    config->sequence_strategy = MPS_LSTM_SEQUENCE_FIXED;
    config->prediction_threshold = 0.01f;
    config->quantization_scale = 1.0f;
    config->enable_temporal_coherence = true;
    config->enable_adaptive_prediction = false;
    config->enable_entropy_coding = true;
    config->use_bidirectional = false;
    config->enable_attention = false;
    config->dropout_rate = 0.0f;
    config->num_threads = 1;
    config->enable_gpu_acceleration = true;
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

MPSLSTMCompressorError mps_lstm_compressor_config_validate(const MPSLSTMCompressorConfig* config) {
    if (!config) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    if (config->input_size == 0 || config->input_size > 8192) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->hidden_size == 0 || config->hidden_size > 4096) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->num_layers == 0 || config->num_layers > 8) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->sequence_length == 0 || config->sequence_length > 32768) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->batch_size == 0 || config->batch_size > 128) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->compression_ratio_target <= 1.0f || config->compression_ratio_target > 1000.0f) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_COMPRESSION_RATIO;
    }
    
    if (config->prediction_threshold < 0.0f || config->prediction_threshold > 1.0f) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    if (config->dropout_rate < 0.0f || config->dropout_rate >= 1.0f) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

MPSLSTMCompressorError mps_lstm_compressor_calculate_memory_requirements(const MPSLSTMCompressorConfig* config,
                                                                        uint32_t* memory_mb) {
    if (!config || !memory_mb) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    uint64_t total_memory = 0;
    
    // LSTM weights memory (encoder + decoder + predictor)
    uint64_t weights_per_layer = (uint64_t)config->input_size * config->hidden_size * 4 + // Input weights
                                (uint64_t)config->hidden_size * config->hidden_size * 4 + // Hidden weights
                                (uint64_t)config->hidden_size * 4; // Biases
    
    total_memory += weights_per_layer * config->num_layers * 3; // Encoder, decoder, predictor
    
    // Activation memory
    uint64_t activation_memory = (uint64_t)config->batch_size * config->sequence_length * 
                                config->hidden_size * sizeof(float) * config->num_layers * 2; // Hidden + cell states
    
    total_memory += activation_memory * 3; // Encoder, decoder, predictor
    
    // Input/output buffers
    uint64_t io_memory = (uint64_t)config->batch_size * config->sequence_length * 
                        config->input_size * sizeof(float) * 4; // Input, compressed, decompressed, residual
    
    total_memory += io_memory;
    
    // Temporary buffers
    total_memory += activation_memory; // Additional temporary storage
    
    // Add overhead for Metal buffers (estimated 20% overhead)
    total_memory = total_memory * 6 / 5;
    
    *memory_mb = (uint32_t)((total_memory + 1024 * 1024 - 1) / (1024 * 1024));
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

// Weight Management Functions

MPSLSTMCompressorError mps_lstm_compressor_weights_create(MPSLSTMCompressorWeights** weights,
                                                         const MPSLSTMCompressorConfig* config) {
    if (!weights || !config) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    *weights = (MPSLSTMCompressorWeights*)calloc(1, sizeof(MPSLSTMCompressorWeights));
    if (!*weights) {
        return MPS_LSTM_COMPRESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Calculate weight sizes
    size_t input_weights_size = config->input_size * config->hidden_size * 4 * sizeof(float);
    size_t hidden_weights_size = config->hidden_size * config->hidden_size * 4 * sizeof(float);
    size_t bias_size = config->hidden_size * 4 * sizeof(float);
    size_t output_weights_size = config->hidden_size * config->input_size * sizeof(float);
    
    // Allocate encoder weights
    (*weights)->encoder_input_weights = malloc(input_weights_size);
    (*weights)->encoder_hidden_weights = malloc(hidden_weights_size);
    (*weights)->encoder_cell_weights = malloc(hidden_weights_size);
    (*weights)->encoder_output_weights = malloc(output_weights_size);
    (*weights)->encoder_biases = malloc(bias_size);
    
    // Allocate decoder weights
    (*weights)->decoder_input_weights = malloc(input_weights_size);
    (*weights)->decoder_hidden_weights = malloc(hidden_weights_size);
    (*weights)->decoder_cell_weights = malloc(hidden_weights_size);
    (*weights)->decoder_output_weights = malloc(output_weights_size);
    (*weights)->decoder_biases = malloc(bias_size);
    
    // Allocate predictor weights
    (*weights)->predictor_weights = malloc(config->hidden_size * config->input_size * sizeof(float));
    (*weights)->predictor_biases = malloc(config->input_size * sizeof(float));
    
    // Check allocations
    if (!(*weights)->encoder_input_weights || !(*weights)->encoder_hidden_weights ||
        !(*weights)->encoder_cell_weights || !(*weights)->encoder_output_weights ||
        !(*weights)->encoder_biases || !(*weights)->decoder_input_weights ||
        !(*weights)->decoder_hidden_weights || !(*weights)->decoder_cell_weights ||
        !(*weights)->decoder_output_weights || !(*weights)->decoder_biases ||
        !(*weights)->predictor_weights || !(*weights)->predictor_biases) {
        mps_lstm_compressor_weights_destroy(*weights);
        *weights = NULL;
        return MPS_LSTM_COMPRESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Calculate total size
    (*weights)->total_weights_size = (input_weights_size + hidden_weights_size + 
                                     hidden_weights_size + output_weights_size + bias_size) * 2 + // Encoder + decoder
                                    config->hidden_size * config->input_size * sizeof(float) + // Predictor weights
                                    config->input_size * sizeof(float); // Predictor biases
    
    (*weights)->weights_version = 1;
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

MPSLSTMCompressorError mps_lstm_compressor_weights_init_random(MPSLSTMCompressorWeights* weights,
                                                              const MPSLSTMCompressorConfig* config,
                                                              uint32_t seed) {
    if (!weights || !config) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    srand(seed);
    
    // Xavier/Glorot initialization
    float input_scale = sqrtf(2.0f / (config->input_size + config->hidden_size));
    float hidden_scale = sqrtf(2.0f / (config->hidden_size + config->hidden_size));
    
    // Initialize encoder weights
    float* weight_arrays[] = {
        (float*)weights->encoder_input_weights,
        (float*)weights->encoder_hidden_weights,
        (float*)weights->encoder_cell_weights,
        (float*)weights->encoder_output_weights,
        (float*)weights->decoder_input_weights,
        (float*)weights->decoder_hidden_weights,
        (float*)weights->decoder_cell_weights,
        (float*)weights->decoder_output_weights,
        (float*)weights->predictor_weights
    };
    
    size_t weight_sizes[] = {
        config->input_size * config->hidden_size * 4,
        config->hidden_size * config->hidden_size * 4,
        config->hidden_size * config->hidden_size * 4,
        config->hidden_size * config->input_size,
        config->input_size * config->hidden_size * 4,
        config->hidden_size * config->hidden_size * 4,
        config->hidden_size * config->hidden_size * 4,
        config->hidden_size * config->input_size,
        config->hidden_size * config->input_size
    };
    
    float scales[] = {
        input_scale, hidden_scale, hidden_scale, hidden_scale,
        input_scale, hidden_scale, hidden_scale, hidden_scale,
        hidden_scale
    };
    
    for (int i = 0; i < 9; i++) {
        if (weight_arrays[i]) {
            for (size_t j = 0; j < weight_sizes[i]; j++) {
                float u1 = (float)rand() / RAND_MAX;
                float u2 = (float)rand() / RAND_MAX;
                float normal = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
                weight_arrays[i][j] = normal * scales[i];
            }
        }
    }
    
    // Initialize biases to zero
    if (weights->encoder_biases) {
        memset(weights->encoder_biases, 0, config->hidden_size * 4 * sizeof(float));
    }
    if (weights->decoder_biases) {
        memset(weights->decoder_biases, 0, config->hidden_size * 4 * sizeof(float));
    }
    if (weights->predictor_biases) {
        memset(weights->predictor_biases, 0, config->input_size * sizeof(float));
    }
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

void mps_lstm_compressor_weights_destroy(MPSLSTMCompressorWeights* weights) {
    if (!weights) {
        return;
    }
    
    free(weights->encoder_input_weights);
    free(weights->encoder_hidden_weights);
    free(weights->encoder_cell_weights);
    free(weights->encoder_output_weights);
    free(weights->encoder_biases);
    free(weights->decoder_input_weights);
    free(weights->decoder_hidden_weights);
    free(weights->decoder_cell_weights);
    free(weights->decoder_output_weights);
    free(weights->decoder_biases);
    free(weights->predictor_weights);
    free(weights->predictor_biases);
    free(weights->quantizer_codebook);
    free(weights->quantizer_mapping);
    
    free(weights);
}

// Data Management Functions

MPSLSTMCompressorError mps_lstm_compressor_data_create(MPSLSTMCompressorData** data,
                                                      uint32_t input_size,
                                                      MPSLSTMDataType data_type) {
    if (!data) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    *data = (MPSLSTMCompressorData*)calloc(1, sizeof(MPSLSTMCompressorData));
    if (!*data) {
        return MPS_LSTM_COMPRESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    (*data)->input_size = input_size;
    (*data)->input_type = data_type;
    (*data)->owns_memory = false;
    
    // Initialize metadata
    memset(&(*data)->metadata, 0, sizeof(MPSLSTMCompressionMetadata));
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

void mps_lstm_compressor_data_destroy(MPSLSTMCompressorData* data) {
    if (!data) {
        return;
    }
    
    if (data->owns_memory) {
        free(data->input_data);
        free(data->compressed_data);
        free(data->decompressed_data);
        free(data->sequence_lengths);
    }
    
    free(data);
}

MPSLSTMCompressorError mps_lstm_compressor_data_validate(const MPSLSTMCompressorData* data,
                                                        const MPSLSTMCompressorConfig* config) {
    if (!data || !config) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    if (!data->input_data || data->input_size == 0) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    if (data->batch_size > config->batch_size) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_DIMENSIONS;
    }
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

// Statistics Functions

MPSLSTMCompressorError mps_lstm_compressor_get_stats(MPSLSTMCompressorContext* context,
                                                    MPSLSTMCompressorStats* stats) {
    if (!context || !stats) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    *stats = context->stats;
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

MPSLSTMCompressorError mps_lstm_compressor_reset_stats(MPSLSTMCompressorContext* context) {
    if (!context) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    memset(&context->stats, 0, sizeof(MPSLSTMCompressorStats));
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

MPSLSTMCompressorError mps_lstm_compressor_get_memory_usage(MPSLSTMCompressorContext* context,
                                                           uint32_t* memory_usage_mb) {
    if (!context || !memory_usage_mb) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    // Calculate current memory usage from buffers
    uint64_t total_memory = 0;
    for (int i = 0; i < 16; i++) {
        total_memory += context->buffers.buffer_sizes[i];
    }
    
    *memory_usage_mb = (uint32_t)((total_memory + 1024 * 1024 - 1) / (1024 * 1024));
    context->stats.memory_usage_mb = *memory_usage_mb;
    
    if (*memory_usage_mb > context->stats.peak_memory_usage_mb) {
        context->stats.peak_memory_usage_mb = *memory_usage_mb;
    }
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

// Utility Functions

const char* mps_lstm_compressor_get_error_string(MPSLSTMCompressorError error_code) {
    if (error_code < 0 || error_code >= sizeof(compressor_error_messages) / sizeof(compressor_error_messages[0])) {
        return "Unknown error";
    }
    return compressor_error_messages[error_code];
}

bool mps_lstm_compressor_is_available(void) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device && [device supportsFamily:MTLGPUFamilyApple4]) {
            return true;
        }
    }
#endif
    return false;
}

MPSLSTMCompressorError mps_lstm_compressor_get_device_info(char* device_name,
                                                          size_t buffer_size,
                                                          uint32_t* compute_units,
                                                          uint32_t* max_memory_mb) {
    if (!device_name || !compute_units || !max_memory_mb) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return MPS_LSTM_COMPRESSOR_ERROR_DEVICE_NOT_FOUND;
        }
        
        NSString* name = device.name;
        strncpy(device_name, [name UTF8String], buffer_size - 1);
        device_name[buffer_size - 1] = '\0';
        
        if (@available(macOS 10.15, *)) {
            *compute_units = (uint32_t)device.maxThreadsPerThreadgroup.width;
            *max_memory_mb = (uint32_t)(device.recommendedMaxWorkingSetSize / (1024 * 1024));
        } else {
            *compute_units = 256;
            *max_memory_mb = 1024;
        }
    }
#else
    strncpy(device_name, "MPS Not Available", buffer_size - 1);
    device_name[buffer_size - 1] = '\0';
    *compute_units = 0;
    *max_memory_mb = 0;
#endif
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

void mps_lstm_compressor_destroy(MPSLSTMCompressorContext* context) {
    if (!context) {
        return;
    }
    
    // Destroy LSTM contexts
    if (context->encoder_context) {
        mps_lstm_destroy(context->encoder_context);
    }
    if (context->decoder_context) {
        mps_lstm_destroy(context->decoder_context);
    }
    if (context->predictor_context) {
        mps_lstm_destroy(context->predictor_context);
    }
    
    // Clear Metal objects
#if MPS_AVAILABLE
    context->device = NULL;
    context->command_queue = NULL;
#endif
    
    free(context);
}

// Internal helper function implementations

static MPSLSTMCompressorError create_compressor_device(MPSLSTMCompressorContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return MPS_LSTM_COMPRESSOR_ERROR_DEVICE_NOT_FOUND;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            return MPS_LSTM_COMPRESSOR_ERROR_DEVICE_NOT_FOUND;
        }
        
        context->device = (__bridge void*)device;
        context->command_queue = (__bridge void*)commandQueue;
    }
    return MPS_LSTM_COMPRESSOR_SUCCESS;
#else
    return MPS_LSTM_COMPRESSOR_ERROR_DEVICE_NOT_FOUND;
#endif
}

static MPSLSTMCompressorError initialize_lstm_contexts(MPSLSTMCompressorContext* context) {
    // Create LSTM configurations
    MPSLSTMConfig encoder_config;
    MPSLSTMError error = mps_lstm_config_create_default(&encoder_config,
                                                       context->config.input_size,
                                                       context->config.hidden_size,
                                                       context->config.num_layers,
                                                       context->config.batch_size,
                                                       context->config.sequence_length);
    if (error != MPS_LSTM_SUCCESS) {
        return MPS_LSTM_COMPRESSOR_ERROR_COMPUTE_FAILED;
    }
    
    // Create encoder context
    error = mps_lstm_create(&context->encoder_context, &encoder_config);
    if (error != MPS_LSTM_SUCCESS) {
        return MPS_LSTM_COMPRESSOR_ERROR_COMPUTE_FAILED;
    }
    
    // Create decoder context (same configuration for now)
    error = mps_lstm_create(&context->decoder_context, &encoder_config);
    if (error != MPS_LSTM_SUCCESS) {
        return MPS_LSTM_COMPRESSOR_ERROR_COMPUTE_FAILED;
    }
    
    // Create predictor context (smaller hidden size)
    MPSLSTMConfig predictor_config = encoder_config;
    predictor_config.hidden_size = context->config.hidden_size / 2;
    
    error = mps_lstm_create(&context->predictor_context, &predictor_config);
    if (error != MPS_LSTM_SUCCESS) {
        return MPS_LSTM_COMPRESSOR_ERROR_COMPUTE_FAILED;
    }
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

static MPSLSTMCompressorError build_compression_graphs(MPSLSTMCompressorContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        // Create computation graphs for compression pipeline
        MPSGraph* encoderGraph = [[MPSGraph alloc] init];
        MPSGraph* decoderGraph = [[MPSGraph alloc] init];
        MPSGraph* predictorGraph = [[MPSGraph alloc] init];
        
        if (!encoderGraph || !decoderGraph || !predictorGraph) {
            return MPS_LSTM_COMPRESSOR_ERROR_GRAPH_COMPILATION;
        }
        
        context->graphs.encoder_graph = (__bridge void*)encoderGraph;
        context->graphs.decoder_graph = (__bridge void*)decoderGraph;
        context->graphs.predictor_graph = (__bridge void*)predictorGraph;
        
        // In a full implementation, this would build the complete compression graphs
        context->graphs.graphs_compiled = true;
        
        return MPS_LSTM_COMPRESSOR_SUCCESS;
    }
#else
    return MPS_LSTM_COMPRESSOR_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static MPSLSTMCompressorError allocate_compressor_buffers(MPSLSTMCompressorContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        
        // Calculate buffer sizes
        size_t input_size = context->config.batch_size * context->config.sequence_length * 
                           context->config.input_size * sizeof(float);
        size_t hidden_size = context->config.batch_size * context->config.sequence_length * 
                            context->config.hidden_size * sizeof(float);
        
        // Allocate main buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithLength:input_size
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> compressedBuffer = [device newBufferWithLength:input_size
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> hiddenStatesBuffer = [device newBufferWithLength:hidden_size
                                                               options:MTLResourceStorageModeShared];
        
        if (!inputBuffer || !compressedBuffer || !hiddenStatesBuffer) {
            return MPS_LSTM_COMPRESSOR_ERROR_BUFFER_ALLOCATION;
        }
        
        context->buffers.input_buffer = (__bridge void*)inputBuffer;
        context->buffers.compressed_buffer = (__bridge void*)compressedBuffer;
        context->buffers.hidden_states_buffer = (__bridge void*)hiddenStatesBuffer;
        
        // Store buffer sizes
        context->buffers.buffer_sizes[0] = input_size;
        context->buffers.buffer_sizes[1] = input_size;
        context->buffers.buffer_sizes[2] = hidden_size;
        
        // Allocate temporary buffers
        for (int i = 0; i < 8; i++) {
            id<MTLBuffer> tempBuffer = [device newBufferWithLength:hidden_size
                                                           options:MTLResourceStorageModePrivate];
            if (!tempBuffer) {
                return MPS_LSTM_COMPRESSOR_ERROR_BUFFER_ALLOCATION;
            }
            context->buffers.temp_buffers[i] = (__bridge void*)tempBuffer;
            context->buffers.buffer_sizes[8 + i] = hidden_size;
        }
        
        context->buffers.num_allocated_buffers = 11;
    }
    return MPS_LSTM_COMPRESSOR_SUCCESS;
#else
    return MPS_LSTM_COMPRESSOR_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static MPSLSTMCompressorError execute_predictive_compression(MPSLSTMCompressorContext* context,
                                                           const float* input_data,
                                                           uint32_t input_size,
                                                           void* compressed_output,
                                                           uint32_t* compressed_size) {
    // Simplified predictive compression implementation
    // In a real implementation, this would use the LSTM to predict future values
    // and encode only the prediction errors
    
    if (!input_data || !compressed_output || !compressed_size) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    // For now, implement a simple quantization-based compression
    float* output = (float*)compressed_output;
    uint32_t num_elements = input_size / sizeof(float);
    
    // Simple prediction using linear extrapolation
    for (uint32_t i = 0; i < num_elements; i++) {
        float predicted = (i > 0) ? input_data[i - 1] : 0.0f;
        float residual = input_data[i] - predicted;
        
        // Quantize residual
        output[i] = roundf(residual * context->config.quantization_scale) / context->config.quantization_scale;
    }
    
    *compressed_size = input_size; // For this simple implementation
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

static MPSLSTMCompressorError execute_temporal_compression(MPSLSTMCompressorContext* context,
                                                         const float* input_data,
                                                         uint32_t input_size,
                                                         void* compressed_output,
                                                         uint32_t* compressed_size) {
    // Simplified temporal compression implementation
    // This would analyze temporal patterns and compress based on repetition
    
    if (!input_data || !compressed_output || !compressed_size) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    // For now, implement simple downsampling
    float* output = (float*)compressed_output;
    uint32_t num_elements = input_size / sizeof(float);
    uint32_t downsample_factor = 2;
    
    for (uint32_t i = 0; i < num_elements / downsample_factor; i++) {
        output[i] = input_data[i * downsample_factor];
    }
    
    *compressed_size = input_size / downsample_factor;
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

static MPSLSTMCompressorError execute_residual_compression(MPSLSTMCompressorContext* context,
                                                         const float* input_data,
                                                         uint32_t input_size,
                                                         void* compressed_output,
                                                         uint32_t* compressed_size) {
    // Simplified residual compression implementation
    // This would encode differences between consecutive values
    
    if (!input_data || !compressed_output || !compressed_size) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    int16_t* output = (int16_t*)compressed_output;
    uint32_t num_elements = input_size / sizeof(float);
    
    // First value stored as-is (quantized)
    output[0] = (int16_t)(input_data[0] * 1000.0f);
    
    // Store differences
    for (uint32_t i = 1; i < num_elements; i++) {
        float diff = input_data[i] - input_data[i - 1];
        output[i] = (int16_t)(diff * 1000.0f);
    }
    
    *compressed_size = num_elements * sizeof(int16_t);
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

static MPSLSTMCompressorError execute_decompression_pipeline(MPSLSTMCompressorContext* context,
                                                           const void* compressed_data,
                                                           uint32_t compressed_size,
                                                           float* output_data,
                                                           uint32_t* output_size) {
    // Simplified decompression implementation
    // This would reverse the compression process
    
    if (!compressed_data || !output_data || !output_size) {
        return MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM;
    }
    
    // For now, implement simple decompression based on compression mode
    switch (context->config.compression_mode) {
        case MPS_LSTM_COMPRESSION_PREDICTIVE: {
            const float* input = (const float*)compressed_data;
            uint32_t num_elements = compressed_size / sizeof(float);
            
            for (uint32_t i = 0; i < num_elements; i++) {
                float predicted = (i > 0) ? output_data[i - 1] : 0.0f;
                output_data[i] = predicted + input[i];
            }
            
            *output_size = compressed_size;
            break;
        }
        
        case MPS_LSTM_COMPRESSION_TEMPORAL: {
            const float* input = (const float*)compressed_data;
            uint32_t num_elements = compressed_size / sizeof(float);
            uint32_t upsample_factor = 2;
            
            for (uint32_t i = 0; i < num_elements; i++) {
                output_data[i * upsample_factor] = input[i];
                if (i * upsample_factor + 1 < num_elements * upsample_factor) {
                    output_data[i * upsample_factor + 1] = input[i]; // Simple duplication
                }
            }
            
            *output_size = compressed_size * upsample_factor;
            break;
        }
        
        case MPS_LSTM_COMPRESSION_RESIDUAL: {
            const int16_t* input = (const int16_t*)compressed_data;
            uint32_t num_elements = compressed_size / sizeof(int16_t);
            
            output_data[0] = (float)input[0] / 1000.0f;
            
            for (uint32_t i = 1; i < num_elements; i++) {
                float diff = (float)input[i] / 1000.0f;
                output_data[i] = output_data[i - 1] + diff;
            }
            
            *output_size = num_elements * sizeof(float);
            break;
        }
        
        default:
            return MPS_LSTM_COMPRESSOR_ERROR_UNSUPPORTED_OPERATION;
    }
    
    return MPS_LSTM_COMPRESSOR_SUCCESS;
}

static uint64_t get_compressor_timestamp_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

static uint32_t calculate_checksum(const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t checksum = 0;
    
    for (size_t i = 0; i < size; i++) {
        checksum = checksum * 31 + bytes[i];
    }
    
    return checksum;
}

static float calculate_mse(const float* original, const float* reconstructed, uint32_t size) {
    float sum = 0.0f;
    
    for (uint32_t i = 0; i < size; i++) {
        float diff = original[i] - reconstructed[i];
        sum += diff * diff;
    }
    
    return sum / size;
}

static float calculate_psnr(float mse, float max_value) {
    if (mse == 0.0f) {
        return INFINITY;
    }
    
    return 20.0f * log10f(max_value / sqrtf(mse));
}
