#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "../../src/neural/engines/mps_lstm_compressor.h"

// Test framework
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("FAIL: %s - %s\n", __func__, message); \
            return 0; \
        } \
    } while(0)

#define TEST_PASS() \
    do { \
        printf("PASS: %s\n", __func__); \
        return 1; \
    } while(0)

#define FLOAT_TOLERANCE 1e-4f

// Helper function to check if two floats are approximately equal
static bool floats_equal(float a, float b, float tolerance) {
    return fabsf(a - b) < tolerance;
}

// Helper function to generate test time series data
static void generate_test_time_series(float* data, size_t size, float amplitude, float frequency) {
    for (size_t i = 0; i < size; i++) {
        float t = (float)i / size;
        data[i] = amplitude * sinf(2.0f * M_PI * frequency * t) + 
                  0.1f * amplitude * sinf(2.0f * M_PI * frequency * 3.0f * t) +
                  0.05f * amplitude * ((float)rand() / RAND_MAX - 0.5f);
    }
}

// Helper function to generate synthetic sequence data
static void generate_test_sequences(float* data, size_t total_size, uint32_t sequence_length) {
    for (size_t i = 0; i < total_size; i += sequence_length) {
        float base_value = (float)(i / sequence_length) * 0.1f;
        for (uint32_t j = 0; j < sequence_length && i + j < total_size; j++) {
            data[i + j] = base_value + (float)j / sequence_length;
        }
    }
}

// Test functions

int test_compressor_availability() {
    bool available = mps_lstm_compressor_is_available();
    printf("  MPS LSTM Compressor available: %s\n", available ? "Yes" : "No");
    
    if (available) {
        char device_name[256];
        uint32_t compute_units, max_memory_mb;
        MPSLSTMCompressorError error = mps_lstm_compressor_get_device_info(device_name, sizeof(device_name),
                                                                          &compute_units, &max_memory_mb);
        TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Getting device info should succeed");
        
        printf("  Device: %s\n", device_name);
        printf("  Compute units: %u\n", compute_units);
        printf("  Max memory: %u MB\n", max_memory_mb);
    }
    
    TEST_PASS();
}

int test_config_creation_and_validation() {
    MPSLSTMCompressorConfig config;
    
    // Test lossless compression configuration
    MPSLSTMCompressorError error = mps_lstm_compressor_config_create_default(&config,
                                                                            MPS_LSTM_COMPRESSION_PREDICTIVE,
                                                                            16, 128,
                                                                            MPS_LSTM_QUALITY_LOSSLESS);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Lossless config creation should succeed");
    TEST_ASSERT(config.input_size == 16, "Input size should be 16");
    TEST_ASSERT(config.sequence_length == 128, "Sequence length should be 128");
    TEST_ASSERT(config.compression_mode == MPS_LSTM_COMPRESSION_PREDICTIVE, "Should be predictive compression");
    TEST_ASSERT(config.quality_level == MPS_LSTM_QUALITY_LOSSLESS, "Should be lossless quality");
    TEST_ASSERT(config.hidden_size == 32, "Hidden size should be 2x input size for lossless");
    TEST_ASSERT(config.compressed_data_type == MPS_LSTM_DATA_TYPE_FLOAT32, "Should use float32 for lossless");
    
    printf("  Lossless config: input=%u, seq_len=%u, hidden=%u, layers=%u\n",
           config.input_size, config.sequence_length, config.hidden_size, config.num_layers);
    
    // Test high quality configuration
    error = mps_lstm_compressor_config_create_default(&config,
                                                     MPS_LSTM_COMPRESSION_TEMPORAL,
                                                     32, 256,
                                                     MPS_LSTM_QUALITY_HIGH);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "High quality config creation should succeed");
    TEST_ASSERT(config.compression_mode == MPS_LSTM_COMPRESSION_TEMPORAL, "Should be temporal compression");
    TEST_ASSERT(config.quality_level == MPS_LSTM_QUALITY_HIGH, "Should be high quality");
    TEST_ASSERT(config.compressed_data_type == MPS_LSTM_DATA_TYPE_FLOAT16, "Should use float16 for high quality");
    
    printf("  High quality config: input=%u, seq_len=%u, hidden=%u, ratio=%.1f\n",
           config.input_size, config.sequence_length, config.hidden_size, config.compression_ratio_target);
    
    // Test low quality configuration
    error = mps_lstm_compressor_config_create_default(&config,
                                                     MPS_LSTM_COMPRESSION_RESIDUAL,
                                                     8, 64,
                                                     MPS_LSTM_QUALITY_LOW);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Low quality config creation should succeed");
    TEST_ASSERT(config.compression_mode == MPS_LSTM_COMPRESSION_RESIDUAL, "Should be residual compression");
    TEST_ASSERT(config.quality_level == MPS_LSTM_QUALITY_LOW, "Should be low quality");
    TEST_ASSERT(config.compressed_data_type == MPS_LSTM_DATA_TYPE_INT8, "Should use int8 for low quality");
    TEST_ASSERT(config.compression_ratio_target == 16.0f, "Should target 16:1 compression for low quality");
    
    printf("  Low quality config: input=%u, seq_len=%u, hidden=%u, ratio=%.1f\n",
           config.input_size, config.sequence_length, config.hidden_size, config.compression_ratio_target);
    
    // Test configuration validation
    error = mps_lstm_compressor_config_validate(&config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Config validation should succeed");
    
    // Test invalid configurations
    MPSLSTMCompressorConfig invalid_config = config;
    invalid_config.input_size = 0;
    error = mps_lstm_compressor_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_ERROR_INVALID_DIMENSIONS, "Zero input size should fail");
    
    invalid_config = config;
    invalid_config.compression_ratio_target = 0.5f;
    error = mps_lstm_compressor_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_ERROR_INVALID_COMPRESSION_RATIO, "Invalid compression ratio should fail");
    
    printf("  Config validation tests passed\n");
    
    TEST_PASS();
}

int test_memory_requirements_calculation() {
    MPSLSTMCompressorConfig config;
    mps_lstm_compressor_config_create_default(&config, MPS_LSTM_COMPRESSION_PREDICTIVE, 16, 128, MPS_LSTM_QUALITY_MEDIUM);
    config.batch_size = 4;
    
    uint32_t memory_mb;
    MPSLSTMCompressorError error = mps_lstm_compressor_calculate_memory_requirements(&config, &memory_mb);
    
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Memory calculation should succeed");
    TEST_ASSERT(memory_mb > 0, "Memory requirement should be positive");
    
    printf("  Memory requirements for medium quality: %u MB\n", memory_mb);
    
    // Test larger configuration
    config.input_size = 64;
    config.sequence_length = 512;
    config.hidden_size = 128;
    config.num_layers = 3;
    config.batch_size = 8;
    
    uint32_t larger_memory_mb;
    error = mps_lstm_compressor_calculate_memory_requirements(&config, &larger_memory_mb);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Larger config memory calculation should succeed");
    TEST_ASSERT(larger_memory_mb > memory_mb, "Larger config should require more memory");
    
    printf("  Memory requirements for large config: %u MB\n", larger_memory_mb);
    
    TEST_PASS();
}

int test_error_string_functions() {
    const char* success_msg = mps_lstm_compressor_get_error_string(MPS_LSTM_COMPRESSOR_SUCCESS);
    TEST_ASSERT(success_msg != NULL, "Success error string should not be NULL");
    TEST_ASSERT(strlen(success_msg) > 0, "Success error string should not be empty");
    
    const char* invalid_param_msg = mps_lstm_compressor_get_error_string(MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM);
    TEST_ASSERT(invalid_param_msg != NULL, "Invalid param error string should not be NULL");
    TEST_ASSERT(strlen(invalid_param_msg) > 0, "Invalid param error string should not be empty");
    
    printf("  MPS_LSTM_COMPRESSOR_SUCCESS: %s\n", success_msg);
    printf("  MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM: %s\n", invalid_param_msg);
    
    // Test unknown error code
    const char* unknown_msg = mps_lstm_compressor_get_error_string((MPSLSTMCompressorError)999);
    TEST_ASSERT(unknown_msg != NULL, "Unknown error should return a message");
    
    TEST_PASS();
}

int test_context_creation_and_destruction() {
    MPSLSTMCompressorConfig config;
    mps_lstm_compressor_config_create_default(&config, MPS_LSTM_COMPRESSION_PREDICTIVE, 8, 32, MPS_LSTM_QUALITY_MEDIUM);
    config.batch_size = 2;
    
    MPSLSTMCompressorContext* context = NULL;
    MPSLSTMCompressorError error = mps_lstm_compressor_create(&context, &config);
    
    if (!mps_lstm_compressor_is_available()) {
        printf("  MPS not available, skipping context creation test\n");
        TEST_PASS();
    }
    
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Context creation should succeed");
    TEST_ASSERT(context != NULL, "Context should not be NULL");
    TEST_ASSERT(context->is_initialized == true, "Context should be initialized");
    TEST_ASSERT(context->config.input_size == 8, "Config should be copied correctly");
    TEST_ASSERT(context->config.sequence_length == 32, "Config should be copied correctly");
    
    printf("  Created compressor context with input=%u, seq_len=%u, hidden=%u\n",
           context->config.input_size, context->config.sequence_length, context->config.hidden_size);
    
    // Test memory usage
    uint32_t memory_usage;
    error = mps_lstm_compressor_get_memory_usage(context, &memory_usage);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Getting memory usage should succeed");
    printf("  Context memory usage: %u MB\n", memory_usage);
    
    // Test LSTM contexts initialization
    TEST_ASSERT(context->encoder_context != NULL, "Encoder context should be initialized");
    TEST_ASSERT(context->decoder_context != NULL, "Decoder context should be initialized");
    TEST_ASSERT(context->predictor_context != NULL, "Predictor context should be initialized");
    printf("  LSTM contexts initialized successfully\n");
    
    mps_lstm_compressor_destroy(context);
    TEST_PASS();
}

int test_weights_creation_and_initialization() {
    MPSLSTMCompressorConfig config;
    mps_lstm_compressor_config_create_default(&config, MPS_LSTM_COMPRESSION_TEMPORAL, 4, 16, MPS_LSTM_QUALITY_HIGH);
    
    MPSLSTMCompressorWeights* weights = NULL;
    MPSLSTMCompressorError error = mps_lstm_compressor_weights_create(&weights, &config);
    
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights creation should succeed");
    TEST_ASSERT(weights != NULL, "Weights should not be NULL");
    TEST_ASSERT(weights->encoder_input_weights != NULL, "Encoder input weights should be allocated");
    TEST_ASSERT(weights->decoder_input_weights != NULL, "Decoder input weights should be allocated");
    TEST_ASSERT(weights->predictor_weights != NULL, "Predictor weights should be allocated");
    
    printf("  Weights created: total_size=%zu, version=%u\n",
           weights->total_weights_size, weights->weights_version);
    
    // Test random initialization
    error = mps_lstm_compressor_weights_init_random(weights, &config, 12345);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Random weight initialization should succeed");
    
    // Check that weights are not all zero
    float* encoder_weights = (float*)weights->encoder_input_weights;
    bool has_nonzero = false;
    for (size_t i = 0; i < config.input_size * config.hidden_size * 4 && !has_nonzero; i++) {
        if (fabsf(encoder_weights[i]) > 1e-8f) {
            has_nonzero = true;
        }
    }
    TEST_ASSERT(has_nonzero, "Weights should not all be zero after initialization");
    printf("  Random initialization successful\n");
    
    mps_lstm_compressor_weights_destroy(weights);
    TEST_PASS();
}

int test_data_creation_and_validation() {
    uint32_t data_size = 256 * sizeof(float);
    
    MPSLSTMCompressorData* data = NULL;
    MPSLSTMCompressorError error = mps_lstm_compressor_data_create(&data, data_size, MPS_LSTM_DATA_TYPE_FLOAT32);
    
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Data creation should succeed");
    TEST_ASSERT(data != NULL, "Data should not be NULL");
    TEST_ASSERT(data->input_size == data_size, "Input size should be set correctly");
    TEST_ASSERT(data->input_type == MPS_LSTM_DATA_TYPE_FLOAT32, "Input type should be set correctly");
    
    printf("  Data created with size=%u, type=%d\n", data->input_size, data->input_type);
    
    // Allocate test data
    data->input_data = malloc(data_size);
    data->compressed_data = malloc(data_size);
    data->decompressed_data = malloc(data_size);
    data->owns_memory = true;
    
    TEST_ASSERT(data->input_data != NULL, "Input data should be allocated");
    TEST_ASSERT(data->compressed_data != NULL, "Compressed data should be allocated");
    TEST_ASSERT(data->decompressed_data != NULL, "Decompressed data should be allocated");
    
    // Initialize test data
    float* input = (float*)data->input_data;
    generate_test_time_series(input, data_size / sizeof(float), 1.0f, 2.0f);
    
    // Test data validation
    MPSLSTMCompressorConfig config;
    mps_lstm_compressor_config_create_default(&config, MPS_LSTM_COMPRESSION_PREDICTIVE, 4, 64, MPS_LSTM_QUALITY_MEDIUM);
    data->batch_size = 1;
    
    error = mps_lstm_compressor_data_validate(data, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Data validation should succeed");
    
    printf("  Data validation passed\n");
    
    mps_lstm_compressor_data_destroy(data);
    TEST_PASS();
}

int test_predictive_compression_basic() {
    if (!mps_lstm_compressor_is_available()) {
        printf("  MPS not available, skipping predictive compression test\n");
        TEST_PASS();
    }
    
    MPSLSTMCompressorConfig config;
    mps_lstm_compressor_config_create_default(&config, MPS_LSTM_COMPRESSION_PREDICTIVE, 4, 32, MPS_LSTM_QUALITY_MEDIUM);
    config.batch_size = 1;
    
    MPSLSTMCompressorContext* context = NULL;
    MPSLSTMCompressorError error = mps_lstm_compressor_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSLSTMCompressorWeights* weights = NULL;
    error = mps_lstm_compressor_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_compressor_weights_init_random(weights, &config, 54321);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights initialization should succeed");
    
    error = mps_lstm_compressor_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Loading weights should succeed");
    
    // Prepare test data
    uint32_t data_size = config.batch_size * config.sequence_length * config.input_size * sizeof(float);
    
    MPSLSTMCompressorData* data = NULL;
    error = mps_lstm_compressor_data_create(&data, data_size, MPS_LSTM_DATA_TYPE_FLOAT32);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Data creation should succeed");
    
    data->input_data = malloc(data_size);
    data->compressed_data = malloc(data_size);
    data->decompressed_data = malloc(data_size);
    data->owns_memory = true;
    data->batch_size = config.batch_size;
    
    // Generate test time series
    float* input = (float*)data->input_data;
    generate_test_time_series(input, data_size / sizeof(float), 1.0f, 1.5f);
    
    // Perform compression
    error = mps_lstm_compressor_compress(context, data);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Predictive compression should succeed");
    
    // Check compression results
    TEST_ASSERT(data->compressed_size > 0, "Compressed size should be positive");
    TEST_ASSERT(data->metadata.compression_ratio > 0.0f, "Compression ratio should be positive");
    TEST_ASSERT(data->metadata.original_size == data_size, "Original size should match input");
    
    printf("  Predictive compression completed successfully\n");
    printf("  Original size: %u bytes, Compressed size: %u bytes\n", 
           data->metadata.original_size, data->metadata.compressed_size);
    printf("  Compression ratio: %.2f:1\n", data->metadata.compression_ratio);
    printf("  Compression time: %.3f ms\n", data->metadata.compression_time_ns / 1000000.0f);
    
    // Perform decompression
    error = mps_lstm_compressor_decompress(context, data);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Decompression should succeed");
    
    // Check decompression results
    TEST_ASSERT(data->decompressed_size > 0, "Decompressed size should be positive");
    TEST_ASSERT(data->metadata.mse >= 0.0f, "MSE should be non-negative");
    
    printf("  Decompression completed: size=%u, MSE=%.6f, PSNR=%.2f dB\n",
           data->decompressed_size, data->metadata.mse, data->metadata.psnr);
    
    // Cleanup
    mps_lstm_compressor_data_destroy(data);
    mps_lstm_compressor_weights_destroy(weights);
    mps_lstm_compressor_destroy(context);
    
    TEST_PASS();
}

int test_temporal_compression_basic() {
    if (!mps_lstm_compressor_is_available()) {
        printf("  MPS not available, skipping temporal compression test\n");
        TEST_PASS();
    }
    
    MPSLSTMCompressorConfig config;
    mps_lstm_compressor_config_create_default(&config, MPS_LSTM_COMPRESSION_TEMPORAL, 8, 64, MPS_LSTM_QUALITY_HIGH);
    config.batch_size = 2;
    
    MPSLSTMCompressorContext* context = NULL;
    MPSLSTMCompressorError error = mps_lstm_compressor_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSLSTMCompressorWeights* weights = NULL;
    error = mps_lstm_compressor_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_compressor_weights_init_random(weights, &config, 98765);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights initialization should succeed");
    
    error = mps_lstm_compressor_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Loading weights should succeed");
    
    // Prepare test data with temporal patterns
    uint32_t data_size = config.batch_size * config.sequence_length * config.input_size * sizeof(float);
    
    float* time_series = malloc(data_size);
    void* compressed_output = malloc(data_size);
    
    TEST_ASSERT(time_series != NULL && compressed_output != NULL, "Memory allocation should succeed");
    
    // Generate temporal pattern data
    generate_test_sequences((float*)time_series, data_size / sizeof(float), config.sequence_length);
    
    // Perform temporal compression
    MPSLSTMCompressionMetadata metadata;
    error = mps_lstm_compressor_compress_time_series(context, time_series, compressed_output, &metadata);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Temporal compression should succeed");
    
    // Check compression results
    TEST_ASSERT(metadata.compressed_size > 0, "Compressed size should be positive");
    TEST_ASSERT(metadata.compression_ratio > 1.0f, "Should achieve compression");
    
    printf("  Temporal compression completed successfully\n");
    printf("  Original size: %u bytes, Compressed size: %u bytes\n", 
           metadata.original_size, metadata.compressed_size);
    printf("  Compression ratio: %.2f:1\n", metadata.compression_ratio);
    printf("  Compression mode: %d\n", metadata.compression_mode);
    
    // Cleanup
    free(time_series);
    free(compressed_output);
    mps_lstm_compressor_weights_destroy(weights);
    mps_lstm_compressor_destroy(context);
    
    TEST_PASS();
}

int test_residual_compression_basic() {
    if (!mps_lstm_compressor_is_available()) {
        printf("  MPS not available, skipping residual compression test\n");
        TEST_PASS();
    }
    
    MPSLSTMCompressorConfig config;
    mps_lstm_compressor_config_create_default(&config, MPS_LSTM_COMPRESSION_RESIDUAL, 6, 48, MPS_LSTM_QUALITY_LOW);
    config.batch_size = 1;
    
    MPSLSTMCompressorContext* context = NULL;
    MPSLSTMCompressorError error = mps_lstm_compressor_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSLSTMCompressorWeights* weights = NULL;
    error = mps_lstm_compressor_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_compressor_weights_init_random(weights, &config, 11111);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights initialization should succeed");
    
    error = mps_lstm_compressor_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Loading weights should succeed");
    
    // Prepare test data
    uint32_t data_size = config.batch_size * config.sequence_length * config.input_size * sizeof(float);
    
    MPSLSTMCompressorData* data = NULL;
    error = mps_lstm_compressor_data_create(&data, data_size, MPS_LSTM_DATA_TYPE_FLOAT32);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Data creation should succeed");
    
    data->input_data = malloc(data_size);
    data->compressed_data = malloc(data_size);
    data->decompressed_data = malloc(data_size);
    data->owns_memory = true;
    data->batch_size = config.batch_size;
    
    // Generate slowly varying test data (good for residual compression)
    float* input = (float*)data->input_data;
    for (size_t i = 0; i < data_size / sizeof(float); i++) {
        input[i] = sinf((float)i * 0.01f) + 0.1f * (float)i / (data_size / sizeof(float));
    }
    
    // Perform compression
    error = mps_lstm_compressor_compress(context, data);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Residual compression should succeed");
    
    // Check compression results
    TEST_ASSERT(data->compressed_size > 0, "Compressed size should be positive");
    TEST_ASSERT(data->metadata.compression_mode == MPS_LSTM_COMPRESSION_RESIDUAL, 
               "Compression mode should be residual");
    
    printf("  Residual compression completed successfully\n");
    printf("  Original size: %u bytes, Compressed size: %u bytes\n", 
           data->metadata.original_size, data->metadata.compressed_size);
    printf("  Compression ratio: %.2f:1\n", data->metadata.compression_ratio);
    printf("  Target compression ratio: %.1f:1\n", config.compression_ratio_target);
    
    // Perform decompression and quality check
    error = mps_lstm_compressor_decompress(context, data);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Decompression should succeed");
    
    printf("  Decompression quality: MSE=%.6f, PSNR=%.2f dB\n",
           data->metadata.mse, data->metadata.psnr);
    
    // Cleanup
    mps_lstm_compressor_data_destroy(data);
    mps_lstm_compressor_weights_destroy(weights);
    mps_lstm_compressor_destroy(context);
    
    TEST_PASS();
}

int test_performance_statistics() {
    if (!mps_lstm_compressor_is_available()) {
        printf("  MPS not available, skipping performance test\n");
        TEST_PASS();
    }
    
    MPSLSTMCompressorConfig config;
    mps_lstm_compressor_config_create_default(&config, MPS_LSTM_COMPRESSION_PREDICTIVE, 4, 32, MPS_LSTM_QUALITY_MEDIUM);
    config.batch_size = 2;
    
    MPSLSTMCompressorContext* context = NULL;
    MPSLSTMCompressorError error = mps_lstm_compressor_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Context creation should succeed");
    
    // Get initial statistics
    MPSLSTMCompressorStats initial_stats;
    error = mps_lstm_compressor_get_stats(context, &initial_stats);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Getting stats should succeed");
    TEST_ASSERT(initial_stats.total_compressions == 0, "Initial compressions should be 0");
    TEST_ASSERT(initial_stats.total_compression_time_ns == 0, "Initial compression time should be 0");
    
    // Create weights and perform operations
    MPSLSTMCompressorWeights* weights = NULL;
    error = mps_lstm_compressor_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_compressor_weights_init_random(weights, &config, 22222);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights initialization should succeed");
    
    error = mps_lstm_compressor_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Loading weights should succeed");
    
    // Prepare test data
    uint32_t data_size = config.batch_size * config.sequence_length * config.input_size * sizeof(float);
    
    MPSLSTMCompressorData* data = NULL;
    error = mps_lstm_compressor_data_create(&data, data_size, MPS_LSTM_DATA_TYPE_FLOAT32);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Data creation should succeed");
    
    data->input_data = malloc(data_size);
    data->compressed_data = malloc(data_size);
    data->decompressed_data = malloc(data_size);
    data->owns_memory = true;
    data->batch_size = config.batch_size;
    
    generate_test_time_series((float*)data->input_data, data_size / sizeof(float), 0.8f, 1.2f);
    
    // Perform multiple operations
    for (int i = 0; i < 3; i++) {
        error = mps_lstm_compressor_compress(context, data);
        TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Compression should succeed");
        
        error = mps_lstm_compressor_decompress(context, data);
        TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Decompression should succeed");
    }
    
    // Get updated statistics
    MPSLSTMCompressorStats final_stats;
    error = mps_lstm_compressor_get_stats(context, &final_stats);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Getting final stats should succeed");
    TEST_ASSERT(final_stats.total_compressions == 3, "Should have 3 compressions");
    TEST_ASSERT(final_stats.total_decompressions == 3, "Should have 3 decompressions");
    TEST_ASSERT(final_stats.total_compression_time_ns > 0, "Should have non-zero compression time");
    TEST_ASSERT(final_stats.average_compression_ratio > 0.0f, "Average compression ratio should be positive");
    
    printf("  Performance stats after 3 compression/decompression cycles:\n");
    printf("    Total compressions: %llu\n", (unsigned long long)final_stats.total_compressions);
    printf("    Total decompressions: %llu\n", (unsigned long long)final_stats.total_decompressions);
    printf("    Bytes compressed: %llu\n", (unsigned long long)final_stats.total_bytes_compressed);
    printf("    Bytes decompressed: %llu\n", (unsigned long long)final_stats.total_bytes_decompressed);
    printf("    Average compression ratio: %.2f:1\n", final_stats.average_compression_ratio);
    printf("    Average compression time: %.3f ms\n", final_stats.average_compression_time_ms);
    printf("    Average decompression time: %.3f ms\n", final_stats.average_decompression_time_ms);
    printf("    Compression throughput: %.2f MB/s\n", final_stats.throughput_mbps_compression);
    printf("    Decompression throughput: %.2f MB/s\n", final_stats.throughput_mbps_decompression);
    printf("    Memory usage: %u MB\n", final_stats.memory_usage_mb);
    printf("    Peak memory: %u MB\n", final_stats.peak_memory_usage_mb);
    
    // Test statistics reset
    error = mps_lstm_compressor_reset_stats(context);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Stats reset should succeed");
    
    MPSLSTMCompressorStats reset_stats;
    error = mps_lstm_compressor_get_stats(context, &reset_stats);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Getting reset stats should succeed");
    TEST_ASSERT(reset_stats.total_compressions == 0, "Reset compressions should be 0");
    TEST_ASSERT(reset_stats.total_compression_time_ns == 0, "Reset compression time should be 0");
    
    printf("  Statistics reset successfully\n");
    
    // Cleanup
    mps_lstm_compressor_data_destroy(data);
    mps_lstm_compressor_weights_destroy(weights);
    mps_lstm_compressor_destroy(context);
    
    TEST_PASS();
}

int test_different_compression_modes() {
    if (!mps_lstm_compressor_is_available()) {
        printf("  MPS not available, skipping compression modes test\n");
        TEST_PASS();
    }
    
    // Test configurations for different compression modes
    MPSLSTMCompressionMode compression_modes[] = {
        MPS_LSTM_COMPRESSION_PREDICTIVE,
        MPS_LSTM_COMPRESSION_TEMPORAL,
        MPS_LSTM_COMPRESSION_RESIDUAL
    };
    
    const char* mode_names[] = {
        "Predictive",
        "Temporal",
        "Residual"
    };
    
    for (int m = 0; m < 3; m++) {
        MPSLSTMCompressorConfig config;
        MPSLSTMCompressorError error = mps_lstm_compressor_config_create_default(&config,
                                                                               compression_modes[m],
                                                                               8, 32,
                                                                               MPS_LSTM_QUALITY_MEDIUM);
        TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Config creation should succeed for all modes");
        
        error = mps_lstm_compressor_config_validate(&config);
        TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Config validation should succeed for all modes");
        
        printf("  %s compression mode validated\n", mode_names[m]);
        
        // Quick context creation test
        MPSLSTMCompressorContext* context = NULL;
        error = mps_lstm_compressor_create(&context, &config);
        if (error == MPS_LSTM_COMPRESSOR_SUCCESS) {
            printf("    %s context created successfully\n", mode_names[m]);
            mps_lstm_compressor_destroy(context);
        }
    }
    
    TEST_PASS();
}

int test_quality_levels() {
    if (!mps_lstm_compressor_is_available()) {
        printf("  MPS not available, skipping quality levels test\n");
        TEST_PASS();
    }
    
    // Test configurations for different quality levels
    MPSLSTMQualityLevel quality_levels[] = {
        MPS_LSTM_QUALITY_LOSSLESS,
        MPS_LSTM_QUALITY_HIGH,
        MPS_LSTM_QUALITY_MEDIUM,
        MPS_LSTM_QUALITY_LOW
    };
    
    const char* quality_names[] = {
        "Lossless",
        "High",
        "Medium",
        "Low"
    };
    
    for (int q = 0; q < 4; q++) {
        MPSLSTMCompressorConfig config;
        MPSLSTMCompressorError error = mps_lstm_compressor_config_create_default(&config,
                                                                               MPS_LSTM_COMPRESSION_PREDICTIVE,
                                                                               8, 32,
                                                                               quality_levels[q]);
        TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Config creation should succeed for all quality levels");
        
        printf("  %s quality: ratio=%.1f:1, data_type=%d, residual_bits=%u\n",
               quality_names[q], config.compression_ratio_target, 
               config.compressed_data_type, config.residual_bits);
    }
    
    TEST_PASS();
}

int test_memory_management() {
    if (!mps_lstm_compressor_is_available()) {
        printf("  MPS not available, skipping memory management test\n");
        TEST_PASS();
    }
    
    MPSLSTMCompressorConfig config;
    mps_lstm_compressor_config_create_default(&config, MPS_LSTM_COMPRESSION_TEMPORAL, 16, 64, MPS_LSTM_QUALITY_HIGH);
    config.batch_size = 4;
    
    MPSLSTMCompressorContext* context = NULL;
    MPSLSTMCompressorError error = mps_lstm_compressor_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Context creation should succeed");
    
    // Check initial memory usage
    uint32_t initial_memory;
    error = mps_lstm_compressor_get_memory_usage(context, &initial_memory);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Getting memory usage should succeed");
    printf("  Initial memory usage: %u MB\n", initial_memory);
    
    // Create weights and check memory increase
    MPSLSTMCompressorWeights* weights = NULL;
    error = mps_lstm_compressor_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_compressor_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Loading weights should succeed");
    
    uint32_t after_weights_memory;
    error = mps_lstm_compressor_get_memory_usage(context, &after_weights_memory);
    TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Getting memory usage should succeed");
    printf("  Memory usage after loading weights: %u MB\n", after_weights_memory);
    
    // Perform operations and check peak memory
    uint32_t data_size = config.batch_size * config.sequence_length * config.input_size * sizeof(float);
    float* time_series = malloc(data_size);
    void* compressed_output = malloc(data_size);
    
    if (time_series && compressed_output) {
        generate_test_time_series(time_series, data_size / sizeof(float), 1.0f, 2.0f);
        
        MPSLSTMCompressionMetadata metadata;
        error = mps_lstm_compressor_compress_time_series(context, time_series, compressed_output, &metadata);
        TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Compression should succeed");
        
        MPSLSTMCompressorStats stats;
        error = mps_lstm_compressor_get_stats(context, &stats);
        TEST_ASSERT(error == MPS_LSTM_COMPRESSOR_SUCCESS, "Getting stats should succeed");
        
        printf("  Peak memory usage: %u MB\n", stats.peak_memory_usage_mb);
        TEST_ASSERT(stats.peak_memory_usage_mb >= after_weights_memory,
                   "Peak memory should be at least as much as after loading weights");
    }
    
    // Cleanup
    free(time_series);
    free(compressed_output);
    mps_lstm_compressor_weights_destroy(weights);
    mps_lstm_compressor_destroy(context);
    
    TEST_PASS();
}

// Test runner
int run_test(const char* test_name, int (*test_func)()) {
    printf("Running %s...\n", test_name);
    int result = test_func();
    if (result) {
        printf("✓ %s passed\n\n", test_name);
    } else {
        printf("✗ %s failed\n\n", test_name);
    }
    return result;
}

int main() {
    printf("MPS LSTM Compression Engine Test Suite\n");
    printf("======================================\n\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    total_tests++; passed_tests += run_test("Compressor Availability", test_compressor_availability);
    total_tests++; passed_tests += run_test("Config Creation and Validation", test_config_creation_and_validation);
    total_tests++; passed_tests += run_test("Memory Requirements Calculation", test_memory_requirements_calculation);
    total_tests++; passed_tests += run_test("Error String Functions", test_error_string_functions);
    total_tests++; passed_tests += run_test("Context Creation and Destruction", test_context_creation_and_destruction);
    total_tests++; passed_tests += run_test("Weights Creation and Initialization", test_weights_creation_and_initialization);
    total_tests++; passed_tests += run_test("Data Creation and Validation", test_data_creation_and_validation);
    total_tests++; passed_tests += run_test("Predictive Compression Basic", test_predictive_compression_basic);
    total_tests++; passed_tests += run_test("Temporal Compression Basic", test_temporal_compression_basic);
    total_tests++; passed_tests += run_test("Residual Compression Basic", test_residual_compression_basic);
    total_tests++; passed_tests += run_test("Performance Statistics", test_performance_statistics);
    total_tests++; passed_tests += run_test("Different Compression Modes", test_different_compression_modes);
    total_tests++; passed_tests += run_test("Quality Levels", test_quality_levels);
    total_tests++; passed_tests += run_test("Memory Management", test_memory_management);
    
    // Summary
    printf("Test Results\n");
    printf("============\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! MPS LSTM compression engine is ready.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the implementation.\n");
        return 1;
    }
}
