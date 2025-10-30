#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "../../src/neural/ops/mps_attention.h"

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

#define FLOAT_TOLERANCE 1e-5f

// Helper function to check if two floats are approximately equal
static bool floats_equal(float a, float b, float tolerance) {
    return fabsf(a - b) < tolerance;
}

// Helper function to initialize test data
static void initialize_test_data(float* data, size_t size, float value) {
    for (size_t i = 0; i < size; i++) {
        data[i] = value + (float)(i % 100) * 0.01f;
    }
}

// Test functions

int test_mps_availability() {
    bool available = mps_attention_is_available();
    printf("  MPS Attention available: %s\n", available ? "Yes" : "No");
    
    if (available) {
        char device_name[256];
        uint32_t compute_units, max_memory_mb;
        MPSAttentionError error = mps_attention_get_device_info(device_name, sizeof(device_name),
                                                              &compute_units, &max_memory_mb);
        TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Getting device info should succeed");
        
        printf("  Device: %s\n", device_name);
        printf("  Compute units: %u\n", compute_units);
        printf("  Max memory: %u MB\n", max_memory_mb);
    }
    
    TEST_PASS();
}

int test_config_creation_and_validation() {
    MPSAttentionConfig config;
    MPSAttentionError error = mps_attention_config_create_default(&config, 512, 768, 12);
    
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Default config creation should succeed");
    TEST_ASSERT(config.sequence_length == 512, "Sequence length should be 512");
    TEST_ASSERT(config.hidden_size == 768, "Hidden size should be 768");
    TEST_ASSERT(config.num_heads == 12, "Number of heads should be 12");
    TEST_ASSERT(config.head_dim == 64, "Head dimension should be 64");
    TEST_ASSERT(config.batch_size == 1, "Default batch size should be 1");
    TEST_ASSERT(config.use_bias == true, "Use bias should be enabled by default");
    TEST_ASSERT(config.attention_type == MPS_ATTENTION_MULTI_HEAD, "Should be multi-head attention");
    
    printf("  Default config: seq_len=%u, hidden=%u, heads=%u, head_dim=%u\n",
           config.sequence_length, config.hidden_size, config.num_heads, config.head_dim);
    
    // Test configuration validation
    error = mps_attention_config_validate(&config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Config validation should succeed");
    
    // Test invalid configurations
    MPSAttentionConfig invalid_config = config;
    invalid_config.hidden_size = 777; // Not divisible by num_heads
    error = mps_attention_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_ATTENTION_ERROR_INVALID_DIMENSIONS, "Invalid config should fail validation");
    
    invalid_config = config;
    invalid_config.sequence_length = 0;
    error = mps_attention_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_ATTENTION_ERROR_INVALID_DIMENSIONS, "Zero sequence length should fail");
    
    printf("  Config validation tests passed\n");
    
    TEST_PASS();
}

int test_memory_requirements_calculation() {
    MPSAttentionConfig config;
    mps_attention_config_create_default(&config, 256, 512, 8);
    config.batch_size = 4;
    
    uint32_t memory_mb;
    MPSAttentionError error = mps_attention_calculate_memory_requirements(&config, &memory_mb);
    
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Memory calculation should succeed");
    TEST_ASSERT(memory_mb > 0, "Memory requirement should be positive");
    
    printf("  Memory requirements for config: %u MB\n", memory_mb);
    
    // Test larger configuration
    config.sequence_length = 1024;
    config.hidden_size = 1024;
    config.num_heads = 16;
    config.head_dim = 64;
    config.batch_size = 8;
    
    uint32_t larger_memory_mb;
    error = mps_attention_calculate_memory_requirements(&config, &larger_memory_mb);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Larger config memory calculation should succeed");
    TEST_ASSERT(larger_memory_mb > memory_mb, "Larger config should require more memory");
    
    printf("  Memory requirements for larger config: %u MB\n", larger_memory_mb);
    
    TEST_PASS();
}

int test_error_string_functions() {
    const char* success_msg = mps_attention_get_error_string(MPS_ATTENTION_SUCCESS);
    TEST_ASSERT(success_msg != NULL, "Success error string should not be NULL");
    TEST_ASSERT(strlen(success_msg) > 0, "Success error string should not be empty");
    
    const char* invalid_param_msg = mps_attention_get_error_string(MPS_ATTENTION_ERROR_INVALID_PARAM);
    TEST_ASSERT(invalid_param_msg != NULL, "Invalid param error string should not be NULL");
    TEST_ASSERT(strlen(invalid_param_msg) > 0, "Invalid param error string should not be empty");
    
    printf("  MPS_ATTENTION_SUCCESS: %s\n", success_msg);
    printf("  MPS_ATTENTION_ERROR_INVALID_PARAM: %s\n", invalid_param_msg);
    
    // Test unknown error code
    const char* unknown_msg = mps_attention_get_error_string((MPSAttentionError)999);
    TEST_ASSERT(unknown_msg != NULL, "Unknown error should return a message");
    
    TEST_PASS();
}

int test_context_creation_and_destruction() {
    MPSAttentionConfig config;
    mps_attention_config_create_default(&config, 128, 256, 4);
    config.batch_size = 2;
    
    MPSAttentionContext* context = NULL;
    MPSAttentionError error = mps_attention_create(&context, &config);
    
    if (!mps_attention_is_available()) {
        printf("  MPS not available, skipping context creation test\n");
        TEST_PASS();
    }
    
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Context creation should succeed");
    TEST_ASSERT(context != NULL, "Context should not be NULL");
    TEST_ASSERT(context->is_initialized == true, "Context should be initialized");
    TEST_ASSERT(context->config.sequence_length == 128, "Config should be copied correctly");
    TEST_ASSERT(context->config.hidden_size == 256, "Config should be copied correctly");
    TEST_ASSERT(context->config.num_heads == 4, "Config should be copied correctly");
    
    printf("  Created context with seq_len=%u, hidden=%u, heads=%u\n",
           context->config.sequence_length, context->config.hidden_size, context->config.num_heads);
    
    // Test memory usage
    uint32_t memory_usage;
    error = mps_attention_get_memory_usage(context, &memory_usage);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Getting memory usage should succeed");
    printf("  Context memory usage: %u MB\n", memory_usage);
    
    mps_attention_destroy(context);
    TEST_PASS();
}

int test_weights_creation_and_initialization() {
    MPSAttentionConfig config;
    mps_attention_config_create_default(&config, 64, 128, 2);
    
    MPSAttentionWeights* weights = NULL;
    MPSAttentionError error = mps_attention_weights_create(&weights, &config);
    
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Weights creation should succeed");
    TEST_ASSERT(weights != NULL, "Weights should not be NULL");
    TEST_ASSERT(weights->query_weights != NULL, "Query weights should be allocated");
    TEST_ASSERT(weights->key_weights != NULL, "Key weights should be allocated");
    TEST_ASSERT(weights->value_weights != NULL, "Value weights should be allocated");
    TEST_ASSERT(weights->output_weights != NULL, "Output weights should be allocated");
    
    if (config.use_bias) {
        TEST_ASSERT(weights->query_bias != NULL, "Query bias should be allocated");
        TEST_ASSERT(weights->key_bias != NULL, "Key bias should be allocated");
        TEST_ASSERT(weights->value_bias != NULL, "Value bias should be allocated");
        TEST_ASSERT(weights->output_bias != NULL, "Output bias should be allocated");
    }
    
    printf("  Weights created: weights_size=%zu, bias_size=%zu\n",
           weights->weights_size, weights->bias_size);
    
    // Test random initialization
    error = mps_attention_weights_init_random(weights, &config, 12345);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Random weight initialization should succeed");
    
    // Check that weights are not all zero
    float* query_weights = (float*)weights->query_weights;
    bool has_nonzero = false;
    for (size_t i = 0; i < config.hidden_size * config.hidden_size && !has_nonzero; i++) {
        if (fabsf(query_weights[i]) > 1e-8f) {
            has_nonzero = true;
        }
    }
    TEST_ASSERT(has_nonzero, "Weights should not all be zero after initialization");
    printf("  Random initialization successful\n");
    
    mps_attention_weights_destroy(weights);
    TEST_PASS();
}

int test_tensors_creation_and_validation() {
    MPSAttentionShape shape = {
        .batch_size = 2,
        .sequence_length = 64,
        .hidden_size = 128,
        .num_heads = 4,
        .head_dim = 32
    };
    
    MPSAttentionTensors* tensors = NULL;
    MPSAttentionError error = mps_attention_tensors_create(&tensors, &shape);
    
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Tensors creation should succeed");
    TEST_ASSERT(tensors != NULL, "Tensors should not be NULL");
    TEST_ASSERT(tensors->shape.batch_size == 2, "Batch size should be copied");
    TEST_ASSERT(tensors->shape.sequence_length == 64, "Sequence length should be copied");
    TEST_ASSERT(tensors->shape.hidden_size == 128, "Hidden size should be copied");
    TEST_ASSERT(tensors->shape.num_heads == 4, "Number of heads should be copied");
    TEST_ASSERT(tensors->shape.head_dim == 32, "Head dimension should be copied");
    
    printf("  Tensors created with shape: [%u, %u, %u], heads=%u, head_dim=%u\n",
           shape.batch_size, shape.sequence_length, shape.hidden_size, 
           shape.num_heads, shape.head_dim);
    
    // Allocate test data
    size_t tensor_size = shape.batch_size * shape.sequence_length * shape.hidden_size;
    tensors->query = malloc(tensor_size * sizeof(float));
    tensors->key = malloc(tensor_size * sizeof(float));
    tensors->value = malloc(tensor_size * sizeof(float));
    tensors->output = malloc(tensor_size * sizeof(float));
    tensors->owns_memory = true;
    
    TEST_ASSERT(tensors->query != NULL, "Query tensor should be allocated");
    TEST_ASSERT(tensors->key != NULL, "Key tensor should be allocated");
    TEST_ASSERT(tensors->value != NULL, "Value tensor should be allocated");
    TEST_ASSERT(tensors->output != NULL, "Output tensor should be allocated");
    
    // Initialize test data
    initialize_test_data(tensors->query, tensor_size, 0.1f);
    initialize_test_data(tensors->key, tensor_size, 0.2f);
    initialize_test_data(tensors->value, tensor_size, 0.3f);
    memset(tensors->output, 0, tensor_size * sizeof(float));
    
    // Test tensor validation
    MPSAttentionConfig config;
    mps_attention_config_create_default(&config, shape.sequence_length, 
                                       shape.hidden_size, shape.num_heads);
    config.batch_size = shape.batch_size;
    
    error = mps_attention_tensors_validate(tensors, &config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Tensor validation should succeed");
    
    printf("  Tensor validation passed\n");
    
    mps_attention_tensors_destroy(tensors);
    TEST_PASS();
}

int test_self_attention_basic() {
    if (!mps_attention_is_available()) {
        printf("  MPS not available, skipping self-attention test\n");
        TEST_PASS();
    }
    
    MPSAttentionConfig config;
    mps_attention_config_create_default(&config, 32, 64, 2);
    config.batch_size = 1;
    
    MPSAttentionContext* context = NULL;
    MPSAttentionError error = mps_attention_create(&context, &config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSAttentionWeights* weights = NULL;
    error = mps_attention_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Weights creation should succeed");
    
    error = mps_attention_weights_init_random(weights, &config, 54321);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Weights initialization should succeed");
    
    error = mps_attention_load_weights(context, weights);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Loading weights should succeed");
    
    // Prepare input and output
    size_t tensor_size = config.batch_size * config.sequence_length * config.hidden_size;
    float* input = malloc(tensor_size * sizeof(float));
    float* output = malloc(tensor_size * sizeof(float));
    
    TEST_ASSERT(input != NULL && output != NULL, "Memory allocation should succeed");
    
    // Initialize input with test pattern
    initialize_test_data(input, tensor_size, 0.5f);
    memset(output, 0, tensor_size * sizeof(float));
    
    // Perform self-attention
    error = mps_attention_self_attention(context, input, output, NULL);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Self-attention should succeed");
    
    // Check that output is different from input (attention was applied)
    bool output_changed = false;
    for (size_t i = 0; i < tensor_size && !output_changed; i++) {
        if (!floats_equal(input[i], output[i], FLOAT_TOLERANCE)) {
            output_changed = true;
        }
    }
    TEST_ASSERT(output_changed, "Output should be different from input");
    
    // Check that output is not all zeros
    bool has_nonzero_output = false;
    for (size_t i = 0; i < tensor_size && !has_nonzero_output; i++) {
        if (fabsf(output[i]) > FLOAT_TOLERANCE) {
            has_nonzero_output = true;
        }
    }
    TEST_ASSERT(has_nonzero_output, "Output should not be all zeros");
    
    printf("  Self-attention computation completed successfully\n");
    printf("  Input sample: [%.3f, %.3f, %.3f, ...]\n", input[0], input[1], input[2]);
    printf("  Output sample: [%.3f, %.3f, %.3f, ...]\n", output[0], output[1], output[2]);
    
    // Cleanup
    free(input);
    free(output);
    mps_attention_weights_destroy(weights);
    mps_attention_destroy(context);
    
    TEST_PASS();
}

int test_cross_attention_basic() {
    if (!mps_attention_is_available()) {
        printf("  MPS not available, skipping cross-attention test\n");
        TEST_PASS();
    }
    
    MPSAttentionConfig config;
    mps_attention_config_create_default(&config, 16, 32, 2);
    config.batch_size = 1;
    
    MPSAttentionContext* context = NULL;
    MPSAttentionError error = mps_attention_create(&context, &config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSAttentionWeights* weights = NULL;
    error = mps_attention_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Weights creation should succeed");
    
    error = mps_attention_weights_init_random(weights, &config, 98765);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Weights initialization should succeed");
    
    error = mps_attention_load_weights(context, weights);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Loading weights should succeed");
    
    // Prepare separate query, key, value, and output tensors
    size_t tensor_size = config.batch_size * config.sequence_length * config.hidden_size;
    float* query = malloc(tensor_size * sizeof(float));
    float* key = malloc(tensor_size * sizeof(float));
    float* value = malloc(tensor_size * sizeof(float));
    float* output = malloc(tensor_size * sizeof(float));
    
    TEST_ASSERT(query != NULL && key != NULL && value != NULL && output != NULL, 
               "Memory allocation should succeed");
    
    // Initialize with different patterns
    initialize_test_data(query, tensor_size, 0.1f);
    initialize_test_data(key, tensor_size, 0.2f);
    initialize_test_data(value, tensor_size, 0.3f);
    memset(output, 0, tensor_size * sizeof(float));
    
    // Perform cross-attention
    error = mps_attention_cross_attention(context, query, key, value, output, NULL);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Cross-attention should succeed");
    
    // Verify output is computed
    bool has_nonzero_output = false;
    for (size_t i = 0; i < tensor_size && !has_nonzero_output; i++) {
        if (fabsf(output[i]) > FLOAT_TOLERANCE) {
            has_nonzero_output = true;
        }
    }
    TEST_ASSERT(has_nonzero_output, "Output should not be all zeros");
    
    printf("  Cross-attention computation completed successfully\n");
    printf("  Query sample: [%.3f, %.3f, %.3f, ...]\n", query[0], query[1], query[2]);
    printf("  Key sample: [%.3f, %.3f, %.3f, ...]\n", key[0], key[1], key[2]);
    printf("  Value sample: [%.3f, %.3f, %.3f, ...]\n", value[0], value[1], value[2]);
    printf("  Output sample: [%.3f, %.3f, %.3f, ...]\n", output[0], output[1], output[2]);
    
    // Cleanup
    free(query);
    free(key);
    free(value);
    free(output);
    mps_attention_weights_destroy(weights);
    mps_attention_destroy(context);
    
    TEST_PASS();
}

int test_performance_statistics() {
    if (!mps_attention_is_available()) {
        printf("  MPS not available, skipping performance test\n");
        TEST_PASS();
    }
    
    MPSAttentionConfig config;
    mps_attention_config_create_default(&config, 64, 128, 4);
    config.batch_size = 2;
    
    MPSAttentionContext* context = NULL;
    MPSAttentionError error = mps_attention_create(&context, &config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Context creation should succeed");
    
    // Get initial statistics
    MPSAttentionStats initial_stats;
    error = mps_attention_get_stats(context, &initial_stats);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Getting stats should succeed");
    TEST_ASSERT(initial_stats.total_operations == 0, "Initial operations should be 0");
    TEST_ASSERT(initial_stats.total_compute_time_ns == 0, "Initial compute time should be 0");
    
    // Create weights and perform some operations
    MPSAttentionWeights* weights = NULL;
    error = mps_attention_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Weights creation should succeed");
    
    error = mps_attention_weights_init_random(weights, &config, 11111);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Weights initialization should succeed");
    
    error = mps_attention_load_weights(context, weights);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Loading weights should succeed");
    
    // Prepare test data
    size_t tensor_size = config.batch_size * config.sequence_length * config.hidden_size;
    float* input = malloc(tensor_size * sizeof(float));
    float* output = malloc(tensor_size * sizeof(float));
    
    initialize_test_data(input, tensor_size, 0.25f);
    
    // Perform multiple operations
    for (int i = 0; i < 3; i++) {
        error = mps_attention_self_attention(context, input, output, NULL);
        TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Self-attention should succeed");
    }
    
    // Get updated statistics
    MPSAttentionStats final_stats;
    error = mps_attention_get_stats(context, &final_stats);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Getting final stats should succeed");
    TEST_ASSERT(final_stats.total_operations == 3, "Should have 3 operations");
    TEST_ASSERT(final_stats.total_compute_time_ns > 0, "Should have non-zero compute time");
    TEST_ASSERT(final_stats.average_compute_time_ms >= 0.0f, "Average compute time should be non-negative");
    
    printf("  Performance stats after 3 operations:\n");
    printf("    Total operations: %llu\n", (unsigned long long)final_stats.total_operations);
    printf("    Total compute time: %llu ns\n", final_stats.total_compute_time_ns);
    printf("    Average compute time: %.3f ms\n", final_stats.average_compute_time_ms);
    printf("    GFLOPS achieved: %.3f\n", final_stats.gflops_achieved);
    printf("    Memory usage: %u MB\n", final_stats.memory_usage_mb);
    printf("    Peak memory: %u MB\n", final_stats.peak_memory_usage_mb);
    
    // Test statistics reset
    error = mps_attention_reset_stats(context);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Stats reset should succeed");
    
    MPSAttentionStats reset_stats;
    error = mps_attention_get_stats(context, &reset_stats);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Getting reset stats should succeed");
    TEST_ASSERT(reset_stats.total_operations == 0, "Reset operations should be 0");
    TEST_ASSERT(reset_stats.total_compute_time_ns == 0, "Reset compute time should be 0");
    
    printf("  Statistics reset successfully\n");
    
    // Cleanup
    free(input);
    free(output);
    mps_attention_weights_destroy(weights);
    mps_attention_destroy(context);
    
    TEST_PASS();
}

int test_different_attention_types() {
    if (!mps_attention_is_available()) {
        printf("  MPS not available, skipping attention types test\n");
        TEST_PASS();
    }
    
    // Test configurations for different attention types
    MPSAttentionType attention_types[] = {
        MPS_ATTENTION_SCALED_DOT_PRODUCT,
        MPS_ATTENTION_MULTI_HEAD,
        MPS_ATTENTION_SELF_ATTENTION,
        MPS_ATTENTION_CROSS_ATTENTION
    };
    
    const char* type_names[] = {
        "Scaled Dot-Product",
        "Multi-Head",
        "Self-Attention",
        "Cross-Attention"
    };
    
    for (int t = 0; t < 4; t++) {
        MPSAttentionConfig config;
        mps_attention_config_create_default(&config, 32, 64, 2);
        config.attention_type = attention_types[t];
        
        MPSAttentionError error = mps_attention_config_validate(&config);
        TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Config validation should succeed for all types");
        
        printf("  %s attention type validated\n", type_names[t]);
    }
    
    TEST_PASS();
}

int test_memory_management() {
    if (!mps_attention_is_available()) {
        printf("  MPS not available, skipping memory management test\n");
        TEST_PASS();
    }
    
    MPSAttentionConfig config;
    mps_attention_config_create_default(&config, 128, 256, 8);
    config.batch_size = 4;
    
    MPSAttentionContext* context = NULL;
    MPSAttentionError error = mps_attention_create(&context, &config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Context creation should succeed");
    
    // Check initial memory usage
    uint32_t initial_memory;
    error = mps_attention_get_memory_usage(context, &initial_memory);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Getting memory usage should succeed");
    printf("  Initial memory usage: %u MB\n", initial_memory);
    
    // Create weights and check memory increase
    MPSAttentionWeights* weights = NULL;
    error = mps_attention_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Weights creation should succeed");
    
    error = mps_attention_load_weights(context, weights);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Loading weights should succeed");
    
    uint32_t after_weights_memory;
    error = mps_attention_get_memory_usage(context, &after_weights_memory);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Getting memory usage should succeed");
    printf("  Memory usage after loading weights: %u MB\n", after_weights_memory);
    
    // Perform operations and check peak memory
    size_t tensor_size = config.batch_size * config.sequence_length * config.hidden_size;
    float* input = malloc(tensor_size * sizeof(float));
    float* output = malloc(tensor_size * sizeof(float));
    
    initialize_test_data(input, tensor_size, 0.4f);
    
    error = mps_attention_self_attention(context, input, output, NULL);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Self-attention should succeed");
    
    MPSAttentionStats stats;
    error = mps_attention_get_stats(context, &stats);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Getting stats should succeed");
    
    printf("  Peak memory usage: %u MB\n", stats.peak_memory_usage_mb);
    TEST_ASSERT(stats.peak_memory_usage_mb >= after_weights_memory, 
               "Peak memory should be at least as much as after loading weights");
    
    // Test buffer management
    error = mps_attention_free_buffers(context);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Freeing buffers should succeed");
    
    uint32_t after_free_memory;
    error = mps_attention_get_memory_usage(context, &after_free_memory);
    TEST_ASSERT(error == MPS_ATTENTION_SUCCESS, "Getting memory usage should succeed");
    printf("  Memory usage after freeing buffers: %u MB\n", after_free_memory);
    
    // Cleanup
    free(input);
    free(output);
    mps_attention_weights_destroy(weights);
    mps_attention_destroy(context);
    
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
    printf("MPS Attention Implementation Test Suite\n");
    printf("=======================================\n\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    total_tests++; passed_tests += run_test("MPS Availability", test_mps_availability);
    total_tests++; passed_tests += run_test("Config Creation and Validation", test_config_creation_and_validation);
    total_tests++; passed_tests += run_test("Memory Requirements Calculation", test_memory_requirements_calculation);
    total_tests++; passed_tests += run_test("Error String Functions", test_error_string_functions);
    total_tests++; passed_tests += run_test("Context Creation and Destruction", test_context_creation_and_destruction);
    total_tests++; passed_tests += run_test("Weights Creation and Initialization", test_weights_creation_and_initialization);
    total_tests++; passed_tests += run_test("Tensors Creation and Validation", test_tensors_creation_and_validation);
    total_tests++; passed_tests += run_test("Self-Attention Basic", test_self_attention_basic);
    total_tests++; passed_tests += run_test("Cross-Attention Basic", test_cross_attention_basic);
    total_tests++; passed_tests += run_test("Performance Statistics", test_performance_statistics);
    total_tests++; passed_tests += run_test("Different Attention Types", test_different_attention_types);
    total_tests++; passed_tests += run_test("Memory Management", test_memory_management);
    
    // Summary
    printf("Test Results\n");
    printf("============\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! MPS Attention implementation is ready.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the implementation.\n");
        return 1;
    }
}