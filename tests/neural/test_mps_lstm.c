#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "../../src/neural/ops/mps_lstm.h"

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

int test_mps_lstm_availability() {
    bool available = mps_lstm_is_available();
    printf("  MPS LSTM available: %s\n", available ? "Yes" : "No");
    
    if (available) {
        char device_name[256];
        uint32_t compute_units, max_memory_mb;
        MPSLSTMError error = mps_lstm_get_device_info(device_name, sizeof(device_name),
                                                     &compute_units, &max_memory_mb);
        TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting device info should succeed");
        
        printf("  Device: %s\n", device_name);
        printf("  Compute units: %u\n", compute_units);
        printf("  Max memory: %u MB\n", max_memory_mb);
    }
    
    TEST_PASS();
}

int test_config_creation_and_validation() {
    MPSLSTMConfig config;
    MPSLSTMError error = mps_lstm_config_create_default(&config, 128, 256, 2);
    
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Default config creation should succeed");
    TEST_ASSERT(config.input_size == 128, "Input size should be 128");
    TEST_ASSERT(config.hidden_size == 256, "Hidden size should be 256");
    TEST_ASSERT(config.num_layers == 2, "Number of layers should be 2");
    TEST_ASSERT(config.batch_size == 1, "Default batch size should be 1");
    TEST_ASSERT(config.sequence_length == 128, "Default sequence length should be 128");
    TEST_ASSERT(config.direction == MPS_LSTM_FORWARD, "Should be forward LSTM");
    TEST_ASSERT(config.cell_type == MPS_LSTM_VANILLA, "Should be vanilla LSTM");
    TEST_ASSERT(config.use_bias == true, "Use bias should be enabled by default");
    
    printf("  Default config: input=%u, hidden=%u, layers=%u, batch=%u\n",
           config.input_size, config.hidden_size, config.num_layers, config.batch_size);
    
    // Test configuration validation
    error = mps_lstm_config_validate(&config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Config validation should succeed");
    
    // Test invalid configurations
    MPSLSTMConfig invalid_config = config;
    invalid_config.input_size = 0;
    error = mps_lstm_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_LSTM_ERROR_INVALID_DIMENSIONS, "Zero input size should fail");
    
    invalid_config = config;
    invalid_config.hidden_size = 0;
    error = mps_lstm_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_LSTM_ERROR_INVALID_DIMENSIONS, "Zero hidden size should fail");
    
    invalid_config = config;
    invalid_config.dropout_rate = 1.5f;
    error = mps_lstm_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_LSTM_ERROR_INVALID_PARAM, "Invalid dropout rate should fail");
    
    printf("  Config validation tests passed\n");
    
    TEST_PASS();
}

int test_memory_requirements_calculation() {
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, 64, 128, 1);
    config.batch_size = 4;
    config.sequence_length = 64;
    
    uint32_t memory_mb;
    MPSLSTMError error = mps_lstm_calculate_memory_requirements(&config, &memory_mb);
    
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Memory calculation should succeed");
    TEST_ASSERT(memory_mb > 0, "Memory requirement should be positive");
    
    printf("  Memory requirements for config: %u MB\n", memory_mb);
    
    // Test larger configuration
    config.input_size = 512;
    config.hidden_size = 512;
    config.num_layers = 3;
    config.batch_size = 8;
    config.sequence_length = 256;
    config.direction = MPS_LSTM_BIDIRECTIONAL;
    
    uint32_t larger_memory_mb;
    error = mps_lstm_calculate_memory_requirements(&config, &larger_memory_mb);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Larger config memory calculation should succeed");
    TEST_ASSERT(larger_memory_mb > memory_mb, "Larger config should require more memory");
    
    printf("  Memory requirements for larger config: %u MB\n", larger_memory_mb);
    
    TEST_PASS();
}

int test_error_string_functions() {
    const char* success_msg = mps_lstm_get_error_string(MPS_LSTM_SUCCESS);
    TEST_ASSERT(success_msg != NULL, "Success error string should not be NULL");
    TEST_ASSERT(strlen(success_msg) > 0, "Success error string should not be empty");
    
    const char* invalid_param_msg = mps_lstm_get_error_string(MPS_LSTM_ERROR_INVALID_PARAM);
    TEST_ASSERT(invalid_param_msg != NULL, "Invalid param error string should not be NULL");
    TEST_ASSERT(strlen(invalid_param_msg) > 0, "Invalid param error string should not be empty");
    
    printf("  MPS_LSTM_SUCCESS: %s\n", success_msg);
    printf("  MPS_LSTM_ERROR_INVALID_PARAM: %s\n", invalid_param_msg);
    
    // Test unknown error code
    const char* unknown_msg = mps_lstm_get_error_string((MPSLSTMError)999);
    TEST_ASSERT(unknown_msg != NULL, "Unknown error should return a message");
    
    TEST_PASS();
}

int test_context_creation_and_destruction() {
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, 64, 128, 1);
    config.batch_size = 2;
    config.sequence_length = 32;
    
    MPSLSTMContext* context = NULL;
    MPSLSTMError error = mps_lstm_create(&context, &config);
    
    if (!mps_lstm_is_available()) {
        printf("  MPS not available, skipping context creation test\n");
        TEST_PASS();
    }
    
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Context creation should succeed");
    TEST_ASSERT(context != NULL, "Context should not be NULL");
    TEST_ASSERT(context->is_initialized == true, "Context should be initialized");
    TEST_ASSERT(context->config.input_size == 64, "Config should be copied correctly");
    TEST_ASSERT(context->config.hidden_size == 128, "Config should be copied correctly");
    TEST_ASSERT(context->config.num_layers == 1, "Config should be copied correctly");
    
    printf("  Created context with input=%u, hidden=%u, layers=%u\n",
           context->config.input_size, context->config.hidden_size, context->config.num_layers);
    
    // Test memory usage
    uint32_t memory_usage;
    error = mps_lstm_get_memory_usage(context, &memory_usage);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting memory usage should succeed");
    printf("  Context memory usage: %u MB\n", memory_usage);
    
    mps_lstm_destroy(context);
    TEST_PASS();
}

int test_weights_creation_and_initialization() {
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, 32, 64, 1);
    
    MPSLSTMWeights* weights = NULL;
    MPSLSTMError error = mps_lstm_weights_create(&weights, &config);
    
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights creation should succeed");
    TEST_ASSERT(weights != NULL, "Weights should not be NULL");
    TEST_ASSERT(weights->input_weights != NULL, "Input weights should be allocated");
    TEST_ASSERT(weights->hidden_weights != NULL, "Hidden weights should be allocated");
    
    if (config.use_bias) {
        TEST_ASSERT(weights->input_bias != NULL, "Input bias should be allocated");
        TEST_ASSERT(weights->hidden_bias != NULL, "Hidden bias should be allocated");
    }
    
    printf("  Weights created: input_size=%zu, hidden_size=%zu, bias_size=%zu\n",
           weights->input_weights_size, weights->hidden_weights_size, weights->bias_size);
    
    // Test random initialization
    error = mps_lstm_weights_init_random(weights, &config, 12345);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Random weight initialization should succeed");
    
    // Check that weights are not all zero
    float* input_weights = (float*)weights->input_weights;
    bool has_nonzero = false;
    for (size_t i = 0; i < config.input_size * config.hidden_size * 4 && !has_nonzero; i++) {
        if (fabsf(input_weights[i]) > 1e-8f) {
            has_nonzero = true;
        }
    }
    TEST_ASSERT(has_nonzero, "Weights should not all be zero after initialization");
    
    printf("  Random initialization successful\n");
    
    // Test Xavier initialization
    error = mps_lstm_weights_init_xavier(weights, &config, 54321);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Xavier weight initialization should succeed");
    
    printf("  Xavier initialization successful\n");
    
    mps_lstm_weights_destroy(weights);
    TEST_PASS();
}

int test_tensors_creation_and_validation() {
    MPSLSTMShape shape = {
        .batch_size = 2,
        .sequence_length = 16,
        .input_size = 32,
        .hidden_size = 64,
        .num_layers = 1,
        .num_directions = 1
    };
    
    MPSLSTMTensors* tensors = NULL;
    MPSLSTMError error = mps_lstm_tensors_create(&tensors, &shape);
    
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Tensors creation should succeed");
    TEST_ASSERT(tensors != NULL, "Tensors should not be NULL");
    TEST_ASSERT(tensors->shape.batch_size == 2, "Batch size should be copied");
    TEST_ASSERT(tensors->shape.sequence_length == 16, "Sequence length should be copied");
    TEST_ASSERT(tensors->shape.input_size == 32, "Input size should be copied");
    TEST_ASSERT(tensors->shape.hidden_size == 64, "Hidden size should be copied");
    TEST_ASSERT(tensors->shape.num_layers == 1, "Number of layers should be copied");
    TEST_ASSERT(tensors->shape.num_directions == 1, "Number of directions should be copied");
    
    printf("  Tensors created with shape: [%u, %u, %u], hidden=%u, layers=%u, dirs=%u\n",
           shape.batch_size, shape.sequence_length, shape.input_size,
           shape.hidden_size, shape.num_layers, shape.num_directions);
    
    // Allocate test data
    size_t input_size = shape.batch_size * shape.sequence_length * shape.input_size;
    size_t output_size = shape.batch_size * shape.sequence_length * shape.hidden_size * shape.num_directions;
    size_t state_size = shape.batch_size * shape.num_layers * shape.num_directions * shape.hidden_size;
    
    tensors->input = malloc(input_size * sizeof(float));
    tensors->output = malloc(output_size * sizeof(float));
    tensors->initial_hidden = malloc(state_size * sizeof(float));
    tensors->initial_cell = malloc(state_size * sizeof(float));
    tensors->final_hidden = malloc(state_size * sizeof(float));
    tensors->final_cell = malloc(state_size * sizeof(float));
    tensors->owns_memory = true;
    
    TEST_ASSERT(tensors->input != NULL, "Input tensor should be allocated");
    TEST_ASSERT(tensors->output != NULL, "Output tensor should be allocated");
    TEST_ASSERT(tensors->initial_hidden != NULL, "Initial hidden should be allocated");
    TEST_ASSERT(tensors->initial_cell != NULL, "Initial cell should be allocated");
    
    // Initialize test data
    initialize_test_data(tensors->input, input_size, 0.1f);
    initialize_test_data(tensors->initial_hidden, state_size, 0.0f);
    initialize_test_data(tensors->initial_cell, state_size, 0.0f);
    memset(tensors->output, 0, output_size * sizeof(float));
    memset(tensors->final_hidden, 0, state_size * sizeof(float));
    memset(tensors->final_cell, 0, state_size * sizeof(float));
    
    // Test tensor validation
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, shape.input_size, shape.hidden_size, shape.num_layers);
    config.batch_size = shape.batch_size;
    config.sequence_length = shape.sequence_length;
    
    error = mps_lstm_tensors_validate(tensors, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Tensor validation should succeed");
    
    printf("  Tensor validation passed\n");
    
    mps_lstm_tensors_destroy(tensors);
    TEST_PASS();
}

int test_sequence_processing() {
    if (!mps_lstm_is_available()) {
        printf("  MPS not available, skipping sequence processing test\n");
        TEST_PASS();
    }
    
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, 16, 32, 1);
    config.batch_size = 1;
    config.sequence_length = 8;
    
    MPSLSTMContext* context = NULL;
    MPSLSTMError error = mps_lstm_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSLSTMWeights* weights = NULL;
    error = mps_lstm_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_weights_init_random(weights, &config, 54321);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights initialization should succeed");
    
    error = mps_lstm_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Loading weights should succeed");
    
    // Prepare input and output
    uint32_t sequence_length = 8;
    size_t input_size = sequence_length * config.input_size;
    size_t output_size = sequence_length * config.hidden_size;
    size_t state_size = config.num_layers * config.hidden_size;
    
    float* input = malloc(input_size * sizeof(float));
    float* output = malloc(output_size * sizeof(float));
    float* initial_hidden = malloc(state_size * sizeof(float));
    float* initial_cell = malloc(state_size * sizeof(float));
    float* final_hidden = malloc(state_size * sizeof(float));
    float* final_cell = malloc(state_size * sizeof(float));
    
    TEST_ASSERT(input != NULL && output != NULL && initial_hidden != NULL && 
               initial_cell != NULL && final_hidden != NULL && final_cell != NULL,
               "Memory allocation should succeed");
    
    // Initialize input with test pattern
    initialize_test_data(input, input_size, 0.5f);
    memset(initial_hidden, 0, state_size * sizeof(float));
    memset(initial_cell, 0, state_size * sizeof(float));
    memset(output, 0, output_size * sizeof(float));
    memset(final_hidden, 0, state_size * sizeof(float));
    memset(final_cell, 0, state_size * sizeof(float));
    
    // Process sequence
    error = mps_lstm_process_sequence(context, input, output, sequence_length,
                                     initial_hidden, initial_cell,
                                     final_hidden, final_cell);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Sequence processing should succeed");
    
    // Check that output is not all zeros
    bool has_nonzero_output = false;
    for (size_t i = 0; i < output_size && !has_nonzero_output; i++) {
        if (fabsf(output[i]) > FLOAT_TOLERANCE) {
            has_nonzero_output = true;
        }
    }
    TEST_ASSERT(has_nonzero_output, "Output should not be all zeros");
    
    // Check that final states are not all zeros
    bool has_nonzero_hidden = false;
    for (size_t i = 0; i < state_size && !has_nonzero_hidden; i++) {
        if (fabsf(final_hidden[i]) > FLOAT_TOLERANCE) {
            has_nonzero_hidden = true;
        }
    }
    TEST_ASSERT(has_nonzero_hidden, "Final hidden state should not be all zeros");
    
    printf("  Sequence processing completed successfully\n");
    printf("  Input sample: [%.3f, %.3f, %.3f, ...]\n", input[0], input[1], input[2]);
    printf("  Output sample: [%.3f, %.3f, %.3f, ...]\n", output[0], output[1], output[2]);
    printf("  Final hidden sample: [%.3f, %.3f, %.3f, ...]\n", 
           final_hidden[0], final_hidden[1], final_hidden[2]);
    
    // Cleanup
    free(input);
    free(output);
    free(initial_hidden);
    free(initial_cell);
    free(final_hidden);
    free(final_cell);
    mps_lstm_weights_destroy(weights);
    mps_lstm_destroy(context);
    
    TEST_PASS();
}

int test_batch_processing() {
    if (!mps_lstm_is_available()) {
        printf("  MPS not available, skipping batch processing test\n");
        TEST_PASS();
    }
    
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, 8, 16, 1);
    config.batch_size = 3;
    config.sequence_length = 6;
    
    MPSLSTMContext* context = NULL;
    MPSLSTMError error = mps_lstm_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSLSTMWeights* weights = NULL;
    error = mps_lstm_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_weights_init_xavier(weights, &config, 98765);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights initialization should succeed");
    
    error = mps_lstm_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Loading weights should succeed");
    
    // Prepare batch data
    uint32_t batch_size = 3;
    uint32_t max_sequence_length = 6;
    uint32_t sequence_lengths[] = {4, 6, 3}; // Variable length sequences
    
    size_t input_size = batch_size * max_sequence_length * config.input_size;
    size_t output_size = batch_size * max_sequence_length * config.hidden_size;
    
    float* input = malloc(input_size * sizeof(float));
    float* output = malloc(output_size * sizeof(float));
    
    TEST_ASSERT(input != NULL && output != NULL, "Memory allocation should succeed");
    
    // Initialize input with different patterns for each batch item
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < max_sequence_length; s++) {
            for (uint32_t i = 0; i < config.input_size; i++) {
                size_t idx = b * max_sequence_length * config.input_size + 
                            s * config.input_size + i;
                if (s < sequence_lengths[b]) {
                    input[idx] = 0.1f * (b + 1) + 0.01f * s + 0.001f * i;
                } else {
                    input[idx] = 0.0f; // Padding
                }
            }
        }
    }
    memset(output, 0, output_size * sizeof(float));
    
    // Process batch
    error = mps_lstm_process_batch(context, input, output, sequence_lengths, batch_size);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Batch processing should succeed");
    
    // Verify output
    bool has_nonzero_output = false;
    for (size_t i = 0; i < output_size && !has_nonzero_output; i++) {
        if (fabsf(output[i]) > FLOAT_TOLERANCE) {
            has_nonzero_output = true;
        }
    }
    TEST_ASSERT(has_nonzero_output, "Output should not be all zeros");
    
    printf("  Batch processing completed successfully\n");
    printf("  Batch sizes: [%u, %u, %u]\n", sequence_lengths[0], sequence_lengths[1], sequence_lengths[2]);
    printf("  Output samples from each batch item:\n");
    for (uint32_t b = 0; b < batch_size; b++) {
        size_t idx = b * max_sequence_length * config.hidden_size;
        printf("    Batch %u: [%.3f, %.3f, %.3f, ...]\n", 
               b, output[idx], output[idx + 1], output[idx + 2]);
    }
    
    // Cleanup
    free(input);
    free(output);
    mps_lstm_weights_destroy(weights);
    mps_lstm_destroy(context);
    
    TEST_PASS();
}

int test_state_management() {
    if (!mps_lstm_is_available()) {
        printf("  MPS not available, skipping state management test\n");
        TEST_PASS();
    }
    
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, 4, 8, 1);
    config.stateful = true;
    
    MPSLSTMContext* context = NULL;
    MPSLSTMError error = mps_lstm_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Context creation should succeed");
    
    // Create weights
    MPSLSTMWeights* weights = NULL;
    error = mps_lstm_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_weights_init_random(weights, &config, 11111);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights initialization should succeed");
    
    error = mps_lstm_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Loading weights should succeed");
    
    // Test state reset
    error = mps_lstm_reset_state(context);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "State reset should succeed");
    
    // Set custom initial state
    size_t state_size = config.num_layers * config.hidden_size;
    float* hidden_state = malloc(state_size * sizeof(float));
    float* cell_state = malloc(state_size * sizeof(float));
    float* retrieved_hidden = malloc(state_size * sizeof(float));
    float* retrieved_cell = malloc(state_size * sizeof(float));
    
    TEST_ASSERT(hidden_state != NULL && cell_state != NULL && 
               retrieved_hidden != NULL && retrieved_cell != NULL,
               "Memory allocation should succeed");
    
    // Initialize states with test values
    for (size_t i = 0; i < state_size; i++) {
        hidden_state[i] = 0.1f + 0.01f * i;
        cell_state[i] = 0.2f + 0.01f * i;
    }
    
    // Set state
    error = mps_lstm_set_state(context, hidden_state, cell_state);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Setting state should succeed");
    
    // Get state back
    error = mps_lstm_get_state(context, retrieved_hidden, retrieved_cell);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting state should succeed");
    
    // Verify state values
    bool states_match = true;
    for (size_t i = 0; i < state_size && states_match; i++) {
        if (!floats_equal(hidden_state[i], retrieved_hidden[i], FLOAT_TOLERANCE) ||
            !floats_equal(cell_state[i], retrieved_cell[i], FLOAT_TOLERANCE)) {
            states_match = false;
        }
    }
    TEST_ASSERT(states_match, "Retrieved states should match set states");
    
    printf("  State management test passed\n");
    printf("  Set hidden state: [%.3f, %.3f, %.3f, ...]\n", 
           hidden_state[0], hidden_state[1], hidden_state[2]);
    printf("  Retrieved hidden state: [%.3f, %.3f, %.3f, ...]\n", 
           retrieved_hidden[0], retrieved_hidden[1], retrieved_hidden[2]);
    
    // Cleanup
    free(hidden_state);
    free(cell_state);
    free(retrieved_hidden);
    free(retrieved_cell);
    mps_lstm_weights_destroy(weights);
    mps_lstm_destroy(context);
    
    TEST_PASS();
}

int test_performance_statistics() {
    if (!mps_lstm_is_available()) {
        printf("  MPS not available, skipping performance test\n");
        TEST_PASS();
    }
    
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, 16, 32, 1);
    config.batch_size = 2;
    config.sequence_length = 10;
    
    MPSLSTMContext* context = NULL;
    MPSLSTMError error = mps_lstm_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Context creation should succeed");
    
    // Get initial statistics
    MPSLSTMStats initial_stats;
    error = mps_lstm_get_stats(context, &initial_stats);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting stats should succeed");
    TEST_ASSERT(initial_stats.total_operations == 0, "Initial operations should be 0");
    TEST_ASSERT(initial_stats.total_compute_time_ns == 0, "Initial compute time should be 0");
    
    // Create weights and perform operations
    MPSLSTMWeights* weights = NULL;
    error = mps_lstm_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_weights_init_random(weights, &config, 22222);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights initialization should succeed");
    
    error = mps_lstm_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Loading weights should succeed");
    
    // Prepare test data
    size_t input_size = config.batch_size * config.sequence_length * config.input_size;
    size_t output_size = config.batch_size * config.sequence_length * config.hidden_size;
    
    float* input = malloc(input_size * sizeof(float));
    float* output = malloc(output_size * sizeof(float));
    
    initialize_test_data(input, input_size, 0.25f);
    
    // Create tensors
    MPSLSTMShape shape = {
        .batch_size = config.batch_size,
        .sequence_length = config.sequence_length,
        .input_size = config.input_size,
        .hidden_size = config.hidden_size,
        .num_layers = config.num_layers,
        .num_directions = 1
    };
    
    MPSLSTMTensors* tensors = NULL;
    error = mps_lstm_tensors_create(&tensors, &shape);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Tensors creation should succeed");
    
    tensors->input = input;
    tensors->output = output;
    tensors->owns_memory = false;
    
    // Perform multiple operations
    for (int i = 0; i < 3; i++) {
        error = mps_lstm_forward(context, tensors);
        TEST_ASSERT(error == MPS_LSTM_SUCCESS, "LSTM forward should succeed");
    }
    
    // Get updated statistics
    MPSLSTMStats final_stats;
    error = mps_lstm_get_stats(context, &final_stats);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting final stats should succeed");
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
    error = mps_lstm_reset_stats(context);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Stats reset should succeed");
    
    MPSLSTMStats reset_stats;
    error = mps_lstm_get_stats(context, &reset_stats);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting reset stats should succeed");
    TEST_ASSERT(reset_stats.total_operations == 0, "Reset operations should be 0");
    TEST_ASSERT(reset_stats.total_compute_time_ns == 0, "Reset compute time should be 0");
    
    printf("  Statistics reset successfully\n");
    
    // Cleanup
    free(input);
    free(output);
    mps_lstm_tensors_destroy(tensors);
    mps_lstm_weights_destroy(weights);
    mps_lstm_destroy(context);
    
    TEST_PASS();
}

int test_bidirectional_lstm() {
    if (!mps_lstm_is_available()) {
        printf("  MPS not available, skipping bidirectional LSTM test\n");
        TEST_PASS();
    }
    
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, 8, 16, 1);
    config.direction = MPS_LSTM_BIDIRECTIONAL;
    config.batch_size = 1;
    config.sequence_length = 5;
    
    MPSLSTMContext* context = NULL;
    MPSLSTMError error = mps_lstm_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Bidirectional context creation should succeed");
    
    // Create weights
    MPSLSTMWeights* weights = NULL;
    error = mps_lstm_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Bidirectional weights creation should succeed");
    
    error = mps_lstm_weights_init_xavier(weights, &config, 33333);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Bidirectional weights initialization should succeed");
    
    error = mps_lstm_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Loading bidirectional weights should succeed");
    
    // Test processing
    size_t input_size = config.batch_size * config.sequence_length * config.input_size;
    size_t output_size = config.batch_size * config.sequence_length * config.hidden_size * 2; // 2 directions
    
    float* input = malloc(input_size * sizeof(float));
    float* output = malloc(output_size * sizeof(float));
    
    TEST_ASSERT(input != NULL && output != NULL, "Memory allocation should succeed");
    
    initialize_test_data(input, input_size, 0.3f);
    memset(output, 0, output_size * sizeof(float));
    
    error = mps_lstm_process_sequence(context, input, output, config.sequence_length,
                                     NULL, NULL, NULL, NULL);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Bidirectional sequence processing should succeed");
    
    // Verify output dimensions and content
    bool has_nonzero_output = false;
    for (size_t i = 0; i < output_size && !has_nonzero_output; i++) {
        if (fabsf(output[i]) > FLOAT_TOLERANCE) {
            has_nonzero_output = true;
        }
    }
    TEST_ASSERT(has_nonzero_output, "Bidirectional output should not be all zeros");
    
    printf("  Bidirectional LSTM processing completed successfully\n");
    printf("  Input sequence length: %u, hidden size: %u, directions: 2\n", 
           config.sequence_length, config.hidden_size);
    printf("  Output shape: [%u, %u, %u]\n", 
           config.batch_size, config.sequence_length, config.hidden_size * 2);
    printf("  Output sample: [%.3f, %.3f, %.3f, ...]\n", output[0], output[1], output[2]);
    
    // Cleanup
    free(input);
    free(output);
    mps_lstm_weights_destroy(weights);
    mps_lstm_destroy(context);
    
    TEST_PASS();
}

int test_different_lstm_variants() {
    // Test different LSTM cell types and activations
    MPSLSTMCellType cell_types[] = {
        MPS_LSTM_VANILLA,
        MPS_LSTM_PEEPHOLE,
        MPS_LSTM_COUPLED_INPUT_FORGET
    };
    
    const char* type_names[] = {
        "Vanilla",
        "Peephole",
        "Coupled Input-Forget"
    };
    
    for (int t = 0; t < 3; t++) {
        MPSLSTMConfig config;
        mps_lstm_config_create_default(&config, 4, 8, 1);
        config.cell_type = cell_types[t];
        
        if (cell_types[t] == MPS_LSTM_PEEPHOLE) {
            config.use_peepholes = true;
        }
        
        MPSLSTMError error = mps_lstm_config_validate(&config);
        TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Config validation should succeed for all cell types");
        
        printf("  %s LSTM cell type validated\n", type_names[t]);
    }
    
    // Test different activation functions
    MPSLSTMActivation activations[] = {
        MPS_LSTM_ACTIVATION_TANH,
        MPS_LSTM_ACTIVATION_SIGMOID,
        MPS_LSTM_ACTIVATION_RELU,
        MPS_LSTM_ACTIVATION_HARD_SIGMOID
    };
    
    const char* activation_names[] = {
        "Tanh",
        "Sigmoid", 
        "ReLU",
        "Hard Sigmoid"
    };
    
    for (int a = 0; a < 4; a++) {
        MPSLSTMConfig config;
        mps_lstm_config_create_default(&config, 4, 8, 1);
        config.input_activation = activations[a];
        config.forget_activation = activations[a];
        config.output_activation = activations[a];
        config.cell_activation = activations[a];
        
        MPSLSTMError error = mps_lstm_config_validate(&config);
        TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Config validation should succeed for all activations");
        
        printf("  %s activation function validated\n", activation_names[a]);
    }
    
    TEST_PASS();
}

int test_memory_management() {
    if (!mps_lstm_is_available()) {
        printf("  MPS not available, skipping memory management test\n");
        TEST_PASS();
    }
    
    MPSLSTMConfig config;
    mps_lstm_config_create_default(&config, 32, 64, 2);
    config.batch_size = 4;
    config.sequence_length = 16;
    
    MPSLSTMContext* context = NULL;
    MPSLSTMError error = mps_lstm_create(&context, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Context creation should succeed");
    
    // Check initial memory usage
    uint32_t initial_memory;
    error = mps_lstm_get_memory_usage(context, &initial_memory);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting memory usage should succeed");
    printf("  Initial memory usage: %u MB\n", initial_memory);
    
    // Create weights and check memory increase
    MPSLSTMWeights* weights = NULL;
    error = mps_lstm_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Weights creation should succeed");
    
    error = mps_lstm_load_weights(context, weights);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Loading weights should succeed");
    
    uint32_t after_weights_memory;
    error = mps_lstm_get_memory_usage(context, &after_weights_memory);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting memory usage should succeed");
    printf("  Memory usage after loading weights: %u MB\n", after_weights_memory);
    
    // Perform operations and check peak memory
    size_t input_size = config.batch_size * config.sequence_length * config.input_size;
    size_t output_size = config.batch_size * config.sequence_length * config.hidden_size;
    
    float* input = malloc(input_size * sizeof(float));
    float* output = malloc(output_size * sizeof(float));
    
    initialize_test_data(input, input_size, 0.4f);
    
    error = mps_lstm_process_sequence(context, input, output, config.sequence_length,
                                     NULL, NULL, NULL, NULL);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Sequence processing should succeed");
    
    MPSLSTMStats stats;
    error = mps_lstm_get_stats(context, &stats);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting stats should succeed");
    
    printf("  Peak memory usage: %u MB\n", stats.peak_memory_usage_mb);
    TEST_ASSERT(stats.peak_memory_usage_mb >= after_weights_memory, 
               "Peak memory should be at least as much as after loading weights");
    
    // Test buffer management
    error = mps_lstm_free_buffers(context);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Freeing buffers should succeed");
    
    uint32_t after_free_memory;
    error = mps_lstm_get_memory_usage(context, &after_free_memory);
    TEST_ASSERT(error == MPS_LSTM_SUCCESS, "Getting memory usage should succeed");
    printf("  Memory usage after freeing buffers: %u MB\n", after_free_memory);
    
    // Cleanup
    free(input);
    free(output);
    mps_lstm_weights_destroy(weights);
    mps_lstm_destroy(context);
    
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
    printf("MPS LSTM Implementation Test Suite\n");
    printf("==================================\n\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    total_tests++; passed_tests += run_test("MPS LSTM Availability", test_mps_lstm_availability);
    total_tests++; passed_tests += run_test("Config Creation and Validation", test_config_creation_and_validation);
    total_tests++; passed_tests += run_test("Memory Requirements Calculation", test_memory_requirements_calculation);
    total_tests++; passed_tests += run_test("Error String Functions", test_error_string_functions);
    total_tests++; passed_tests += run_test("Context Creation and Destruction", test_context_creation_and_destruction);
    total_tests++; passed_tests += run_test("Weights Creation and Initialization", test_weights_creation_and_initialization);
    total_tests++; passed_tests += run_test("Tensors Creation and Validation", test_tensors_creation_and_validation);
    total_tests++; passed_tests += run_test("Sequence Processing", test_sequence_processing);
    total_tests++; passed_tests += run_test("Batch Processing", test_batch_processing);
    total_tests++; passed_tests += run_test("State Management", test_state_management);
    total_tests++; passed_tests += run_test("Performance Statistics", test_performance_statistics);
    total_tests++; passed_tests += run_test("Bidirectional LSTM", test_bidirectional_lstm);
    total_tests++; passed_tests += run_test("Different LSTM Variants", test_different_lstm_variants);
    total_tests++; passed_tests += run_test("Memory Management", test_memory_management);
    
    // Summary
    printf("Test Results\n");
    printf("============\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! MPS LSTM implementation is ready.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the implementation.\n");
        return 1;
    }
}