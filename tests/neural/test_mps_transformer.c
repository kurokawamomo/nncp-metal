#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "../../src/neural/engines/mps_transformer.h"

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
static void initialize_test_tokens(uint32_t* tokens, size_t size, uint32_t vocab_size) {
    for (size_t i = 0; i < size; i++) {
        tokens[i] = i % vocab_size;
    }
}

static void initialize_test_floats(float* data, size_t size, float base_value) {
    for (size_t i = 0; i < size; i++) {
        data[i] = base_value + (float)(i % 100) * 0.01f;
    }
}

// Test functions

int test_transformer_availability() {
    bool available = mps_transformer_is_available();
    printf("  MPS Transformer available: %s\n", available ? "Yes" : "No");
    
    if (available) {
        char device_name[256];
        uint32_t compute_units, max_memory_mb;
        MPSTransformerError error = mps_transformer_get_device_info(device_name, sizeof(device_name),
                                                                   &compute_units, &max_memory_mb);
        TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Getting device info should succeed");
        
        printf("  Device: %s\n", device_name);
        printf("  Compute units: %u\n", compute_units);
        printf("  Max memory: %u MB\n", max_memory_mb);
    }
    
    TEST_PASS();
}

int test_config_creation_and_validation() {
    MPSTransformerConfig config;
    
    // Test BERT-style configuration
    MPSTransformerError error = mps_transformer_config_create_default(&config, "bert", 30000, 512, 768, 12, 12);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "BERT config creation should succeed");
    TEST_ASSERT(config.vocab_size == 30000, "Vocab size should be 30000");
    TEST_ASSERT(config.sequence_length == 512, "Sequence length should be 512");
    TEST_ASSERT(config.hidden_size == 768, "Hidden size should be 768");
    TEST_ASSERT(config.num_layers == 12, "Number of layers should be 12");
    TEST_ASSERT(config.num_heads == 12, "Number of heads should be 12");
    TEST_ASSERT(config.is_encoder_only == true, "Should be encoder-only");
    TEST_ASSERT(config.is_decoder_only == false, "Should not be decoder-only");
    TEST_ASSERT(config.use_token_type_embeddings == true, "Should use token type embeddings");
    
    printf("  BERT config: vocab=%u, seq_len=%u, hidden=%u, layers=%u, heads=%u\n",
           config.vocab_size, config.sequence_length, config.hidden_size, 
           config.num_layers, config.num_heads);
    
    // Test GPT-style configuration
    error = mps_transformer_config_create_default(&config, "gpt", 50000, 1024, 1024, 24, 16);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "GPT config creation should succeed");
    TEST_ASSERT(config.is_decoder_only == true, "Should be decoder-only");
    TEST_ASSERT(config.tie_word_embeddings == true, "Should tie word embeddings");
    TEST_ASSERT(config.use_token_type_embeddings == false, "Should not use token type embeddings");
    
    printf("  GPT config: vocab=%u, seq_len=%u, hidden=%u, layers=%u, heads=%u\n",
           config.vocab_size, config.sequence_length, config.hidden_size, 
           config.num_layers, config.num_heads);
    
    // Test T5-style configuration
    error = mps_transformer_config_create_default(&config, "t5", 32000, 512, 512, 6, 8);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "T5 config creation should succeed");
    TEST_ASSERT(config.is_encoder_decoder == true, "Should be encoder-decoder");
    TEST_ASSERT(config.pos_encoding_type == MPS_TRANSFORMER_POSITIONAL_SINUSOIDAL, 
               "Should use sinusoidal positional encoding");
    
    printf("  T5 config: vocab=%u, seq_len=%u, hidden=%u, layers=%u, heads=%u\n",
           config.vocab_size, config.sequence_length, config.hidden_size, 
           config.num_layers, config.num_heads);
    
    // Test configuration validation
    error = mps_transformer_config_validate(&config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Config validation should succeed");
    
    // Test invalid configurations
    MPSTransformerConfig invalid_config = config;
    invalid_config.hidden_size = 777; // Not divisible by num_heads
    error = mps_transformer_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS, "Invalid config should fail validation");
    
    invalid_config = config;
    invalid_config.vocab_size = 0;
    error = mps_transformer_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS, "Zero vocab size should fail");
    
    invalid_config = config;
    invalid_config.is_encoder_only = true;
    invalid_config.is_decoder_only = true;
    error = mps_transformer_config_validate(&invalid_config);
    TEST_ASSERT(error == MPS_TRANSFORMER_ERROR_INCOMPATIBLE_CONFIG, "Multiple model types should fail");
    
    printf("  Config validation tests passed\n");
    
    TEST_PASS();
}

int test_memory_requirements_calculation() {
    MPSTransformerConfig config;
    mps_transformer_config_create_default(&config, "bert", 30000, 256, 512, 6, 8);
    config.batch_size = 4;
    
    uint32_t memory_mb;
    MPSTransformerError error = mps_transformer_calculate_memory_requirements(&config, &memory_mb);
    
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Memory calculation should succeed");
    TEST_ASSERT(memory_mb > 0, "Memory requirement should be positive");
    
    printf("  Memory requirements for small model: %u MB\n", memory_mb);
    
    // Test larger configuration
    config.vocab_size = 50000;
    config.sequence_length = 1024;
    config.hidden_size = 1024;
    config.num_layers = 24;
    config.num_heads = 16;
    config.batch_size = 8;
    
    uint32_t larger_memory_mb;
    error = mps_transformer_calculate_memory_requirements(&config, &larger_memory_mb);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Larger config memory calculation should succeed");
    TEST_ASSERT(larger_memory_mb > memory_mb, "Larger config should require more memory");
    
    printf("  Memory requirements for large model: %u MB\n", larger_memory_mb);
    
    TEST_PASS();
}

int test_error_string_functions() {
    const char* success_msg = mps_transformer_get_error_string(MPS_TRANSFORMER_SUCCESS);
    TEST_ASSERT(success_msg != NULL, "Success error string should not be NULL");
    TEST_ASSERT(strlen(success_msg) > 0, "Success error string should not be empty");
    
    const char* invalid_param_msg = mps_transformer_get_error_string(MPS_TRANSFORMER_ERROR_INVALID_PARAM);
    TEST_ASSERT(invalid_param_msg != NULL, "Invalid param error string should not be NULL");
    TEST_ASSERT(strlen(invalid_param_msg) > 0, "Invalid param error string should not be empty");
    
    printf("  MPS_TRANSFORMER_SUCCESS: %s\n", success_msg);
    printf("  MPS_TRANSFORMER_ERROR_INVALID_PARAM: %s\n", invalid_param_msg);
    
    // Test unknown error code
    const char* unknown_msg = mps_transformer_get_error_string((MPSTransformerError)999);
    TEST_ASSERT(unknown_msg != NULL, "Unknown error should return a message");
    
    TEST_PASS();
}

int test_context_creation_and_destruction() {
    MPSTransformerConfig config;
    mps_transformer_config_create_default(&config, "bert", 1000, 64, 128, 2, 4);
    config.batch_size = 2;
    
    MPSTransformerContext* context = NULL;
    MPSTransformerError error = mps_transformer_create(&context, &config);
    
    if (!mps_transformer_is_available()) {
        printf("  MPS not available, skipping context creation test\n");
        TEST_PASS();
    }
    
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Context creation should succeed");
    TEST_ASSERT(context != NULL, "Context should not be NULL");
    TEST_ASSERT(context->is_initialized == true, "Context should be initialized");
    TEST_ASSERT(context->config.vocab_size == 1000, "Config should be copied correctly");
    TEST_ASSERT(context->config.hidden_size == 128, "Config should be copied correctly");
    TEST_ASSERT(context->config.num_layers == 2, "Config should be copied correctly");
    
    printf("  Created transformer context with vocab=%u, hidden=%u, layers=%u\n",
           context->config.vocab_size, context->config.hidden_size, context->config.num_layers);
    
    // Test memory usage
    uint32_t memory_usage;
    error = mps_transformer_get_memory_usage(context, &memory_usage);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Getting memory usage should succeed");
    printf("  Context memory usage: %u MB\n", memory_usage);
    
    // Test attention contexts initialization
    TEST_ASSERT(context->attention_contexts != NULL, "Attention contexts should be allocated");
    for (uint32_t i = 0; i < context->config.num_layers; i++) {
        TEST_ASSERT(context->attention_contexts[i] != NULL, "Each attention context should be initialized");
    }
    printf("  %u attention contexts initialized\n", context->config.num_layers);
    
    mps_transformer_destroy(context);
    TEST_PASS();
}

int test_weights_creation_and_initialization() {
    MPSTransformerConfig config;
    mps_transformer_config_create_default(&config, "gpt", 5000, 32, 64, 2, 2);
    
    MPSTransformerWeights* weights = NULL;
    MPSTransformerError error = mps_transformer_weights_create(&weights, &config);
    
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights creation should succeed");
    TEST_ASSERT(weights != NULL, "Weights should not be NULL");
    TEST_ASSERT(weights->word_embeddings != NULL, "Word embeddings should be allocated");
    TEST_ASSERT(weights->layer_weights != NULL, "Layer weights array should be allocated");
    
    if (config.pos_encoding_type == MPS_TRANSFORMER_POSITIONAL_LEARNED) {
        TEST_ASSERT(weights->position_embeddings != NULL, "Position embeddings should be allocated");
    }
    
    printf("  Weights created: embedding_size=%zu, layer_weights_size=%zu\n",
           weights->embedding_size, weights->layer_weights_size);
    
    // Test random initialization
    error = mps_transformer_weights_init_random(weights, &config, 12345);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Random weight initialization should succeed");
    
    // Check that embeddings are not all zero
    float* word_embeddings = (float*)weights->word_embeddings;
    bool has_nonzero = false;
    for (size_t i = 0; i < config.vocab_size * config.hidden_size && !has_nonzero; i++) {
        if (fabsf(word_embeddings[i]) > 1e-8f) {
            has_nonzero = true;
        }
    }
    TEST_ASSERT(has_nonzero, "Word embeddings should not all be zero after initialization");
    printf("  Random initialization successful\n");
    
    mps_transformer_weights_destroy(weights);
    TEST_PASS();
}

int test_tensors_creation_and_validation() {
    MPSTransformerConfig config;
    mps_transformer_config_create_default(&config, "bert", 1000, 32, 64, 2, 2);
    config.batch_size = 2;
    
    uint32_t batch_size = 2;
    uint32_t sequence_length = 16;
    
    MPSTransformerTensors* tensors = NULL;
    MPSTransformerError error = mps_transformer_tensors_create(&tensors, &config, batch_size, sequence_length);
    
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Tensors creation should succeed");
    TEST_ASSERT(tensors != NULL, "Tensors should not be NULL");
    TEST_ASSERT(tensors->batch_size == batch_size, "Batch size should be set correctly");
    TEST_ASSERT(tensors->sequence_length == sequence_length, "Sequence length should be set correctly");
    
    printf("  Tensors created with batch_size=%u, sequence_length=%u\n",
           tensors->batch_size, tensors->sequence_length);
    
    // Allocate test data
    size_t token_size = batch_size * sequence_length;
    size_t hidden_size = config.hidden_size;
    size_t embeddings_size = batch_size * sequence_length * hidden_size;
    size_t logits_size = batch_size * sequence_length * config.vocab_size;
    
    tensors->input_ids = malloc(token_size * sizeof(uint32_t));
    tensors->embeddings = malloc(embeddings_size * sizeof(float));
    tensors->hidden_states = malloc(embeddings_size * sizeof(float));
    tensors->logits = malloc(logits_size * sizeof(float));
    tensors->owns_memory = true;
    
    TEST_ASSERT(tensors->input_ids != NULL, "Input IDs should be allocated");
    TEST_ASSERT(tensors->embeddings != NULL, "Embeddings should be allocated");
    TEST_ASSERT(tensors->hidden_states != NULL, "Hidden states should be allocated");
    TEST_ASSERT(tensors->logits != NULL, "Logits should be allocated");
    
    // Initialize test data
    initialize_test_tokens(tensors->input_ids, token_size, config.vocab_size);
    initialize_test_floats(tensors->embeddings, embeddings_size, 0.1f);
    memset(tensors->hidden_states, 0, embeddings_size * sizeof(float));
    memset(tensors->logits, 0, logits_size * sizeof(float));
    
    // Test tensor validation
    error = mps_transformer_tensors_validate(tensors, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Tensor validation should succeed");
    
    printf("  Tensor validation passed\n");
    
    mps_transformer_tensors_destroy(tensors);
    TEST_PASS();
}

int test_bert_encoder_basic() {
    if (!mps_transformer_is_available()) {
        printf("  MPS not available, skipping BERT encoder test\n");
        TEST_PASS();
    }
    
    MPSTransformerConfig config;
    mps_transformer_config_create_default(&config, "bert", 1000, 32, 64, 2, 2);
    config.batch_size = 1;
    
    MPSTransformerContext* context = NULL;
    MPSTransformerError error = mps_transformer_create(&context, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSTransformerWeights* weights = NULL;
    error = mps_transformer_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights creation should succeed");
    
    error = mps_transformer_weights_init_random(weights, &config, 54321);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights initialization should succeed");
    
    error = mps_transformer_load_weights(context, weights);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Loading weights should succeed");
    
    // Prepare input
    uint32_t batch_size = 1;
    uint32_t sequence_length = 16;
    size_t token_size = batch_size * sequence_length;
    size_t hidden_size = config.hidden_size;
    size_t hidden_states_size = batch_size * sequence_length * hidden_size;
    
    uint32_t* input_ids = malloc(token_size * sizeof(uint32_t));
    float* output_hidden_states = malloc(hidden_states_size * sizeof(float));
    
    TEST_ASSERT(input_ids != NULL && output_hidden_states != NULL, "Memory allocation should succeed");
    
    // Initialize input with test tokens
    initialize_test_tokens(input_ids, token_size, config.vocab_size);
    memset(output_hidden_states, 0, hidden_states_size * sizeof(float));
    
    // Perform encoding
    error = mps_transformer_encode(context, input_ids, NULL, output_hidden_states);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "BERT encoding should succeed");
    
    // Check that output is not all zeros
    bool has_nonzero_output = false;
    for (size_t i = 0; i < batch_size * sequence_length * hidden_size && !has_nonzero_output; i++) {
        if (fabsf(output_hidden_states[i]) > FLOAT_TOLERANCE) {
            has_nonzero_output = true;
        }
    }
    TEST_ASSERT(has_nonzero_output, "Output should not be all zeros");
    
    printf("  BERT encoding completed successfully\n");
    printf("  Input tokens: [%u, %u, %u, ...]\n", input_ids[0], input_ids[1], input_ids[2]);
    printf("  Output sample: [%.3f, %.3f, %.3f, ...]\n", 
           output_hidden_states[0], output_hidden_states[1], output_hidden_states[2]);
    
    // Cleanup
    free(input_ids);
    free(output_hidden_states);
    mps_transformer_weights_destroy(weights);
    mps_transformer_destroy(context);
    
    TEST_PASS();
}

int test_gpt_generation_basic() {
    if (!mps_transformer_is_available()) {
        printf("  MPS not available, skipping GPT generation test\n");
        TEST_PASS();
    }
    
    MPSTransformerConfig config;
    mps_transformer_config_create_default(&config, "gpt", 1000, 32, 64, 2, 2);
    config.batch_size = 1;
    
    MPSTransformerContext* context = NULL;
    MPSTransformerError error = mps_transformer_create(&context, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSTransformerWeights* weights = NULL;
    error = mps_transformer_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights creation should succeed");
    
    error = mps_transformer_weights_init_random(weights, &config, 98765);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights initialization should succeed");
    
    error = mps_transformer_load_weights(context, weights);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Loading weights should succeed");
    
    // Prepare input for generation
    uint32_t batch_size = 1;
    uint32_t sequence_length = 8;
    size_t token_size = batch_size * sequence_length;
    size_t logits_size = batch_size * sequence_length * config.vocab_size;
    
    uint32_t* input_ids = malloc(token_size * sizeof(uint32_t));
    float* logits = malloc(logits_size * sizeof(float));
    
    TEST_ASSERT(input_ids != NULL && logits != NULL, "Memory allocation should succeed");
    
    // Initialize input with test tokens
    initialize_test_tokens(input_ids, token_size, config.vocab_size);
    memset(logits, 0, logits_size * sizeof(float));
    
    // Perform logits generation
    error = mps_transformer_generate_logits(context, input_ids, NULL, logits);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "GPT generation should succeed");
    
    // Check that logits are computed
    bool has_nonzero_logits = false;
    for (size_t i = 0; i < logits_size && !has_nonzero_logits; i++) {
        if (fabsf(logits[i]) > FLOAT_TOLERANCE) {
            has_nonzero_logits = true;
        }
    }
    TEST_ASSERT(has_nonzero_logits, "Logits should not be all zeros");
    
    printf("  GPT generation completed successfully\n");
    printf("  Input tokens: [%u, %u, %u, ...]\n", input_ids[0], input_ids[1], input_ids[2]);
    printf("  Logits sample: [%.3f, %.3f, %.3f, ...]\n", logits[0], logits[1], logits[2]);
    
    // Cleanup
    free(input_ids);
    free(logits);
    mps_transformer_weights_destroy(weights);
    mps_transformer_destroy(context);
    
    TEST_PASS();
}

int test_transformer_forward_pass() {
    if (!mps_transformer_is_available()) {
        printf("  MPS not available, skipping transformer forward pass test\n");
        TEST_PASS();
    }
    
    MPSTransformerConfig config;
    mps_transformer_config_create_default(&config, "bert", 1000, 16, 32, 2, 2);
    config.batch_size = 2;
    
    MPSTransformerContext* context = NULL;
    MPSTransformerError error = mps_transformer_create(&context, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Context creation should succeed");
    
    // Create and load weights
    MPSTransformerWeights* weights = NULL;
    error = mps_transformer_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights creation should succeed");
    
    error = mps_transformer_weights_init_random(weights, &config, 11111);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights initialization should succeed");
    
    error = mps_transformer_load_weights(context, weights);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Loading weights should succeed");
    
    // Create tensors
    MPSTransformerTensors* tensors = NULL;
    error = mps_transformer_tensors_create(&tensors, &config, 2, 8);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Tensors creation should succeed");
    
    // Allocate tensor memory
    size_t token_size = tensors->batch_size * tensors->sequence_length;
    size_t hidden_states_size = token_size * config.hidden_size;
    size_t logits_size = token_size * config.vocab_size;
    
    tensors->input_ids = malloc(token_size * sizeof(uint32_t));
    tensors->embeddings = malloc(hidden_states_size * sizeof(float));
    tensors->hidden_states = malloc(hidden_states_size * sizeof(float));
    tensors->logits = malloc(logits_size * sizeof(float));
    tensors->owns_memory = true;
    
    TEST_ASSERT(tensors->input_ids && tensors->embeddings && 
               tensors->hidden_states && tensors->logits, "Tensor memory allocation should succeed");
    
    // Initialize tensors
    initialize_test_tokens(tensors->input_ids, token_size, config.vocab_size);
    initialize_test_floats(tensors->embeddings, hidden_states_size, 0.1f);
    memset(tensors->hidden_states, 0, hidden_states_size * sizeof(float));
    memset(tensors->logits, 0, logits_size * sizeof(float));
    
    // Perform forward pass
    error = mps_transformer_forward(context, tensors);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Forward pass should succeed");
    
    // Verify outputs
    bool has_hidden_states = false;
    bool has_logits = false;
    
    for (size_t i = 0; i < hidden_states_size && !has_hidden_states; i++) {
        if (fabsf(tensors->hidden_states[i]) > FLOAT_TOLERANCE) {
            has_hidden_states = true;
        }
    }
    
    for (size_t i = 0; i < logits_size && !has_logits; i++) {
        if (fabsf(tensors->logits[i]) > FLOAT_TOLERANCE) {
            has_logits = true;
        }
    }
    
    TEST_ASSERT(has_hidden_states, "Hidden states should be computed");
    TEST_ASSERT(has_logits, "Logits should be computed");
    
    printf("  Transformer forward pass completed successfully\n");
    printf("  Hidden states sample: [%.3f, %.3f, %.3f, ...]\n", 
           tensors->hidden_states[0], tensors->hidden_states[1], tensors->hidden_states[2]);
    printf("  Logits sample: [%.3f, %.3f, %.3f, ...]\n", 
           tensors->logits[0], tensors->logits[1], tensors->logits[2]);
    
    // Cleanup
    mps_transformer_tensors_destroy(tensors);
    mps_transformer_weights_destroy(weights);
    mps_transformer_destroy(context);
    
    TEST_PASS();
}

int test_performance_statistics() {
    if (!mps_transformer_is_available()) {
        printf("  MPS not available, skipping performance test\n");
        TEST_PASS();
    }
    
    MPSTransformerConfig config;
    mps_transformer_config_create_default(&config, "gpt", 1000, 32, 64, 2, 2);
    config.batch_size = 2;
    
    MPSTransformerContext* context = NULL;
    MPSTransformerError error = mps_transformer_create(&context, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Context creation should succeed");
    
    // Get initial statistics
    MPSTransformerStats initial_stats;
    error = mps_transformer_get_stats(context, &initial_stats);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Getting stats should succeed");
    TEST_ASSERT(initial_stats.total_forward_passes == 0, "Initial forward passes should be 0");
    TEST_ASSERT(initial_stats.total_compute_time_ns == 0, "Initial compute time should be 0");
    
    // Create weights and perform operations
    MPSTransformerWeights* weights = NULL;
    error = mps_transformer_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights creation should succeed");
    
    error = mps_transformer_weights_init_random(weights, &config, 22222);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights initialization should succeed");
    
    error = mps_transformer_load_weights(context, weights);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Loading weights should succeed");
    
    // Prepare test data
    uint32_t batch_size = 2;
    uint32_t sequence_length = 8;
    size_t token_size = batch_size * sequence_length;
    size_t logits_size = token_size * config.vocab_size;
    
    uint32_t* input_ids = malloc(token_size * sizeof(uint32_t));
    float* logits = malloc(logits_size * sizeof(float));
    
    initialize_test_tokens(input_ids, token_size, config.vocab_size);
    
    // Perform multiple operations
    for (int i = 0; i < 3; i++) {
        error = mps_transformer_generate_logits(context, input_ids, NULL, logits);
        TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Generation should succeed");
    }
    
    // Get updated statistics
    MPSTransformerStats final_stats;
    error = mps_transformer_get_stats(context, &final_stats);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Getting final stats should succeed");
    TEST_ASSERT(final_stats.total_forward_passes == 3, "Should have 3 forward passes");
    TEST_ASSERT(final_stats.total_compute_time_ns > 0, "Should have non-zero compute time");
    TEST_ASSERT(final_stats.average_forward_time_ms >= 0.0f, "Average forward time should be non-negative");
    
    printf("  Performance stats after 3 operations:\n");
    printf("    Total forward passes: %llu\n", (unsigned long long)final_stats.total_forward_passes);
    printf("    Total compute time: %llu ns\n", final_stats.total_compute_time_ns);
    printf("    Average forward time: %.3f ms\n", final_stats.average_forward_time_ms);
    printf("    Tokens per second: %.1f\n", final_stats.tokens_per_second);
    printf("    Memory usage: %u MB\n", final_stats.memory_usage_mb);
    printf("    Peak memory: %u MB\n", final_stats.peak_memory_usage_mb);
    
    // Test statistics reset
    error = mps_transformer_reset_stats(context);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Stats reset should succeed");
    
    MPSTransformerStats reset_stats;
    error = mps_transformer_get_stats(context, &reset_stats);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Getting reset stats should succeed");
    TEST_ASSERT(reset_stats.total_forward_passes == 0, "Reset forward passes should be 0");
    TEST_ASSERT(reset_stats.total_compute_time_ns == 0, "Reset compute time should be 0");
    
    printf("  Statistics reset successfully\n");
    
    // Cleanup
    free(input_ids);
    free(logits);
    mps_transformer_weights_destroy(weights);
    mps_transformer_destroy(context);
    
    TEST_PASS();
}

int test_different_model_types() {
    if (!mps_transformer_is_available()) {
        printf("  MPS not available, skipping model types test\n");
        TEST_PASS();
    }
    
    // Test configurations for different model types
    const char* model_types[] = {"bert", "gpt", "t5", "custom"};
    
    for (int t = 0; t < 4; t++) {
        MPSTransformerConfig config;
        MPSTransformerError error = mps_transformer_config_create_default(&config, model_types[t], 
                                                                         1000, 32, 64, 2, 2);
        TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Config creation should succeed for all types");
        
        error = mps_transformer_config_validate(&config);
        TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Config validation should succeed for all types");
        
        printf("  %s model type validated\n", model_types[t]);
        
        // Quick context creation test
        MPSTransformerContext* context = NULL;
        error = mps_transformer_create(&context, &config);
        if (error == MPS_TRANSFORMER_SUCCESS) {
            printf("    %s context created successfully\n", model_types[t]);
            mps_transformer_destroy(context);
        }
    }
    
    TEST_PASS();
}

int test_memory_management() {
    if (!mps_transformer_is_available()) {
        printf("  MPS not available, skipping memory management test\n");
        TEST_PASS();
    }
    
    MPSTransformerConfig config;
    mps_transformer_config_create_default(&config, "bert", 2000, 64, 128, 3, 4);
    config.batch_size = 4;
    
    MPSTransformerContext* context = NULL;
    MPSTransformerError error = mps_transformer_create(&context, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Context creation should succeed");
    
    // Check initial memory usage
    uint32_t initial_memory;
    error = mps_transformer_get_memory_usage(context, &initial_memory);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Getting memory usage should succeed");
    printf("  Initial memory usage: %u MB\n", initial_memory);
    
    // Create weights and check memory increase
    MPSTransformerWeights* weights = NULL;
    error = mps_transformer_weights_create(&weights, &config);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Weights creation should succeed");
    
    error = mps_transformer_load_weights(context, weights);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Loading weights should succeed");
    
    uint32_t after_weights_memory;
    error = mps_transformer_get_memory_usage(context, &after_weights_memory);
    TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Getting memory usage should succeed");
    printf("  Memory usage after loading weights: %u MB\n", after_weights_memory);
    
    // Perform operations and check peak memory
    uint32_t* input_ids = malloc(config.batch_size * 32 * sizeof(uint32_t));
    float* hidden_states = malloc(config.batch_size * 32 * config.hidden_size * sizeof(float));
    
    if (input_ids && hidden_states) {
        initialize_test_tokens(input_ids, config.batch_size * 32, config.vocab_size);
        
        error = mps_transformer_encode(context, input_ids, NULL, hidden_states);
        TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Encoding should succeed");
        
        MPSTransformerStats stats;
        error = mps_transformer_get_stats(context, &stats);
        TEST_ASSERT(error == MPS_TRANSFORMER_SUCCESS, "Getting stats should succeed");
        
        printf("  Peak memory usage: %u MB\n", stats.peak_memory_usage_mb);
        TEST_ASSERT(stats.peak_memory_usage_mb >= after_weights_memory, 
                   "Peak memory should be at least as much as after loading weights");
    }
    
    // Cleanup
    free(input_ids);
    free(hidden_states);
    mps_transformer_weights_destroy(weights);
    mps_transformer_destroy(context);
    
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
    printf("MPS Transformer Engine Test Suite\n");
    printf("==================================\n\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    total_tests++; passed_tests += run_test("Transformer Availability", test_transformer_availability);
    total_tests++; passed_tests += run_test("Config Creation and Validation", test_config_creation_and_validation);
    total_tests++; passed_tests += run_test("Memory Requirements Calculation", test_memory_requirements_calculation);
    total_tests++; passed_tests += run_test("Error String Functions", test_error_string_functions);
    total_tests++; passed_tests += run_test("Context Creation and Destruction", test_context_creation_and_destruction);
    total_tests++; passed_tests += run_test("Weights Creation and Initialization", test_weights_creation_and_initialization);
    total_tests++; passed_tests += run_test("Tensors Creation and Validation", test_tensors_creation_and_validation);
    total_tests++; passed_tests += run_test("BERT Encoder Basic", test_bert_encoder_basic);
    total_tests++; passed_tests += run_test("GPT Generation Basic", test_gpt_generation_basic);
    total_tests++; passed_tests += run_test("Transformer Forward Pass", test_transformer_forward_pass);
    total_tests++; passed_tests += run_test("Performance Statistics", test_performance_statistics);
    total_tests++; passed_tests += run_test("Different Model Types", test_different_model_types);
    total_tests++; passed_tests += run_test("Memory Management", test_memory_management);
    
    // Summary
    printf("Test Results\n");
    printf("============\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! MPS Transformer engine is ready.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the implementation.\n");
        return 1;
    }
}
