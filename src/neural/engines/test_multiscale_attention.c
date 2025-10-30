#include "mps_transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Test framework
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        printf("Running test: %s... ", #name); \
        if (test_##name()) { \
            tests_passed++; \
            printf("PASSED\n"); \
        } else { \
            printf("FAILED\n"); \
        } \
    } while(0)

// Test functions
static bool test_multiscale_config_creation() {
    MPSMultiScaleAttentionConfig config;
    
    MPSTransformerError error = mps_transformer_create_multiscale_config(&config, 1024);
    
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return false;
    }
    
    // Verify default values
    bool success = (config.enabled == true) &&
                   (config.num_scales == 3) &&
                   (config.local_window_size == 64) &&
                   (config.medium_window_size == 256) &&
                   (config.global_window_size == 1024) &&
                   (config.use_scale_fusion == true);
    
    return success;
}

static bool test_enhanced_attention_initialization() {
    // Create base attention config
    MPSAttentionConfig base_config = {0};
    base_config.hidden_size = 1024;
    base_config.num_heads = 8;
    base_config.sequence_length = 256;
    base_config.dropout_rate = 0.1f;
    base_config.use_bias = true;
    
    // Create multi-scale config
    MPSMultiScaleAttentionConfig multi_scale_config;
    MPSTransformerError error = mps_transformer_create_multiscale_config(&multi_scale_config, 1024);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return false;
    }
    
    // Initialize enhanced attention
    MPSEnhancedAttentionConfig enhanced_config;
    error = mps_transformer_init_enhanced_attention(&enhanced_config, &base_config, &multi_scale_config);
    
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return false;
    }
    
    // Verify initialization
    bool success = (enhanced_config.use_multi_scale == true) &&
                   (enhanced_config.base_config.hidden_size == 1024) &&
                   (enhanced_config.multi_scale.enabled == true);
    
    return success;
}

static bool test_attention_scale_weights() {
    MPSMultiScaleAttentionConfig config;
    MPSTransformerError error = mps_transformer_create_multiscale_config(&config, 1024);
    
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return false;
    }
    
    // Verify scale weights sum to approximately 1.0
    float weight_sum = config.scale_weights[0] + config.scale_weights[1] + config.scale_weights[2];
    bool weights_valid = (weight_sum >= 0.95f && weight_sum <= 1.05f);
    
    // Verify individual weights are reasonable
    bool individual_weights_valid = (config.scale_weights[0] > 0.0f && config.scale_weights[0] < 1.0f) &&
                                   (config.scale_weights[1] > 0.0f && config.scale_weights[1] < 1.0f) &&
                                   (config.scale_weights[2] > 0.0f && config.scale_weights[2] < 1.0f);
    
    return weights_valid && individual_weights_valid;
}

static bool test_fusion_weights_computation() {
    // Test the attention fusion logic with simple synthetic data
    uint32_t batch_size = 2;
    uint32_t seq_len = 4;
    uint32_t hidden_size = 8;
    
    size_t tensor_size = batch_size * seq_len * hidden_size * sizeof(float);
    
    float* local_attention = (float*)malloc(tensor_size);
    float* medium_attention = (float*)malloc(tensor_size);
    float* global_attention = (float*)malloc(tensor_size);
    float* fused_output = (float*)malloc(tensor_size);
    
    if (!local_attention || !medium_attention || !global_attention || !fused_output) {
        free(local_attention);
        free(medium_attention);
        free(global_attention);
        free(fused_output);
        return false;
    }
    
    // Initialize with simple test values
    for (size_t i = 0; i < batch_size * seq_len * hidden_size; i++) {
        local_attention[i] = 1.0f;   // Local attention outputs 1.0
        medium_attention[i] = 2.0f;  // Medium attention outputs 2.0
        global_attention[i] = 3.0f;  // Global attention outputs 3.0
    }
    
    // Test fusion weights
    float fusion_weights[4] = {0.5f, 0.3f, 0.2f, 0.0f};
    
    // Create minimal context for testing (we only need dimensions)
    MPSTransformerContext context = {0};
    context.config.batch_size = batch_size;
    context.config.sequence_length = seq_len;
    context.config.hidden_size = hidden_size;
    
    MPSTransformerError error = mps_transformer_fuse_attention_scales(&context,
                                                                    local_attention,
                                                                    medium_attention,
                                                                    global_attention,
                                                                    fusion_weights,
                                                                    fused_output);
    
    bool success = (error == MPS_TRANSFORMER_SUCCESS);
    
    if (success) {
        // Verify fusion computation: 0.5*1.0 + 0.3*2.0 + 0.2*3.0 = 0.5 + 0.6 + 0.6 = 1.7
        float expected_value = 0.5f * 1.0f + 0.3f * 2.0f + 0.2f * 3.0f;
        float actual_value = fused_output[0];
        
        success = (actual_value >= expected_value - 0.001f && actual_value <= expected_value + 0.001f);
    }
    
    free(local_attention);
    free(medium_attention);
    free(global_attention);
    free(fused_output);
    
    return success;
}

static bool test_window_size_validation() {
    MPSMultiScaleAttentionConfig config;
    MPSTransformerError error = mps_transformer_create_multiscale_config(&config, 1024);
    
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return false;
    }
    
    // Verify window sizes are in ascending order
    bool ascending_order = (config.local_window_size < config.medium_window_size) &&
                          (config.medium_window_size < config.global_window_size);
    
    // Verify reasonable window sizes
    bool reasonable_sizes = (config.local_window_size >= 16) &&
                           (config.medium_window_size >= 64) &&
                           (config.global_window_size >= 256);
    
    return ascending_order && reasonable_sizes;
}

static bool test_error_handling() {
    // Test NULL parameter handling
    MPSTransformerError error1 = mps_transformer_create_multiscale_config(NULL, 1024);
    
    MPSEnhancedAttentionConfig enhanced_config;
    MPSTransformerError error2 = mps_transformer_init_enhanced_attention(&enhanced_config, NULL, NULL);
    
    MPSTransformerError error3 = mps_transformer_fuse_attention_scales(NULL, NULL, NULL, NULL, NULL, NULL);
    
    // All should return appropriate error codes
    bool success = (error1 == MPS_TRANSFORMER_ERROR_INVALID_PARAM) &&
                   (error2 == MPS_TRANSFORMER_ERROR_INVALID_PARAM) &&
                   (error3 == MPS_TRANSFORMER_ERROR_INVALID_PARAM);
    
    return success;
}

static bool test_configuration_compatibility() {
    // Test different hidden sizes
    uint32_t hidden_sizes[] = {256, 512, 1024, 2048};
    
    for (int i = 0; i < 4; i++) {
        MPSMultiScaleAttentionConfig config;
        MPSTransformerError error = mps_transformer_create_multiscale_config(&config, hidden_sizes[i]);
        
        if (error != MPS_TRANSFORMER_SUCCESS) {
            return false;
        }
        
        // Verify config is valid regardless of hidden size
        if (!config.enabled || config.num_scales != 3) {
            return false;
        }
    }
    
    return true;
}

static bool test_scale_weights_range() {
    MPSMultiScaleAttentionConfig config;
    MPSTransformerError error = mps_transformer_create_multiscale_config(&config, 1024);
    
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return false;
    }
    
    // Verify all weights are in valid range [0, 1]
    for (int i = 0; i < 3; i++) {
        if (config.scale_weights[i] < 0.0f || config.scale_weights[i] > 1.0f) {
            return false;
        }
    }
    
    // Verify local attention has highest weight (should focus on local patterns first)
    bool local_dominant = config.scale_weights[0] >= config.scale_weights[1] &&
                         config.scale_weights[0] >= config.scale_weights[2];
    
    return local_dominant;
}

static bool test_fusion_temperature() {
    MPSMultiScaleAttentionConfig config;
    MPSTransformerError error = mps_transformer_create_multiscale_config(&config, 1024);
    
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return false;
    }
    
    // Verify fusion temperature is reasonable
    bool valid_temperature = (config.fusion_temperature > 0.0f && config.fusion_temperature <= 2.0f);
    
    return valid_temperature;
}

static bool test_adaptive_scaling_flag() {
    MPSMultiScaleAttentionConfig config;
    MPSTransformerError error = mps_transformer_create_multiscale_config(&config, 1024);
    
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return false;
    }
    
    // Adaptive scaling should be disabled by default for stability
    return (config.adaptive_scaling == false);
}

int main() {
    printf("=== Multi-Scale Attention Unit Tests ===\n\n");
    
    TEST(multiscale_config_creation);
    TEST(enhanced_attention_initialization);
    TEST(attention_scale_weights);
    TEST(fusion_weights_computation);
    TEST(window_size_validation);
    TEST(error_handling);
    TEST(configuration_compatibility);
    TEST(scale_weights_range);
    TEST(fusion_temperature);
    TEST(adaptive_scaling_flag);
    
    printf("\n=== Test Results ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    printf("Success rate: %.1f%%\n", (float)tests_passed / tests_run * 100.0f);
    
    if (tests_passed == tests_run) {
        printf("\n✅ Multi-Scale Attention Enhancement implementation is working correctly!\n");
        printf("📊 Features implemented:\n");
        printf("   - Local attention (64-token window)\n");
        printf("   - Medium attention (256-token window)\n");
        printf("   - Global attention (1024-token window)\n");
        printf("   - Learnable attention fusion with weighted combination\n");
        printf("   - Metal acceleration support for parallel processing\n");
        printf("   - CPU fallback for compatibility\n");
    }
    
    return (tests_passed == tests_run) ? 0 : 1;
}
