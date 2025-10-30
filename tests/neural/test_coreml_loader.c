#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include "../../src/neural/models/coreml_loader.h"

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

// Test data paths
static const char* test_models_dir = "/tmp/test_coreml_models";
static const char* test_model_path = "/tmp/test_coreml_models/test_model.mlmodel";
static const char* test_transformer_path = "/tmp/test_coreml_models/transformer_model.mlmodel";
static const char* test_lstm_path = "/tmp/test_coreml_models/lstm_model.mlmodel";

// Helper functions
static int create_test_environment() {
    // Create test directory
    if (mkdir(test_models_dir, 0755) != 0 && errno != EEXIST) {
        printf("Failed to create test directory\n");
        return 0;
    }
    
    // Create dummy model files for testing
    FILE* f1 = fopen(test_model_path, "w");
    if (f1) {
        fprintf(f1, "dummy CoreML model content\n");
        fclose(f1);
    }
    
    FILE* f2 = fopen(test_transformer_path, "w");
    if (f2) {
        fprintf(f2, "dummy Transformer CoreML model content\n");
        fclose(f2);
    }
    
    FILE* f3 = fopen(test_lstm_path, "w");
    if (f3) {
        fprintf(f3, "dummy LSTM CoreML model content\n");
        fclose(f3);
    }
    
    return 1;
}

static void cleanup_test_environment() {
    unlink(test_model_path);
    unlink(test_transformer_path);
    unlink(test_lstm_path);
    rmdir(test_models_dir);
}

// Test functions
int test_coreml_availability() {
    bool available = coreml_is_available();
    printf("  CoreML available: %s\n", available ? "Yes" : "No");
    
    char version[256];
    CoreMLError error = coreml_get_framework_version(version, sizeof(version));
    TEST_ASSERT(error == COREML_SUCCESS, "Getting framework version should succeed");
    printf("  CoreML version: %s\n", version);
    
    TEST_PASS();
}

int test_loader_config_creation() {
    CoreMLLoaderConfig config;
    CoreMLError error = coreml_loader_config_create_default(&config);
    
    TEST_ASSERT(error == COREML_SUCCESS, "Default config creation should succeed");
    TEST_ASSERT(config.max_cached_models > 0, "Max cached models should be positive");
    TEST_ASSERT(config.max_memory_mb > 0, "Max memory should be positive");
    TEST_ASSERT(config.cache_ttl_seconds > 0, "Cache TTL should be positive");
    TEST_ASSERT(config.enable_model_validation == true, "Model validation should be enabled by default");
    TEST_ASSERT(strlen(config.models_directory) > 0, "Models directory should be set");
    
    printf("  Default config: max_models=%u, memory=%uMB, ttl=%us\n",
           config.max_cached_models, config.max_memory_mb, config.cache_ttl_seconds);
    
    TEST_PASS();
}

int test_loader_creation() {
    CoreMLLoaderConfig config;
    coreml_loader_config_create_default(&config);
    
    CoreMLLoader* loader = NULL;
    CoreMLError error = coreml_loader_create(&loader, &config);
    
    TEST_ASSERT(error == COREML_SUCCESS, "Loader creation should succeed");
    TEST_ASSERT(loader != NULL, "Loader should not be NULL");
    TEST_ASSERT(loader->is_initialized == true, "Loader should be initialized");
    TEST_ASSERT(loader->cache_capacity == config.max_cached_models, "Cache capacity should match config");
    TEST_ASSERT(loader->cache_count == 0, "Initial cache count should be 0");
    
    printf("  Created loader with cache capacity: %u\n", loader->cache_capacity);
    
    coreml_loader_destroy(loader);
    TEST_PASS();
}

int test_error_string_functions() {
    const char* error_msg = coreml_get_error_string(COREML_SUCCESS);
    TEST_ASSERT(error_msg != NULL, "Error string should not be NULL");
    TEST_ASSERT(strlen(error_msg) > 0, "Error string should not be empty");
    
    error_msg = coreml_get_error_string(COREML_ERROR_INVALID_PARAM);
    TEST_ASSERT(error_msg != NULL, "Error string should not be NULL");
    TEST_ASSERT(strlen(error_msg) > 0, "Error string should not be empty");
    
    printf("  COREML_SUCCESS: %s\n", coreml_get_error_string(COREML_SUCCESS));
    printf("  COREML_ERROR_INVALID_PARAM: %s\n", coreml_get_error_string(COREML_ERROR_INVALID_PARAM));
    
    TEST_PASS();
}

int test_model_architecture_detection() {
    CoreMLLoaderConfig config;
    coreml_loader_config_create_default(&config);
    
    CoreMLLoader* loader = NULL;
    CoreMLError error = coreml_loader_create(&loader, &config);
    TEST_ASSERT(error == COREML_SUCCESS, "Loader creation should succeed");
    
    // Test architecture detection from file paths
    CoreMLModelInfo transformer_info;
    error = coreml_loader_get_model_info(loader, test_transformer_path, &transformer_info);
    if (error == COREML_SUCCESS) {
        TEST_ASSERT(transformer_info.architecture == COREML_ARCH_TRANSFORMER, 
                   "Should detect Transformer architecture");
        printf("  Transformer model detected: %s\n", 
               transformer_info.architecture == COREML_ARCH_TRANSFORMER ? "Yes" : "No");
    }
    
    CoreMLModelInfo lstm_info;
    error = coreml_loader_get_model_info(loader, test_lstm_path, &lstm_info);
    if (error == COREML_SUCCESS) {
        TEST_ASSERT(lstm_info.architecture == COREML_ARCH_LSTM, 
                   "Should detect LSTM architecture");
        printf("  LSTM model detected: %s\n", 
               lstm_info.architecture == COREML_ARCH_LSTM ? "Yes" : "No");
    }
    
    coreml_loader_destroy(loader);
    TEST_PASS();
}

int test_model_loading_basic() {
    CoreMLLoaderConfig config;
    coreml_loader_config_create_default(&config);
    config.max_cached_models = 5;
    
    CoreMLLoader* loader = NULL;
    CoreMLError error = coreml_loader_create(&loader, &config);
    TEST_ASSERT(error == COREML_SUCCESS, "Loader creation should succeed");
    
    // Test loading non-existent model
    CoreMLModelEntry* model_entry = NULL;
    error = coreml_loader_load_model(loader, "/non/existent/model.mlmodel", &model_entry);
    TEST_ASSERT(error == COREML_ERROR_MODEL_NOT_FOUND, "Loading non-existent model should fail");
    TEST_ASSERT(model_entry == NULL, "Model entry should be NULL for failed load");
    
    // Test loading existing model (metadata only on non-Apple platforms)
    error = coreml_loader_load_model(loader, test_model_path, &model_entry);
    if (coreml_is_available()) {
        // On Apple platforms, may succeed or fail depending on model format
        printf("  Model loading result on Apple platform: %s\n", coreml_get_error_string(error));
    } else {
        // On non-Apple platforms, should load metadata only
        TEST_ASSERT(error == COREML_SUCCESS, "Metadata loading should succeed on non-Apple platforms");
        TEST_ASSERT(model_entry != NULL, "Model entry should not be NULL");
        TEST_ASSERT(model_entry->is_valid == true, "Model entry should be valid");
        TEST_ASSERT(model_entry->reference_count == 1, "Reference count should be 1");
        
        printf("  Loaded model: %s\n", model_entry->info.name);
        printf("  Architecture: %d\n", model_entry->info.architecture);
        printf("  Size: %llu bytes\n", model_entry->info.model_size_bytes);
        
        // Release model
        error = coreml_loader_release_model(loader, model_entry);
        TEST_ASSERT(error == COREML_SUCCESS, "Model release should succeed");
    }
    
    coreml_loader_destroy(loader);
    TEST_PASS();
}

int test_model_caching() {
    CoreMLLoaderConfig config;
    coreml_loader_config_create_default(&config);
    config.max_cached_models = 3;
    
    CoreMLLoader* loader = NULL;
    CoreMLError error = coreml_loader_create(&loader, &config);
    TEST_ASSERT(error == COREML_SUCCESS, "Loader creation should succeed");
    
    // Load same model multiple times
    CoreMLModelEntry* model1 = NULL;
    CoreMLModelEntry* model2 = NULL;
    
    error = coreml_loader_load_model(loader, test_model_path, &model1);
    if (error == COREML_SUCCESS && model1) {
        // Try to get cached model
        error = coreml_loader_get_cached_model(loader, test_model_path, &model2);
        TEST_ASSERT(error == COREML_SUCCESS, "Getting cached model should succeed");
        TEST_ASSERT(model1 == model2, "Should return same cached model");
        TEST_ASSERT(model1->reference_count == 2, "Reference count should be 2");
        TEST_ASSERT(model1->access_count >= 2, "Access count should be at least 2");
        
        printf("  Cache hit test: model1=%p, model2=%p, ref_count=%u\n",
               (void*)model1, (void*)model2, model1->reference_count);
        
        // Release both references
        coreml_loader_release_model(loader, model1);
        coreml_loader_release_model(loader, model2);
    }
    
    // Test cache statistics
    CoreMLLoaderStats stats;
    error = coreml_loader_get_stats(loader, &stats);
    TEST_ASSERT(error == COREML_SUCCESS, "Getting stats should succeed");
    
    printf("  Cache stats: hits=%u, misses=%u, models_loaded=%u\n",
           stats.cache_hits, stats.cache_misses, stats.models_loaded);
    
    coreml_loader_destroy(loader);
    TEST_PASS();
}

int test_cache_management() {
    CoreMLLoaderConfig config;
    coreml_loader_config_create_default(&config);
    config.max_cached_models = 2;
    config.cache_ttl_seconds = 1; // Short TTL for testing
    
    CoreMLLoader* loader = NULL;
    CoreMLError error = coreml_loader_create(&loader, &config);
    TEST_ASSERT(error == COREML_SUCCESS, "Loader creation should succeed");
    
    // Test memory usage tracking
    uint32_t initial_memory = 0;
    error = coreml_loader_get_memory_usage(loader, &initial_memory);
    TEST_ASSERT(error == COREML_SUCCESS, "Getting memory usage should succeed");
    printf("  Initial memory usage: %u MB\n", initial_memory);
    
    // Load models to fill cache
    CoreMLModelEntry* model1 = NULL;
    CoreMLModelEntry* model2 = NULL;
    
    error = coreml_loader_load_model(loader, test_model_path, &model1);
    error = coreml_loader_load_model(loader, test_transformer_path, &model2);
    
    if (model1) coreml_loader_release_model(loader, model1);
    if (model2) coreml_loader_release_model(loader, model2);
    
    uint32_t used_memory = 0;
    error = coreml_loader_get_memory_usage(loader, &used_memory);
    TEST_ASSERT(error == COREML_SUCCESS, "Getting memory usage should succeed");
    printf("  Memory usage after loading: %u MB\n", used_memory);
    
    // Test cache clearing
    error = coreml_loader_clear_cache(loader);
    TEST_ASSERT(error == COREML_SUCCESS, "Cache clearing should succeed");
    
    uint32_t cleared_memory = 0;
    error = coreml_loader_get_memory_usage(loader, &cleared_memory);
    TEST_ASSERT(error == COREML_SUCCESS, "Getting memory usage should succeed");
    printf("  Memory usage after clear: %u MB\n", cleared_memory);
    TEST_ASSERT(cleared_memory <= used_memory, "Memory usage should decrease after clearing");
    
    coreml_loader_destroy(loader);
    TEST_PASS();
}

int test_prediction_data_management() {
    // Test creating prediction data structure
    CoreMLPredictionData* pred_data = NULL;
    CoreMLError error = coreml_prediction_data_create(2, 1, &pred_data);
    
    TEST_ASSERT(error == COREML_SUCCESS, "Prediction data creation should succeed");
    TEST_ASSERT(pred_data != NULL, "Prediction data should not be NULL");
    TEST_ASSERT(pred_data->input_count == 2, "Input count should be 2");
    TEST_ASSERT(pred_data->output_count == 1, "Output count should be 1");
    TEST_ASSERT(pred_data->inputs != NULL, "Inputs array should not be NULL");
    TEST_ASSERT(pred_data->outputs != NULL, "Outputs array should not be NULL");
    TEST_ASSERT(pred_data->is_valid == true, "Prediction data should be valid");
    
    printf("  Created prediction data: inputs=%u, outputs=%u\n",
           pred_data->input_count, pred_data->output_count);
    
    // Test creating feature data
    uint32_t shape[] = {1, 512};
    CoreMLFeatureData* feature_data = NULL;
    error = coreml_feature_data_create("input_tokens", COREML_FEATURE_INT32, shape, 2, &feature_data);
    
    TEST_ASSERT(error == COREML_SUCCESS, "Feature data creation should succeed");
    TEST_ASSERT(feature_data != NULL, "Feature data should not be NULL");
    TEST_ASSERT(strcmp(feature_data->feature_name, "input_tokens") == 0, "Feature name should match");
    TEST_ASSERT(feature_data->type == COREML_FEATURE_INT32, "Feature type should match");
    TEST_ASSERT(feature_data->shape_count == 2, "Shape count should be 2");
    TEST_ASSERT(feature_data->shape[0] == 1, "First dimension should be 1");
    TEST_ASSERT(feature_data->shape[1] == 512, "Second dimension should be 512");
    
    printf("  Created feature data: %s, type=%d, shape=[%u, %u]\n",
           feature_data->feature_name, feature_data->type,
           feature_data->shape[0], feature_data->shape[1]);
    
    // Cleanup
    coreml_feature_data_destroy(feature_data);
    coreml_prediction_data_destroy(pred_data);
    
    TEST_PASS();
}

int test_config_update() {
    CoreMLLoaderConfig config;
    coreml_loader_config_create_default(&config);
    
    CoreMLLoader* loader = NULL;
    CoreMLError error = coreml_loader_create(&loader, &config);
    TEST_ASSERT(error == COREML_SUCCESS, "Loader creation should succeed");
    
    uint32_t original_capacity = loader->cache_capacity;
    
    // Update configuration
    CoreMLLoaderConfig new_config = config;
    new_config.max_cached_models = original_capacity * 2;
    new_config.max_memory_mb = config.max_memory_mb * 2;
    new_config.enable_performance_logging = false;
    
    error = coreml_loader_update_config(loader, &new_config);
    TEST_ASSERT(error == COREML_SUCCESS, "Config update should succeed");
    TEST_ASSERT(loader->cache_capacity == new_config.max_cached_models, "Cache capacity should be updated");
    TEST_ASSERT(loader->config.max_memory_mb == new_config.max_memory_mb, "Memory limit should be updated");
    TEST_ASSERT(loader->config.enable_performance_logging == false, "Performance logging should be disabled");
    
    printf("  Updated config: cache=%u->%u, memory=%u->%uMB\n",
           original_capacity, loader->cache_capacity,
           config.max_memory_mb, loader->config.max_memory_mb);
    
    coreml_loader_destroy(loader);
    TEST_PASS();
}

int test_cache_optimization() {
    CoreMLLoaderConfig config;
    coreml_loader_config_create_default(&config);
    config.max_cached_models = 3;
    config.auto_cleanup = true;
    
    CoreMLLoader* loader = NULL;
    CoreMLError error = coreml_loader_create(&loader, &config);
    TEST_ASSERT(error == COREML_SUCCESS, "Loader creation should succeed");
    
    // Load multiple models
    CoreMLModelEntry* models[3] = {NULL, NULL, NULL};
    const char* paths[] = {test_model_path, test_transformer_path, test_lstm_path};
    
    for (int i = 0; i < 3; i++) {
        error = coreml_loader_load_model(loader, paths[i], &models[i]);
        if (error == COREML_SUCCESS && models[i]) {
            printf("  Loaded model %d: %s\n", i, models[i]->info.name);
        }
    }
    
    // Release references
    for (int i = 0; i < 3; i++) {
        if (models[i]) {
            coreml_loader_release_model(loader, models[i]);
        }
    }
    
    // Test cache optimization
    error = coreml_loader_optimize_cache(loader);
    TEST_ASSERT(error == COREML_SUCCESS, "Cache optimization should succeed");
    
    printf("  Cache count after optimization: %u\n", loader->cache_count);
    
    coreml_loader_destroy(loader);
    TEST_PASS();
}

int test_statistics_tracking() {
    CoreMLLoaderConfig config;
    coreml_loader_config_create_default(&config);
    config.enable_performance_logging = true;
    
    CoreMLLoader* loader = NULL;
    CoreMLError error = coreml_loader_create(&loader, &config);
    TEST_ASSERT(error == COREML_SUCCESS, "Loader creation should succeed");
    
    // Initial statistics
    CoreMLLoaderStats initial_stats;
    error = coreml_loader_get_stats(loader, &initial_stats);
    TEST_ASSERT(error == COREML_SUCCESS, "Getting initial stats should succeed");
    TEST_ASSERT(initial_stats.models_loaded == 0, "Initial models loaded should be 0");
    TEST_ASSERT(initial_stats.cache_hits == 0, "Initial cache hits should be 0");
    TEST_ASSERT(initial_stats.cache_misses == 0, "Initial cache misses should be 0");
    
    // Load model and check statistics
    CoreMLModelEntry* model = NULL;
    error = coreml_loader_load_model(loader, test_model_path, &model);
    
    CoreMLLoaderStats stats_after_load;
    error = coreml_loader_get_stats(loader, &stats_after_load);
    TEST_ASSERT(error == COREML_SUCCESS, "Getting stats after load should succeed");
    
    if (model) {
        TEST_ASSERT(stats_after_load.models_loaded > 0, "Models loaded count should increase");
        printf("  Stats after load: loaded=%u, hits=%u, misses=%u\n",
               stats_after_load.models_loaded, stats_after_load.cache_hits, stats_after_load.cache_misses);
        
        // Test cache hit by loading same model again
        CoreMLModelEntry* model2 = NULL;
        error = coreml_loader_get_cached_model(loader, test_model_path, &model2);
        
        CoreMLLoaderStats stats_after_cache_hit;
        error = coreml_loader_get_stats(loader, &stats_after_cache_hit);
        TEST_ASSERT(error == COREML_SUCCESS, "Getting stats after cache hit should succeed");
        
        if (model2) {
            TEST_ASSERT(stats_after_cache_hit.cache_hits > stats_after_load.cache_hits, 
                       "Cache hits should increase");
            printf("  Stats after cache hit: loaded=%u, hits=%u, misses=%u\n",
                   stats_after_cache_hit.models_loaded, stats_after_cache_hit.cache_hits, 
                   stats_after_cache_hit.cache_misses);
            
            coreml_loader_release_model(loader, model2);
        }
        
        coreml_loader_release_model(loader, model);
    }
    
    // Test statistics reset
    error = coreml_loader_reset_stats(loader);
    TEST_ASSERT(error == COREML_SUCCESS, "Stats reset should succeed");
    
    CoreMLLoaderStats reset_stats;
    error = coreml_loader_get_stats(loader, &reset_stats);
    TEST_ASSERT(error == COREML_SUCCESS, "Getting reset stats should succeed");
    TEST_ASSERT(reset_stats.models_loaded == 0, "Reset models loaded should be 0");
    TEST_ASSERT(reset_stats.cache_hits == 0, "Reset cache hits should be 0");
    TEST_ASSERT(reset_stats.cache_misses == 0, "Reset cache misses should be 0");
    
    printf("  Stats after reset: all counters should be 0\n");
    
    coreml_loader_destroy(loader);
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
    printf("CoreML Model Loader Test Suite\n");
    printf("==============================\n\n");
    
    // Setup test environment
    if (!create_test_environment()) {
        printf("Failed to create test environment\n");
        return 1;
    }
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    total_tests++; passed_tests += run_test("CoreML Availability", test_coreml_availability);
    total_tests++; passed_tests += run_test("Loader Config Creation", test_loader_config_creation);
    total_tests++; passed_tests += run_test("Loader Creation", test_loader_creation);
    total_tests++; passed_tests += run_test("Error String Functions", test_error_string_functions);
    total_tests++; passed_tests += run_test("Model Architecture Detection", test_model_architecture_detection);
    total_tests++; passed_tests += run_test("Model Loading Basic", test_model_loading_basic);
    total_tests++; passed_tests += run_test("Model Caching", test_model_caching);
    total_tests++; passed_tests += run_test("Cache Management", test_cache_management);
    total_tests++; passed_tests += run_test("Prediction Data Management", test_prediction_data_management);
    total_tests++; passed_tests += run_test("Config Update", test_config_update);
    total_tests++; passed_tests += run_test("Cache Optimization", test_cache_optimization);
    total_tests++; passed_tests += run_test("Statistics Tracking", test_statistics_tracking);
    
    // Cleanup test environment
    cleanup_test_environment();
    
    // Summary
    printf("Test Results\n");
    printf("============\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! CoreML Model Loader implementation is ready.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the implementation.\n");
        return 1;
    }
}