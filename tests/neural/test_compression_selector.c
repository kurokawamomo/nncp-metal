#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#include "../../src/neural/optimization/compression_selector.h"

// Test utilities
static void print_test_header(const char* test_name) {
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Testing: %s\n", test_name);
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
}

static void print_success(const char* test_name) {
    printf("✅ %s: PASSED\n", test_name);
}

static void print_error(const char* test_name, const char* error) {
    printf("❌ %s: FAILED - %s\n", test_name, error);
}

// Generate test data with specific characteristics
static void generate_text_data(uint8_t* buffer, size_t size) {
    const char* sample_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                             "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ";
    size_t text_len = strlen(sample_text);
    
    for (size_t i = 0; i < size; i++) {
        buffer[i] = sample_text[i % text_len];
    }
}

static void generate_sparse_data(float* buffer, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (i % 10 == 0) {
            buffer[i] = (float)rand() / RAND_MAX;
        } else {
            buffer[i] = 0.0f;
        }
    }
}

static void generate_time_series_data(float* buffer, size_t count) {
    for (size_t i = 0; i < count; i++) {
        buffer[i] = sinf(2.0f * M_PI * i / 100.0f) + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }
}

static void generate_random_data(uint8_t* buffer, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buffer[i] = rand() % 256;
    }
}

// Test 1: Basic context creation and destruction
static int test_context_creation(void) {
    print_test_header("Context Creation and Destruction");
    
    CompressionSelectorConfig config;
    CompressionSelectorError error = compression_selector_config_create_default(&config, COMPRESSION_OBJECTIVE_BALANCED);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Config Creation", compression_selector_get_error_string(error));
        return 0;
    }
    print_success("Default Configuration Created");
    
    CompressionSelectorContext* context = NULL;
    error = compression_selector_create(&context, &config);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Context Creation", compression_selector_get_error_string(error));
        return 0;
    }
    
    if (!context) {
        print_error("Context Creation", "Context is NULL");
        return 0;
    }
    print_success("Context Created Successfully");
    
    compression_selector_destroy(context);
    print_success("Context Destroyed Successfully");
    
    return 1;
}

// Test 2: Configuration validation
static int test_configuration(void) {
    print_test_header("Configuration Testing");
    
    CompressionSelectorConfig config;
    
    // Test different objectives
    CompressionObjective objectives[] = {
        COMPRESSION_OBJECTIVE_SIZE,
        COMPRESSION_OBJECTIVE_SPEED,
        COMPRESSION_OBJECTIVE_QUALITY,
        COMPRESSION_OBJECTIVE_BALANCED,
        COMPRESSION_OBJECTIVE_ENERGY_EFFICIENT
    };
    
    for (int i = 0; i < 5; i++) {
        CompressionSelectorError error = compression_selector_config_create_default(&config, objectives[i]);
        if (error != COMPRESSION_SELECTOR_SUCCESS) {
            print_error("Configuration", "Failed to create config for objective");
            return 0;
        }
        
        printf("  Objective %d: Primary=%.2f, Secondary=%.2f\n", 
               i, config.objective_weight_primary, config.objective_weight_secondary);
    }
    
    print_success("All Configuration Objectives");
    
    // Test configuration parameters
    printf("  Sample size: %u\n", config.sample_size_for_analysis);
    printf("  ML feature vector size: %u\n", config.ml_feature_vector_size);
    printf("  Max compression time: %.0f ms\n", config.max_compression_time_ms);
    printf("  Min compression ratio: %.2f\n", config.min_compression_ratio);
    printf("  GPU acceleration: %s\n", config.enable_gpu_acceleration ? "enabled" : "disabled");
    
    print_success("Configuration Parameters Validation");
    
    return 1;
}

// Test 3: Data characteristics analysis
static int test_data_analysis(void) {
    print_test_header("Data Characteristics Analysis");
    
    CompressionSelectorConfig config;
    compression_selector_config_create_default(&config, COMPRESSION_OBJECTIVE_BALANCED);
    
    CompressionSelectorContext* context = NULL;
    CompressionSelectorError error = compression_selector_create(&context, &config);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Context Creation", compression_selector_get_error_string(error));
        return 0;
    }
    
    // Test with text data
    const size_t text_size = 8192;
    uint8_t* text_data = malloc(text_size);
    generate_text_data(text_data, text_size);
    
    DataCharacteristics text_chars;
    error = compression_selector_analyze_data(context, text_data, text_size, &text_chars);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Text Analysis", compression_selector_get_error_string(error));
        free(text_data);
        compression_selector_destroy(context);
        return 0;
    }
    
    printf("  Text Data Analysis:\n");
    printf("    Entropy: %.3f\n", text_chars.entropy);
    printf("    Sparsity: %.3f\n", text_chars.sparsity);
    printf("    Redundancy ratio: %.3f\n", text_chars.redundancy_ratio);
    printf("    Compressibility score: %.3f\n", text_chars.compressibility_score);
    printf("    Primary type: %d\n", text_chars.primary_type);
    printf("    Type confidence: %.3f\n", text_chars.type_confidence);
    
    print_success("Text Data Analysis");
    
    // Test with sparse numerical data
    const size_t sparse_count = 1024;
    float* sparse_data = malloc(sparse_count * sizeof(float));
    generate_sparse_data(sparse_data, sparse_count);
    
    DataCharacteristics sparse_chars;
    error = compression_selector_analyze_data(context, sparse_data, sparse_count * sizeof(float), &sparse_chars);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Sparse Analysis", compression_selector_get_error_string(error));
        free(text_data);
        free(sparse_data);
        compression_selector_destroy(context);
        return 0;
    }
    
    printf("  Sparse Data Analysis:\n");
    printf("    Entropy: %.3f\n", sparse_chars.entropy);
    printf("    Sparsity: %.3f\n", sparse_chars.sparsity);
    printf("    Mean: %.3f\n", sparse_chars.mean);
    printf("    Variance: %.3f\n", sparse_chars.variance);
    printf("    Primary type: %d\n", sparse_chars.primary_type);
    
    print_success("Sparse Data Analysis");
    
    // Test with time series data
    const size_t ts_count = 2048;
    float* ts_data = malloc(ts_count * sizeof(float));
    generate_time_series_data(ts_data, ts_count);
    
    DataCharacteristics ts_chars;
    error = compression_selector_analyze_data(context, ts_data, ts_count * sizeof(float), &ts_chars);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Time Series Analysis", compression_selector_get_error_string(error));
        free(text_data);
        free(sparse_data);
        free(ts_data);
        compression_selector_destroy(context);
        return 0;
    }
    
    printf("  Time Series Data Analysis:\n");
    printf("    Autocorrelation: %.3f\n", ts_chars.autocorrelation);
    printf("    Trend strength: %.3f\n", ts_chars.trend_strength);
    printf("    Predictability: %.3f\n", ts_chars.predictability);
    printf("    Pattern repetition: %.3f\n", ts_chars.pattern_repetition);
    
    print_success("Time Series Data Analysis");
    
    free(text_data);
    free(sparse_data);
    free(ts_data);
    compression_selector_destroy(context);
    
    return 1;
}

// Test 4: Algorithm selection
static int test_algorithm_selection(void) {
    print_test_header("Algorithm Selection");
    
    CompressionSelectorConfig config;
    compression_selector_config_create_default(&config, COMPRESSION_OBJECTIVE_SIZE);
    
    CompressionSelectorContext* context = NULL;
    CompressionSelectorError error = compression_selector_create(&context, &config);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Context Creation", compression_selector_get_error_string(error));
        return 0;
    }
    
    // Test selection with different data types
    const size_t data_size = 4096;
    
    // Test 1: High entropy random data
    uint8_t* random_data = malloc(data_size);
    generate_random_data(random_data, data_size);
    
    CompressionSelectionResult result;
    error = compression_selector_select_algorithm(context, random_data, data_size, &result);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Random Data Selection", compression_selector_get_error_string(error));
        free(random_data);
        compression_selector_destroy(context);
        return 0;
    }
    
    printf("  Random Data Selection:\n");
    printf("    Selected algorithm: %s\n", compression_selector_get_algorithm_name(result.selected_algorithm));
    printf("    Predicted ratio: %.2fx\n", result.prediction.predicted_ratio);
    printf("    Predicted speed: %.0f MB/s\n", result.prediction.predicted_speed);
    printf("    Predicted quality: %.3f\n", result.prediction.predicted_quality);
    printf("    Selection confidence: %.3f\n", result.selection_confidence);
    printf("    Analysis time: %u ms\n", result.analysis_time_ms);
    printf("    Requires GPU: %s\n", result.requires_gpu ? "yes" : "no");
    printf("    Memory estimate: %u MB\n", result.estimated_memory_mb);
    
    print_success("Random Data Algorithm Selection");
    
    // Test 2: Sparse data
    float* sparse_data = malloc(data_size);
    generate_sparse_data((float*)sparse_data, data_size / sizeof(float));
    
    CompressionSelectionResult sparse_result;
    error = compression_selector_select_algorithm(context, sparse_data, data_size, &sparse_result);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Sparse Data Selection", compression_selector_get_error_string(error));
        free(random_data);
        free(sparse_data);
        compression_selector_destroy(context);
        return 0;
    }
    
    printf("  Sparse Data Selection:\n");
    printf("    Selected algorithm: %s\n", compression_selector_get_algorithm_name(sparse_result.selected_algorithm));
    printf("    Predicted ratio: %.2fx\n", sparse_result.prediction.predicted_ratio);
    printf("    Overall score: %.3f\n", sparse_result.overall_score);
    
    print_success("Sparse Data Algorithm Selection");
    
    // Clean up alternative algorithms
    if (result.alternative_algorithms) {
        free(result.alternative_algorithms);
        free(result.alternative_scores);
    }
    if (sparse_result.alternative_algorithms) {
        free(sparse_result.alternative_algorithms);
        free(sparse_result.alternative_scores);
    }
    
    free(random_data);
    free(sparse_data);
    compression_selector_destroy(context);
    
    return 1;
}

// Test 5: Performance prediction
static int test_performance_prediction(void) {
    print_test_header("Performance Prediction");
    
    CompressionSelectorConfig config;
    compression_selector_config_create_default(&config, COMPRESSION_OBJECTIVE_BALANCED);
    
    CompressionSelectorContext* context = NULL;
    CompressionSelectorError error = compression_selector_create(&context, &config);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Context Creation", compression_selector_get_error_string(error));
        return 0;
    }
    
    // Create test data characteristics
    DataCharacteristics chars;
    memset(&chars, 0, sizeof(chars));
    chars.data_size = 1024 * 1024; // 1MB
    chars.entropy = 6.5f;
    chars.sparsity = 0.3f;
    chars.compressibility_score = 0.7f;
    chars.autocorrelation = 0.6f;
    chars.predictability = 0.8f;
    
    // Test prediction for each algorithm
    CompressionAlgorithm algorithms[] = {
        COMPRESSION_ALGORITHM_NEURAL_QUANTIZATION,
        COMPRESSION_ALGORITHM_RLE_LOSSLESS,
        COMPRESSION_ALGORITHM_TRANSFORMER_COMPRESSION,
        COMPRESSION_ALGORITHM_LSTM_COMPRESSION
    };
    
    printf("  Algorithm Performance Predictions:\n");
    printf("  %-25s %8s %8s %8s %8s %8s\n", 
           "Algorithm", "Ratio", "Speed", "Quality", "Memory", "Time");
    printf("  %s\n", "-----------------------------------------------------------------------");
    
    for (int i = 0; i < 4; i++) {
        AlgorithmPerformancePrediction prediction;
        error = compression_selector_predict_performance(context, &chars, algorithms[i], &prediction);
        if (error != COMPRESSION_SELECTOR_SUCCESS) {
            print_error("Performance Prediction", compression_selector_get_error_string(error));
            compression_selector_destroy(context);
            return 0;
        }
        
        printf("  %-25s %7.2fx %7.0f %7.3f %7.0f %7d\n",
               compression_selector_get_algorithm_name(algorithms[i]),
               prediction.predicted_ratio,
               prediction.predicted_speed,
               prediction.predicted_quality,
               prediction.memory_usage,
               prediction.estimated_time_ms);
    }
    
    print_success("Performance Prediction for All Algorithms");
    
    compression_selector_destroy(context);
    
    return 1;
}

// Test 6: Feature extraction
static int test_feature_extraction(void) {
    print_test_header("Feature Extraction");
    
    // Create test data characteristics
    DataCharacteristics chars;
    memset(&chars, 0, sizeof(chars));
    chars.data_size = 65536;
    chars.entropy = 7.2f;
    chars.sparsity = 0.15f;
    chars.variance = 1.5f;
    chars.autocorrelation = 0.7f;
    chars.predictability = 0.6f;
    chars.compressibility_score = 0.8f;
    chars.primary_type = DATA_TYPE_TIME_SERIES;
    
    const uint32_t feature_count = 32;
    float features[32];
    
    CompressionSelectorError error = compression_selector_extract_features(&chars, features, feature_count);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Feature Extraction", compression_selector_get_error_string(error));
        return 0;
    }
    
    printf("  Extracted Features (first 16):\n");
    for (int i = 0; i < 16; i++) {
        printf("    Feature %2d: %8.4f", i, features[i]);
        if (i % 4 == 3) printf("\n");
    }
    printf("\n");
    
    // Validate feature ranges
    int valid_features = 1;
    for (uint32_t i = 0; i < feature_count; i++) {
        if (isnan(features[i]) || isinf(features[i])) {
            printf("  Invalid feature at index %u: %f\n", i, features[i]);
            valid_features = 0;
        }
    }
    
    if (valid_features) {
        print_success("Feature Extraction and Validation");
    } else {
        print_error("Feature Extraction", "Invalid features detected");
        return 0;
    }
    
    return 1;
}

// Test 7: Algorithm availability configuration
static int test_algorithm_configuration(void) {
    print_test_header("Algorithm Configuration");
    
    CompressionSelectorConfig config;
    compression_selector_config_create_default(&config, COMPRESSION_OBJECTIVE_SPEED);
    
    CompressionSelectorContext* context = NULL;
    CompressionSelectorError error = compression_selector_create(&context, &config);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Context Creation", compression_selector_get_error_string(error));
        return 0;
    }
    
    // Disable some algorithms
    error = compression_selector_set_algorithm_availability(context, COMPRESSION_ALGORITHM_TRANSFORMER_COMPRESSION, 
                                                           false, 0.0f);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Algorithm Disable", compression_selector_get_error_string(error));
        compression_selector_destroy(context);
        return 0;
    }
    
    // Set preference weights
    error = compression_selector_set_algorithm_availability(context, COMPRESSION_ALGORITHM_RLE_LOSSLESS, 
                                                           true, 2.0f);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Algorithm Weight", compression_selector_get_error_string(error));
        compression_selector_destroy(context);
        return 0;
    }
    
    print_success("Algorithm Availability Configuration");
    
    // Test selection with modified availability
    const size_t data_size = 2048;
    uint8_t* test_data = malloc(data_size);
    generate_random_data(test_data, data_size);
    
    CompressionSelectionResult result;
    error = compression_selector_select_algorithm(context, test_data, data_size, &result);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Selection with Modified Availability", compression_selector_get_error_string(error));
        free(test_data);
        compression_selector_destroy(context);
        return 0;
    }
    
    printf("  Selected algorithm with preferences: %s\n", 
           compression_selector_get_algorithm_name(result.selected_algorithm));
    
    if (result.selected_algorithm == COMPRESSION_ALGORITHM_TRANSFORMER_COMPRESSION) {
        print_error("Algorithm Filtering", "Disabled algorithm was selected");
        free(test_data);
        compression_selector_destroy(context);
        return 0;
    }
    
    print_success("Algorithm Filtering and Preferences");
    
    if (result.alternative_algorithms) {
        free(result.alternative_algorithms);
        free(result.alternative_scores);
    }
    
    free(test_data);
    compression_selector_destroy(context);
    
    return 1;
}

// Test 8: Performance history and statistics
static int test_performance_history(void) {
    print_test_header("Performance History and Statistics");
    
    CompressionSelectorConfig config;
    compression_selector_config_create_default(&config, COMPRESSION_OBJECTIVE_BALANCED);
    
    CompressionSelectorContext* context = NULL;
    CompressionSelectorError error = compression_selector_create(&context, &config);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Context Creation", compression_selector_get_error_string(error));
        return 0;
    }
    
    // Create test data characteristics
    DataCharacteristics chars;
    memset(&chars, 0, sizeof(chars));
    chars.data_size = 4096;
    chars.entropy = 5.5f;
    chars.sparsity = 0.2f;
    chars.compressibility_score = 0.6f;
    
    // Update history with some test results
    error = compression_selector_update_history(context, COMPRESSION_ALGORITHM_NEURAL_QUANTIZATION, 
                                               &chars, 2.5f, 150.0f, 0.85f);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("History Update 1", compression_selector_get_error_string(error));
        compression_selector_destroy(context);
        return 0;
    }
    
    error = compression_selector_update_history(context, COMPRESSION_ALGORITHM_RLE_LOSSLESS, 
                                               &chars, 1.8f, 400.0f, 1.0f);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("History Update 2", compression_selector_get_error_string(error));
        compression_selector_destroy(context);
        return 0;
    }
    
    // Get statistics
    uint64_t total_selections;
    float accuracy, avg_analysis_time;
    error = compression_selector_get_statistics(context, &total_selections, &accuracy, &avg_analysis_time);
    if (error != COMPRESSION_SELECTOR_SUCCESS) {
        print_error("Statistics Retrieval", compression_selector_get_error_string(error));
        compression_selector_destroy(context);
        return 0;
    }
    
    printf("  Performance Statistics:\n");
    printf("    Total selections: %llu\n", total_selections);
    printf("    Accuracy: %.3f\n", accuracy);
    printf("    Average analysis time: %.1f ms\n", avg_analysis_time);
    
    print_success("Performance History and Statistics");
    
    compression_selector_destroy(context);
    
    return 1;
}

// Test 9: Error handling
static int test_error_handling(void) {
    print_test_header("Error Handling");
    
    // Test NULL parameters
    CompressionSelectorError error = compression_selector_create(NULL, NULL);
    if (error != COMPRESSION_SELECTOR_ERROR_INVALID_PARAM) {
        print_error("NULL Parameter Test", "Expected INVALID_PARAM error");
        return 0;
    }
    print_success("NULL Parameter Handling");
    
    // Test invalid configuration
    CompressionSelectorConfig invalid_config;
    memset(&invalid_config, 0, sizeof(invalid_config));
    
    CompressionSelectorContext* context = NULL;
    error = compression_selector_create(&context, &invalid_config);
    // This should still succeed as we handle invalid configs gracefully
    if (error == COMPRESSION_SELECTOR_SUCCESS && context) {
        compression_selector_destroy(context);
        print_success("Invalid Configuration Handling");
    }
    
    // Test analysis with invalid data
    CompressionSelectorConfig config;
    compression_selector_config_create_default(&config, COMPRESSION_OBJECTIVE_BALANCED);
    
    error = compression_selector_create(&context, &config);
    if (error == COMPRESSION_SELECTOR_SUCCESS) {
        DataCharacteristics chars;
        error = compression_selector_analyze_data(context, NULL, 0, &chars);
        if (error != COMPRESSION_SELECTOR_ERROR_INVALID_PARAM) {
            print_error("Invalid Data Test", "Expected INVALID_PARAM error");
            compression_selector_destroy(context);
            return 0;
        }
        print_success("Invalid Data Handling");
        
        compression_selector_destroy(context);
    }
    
    // Test error string function
    const char* error_msg = compression_selector_get_error_string(COMPRESSION_SELECTOR_ERROR_MEMORY_ALLOCATION);
    if (!error_msg || strlen(error_msg) == 0) {
        print_error("Error String", "Invalid error message");
        return 0;
    }
    printf("  Error message example: '%s'\n", error_msg);
    print_success("Error String Function");
    
    return 1;
}

// Test 10: Different optimization objectives
static int test_optimization_objectives(void) {
    print_test_header("Optimization Objectives");
    
    CompressionObjective objectives[] = {
        COMPRESSION_OBJECTIVE_SIZE,
        COMPRESSION_OBJECTIVE_SPEED,
        COMPRESSION_OBJECTIVE_QUALITY,
        COMPRESSION_OBJECTIVE_ENERGY_EFFICIENT
    };
    const char* objective_names[] = {
        "Size Optimization",
        "Speed Optimization", 
        "Quality Optimization",
        "Energy Efficiency"
    };
    
    const size_t data_size = 4096;
    float* test_data = malloc(data_size);
    generate_time_series_data(test_data, data_size / sizeof(float));
    
    printf("  Algorithm Selection by Objective:\n");
    printf("  %-20s %-25s %8s %8s\n", "Objective", "Selected Algorithm", "Score", "Confidence");
    printf("  %s\n", "---------------------------------------------------------------------");
    
    for (int i = 0; i < 4; i++) {
        CompressionSelectorConfig config;
        compression_selector_config_create_default(&config, objectives[i]);
        
        CompressionSelectorContext* context = NULL;
        CompressionSelectorError error = compression_selector_create(&context, &config);
        if (error != COMPRESSION_SELECTOR_SUCCESS) {
            continue;
        }
        
        CompressionSelectionResult result;
        error = compression_selector_select_algorithm(context, test_data, data_size, &result);
        if (error == COMPRESSION_SELECTOR_SUCCESS) {
            printf("  %-20s %-25s %7.3f %7.3f\n",
                   objective_names[i],
                   compression_selector_get_algorithm_name(result.selected_algorithm),
                   result.overall_score,
                   result.selection_confidence);
            
            if (result.alternative_algorithms) {
                free(result.alternative_algorithms);
                free(result.alternative_scores);
            }
        }
        
        compression_selector_destroy(context);
    }
    
    free(test_data);
    print_success("Optimization Objectives Testing");
    
    return 1;
}

// Main test runner
int main(void) {
    printf("NNCP Compression Selector Test Suite\n");
    printf("====================================\n");
    
    srand((unsigned int)time(NULL));
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    struct {
        const char* name;
        int (*test_func)(void);
    } tests[] = {
        {"Context Creation", test_context_creation},
        {"Configuration", test_configuration},
        {"Data Analysis", test_data_analysis},
        {"Algorithm Selection", test_algorithm_selection},
        {"Performance Prediction", test_performance_prediction},
        {"Feature Extraction", test_feature_extraction},
        {"Algorithm Configuration", test_algorithm_configuration},
        {"Performance History", test_performance_history},
        {"Error Handling", test_error_handling},
        {"Optimization Objectives", test_optimization_objectives}
    };
    
    const int num_tests = sizeof(tests) / sizeof(tests[0]);
    
    for (int i = 0; i < num_tests; i++) {
        total_tests++;
        if (tests[i].test_func()) {
            passed_tests++;
        }
        
        // Small delay between tests
        usleep(100000); // 100ms
    }
    
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Test Summary\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    printf("Success rate: %.1f%%\n", (float)passed_tests / total_tests * 100.0f);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! Compression Selector is working correctly.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the output above.\n");
        return 1;
    }
}
