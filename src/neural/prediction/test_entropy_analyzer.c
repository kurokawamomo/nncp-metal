#include "entropy_analyzer.h"
#include <stdio.h>
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
static bool test_config_creation() {
    EntropyAnalyzerConfig config = entropy_analyzer_config_default();
    return entropy_analyzer_config_validate(&config);
}

static bool test_analyzer_creation() {
    EntropyAnalyzerConfig config = entropy_analyzer_config_default();
    EntropyAnalyzer *analyzer = entropy_analyzer_create(&config);
    
    bool success = (analyzer != NULL) && entropy_analyzer_is_ready(analyzer);
    
    entropy_analyzer_destroy(analyzer);
    return success;
}

static bool test_basic_entropy_calculation() {
    // Test with known data
    const uint8_t uniform_data[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const uint8_t random_data[] = {42, 17, 88, 3, 156, 91, 204, 37, 79, 122, 15, 246, 63, 189, 8, 133};
    
    float uniform_entropy = entropy_analyzer_local_entropy(uniform_data, sizeof(uniform_data), 8, 8);
    float random_entropy = entropy_analyzer_local_entropy(random_data, sizeof(random_data), 8, 8);
    
    // Random data should have higher entropy than uniform data
    return random_entropy > uniform_entropy && uniform_entropy > 0.0f;
}

static bool test_data_processing() {
    EntropyAnalyzerConfig config = entropy_analyzer_config_default();
    EntropyAnalyzer *analyzer = entropy_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    const char *test_data = "This is a test string for entropy analysis with repeated patterns.";
    EntropyAnalyzerResult result = entropy_analyzer_process_data(
        analyzer, (const uint8_t*)test_data, strlen(test_data));
    
    bool success = (result == ENTROPY_ANALYZER_SUCCESS);
    
    if (success) {
        EntropyMetrics metrics = entropy_analyzer_get_current_metrics(analyzer);
        success = (metrics.shannon_entropy > 0.0f) && (metrics.unique_symbols > 0);
    }
    
    entropy_analyzer_destroy(analyzer);
    return success;
}

static bool test_metrics_calculation() {
    EntropyAnalyzerConfig config = entropy_analyzer_config_default();
    EntropyAnalyzer *analyzer = entropy_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    const char *test_data = "Hello, World! This is a comprehensive test for entropy metrics calculation.";
    
    // First process the data to build frequency tables
    entropy_analyzer_process_data(analyzer, (const uint8_t*)test_data, strlen(test_data));
    
    EntropyMetrics metrics = entropy_analyzer_get_current_metrics(analyzer);
    
    printf("\n  Debug: entropy=%.2f, symbols=%zu, potential=%.2f, distribution=%.2f\n", 
           metrics.shannon_entropy, metrics.unique_symbols, 
           metrics.compression_potential, metrics.symbol_distribution);
    
    bool success = (metrics.shannon_entropy > 0.0f) &&
                   (metrics.unique_symbols > 10) &&
                   (metrics.compression_potential >= 0.0f) &&
                   (metrics.symbol_distribution > 0.0f);
    
    entropy_analyzer_destroy(analyzer);
    return success;
}

static bool test_conditional_entropy() {
    const uint8_t pattern_data[] = "abcabcabcabcabcabc";
    
    float conditional = entropy_analyzer_conditional_entropy(
        pattern_data, sizeof(pattern_data) - 1, 2);
    
    // Conditional entropy should be relatively low for patterned data
    return conditional >= 0.0f && conditional < 3.0f;
}

static bool test_content_characterization() {
    // Test text data detection
    const char *text_data = "The quick brown fox jumps over the lazy dog.";
    
    EntropyAnalyzerConfig config = entropy_analyzer_config_default();
    EntropyAnalyzer *analyzer = entropy_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    entropy_analyzer_process_data(analyzer, (const uint8_t*)text_data, strlen(text_data));
    EntropyMetrics metrics = entropy_analyzer_get_current_metrics(analyzer);
    
    printf("\n  Debug text: entropy=%.2f, symbols=%zu\n", 
           metrics.shannon_entropy, metrics.unique_symbols);
    
    bool is_text = entropy_analyzer_is_text_data(&metrics);
    bool not_random = !entropy_analyzer_is_random_data(&metrics);
    
    entropy_analyzer_destroy(analyzer);
    return is_text && not_random;
}

static bool test_prediction_integration() {
    EntropyAnalyzerConfig config = entropy_analyzer_config_default();
    EntropyAnalyzer *analyzer = entropy_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    const uint8_t context[] = "Hello, Wor";
    uint8_t candidate = 'l';  // Common in context
    
    PredictionEntropyInfo info = entropy_analyzer_analyze_for_prediction(
        analyzer, context, sizeof(context) - 1, candidate);
    
    bool success = (info.entropy_score >= 0.0f) &&
                   (info.context_relevance >= 0.0f) &&
                   (info.prediction_confidence >= 0.0f);
    
    entropy_analyzer_destroy(analyzer);
    return success;
}

static bool test_compression_prediction() {
    // Test with different data types
    const uint8_t structured_data[] = "{{{{}}}}[[[[]]]]{{{{}}}}[[[[]]]]";
    const uint8_t text_data[] = "The entropy analyzer predicts compression ratios.";
    
    EntropyAnalyzerConfig config = entropy_analyzer_config_default();
    EntropyAnalyzer *analyzer = entropy_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    entropy_analyzer_process_data(analyzer, structured_data, sizeof(structured_data) - 1);
    EntropyMetrics structured_metrics = entropy_analyzer_get_current_metrics(analyzer);
    
    entropy_analyzer_process_data(analyzer, text_data, sizeof(text_data) - 1);
    EntropyMetrics text_metrics = entropy_analyzer_get_current_metrics(analyzer);
    
    float structured_ratio = entropy_analyzer_predict_compression_ratio(&structured_metrics);
    float text_ratio = entropy_analyzer_predict_compression_ratio(&text_metrics);
    
    printf("\n  Debug compression: structured=%.2f, text=%.2f\n", 
           structured_ratio, text_ratio);
    
    // Both ratios should be reasonable
    bool success = structured_ratio > 0.0f && text_ratio > 0.0f && 
                   structured_ratio < 1.0f && text_ratio < 1.0f;
    
    entropy_analyzer_destroy(analyzer);
    return success;
}

static bool test_performance_configs() {
    EntropyAnalyzerConfig fast_config = entropy_analyzer_config_fast();
    EntropyAnalyzerConfig accurate_config = entropy_analyzer_config_accurate();
    
    bool fast_valid = entropy_analyzer_config_validate(&fast_config);
    bool accurate_valid = entropy_analyzer_config_validate(&accurate_config);
    
    // Fast config should have smaller parameters
    bool fast_smaller = (fast_config.window_size < accurate_config.window_size) &&
                        (fast_config.context_length < accurate_config.context_length);
    
    return fast_valid && accurate_valid && fast_smaller;
}

int main() {
    printf("=== Entropy Analyzer Unit Tests ===\n\n");
    
    TEST(config_creation);
    TEST(analyzer_creation);
    TEST(basic_entropy_calculation);
    TEST(data_processing);
    TEST(metrics_calculation);
    TEST(conditional_entropy);
    TEST(content_characterization);
    TEST(prediction_integration);
    TEST(compression_prediction);
    TEST(performance_configs);
    
    printf("\n=== Test Results ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    printf("Success rate: %.1f%%\n", (float)tests_passed / tests_run * 100.0f);
    
    return (tests_passed == tests_run) ? 0 : 1;
}
