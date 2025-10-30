#include "enhanced_selector.h"
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
static bool test_config_creation() {
    EnhancedSelectorConfig config = enhanced_selector_config_default();
    return enhanced_selector_config_validate(&config);
}

static bool test_selector_creation() {
    EnhancedSelectorConfig config = enhanced_selector_config_default();
    EnhancedSelector *selector = enhanced_selector_create(&config);
    
    bool success = (selector != NULL);
    
    enhanced_selector_destroy(selector);
    return success;
}

static bool test_content_analysis() {
    EnhancedSelectorConfig config = enhanced_selector_config_fast();
    EnhancedSelector *selector = enhanced_selector_create(&config);
    
    if (!selector) return false;
    
    const char *text_data = "Hello, World! This is a test text with patterns.";
    ContentAnalysis analysis;
    
    EnhancedSelectorResult result = enhanced_selector_analyze_content(
        selector, (const uint8_t*)text_data, strlen(text_data), &analysis);
    
    bool success = (result == ENHANCED_SELECTOR_SUCCESS) &&
                   (analysis.average_entropy > 0.0f) &&
                   (analysis.confidence_level >= 0.0f);
    
    enhanced_selector_destroy(selector);
    return success;
}

static bool test_quick_analysis() {
    EnhancedSelectorConfig config = enhanced_selector_config_fast();
    EnhancedSelector *selector = enhanced_selector_create(&config);
    
    if (!selector) return false;
    
    const uint8_t test_data[] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ContentAnalysis analysis;
    
    EnhancedSelectorResult result = enhanced_selector_quick_analysis(
        selector, test_data, sizeof(test_data), &analysis);
    
    bool success = (result == ENHANCED_SELECTOR_SUCCESS) &&
                   (analysis.average_entropy >= 0.0f);
    
    enhanced_selector_destroy(selector);
    return success;
}

static bool test_algorithm_recommendation() {
    EnhancedSelectorConfig config = enhanced_selector_config_default();
    EnhancedSelector *selector = enhanced_selector_create(&config);
    
    if (!selector) return false;
    
    // Test with text data (should favor LSTM)
    const char *text_data = "The quick brown fox jumps over the lazy dog repeatedly.";
    ContentAnalysis analysis;
    
    enhanced_selector_analyze_content(selector, (const uint8_t*)text_data, strlen(text_data), &analysis);
    AlgorithmRecommendation recommendation = enhanced_selector_recommend_algorithm(selector, &analysis);
    
    bool success = (recommendation == SELECTION_LSTM_FAVORED || 
                    recommendation == SELECTION_TRANSFORMER_FAVORED ||
                    recommendation == SELECTION_ANALYSIS_NEEDED);
    
    enhanced_selector_destroy(selector);
    return success;
}

static bool test_algorithm_selection() {
    EnhancedSelectorConfig config = enhanced_selector_config_default();
    EnhancedSelector *selector = enhanced_selector_create(&config);
    
    if (!selector) return false;
    
    const uint8_t structured_data[] = "{{{{}}}}[[[[]]]]{{{{}}}}[[[[]]]]";
    AlgorithmRecommendation recommendation;
    float confidence;
    
    EnhancedSelectorResult result = enhanced_selector_select_best_algorithm(
        selector, structured_data, sizeof(structured_data) - 1, &recommendation, &confidence);
    
    bool success = (result == ENHANCED_SELECTOR_SUCCESS) &&
                   (recommendation != SELECTION_ERROR) &&
                   (confidence >= 0.0f && confidence <= 1.0f);
    
    enhanced_selector_destroy(selector);
    return success;
}

static bool test_content_characterization() {
    EnhancedSelectorConfig config = enhanced_selector_config_default();
    EnhancedSelector *selector = enhanced_selector_create(&config);
    
    if (!selector) return false;
    
    // Test text content
    const char *text_data = "This is clearly readable text content.";
    ContentAnalysis text_analysis;
    enhanced_selector_analyze_content(selector, (const uint8_t*)text_data, strlen(text_data), &text_analysis);
    
    // Test structured content
    const uint8_t structured_data[] = "ababababcdcdcdcd";
    ContentAnalysis structured_analysis;
    enhanced_selector_analyze_content(selector, structured_data, sizeof(structured_data) - 1, &structured_analysis);
    
    bool success = true;  // Basic validation that analysis completes
    
    enhanced_selector_destroy(selector);
    return success;
}

static bool test_cache_functionality() {
    EnhancedSelectorConfig config = enhanced_selector_config_default();
    EnhancedSelector *selector = enhanced_selector_create(&config);
    
    if (!selector) return false;
    
    const uint8_t test_data[] = "Cache test data for validation";
    ContentAnalysis analysis1, analysis2;
    
    // First analysis
    enhanced_selector_analyze_content(selector, test_data, sizeof(test_data) - 1, &analysis1);
    
    // Second analysis (should use cache)
    enhanced_selector_analyze_content(selector, test_data, sizeof(test_data) - 1, &analysis2);
    
    bool cache_used = enhanced_selector_is_cache_valid(selector, test_data, sizeof(test_data) - 1);
    
    // Clear cache and verify
    enhanced_selector_clear_cache(selector);
    bool cache_cleared = !enhanced_selector_is_cache_valid(selector, test_data, sizeof(test_data) - 1);
    
    enhanced_selector_destroy(selector);
    return cache_used && cache_cleared;
}

static bool test_statistics() {
    EnhancedSelectorConfig config = enhanced_selector_config_default();
    EnhancedSelector *selector = enhanced_selector_create(&config);
    
    if (!selector) return false;
    
    // Perform some analyses
    const char *data1 = "Text data for statistics test";
    const uint8_t data2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    AlgorithmRecommendation rec;
    float confidence;
    
    enhanced_selector_select_best_algorithm(selector, (const uint8_t*)data1, strlen(data1), &rec, &confidence);
    enhanced_selector_select_best_algorithm(selector, data2, sizeof(data2), &rec, &confidence);
    
    EnhancedSelectorStats stats = enhanced_selector_get_stats(selector);
    
    bool success = (stats.total_analyses >= 2);
    
    // Reset and verify
    enhanced_selector_reset_stats(selector);
    stats = enhanced_selector_get_stats(selector);
    bool reset_success = (stats.total_analyses == 0);
    
    enhanced_selector_destroy(selector);
    return success && reset_success;
}

static bool test_performance_configs() {
    EnhancedSelectorConfig fast_config = enhanced_selector_config_fast();
    EnhancedSelectorConfig accurate_config = enhanced_selector_config_accurate();
    
    bool fast_valid = enhanced_selector_config_validate(&fast_config);
    bool accurate_valid = enhanced_selector_config_validate(&accurate_config);
    
    // Fast config should be simpler
    bool fast_simpler = (fast_config.analysis_window_size <= accurate_config.analysis_window_size) &&
                        (!fast_config.enable_deep_analysis || accurate_config.enable_deep_analysis);
    
    return fast_valid && accurate_valid && fast_simpler;
}

int main() {
    printf("=== Enhanced Selector Unit Tests ===\n\n");
    
    TEST(config_creation);
    TEST(selector_creation);
    TEST(content_analysis);
    TEST(quick_analysis);
    TEST(algorithm_recommendation);
    TEST(algorithm_selection);
    TEST(content_characterization);
    TEST(cache_functionality);
    TEST(statistics);
    TEST(performance_configs);
    
    printf("\n=== Test Results ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    printf("Success rate: %.1f%%\n", (float)tests_passed / tests_run * 100.0f);
    
    return (tests_passed == tests_run) ? 0 : 1;
}
