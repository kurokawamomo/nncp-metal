#include "pattern_matcher.h"
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
    PatternMatcherConfig config = pattern_matcher_config_default();
    return pattern_matcher_config_validate(&config);
}

static bool test_matcher_creation() {
    PatternMatcherConfig config = pattern_matcher_config_default();
    PatternMatcher *matcher = pattern_matcher_create(&config);
    
    bool success = (matcher != NULL);
    
    pattern_matcher_destroy(matcher);
    return success;
}

static bool test_pattern_addition() {
    PatternMatcherConfig config = pattern_matcher_config_default();
    PatternMatcher *matcher = pattern_matcher_create(&config);
    
    if (!matcher) return false;
    
    const uint8_t pattern[] = {1, 2, 3, 4};
    PatternMatcherResult result = pattern_matcher_add_pattern(
        matcher, pattern, sizeof(pattern), 0);
    
    bool success = (result == PATTERN_MATCHER_SUCCESS);
    
    pattern_matcher_destroy(matcher);
    return success;
}

static bool test_data_processing() {
    PatternMatcherConfig config = pattern_matcher_config_fast();  // Use fast config
    PatternMatcher *matcher = pattern_matcher_create(&config);
    
    if (!matcher) return false;
    
    const char *test_data = "abcabc";  // Shorter test data
    PatternMatcherResult result = pattern_matcher_add_data(
        matcher, (const uint8_t*)test_data, strlen(test_data));
    
    bool success = (result == PATTERN_MATCHER_SUCCESS);
    
    if (success) {
        PatternMatcherStats stats = pattern_matcher_get_stats(matcher);
        success = (stats.total_patterns > 0);
    }
    
    pattern_matcher_destroy(matcher);
    return success;
}

static bool test_boyer_moore_search() {
    const uint8_t text[] = "abcdefabcdefabcdef";
    const uint8_t pattern[] = "abc";
    
    size_t num_matches;
    size_t *positions = pattern_matcher_boyer_moore_search(
        text, sizeof(text) - 1, pattern, sizeof(pattern) - 1, &num_matches);
    
    bool success = (positions != NULL) && (num_matches == 3);
    
    if (success) {
        success = (positions[0] == 0) && (positions[1] == 6) && (positions[2] == 12);
    }
    
    free(positions);
    return success;
}

static bool test_pattern_search() {
    PatternMatcherConfig config = pattern_matcher_config_default();
    PatternMatcher *matcher = pattern_matcher_create(&config);
    
    if (!matcher) return false;
    
    const uint8_t text[] = "hello world hello";
    const uint8_t pattern[] = "hello";
    
    size_t *positions;
    size_t num_positions;
    
    PatternMatcherResult result = pattern_matcher_search_pattern(
        matcher, pattern, 5, text, sizeof(text) - 1, &positions, &num_positions);
    
    bool success = (result == PATTERN_MATCHER_SUCCESS) && 
                   (num_positions == 2) && 
                   (positions != NULL);
    
    if (success) {
        success = (positions[0] == 0) && (positions[1] == 12);
    }
    
    free(positions);
    pattern_matcher_destroy(matcher);
    return success;
}

static bool test_pattern_scoring() {
    PatternMatcherConfig config = pattern_matcher_config_default();
    PatternMatcher *matcher = pattern_matcher_create(&config);
    
    if (!matcher) return false;
    
    // Add pattern multiple times to increase frequency
    const uint8_t pattern[] = "test";
    for (int i = 0; i < 5; i++) {
        pattern_matcher_add_pattern(matcher, pattern, 4, i * 10);
    }
    
    // Get pattern info and calculate score
    bool found_pattern = false;
    for (size_t i = 0; i < matcher->cache_count; i++) {
        if (pattern_matcher_patterns_equal(pattern, 4,
                                          matcher->pattern_cache[i].pattern,
                                          matcher->pattern_cache[i].length)) {
            float score = pattern_matcher_calculate_score(
                matcher, &matcher->pattern_cache[i], 50);
            found_pattern = (score > 0.0f);
            break;
        }
    }
    
    pattern_matcher_destroy(matcher);
    return found_pattern;
}

static bool test_pattern_matching() {
    PatternMatcherConfig config = pattern_matcher_config_fast();
    PatternMatcher *matcher = pattern_matcher_create(&config);
    
    if (!matcher) return false;
    
    // Add some test data with patterns
    const uint8_t data[] = "abcabc";  // Shorter data
    pattern_matcher_add_data(matcher, data, sizeof(data) - 1);
    
    // Find patterns in the data
    PatternMatch *matches;
    size_t num_matches;
    
    PatternMatcherResult result = pattern_matcher_find_patterns(
        matcher, data, sizeof(data) - 1, &matches, &num_matches);
    
    bool success = (result == PATTERN_MATCHER_SUCCESS);
    
    // Clean up matches
    if (matches) {
        for (size_t i = 0; i < num_matches; i++) {
            free(matches[i].pattern_data);
        }
        free(matches);
    }
    
    pattern_matcher_destroy(matcher);
    return success;
}

static bool test_prediction_integration() {
    PatternMatcherConfig config = pattern_matcher_config_default();
    PatternMatcher *matcher = pattern_matcher_create(&config);
    
    if (!matcher) return false;
    
    // Add pattern data
    const uint8_t training_data[] = "abcabcabcabc";
    pattern_matcher_add_data(matcher, training_data, sizeof(training_data) - 1);
    
    // Test prediction for 'c' after "ab"
    const uint8_t context[] = "ab";
    uint8_t candidate = 'c';
    
    PredictionPatternInfo info = pattern_matcher_analyze_for_prediction(
        matcher, context, sizeof(context) - 1, candidate);
    
    bool success = (info.pattern_score >= 0.0f);
    
    pattern_matcher_destroy(matcher);
    return success;
}

static bool test_cache_management() {
    // Create matcher with small cache
    PatternMatcherConfig config = pattern_matcher_config_default();
    config.max_patterns = 4;  // Small cache for testing eviction
    
    PatternMatcher *matcher = pattern_matcher_create(&config);
    if (!matcher) return false;
    
    // Add more patterns than cache can hold
    const uint8_t patterns[][4] = {
        {1, 2, 3, 4},
        {2, 3, 4, 5},
        {3, 4, 5, 6},
        {4, 5, 6, 7},
        {5, 6, 7, 8}  // This should trigger eviction
    };
    
    for (int i = 0; i < 5; i++) {
        pattern_matcher_add_pattern(matcher, patterns[i], 4, i);
    }
    
    PatternMatcherStats stats = pattern_matcher_get_stats(matcher);
    bool success = (stats.total_patterns <= config.max_patterns);
    
    pattern_matcher_destroy(matcher);
    return success;
}

static bool test_performance_configs() {
    PatternMatcherConfig fast_config = pattern_matcher_config_fast();
    PatternMatcherConfig accurate_config = pattern_matcher_config_accurate();
    
    bool fast_valid = pattern_matcher_config_validate(&fast_config);
    bool accurate_valid = pattern_matcher_config_validate(&accurate_config);
    
    // Fast config should have smaller parameters
    bool fast_smaller = (fast_config.max_patterns < accurate_config.max_patterns) &&
                        (fast_config.cache_size < accurate_config.cache_size);
    
    return fast_valid && accurate_valid && fast_smaller;
}

static bool test_hash_function() {
    const uint8_t data1[] = {1, 2, 3, 4};
    const uint8_t data2[] = {1, 2, 3, 4};
    const uint8_t data3[] = {1, 2, 3, 5};
    
    uint32_t hash1 = pattern_matcher_hash(data1, 4);
    uint32_t hash2 = pattern_matcher_hash(data2, 4);
    uint32_t hash3 = pattern_matcher_hash(data3, 4);
    
    // Same data should produce same hash
    bool same_hash = (hash1 == hash2);
    
    // Different data should produce different hash (very likely)
    bool different_hash = (hash1 != hash3);
    
    return same_hash && different_hash;
}

int main() {
    printf("=== Pattern Matcher Unit Tests ===\n\n");
    
    TEST(config_creation);
    TEST(matcher_creation);
    TEST(pattern_addition);
    TEST(data_processing);
    TEST(boyer_moore_search);
    TEST(pattern_search);
    TEST(pattern_scoring);
    TEST(pattern_matching);
    TEST(prediction_integration);
    TEST(cache_management);
    TEST(performance_configs);
    TEST(hash_function);
    
    printf("\n=== Test Results ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    printf("Success rate: %.1f%%\n", (float)tests_passed / tests_run * 100.0f);
    
    return (tests_passed == tests_run) ? 0 : 1;
}
