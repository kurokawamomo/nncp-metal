#include "content_analyzer.h"
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
    ContentAnalyzerConfig config = content_analyzer_config_default();
    return content_analyzer_config_validate(&config);
}

static bool test_analyzer_creation() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    bool success = (analyzer != NULL);
    
    content_analyzer_destroy(analyzer);
    return success;
}

static bool test_text_analysis() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    const char *text_data = "This is a sample text for analysis. It contains multiple sentences and words that should be detected as natural language content.";
    ContentCharacteristics characteristics;
    
    ContentAnalyzerResult result = content_analyzer_analyze_content(
        analyzer, (const uint8_t*)text_data, strlen(text_data), &characteristics);
    
    bool success = (result == CONTENT_ANALYZER_SUCCESS) &&
                   (characteristics.text_probability > 0.7f) &&
                   (characteristics.word_count > 10) &&
                   (characteristics.primary_type == CONTENT_TYPE_NATURAL_LANGUAGE);
    
    content_analyzer_destroy(analyzer);
    return success;
}

static bool test_json_analysis() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    const char *json_data = "{\"name\":\"test\",\"values\":[1,2,3],\"nested\":{\"key\":\"value\"}}";
    ContentCharacteristics characteristics;
    
    ContentAnalyzerResult result = content_analyzer_analyze_content(
        analyzer, (const uint8_t*)json_data, strlen(json_data), &characteristics);
    
    bool success = (result == CONTENT_ANALYZER_SUCCESS) &&
                   (characteristics.has_json_markers) &&
                   (characteristics.primary_type == CONTENT_TYPE_JSON);
    
    content_analyzer_destroy(analyzer);
    return success;
}

static bool test_xml_analysis() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    const char *xml_data = "<root><item id=\"1\">value</item><item id=\"2\">another</item></root>";
    ContentCharacteristics characteristics;
    
    ContentAnalyzerResult result = content_analyzer_analyze_content(
        analyzer, (const uint8_t*)xml_data, strlen(xml_data), &characteristics);
    
    bool success = (result == CONTENT_ANALYZER_SUCCESS) &&
                   (characteristics.has_xml_markers) &&
                   (characteristics.primary_type == CONTENT_TYPE_XML);
    
    content_analyzer_destroy(analyzer);
    return success;
}

static bool test_binary_analysis() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    // Create binary data with low entropy
    uint8_t binary_data[256];
    for (int i = 0; i < 256; i++) {
        binary_data[i] = i % 16;  // Repetitive pattern
    }
    
    ContentCharacteristics characteristics;
    
    ContentAnalyzerResult result = content_analyzer_analyze_content(
        analyzer, binary_data, sizeof(binary_data), &characteristics);
    
    bool success = (result == CONTENT_ANALYZER_SUCCESS) &&
                   (characteristics.text_probability < 0.5f) &&
                   (characteristics.primary_type == CONTENT_TYPE_BINARY);
    
    content_analyzer_destroy(analyzer);
    return success;
}

static bool test_random_data_analysis() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    // Create high-entropy random-like data
    uint8_t random_data[1024];
    for (int i = 0; i < 1024; i++) {
        random_data[i] = (i * 17 + 123) % 256;  // Pseudo-random
    }
    
    ContentCharacteristics characteristics;
    
    ContentAnalyzerResult result = content_analyzer_analyze_content(
        analyzer, random_data, sizeof(random_data), &characteristics);
    
    bool success = (result == CONTENT_ANALYZER_SUCCESS) &&
                   (characteristics.entropy > 7.0f);
    
    content_analyzer_destroy(analyzer);
    return success;
}

static bool test_content_type_utilities() {
    ContentCharacteristics text_chars = {0};
    text_chars.primary_type = CONTENT_TYPE_NATURAL_LANGUAGE;
    text_chars.text_probability = 0.9f;
    
    ContentCharacteristics json_chars = {0};
    json_chars.primary_type = CONTENT_TYPE_JSON;
    json_chars.has_json_markers = true;
    
    ContentCharacteristics binary_chars = {0};
    binary_chars.primary_type = CONTENT_TYPE_BINARY;
    binary_chars.text_probability = 0.1f;
    
    ContentCharacteristics compressed_chars = {0};
    compressed_chars.primary_type = CONTENT_TYPE_COMPRESSED;
    compressed_chars.entropy = 7.8f;
    
    return content_analyzer_is_text_data(&text_chars) &&
           content_analyzer_is_structured_data(&json_chars) &&
           content_analyzer_is_binary_data(&binary_chars) &&
           content_analyzer_is_compressed_data(&compressed_chars);
}

static bool test_algorithm_suitability() {
    ContentCharacteristics text_chars = {0};
    text_chars.primary_type = CONTENT_TYPE_NATURAL_LANGUAGE;
    text_chars.text_probability = 0.9f;
    text_chars.entropy = 5.5f;
    text_chars.repetition_factor = 0.1f;
    
    ContentCharacteristics structured_chars = {0};
    structured_chars.primary_type = CONTENT_TYPE_JSON;
    structured_chars.has_json_markers = true;
    structured_chars.entropy = 5.0f;
    structured_chars.pattern_regularity = 0.2f;
    
    float text_lstm = content_analyzer_lstm_suitability(&text_chars);
    float text_transformer = content_analyzer_transformer_suitability(&text_chars);
    float structured_lstm = content_analyzer_lstm_suitability(&structured_chars);
    float structured_transformer = content_analyzer_transformer_suitability(&structured_chars);
    
    // Text should favor LSTM, structured should favor Transformer
    return (text_lstm > text_transformer) && (structured_transformer > structured_lstm);
}

static bool test_configuration_variants() {
    ContentAnalyzerConfig default_config = content_analyzer_config_default();
    ContentAnalyzerConfig fast_config = content_analyzer_config_fast();
    ContentAnalyzerConfig thorough_config = content_analyzer_config_thorough();
    
    bool default_valid = content_analyzer_config_validate(&default_config);
    bool fast_valid = content_analyzer_config_validate(&fast_config);
    bool thorough_valid = content_analyzer_config_validate(&thorough_config);
    
    // Fast config should be simpler than thorough
    bool hierarchy_correct = (fast_config.max_analysis_size <= default_config.max_analysis_size) &&
                            (default_config.max_analysis_size <= thorough_config.max_analysis_size) &&
                            (!fast_config.enable_deep_analysis || thorough_config.enable_deep_analysis);
    
    return default_valid && fast_valid && thorough_valid && hierarchy_correct;
}

static bool test_statistics() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    // Perform some analyses
    const char *text_data = "This is text content for testing statistics.";
    const uint8_t binary_data[] = {0x00, 0x01, 0x02, 0x03, 0x04};
    const char *json_data = "{\"test\": \"value\"}";
    
    ContentCharacteristics characteristics;
    
    content_analyzer_analyze_content(analyzer, (const uint8_t*)text_data, strlen(text_data), &characteristics);
    content_analyzer_analyze_content(analyzer, binary_data, sizeof(binary_data), &characteristics);
    content_analyzer_analyze_content(analyzer, (const uint8_t*)json_data, strlen(json_data), &characteristics);
    
    ContentAnalyzerStats stats = content_analyzer_get_stats(analyzer);
    
    bool success = (stats.total_analyses >= 3) && (stats.successful_analyses >= 3);
    
    // Reset and verify
    content_analyzer_reset_stats(analyzer);
    stats = content_analyzer_get_stats(analyzer);
    bool reset_success = (stats.total_analyses == 0) && (stats.successful_analyses == 0);
    
    content_analyzer_destroy(analyzer);
    return success && reset_success;
}

static bool test_cache_management() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    size_t initial_cache_size = content_analyzer_get_cache_size(analyzer);
    
    // Clear cache and verify
    content_analyzer_clear_pattern_cache(analyzer);
    size_t cleared_cache_size = content_analyzer_get_cache_size(analyzer);
    
    content_analyzer_destroy(analyzer);
    return cleared_cache_size == 0;
}

static bool test_pattern_detection() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    // Create data with obvious patterns
    const char *pattern_data = "ababababcdcdcdcdefefefefghghghgh";
    ContentCharacteristics characteristics;
    
    ContentAnalyzerResult result = content_analyzer_analyze_content(
        analyzer, (const uint8_t*)pattern_data, strlen(pattern_data), &characteristics);
    
    bool success = (result == CONTENT_ANALYZER_SUCCESS) &&
                   (characteristics.repeated_sequences > 0) &&
                   (characteristics.pattern_regularity > 0.0f);
    
    content_analyzer_destroy(analyzer);
    return success;
}

static bool test_entropy_calculation() {
    ContentAnalyzerConfig config = content_analyzer_config_default();
    ContentAnalyzer *analyzer = content_analyzer_create(&config);
    
    if (!analyzer) return false;
    
    // Low entropy data (all same character)
    uint8_t low_entropy_data[100];
    memset(low_entropy_data, 'A', sizeof(low_entropy_data));
    
    // High entropy data (more varied)
    uint8_t high_entropy_data[256];
    for (int i = 0; i < 256; i++) {
        high_entropy_data[i] = i;
    }
    
    ContentCharacteristics low_characteristics, high_characteristics;
    
    content_analyzer_analyze_content(analyzer, low_entropy_data, sizeof(low_entropy_data), &low_characteristics);
    content_analyzer_analyze_content(analyzer, high_entropy_data, sizeof(high_entropy_data), &high_characteristics);
    
    bool success = (low_characteristics.entropy < high_characteristics.entropy) &&
                   (low_characteristics.entropy < 2.0f) &&
                   (high_characteristics.entropy > 6.0f);
    
    content_analyzer_destroy(analyzer);
    return success;
}

static bool test_error_handling() {
    // Test NULL parameters
    ContentAnalyzerResult result1 = content_analyzer_analyze_content(NULL, NULL, 0, NULL);
    
    // Test invalid config
    ContentAnalyzerConfig invalid_config = {0};
    ContentAnalyzer *analyzer = content_analyzer_create(&invalid_config);
    
    bool success = (result1 == CONTENT_ANALYZER_ERROR_INVALID_PARAM) &&
                   (analyzer == NULL);
    
    return success;
}

int main() {
    printf("=== Content Analyzer Unit Tests ===\n\n");
    
    TEST(config_creation);
    TEST(analyzer_creation);
    TEST(text_analysis);
    TEST(json_analysis);
    TEST(xml_analysis);
    TEST(binary_analysis);
    TEST(random_data_analysis);
    TEST(content_type_utilities);
    TEST(algorithm_suitability);
    TEST(configuration_variants);
    TEST(statistics);
    TEST(cache_management);
    TEST(pattern_detection);
    TEST(entropy_calculation);
    TEST(error_handling);
    
    printf("\n=== Test Results ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    printf("Success rate: %.1f%%\n", (float)tests_passed / tests_run * 100.0f);
    
    return (tests_passed == tests_run) ? 0 : 1;
}
