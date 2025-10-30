#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "../../src/neural/quality/quality_validator.h"

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
static void generate_test_data(uint8_t* buffer, size_t size, double entropy_target) {
    if (entropy_target < 1.0) {
        // Low entropy data (mostly zeros)
        memset(buffer, 0, size);
        for (size_t i = 0; i < size / 10; i++) {
            buffer[rand() % size] = rand() % 256;
        }
    } else if (entropy_target > 7.0) {
        // High entropy data (random)
        for (size_t i = 0; i < size; i++) {
            buffer[i] = rand() % 256;
        }
    } else {
        // Medium entropy data (pattern with some randomness)
        for (size_t i = 0; i < size; i++) {
            buffer[i] = (i % 16) + (rand() % 32);
        }
    }
}

// Custom validator function for testing
static QualityValidatorError test_custom_validator(const void* data, size_t data_size,
                                                  void* context, double* quality_score) {
    if (!data || data_size == 0 || !quality_score) {
        return QV_ERROR_INVALID_PARAM;
    }
    
    // Simple custom validator: check for magic number at start
    const uint8_t* bytes = (const uint8_t*)data;
    bool has_magic = (data_size >= 4 && 
                     bytes[0] == 0xDE && bytes[1] == 0xAD && 
                     bytes[2] == 0xBE && bytes[3] == 0xEF);
    
    *quality_score = has_magic ? 1.0 : 0.5;
    return QV_SUCCESS;
}

// Test 1: Basic validator creation and destruction
static int test_validator_creation(void) {
    print_test_header("Validator Creation and Destruction");
    
    ValidatorConfig config;
    QualityValidatorError error = quality_validator_config_default(&config, QV_MODE_STANDARD);
    if (error != QV_SUCCESS) {
        print_error("Default Config", quality_validator_error_string(error));
        return 0;
    }
    
    print_success("Default Configuration");
    
    QualityValidator* validator = NULL;
    error = quality_validator_create(&validator, &config);
    if (error != QV_SUCCESS) {
        print_error("Validator Creation", quality_validator_error_string(error));
        return 0;
    }
    
    if (!validator) {
        print_error("Validator Creation", "Validator is NULL");
        return 0;
    }
    
    print_success("Validator Created");
    
    // Test configuration modes
    const ValidationMode modes[] = {
        QV_MODE_BASIC, QV_MODE_STANDARD, QV_MODE_COMPREHENSIVE,
        QV_MODE_CONTINUOUS, QV_MODE_ADAPTIVE
    };
    
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
        ValidatorConfig test_config;
        error = quality_validator_config_default(&test_config, modes[i]);
        if (error != QV_SUCCESS) {
            print_error("Config Mode", "Failed to create config for mode");
            quality_validator_destroy(validator);
            return 0;
        }
    }
    
    print_success("Configuration Modes");
    
    // Destroy validator
    quality_validator_destroy(validator);
    print_success("Validator Destroyed");
    
    return 1;
}

// Test 2: Basic data validation
static int test_basic_validation(void) {
    print_test_header("Basic Data Validation");
    
    ValidatorConfig config;
    quality_validator_config_default(&config, QV_MODE_STANDARD);
    
    QualityValidator* validator = NULL;
    QualityValidatorError error = quality_validator_create(&validator, &config);
    if (error != QV_SUCCESS) {
        print_error("Validator Creation", quality_validator_error_string(error));
        return 0;
    }
    
    // Test with normal data
    const size_t data_size = 4096;
    uint8_t* test_data = malloc(data_size);
    if (!test_data) {
        print_error("Memory Allocation", "Failed to allocate test data");
        quality_validator_destroy(validator);
        return 0;
    }
    
    generate_test_data(test_data, data_size, 5.0);  // Medium entropy
    
    ValidationResult result;
    error = quality_validator_validate(validator, test_data, data_size, NULL, &result);
    if (error != QV_SUCCESS) {
        print_error("Data Validation", quality_validator_error_string(error));
        free(test_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  Validation result: %s\n", result.is_valid ? "Valid" : "Invalid");
    printf("  Quality level: %s\n", quality_validator_level_name(result.quality_level));
    printf("  Overall score: %.3f\n", result.overall_score);
    printf("  Number of metrics: %u\n", result.num_metrics);
    printf("  Number of errors: %u\n", result.num_errors);
    printf("  Validation time: %llu μs\n", result.validation_time_us);
    
    if (result.num_metrics == 0) {
        print_error("Metrics", "No metrics calculated");
        free(test_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    print_success("Basic Validation");
    
    // Test with invalid data (NULL)
    error = quality_validator_validate(validator, NULL, data_size, NULL, &result);
    if (error != QV_ERROR_INVALID_PARAM) {
        print_error("NULL Data", "Expected INVALID_PARAM error");
        free(test_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    print_success("NULL Parameter Handling");
    
    free(test_data);
    quality_validator_destroy(validator);
    
    return 1;
}

// Test 3: Quality metrics calculation
static int test_quality_metrics(void) {
    print_test_header("Quality Metrics Calculation");
    
    ValidatorConfig config;
    quality_validator_config_default(&config, QV_MODE_COMPREHENSIVE);
    
    QualityValidator* validator = NULL;
    QualityValidatorError error = quality_validator_create(&validator, &config);
    if (error != QV_SUCCESS) {
        print_error("Validator Creation", quality_validator_error_string(error));
        return 0;
    }
    
    // Test with different data characteristics
    const size_t data_size = 2048;
    uint8_t* test_data = malloc(data_size);
    if (!test_data) {
        print_error("Memory Allocation", "Failed to allocate test data");
        quality_validator_destroy(validator);
        return 0;
    }
    
    // Test high entropy data
    generate_test_data(test_data, data_size, 8.0);
    
    QualityMetric metrics[QV_MAX_QUALITY_METRICS];
    uint32_t num_metrics;
    error = quality_validator_calculate_metrics(validator, test_data, data_size,
                                               metrics, QV_MAX_QUALITY_METRICS, &num_metrics);
    if (error != QV_SUCCESS) {
        print_error("High Entropy Metrics", quality_validator_error_string(error));
        free(test_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  High entropy data - Metrics calculated: %u\n", num_metrics);
    for (uint32_t i = 0; i < num_metrics; i++) {
        printf("    %s: %.3f (valid: %s)\n", 
               metrics[i].name, metrics[i].value, 
               metrics[i].is_valid ? "yes" : "no");
    }
    
    // Test low entropy data
    generate_test_data(test_data, data_size, 0.5);
    
    error = quality_validator_calculate_metrics(validator, test_data, data_size,
                                               metrics, QV_MAX_QUALITY_METRICS, &num_metrics);
    if (error != QV_SUCCESS) {
        print_error("Low Entropy Metrics", quality_validator_error_string(error));
        free(test_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  Low entropy data - Metrics calculated: %u\n", num_metrics);
    for (uint32_t i = 0; i < num_metrics; i++) {
        printf("    %s: %.3f (valid: %s)\n", 
               metrics[i].name, metrics[i].value, 
               metrics[i].is_valid ? "yes" : "no");
    }
    
    print_success("Quality Metrics Calculation");
    
    free(test_data);
    quality_validator_destroy(validator);
    
    return 1;
}

// Test 4: Compression validation
static int test_compression_validation(void) {
    print_test_header("Compression Validation");
    
    ValidatorConfig config;
    quality_validator_config_default(&config, QV_MODE_STANDARD);
    config.thresholds.compression_ratio_min = 1.2;
    config.thresholds.compression_ratio_max = 10.0;
    
    QualityValidator* validator = NULL;
    QualityValidatorError error = quality_validator_create(&validator, &config);
    if (error != QV_SUCCESS) {
        print_error("Validator Creation", quality_validator_error_string(error));
        return 0;
    }
    
    // Create original and "compressed" data
    const size_t original_size = 8192;
    const size_t compressed_size = 4096;  // 2:1 compression ratio
    
    uint8_t* original_data = malloc(original_size);
    uint8_t* compressed_data = malloc(compressed_size);
    
    if (!original_data || !compressed_data) {
        print_error("Memory Allocation", "Failed to allocate test data");
        free(original_data);
        free(compressed_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    generate_test_data(original_data, original_size, 6.0);
    generate_test_data(compressed_data, compressed_size, 7.0);
    
    ValidationResult result;
    error = quality_validator_validate_compression(validator, 
                                                  original_data, original_size,
                                                  compressed_data, compressed_size,
                                                  &result);
    if (error != QV_SUCCESS) {
        print_error("Compression Validation", quality_validator_error_string(error));
        free(original_data);
        free(compressed_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  Compression validation result: %s\n", result.is_valid ? "Valid" : "Invalid");
    printf("  Quality level: %s\n", quality_validator_level_name(result.quality_level));
    printf("  Number of metrics: %u\n", result.num_metrics);
    printf("  Number of errors: %u\n", result.num_errors);
    
    // Look for compression ratio metric
    bool found_ratio = false;
    for (uint32_t i = 0; i < result.num_metrics; i++) {
        if (result.metrics[i].type == QV_METRIC_COMPRESSION_RATIO) {
            printf("  Compression ratio: %.2f\n", result.metrics[i].value);
            found_ratio = true;
            break;
        }
    }
    
    if (!found_ratio) {
        print_error("Compression Ratio", "Compression ratio metric not found");
        free(original_data);
        free(compressed_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    print_success("Compression Validation");
    
    free(original_data);
    free(compressed_data);
    quality_validator_destroy(validator);
    
    return 1;
}

// Test 5: Error detection
static int test_error_detection(void) {
    print_test_header("Error Detection");
    
    ValidatorConfig config;
    quality_validator_config_default(&config, QV_MODE_COMPREHENSIVE);
    
    QualityValidator* validator = NULL;
    QualityValidatorError error = quality_validator_create(&validator, &config);
    if (error != QV_SUCCESS) {
        print_error("Validator Creation", quality_validator_error_string(error));
        return 0;
    }
    
    // Test with corrupted data (long zero runs)
    const size_t data_size = 1024;
    uint8_t* corrupted_data = malloc(data_size);
    if (!corrupted_data) {
        print_error("Memory Allocation", "Failed to allocate test data");
        quality_validator_destroy(validator);
        return 0;
    }
    
    // Create data with corruption pattern
    memset(corrupted_data, 0, data_size);
    for (size_t i = 0; i < 100; i++) {
        corrupted_data[i] = i % 256;
    }
    // Leave large zero section to simulate corruption
    
    ValidationError errors[QV_MAX_ERROR_DETAILS];
    uint32_t num_errors;
    error = quality_validator_detect_errors(validator, corrupted_data, data_size,
                                           errors, QV_MAX_ERROR_DETAILS, &num_errors);
    if (error != QV_SUCCESS) {
        print_error("Error Detection", quality_validator_error_string(error));
        free(corrupted_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  Errors detected: %u\n", num_errors);
    for (uint32_t i = 0; i < num_errors; i++) {
        printf("    Error %u: %s (%s)\n", i + 1, errors[i].message,
               quality_validator_severity_name(errors[i].severity));
    }
    
    if (num_errors == 0) {
        print_error("Error Detection", "Expected to detect errors in corrupted data");
        free(corrupted_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    print_success("Error Detection");
    
    // Test with very small data
    uint8_t small_data[4] = {1, 2, 3, 4};
    error = quality_validator_detect_errors(validator, small_data, sizeof(small_data),
                                           errors, QV_MAX_ERROR_DETAILS, &num_errors);
    if (error != QV_SUCCESS) {
        print_error("Small Data Error Detection", quality_validator_error_string(error));
        free(corrupted_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  Small data errors detected: %u\n", num_errors);
    
    print_success("Small Data Error Detection");
    
    free(corrupted_data);
    quality_validator_destroy(validator);
    
    return 1;
}

// Test 6: Custom validators
static int test_custom_validators(void) {
    print_test_header("Custom Validators");
    
    ValidatorConfig config;
    quality_validator_config_default(&config, QV_MODE_STANDARD);
    
    QualityValidator* validator = NULL;
    QualityValidatorError error = quality_validator_create(&validator, &config);
    if (error != QV_SUCCESS) {
        print_error("Validator Creation", quality_validator_error_string(error));
        return 0;
    }
    
    // Add custom validator
    error = quality_validator_add_custom(validator, test_custom_validator, NULL);
    if (error != QV_SUCCESS) {
        print_error("Add Custom Validator", quality_validator_error_string(error));
        quality_validator_destroy(validator);
        return 0;
    }
    
    print_success("Custom Validator Added");
    
    // Test with data that has magic number
    uint8_t magic_data[100];
    magic_data[0] = 0xDE;
    magic_data[1] = 0xAD;
    magic_data[2] = 0xBE;
    magic_data[3] = 0xEF;
    for (size_t i = 4; i < sizeof(magic_data); i++) {
        magic_data[i] = rand() % 256;
    }
    
    ValidationResult result;
    error = quality_validator_validate(validator, magic_data, sizeof(magic_data), NULL, &result);
    if (error != QV_SUCCESS) {
        print_error("Magic Data Validation", quality_validator_error_string(error));
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  Magic data validation: %s\n", result.is_valid ? "Valid" : "Invalid");
    printf("  Number of errors: %u\n", result.num_errors);
    
    // Test with data without magic number
    uint8_t normal_data[100];
    for (size_t i = 0; i < sizeof(normal_data); i++) {
        normal_data[i] = rand() % 256;
    }
    
    error = quality_validator_validate(validator, normal_data, sizeof(normal_data), NULL, &result);
    if (error != QV_SUCCESS) {
        print_error("Normal Data Validation", quality_validator_error_string(error));
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  Normal data validation: %s\n", result.is_valid ? "Valid" : "Invalid");
    printf("  Number of errors: %u\n", result.num_errors);
    
    print_success("Custom Validator Testing");
    
    quality_validator_destroy(validator);
    
    return 1;
}

// Test 7: Validation rules and thresholds
static int test_validation_rules(void) {
    print_test_header("Validation Rules and Thresholds");
    
    ValidatorConfig config;
    quality_validator_config_default(&config, QV_MODE_STANDARD);
    
    // Add custom validation rule
    ValidationRule rule;
    rule.metric_type = QV_METRIC_ENTROPY;
    rule.min_value = 3.0;
    rule.max_value = 7.0;
    rule.severity = QV_SEVERITY_WARNING;
    rule.enabled = true;
    snprintf(rule.description, sizeof(rule.description), "Entropy must be between 3.0 and 7.0");
    
    config.thresholds.rules[config.thresholds.num_rules++] = rule;
    
    QualityValidator* validator = NULL;
    QualityValidatorError error = quality_validator_create(&validator, &config);
    if (error != QV_SUCCESS) {
        print_error("Validator Creation", quality_validator_error_string(error));
        return 0;
    }
    
    // Test with data that violates entropy rule (too low entropy)
    const size_t data_size = 1024;
    uint8_t* low_entropy_data = malloc(data_size);
    if (!low_entropy_data) {
        print_error("Memory Allocation", "Failed to allocate test data");
        quality_validator_destroy(validator);
        return 0;
    }
    
    generate_test_data(low_entropy_data, data_size, 0.5);  // Very low entropy
    
    ValidationResult result;
    error = quality_validator_validate(validator, low_entropy_data, data_size, NULL, &result);
    if (error != QV_SUCCESS) {
        print_error("Low Entropy Validation", quality_validator_error_string(error));
        free(low_entropy_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  Low entropy validation result: %s\n", result.is_valid ? "Valid" : "Invalid");
    printf("  Number of errors: %u\n", result.num_errors);
    
    for (uint32_t i = 0; i < result.num_errors; i++) {
        printf("    Error %u: %s\n", i + 1, result.errors[i].message);
    }
    
    print_success("Validation Rules Testing");
    
    free(low_entropy_data);
    quality_validator_destroy(validator);
    
    return 1;
}

// Test 8: Statistics and reporting
static int test_statistics(void) {
    print_test_header("Statistics and Reporting");
    
    ValidatorConfig config;
    quality_validator_config_default(&config, QV_MODE_STANDARD);
    
    QualityValidator* validator = NULL;
    QualityValidatorError error = quality_validator_create(&validator, &config);
    if (error != QV_SUCCESS) {
        print_error("Validator Creation", quality_validator_error_string(error));
        return 0;
    }
    
    // Perform several validations
    const size_t data_size = 512;
    uint8_t* test_data = malloc(data_size);
    if (!test_data) {
        print_error("Memory Allocation", "Failed to allocate test data");
        quality_validator_destroy(validator);
        return 0;
    }
    
    const int num_validations = 5;
    for (int i = 0; i < num_validations; i++) {
        generate_test_data(test_data, data_size, 4.0 + i);  // Varying entropy
        
        ValidationResult result;
        error = quality_validator_validate(validator, test_data, data_size, NULL, &result);
        if (error != QV_SUCCESS) {
            printf("  Validation %d failed: %s\n", i + 1, quality_validator_error_string(error));
        }
    }
    
    // Get statistics
    uint64_t total_validations;
    double success_rate;
    double average_score;
    double average_time;
    
    error = quality_validator_get_statistics(validator, &total_validations, &success_rate,
                                            &average_score, &average_time);
    if (error != QV_SUCCESS) {
        print_error("Get Statistics", quality_validator_error_string(error));
        free(test_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    printf("  Statistics:\n");
    printf("    Total validations: %llu\n", total_validations);
    printf("    Success rate: %.1f%%\n", success_rate * 100.0);
    printf("    Average quality score: %.3f\n", average_score);
    printf("    Average validation time: %.3f seconds\n", average_time);
    
    if (total_validations != num_validations) {
        print_error("Statistics", "Incorrect validation count");
        free(test_data);
        quality_validator_destroy(validator);
        return 0;
    }
    
    print_success("Statistics Collection");
    
    free(test_data);
    quality_validator_destroy(validator);
    
    return 1;
}

// Test 9: Utility functions
static int test_utility_functions(void) {
    print_test_header("Utility Functions");
    
    // Test error string function
    const char* error_msg = quality_validator_error_string(QV_ERROR_VALIDATION_FAILED);
    if (!error_msg || strlen(error_msg) == 0) {
        print_error("Error String", "Invalid error message");
        return 0;
    }
    printf("  Example error message: '%s'\n", error_msg);
    
    print_success("Error String Function");
    
    // Test quality level names
    for (int level = QV_QUALITY_UNACCEPTABLE; level <= QV_QUALITY_EXCELLENT; level++) {
        const char* level_name = quality_validator_level_name((QualityLevel)level);
        if (!level_name || strlen(level_name) == 0) {
            print_error("Level Name", "Invalid level name");
            return 0;
        }
        printf("  Quality level %d: %s\n", level, level_name);
    }
    
    print_success("Quality Level Names");
    
    // Test metric type names
    const QualityMetricType test_metrics[] = {
        QV_METRIC_COMPRESSION_RATIO, QV_METRIC_QUALITY_SCORE, QV_METRIC_ENTROPY,
        QV_METRIC_PSNR, QV_METRIC_SSIM, QV_METRIC_PROCESSING_TIME
    };
    
    for (size_t i = 0; i < sizeof(test_metrics) / sizeof(test_metrics[0]); i++) {
        const char* metric_name = quality_validator_metric_name(test_metrics[i]);
        if (!metric_name || strlen(metric_name) == 0) {
            print_error("Metric Name", "Invalid metric name");
            return 0;
        }
        printf("  Metric %d: %s\n", (int)test_metrics[i], metric_name);
    }
    
    print_success("Metric Type Names");
    
    // Test score to level conversion
    struct {
        double score;
        QualityLevel expected_level;
    } score_tests[] = {
        {0.99, QV_QUALITY_EXCELLENT},
        {0.90, QV_QUALITY_GOOD},
        {0.80, QV_QUALITY_ACCEPTABLE},
        {0.70, QV_QUALITY_POOR},
        {0.50, QV_QUALITY_UNACCEPTABLE}
    };
    
    for (size_t i = 0; i < sizeof(score_tests) / sizeof(score_tests[0]); i++) {
        QualityLevel level = quality_validator_score_to_level(score_tests[i].score);
        if (level != score_tests[i].expected_level) {
            print_error("Score to Level", "Incorrect level conversion");
            return 0;
        }
        printf("  Score %.2f -> %s\n", score_tests[i].score, 
               quality_validator_level_name(level));
    }
    
    print_success("Score to Level Conversion");
    
    return 1;
}

// Test 10: Overall score calculation
static int test_overall_score_calculation(void) {
    print_test_header("Overall Score Calculation");
    
    // Create test metrics
    QualityMetric metrics[3];
    
    // Metric 1: High value, high weight
    metrics[0].type = QV_METRIC_QUALITY_SCORE;
    metrics[0].value = 0.9;
    metrics[0].weight = 0.5;
    metrics[0].is_valid = true;
    
    // Metric 2: Medium value, medium weight
    metrics[1].type = QV_METRIC_COMPRESSION_RATIO;
    metrics[1].value = 0.7;
    metrics[1].weight = 0.3;
    metrics[1].is_valid = true;
    
    // Metric 3: Low value, low weight
    metrics[2].type = QV_METRIC_ENTROPY;
    metrics[2].value = 0.5;
    metrics[2].weight = 0.2;
    metrics[2].is_valid = true;
    
    double overall_score;
    QualityValidatorError error = quality_validator_calculate_overall_score(metrics, 3, &overall_score);
    if (error != QV_SUCCESS) {
        print_error("Overall Score Calculation", quality_validator_error_string(error));
        return 0;
    }
    
    // Expected: (0.9*0.5 + 0.7*0.3 + 0.5*0.2) / (0.5+0.3+0.2) = 0.76
    double expected = (0.9*0.5 + 0.7*0.3 + 0.5*0.2) / (0.5+0.3+0.2);
    
    printf("  Calculated overall score: %.3f\n", overall_score);
    printf("  Expected overall score: %.3f\n", expected);
    
    if (fabs(overall_score - expected) > 0.001) {
        print_error("Score Calculation", "Incorrect overall score");
        return 0;
    }
    
    print_success("Overall Score Calculation");
    
    // Test with invalid metrics
    metrics[1].is_valid = false;
    
    error = quality_validator_calculate_overall_score(metrics, 3, &overall_score);
    if (error != QV_SUCCESS) {
        print_error("Invalid Metrics", quality_validator_error_string(error));
        return 0;
    }
    
    // Expected: (0.9*0.5 + 0.5*0.2) / (0.5+0.2) = 0.786
    expected = (0.9*0.5 + 0.5*0.2) / (0.5+0.2);
    
    printf("  Score with invalid metric: %.3f\n", overall_score);
    printf("  Expected score: %.3f\n", expected);
    
    if (fabs(overall_score - expected) > 0.001) {
        print_error("Invalid Metrics", "Incorrect score with invalid metrics");
        return 0;
    }
    
    print_success("Invalid Metrics Handling");
    
    return 1;
}

// Main test runner
int main(void) {
    printf("Quality Validator Test Suite\n");
    printf("===========================\n");
    
    srand((unsigned int)time(NULL));
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    struct {
        const char* name;
        int (*test_func)(void);
    } tests[] = {
        {"Validator Creation", test_validator_creation},
        {"Basic Validation", test_basic_validation},
        {"Quality Metrics", test_quality_metrics},
        {"Compression Validation", test_compression_validation},
        {"Error Detection", test_error_detection},
        {"Custom Validators", test_custom_validators},
        {"Validation Rules", test_validation_rules},
        {"Statistics", test_statistics},
        {"Utility Functions", test_utility_functions},
        {"Overall Score Calculation", test_overall_score_calculation}
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
        printf("\n🎉 All tests passed! Quality Validator is working correctly.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the output above.\n");
        return 1;
    }
}
