/*
 * NeuralCompressionTestSuite.h
 * 
 * Comprehensive Test Suite for Neural Network Compression Protocol (NNCP)
 * Comprehensive testing framework for all NNCP components including
 * unit tests, integration tests, performance benchmarks, and validation tests
 */

#ifndef NEURAL_COMPRESSION_TEST_SUITE_H
#define NEURAL_COMPRESSION_TEST_SUITE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Include all NNCP components for testing
#include "../src/neural/core/ProgressiveCompressionEngine.h"
#include "../src/neural/validation/IntegrityValidator.h"
#include "../src/neural/robustness/RobustCompressionEngine.h"
#include "../src/neural/verification/PerformanceVerifier.h"
#include "../src/neural/precision/MixedPrecisionEngine.h"
#include "../src/neural/optimization/CacheOptimizer.h"
#include "../src/neural/parallel/ParallelProcessor.h"

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct NeuralCompressionTestSuite NeuralCompressionTestSuite;

// Test result severity levels
typedef enum {
    TEST_SEVERITY_PASS = 0,              // Test passed successfully
    TEST_SEVERITY_WARNING,               // Test passed with warnings
    TEST_SEVERITY_MINOR_FAIL,            // Minor test failure
    TEST_SEVERITY_MAJOR_FAIL,            // Major test failure
    TEST_SEVERITY_CRITICAL_FAIL          // Critical test failure
} TestSeverity;

// Test categories for comprehensive coverage
typedef enum {
    TEST_CATEGORY_UNIT = 0,              // Unit tests
    TEST_CATEGORY_INTEGRATION,           // Integration tests
    TEST_CATEGORY_PERFORMANCE,           // Performance benchmarks
    TEST_CATEGORY_STRESS,                // Stress tests
    TEST_CATEGORY_VALIDATION,            // Data validation tests
    TEST_CATEGORY_ROBUSTNESS,            // Error handling/robustness tests
    TEST_CATEGORY_COMPATIBILITY,        // Hardware compatibility tests
    TEST_CATEGORY_REGRESSION,            // Regression tests
    TEST_CATEGORY_SECURITY,              // Security tests
    TEST_CATEGORY_MEMORY,                // Memory leak/management tests
    TEST_CATEGORY_END_TO_END            // End-to-end system tests
} TestCategory;

// Test execution modes
typedef enum {
    TEST_MODE_QUICK = 0,                 // Quick test suite (5-10 minutes)
    TEST_MODE_STANDARD,                  // Standard test suite (30-60 minutes)
    TEST_MODE_COMPREHENSIVE,             // Comprehensive test suite (2-4 hours)
    TEST_MODE_STRESS,                    // Stress test mode (8+ hours)
    TEST_MODE_CUSTOM                     // Custom test configuration
} TestExecutionMode;

// Individual test result
typedef struct {
    uint32_t test_id;                    // Unique test identifier
    char test_name[128];                 // Test name
    TestCategory category;               // Test category
    TestSeverity severity;               // Test result severity
    bool test_passed;                    // Test pass/fail status
    uint64_t execution_time_ns;          // Test execution time
    uint64_t memory_peak_usage_bytes;    // Peak memory usage during test
    float performance_score;             // Performance score (0-100)
    uint32_t assertions_total;           // Total number of assertions
    uint32_t assertions_passed;          // Number of passed assertions
    uint32_t assertions_failed;          // Number of failed assertions
    char failure_reason[512];            // Failure reason (if failed)
    char test_details[1024];             // Detailed test information
    void* test_data;                     // Test-specific data
    size_t test_data_size;               // Size of test-specific data
} TestResult;

// Test suite configuration
typedef struct {
    TestExecutionMode execution_mode;    // Test execution mode
    bool enable_performance_benchmarks;  // Enable performance benchmarking
    bool enable_memory_leak_detection;   // Enable memory leak detection
    bool enable_stress_testing;          // Enable stress testing
    bool enable_hardware_validation;     // Enable hardware-specific validation
    bool enable_parallel_execution;      // Enable parallel test execution
    bool enable_regression_testing;      // Enable regression testing
    bool generate_detailed_reports;      // Generate detailed test reports
    bool stop_on_first_failure;          // Stop execution on first failure
    uint32_t test_timeout_seconds;       // Timeout for individual tests
    uint32_t max_concurrent_tests;       // Maximum concurrent tests
    float performance_regression_threshold; // Performance regression threshold
    char output_directory[256];          // Output directory for test results
    char test_data_directory[256];       // Directory containing test data
} TestSuiteConfiguration;

// Test suite statistics
typedef struct {
    uint32_t total_tests_executed;       // Total number of tests executed
    uint32_t tests_passed;               // Number of tests passed
    uint32_t tests_failed;               // Number of tests failed
    uint32_t tests_skipped;              // Number of tests skipped
    uint32_t tests_with_warnings;        // Number of tests with warnings
    uint64_t total_execution_time_ns;    // Total execution time
    uint64_t peak_memory_usage_bytes;    // Peak memory usage across all tests
    float overall_performance_score;     // Overall performance score
    uint32_t performance_regressions;    // Number of performance regressions
    uint32_t memory_leaks_detected;      // Number of memory leaks detected
    float test_coverage_percentage;      // Test coverage percentage
    char execution_summary[1024];        // Human-readable execution summary
} TestSuiteStatistics;

// Comprehensive test suite results
typedef struct {
    TestSuiteConfiguration configuration; // Test configuration used
    TestSuiteStatistics statistics;      // Test execution statistics
    TestResult* individual_results;      // Array of individual test results
    uint32_t result_count;               // Number of test results
    uint32_t result_capacity;            // Capacity of results array
    bool all_tests_passed;               // Overall test suite success
    bool performance_acceptable;         // Performance within acceptable limits
    bool memory_usage_acceptable;        // Memory usage within limits
    char overall_assessment[512];        // Overall test assessment
    char recommendations[2048];          // Recommendations for improvements
} TestSuiteResults;

// Test data generators for various scenarios
typedef struct {
    void* (*generate_random_neural_data)(size_t layer_count, size_t* layer_sizes, LayerType* layer_types);
    void* (*generate_pathological_data)(size_t data_size, const char* pathology_type);
    void* (*generate_benchmark_data)(size_t data_size, const char* benchmark_type);
    void* (*generate_regression_data)(const char* regression_scenario);
    void (*cleanup_test_data)(void* test_data);
} TestDataGenerators;

// Error codes for test suite
typedef enum {
    TEST_SUITE_SUCCESS = 0,
    TEST_SUITE_ERROR_INVALID_PARAM,
    TEST_SUITE_ERROR_MEMORY_ALLOCATION,
    TEST_SUITE_ERROR_INITIALIZATION_FAILED,
    TEST_SUITE_ERROR_TEST_EXECUTION_FAILED,
    TEST_SUITE_ERROR_CONFIGURATION_INVALID,
    TEST_SUITE_ERROR_TIMEOUT,
    TEST_SUITE_ERROR_RESOURCE_EXHAUSTED,
    TEST_SUITE_ERROR_HARDWARE_INCOMPATIBLE
} TestSuiteError;

// Core API Functions

/**
 * Create neural compression test suite instance
 * @param test_suite Pointer to store created test suite
 * @param config Test suite configuration
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_create(NeuralCompressionTestSuite** test_suite,
                                const TestSuiteConfiguration* config);

/**
 * Initialize test suite with system detection and preparation
 * @param test_suite Test suite instance
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_initialize(NeuralCompressionTestSuite* test_suite);

/**
 * Execute complete test suite
 * @param test_suite Test suite instance
 * @param results Output test suite results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_execute_all(NeuralCompressionTestSuite* test_suite,
                                      TestSuiteResults* results);

/**
 * Execute specific test category
 * @param test_suite Test suite instance
 * @param category Test category to execute
 * @param results Output test results for category
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_execute_category(NeuralCompressionTestSuite* test_suite,
                                           TestCategory category,
                                           TestSuiteResults* results);

// Unit Tests

/**
 * Execute unit tests for ProgressiveCompressionEngine
 * @param test_suite Test suite instance
 * @param results Output unit test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_unit_progressive_engine(NeuralCompressionTestSuite* test_suite,
                                                 TestSuiteResults* results);

/**
 * Execute unit tests for IntegrityValidator
 * @param test_suite Test suite instance
 * @param results Output unit test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_unit_integrity_validator(NeuralCompressionTestSuite* test_suite,
                                                  TestSuiteResults* results);

/**
 * Execute unit tests for RobustCompressionEngine
 * @param test_suite Test suite instance
 * @param results Output unit test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_unit_robust_engine(NeuralCompressionTestSuite* test_suite,
                                             TestSuiteResults* results);

/**
 * Execute unit tests for PerformanceVerifier
 * @param test_suite Test suite instance
 * @param results Output unit test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_unit_performance_verifier(NeuralCompressionTestSuite* test_suite,
                                                   TestSuiteResults* results);

/**
 * Execute unit tests for MixedPrecisionEngine
 * @param test_suite Test suite instance
 * @param results Output unit test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_unit_precision_engine(NeuralCompressionTestSuite* test_suite,
                                                TestSuiteResults* results);

// Integration Tests

/**
 * Execute end-to-end compression/decompression integration test
 * @param test_suite Test suite instance
 * @param model_data Neural network model data
 * @param model_size Size of model data
 * @param layer_info Layer information array
 * @param layer_count Number of layers
 * @param results Output integration test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_integration_end_to_end(NeuralCompressionTestSuite* test_suite,
                                                 const void* model_data,
                                                 size_t model_size,
                                                 const void* layer_info,
                                                 uint32_t layer_count,
                                                 TestSuiteResults* results);

/**
 * Execute multi-component integration tests
 * @param test_suite Test suite instance
 * @param results Output integration test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_integration_multi_component(NeuralCompressionTestSuite* test_suite,
                                                     TestSuiteResults* results);

/**
 * Execute Apple Silicon hardware integration tests
 * @param test_suite Test suite instance
 * @param results Output integration test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_integration_apple_silicon(NeuralCompressionTestSuite* test_suite,
                                                   TestSuiteResults* results);

// Performance Benchmarks

/**
 * Execute comprehensive performance benchmarks
 * @param test_suite Test suite instance
 * @param benchmark_datasets Array of benchmark datasets
 * @param dataset_count Number of benchmark datasets
 * @param results Output benchmark results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_benchmark_performance(NeuralCompressionTestSuite* test_suite,
                                               void* const* benchmark_datasets,
                                               uint32_t dataset_count,
                                               TestSuiteResults* results);

/**
 * Execute compression ratio benchmarks across different model types
 * @param test_suite Test suite instance
 * @param results Output benchmark results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_benchmark_compression_ratio(NeuralCompressionTestSuite* test_suite,
                                                     TestSuiteResults* results);

/**
 * Execute throughput benchmarks for different data sizes
 * @param test_suite Test suite instance
 * @param data_sizes Array of data sizes to benchmark
 * @param size_count Number of data sizes
 * @param results Output benchmark results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_benchmark_throughput(NeuralCompressionTestSuite* test_suite,
                                               const size_t* data_sizes,
                                               uint32_t size_count,
                                               TestSuiteResults* results);

// Stress Tests

/**
 * Execute memory stress tests with large models
 * @param test_suite Test suite instance
 * @param max_memory_usage_gb Maximum memory usage to test (GB)
 * @param results Output stress test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_stress_memory(NeuralCompressionTestSuite* test_suite,
                                       float max_memory_usage_gb,
                                       TestSuiteResults* results);

/**
 * Execute thermal stress tests with sustained load
 * @param test_suite Test suite instance
 * @param stress_duration_minutes Duration of stress test
 * @param results Output stress test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_stress_thermal(NeuralCompressionTestSuite* test_suite,
                                         uint32_t stress_duration_minutes,
                                         TestSuiteResults* results);

/**
 * Execute concurrent processing stress tests
 * @param test_suite Test suite instance
 * @param max_concurrent_operations Maximum concurrent operations
 * @param results Output stress test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_stress_concurrency(NeuralCompressionTestSuite* test_suite,
                                             uint32_t max_concurrent_operations,
                                             TestSuiteResults* results);

// Validation and Regression Tests

/**
 * Execute numerical precision validation tests
 * @param test_suite Test suite instance
 * @param precision_tolerance Maximum acceptable precision loss
 * @param results Output validation test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_validate_numerical_precision(NeuralCompressionTestSuite* test_suite,
                                                      float precision_tolerance,
                                                      TestSuiteResults* results);

/**
 * Execute data integrity validation tests
 * @param test_suite Test suite instance
 * @param results Output validation test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_validate_data_integrity(NeuralCompressionTestSuite* test_suite,
                                                  TestSuiteResults* results);

/**
 * Execute regression tests against known reference results
 * @param test_suite Test suite instance
 * @param reference_data_path Path to reference test data
 * @param results Output regression test results
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_regression_reference_validation(NeuralCompressionTestSuite* test_suite,
                                                         const char* reference_data_path,
                                                         TestSuiteResults* results);

// Test Data Management and Generation

/**
 * Generate synthetic neural network models for testing
 * @param test_suite Test suite instance
 * @param model_specifications Array of model specifications
 * @param spec_count Number of model specifications
 * @param generated_models Output array of generated models
 * @param model_count Output number of generated models
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_generate_test_models(NeuralCompressionTestSuite* test_suite,
                                               const void* model_specifications,
                                               uint32_t spec_count,
                                               void** generated_models,
                                               uint32_t* model_count);

/**
 * Load real-world neural network models for testing
 * @param test_suite Test suite instance
 * @param model_directory Directory containing real models
 * @param loaded_models Output array of loaded models
 * @param model_count Output number of loaded models
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_load_real_models(NeuralCompressionTestSuite* test_suite,
                                          const char* model_directory,
                                          void** loaded_models,
                                          uint32_t* model_count);

/**
 * Create pathological test cases for edge case testing
 * @param test_suite Test suite instance
 * @param pathology_types Array of pathology type specifications
 * @param pathology_count Number of pathology types
 * @param pathological_data Output pathological test data
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_create_pathological_cases(NeuralCompressionTestSuite* test_suite,
                                                   const char* const* pathology_types,
                                                   uint32_t pathology_count,
                                                   void** pathological_data);

// Reporting and Analysis

/**
 * Generate comprehensive test report
 * @param test_suite Test suite instance
 * @param results Test suite results to analyze
 * @param report_format Output report format ("html", "json", "xml", "text")
 * @param output_file Output file path for report
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_generate_report(NeuralCompressionTestSuite* test_suite,
                                         const TestSuiteResults* results,
                                         const char* report_format,
                                         const char* output_file);

/**
 * Analyze performance trends and regressions
 * @param test_suite Test suite instance
 * @param historical_results Array of historical test results
 * @param result_count Number of historical results
 * @param trend_analysis Output trend analysis
 * @param analysis_buffer_size Size of trend analysis buffer
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_analyze_performance_trends(NeuralCompressionTestSuite* test_suite,
                                                    const TestSuiteResults* historical_results,
                                                    uint32_t result_count,
                                                    char* trend_analysis,
                                                    size_t analysis_buffer_size);

/**
 * Compare test results between different system configurations
 * @param test_suite Test suite instance
 * @param baseline_results Baseline test results
 * @param comparison_results Comparison test results
 * @param comparison_report Output comparison report
 * @param report_buffer_size Size of comparison report buffer
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_compare_configurations(NeuralCompressionTestSuite* test_suite,
                                                const TestSuiteResults* baseline_results,
                                                const TestSuiteResults* comparison_results,
                                                char* comparison_report,
                                                size_t report_buffer_size);

// Configuration and Utility Functions

/**
 * Create default test suite configuration
 * @param config Output default configuration
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_create_default_config(TestSuiteConfiguration* config);

/**
 * Create quick test configuration for rapid validation
 * @param config Output quick test configuration
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_create_quick_config(TestSuiteConfiguration* config);

/**
 * Create comprehensive test configuration for thorough validation
 * @param config Output comprehensive test configuration
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_create_comprehensive_config(TestSuiteConfiguration* config);

/**
 * Validate test suite configuration
 * @param config Configuration to validate
 * @param is_valid Output boolean for configuration validity
 * @param validation_message Output validation message buffer
 * @param message_size Size of validation message buffer
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_validate_config(const TestSuiteConfiguration* config,
                                         bool* is_valid,
                                         char* validation_message,
                                         size_t message_size);

/**
 * Estimate test execution time for given configuration
 * @param config Test suite configuration
 * @param estimated_time_minutes Output estimated execution time
 * @return TEST_SUITE_SUCCESS on success, error code on failure
 */
TestSuiteError test_suite_estimate_execution_time(const TestSuiteConfiguration* config,
                                                 uint32_t* estimated_time_minutes);

/**
 * Destroy test suite and free resources
 * @param test_suite Test suite instance to destroy
 */
void test_suite_destroy(NeuralCompressionTestSuite* test_suite);

// Utility Functions

/**
 * Get error string for test suite error code
 * @param error_code TestSuiteError code
 * @return Human-readable error message
 */
const char* test_suite_get_error_string(TestSuiteError error_code);

/**
 * Get test category string
 * @param category Test category enum
 * @return Human-readable category name
 */
const char* test_suite_get_category_string(TestCategory category);

/**
 * Get test severity string
 * @param severity Test severity enum
 * @return Human-readable severity name
 */
const char* test_suite_get_severity_string(TestSeverity severity);

/**
 * Get test execution mode string
 * @param mode Test execution mode enum
 * @return Human-readable mode name
 */
const char* test_suite_get_mode_string(TestExecutionMode mode);

/**
 * Calculate test coverage percentage
 * @param total_possible_tests Total possible tests
 * @param executed_tests Tests actually executed
 * @return Coverage percentage (0-100)
 */
float test_suite_calculate_coverage(uint32_t total_possible_tests, uint32_t executed_tests);

/**
 * Determine if performance regression occurred
 * @param baseline_score Baseline performance score
 * @param current_score Current performance score
 * @param regression_threshold Regression threshold
 * @return True if regression detected
 */
bool test_suite_detect_performance_regression(float baseline_score,
                                             float current_score,
                                             float regression_threshold);

// Constants for test suite

// Test execution timing
#define QUICK_TEST_TIMEOUT_SECONDS 300        // 5 minutes for quick tests
#define STANDARD_TEST_TIMEOUT_SECONDS 1800    // 30 minutes for standard tests
#define COMPREHENSIVE_TEST_TIMEOUT_SECONDS 7200 // 2 hours for comprehensive tests

// Performance thresholds
#define PERFORMANCE_REGRESSION_THRESHOLD 0.05f  // 5% performance regression threshold
#define MEMORY_LEAK_THRESHOLD_BYTES 1048576    // 1MB memory leak threshold
#define THERMAL_STRESS_THRESHOLD_CELSIUS 85.0f // Thermal stress threshold

// Test data sizes
#define SMALL_MODEL_SIZE_MB 10                 // Small model size for testing
#define MEDIUM_MODEL_SIZE_MB 100               // Medium model size for testing
#define LARGE_MODEL_SIZE_MB 1000               // Large model size for testing
#define STRESS_MODEL_SIZE_MB 10000             // Stress test model size

// Coverage targets
#define TARGET_TEST_COVERAGE_PERCENTAGE 90.0f  // Target test coverage
#define MINIMUM_TEST_COVERAGE_PERCENTAGE 80.0f // Minimum acceptable coverage

#ifdef __cplusplus
}
#endif

#endif // NEURAL_COMPRESSION_TEST_SUITE_H
