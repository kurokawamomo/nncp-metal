/*
 * Comprehensive CUDA Compatibility Tests
 * 
 * Test suite for validating CUDA compatibility across all profiles, file sizes,
 * and mathematical operations to ensure perfect equivalence with original CUDA implementation.
 */

#ifndef CUDA_COMPATIBILITY_TESTS_H
#define CUDA_COMPATIBILITY_TESTS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Test result structures
typedef enum {
    CUDA_TEST_SUCCESS = 0,
    CUDA_TEST_FAILURE,
    CUDA_TEST_SKIP,
    CUDA_TEST_ERROR
} CUDATestResult;

typedef struct {
    const char* test_name;
    CUDATestResult result;
    double execution_time_ms;
    size_t memory_used_bytes;
    const char* failure_reason;
    float precision_error;
    bool cuda_equivalent;
} CUDATestCase;

typedef struct {
    size_t total_tests;
    size_t passed_tests;
    size_t failed_tests;
    size_t skipped_tests;
    size_t error_tests;
    double total_execution_time_ms;
    size_t total_memory_used;
    float max_precision_error;
    bool all_cuda_equivalent;
} CUDATestSuite;

// Test categories
typedef enum {
    CUDA_TEST_CATEGORY_PROFILES = 0,
    CUDA_TEST_CATEGORY_MATH_OPERATIONS,
    CUDA_TEST_CATEGORY_TENSOR_OPERATIONS,
    CUDA_TEST_CATEGORY_LSTM_OPERATIONS,
    CUDA_TEST_CATEGORY_TRANSFORMER_OPERATIONS,
    CUDA_TEST_CATEGORY_FILE_HANDLING,
    CUDA_TEST_CATEGORY_ERROR_HANDLING,
    CUDA_TEST_CATEGORY_PARAMETER_VALIDATION,
    CUDA_TEST_CATEGORY_MEMORY_MANAGEMENT,
    CUDA_TEST_CATEGORY_PERFORMANCE,
    CUDA_TEST_CATEGORY_COUNT
} CUDATestCategory;

// Main test functions
CUDATestSuite* cuda_compatibility_test_run_all(void);
CUDATestSuite* cuda_compatibility_test_run_category(CUDATestCategory category);
void cuda_compatibility_test_print_results(const CUDATestSuite* suite);
void cuda_compatibility_test_free_suite(CUDATestSuite* suite);

// Profile compatibility tests
CUDATestSuite* cuda_test_profiles_compatibility(void);
bool cuda_test_profile_parameters(const char* profile_name);
bool cuda_test_profile_seg_len_relationship(const char* profile_name);
bool cuda_test_profile_memory_constraints(const char* profile_name);
bool cuda_test_profile_file_size_compatibility(const char* profile_name);

// Mathematical operations tests
CUDATestSuite* cuda_test_math_operations(void);
bool cuda_test_matrix_multiplication(void);
bool cuda_test_tensor_arithmetic(void);
bool cuda_test_activation_functions(void);
bool cuda_test_softmax_operation(void);
bool cuda_test_layer_norm_operation(void);
bool cuda_test_random_generation(void);

// LSTM operations tests
CUDATestSuite* cuda_test_lstm_operations(void);
bool cuda_test_lstm_cell_computation(void);
bool cuda_test_lstm_state_management(void);
bool cuda_test_lstm_sequence_processing(void);
bool cuda_test_enhanced_lstm_context(void);

// Transformer operations tests
CUDATestSuite* cuda_test_transformer_operations(void);
bool cuda_test_attention_mechanism(void);
bool cuda_test_multi_head_attention(void);
bool cuda_test_feed_forward_network(void);
bool cuda_test_positional_encoding(void);

// File handling tests
CUDATestSuite* cuda_test_file_handling(void);
bool cuda_test_small_file_processing(void);
bool cuda_test_large_file_processing(void);
bool cuda_test_file_size_thresholds(void);
bool cuda_test_compression_ratios(void);

// Error handling tests
CUDATestSuite* cuda_test_error_handling(void);
bool cuda_test_error_codes_compatibility(void);
bool cuda_test_error_severity_mapping(void);
bool cuda_test_error_chain_management(void);
bool cuda_test_error_recovery(void);

// Parameter validation tests
CUDATestSuite* cuda_test_parameter_validation(void);
bool cuda_test_seg_len_validation(void);
bool cuda_test_batch_size_validation(void);
bool cuda_test_memory_budget_validation(void);
bool cuda_test_tensor_dimension_validation(void);

// Memory management tests
CUDATestSuite* cuda_test_memory_management(void);
bool cuda_test_tensor_allocation(void);
bool cuda_test_metal_buffer_integration(void);
bool cuda_test_memory_leak_detection(void);
bool cuda_test_memory_alignment(void);

// Performance tests
CUDATestSuite* cuda_test_performance(void);
bool cuda_test_operation_throughput(void);
bool cuda_test_memory_bandwidth(void);
bool cuda_test_computation_latency(void);
bool cuda_test_scalability(void);

// Stress tests
CUDATestSuite* cuda_test_stress_scenarios(void);
bool cuda_test_large_tensor_operations(void);
bool cuda_test_long_sequence_processing(void);
bool cuda_test_concurrent_operations(void);
bool cuda_test_memory_pressure(void);

// Precision and accuracy tests
CUDATestSuite* cuda_test_precision_accuracy(void);
bool cuda_test_numerical_stability(void);
bool cuda_test_floating_point_precision(void);
bool cuda_test_deterministic_computation(void);
bool cuda_test_reference_comparison(void);

// Integration tests
CUDATestSuite* cuda_test_integration_scenarios(void);
bool cuda_test_end_to_end_compression(void);
bool cuda_test_profile_switching(void);
bool cuda_test_multi_algorithm_pipeline(void);
bool cuda_test_real_world_scenarios(void);

// Utility functions
bool cuda_test_setup_environment(void);
void cuda_test_cleanup_environment(void);
bool cuda_test_create_test_data(size_t file_size, const char* pattern);
bool cuda_test_verify_cuda_equivalence(const void* metal_result, const void* cuda_reference, size_t size);
float cuda_test_measure_precision_error(const float* a, const float* b, size_t count);
bool cuda_test_validate_compression_ratio(float ratio, float expected_min, float expected_max);

// Test data generators
void* cuda_test_generate_random_tensor(const size_t* shape, size_t ndim);
void* cuda_test_generate_structured_data(size_t size, const char* pattern);
void* cuda_test_generate_enwik8_sample(size_t size);
void* cuda_test_generate_lstm_test_sequence(size_t length);

// Benchmark and comparison utilities
typedef struct {
    double cuda_time_ms;
    double metal_time_ms;
    float precision_error;
    bool results_equivalent;
    size_t memory_used;
} CUDAComparison;

CUDAComparison* cuda_test_benchmark_operation(const char* operation_name, 
                                             void (*metal_func)(void), 
                                             void (*cuda_ref_func)(void));

// Configuration for test execution
typedef struct {
    bool enable_performance_tests;
    bool enable_stress_tests;
    bool enable_precision_tests;
    bool enable_memory_tests;
    float precision_tolerance;
    size_t max_test_duration_ms;
    bool abort_on_first_failure;
    bool verbose_output;
    const char* test_data_directory;
    const char* reference_data_directory;
} CUDATestConfig;

void cuda_test_set_config(const CUDATestConfig* config);
const CUDATestConfig* cuda_test_get_config(void);

#ifdef __cplusplus
}
#endif

#endif // CUDA_COMPATIBILITY_TESTS_H