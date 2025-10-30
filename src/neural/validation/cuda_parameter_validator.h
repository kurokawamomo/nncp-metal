/*
 * CUDA Parameter Validation System
 * 
 * Comprehensive validation system that ensures all parameters match
 * original CUDA implementation constraints and behavioral requirements.
 * 
 * Based on original CUDA NNCP implementation analysis for exact compatibility.
 */

#ifndef CUDA_PARAMETER_VALIDATOR_H
#define CUDA_PARAMETER_VALIDATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../compatibility/cuda_math_compat.h"
#include "../config/cuda_profiles.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validation severity levels
typedef enum {
    CUDA_VALIDATION_INFO = 0,
    CUDA_VALIDATION_WARNING,
    CUDA_VALIDATION_ERROR,
    CUDA_VALIDATION_CRITICAL
} CUDAValidationSeverity;

// Validation result for a single parameter
typedef struct {
    const char* parameter_name;
    bool is_valid;
    CUDAValidationSeverity severity;
    char message[256];
    double expected_value;
    double actual_value;
    const char* cuda_constraint_description;
} CUDAParameterValidationResult;

// Overall validation report
typedef struct {
    bool overall_valid;
    size_t total_checks;
    size_t passed_checks;
    size_t warning_count;
    size_t error_count;
    size_t critical_error_count;
    
    // Detailed results
    CUDAParameterValidationResult* results;
    size_t result_count;
    size_t result_capacity;
    
    // Summary information
    char profile_name[64];
    char validation_timestamp[64];
    double validation_duration_ms;
} CUDAValidationReport;

// Core validation functions
CUDAValidationReport* cuda_validator_validate_profile(const CUDAProfile* profile);
CUDAValidationReport* cuda_validator_validate_model_params(const CUDAModelParams* params);
CUDAValidationReport* cuda_validator_validate_runtime_config(const CUDAModelParams* params);
CUDAValidationReport* cuda_validator_validate_tensor_shapes(const CUDAModelParams* params);

// Specific parameter validation functions
bool cuda_validator_check_seg_len_constraints(int32_t seg_len, CUDAValidationReport* report);
bool cuda_validator_check_batch_size_constraints(int32_t batch_size, CUDAValidationReport* report);
bool cuda_validator_check_model_dimension_constraints(const CUDAModelParams* params, CUDAValidationReport* report);
bool cuda_validator_check_lstm_constraints(const CUDAModelParams* params, CUDAValidationReport* report);
bool cuda_validator_check_memory_constraints(const CUDAModelParams* params, CUDAValidationReport* report);
bool cuda_validator_check_numerical_precision_constraints(const CUDAMathConfig* config, CUDAValidationReport* report);

// Critical CUDA compatibility checks
bool cuda_validator_verify_train_len_seg_len_relationship(int32_t train_len, int32_t seg_len, CUDAValidationReport* report);
bool cuda_validator_verify_cuda_profile_consistency(const CUDAProfile* profile, CUDAValidationReport* report);
bool cuda_validator_verify_mathematical_compatibility(const CUDAMathConfig* math_config, CUDAValidationReport* report);
bool cuda_validator_verify_tensor_layout_compatibility(const CUDAModelParams* params, CUDAValidationReport* report);

// Memory and performance validation
bool cuda_validator_check_memory_budget_feasibility(const CUDAModelParams* params, CUDAValidationReport* report);
bool cuda_validator_check_performance_expectations(const CUDAModelParams* params, CUDAValidationReport* report);
bool cuda_validator_check_hardware_compatibility(const CUDAModelParams* params, CUDAValidationReport* report);

// File size and data validation
bool cuda_validator_validate_file_size_compatibility(const CUDAProfile* profile, size_t file_size, CUDAValidationReport* report);
bool cuda_validator_validate_input_data_format(const uint8_t* data, size_t data_size, 
                                               const CUDAModelParams* params, CUDAValidationReport* report);

// Cross-parameter validation (checks relationships between parameters)
bool cuda_validator_check_parameter_relationships(const CUDAModelParams* params, CUDAValidationReport* report);
bool cuda_validator_check_scaling_constraints(const CUDAModelParams* params, CUDAValidationReport* report);
bool cuda_validator_check_architectural_consistency(const CUDAModelParams* params, CUDAValidationReport* report);

// Validation report management
CUDAValidationReport* cuda_validation_report_create(const char* profile_name);
void cuda_validation_report_free(CUDAValidationReport* report);
void cuda_validation_report_add_result(CUDAValidationReport* report, 
                                      const char* parameter_name,
                                      bool is_valid,
                                      CUDAValidationSeverity severity,
                                      const char* message,
                                      double expected_value,
                                      double actual_value,
                                      const char* constraint_description);

// Report output and analysis
void cuda_validation_report_print(const CUDAValidationReport* report, bool verbose);
void cuda_validation_report_print_summary(const CUDAValidationReport* report);
void cuda_validation_report_print_errors_only(const CUDAValidationReport* report);
bool cuda_validation_report_save_to_file(const CUDAValidationReport* report, const char* filename);

// Validation presets for different scenarios
CUDAValidationReport* cuda_validator_validate_for_compression(const CUDAProfile* profile, 
                                                             size_t file_size,
                                                             bool strict_mode);
CUDAValidationReport* cuda_validator_validate_for_decompression(const CUDAModelParams* params);
CUDAValidationReport* cuda_validator_validate_for_training(const CUDAModelParams* params);

// Advanced validation features
bool cuda_validator_compare_with_reference_implementation(const CUDAModelParams* params,
                                                         const char* reference_config_path,
                                                         CUDAValidationReport* report);
bool cuda_validator_benchmark_parameter_performance(const CUDAModelParams* params,
                                                   CUDAValidationReport* report);

// Validation configuration
typedef struct {
    bool enable_strict_mode;           // Enforce strictest CUDA compatibility
    bool enable_performance_checks;    // Include performance validation
    bool enable_memory_checks;         // Include memory usage validation
    bool enable_cross_validation;      // Check parameter relationships
    bool enable_benchmarking;          // Include performance benchmarks
    double precision_tolerance;        // Numerical precision tolerance
    bool fail_on_warnings;            // Treat warnings as errors
} CUDAValidatorConfig;

CUDAValidatorConfig* cuda_validator_config_create_default(void);
CUDAValidatorConfig* cuda_validator_config_create_strict(void);
CUDAValidatorConfig* cuda_validator_config_create_permissive(void);
void cuda_validator_config_free(CUDAValidatorConfig* config);

// Context-aware validation (uses validation config)
CUDAValidationReport* cuda_validator_validate_with_config(const CUDAProfile* profile,
                                                         const CUDAValidatorConfig* validator_config);

// Error handling and diagnostics
typedef enum {
    CUDA_VALIDATOR_SUCCESS = 0,
    CUDA_VALIDATOR_ERROR_INVALID_INPUT,
    CUDA_VALIDATOR_ERROR_ALLOCATION_FAILED,
    CUDA_VALIDATOR_ERROR_PROFILE_INVALID,
    CUDA_VALIDATOR_ERROR_CONSTRAINT_VIOLATED,
    CUDA_VALIDATOR_ERROR_COMPATIBILITY_CHECK_FAILED,
    CUDA_VALIDATOR_ERROR_REFERENCE_MISMATCH
} CUDAValidatorError;

const char* cuda_validator_error_string(CUDAValidatorError error);

// Utility functions for constraint definitions
typedef struct {
    const char* name;
    double min_value;
    double max_value;
    bool must_be_power_of_two;
    bool must_be_multiple_of;
    int32_t multiple_of_value;
    const char* description;
} CUDAParameterConstraint;

const CUDAParameterConstraint* cuda_validator_get_seg_len_constraints(void);
const CUDAParameterConstraint* cuda_validator_get_batch_size_constraints(void);
const CUDAParameterConstraint* cuda_validator_get_hidden_size_constraints(void);
const CUDAParameterConstraint* cuda_validator_get_learning_rate_constraints(void);

// Quick validation helpers
bool cuda_validator_quick_check_compatibility(const CUDAProfile* profile);
bool cuda_validator_quick_check_seg_len(int32_t seg_len);
bool cuda_validator_quick_check_batch_size(int32_t batch_size);
bool cuda_validator_quick_check_train_len_relationship(int32_t train_len, int32_t seg_len);

#ifdef __cplusplus
}
#endif

#endif // CUDA_PARAMETER_VALIDATOR_H