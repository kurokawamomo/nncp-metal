/*
 * CUDA Parameter Validation System Implementation
 * 
 * Comprehensive validation ensuring all parameters match original CUDA 
 * implementation constraints and behavioral requirements.
 */

#include "cuda_parameter_validator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Error message strings
static const char* cuda_validator_error_messages[] = {
    "Success",
    "Invalid input parameters",
    "Memory allocation failed",
    "Profile validation failed",
    "CUDA constraint violated",
    "Compatibility check failed",
    "Reference implementation mismatch"
};

// CUDA parameter constraints (based on original implementation analysis)
static const CUDAParameterConstraint seg_len_constraints = {
    .name = "seg_len",
    .min_value = 1,
    .max_value = 1024,
    .must_be_power_of_two = false,
    .must_be_multiple_of = false,
    .multiple_of_value = 1,
    .description = "Segment length - critical for CUDA compatibility (standard values: 20, 32, 64)"
};

static const CUDAParameterConstraint batch_size_constraints = {
    .name = "batch_size",
    .min_value = 1,
    .max_value = 1024,
    .must_be_power_of_two = true,  // Optimal for CUDA performance
    .must_be_multiple_of = false,
    .multiple_of_value = 1,
    .description = "Batch size - should be power of 2 for optimal CUDA performance"
};

static const CUDAParameterConstraint hidden_size_constraints = {
    .name = "hidden_size",
    .min_value = 32,
    .max_value = 4096,
    .must_be_power_of_two = false,
    .must_be_multiple_of = true,
    .multiple_of_value = 32,  // 32-byte alignment for CUDA memory coalescing
    .description = "Hidden size - must be multiple of 32 for CUDA memory alignment"
};

static const CUDAParameterConstraint learning_rate_constraints = {
    .name = "learning_rate",
    .min_value = 1e-6,
    .max_value = 1.0,
    .must_be_power_of_two = false,
    .must_be_multiple_of = false,
    .multiple_of_value = 1,
    .description = "Learning rate - must be within numerical stability range"
};

// Validation report management
CUDAValidationReport* cuda_validation_report_create(const char* profile_name) {
    CUDAValidationReport* report = (CUDAValidationReport*)calloc(1, sizeof(CUDAValidationReport));
    if (!report) {
        return NULL;
    }
    
    report->overall_valid = true;
    report->result_capacity = 32;  // Initial capacity
    report->results = (CUDAParameterValidationResult*)calloc(report->result_capacity, 
                                                             sizeof(CUDAParameterValidationResult));
    if (!report->results) {
        free(report);
        return NULL;
    }
    
    if (profile_name) {
        strncpy(report->profile_name, profile_name, sizeof(report->profile_name) - 1);
    }
    
    // Set timestamp
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    strftime(report->validation_timestamp, sizeof(report->validation_timestamp), 
             "%Y-%m-%d %H:%M:%S", tm_info);
    
    return report;
}

void cuda_validation_report_free(CUDAValidationReport* report) {
    if (report) {
        if (report->results) {
            free(report->results);
        }
        free(report);
    }
}

void cuda_validation_report_add_result(CUDAValidationReport* report, 
                                      const char* parameter_name,
                                      bool is_valid,
                                      CUDAValidationSeverity severity,
                                      const char* message,
                                      double expected_value,
                                      double actual_value,
                                      const char* constraint_description) {
    if (!report) {
        return;
    }
    
    // Resize results array if needed
    if (report->result_count >= report->result_capacity) {
        report->result_capacity *= 2;
        report->results = (CUDAParameterValidationResult*)realloc(report->results,
            report->result_capacity * sizeof(CUDAParameterValidationResult));
        if (!report->results) {
            return;
        }
    }
    
    CUDAParameterValidationResult* result = &report->results[report->result_count];
    memset(result, 0, sizeof(CUDAParameterValidationResult));
    
    result->parameter_name = parameter_name;
    result->is_valid = is_valid;
    result->severity = severity;
    result->expected_value = expected_value;
    result->actual_value = actual_value;
    result->cuda_constraint_description = constraint_description;
    
    if (message) {
        strncpy(result->message, message, sizeof(result->message) - 1);
    }
    
    // Update report statistics
    report->result_count++;
    report->total_checks++;
    
    if (is_valid) {
        report->passed_checks++;
    } else {
        report->overall_valid = false;
        
        switch (severity) {
            case CUDA_VALIDATION_WARNING:
                report->warning_count++;
                break;
            case CUDA_VALIDATION_ERROR:
                report->error_count++;
                break;
            case CUDA_VALIDATION_CRITICAL:
                report->critical_error_count++;
                break;
            default:
                break;
        }
    }
}

// Core validation functions
CUDAValidationReport* cuda_validator_validate_profile(const CUDAProfile* profile) {
    if (!profile) {
        return NULL;
    }
    
    CUDAValidationReport* report = cuda_validation_report_create(profile->name);
    if (!report) {
        return NULL;
    }
    
    clock_t start_time = clock();
    
    // Validate model parameters
    if (!cuda_validator_check_seg_len_constraints(profile->params.seg_len, report)) {
        // Error already added to report
    }
    
    if (!cuda_validator_check_batch_size_constraints(profile->params.batch_size, report)) {
        // Error already added to report
    }
    
    if (!cuda_validator_check_model_dimension_constraints(&profile->params, report)) {
        // Error already added to report
    }
    
    if (!cuda_validator_check_lstm_constraints(&profile->params, report)) {
        // Error already added to report
    }
    
    if (!cuda_validator_check_memory_constraints(&profile->params, report)) {
        // Error already added to report
    }
    
    // Critical CUDA compatibility checks
    if (!cuda_validator_verify_cuda_profile_consistency(profile, report)) {
        // Error already added to report
    }
    
    // Parameter relationship checks
    if (!cuda_validator_check_parameter_relationships(&profile->params, report)) {
        // Error already added to report
    }
    
    clock_t end_time = clock();
    report->validation_duration_ms = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000.0;
    
    return report;
}

CUDAValidationReport* cuda_validator_validate_runtime_config(const CUDAModelParams* params) {
    if (!params) {
        return NULL;
    }
    
    CUDAValidationReport* report = cuda_validation_report_create("runtime_config");
    if (!report) {
        return NULL;
    }
    
    // Critical train_len = seg_len check
    if (!cuda_validator_verify_train_len_seg_len_relationship(params->train_len, params->seg_len, report)) {
        // Error already added to report
    }
    
    // Basic parameter validation
    if (!cuda_validator_check_seg_len_constraints(params->seg_len, report)) {
        // Error already added to report
    }
    
    if (!cuda_validator_check_batch_size_constraints(params->batch_size, report)) {
        // Error already added to report
    }
    
    // Mathematical compatibility validation
    if (params->math_config) {
        if (!cuda_validator_check_numerical_precision_constraints(params->math_config, report)) {
            // Error already added to report
        }
    }
    
    return report;
}

// Specific parameter validation functions
bool cuda_validator_check_seg_len_constraints(int32_t seg_len, CUDAValidationReport* report) {
    const CUDAParameterConstraint* constraint = &seg_len_constraints;
    bool is_valid = true;
    
    // Check basic range
    if (seg_len < constraint->min_value || seg_len > constraint->max_value) {
        cuda_validation_report_add_result(report, constraint->name, false, CUDA_VALIDATION_ERROR,
            "seg_len outside valid range", 
            (constraint->min_value + constraint->max_value) / 2, seg_len,
            constraint->description);
        is_valid = false;
    }
    
    // Check for standard CUDA values
    if (seg_len != 20 && seg_len != 32 && seg_len != 64) {
        cuda_validation_report_add_result(report, constraint->name, false, CUDA_VALIDATION_WARNING,
            "seg_len is non-standard (CUDA uses 20, 32, 64)",
            32, seg_len,  // Default is 32
            "Standard CUDA seg_len values for compatibility");
        // This is a warning, not an error
    }
    
    if (is_valid) {
        cuda_validation_report_add_result(report, constraint->name, true, CUDA_VALIDATION_INFO,
            "seg_len within valid range", seg_len, seg_len, constraint->description);
    }
    
    return is_valid;
}

bool cuda_validator_check_batch_size_constraints(int32_t batch_size, CUDAValidationReport* report) {
    const CUDAParameterConstraint* constraint = &batch_size_constraints;
    bool is_valid = true;
    
    // Check basic range
    if (batch_size < constraint->min_value || batch_size > constraint->max_value) {
        cuda_validation_report_add_result(report, constraint->name, false, CUDA_VALIDATION_ERROR,
            "batch_size outside valid range",
            (constraint->min_value + constraint->max_value) / 2, batch_size,
            constraint->description);
        is_valid = false;
    }
    
    // Check power-of-2 requirement for optimal CUDA performance
    if (constraint->must_be_power_of_two && (batch_size & (batch_size - 1)) != 0) {
        cuda_validation_report_add_result(report, constraint->name, false, CUDA_VALIDATION_WARNING,
            "batch_size is not power-of-2 (may impact CUDA performance)",
            32, batch_size,  // Suggest 32 as default
            "Power-of-2 batch sizes optimize CUDA memory access patterns");
        // This is a warning for performance, not correctness
    }
    
    if (is_valid) {
        cuda_validation_report_add_result(report, constraint->name, true, CUDA_VALIDATION_INFO,
            "batch_size within valid range", batch_size, batch_size, constraint->description);
    }
    
    return is_valid;
}

bool cuda_validator_check_model_dimension_constraints(const CUDAModelParams* params, CUDAValidationReport* report) {
    if (!params || !report) {
        return false;
    }
    
    bool is_valid = true;
    
    // Check hidden size alignment
    const CUDAParameterConstraint* hidden_constraint = &hidden_size_constraints;
    if (params->lstm_hidden_size < hidden_constraint->min_value || 
        params->lstm_hidden_size > hidden_constraint->max_value) {
        cuda_validation_report_add_result(report, "lstm_hidden_size", false, CUDA_VALIDATION_ERROR,
            "LSTM hidden size outside valid range",
            256, params->lstm_hidden_size, hidden_constraint->description);
        is_valid = false;
    }
    
    // Check 32-byte alignment for CUDA memory coalescing
    if (hidden_constraint->must_be_multiple_of && 
        (params->lstm_hidden_size % hidden_constraint->multiple_of_value) != 0) {
        cuda_validation_report_add_result(report, "lstm_hidden_size", false, CUDA_VALIDATION_WARNING,
            "LSTM hidden size not aligned to 32 bytes (may impact CUDA performance)",
            ((params->lstm_hidden_size / 32) + 1) * 32, params->lstm_hidden_size,
            "32-byte alignment optimizes CUDA memory access");
    }
    
    // Check model dimension consistency
    if (params->d_model <= 0 || params->d_model > 4096) {
        cuda_validation_report_add_result(report, "d_model", false, CUDA_VALIDATION_ERROR,
            "Model dimension outside valid range",
            512, params->d_model, "Model dimension must be positive and reasonable");
        is_valid = false;
    }
    
    // Check feed-forward dimension
    if (params->d_ff <= 0 || params->d_ff > 16384) {
        cuda_validation_report_add_result(report, "d_ff", false, CUDA_VALIDATION_ERROR,
            "Feed-forward dimension outside valid range",
            2048, params->d_ff, "Feed-forward dimension must be positive and reasonable");
        is_valid = false;
    }
    
    // Check typical relationship: d_ff = 4 * d_model
    if (params->d_ff != 4 * params->d_model) {
        cuda_validation_report_add_result(report, "d_ff_relationship", false, CUDA_VALIDATION_WARNING,
            "d_ff != 4 * d_model (non-standard Transformer architecture)",
            4 * params->d_model, params->d_ff,
            "Standard Transformer uses d_ff = 4 * d_model");
    }
    
    return is_valid;
}

bool cuda_validator_check_lstm_constraints(const CUDAModelParams* params, CUDAValidationReport* report) {
    if (!params || !report) {
        return false;
    }
    
    bool is_valid = true;
    
    // Check LSTM layer count
    if (params->lstm_num_layers <= 0 || params->lstm_num_layers > 16) {
        cuda_validation_report_add_result(report, "lstm_num_layers", false, CUDA_VALIDATION_ERROR,
            "LSTM layer count outside valid range",
            2, params->lstm_num_layers, "LSTM layers must be between 1 and 16");
        is_valid = false;
    }
    
    // Check LSTM hidden size
    if (params->lstm_hidden_size <= 0 || params->lstm_hidden_size > 4096) {
        cuda_validation_report_add_result(report, "lstm_hidden_size", false, CUDA_VALIDATION_ERROR,
            "LSTM hidden size outside valid range",
            256, params->lstm_hidden_size, "LSTM hidden size must be between 1 and 4096");
        is_valid = false;
    }
    
    return is_valid;
}

bool cuda_validator_check_memory_constraints(const CUDAModelParams* params, CUDAValidationReport* report) {
    if (!params || !report) {
        return false;
    }
    
    bool is_valid = true;
    
    // Estimate memory requirements (rough calculation)
    size_t model_params_memory = 0;
    
    // LSTM weights: (input_size + hidden_size) * hidden_size * 4 * num_layers
    model_params_memory += (params->n_symbols + params->lstm_hidden_size) * 
                          params->lstm_hidden_size * 4 * params->lstm_num_layers * sizeof(float);
    
    // Transformer weights (if applicable)
    model_params_memory += params->n_layers * params->d_model * params->d_ff * sizeof(float);
    
    // Batch processing memory
    size_t batch_memory = params->batch_size * params->max_seq_len * params->d_model * sizeof(float);
    
    size_t total_estimated_mb = (model_params_memory + batch_memory) / (1024 * 1024);
    
    if (total_estimated_mb > (size_t)params->memory_budget_mb) {
        cuda_validation_report_add_result(report, "memory_budget", false, CUDA_VALIDATION_WARNING,
            "Estimated memory usage exceeds budget",
            params->memory_budget_mb, total_estimated_mb,
            "Memory budget may be insufficient for these parameters");
        // This is a warning, not an error - actual usage may be different
    }
    
    return is_valid;
}

bool cuda_validator_check_numerical_precision_constraints(const CUDAMathConfig* config, CUDAValidationReport* report) {
    if (!config || !report) {
        return false;
    }
    
    bool is_valid = true;
    
    // Check precision tolerance range
    if (config->precision_tolerance < 1e-10 || config->precision_tolerance > 1e-3) {
        cuda_validation_report_add_result(report, "precision_tolerance", false, CUDA_VALIDATION_WARNING,
            "Precision tolerance outside typical range",
            1e-6, config->precision_tolerance,
            "Typical precision tolerance is between 1e-10 and 1e-3");
    }
    
    return is_valid;
}

// Critical CUDA compatibility checks
bool cuda_validator_verify_train_len_seg_len_relationship(int32_t train_len, int32_t seg_len, CUDAValidationReport* report) {
    if (!report) {
        return false;
    }
    
    // CRITICAL: train_len MUST equal seg_len for CUDA mathematical compatibility
    if (train_len != seg_len) {
        cuda_validation_report_add_result(report, "train_len_seg_len_relationship", false, CUDA_VALIDATION_CRITICAL,
            "train_len != seg_len - CRITICAL CUDA compatibility violation",
            seg_len, train_len,
            "CUDA NNCP requires train_len = seg_len for mathematical correctness");
        return false;
    }
    
    cuda_validation_report_add_result(report, "train_len_seg_len_relationship", true, CUDA_VALIDATION_INFO,
        "train_len = seg_len relationship maintained",
        seg_len, train_len,
        "Critical CUDA compatibility requirement satisfied");
    
    return true;
}

bool cuda_validator_verify_cuda_profile_consistency(const CUDAProfile* profile, CUDAValidationReport* report) {
    if (!profile || !report) {
        return false;
    }
    
    bool is_valid = true;
    
    // Check profile type consistency
    const char* expected_name = cuda_profile_type_to_string(profile->type);
    if (expected_name && strcmp(profile->name, expected_name) != 0) {
        cuda_validation_report_add_result(report, "profile_name_consistency", false, CUDA_VALIDATION_ERROR,
            "Profile name does not match profile type",
            0, 0, "Profile name and type must be consistent");
        is_valid = false;
    }
    
    // Check seg_len matches profile expectations
    int32_t expected_seg_len = 32;  // Default
    if (profile->type == CUDA_PROFILE_LSTM || profile->type == CUDA_PROFILE_LSTM_FAST) {
        expected_seg_len = 20;
    } else if (profile->type == CUDA_PROFILE_ENWIK8 || profile->type == CUDA_PROFILE_ENWIK9) {
        expected_seg_len = 64;
    }
    
    if (profile->params.seg_len != expected_seg_len) {
        cuda_validation_report_add_result(report, "profile_seg_len_consistency", false, CUDA_VALIDATION_ERROR,
            "seg_len does not match profile type expectations",
            expected_seg_len, profile->params.seg_len,
            "Each profile type has specific seg_len requirements");
        is_valid = false;
    }
    
    return is_valid;
}

bool cuda_validator_check_parameter_relationships(const CUDAModelParams* params, CUDAValidationReport* report) {
    if (!params || !report) {
        return false;
    }
    
    bool is_valid = true;
    
    // Check attention head divisibility
    if (params->d_model % params->n_heads != 0) {
        cuda_validation_report_add_result(report, "attention_head_divisibility", false, CUDA_VALIDATION_ERROR,
            "d_model must be divisible by n_heads",
            params->n_heads * ((params->d_model / params->n_heads) + 1), params->d_model,
            "Attention mechanism requires d_model divisible by n_heads");
        is_valid = false;
    }
    
    // Check reasonable layer count
    if (params->n_layers > 32) {
        cuda_validation_report_add_result(report, "excessive_layers", false, CUDA_VALIDATION_WARNING,
            "Very high layer count may impact performance",
            16, params->n_layers,
            "High layer counts increase memory usage and computation time");
    }
    
    return is_valid;
}

// Report output functions
void cuda_validation_report_print_summary(const CUDAValidationReport* report) {
    if (!report) {
        return;
    }
    
    printf("CUDA Validation Report Summary\n");
    printf("==============================\n");
    printf("Profile: %s\n", report->profile_name);
    printf("Timestamp: %s\n", report->validation_timestamp);
    printf("Duration: %.2f ms\n", report->validation_duration_ms);
    printf("\n");
    printf("Overall Valid: %s\n", report->overall_valid ? "YES" : "NO");
    printf("Total Checks: %zu\n", report->total_checks);
    printf("Passed: %zu\n", report->passed_checks);
    printf("Warnings: %zu\n", report->warning_count);
    printf("Errors: %zu\n", report->error_count);
    printf("Critical Errors: %zu\n", report->critical_error_count);
    printf("\n");
}

void cuda_validation_report_print_errors_only(const CUDAValidationReport* report) {
    if (!report) {
        return;
    }
    
    printf("CUDA Validation Errors\n");
    printf("======================\n");
    
    for (size_t i = 0; i < report->result_count; i++) {
        const CUDAParameterValidationResult* result = &report->results[i];
        if (!result->is_valid && result->severity >= CUDA_VALIDATION_ERROR) {
            printf("[%s] %s: %s\n", 
                   result->severity == CUDA_VALIDATION_CRITICAL ? "CRITICAL" : "ERROR",
                   result->parameter_name, result->message);
            printf("  Expected: %.2f, Actual: %.2f\n", 
                   result->expected_value, result->actual_value);
            printf("  Constraint: %s\n\n", result->cuda_constraint_description);
        }
    }
}

// Quick validation helpers
bool cuda_validator_quick_check_compatibility(const CUDAProfile* profile) {
    if (!profile) {
        return false;
    }
    
    // Quick checks for most critical compatibility issues
    return cuda_validator_quick_check_seg_len(profile->params.seg_len) &&
           cuda_validator_quick_check_batch_size(profile->params.batch_size);
}

bool cuda_validator_quick_check_seg_len(int32_t seg_len) {
    return (seg_len > 0 && seg_len <= 1024);
}

bool cuda_validator_quick_check_batch_size(int32_t batch_size) {
    return (batch_size > 0 && batch_size <= 1024);
}

bool cuda_validator_quick_check_train_len_relationship(int32_t train_len, int32_t seg_len) {
    return (train_len == seg_len);
}

// Constraint accessors
const CUDAParameterConstraint* cuda_validator_get_seg_len_constraints(void) {
    return &seg_len_constraints;
}

const CUDAParameterConstraint* cuda_validator_get_batch_size_constraints(void) {
    return &batch_size_constraints;
}

const CUDAParameterConstraint* cuda_validator_get_hidden_size_constraints(void) {
    return &hidden_size_constraints;
}

const CUDAParameterConstraint* cuda_validator_get_learning_rate_constraints(void) {
    return &learning_rate_constraints;
}

// Error handling
const char* cuda_validator_error_string(CUDAValidatorError error) {
    if (error < 0 || error >= sizeof(cuda_validator_error_messages) / sizeof(cuda_validator_error_messages[0])) {
        return "Unknown error";
    }
    return cuda_validator_error_messages[error];
}