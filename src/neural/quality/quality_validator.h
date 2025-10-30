#ifndef QUALITY_VALIDATOR_H
#define QUALITY_VALIDATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum limits
#define QV_MAX_VALIDATION_LEVELS     10
#define QV_MAX_QUALITY_METRICS       32
#define QV_MAX_ERROR_DETAILS         16
#define QV_MAX_REPAIR_ATTEMPTS       5
#define QV_MAX_THRESHOLD_RULES       64
#define QV_MAX_CUSTOM_VALIDATORS     16

// Error codes
typedef enum {
    QV_SUCCESS = 0,
    QV_ERROR_INVALID_PARAM,
    QV_ERROR_MEMORY_ALLOCATION,
    QV_ERROR_VALIDATION_FAILED,
    QV_ERROR_THRESHOLD_EXCEEDED,
    QV_ERROR_REPAIR_FAILED,
    QV_ERROR_UNSUPPORTED_FORMAT,
    QV_ERROR_CHECKSUM_MISMATCH,
    QV_ERROR_COMPRESSION_CORRUPTED,
    QV_ERROR_METADATA_INVALID,
    QV_ERROR_VERSION_INCOMPATIBLE,
    QV_ERROR_INSUFFICIENT_DATA,
    QV_ERROR_QUALITY_TOO_LOW,
    QV_ERROR_PERFORMANCE_DEGRADED
} QualityValidatorError;

// Data quality levels
typedef enum {
    QV_QUALITY_EXCELLENT = 5,     // >95% quality
    QV_QUALITY_GOOD = 4,          // 85-95% quality
    QV_QUALITY_ACCEPTABLE = 3,    // 75-85% quality
    QV_QUALITY_POOR = 2,          // 60-75% quality
    QV_QUALITY_UNACCEPTABLE = 1   // <60% quality
} QualityLevel;

// Validation severity levels
typedef enum {
    QV_SEVERITY_INFO = 0,
    QV_SEVERITY_WARNING,
    QV_SEVERITY_ERROR,
    QV_SEVERITY_CRITICAL
} ValidationSeverity;

// Quality metric types
typedef enum {
    QV_METRIC_COMPRESSION_RATIO = 0,    // Compression efficiency
    QV_METRIC_QUALITY_SCORE,            // Overall quality score
    QV_METRIC_FIDELITY,                 // Data fidelity
    QV_METRIC_ENTROPY,                  // Information entropy
    QV_METRIC_SNR,                      // Signal-to-noise ratio
    QV_METRIC_PSNR,                     // Peak signal-to-noise ratio
    QV_METRIC_SSIM,                     // Structural similarity index
    QV_METRIC_MSE,                      // Mean squared error
    QV_METRIC_MAE,                      // Mean absolute error
    QV_METRIC_PERPLEXITY,              // Model perplexity
    QV_METRIC_BLEU_SCORE,              // Text quality (BLEU)
    QV_METRIC_ROUGE_SCORE,             // Text summarization quality
    QV_METRIC_PROCESSING_TIME,         // Performance metric
    QV_METRIC_MEMORY_USAGE,            // Resource usage
    QV_METRIC_ENERGY_EFFICIENCY,       // Power consumption
    QV_METRIC_CUSTOM                   // User-defined metric
} QualityMetricType;

// Validation modes
typedef enum {
    QV_MODE_BASIC = 0,              // Basic validation only
    QV_MODE_STANDARD,               // Standard validation
    QV_MODE_COMPREHENSIVE,          // Full validation suite
    QV_MODE_CONTINUOUS,             // Continuous monitoring
    QV_MODE_ADAPTIVE               // Adaptive validation
} ValidationMode;

// Error repair strategies
typedef enum {
    QV_REPAIR_NONE = 0,            // No repair attempted
    QV_REPAIR_RECOMPRESS,          // Re-compress with different settings
    QV_REPAIR_FALLBACK_ALGORITHM,  // Use fallback compression
    QV_REPAIR_INTERPOLATE,         // Interpolate missing data
    QV_REPAIR_REDUNDANCY,          // Use redundancy information
    QV_REPAIR_MANUAL_REVIEW        // Require manual review
} RepairStrategy;

// Forward declarations
typedef struct QualityValidator QualityValidator;
typedef struct ValidationResult ValidationResult;
typedef struct QualityMetric QualityMetric;
typedef struct ValidationRule ValidationRule;

// Custom validator function
typedef QualityValidatorError (*CustomValidatorFunction)(const void* data, 
                                                        size_t data_size,
                                                        void* context,
                                                        double* quality_score);

// Repair function
typedef QualityValidatorError (*RepairFunction)(void* data, 
                                               size_t* data_size,
                                               const ValidationResult* validation,
                                               void* context);

// Quality metric structure
struct QualityMetric {
    QualityMetricType type;             // Metric type
    char name[64];                      // Metric name
    char description[256];              // Metric description
    double value;                       // Current value
    double threshold_min;               // Minimum acceptable value
    double threshold_max;               // Maximum acceptable value
    double weight;                      // Weight in overall score
    bool is_valid;                      // Whether metric is valid
    uint64_t last_updated;             // Last update timestamp
};

// Validation error detail
typedef struct {
    ValidationSeverity severity;        // Error severity
    QualityMetricType metric_type;      // Related metric
    uint32_t error_code;               // Specific error code
    char message[256];                 // Error message
    uint64_t offset;                   // Data offset (if applicable)
    size_t length;                     // Error length (if applicable)
    RepairStrategy suggested_repair;    // Suggested repair strategy
} ValidationError;

// Validation result
struct ValidationResult {
    bool is_valid;                     // Overall validation result
    QualityLevel quality_level;        // Overall quality level
    double overall_score;              // Overall quality score (0-1)
    
    QualityMetric metrics[QV_MAX_QUALITY_METRICS];  // Quality metrics
    uint32_t num_metrics;              // Number of metrics
    
    ValidationError errors[QV_MAX_ERROR_DETAILS];   // Validation errors
    uint32_t num_errors;               // Number of errors
    
    uint64_t validation_time_us;       // Validation time in microseconds
    uint64_t data_size;                // Size of validated data
    char validator_version[32];        // Validator version
    uint64_t timestamp;                // Validation timestamp
};

// Validation rule
struct ValidationRule {
    QualityMetricType metric_type;     // Target metric
    double min_value;                  // Minimum acceptable value
    double max_value;                  // Maximum acceptable value
    ValidationSeverity severity;       // Violation severity
    bool enabled;                      // Whether rule is active
    char description[256];             // Rule description
};

// Threshold configuration
typedef struct {
    ValidationRule rules[QV_MAX_THRESHOLD_RULES];   // Validation rules
    uint32_t num_rules;                // Number of rules
    
    double quality_threshold;          // Minimum quality threshold
    double compression_ratio_min;      // Minimum compression ratio
    double compression_ratio_max;      // Maximum compression ratio
    double processing_time_max;        // Maximum processing time (seconds)
    
    bool enable_auto_repair;           // Enable automatic repair
    uint32_t max_repair_attempts;      // Maximum repair attempts
    bool require_manual_review;        // Require manual review for errors
} ValidationThresholds;

// Validator configuration
typedef struct {
    ValidationMode mode;               // Validation mode
    ValidationThresholds thresholds;   // Quality thresholds
    
    bool enable_checksum_validation;   // Enable checksum validation
    bool enable_format_validation;     // Enable format validation
    bool enable_content_validation;    // Enable content validation
    bool enable_performance_validation; // Enable performance validation
    
    uint32_t validation_timeout_ms;    // Validation timeout
    bool enable_parallel_validation;   // Enable parallel validation
    uint32_t num_validation_threads;   // Number of validation threads
    
    void* custom_context;             // Custom context
} ValidatorConfig;

// Main quality validator structure
struct QualityValidator {
    ValidatorConfig config;            // Configuration
    bool is_initialized;               // Initialization flag
    
    // Custom validators
    CustomValidatorFunction custom_validators[QV_MAX_CUSTOM_VALIDATORS];
    void* custom_contexts[QV_MAX_CUSTOM_VALIDATORS];
    uint32_t num_custom_validators;
    
    // Repair functions
    RepairFunction repair_functions[16];
    uint32_t num_repair_functions;
    
    // Statistics
    uint64_t total_validations;        // Total validations performed
    uint64_t successful_validations;   // Successful validations
    uint64_t failed_validations;       // Failed validations
    uint64_t repairs_attempted;        // Repairs attempted
    uint64_t repairs_successful;       // Successful repairs
    
    // Performance tracking
    double average_validation_time;    // Average validation time
    double average_quality_score;      // Average quality score
    uint64_t total_data_processed;     // Total data processed
    
    // Thread safety
    pthread_mutex_t validator_mutex;   // Validator mutex
    pthread_rwlock_t rules_lock;       // Rules read-write lock
};

// Core API Functions

/**
 * Create and initialize quality validator
 * @param validator Pointer to store created validator
 * @param config Validator configuration
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_create(QualityValidator** validator,
                                              const ValidatorConfig* config);

/**
 * Destroy quality validator and free resources
 * @param validator Validator to destroy
 */
void quality_validator_destroy(QualityValidator* validator);

// Validation Functions

/**
 * Validate data quality
 * @param validator Quality validator instance
 * @param data Data to validate
 * @param data_size Size of data
 * @param format_info Format information (optional)
 * @param result Output validation result
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_validate(QualityValidator* validator,
                                                 const void* data,
                                                 size_t data_size,
                                                 const void* format_info,
                                                 ValidationResult* result);

/**
 * Validate compressed data integrity
 * @param validator Quality validator instance
 * @param original_data Original data
 * @param original_size Original data size
 * @param compressed_data Compressed data
 * @param compressed_size Compressed data size
 * @param result Output validation result
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_validate_compression(QualityValidator* validator,
                                                           const void* original_data,
                                                           size_t original_size,
                                                           const void* compressed_data,
                                                           size_t compressed_size,
                                                           ValidationResult* result);

/**
 * Continuous quality monitoring
 * @param validator Quality validator instance
 * @param data_stream Data stream to monitor
 * @param stream_size Stream size
 * @param callback Callback for validation results
 * @param context Callback context
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_monitor(QualityValidator* validator,
                                               const void* data_stream,
                                               size_t stream_size,
                                               void (*callback)(const ValidationResult*, void*),
                                               void* context);

// Quality Metrics Functions

/**
 * Calculate quality metrics for data
 * @param validator Quality validator instance
 * @param data Data to analyze
 * @param data_size Size of data
 * @param metrics Output metrics array
 * @param max_metrics Maximum number of metrics
 * @param num_metrics Output number of metrics
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_calculate_metrics(QualityValidator* validator,
                                                         const void* data,
                                                         size_t data_size,
                                                         QualityMetric* metrics,
                                                         uint32_t max_metrics,
                                                         uint32_t* num_metrics);

/**
 * Get specific quality metric
 * @param validator Quality validator instance
 * @param data Data to analyze
 * @param data_size Size of data
 * @param metric_type Type of metric to calculate
 * @param metric Output metric
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_get_metric(QualityValidator* validator,
                                                   const void* data,
                                                   size_t data_size,
                                                   QualityMetricType metric_type,
                                                   QualityMetric* metric);

/**
 * Compare two datasets for quality differences
 * @param validator Quality validator instance
 * @param data1 First dataset
 * @param size1 First dataset size
 * @param data2 Second dataset
 * @param size2 Second dataset size
 * @param comparison_result Output comparison result
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_compare(QualityValidator* validator,
                                               const void* data1, size_t size1,
                                               const void* data2, size_t size2,
                                               ValidationResult* comparison_result);

// Error Detection and Repair Functions

/**
 * Detect errors in data
 * @param validator Quality validator instance
 * @param data Data to check
 * @param data_size Size of data
 * @param errors Output error array
 * @param max_errors Maximum number of errors
 * @param num_errors Output number of errors
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_detect_errors(QualityValidator* validator,
                                                     const void* data,
                                                     size_t data_size,
                                                     ValidationError* errors,
                                                     uint32_t max_errors,
                                                     uint32_t* num_errors);

/**
 * Attempt to repair detected errors
 * @param validator Quality validator instance
 * @param data Data to repair (modified in-place)
 * @param data_size Pointer to data size (may be modified)
 * @param errors Detected errors
 * @param num_errors Number of errors
 * @param repair_result Output repair result
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_repair(QualityValidator* validator,
                                              void* data,
                                              size_t* data_size,
                                              const ValidationError* errors,
                                              uint32_t num_errors,
                                              ValidationResult* repair_result);

/**
 * Validate and repair data in one operation
 * @param validator Quality validator instance
 * @param data Data to validate and repair
 * @param data_size Pointer to data size
 * @param max_repair_attempts Maximum repair attempts
 * @param result Output validation/repair result
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_validate_and_repair(QualityValidator* validator,
                                                           void* data,
                                                           size_t* data_size,
                                                           uint32_t max_repair_attempts,
                                                           ValidationResult* result);

// Configuration Functions

/**
 * Create default validator configuration
 * @param config Output configuration
 * @param mode Validation mode
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_config_default(ValidatorConfig* config,
                                                       ValidationMode mode);

/**
 * Update validator configuration
 * @param validator Quality validator instance
 * @param config New configuration
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_update_config(QualityValidator* validator,
                                                      const ValidatorConfig* config);

/**
 * Add validation rule
 * @param validator Quality validator instance
 * @param rule Validation rule to add
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_add_rule(QualityValidator* validator,
                                                 const ValidationRule* rule);

/**
 * Remove validation rule
 * @param validator Quality validator instance
 * @param metric_type Metric type to remove rule for
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_remove_rule(QualityValidator* validator,
                                                    QualityMetricType metric_type);

/**
 * Add custom validator function
 * @param validator Quality validator instance
 * @param custom_validator Custom validator function
 * @param context Custom context
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_add_custom(QualityValidator* validator,
                                                   CustomValidatorFunction custom_validator,
                                                   void* context);

// Statistics and Reporting Functions

/**
 * Get validator statistics
 * @param validator Quality validator instance
 * @param total_validations Output total validations
 * @param success_rate Output success rate (0-1)
 * @param average_score Output average quality score
 * @param average_time Output average validation time
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_get_statistics(QualityValidator* validator,
                                                       uint64_t* total_validations,
                                                       double* success_rate,
                                                       double* average_score,
                                                       double* average_time);

/**
 * Reset validator statistics
 * @param validator Quality validator instance
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_reset_statistics(QualityValidator* validator);

/**
 * Generate quality report
 * @param validator Quality validator instance
 * @param result Validation result
 * @param report_buffer Output report buffer
 * @param buffer_size Buffer size
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_generate_report(QualityValidator* validator,
                                                        const ValidationResult* result,
                                                        char* report_buffer,
                                                        size_t buffer_size);

// Utility Functions

/**
 * Get error string for error code
 * @param error Error code
 * @return Human-readable error string
 */
const char* quality_validator_error_string(QualityValidatorError error);

/**
 * Get quality level name
 * @param level Quality level
 * @return Human-readable quality level name
 */
const char* quality_validator_level_name(QualityLevel level);

/**
 * Get metric type name
 * @param metric_type Metric type
 * @return Human-readable metric type name
 */
const char* quality_validator_metric_name(QualityMetricType metric_type);

/**
 * Get severity name
 * @param severity Validation severity
 * @return Human-readable severity name
 */
const char* quality_validator_severity_name(ValidationSeverity severity);

/**
 * Convert quality score to quality level
 * @param score Quality score (0-1)
 * @return Corresponding quality level
 */
QualityLevel quality_validator_score_to_level(double score);

/**
 * Calculate overall quality score from metrics
 * @param metrics Quality metrics array
 * @param num_metrics Number of metrics
 * @param overall_score Output overall score
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_calculate_overall_score(const QualityMetric* metrics,
                                                               uint32_t num_metrics,
                                                               double* overall_score);

/**
 * Enhanced validation for neural compression with SHA256 verification
 * @param validator Quality validator instance
 * @param original_data Original uncompressed data
 * @param original_size Size of original data
 * @param compressed_data Compressed data
 * @param compressed_size Size of compressed data
 * @param decompressed_data Decompressed data
 * @param decompressed_size Size of decompressed data
 * @param result Output validation result with quality metrics
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_validate_neural_compression(
    QualityValidator* validator,
    const uint8_t* original_data, size_t original_size,
    const uint8_t* compressed_data, size_t compressed_size,
    const uint8_t* decompressed_data, size_t decompressed_size,
    ValidationResult* result);

// Cross-Algorithm Validation Functions for Compression Optimization

/**
 * Validate consistency between Transformer and LSTM compression algorithms
 * @param validator Quality validator instance
 * @param test_data Input test data
 * @param test_size Size of test data
 * @param transformer_result Transformer compression/decompression result
 * @param lstm_result LSTM compression/decompression result
 * @param consistency_result Output consistency validation result
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_validate_cross_algorithm_consistency(
    QualityValidator* validator,
    const uint8_t* test_data, size_t test_size,
    const void* transformer_result,
    const void* lstm_result,
    ValidationResult* consistency_result);

/**
 * Validate compression improvement with prediction scoring integration
 * @param validator Quality validator instance
 * @param baseline_ratio Baseline compression ratio (without prediction scoring)
 * @param enhanced_ratio Enhanced compression ratio (with prediction scoring)
 * @param algorithm_name Algorithm name ("Transformer" or "LSTM")
 * @param improvement_result Output improvement validation result
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_validate_compression_improvement(
    QualityValidator* validator,
    double baseline_ratio,
    double enhanced_ratio,
    const char* algorithm_name,
    ValidationResult* improvement_result);

/**
 * Perform comprehensive quality regression testing
 * @param validator Quality validator instance
 * @param test_datasets Array of test datasets
 * @param num_datasets Number of test datasets
 * @param regression_results Output regression test results
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_perform_regression_testing(
    QualityValidator* validator,
    const void* test_datasets,
    uint32_t num_datasets,
    ValidationResult* regression_results);

/**
 * Validate lossless integrity across both algorithms
 * @param validator Quality validator instance
 * @param original_data Original input data
 * @param original_size Size of original data
 * @param transformer_decompressed Transformer decompressed data
 * @param transformer_size Transformer decompressed size
 * @param lstm_decompressed LSTM decompressed data
 * @param lstm_size LSTM decompressed size
 * @param integrity_result Output integrity validation result
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_validate_lossless_integrity(
    QualityValidator* validator,
    const uint8_t* original_data, size_t original_size,
    const uint8_t* transformer_decompressed, size_t transformer_size,
    const uint8_t* lstm_decompressed, size_t lstm_size,
    ValidationResult* integrity_result);

/**
 * Generate comprehensive quality report for compression optimization
 * @param validator Quality validator instance
 * @param test_results Array of validation results
 * @param num_results Number of results
 * @param report_buffer Output buffer for quality report
 * @param buffer_size Size of report buffer
 * @return QV_SUCCESS on success, error code on failure
 */
QualityValidatorError quality_validator_generate_compression_optimization_report(
    QualityValidator* validator,
    const ValidationResult* test_results,
    uint32_t num_results,
    char* report_buffer,
    size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif // QUALITY_VALIDATOR_H
