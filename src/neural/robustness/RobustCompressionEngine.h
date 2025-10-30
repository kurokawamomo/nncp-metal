/*
 * RobustCompressionEngine.h
 * 
 * Robust Error Handling Engine for Neural Compression
 * Advanced error detection, automatic fallback mechanisms, 
 * progressive recovery strategies, and system resilience
 */

#ifndef ROBUST_COMPRESSION_ENGINE_H
#define ROBUST_COMPRESSION_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../validation/IntegrityValidator.h"
#include "../verification/PerformanceVerifier.h"

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <os/log.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct RobustCompressionEngine RobustCompressionEngine;

// Error severity levels
typedef enum {
    ERROR_SEVERITY_INFO = 0,              // Informational - no action needed
    ERROR_SEVERITY_WARNING,               // Warning - monitoring required
    ERROR_SEVERITY_MINOR,                 // Minor error - graceful degradation
    ERROR_SEVERITY_MODERATE,              // Moderate error - fallback required
    ERROR_SEVERITY_SEVERE,                // Severe error - immediate recovery
    ERROR_SEVERITY_CRITICAL,              // Critical error - system protection
    ERROR_SEVERITY_FATAL                  // Fatal error - complete shutdown
} ErrorSeverity;

// Error category classification
typedef enum {
    ERROR_CATEGORY_MEMORY = 0,            // Memory allocation/access errors
    ERROR_CATEGORY_COMPUTATION,           // Computation/numerical errors
    ERROR_CATEGORY_IO,                    // Input/output errors
    ERROR_CATEGORY_VALIDATION,            // Data validation errors
    ERROR_CATEGORY_HARDWARE,              // Hardware-related errors
    ERROR_CATEGORY_NETWORK,               // Network/communication errors
    ERROR_CATEGORY_TIMEOUT,               // Timeout errors
    ERROR_CATEGORY_SECURITY,              // Security-related errors
    ERROR_CATEGORY_CONFIGURATION,         // Configuration errors
    ERROR_CATEGORY_SYSTEM,                // System-level errors
    ERROR_CATEGORY_UNKNOWN               // Unknown/unclassified errors
} ErrorCategory;

// Recovery strategy types
typedef enum {
    RECOVERY_STRATEGY_NONE = 0,           // No recovery action
    RECOVERY_STRATEGY_RETRY,              // Simple retry with backoff
    RECOVERY_STRATEGY_FALLBACK,           // Use alternative algorithm
    RECOVERY_STRATEGY_DEGRADED_MODE,      // Continue with reduced functionality
    RECOVERY_STRATEGY_PROGRESSIVE_RETRY,  // Progressive complexity reduction
    RECOVERY_STRATEGY_ISOLATION,         // Isolate and continue
    RECOVERY_STRATEGY_RESTART,            // Restart component/subsystem
    RECOVERY_STRATEGY_SAFE_SHUTDOWN      // Controlled safe shutdown
} RecoveryStrategy;

// Fallback compression modes
typedef enum {
    FALLBACK_MODE_NONE = 0,               // No fallback - fail immediately
    FALLBACK_MODE_STANDARD_COMPRESSION,   // Use standard compression (gzip/zlib)
    FALLBACK_MODE_SIMPLE_NEURAL,          // Use simplified neural model
    FALLBACK_MODE_REDUCED_PRECISION,      // Reduce precision requirements
    FALLBACK_MODE_SMALLER_CONTEXT,        // Use smaller context window
    FALLBACK_MODE_CPU_ONLY,               // Disable GPU acceleration
    FALLBACK_MODE_NO_OPTIMIZATION,        // Disable all optimizations
    FALLBACK_MODE_PASSTHROUGH            // No compression (copy only)
} FallbackMode;

// Error detection configuration
typedef struct {
    bool enable_memory_monitoring;        // Monitor memory allocation patterns
    bool enable_computation_validation;   // Validate computation results
    bool enable_performance_monitoring;   // Monitor performance degradation
    bool enable_thermal_monitoring;       // Monitor thermal state
    bool enable_hardware_monitoring;      // Monitor hardware health
    bool enable_deadlock_detection;       // Detect potential deadlocks
    bool enable_infinite_loop_detection;  // Detect infinite loops
    bool enable_stack_overflow_detection; // Detect stack overflow
    uint32_t error_detection_interval_ms; // Error detection interval
    uint32_t performance_degradation_threshold_percent; // Performance threshold
    float memory_pressure_threshold;      // Memory pressure threshold (0-1)
    float thermal_throttle_threshold;     // Thermal threshold (°C)
} ErrorDetectionConfig;

// Robustness configuration
typedef struct {
    ErrorSeverity minimum_severity_for_recovery; // Minimum severity to trigger recovery
    uint32_t max_retry_attempts;          // Maximum retry attempts per error
    uint32_t retry_backoff_base_ms;       // Base backoff time for retries
    uint32_t max_backoff_time_ms;         // Maximum backoff time
    bool enable_progressive_degradation;  // Enable progressive degradation
    bool enable_automatic_fallback;       // Enable automatic fallback
    bool enable_error_learning;           // Learn from error patterns
    bool enable_preemptive_recovery;      // Preemptive error recovery
    FallbackMode default_fallback_mode;   // Default fallback compression mode
    uint32_t recovery_timeout_ms;         // Recovery operation timeout
    uint32_t max_concurrent_recoveries;   // Maximum concurrent recovery operations
    float recovery_success_threshold;     // Required success rate (0-1)
} RobustnessConfig;

// Error information structure
typedef struct {
    uint64_t error_id;                    // Unique error identifier
    ErrorSeverity severity;               // Error severity level
    ErrorCategory category;               // Error category
    uint32_t error_code;                  // Specific error code
    uint64_t timestamp_ns;                // Error occurrence timestamp
    uint64_t thread_id;                   // Thread where error occurred
    char error_message[512];              // Human-readable error description
    char error_context[256];              // Context where error occurred
    char stack_trace[2048];               // Stack trace (if available)
    size_t data_size_at_error;            // Data size when error occurred
    void* error_data;                     // Additional error-specific data
    size_t error_data_size;               // Size of error-specific data
    bool is_recoverable;                  // Whether error is recoverable
    bool requires_immediate_action;       // Requires immediate action
} ErrorInfo;

// Recovery attempt information
typedef struct {
    uint64_t recovery_attempt_id;         // Recovery attempt identifier
    uint64_t error_id;                    // Associated error ID
    RecoveryStrategy strategy;            // Recovery strategy used
    uint64_t attempt_timestamp_ns;        // Recovery attempt timestamp
    uint64_t recovery_duration_ns;        // Time taken for recovery
    bool recovery_successful;             // Recovery success status
    float recovery_quality_score;         // Quality of recovery (0-1)
    char recovery_description[512];       // Description of recovery action
    uint32_t resources_consumed;          // Resources consumed during recovery
    FallbackMode fallback_mode_used;      // Fallback mode used (if any)
} RecoveryAttempt;

// Error statistics and monitoring
typedef struct {
    uint64_t total_errors_detected;       // Total errors detected
    uint64_t errors_by_severity[7];       // Errors by severity level
    uint64_t errors_by_category[11];      // Errors by category
    uint64_t total_recovery_attempts;     // Total recovery attempts
    uint64_t successful_recoveries;       // Successful recovery attempts
    uint64_t failed_recoveries;           // Failed recovery attempts
    float overall_recovery_success_rate;  // Overall recovery success rate
    uint64_t total_fallback_activations;  // Fallback activations
    uint64_t current_active_errors;       // Currently active errors
    uint64_t mean_time_to_recovery_ns;    // Mean time to recovery
    uint64_t max_recovery_time_ns;        // Maximum recovery time observed
    float system_reliability_score;       // Overall system reliability (0-1)
    uint64_t uptime_since_last_critical_error; // Uptime since last critical error
} ErrorStatistics;

// System health monitoring
typedef struct {
    float cpu_utilization_percent;        // Current CPU utilization
    float memory_utilization_percent;     // Current memory utilization
    float gpu_utilization_percent;        // Current GPU utilization
    float thermal_state_celsius;          // Current thermal state
    float power_consumption_watts;         // Current power consumption
    uint32_t active_threads;              // Number of active threads
    uint32_t pending_operations;          // Number of pending operations
    bool hardware_healthy;                // Hardware health status
    bool memory_pressure_detected;        // Memory pressure detected
    bool thermal_throttling_active;       // Thermal throttling active
    bool performance_degradation_detected; // Performance degradation detected
    float overall_health_score;           // Overall system health (0-1)
} SystemHealthStatus;

// Robust operation result
typedef struct {
    bool operation_successful;            // Overall operation success
    bool fallback_used;                   // Fallback mechanism was used
    FallbackMode fallback_mode;           // Fallback mode used
    uint32_t errors_encountered;          // Number of errors encountered
    uint32_t recovery_attempts;           // Number of recovery attempts
    uint64_t total_operation_time_ns;     // Total operation time
    uint64_t recovery_overhead_time_ns;   // Time spent on recovery
    float compression_ratio_achieved;     // Achieved compression ratio
    float quality_score;                  // Overall quality score (0-1)
    ErrorInfo most_severe_error;          // Most severe error encountered
    char operation_summary[512];          // Summary of operation and any issues
} RobustOperationResult;

// Progressive degradation configuration
typedef struct {
    bool enable_quality_degradation;      // Allow quality degradation
    bool enable_speed_degradation;        // Allow speed degradation
    bool enable_feature_disabling;        // Allow feature disabling
    float minimum_acceptable_quality;     // Minimum quality threshold (0-1)
    float minimum_acceptable_speed;       // Minimum speed threshold (0-1)
    uint32_t degradation_steps;           // Number of degradation steps
    float degradation_step_size;          // Size of each degradation step
} ProgressiveDegradationConfig;

// Error codes for robust compression engine
typedef enum {
    ROBUST_ENGINE_SUCCESS = 0,
    ROBUST_ENGINE_ERROR_INVALID_PARAM,
    ROBUST_ENGINE_ERROR_MEMORY_ALLOCATION,
    ROBUST_ENGINE_ERROR_INITIALIZATION_FAILED,
    ROBUST_ENGINE_ERROR_UNRECOVERABLE_ERROR,
    ROBUST_ENGINE_ERROR_RECOVERY_FAILED,
    ROBUST_ENGINE_ERROR_TIMEOUT,
    ROBUST_ENGINE_ERROR_SYSTEM_OVERLOAD,
    ROBUST_ENGINE_ERROR_HARDWARE_FAILURE,
    ROBUST_ENGINE_ERROR_CONFIGURATION_INVALID,
    ROBUST_ENGINE_ERROR_OPERATION_ABORTED,
    ROBUST_ENGINE_ERROR_FALLBACK_EXHAUSTED,
    ROBUST_ENGINE_ERROR_CRITICAL_SYSTEM_STATE
} RobustEngineError;

// Core API Functions

/**
 * Create robust compression engine instance
 * @param engine Pointer to store created engine
 * @param error_config Error detection configuration
 * @param robustness_config Robustness configuration
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_create(RobustCompressionEngine** engine,
                                       const ErrorDetectionConfig* error_config,
                                       const RobustnessConfig* robustness_config);

/**
 * Initialize robust compression engine
 * @param engine Robust compression engine instance
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_initialize(RobustCompressionEngine* engine);

/**
 * Start error monitoring and health checking
 * @param engine Robust compression engine instance
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_start_monitoring(RobustCompressionEngine* engine);

/**
 * Stop error monitoring
 * @param engine Robust compression engine instance
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_stop_monitoring(RobustCompressionEngine* engine);

// Error Detection and Reporting

/**
 * Report an error to the robust engine
 * @param engine Robust compression engine instance
 * @param error_info Error information
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_report_error(RobustCompressionEngine* engine,
                                             const ErrorInfo* error_info);

/**
 * Detect potential system issues proactively
 * @param engine Robust compression engine instance
 * @param health_status Output current system health status
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_detect_system_issues(RobustCompressionEngine* engine,
                                                     SystemHealthStatus* health_status);

/**
 * Check for specific error patterns
 * @param engine Robust compression engine instance
 * @param pattern_description Description of pattern to check
 * @param pattern_detected Output boolean for pattern detection
 * @param confidence_score Output confidence score (0-1)
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_check_error_pattern(RobustCompressionEngine* engine,
                                                    const char* pattern_description,
                                                    bool* pattern_detected,
                                                    float* confidence_score);

// Recovery and Fallback Mechanisms

/**
 * Attempt error recovery for given error
 * @param engine Robust compression engine instance
 * @param error_info Error to recover from
 * @param recovery_strategy Preferred recovery strategy
 * @param recovery_attempt Output recovery attempt information
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_attempt_recovery(RobustCompressionEngine* engine,
                                                 const ErrorInfo* error_info,
                                                 RecoveryStrategy recovery_strategy,
                                                 RecoveryAttempt* recovery_attempt);

/**
 * Execute automatic fallback mechanism
 * @param engine Robust compression engine instance
 * @param original_operation_context Context of the original operation
 * @param fallback_mode Fallback mode to use
 * @param fallback_result Output fallback operation result
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_execute_fallback(RobustCompressionEngine* engine,
                                                 const void* original_operation_context,
                                                 FallbackMode fallback_mode,
                                                 RobustOperationResult* fallback_result);

/**
 * Implement progressive degradation
 * @param engine Robust compression engine instance
 * @param degradation_config Degradation configuration
 * @param current_quality Current quality level (0-1)
 * @param target_quality Target quality level (0-1)
 * @param degradation_successful Output boolean for degradation success
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_progressive_degradation(RobustCompressionEngine* engine,
                                                       const ProgressiveDegradationConfig* degradation_config,
                                                       float current_quality,
                                                       float target_quality,
                                                       bool* degradation_successful);

/**
 * Restore system from degraded state
 * @param engine Robust compression engine instance
 * @param target_quality Target quality to restore (0-1)
 * @param restoration_timeout_ms Timeout for restoration
 * @param restoration_successful Output boolean for restoration success
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_restore_from_degradation(RobustCompressionEngine* engine,
                                                         float target_quality,
                                                         uint32_t restoration_timeout_ms,
                                                         bool* restoration_successful);

// Robust Operations

/**
 * Perform robust compression with automatic error handling
 * @param engine Robust compression engine instance
 * @param input_data Data to compress
 * @param input_size Size of input data
 * @param output_buffer Output buffer for compressed data
 * @param output_buffer_size Size of output buffer
 * @param compressed_size Output size of compressed data
 * @param operation_result Output operation result with error details
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_compress_with_recovery(RobustCompressionEngine* engine,
                                                       const void* input_data,
                                                       size_t input_size,
                                                       void* output_buffer,
                                                       size_t output_buffer_size,
                                                       size_t* compressed_size,
                                                       RobustOperationResult* operation_result);

/**
 * Perform robust decompression with automatic error handling
 * @param engine Robust compression engine instance
 * @param compressed_data Compressed data to decompress
 * @param compressed_size Size of compressed data
 * @param output_buffer Output buffer for decompressed data
 * @param output_buffer_size Size of output buffer
 * @param decompressed_size Output size of decompressed data
 * @param operation_result Output operation result with error details
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_decompress_with_recovery(RobustCompressionEngine* engine,
                                                         const void* compressed_data,
                                                         size_t compressed_size,
                                                         void* output_buffer,
                                                         size_t output_buffer_size,
                                                         size_t* decompressed_size,
                                                         RobustOperationResult* operation_result);

/**
 * Execute any operation with robust error handling wrapper
 * @param engine Robust compression engine instance
 * @param operation_function Function pointer to operation
 * @param operation_context Context data for operation
 * @param operation_timeout_ms Operation timeout in milliseconds
 * @param operation_result Output operation result
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_execute_with_recovery(RobustCompressionEngine* engine,
                                                      void* operation_function,
                                                      void* operation_context,
                                                      uint32_t operation_timeout_ms,
                                                      RobustOperationResult* operation_result);

// System Health and Monitoring

/**
 * Get current system health status
 * @param engine Robust compression engine instance
 * @param health_status Output system health status
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_get_system_health(RobustCompressionEngine* engine,
                                                  SystemHealthStatus* health_status);

/**
 * Get error statistics and monitoring data
 * @param engine Robust compression engine instance
 * @param error_stats Output error statistics
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_get_error_statistics(RobustCompressionEngine* engine,
                                                     ErrorStatistics* error_stats);

/**
 * Reset error statistics and counters
 * @param engine Robust compression engine instance
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_reset_error_statistics(RobustCompressionEngine* engine);

/**
 * Check if system is in safe operational state
 * @param engine Robust compression engine instance
 * @param is_safe Output boolean for safety status
 * @param safety_score Output safety score (0-1)
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_check_operational_safety(RobustCompressionEngine* engine,
                                                         bool* is_safe,
                                                         float* safety_score);

// Configuration and Learning

/**
 * Update robustness configuration dynamically
 * @param engine Robust compression engine instance
 * @param new_config New robustness configuration
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_update_configuration(RobustCompressionEngine* engine,
                                                     const RobustnessConfig* new_config);

/**
 * Learn from error patterns and adapt behavior
 * @param engine Robust compression engine instance
 * @param learning_data Historical error and recovery data
 * @param learning_data_size Size of learning data
 * @param adaptation_successful Output boolean for adaptation success
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_learn_and_adapt(RobustCompressionEngine* engine,
                                                const void* learning_data,
                                                size_t learning_data_size,
                                                bool* adaptation_successful);

/**
 * Generate robustness improvement recommendations
 * @param engine Robust compression engine instance
 * @param analysis_period_hours Period to analyze (hours)
 * @param recommendations Output recommendations buffer
 * @param recommendations_size Size of recommendations buffer
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_generate_recommendations(RobustCompressionEngine* engine,
                                                        uint32_t analysis_period_hours,
                                                        char* recommendations,
                                                        size_t recommendations_size);

// Emergency and Safety Functions

/**
 * Initiate emergency shutdown sequence
 * @param engine Robust compression engine instance
 * @param shutdown_reason Reason for emergency shutdown
 * @param save_state Whether to save current state
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_emergency_shutdown(RobustCompressionEngine* engine,
                                                   const char* shutdown_reason,
                                                   bool save_state);

/**
 * Perform controlled safe restart
 * @param engine Robust compression engine instance
 * @param restart_mode Restart mode configuration
 * @param restart_successful Output boolean for restart success
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_safe_restart(RobustCompressionEngine* engine,
                                             uint32_t restart_mode,
                                             bool* restart_successful);

/**
 * Create system state snapshot for recovery
 * @param engine Robust compression engine instance
 * @param snapshot_buffer Output buffer for snapshot
 * @param snapshot_buffer_size Size of snapshot buffer
 * @param snapshot_size Output size of snapshot
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_create_state_snapshot(RobustCompressionEngine* engine,
                                                      void* snapshot_buffer,
                                                      size_t snapshot_buffer_size,
                                                      size_t* snapshot_size);

/**
 * Restore system from state snapshot
 * @param engine Robust compression engine instance
 * @param snapshot_data Snapshot data to restore from
 * @param snapshot_size Size of snapshot data
 * @param restore_successful Output boolean for restore success
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_restore_from_snapshot(RobustCompressionEngine* engine,
                                                      const void* snapshot_data,
                                                      size_t snapshot_size,
                                                      bool* restore_successful);

// Configuration Creation Functions

/**
 * Create default error detection configuration
 * @param config Output default error detection configuration
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_create_default_error_config(ErrorDetectionConfig* config);

/**
 * Create default robustness configuration
 * @param config Output default robustness configuration
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_create_default_robustness_config(RobustnessConfig* config);

/**
 * Create high-reliability configuration
 * @param error_config Output high-reliability error detection configuration
 * @param robustness_config Output high-reliability robustness configuration
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_create_high_reliability_config(ErrorDetectionConfig* error_config,
                                                              RobustnessConfig* robustness_config);

/**
 * Create performance-optimized configuration
 * @param error_config Output performance-optimized error detection configuration
 * @param robustness_config Output performance-optimized robustness configuration
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_create_performance_config(ErrorDetectionConfig* error_config,
                                                         RobustnessConfig* robustness_config);

/**
 * Validate configuration settings
 * @param error_config Error detection configuration to validate
 * @param robustness_config Robustness configuration to validate
 * @param is_valid Output boolean for configuration validity
 * @param validation_message Output validation message buffer
 * @param message_size Size of validation message buffer
 * @return ROBUST_ENGINE_SUCCESS on success, error code on failure
 */
RobustEngineError robust_engine_validate_configuration(const ErrorDetectionConfig* error_config,
                                                       const RobustnessConfig* robustness_config,
                                                       bool* is_valid,
                                                       char* validation_message,
                                                       size_t message_size);

/**
 * Destroy robust compression engine and free resources
 * @param engine Robust compression engine instance to destroy
 */
void robust_engine_destroy(RobustCompressionEngine* engine);

// Utility Functions

/**
 * Get error string for robust engine error code
 * @param error_code RobustEngineError code
 * @return Human-readable error message
 */
const char* robust_engine_get_error_string(RobustEngineError error_code);

/**
 * Get error severity string
 * @param severity Error severity enum
 * @return Human-readable severity name
 */
const char* robust_engine_get_severity_string(ErrorSeverity severity);

/**
 * Get error category string
 * @param category Error category enum
 * @return Human-readable category name
 */
const char* robust_engine_get_category_string(ErrorCategory category);

/**
 * Get recovery strategy string
 * @param strategy Recovery strategy enum
 * @return Human-readable strategy name
 */
const char* robust_engine_get_recovery_strategy_string(RecoveryStrategy strategy);

/**
 * Get fallback mode string
 * @param fallback_mode Fallback mode enum
 * @return Human-readable fallback mode name
 */
const char* robust_engine_get_fallback_mode_string(FallbackMode fallback_mode);

/**
 * Calculate error impact score
 * @param severity Error severity
 * @param category Error category
 * @param frequency Error frequency
 * @return Impact score (0-100)
 */
float robust_engine_calculate_error_impact_score(ErrorSeverity severity,
                                                 ErrorCategory category,
                                                 uint32_t frequency);

/**
 * Estimate recovery time for error type
 * @param severity Error severity
 * @param strategy Recovery strategy
 * @param system_load Current system load (0-1)
 * @return Estimated recovery time in milliseconds
 */
uint32_t robust_engine_estimate_recovery_time(ErrorSeverity severity,
                                              RecoveryStrategy strategy,
                                              float system_load);

/**
 * Check if error pattern indicates system degradation
 * @param recent_errors Array of recent errors
 * @param error_count Number of recent errors
 * @param time_window_ms Time window for analysis
 * @return True if degradation pattern detected
 */
bool robust_engine_detect_degradation_pattern(const ErrorInfo* recent_errors,
                                              uint32_t error_count,
                                              uint64_t time_window_ms);

// Constants for robust compression engine

// Error detection timing
#define DEFAULT_ERROR_DETECTION_INTERVAL_MS 100     // Default error detection interval
#define FAST_ERROR_DETECTION_INTERVAL_MS 50         // Fast error detection interval
#define SLOW_ERROR_DETECTION_INTERVAL_MS 500        // Slow error detection interval

// Recovery timing
#define DEFAULT_RETRY_BACKOFF_BASE_MS 100           // Default retry backoff base
#define MAX_RETRY_BACKOFF_MS 30000                  // Maximum retry backoff
#define DEFAULT_RECOVERY_TIMEOUT_MS 5000            // Default recovery timeout
#define EMERGENCY_RECOVERY_TIMEOUT_MS 1000          // Emergency recovery timeout

// System thresholds
#define MEMORY_PRESSURE_WARNING_THRESHOLD 0.8f     // Memory pressure warning
#define MEMORY_PRESSURE_CRITICAL_THRESHOLD 0.95f   // Memory pressure critical
#define THERMAL_WARNING_THRESHOLD 75.0f            // Thermal warning (°C)
#define THERMAL_CRITICAL_THRESHOLD 85.0f           // Thermal critical (°C)
#define PERFORMANCE_DEGRADATION_THRESHOLD 0.5f     // Performance degradation threshold

// Reliability targets
#define TARGET_SYSTEM_RELIABILITY_SCORE 0.99f      // Target reliability score
#define TARGET_RECOVERY_SUCCESS_RATE 0.95f         // Target recovery success rate
#define ACCEPTABLE_ERROR_RATE_PER_HOUR 10          // Acceptable error rate
#define CRITICAL_ERROR_ESCALATION_THRESHOLD 3      // Critical error escalation

// Quality and performance
#define MINIMUM_ACCEPTABLE_QUALITY 0.8f            // Minimum quality threshold
#define MINIMUM_ACCEPTABLE_PERFORMANCE 0.6f        // Minimum performance threshold
#define DEGRADATION_STEP_SIZE_DEFAULT 0.1f         // Default degradation step
#define RESTORATION_QUALITY_TARGET 0.95f           // Target quality for restoration

#ifdef __cplusplus
}
#endif

#endif // ROBUST_COMPRESSION_ENGINE_H
