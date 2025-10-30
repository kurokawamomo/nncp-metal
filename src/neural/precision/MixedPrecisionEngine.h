/*
 * MixedPrecisionEngine.h
 * 
 * Mixed Precision Computing Engine for Apple Silicon
 * Intelligent precision selection with float32 for critical operations
 * and float16 acceleration for FFN intermediate layers
 */

#ifndef MIXED_PRECISION_ENGINE_H
#define MIXED_PRECISION_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../acceleration/MetalComputeAccelerator.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct MixedPrecisionEngine MixedPrecisionEngine;

// Precision types supported
typedef enum {
    PRECISION_TYPE_FLOAT32 = 0,        // Full precision (32-bit)
    PRECISION_TYPE_FLOAT16,            // Half precision (16-bit)
    PRECISION_TYPE_BFLOAT16,           // Brain float (16-bit, Google format)
    PRECISION_TYPE_MIXED_ADAPTIVE,     // Adaptive mixed precision
    PRECISION_TYPE_MIXED_CONSERVATIVE  // Conservative mixed precision
} PrecisionType;

// Operation criticality levels
typedef enum {
    OPERATION_CRITICAL_LEVEL_CRITICAL = 0,  // Must use float32 (attention, gradients)
    OPERATION_CRITICAL_LEVEL_IMPORTANT,     // Should use float32 (layer norm, softmax)
    OPERATION_CRITICAL_LEVEL_STANDARD,      // Can use float16 (FFN intermediate)
    OPERATION_CRITICAL_LEVEL_ACCELERATED    // Prioritize speed with float16
} OperationCriticalLevel;

// Numerical stability monitoring
typedef enum {
    STABILITY_STATUS_STABLE = 0,       // Numerically stable
    STABILITY_STATUS_WARNING,          // Potential instability detected
    STABILITY_STATUS_UNSTABLE,         // Instability detected, fallback needed
    STABILITY_STATUS_CRITICAL          // Critical instability, emergency fallback
} NumericalStabilityStatus;

// Mixed precision configuration
typedef struct {
    bool enable_automatic_precision_selection; // Enable automatic precision selection
    bool enable_stability_monitoring;    // Enable numerical stability monitoring
    bool enable_gradient_scaling;        // Enable gradient scaling for stability
    bool enable_loss_scaling;           // Enable loss scaling
    bool enable_overflow_detection;     // Enable overflow/underflow detection
    bool enable_performance_optimization; // Enable performance-driven precision
    float stability_threshold;          // Numerical stability threshold
    float gradient_scale_factor;        // Gradient scaling factor
    float loss_scale_factor;            // Loss scaling factor
    uint32_t stability_check_interval;  // Stability check interval (operations)
    uint32_t fallback_threshold;        // Fallback threshold (instabilities)
} MixedPrecisionConfig;

// Operation configuration for precision selection
typedef struct {
    OperationCriticalLevel criticality; // Operation criticality level
    PrecisionType forced_precision;     // Force specific precision (optional)
    bool enable_overflow_check;        // Enable overflow checking for this op
    bool enable_underflow_check;       // Enable underflow checking for this op
    bool enable_nan_check;             // Enable NaN checking for this op
    float precision_tolerance;         // Acceptable precision loss (0.0-1.0)
    uint32_t input_size;               // Input tensor size
    uint32_t output_size;              // Output tensor size
    const char* operation_name;        // Operation name for debugging
} MixedPrecisionOpConfig;

// FFN (Feed-Forward Network) specific configuration
typedef struct {
    uint32_t input_dimension;          // Input dimension (d_model = 768)
    uint32_t hidden_dimension;         // Hidden dimension (d_ff = 3072)
    uint32_t batch_size;               // Batch size
    uint32_t sequence_length;          // Sequence length
    bool use_swiglu_activation;        // Use SwiGLU activation
    bool use_gelu_activation;          // Use GELU activation
    bool enable_intermediate_float16;  // Use float16 for intermediate layers
    bool enable_weight_caching;        // Enable weight caching
    float dropout_rate;                // Dropout rate
} FFNPrecisionConfig;

// Attention mechanism precision configuration
typedef struct {
    uint32_t sequence_length;          // Sequence length
    uint32_t d_model;                  // Model dimension (768)
    uint32_t num_heads;                // Number of attention heads (16)
    uint32_t head_dimension;           // Head dimension (48)
    bool force_float32_attention;      // Force float32 for attention computation
    bool force_float32_softmax;        // Force float32 for softmax
    bool enable_flash_attention;       // Enable Flash Attention optimization
    bool enable_attention_scaling;     // Enable attention scaling
    float attention_scale;             // Attention scaling factor
} AttentionPrecisionConfig;

// Precision operation statistics
typedef struct {
    uint64_t total_operations;         // Total operations performed
    uint64_t float32_operations;       // Operations using float32
    uint64_t float16_operations;       // Operations using float16
    uint64_t bfloat16_operations;      // Operations using bfloat16
    uint64_t precision_fallbacks;      // Precision fallbacks due to instability
    uint64_t overflow_detections;      // Overflow detections
    uint64_t underflow_detections;     // Underflow detections
    uint64_t nan_detections;           // NaN detections
    float average_precision_speedup;   // Average speedup from mixed precision
    float numerical_stability_score;   // Overall numerical stability score
    uint64_t total_compute_time_us;    // Total compute time in microseconds
} MixedPrecisionStats;

// Numerical stability analysis results
typedef struct {
    NumericalStabilityStatus status;   // Current stability status
    float stability_score;             // Stability score (0.0-1.0)
    float gradient_norm;               // Gradient norm
    float parameter_norm;              // Parameter norm
    float loss_value;                  // Current loss value
    uint32_t overflow_count;           // Number of overflows detected
    uint32_t underflow_count;          // Number of underflows detected
    uint32_t nan_count;                // Number of NaNs detected
    uint32_t inf_count;                // Number of infinities detected
    bool requires_precision_fallback;  // Requires precision fallback
    PrecisionType recommended_precision; // Recommended precision type
} NumericalStabilityAnalysis;

// Mixed precision operation result
typedef struct {
    PrecisionType used_precision;      // Precision type used for operation
    NumericalStabilityStatus stability_status; // Stability status after operation
    uint64_t operation_time_us;        // Operation execution time
    float precision_speedup;           // Speedup achieved (vs float32)
    bool fallback_occurred;            // Whether fallback occurred
    float numerical_error;             // Estimated numerical error
    size_t memory_saved_bytes;         // Memory saved by using lower precision
} MixedPrecisionOpResult;

// Error codes for mixed precision engine
typedef enum {
    MIXED_PRECISION_SUCCESS = 0,
    MIXED_PRECISION_ERROR_INVALID_PARAM,
    MIXED_PRECISION_ERROR_MEMORY_ALLOCATION,
    MIXED_PRECISION_ERROR_UNSUPPORTED_PRECISION,
    MIXED_PRECISION_ERROR_NUMERICAL_INSTABILITY,
    MIXED_PRECISION_ERROR_OVERFLOW_DETECTED,
    MIXED_PRECISION_ERROR_UNDERFLOW_DETECTED,
    MIXED_PRECISION_ERROR_NAN_DETECTED,
    MIXED_PRECISION_ERROR_METAL_DEVICE_ERROR,
    MIXED_PRECISION_ERROR_CONVERSION_FAILED,
    MIXED_PRECISION_ERROR_STABILITY_CHECK_FAILED,
    MIXED_PRECISION_ERROR_FALLBACK_FAILED
} MixedPrecisionError;

// Core API Functions

/**
 * Create mixed precision engine
 * @param engine Pointer to store created engine
 * @param config Mixed precision configuration
 * @param metal_accelerator Metal compute accelerator for GPU operations
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_create(MixedPrecisionEngine** engine,
                                          const MixedPrecisionConfig* config,
                                          MetalComputeAccelerator* metal_accelerator);

/**
 * Initialize mixed precision engine with Apple Silicon optimizations
 * @param engine Mixed precision engine instance
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_initialize(MixedPrecisionEngine* engine);

/**
 * Automatically select optimal precision for operation
 * @param engine Mixed precision engine instance
 * @param op_config Operation configuration
 * @param input_data Input data for analysis
 * @param input_size Size of input data
 * @param selected_precision Output selected precision type
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_select_optimal_precision(MixedPrecisionEngine* engine,
                                                            const MixedPrecisionOpConfig* op_config,
                                                            const float* input_data,
                                                            size_t input_size,
                                                            PrecisionType* selected_precision);

/**
 * Perform numerical stability analysis
 * @param engine Mixed precision engine instance
 * @param data Data to analyze for stability
 * @param data_size Size of data
 * @param stability_analysis Output stability analysis results
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_analyze_stability(MixedPrecisionEngine* engine,
                                                     const float* data,
                                                     size_t data_size,
                                                     NumericalStabilityAnalysis* stability_analysis);

// Critical Operations (Float32 Required)

/**
 * Compute attention weights with float32 precision
 * @param engine Mixed precision engine instance
 * @param attention_config Attention precision configuration
 * @param query_matrix Query matrix (float32)
 * @param key_matrix Key matrix (float32)
 * @param value_matrix Value matrix (float32)
 * @param attention_output Output attention matrix (float32)
 * @param attention_weights Optional output attention weights (float32)
 * @param op_result Output operation result with statistics
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_compute_attention_float32(MixedPrecisionEngine* engine,
                                                             const AttentionPrecisionConfig* attention_config,
                                                             const float* query_matrix,
                                                             const float* key_matrix,
                                                             const float* value_matrix,
                                                             float* attention_output,
                                                             float* attention_weights,
                                                             MixedPrecisionOpResult* op_result);

/**
 * Compute softmax with float32 precision for numerical stability
 * @param engine Mixed precision engine instance
 * @param input_vectors Input vectors (float32)
 * @param output_vectors Output softmax vectors (float32)
 * @param vector_length Length of each vector
 * @param batch_size Number of vectors
 * @param op_result Output operation result with statistics
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_compute_softmax_float32(MixedPrecisionEngine* engine,
                                                           const float* input_vectors,
                                                           float* output_vectors,
                                                           uint32_t vector_length,
                                                           uint32_t batch_size,
                                                           MixedPrecisionOpResult* op_result);

/**
 * Compute layer normalization with float32 precision
 * @param engine Mixed precision engine instance
 * @param input_vectors Input vectors (float32)
 * @param gamma_weights Gamma scaling weights (float32)
 * @param beta_weights Beta bias weights (float32)
 * @param output_vectors Output normalized vectors (float32)
 * @param vector_length Length of each vector
 * @param batch_size Number of vectors
 * @param epsilon Normalization epsilon
 * @param op_result Output operation result with statistics
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_compute_layer_norm_float32(MixedPrecisionEngine* engine,
                                                              const float* input_vectors,
                                                              const float* gamma_weights,
                                                              const float* beta_weights,
                                                              float* output_vectors,
                                                              uint32_t vector_length,
                                                              uint32_t batch_size,
                                                              float epsilon,
                                                              MixedPrecisionOpResult* op_result);

// Accelerated Operations (Float16 Optimized)

/**
 * Compute FFN intermediate layer with float16 acceleration
 * @param engine Mixed precision engine instance
 * @param ffn_config FFN precision configuration
 * @param input_matrix Input matrix (converted to float16 internally)
 * @param weight_matrix Weight matrix (converted to float16 internally)
 * @param bias_vector Bias vector (converted to float16 internally)
 * @param output_matrix Output matrix (converted back to float32)
 * @param op_result Output operation result with statistics
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_compute_ffn_float16(MixedPrecisionEngine* engine,
                                                       const FFNPrecisionConfig* ffn_config,
                                                       const float* input_matrix,
                                                       const float* weight_matrix,
                                                       const float* bias_vector,
                                                       float* output_matrix,
                                                       MixedPrecisionOpResult* op_result);

/**
 * Compute matrix multiplication with float16 acceleration
 * @param engine Mixed precision engine instance
 * @param matrix_a Input matrix A (converted to float16 internally)
 * @param matrix_b Input matrix B (converted to float16 internally)
 * @param result_matrix Output result matrix (converted back to float32)
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A (rows in matrix B)
 * @param cols_b Number of columns in matrix B
 * @param op_result Output operation result with statistics
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_compute_matrix_multiply_float16(MixedPrecisionEngine* engine,
                                                                   const float* matrix_a,
                                                                   const float* matrix_b,
                                                                   float* result_matrix,
                                                                   uint32_t rows_a,
                                                                   uint32_t cols_a,
                                                                   uint32_t cols_b,
                                                                   MixedPrecisionOpResult* op_result);

/**
 * Compute GELU activation with float16 acceleration
 * @param engine Mixed precision engine instance
 * @param input_vectors Input vectors (converted to float16 internally)
 * @param output_vectors Output GELU vectors (converted back to float32)
 * @param total_elements Total number of elements to process
 * @param op_result Output operation result with statistics
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_compute_gelu_float16(MixedPrecisionEngine* engine,
                                                        const float* input_vectors,
                                                        float* output_vectors,
                                                        uint32_t total_elements,
                                                        MixedPrecisionOpResult* op_result);

/**
 * Compute SwiGLU activation with float16 acceleration
 * @param engine Mixed precision engine instance
 * @param input_vectors Input vectors (converted to float16 internally)
 * @param gate_vectors Gate vectors (converted to float16 internally)
 * @param output_vectors Output SwiGLU vectors (converted back to float32)
 * @param total_elements Total number of elements to process
 * @param op_result Output operation result with statistics
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_compute_swiglu_float16(MixedPrecisionEngine* engine,
                                                          const float* input_vectors,
                                                          const float* gate_vectors,
                                                          float* output_vectors,
                                                          uint32_t total_elements,
                                                          MixedPrecisionOpResult* op_result);

// Data Conversion and Utilities

/**
 * Convert float32 array to float16
 * @param engine Mixed precision engine instance
 * @param float32_data Input float32 data
 * @param float16_data Output float16 data buffer
 * @param element_count Number of elements to convert
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_convert_float32_to_float16(MixedPrecisionEngine* engine,
                                                              const float* float32_data,
                                                              uint16_t* float16_data,
                                                              size_t element_count);

/**
 * Convert float16 array to float32
 * @param engine Mixed precision engine instance
 * @param float16_data Input float16 data
 * @param float32_data Output float32 data buffer
 * @param element_count Number of elements to convert
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_convert_float16_to_float32(MixedPrecisionEngine* engine,
                                                              const uint16_t* float16_data,
                                                              float* float32_data,
                                                              size_t element_count);

/**
 * Check for numerical issues in data
 * @param engine Mixed precision engine instance
 * @param data Data to check
 * @param element_count Number of elements to check
 * @param overflow_count Output number of overflow values
 * @param underflow_count Output number of underflow values
 * @param nan_count Output number of NaN values
 * @param inf_count Output number of infinity values
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_check_numerical_issues(MixedPrecisionEngine* engine,
                                                          const float* data,
                                                          size_t element_count,
                                                          uint32_t* overflow_count,
                                                          uint32_t* underflow_count,
                                                          uint32_t* nan_count,
                                                          uint32_t* inf_count);

// Performance and Statistics

/**
 * Get mixed precision engine performance statistics
 * @param engine Mixed precision engine instance
 * @param stats Output performance statistics
 */
void mixed_precision_get_stats(MixedPrecisionEngine* engine,
                               MixedPrecisionStats* stats);

/**
 * Reset performance counters
 * @param engine Mixed precision engine instance
 */
void mixed_precision_reset_counters(MixedPrecisionEngine* engine);

/**
 * Update mixed precision configuration
 * @param engine Mixed precision engine instance
 * @param config New configuration
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_update_config(MixedPrecisionEngine* engine,
                                                  const MixedPrecisionConfig* config);

/**
 * Destroy mixed precision engine and free resources
 * @param engine Mixed precision engine instance to destroy
 */
void mixed_precision_destroy(MixedPrecisionEngine* engine);

// Configuration Functions

/**
 * Create default mixed precision configuration
 * @param config Output default configuration
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_create_default_config(MixedPrecisionConfig* config);

/**
 * Create Apple Silicon optimized mixed precision configuration
 * @param config Output Apple Silicon optimized configuration
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_create_apple_silicon_config(MixedPrecisionConfig* config);

/**
 * Create conservative mixed precision configuration (prioritizes stability)
 * @param config Output conservative configuration
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_create_conservative_config(MixedPrecisionConfig* config);

/**
 * Create aggressive mixed precision configuration (prioritizes performance)
 * @param config Output aggressive configuration
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_create_aggressive_config(MixedPrecisionConfig* config);

/**
 * Validate mixed precision configuration
 * @param config Configuration to validate
 * @param metal_accelerator Metal accelerator for capability checking
 * @param is_valid Output boolean for validity
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_validate_config(const MixedPrecisionConfig* config,
                                                    MetalComputeAccelerator* metal_accelerator,
                                                    bool* is_valid);

// Utility Functions

/**
 * Get error string for mixed precision error code
 * @param error_code MixedPrecisionError code
 * @return Human-readable error message
 */
const char* mixed_precision_get_error_string(MixedPrecisionError error_code);

/**
 * Get precision type string
 * @param precision_type Precision type
 * @return Human-readable precision type name
 */
const char* mixed_precision_get_precision_type_string(PrecisionType precision_type);

/**
 * Get operation criticality level string
 * @param criticality Criticality level
 * @return Human-readable criticality level name
 */
const char* mixed_precision_get_criticality_string(OperationCriticalLevel criticality);

/**
 * Get stability status string
 * @param status Stability status
 * @return Human-readable stability status name
 */
const char* mixed_precision_get_stability_status_string(NumericalStabilityStatus status);

/**
 * Estimate memory savings with mixed precision
 * @param total_parameters Total number of model parameters
 * @param float16_ratio Ratio of parameters using float16 (0.0-1.0)
 * @param estimated_savings_mb Output estimated memory savings in MB
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_estimate_memory_savings(size_t total_parameters,
                                                           float float16_ratio,
                                                           size_t* estimated_savings_mb);

/**
 * Estimate performance improvement with mixed precision
 * @param operation_type Type of operation
 * @param data_size Size of operation data
 * @param precision_type Target precision type
 * @param estimated_speedup Output estimated speedup factor
 * @return MIXED_PRECISION_SUCCESS on success, error code on failure
 */
MixedPrecisionError mixed_precision_estimate_performance_improvement(const char* operation_type,
                                                                    size_t data_size,
                                                                    PrecisionType precision_type,
                                                                    float* estimated_speedup);

// Constants for mixed precision engine
#define MIXED_PRECISION_DEFAULT_STABILITY_THRESHOLD 1e-6f    // Default stability threshold
#define MIXED_PRECISION_DEFAULT_GRADIENT_SCALE 65536.0f      // Default gradient scaling factor
#define MIXED_PRECISION_DEFAULT_LOSS_SCALE 512.0f            // Default loss scaling factor
#define MIXED_PRECISION_STABILITY_CHECK_INTERVAL 100         // Check stability every 100 ops
#define MIXED_PRECISION_FALLBACK_THRESHOLD 5                 // Fallback after 5 instabilities

// Float16 constants
#define FLOAT16_MAX_NORMAL 65504.0f                          // Maximum normal float16 value
#define FLOAT16_MIN_NORMAL 6.103515625e-05f                  // Minimum normal float16 value
#define FLOAT16_EPSILON 9.765625e-04f                        // Float16 machine epsilon

// Numerical stability thresholds
#define STABILITY_OVERFLOW_THRESHOLD 3.4e38f                 // Overflow threshold
#define STABILITY_UNDERFLOW_THRESHOLD 1.175494e-38f          // Underflow threshold
#define STABILITY_GRADIENT_NORM_THRESHOLD 1000.0f            // Gradient norm threshold
#define STABILITY_LOSS_THRESHOLD 1e6f                        // Loss value threshold

#ifdef __cplusplus
}
#endif

#endif // MIXED_PRECISION_ENGINE_H
