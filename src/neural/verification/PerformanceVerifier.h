/*
 * PerformanceVerifier.h
 * 
 * Performance Verification Engine for Apple Silicon Optimization
 * Comprehensive performance measurement, comparison analysis, bottleneck detection,
 * and optimization validation for neural compression systems
 */

#ifndef PERFORMANCE_VERIFIER_H
#define PERFORMANCE_VERIFIER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../precision/MixedPrecisionEngine.h"
#include "../optimization/CacheOptimizer.h"
#include "../parallel/ParallelProcessor.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct PerformanceVerifier PerformanceVerifier;

// Apple Silicon model detection
typedef enum {
    APPLE_SILICON_MODEL_M1 = 0,           // M1 chip
    APPLE_SILICON_MODEL_M1_PRO,           // M1 Pro chip
    APPLE_SILICON_MODEL_M1_MAX,           // M1 Max chip
    APPLE_SILICON_MODEL_M1_ULTRA,         // M1 Ultra chip
    APPLE_SILICON_MODEL_M2,               // M2 chip
    APPLE_SILICON_MODEL_M2_PRO,           // M2 Pro chip
    APPLE_SILICON_MODEL_M2_MAX,           // M2 Max chip
    APPLE_SILICON_MODEL_M2_ULTRA,         // M2 Ultra chip
    APPLE_SILICON_MODEL_M3,               // M3 chip
    APPLE_SILICON_MODEL_M3_PRO,           // M3 Pro chip
    APPLE_SILICON_MODEL_M3_MAX,           // M3 Max chip
    APPLE_SILICON_MODEL_UNKNOWN          // Unknown or unsupported model
} AppleSiliconModel;

// Performance measurement categories
typedef enum {
    PERFORMANCE_CATEGORY_COMPRESSION = 0,  // Compression performance
    PERFORMANCE_CATEGORY_DECOMPRESSION,    // Decompression performance
    PERFORMANCE_CATEGORY_MEMORY,           // Memory performance
    PERFORMANCE_CATEGORY_GPU,              // GPU acceleration performance
    PERFORMANCE_CATEGORY_CPU,              // CPU performance
    PERFORMANCE_CATEGORY_CACHE,            // Cache efficiency
    PERFORMANCE_CATEGORY_PARALLEL,         // Parallel processing performance
    PERFORMANCE_CATEGORY_PRECISION,        // Mixed precision performance
    PERFORMANCE_CATEGORY_OVERALL           // Overall system performance
} PerformanceCategory;

// System hardware characteristics
typedef struct {
    AppleSiliconModel silicon_model;       // Apple Silicon model
    uint32_t cpu_core_count;               // Total CPU cores
    uint32_t performance_cores;            // Performance cores (P-cores)
    uint32_t efficiency_cores;             // Efficiency cores (E-cores)
    uint32_t gpu_core_count;               // GPU cores
    uint64_t unified_memory_gb;            // Unified memory size (GB)
    uint32_t memory_bandwidth_gbps;        // Memory bandwidth (GB/s)
    uint32_t neural_engine_tops;           // Neural Engine TOPS
    float base_cpu_frequency_ghz;          // Base CPU frequency
    float max_cpu_frequency_ghz;           // Maximum CPU frequency
    float gpu_frequency_ghz;               // GPU frequency
    bool supports_metal_3;                 // Metal 3 support
    bool supports_neural_engine;           // Neural Engine support
    char silicon_name[64];                 // Human-readable silicon name
} AppleSiliconSystemSpecs;

// Performance measurement configuration
typedef struct {
    bool measure_compression_performance;   // Enable compression measurement
    bool measure_decompression_performance; // Enable decompression measurement
    bool measure_memory_performance;        // Enable memory performance measurement
    bool measure_gpu_performance;           // Enable GPU performance measurement
    bool measure_cache_performance;         // Enable cache performance measurement
    bool measure_parallel_performance;      // Enable parallel performance measurement
    bool measure_precision_performance;     // Enable mixed precision measurement
    bool enable_detailed_profiling;        // Enable detailed profiling
    bool enable_thermal_monitoring;        // Enable thermal monitoring
    bool enable_power_monitoring;          // Enable power consumption monitoring
    uint32_t measurement_iterations;        // Number of measurement iterations
    uint32_t warm_up_iterations;           // Number of warm-up iterations
    uint32_t measurement_duration_seconds;  // Measurement duration
    float measurement_confidence_level;     // Statistical confidence level (0.90-0.99)
} PerformanceMeasurementConfig;

// Performance metrics for a single operation
typedef struct {
    uint64_t operation_count;              // Number of operations performed
    uint64_t total_time_ns;                // Total time in nanoseconds
    uint64_t cpu_time_ns;                  // CPU time in nanoseconds
    uint64_t gpu_time_ns;                  // GPU time in nanoseconds
    uint64_t memory_transfer_time_ns;      // Memory transfer time
    uint64_t cache_access_time_ns;         // Cache access time
    uint64_t synchronization_time_ns;      // Synchronization overhead time
    double average_time_ns;                // Average time per operation
    double median_time_ns;                 // Median time per operation
    double min_time_ns;                    // Minimum time per operation
    double max_time_ns;                    // Maximum time per operation
    double std_deviation_ns;               // Standard deviation of times
    double throughput_ops_per_second;      // Operations per second
    double data_throughput_mbps;           // Data throughput (MB/s)
    float cpu_utilization_percent;         // CPU utilization percentage
    float gpu_utilization_percent;         // GPU utilization percentage
    float memory_utilization_percent;      // Memory utilization percentage
    float cache_hit_rate;                  // Cache hit rate (0.0-1.0)
    float parallel_efficiency;             // Parallel efficiency (0.0-1.0)
    uint64_t memory_peak_usage_bytes;      // Peak memory usage
    uint64_t gpu_memory_peak_usage_bytes;  // Peak GPU memory usage
    float power_consumption_watts;         // Power consumption
    float thermal_state_celsius;           // Thermal state (°C)
} PerformanceMetrics;

// Comprehensive performance report
typedef struct {
    AppleSiliconSystemSpecs system_specs;  // System specifications
    PerformanceMetrics compression_metrics; // Compression performance
    PerformanceMetrics decompression_metrics; // Decompression performance
    PerformanceMetrics memory_metrics;      // Memory performance
    PerformanceMetrics gpu_metrics;         // GPU performance
    PerformanceMetrics cache_metrics;       // Cache performance
    PerformanceMetrics parallel_metrics;    // Parallel performance
    PerformanceMetrics precision_metrics;   // Mixed precision performance
    PerformanceMetrics overall_metrics;     // Overall performance
    uint64_t total_measurement_time_ns;     // Total measurement time
    uint32_t measurement_iterations;        // Number of iterations performed
    float overall_performance_score;        // Overall performance score (0-100)
    float optimization_effectiveness_score; // Optimization effectiveness (0-100)
    bool meets_10x_speed_constraint;        // Meets <10x speed constraint
    char measurement_timestamp[64];         // Measurement timestamp
} PerformanceReport;

// Performance comparison analysis
typedef struct {
    PerformanceReport baseline_report;      // Baseline (unoptimized) performance
    PerformanceReport optimized_report;     // Optimized performance
    float compression_speedup_factor;       // Compression speedup factor
    float decompression_speedup_factor;     // Decompression speedup factor
    float memory_efficiency_improvement;    // Memory efficiency improvement
    float gpu_acceleration_benefit;         // GPU acceleration benefit
    float cache_optimization_benefit;       // Cache optimization benefit
    float parallel_processing_benefit;      // Parallel processing benefit
    float mixed_precision_benefit;          // Mixed precision benefit
    float overall_speedup_factor;           // Overall speedup factor
    float performance_consistency_score;    // Performance consistency (0-100)
    float optimization_roi_score;           // Optimization ROI score (0-100)
    bool significant_improvement;           // Statistically significant improvement
    char comparison_summary[1024];          // Human-readable comparison summary
} PerformanceComparison;

// Bottleneck analysis results
typedef struct {
    PerformanceCategory primary_bottleneck; // Primary performance bottleneck
    PerformanceCategory secondary_bottleneck; // Secondary bottleneck
    float bottleneck_severity_score;        // Bottleneck severity (0-100)
    float cpu_bottleneck_factor;            // CPU bottleneck factor (0-1)
    float gpu_bottleneck_factor;            // GPU bottleneck factor (0-1)
    float memory_bottleneck_factor;         // Memory bottleneck factor (0-1)
    float cache_bottleneck_factor;          // Cache bottleneck factor (0-1)
    float parallel_bottleneck_factor;       // Parallelization bottleneck factor (0-1)
    float precision_bottleneck_factor;      // Mixed precision bottleneck factor (0-1)
    uint32_t improvement_opportunities_count; // Number of improvement opportunities
    char improvement_recommendations[2048];  // Detailed improvement recommendations
    float estimated_improvement_potential;   // Estimated improvement potential (%)
} BottleneckAnalysis;

// Optimization validation results
typedef struct {
    bool optimization_successful;           // Overall optimization success
    bool meets_compression_targets;         // Meets compression ratio targets
    bool meets_speed_constraints;           // Meets speed constraints
    bool maintains_data_integrity;          // Maintains 100% data integrity
    bool passes_stability_tests;            // Passes stability tests
    bool passes_stress_tests;               // Passes stress tests
    float compression_ratio_achieved;       // Achieved compression ratio (%)
    float speed_constraint_compliance;      // Speed constraint compliance (0-1)
    float stability_score;                  // Stability score (0-100)
    float robustness_score;                 // Robustness score (0-100)
    uint32_t failed_test_count;             // Number of failed tests
    char validation_summary[1024];          // Validation summary
} OptimizationValidation;

// Error codes for performance verification
typedef enum {
    PERFORMANCE_VERIFIER_SUCCESS = 0,
    PERFORMANCE_VERIFIER_ERROR_INVALID_PARAM,
    PERFORMANCE_VERIFIER_ERROR_MEMORY_ALLOCATION,
    PERFORMANCE_VERIFIER_ERROR_SYSTEM_DETECTION_FAILED,
    PERFORMANCE_VERIFIER_ERROR_MEASUREMENT_FAILED,
    PERFORMANCE_VERIFIER_ERROR_ANALYSIS_FAILED,
    PERFORMANCE_VERIFIER_ERROR_INSUFFICIENT_DATA,
    PERFORMANCE_VERIFIER_ERROR_HARDWARE_INCOMPATIBLE,
    PERFORMANCE_VERIFIER_ERROR_OPTIMIZATION_NOT_FOUND,
    PERFORMANCE_VERIFIER_ERROR_VALIDATION_FAILED,
    PERFORMANCE_VERIFIER_ERROR_STATISTICAL_SIGNIFICANCE
} PerformanceVerifierError;

// Core API Functions

/**
 * Create performance verifier instance
 * @param verifier Pointer to store created verifier
 * @param config Performance measurement configuration
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_create(PerformanceVerifier** verifier,
                                                     const PerformanceMeasurementConfig* config);

/**
 * Initialize performance verifier with system detection
 * @param verifier Performance verifier instance
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_initialize(PerformanceVerifier* verifier);

/**
 * Detect Apple Silicon system specifications
 * @param verifier Performance verifier instance
 * @param system_specs Output system specifications
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_detect_system_specs(PerformanceVerifier* verifier,
                                                                   AppleSiliconSystemSpecs* system_specs);

// Performance Measurement Functions

/**
 * Measure compression performance
 * @param verifier Performance verifier instance
 * @param test_data Test data to compress
 * @param test_data_size Size of test data
 * @param compression_metrics Output compression performance metrics
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_measure_compression(PerformanceVerifier* verifier,
                                                                  const void* test_data,
                                                                  size_t test_data_size,
                                                                  PerformanceMetrics* compression_metrics);

/**
 * Measure decompression performance
 * @param verifier Performance verifier instance
 * @param compressed_data Compressed test data
 * @param compressed_size Size of compressed data
 * @param expected_output_size Expected decompressed size
 * @param decompression_metrics Output decompression performance metrics
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_measure_decompression(PerformanceVerifier* verifier,
                                                                    const void* compressed_data,
                                                                    size_t compressed_size,
                                                                    size_t expected_output_size,
                                                                    PerformanceMetrics* decompression_metrics);

/**
 * Measure GPU acceleration performance
 * @param verifier Performance verifier instance
 * @param gpu_workload GPU workload function pointer
 * @param workload_data Workload data
 * @param workload_size Size of workload data
 * @param gpu_metrics Output GPU performance metrics
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_measure_gpu_performance(PerformanceVerifier* verifier,
                                                                       void* gpu_workload,
                                                                       const void* workload_data,
                                                                       size_t workload_size,
                                                                       PerformanceMetrics* gpu_metrics);

/**
 * Measure cache optimization effectiveness
 * @param verifier Performance verifier instance
 * @param cache_optimizer Cache optimizer instance
 * @param test_workload Test workload for cache measurement
 * @param workload_size Size of test workload
 * @param cache_metrics Output cache performance metrics
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_measure_cache_performance(PerformanceVerifier* verifier,
                                                                         CacheOptimizer* cache_optimizer,
                                                                         const void* test_workload,
                                                                         size_t workload_size,
                                                                         PerformanceMetrics* cache_metrics);

/**
 * Measure parallel processing effectiveness
 * @param verifier Performance verifier instance
 * @param parallel_processor Parallel processor instance
 * @param parallel_workload Parallel workload function
 * @param workload_data Workload data
 * @param workload_size Size of workload data
 * @param parallel_metrics Output parallel processing metrics
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_measure_parallel_performance(PerformanceVerifier* verifier,
                                                                            ParallelProcessor* parallel_processor,
                                                                            void* parallel_workload,
                                                                            const void* workload_data,
                                                                            size_t workload_size,
                                                                            PerformanceMetrics* parallel_metrics);

/**
 * Measure mixed precision performance
 * @param verifier Performance verifier instance
 * @param precision_engine Mixed precision engine instance
 * @param precision_workload Mixed precision workload
 * @param workload_data Workload data
 * @param workload_size Size of workload data
 * @param precision_metrics Output mixed precision metrics
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_measure_precision_performance(PerformanceVerifier* verifier,
                                                                             MixedPrecisionEngine* precision_engine,
                                                                             void* precision_workload,
                                                                             const void* workload_data,
                                                                             size_t workload_size,
                                                                             PerformanceMetrics* precision_metrics);

/**
 * Run comprehensive performance measurement suite
 * @param verifier Performance verifier instance
 * @param test_data_set Array of test data for comprehensive testing
 * @param test_count Number of test data sets
 * @param report Output comprehensive performance report
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_run_comprehensive_measurement(PerformanceVerifier* verifier,
                                                                             void* const* test_data_set,
                                                                             size_t test_count,
                                                                             PerformanceReport* report);

// Performance Comparison and Analysis

/**
 * Compare baseline vs optimized performance
 * @param verifier Performance verifier instance
 * @param baseline_report Baseline performance report
 * @param optimized_report Optimized performance report
 * @param comparison Output performance comparison analysis
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_compare_performance(PerformanceVerifier* verifier,
                                                                   const PerformanceReport* baseline_report,
                                                                   const PerformanceReport* optimized_report,
                                                                   PerformanceComparison* comparison);

/**
 * Analyze performance bottlenecks
 * @param verifier Performance verifier instance
 * @param performance_report Performance report to analyze
 * @param bottleneck_analysis Output bottleneck analysis
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_analyze_bottlenecks(PerformanceVerifier* verifier,
                                                                   const PerformanceReport* performance_report,
                                                                   BottleneckAnalysis* bottleneck_analysis);

/**
 * Generate optimization recommendations
 * @param verifier Performance verifier instance
 * @param bottleneck_analysis Bottleneck analysis results
 * @param current_system_specs Current system specifications
 * @param recommendations Output optimization recommendations buffer
 * @param recommendations_size Size of recommendations buffer
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_generate_recommendations(PerformanceVerifier* verifier,
                                                                        const BottleneckAnalysis* bottleneck_analysis,
                                                                        const AppleSiliconSystemSpecs* current_system_specs,
                                                                        char* recommendations,
                                                                        size_t recommendations_size);

// Optimization Validation

/**
 * Validate optimization effectiveness
 * @param verifier Performance verifier instance
 * @param performance_comparison Performance comparison results
 * @param target_compression_ratio Target compression ratio (%)
 * @param max_speed_factor Maximum allowed speed factor
 * @param validation_result Output validation results
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_validate_optimization(PerformanceVerifier* verifier,
                                                                     const PerformanceComparison* performance_comparison,
                                                                     float target_compression_ratio,
                                                                     float max_speed_factor,
                                                                     OptimizationValidation* validation_result);

/**
 * Verify 10x speed constraint compliance
 * @param verifier Performance verifier instance
 * @param performance_report Performance report to check
 * @param baseline_time_ns Baseline processing time in nanoseconds
 * @param meets_constraint Output boolean for constraint compliance
 * @param actual_speed_factor Output actual speed factor
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_check_speed_constraint(PerformanceVerifier* verifier,
                                                                      const PerformanceReport* performance_report,
                                                                      uint64_t baseline_time_ns,
                                                                      bool* meets_constraint,
                                                                      float* actual_speed_factor);

/**
 * Run stress test for optimization stability
 * @param verifier Performance verifier instance
 * @param test_duration_minutes Duration of stress test in minutes
 * @param stress_intensity Stress test intensity (0.0-1.0)
 * @param stability_score Output stability score (0-100)
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_run_stress_test(PerformanceVerifier* verifier,
                                                              uint32_t test_duration_minutes,
                                                              float stress_intensity,
                                                              float* stability_score);

// Reporting and Export Functions

/**
 * Generate detailed performance report
 * @param verifier Performance verifier instance
 * @param performance_report Performance report data
 * @param report_format Report format ("text", "json", "html")
 * @param output_buffer Output buffer for report
 * @param buffer_size Size of output buffer
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_generate_report(PerformanceVerifier* verifier,
                                                              const PerformanceReport* performance_report,
                                                              const char* report_format,
                                                              char* output_buffer,
                                                              size_t buffer_size);

/**
 * Export performance data for external analysis
 * @param verifier Performance verifier instance
 * @param performance_report Performance report to export
 * @param export_format Export format ("csv", "json", "xml")
 * @param file_path Output file path
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_export_data(PerformanceVerifier* verifier,
                                                           const PerformanceReport* performance_report,
                                                           const char* export_format,
                                                           const char* file_path);

/**
 * Create performance visualization data
 * @param verifier Performance verifier instance
 * @param performance_comparison Performance comparison data
 * @param visualization_type Type of visualization ("speedup", "bottleneck", "trend")
 * @param visualization_data Output visualization data buffer
 * @param data_size Size of visualization data buffer
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_create_visualization(PerformanceVerifier* verifier,
                                                                    const PerformanceComparison* performance_comparison,
                                                                    const char* visualization_type,
                                                                    char* visualization_data,
                                                                    size_t data_size);

// Configuration and Utility Functions

/**
 * Create default performance measurement configuration
 * @param config Output default configuration
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_create_default_config(PerformanceMeasurementConfig* config);

/**
 * Create comprehensive measurement configuration
 * @param config Output comprehensive configuration
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_create_comprehensive_config(PerformanceMeasurementConfig* config);

/**
 * Create quick measurement configuration
 * @param config Output quick measurement configuration
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_create_quick_config(PerformanceMeasurementConfig* config);

/**
 * Validate measurement configuration
 * @param config Configuration to validate
 * @param is_valid Output boolean for validity
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_validate_config(const PerformanceMeasurementConfig* config,
                                                              bool* is_valid);

/**
 * Calculate statistical significance of performance differences
 * @param verifier Performance verifier instance
 * @param baseline_metrics Baseline performance metrics
 * @param optimized_metrics Optimized performance metrics
 * @param confidence_level Required confidence level (0.90-0.99)
 * @param is_significant Output boolean for statistical significance
 * @param p_value Output p-value of statistical test
 * @return PERFORMANCE_VERIFIER_SUCCESS on success, error code on failure
 */
PerformanceVerifierError performance_verifier_calculate_significance(PerformanceVerifier* verifier,
                                                                     const PerformanceMetrics* baseline_metrics,
                                                                     const PerformanceMetrics* optimized_metrics,
                                                                     float confidence_level,
                                                                     bool* is_significant,
                                                                     float* p_value);

/**
 * Destroy performance verifier and free resources
 * @param verifier Performance verifier instance to destroy
 */
void performance_verifier_destroy(PerformanceVerifier* verifier);

// Utility Functions

/**
 * Get error string for performance verifier error code
 * @param error_code PerformanceVerifierError code
 * @return Human-readable error message
 */
const char* performance_verifier_get_error_string(PerformanceVerifierError error_code);

/**
 * Get Apple Silicon model string
 * @param model Apple Silicon model enum
 * @return Human-readable model name
 */
const char* performance_verifier_get_model_string(AppleSiliconModel model);

/**
 * Get performance category string
 * @param category Performance category enum
 * @return Human-readable category name
 */
const char* performance_verifier_get_category_string(PerformanceCategory category);

/**
 * Calculate performance score from metrics
 * @param metrics Performance metrics
 * @param baseline_metrics Baseline metrics for comparison
 * @return Performance score (0-100)
 */
float performance_verifier_calculate_performance_score(const PerformanceMetrics* metrics,
                                                       const PerformanceMetrics* baseline_metrics);

/**
 * Calculate memory efficiency score
 * @param peak_memory_mb Peak memory usage in MB
 * @param available_memory_mb Available system memory in MB
 * @param cache_hit_rate Cache hit rate (0.0-1.0)
 * @return Memory efficiency score (0-100)
 */
float performance_verifier_calculate_memory_efficiency(uint64_t peak_memory_mb,
                                                       uint64_t available_memory_mb,
                                                       float cache_hit_rate);

/**
 * Estimate power efficiency from performance metrics
 * @param performance_metrics Performance metrics
 * @param power_consumption_watts Power consumption in watts
 * @return Power efficiency score (operations per watt)
 */
float performance_verifier_calculate_power_efficiency(const PerformanceMetrics* performance_metrics,
                                                      float power_consumption_watts);

// Constants for Apple Silicon performance verification

// Apple Silicon Model Specifications
#define M1_CPU_CORES 8
#define M1_PRO_CPU_CORES 10
#define M1_MAX_CPU_CORES 10
#define M1_ULTRA_CPU_CORES 20

#define M2_CPU_CORES 8
#define M2_PRO_CPU_CORES 12
#define M2_MAX_CPU_CORES 12
#define M2_ULTRA_CPU_CORES 24

#define M3_CPU_CORES 8
#define M3_PRO_CPU_CORES 12
#define M3_MAX_CPU_CORES 16

// Performance Thresholds
#define PERFORMANCE_SCORE_EXCELLENT 90.0f     // Excellent performance threshold
#define PERFORMANCE_SCORE_GOOD 75.0f          // Good performance threshold
#define PERFORMANCE_SCORE_ACCEPTABLE 60.0f    // Acceptable performance threshold
#define SPEEDUP_FACTOR_SIGNIFICANT 1.5f       // Significant speedup threshold
#define SPEED_CONSTRAINT_FACTOR_MAX 10.0f     // Maximum allowed speed factor
#define CACHE_HIT_RATE_EXCELLENT 0.95f        // Excellent cache hit rate
#define PARALLEL_EFFICIENCY_GOOD 0.8f         // Good parallel efficiency

// Measurement Configuration
#define DEFAULT_MEASUREMENT_ITERATIONS 10      // Default measurement iterations
#define DEFAULT_WARMUP_ITERATIONS 3           // Default warm-up iterations
#define DEFAULT_CONFIDENCE_LEVEL 0.95f        // Default statistical confidence
#define STRESS_TEST_MIN_DURATION_MINUTES 5    // Minimum stress test duration
#define PERFORMANCE_STABILITY_THRESHOLD 0.9f  // Performance stability threshold

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_VERIFIER_H
