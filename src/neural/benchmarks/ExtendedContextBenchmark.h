/*
 * ExtendedContextBenchmark.h
 * 
 * Extended Context Performance Measurement System
 * Comprehensive benchmarking for all context lengths with compression ratio analysis,
 * processing time evaluation, memory usage profiling, and quality-speed tradeoff analysis
 */

#ifndef EXTENDED_CONTEXT_BENCHMARK_H
#define EXTENDED_CONTEXT_BENCHMARK_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <sys/time.h>
#include "../context/AdaptiveContextManager.h"
#include "../context/DynamicContextSelector.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct ExtendedContextBenchmark ExtendedContextBenchmark;

// Context length test configurations
typedef enum {
    CONTEXT_TEST_BASIC = 0,            // 64 tokens - basic processing
    CONTEXT_TEST_MEDIUM,               // 256 tokens - medium range
    CONTEXT_TEST_STANDARD,             // 512 tokens - standard processing
    CONTEXT_TEST_LONG,                 // 1024 tokens - long context
    CONTEXT_TEST_MAXIMUM,              // 2048 tokens - maximum context
    CONTEXT_TEST_ALL                   // Test all context lengths
} ContextTestLevel;

// Benchmark test types
typedef enum {
    BENCHMARK_TYPE_COMPRESSION = 0,    // Compression ratio measurement
    BENCHMARK_TYPE_PERFORMANCE,        // Processing time and throughput
    BENCHMARK_TYPE_MEMORY,             // Memory usage analysis
    BENCHMARK_TYPE_QUALITY,            // Quality metrics evaluation
    BENCHMARK_TYPE_COMPREHENSIVE       // All benchmark types
} BenchmarkType;

// Test content categories
typedef enum {
    TEST_CONTENT_TEXT = 0,             // Natural language text
    TEST_CONTENT_CODE,                 // Source code files
    TEST_CONTENT_BINARY,               // Binary data
    TEST_CONTENT_STRUCTURED,           // Structured data (JSON, XML)
    TEST_CONTENT_MIXED,                // Mixed content types
    TEST_CONTENT_SYNTHETIC,            // Synthetic test patterns
    TEST_CONTENT_REAL_WORLD           // Real-world file samples
} TestContentCategory;

// Compression measurement results
typedef struct {
    uint32_t context_length;           // Context length used
    size_t original_size;              // Original data size (bytes)
    size_t compressed_size;            // Compressed data size (bytes)
    float compression_ratio;           // Compression ratio (0.0-1.0)
    float compression_percentage;      // Compression percentage
    size_t memory_overhead;            // Memory overhead during compression
    uint64_t compression_time_us;      // Compression time (microseconds)
    uint64_t decompression_time_us;    // Decompression time (microseconds)
    float bits_per_token;              // Average bits per token
    bool lossless_verified;            // Lossless compression verified
    float quality_score;               // Compression quality score
    uint32_t context_utilization;      // Context utilization percentage
} CompressionBenchmarkResult;

// Performance measurement results
typedef struct {
    uint32_t context_length;           // Context length used
    uint64_t processing_time_us;       // Total processing time (microseconds)
    uint64_t attention_time_us;        // Attention computation time
    uint64_t ffn_time_us;              // Feed-forward network time
    uint64_t memory_ops_time_us;       // Memory operations time
    float tokens_per_second;           // Processing throughput
    float gpu_utilization;             // GPU utilization percentage
    float cpu_utilization;             // CPU utilization percentage
    uint32_t memory_transfers;         // Number of GPU-CPU transfers
    size_t peak_bandwidth_mbps;        // Peak memory bandwidth usage
    float computational_efficiency;    // Computational efficiency score
    float parallelization_factor;      // Achieved parallelization factor
} PerformanceBenchmarkResult;

// Memory usage analysis results
typedef struct {
    uint32_t context_length;           // Context length used
    size_t peak_memory_usage_mb;       // Peak memory usage (MB)
    size_t base_memory_mb;             // Base memory requirement
    size_t context_memory_mb;          // Additional context memory
    size_t attention_memory_mb;        // Attention mechanism memory
    size_t cache_memory_mb;            // Cache memory usage
    size_t gpu_memory_mb;              // GPU memory usage
    size_t cpu_memory_mb;              // CPU memory usage
    float memory_efficiency;           // Memory usage efficiency
    uint32_t memory_allocations;       // Number of allocations
    uint32_t memory_deallocations;     // Number of deallocations
    float fragmentation_ratio;         // Memory fragmentation ratio
    bool memory_pressure_detected;     // Memory pressure detected
} MemoryBenchmarkResult;

// Quality metrics evaluation results
typedef struct {
    uint32_t context_length;           // Context length used
    float reconstruction_accuracy;      // Data reconstruction accuracy
    float pattern_preservation;        // Pattern preservation score
    float structural_integrity;        // Structural integrity score
    float semantic_coherence;          // Semantic coherence score
    float compression_effectiveness;    // Overall compression effectiveness
    uint32_t artifacts_detected;       // Number of compression artifacts
    float signal_to_noise_ratio;       // Signal-to-noise ratio
    bool quality_threshold_met;        // Quality threshold satisfied
    float user_perceptible_difference; // Perceptible difference score
    float compression_distortion;       // Compression-induced distortion
} QualityBenchmarkResult;

// Comprehensive benchmark results for single context length
typedef struct {
    uint32_t context_length;           // Context length tested
    TestContentCategory content_category; // Content category tested
    CompressionBenchmarkResult compression; // Compression results
    PerformanceBenchmarkResult performance; // Performance results
    MemoryBenchmarkResult memory;       // Memory results
    QualityBenchmarkResult quality;     // Quality results
    float overall_score;               // Overall benchmark score
    bool meets_phase2_criteria;        // Meets Phase 2 completion criteria
    struct timeval benchmark_start;     // Benchmark start time
    struct timeval benchmark_end;       // Benchmark end time
    uint64_t total_benchmark_time_us;   // Total benchmark duration
} ContextBenchmarkResult;

// Tradeoff analysis results
typedef struct {
    uint32_t context_length;           // Context length
    float speed_score;                 // Speed performance score (0-100)
    float quality_score;               // Quality score (0-100)
    float memory_score;                // Memory efficiency score (0-100)
    float compression_score;           // Compression effectiveness score (0-100)
    float balanced_score;              // Balanced overall score
    float speed_quality_tradeoff;      // Speed vs quality tradeoff ratio
    float memory_quality_tradeoff;     // Memory vs quality tradeoff ratio
    float compression_speed_tradeoff;   // Compression vs speed tradeoff ratio
    uint32_t optimal_use_cases;        // Recommended use case flags
    bool recommended_for_production;    // Recommended for production use
    char optimization_suggestions[256]; // Optimization recommendations
} TradeoffAnalysisResult;

// Extended context benchmark configuration
typedef struct {
    bool enable_compression_benchmarks; // Enable compression benchmarks
    bool enable_performance_benchmarks; // Enable performance benchmarks
    bool enable_memory_benchmarks;      // Enable memory benchmarks
    bool enable_quality_benchmarks;     // Enable quality benchmarks
    bool enable_tradeoff_analysis;      // Enable tradeoff analysis
    bool enable_real_time_monitoring;   // Enable real-time monitoring
    uint32_t test_iterations;           // Number of test iterations
    uint32_t warmup_iterations;         // Number of warmup iterations
    size_t test_data_size;              // Test data size (bytes)
    float target_compression_ratio;     // Target compression ratio
    float quality_threshold;            // Minimum quality threshold
    uint64_t max_benchmark_time_us;     // Maximum benchmark time
    bool save_detailed_logs;            // Save detailed benchmark logs
    bool generate_performance_reports;  // Generate performance reports
} ExtendedContextBenchmarkConfig;

// Benchmark statistics and summaries
typedef struct {
    uint32_t total_tests_run;           // Total number of tests run
    uint32_t successful_tests;          // Number of successful tests
    uint32_t failed_tests;              // Number of failed tests
    uint64_t total_benchmark_time_us;   // Total benchmarking time
    float average_compression_ratio;    // Average compression ratio achieved
    float best_compression_ratio;       // Best compression ratio achieved
    uint32_t optimal_context_length;    // Optimal context length found
    float phase2_completion_score;      // Phase 2 completion score
    bool phase2_criteria_met;           // Phase 2 criteria satisfied
    TestContentCategory best_content_type; // Best performing content type
    float overall_system_score;         // Overall system performance score
    char summary_report[1024];          // Text summary report
} ExtendedContextBenchmarkStats;

// Error codes for extended context benchmarking
typedef enum {
    EXTENDED_BENCHMARK_SUCCESS = 0,
    EXTENDED_BENCHMARK_ERROR_INVALID_PARAM,
    EXTENDED_BENCHMARK_ERROR_MEMORY_ALLOCATION,
    EXTENDED_BENCHMARK_ERROR_INITIALIZATION_FAILED,
    EXTENDED_BENCHMARK_ERROR_TEST_DATA_GENERATION_FAILED,
    EXTENDED_BENCHMARK_ERROR_COMPRESSION_FAILED,
    EXTENDED_BENCHMARK_ERROR_PERFORMANCE_MEASUREMENT_FAILED,
    EXTENDED_BENCHMARK_ERROR_MEMORY_MEASUREMENT_FAILED,
    EXTENDED_BENCHMARK_ERROR_QUALITY_MEASUREMENT_FAILED,
    EXTENDED_BENCHMARK_ERROR_BENCHMARK_TIMEOUT,
    EXTENDED_BENCHMARK_ERROR_INSUFFICIENT_RESOURCES,
    EXTENDED_BENCHMARK_ERROR_DATA_CORRUPTION,
    EXTENDED_BENCHMARK_ERROR_CRITERIA_NOT_MET
} ExtendedBenchmarkError;

// Core API Functions

/**
 * Create extended context benchmark system
 * @param benchmark Pointer to store created benchmark system
 * @param config Extended context benchmark configuration
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_create(ExtendedContextBenchmark** benchmark,
                                                 const ExtendedContextBenchmarkConfig* config);

/**
 * Initialize extended context benchmark with adaptive context manager
 * @param benchmark Extended context benchmark instance
 * @param context_manager Adaptive context manager
 * @param context_selector Dynamic context selector
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_initialize(ExtendedContextBenchmark* benchmark,
                                                    AdaptiveContextManager* context_manager,
                                                    DynamicContextSelector* context_selector);

/**
 * Run compression ratio measurement for specific context length
 * @param benchmark Extended context benchmark instance
 * @param context_length Context length to test
 * @param content_category Content category to test
 * @param compression_result Output compression benchmark result
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_measure_compression(ExtendedContextBenchmark* benchmark,
                                                            uint32_t context_length,
                                                            TestContentCategory content_category,
                                                            CompressionBenchmarkResult* compression_result);

/**
 * Run performance measurement for specific context length
 * @param benchmark Extended context benchmark instance
 * @param context_length Context length to test
 * @param content_category Content category to test
 * @param performance_result Output performance benchmark result
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_measure_performance(ExtendedContextBenchmark* benchmark,
                                                            uint32_t context_length,
                                                            TestContentCategory content_category,
                                                            PerformanceBenchmarkResult* performance_result);

/**
 * Run memory usage measurement for specific context length
 * @param benchmark Extended context benchmark instance
 * @param context_length Context length to test
 * @param content_category Content category to test
 * @param memory_result Output memory benchmark result
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_measure_memory(ExtendedContextBenchmark* benchmark,
                                                       uint32_t context_length,
                                                       TestContentCategory content_category,
                                                       MemoryBenchmarkResult* memory_result);

/**
 * Run quality metrics evaluation for specific context length
 * @param benchmark Extended context benchmark instance
 * @param context_length Context length to test
 * @param content_category Content category to test
 * @param quality_result Output quality benchmark result
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_measure_quality(ExtendedContextBenchmark* benchmark,
                                                         uint32_t context_length,
                                                         TestContentCategory content_category,
                                                         QualityBenchmarkResult* quality_result);

/**
 * Run comprehensive benchmark for specific context length
 * @param benchmark Extended context benchmark instance
 * @param context_length Context length to test
 * @param content_category Content category to test
 * @param benchmark_result Output comprehensive benchmark result
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_run_comprehensive(ExtendedContextBenchmark* benchmark,
                                                           uint32_t context_length,
                                                           TestContentCategory content_category,
                                                           ContextBenchmarkResult* benchmark_result);

/**
 * Run benchmarks for all context lengths
 * @param benchmark Extended context benchmark instance
 * @param content_category Content category to test
 * @param results Array to store results (must have space for 5 results)
 * @param num_results Output number of results stored
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_run_all_contexts(ExtendedContextBenchmark* benchmark,
                                                          TestContentCategory content_category,
                                                          ContextBenchmarkResult* results,
                                                          uint32_t* num_results);

/**
 * Perform quality-speed tradeoff analysis
 * @param benchmark Extended context benchmark instance
 * @param benchmark_results Array of benchmark results to analyze
 * @param num_results Number of benchmark results
 * @param tradeoff_results Output tradeoff analysis results
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_analyze_tradeoffs(ExtendedContextBenchmark* benchmark,
                                                           const ContextBenchmarkResult* benchmark_results,
                                                           uint32_t num_results,
                                                           TradeoffAnalysisResult* tradeoff_results);

/**
 * Check Phase 2 completion criteria (30-40% compression ratio target)
 * @param benchmark Extended context benchmark instance
 * @param benchmark_results Array of benchmark results to evaluate
 * @param num_results Number of benchmark results
 * @param phase2_met Output boolean for Phase 2 criteria satisfaction
 * @param completion_score Output Phase 2 completion score
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_check_phase2_completion(ExtendedContextBenchmark* benchmark,
                                                                 const ContextBenchmarkResult* benchmark_results,
                                                                 uint32_t num_results,
                                                                 bool* phase2_met,
                                                                 float* completion_score);

/**
 * Generate comprehensive benchmark report
 * @param benchmark Extended context benchmark instance
 * @param benchmark_results Array of benchmark results
 * @param num_results Number of benchmark results
 * @param report_buffer Buffer to store report text
 * @param buffer_size Size of report buffer
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_generate_report(ExtendedContextBenchmark* benchmark,
                                                         const ContextBenchmarkResult* benchmark_results,
                                                         uint32_t num_results,
                                                         char* report_buffer,
                                                         size_t buffer_size);

/**
 * Get extended context benchmark statistics
 * @param benchmark Extended context benchmark instance
 * @param stats Output benchmark statistics
 */
void extended_benchmark_get_stats(ExtendedContextBenchmark* benchmark,
                                 ExtendedContextBenchmarkStats* stats);

/**
 * Update extended context benchmark configuration
 * @param benchmark Extended context benchmark instance
 * @param config New benchmark configuration
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_update_config(ExtendedContextBenchmark* benchmark,
                                                       const ExtendedContextBenchmarkConfig* config);

/**
 * Destroy extended context benchmark and free resources
 * @param benchmark Extended context benchmark instance to destroy
 */
void extended_benchmark_destroy(ExtendedContextBenchmark* benchmark);

// Test Data Generation Functions

/**
 * Generate test data for specific content category
 * @param content_category Category of content to generate
 * @param data_size Size of data to generate
 * @param test_data Output buffer for generated test data
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_generate_test_data(TestContentCategory content_category,
                                                           size_t data_size,
                                                           uint8_t* test_data);

/**
 * Load real-world test data samples
 * @param content_category Category of content to load
 * @param sample_index Sample index to load
 * @param test_data Output buffer for loaded test data
 * @param data_size Input buffer size, output actual data size
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_load_real_world_data(TestContentCategory content_category,
                                                              uint32_t sample_index,
                                                              uint8_t* test_data,
                                                              size_t* data_size);

/**
 * Validate test data integrity
 * @param original_data Original test data
 * @param processed_data Processed/compressed and decompressed data
 * @param data_size Size of data to compare
 * @param integrity_verified Output boolean for integrity verification
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_validate_data_integrity(const uint8_t* original_data,
                                                                 const uint8_t* processed_data,
                                                                 size_t data_size,
                                                                 bool* integrity_verified);

// Analysis and Reporting Functions

/**
 * Calculate compression effectiveness score
 * @param compression_result Compression benchmark result
 * @param effectiveness_score Output effectiveness score (0-100)
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_calculate_compression_effectiveness(const CompressionBenchmarkResult* compression_result,
                                                                            float* effectiveness_score);

/**
 * Calculate performance efficiency score
 * @param performance_result Performance benchmark result
 * @param efficiency_score Output efficiency score (0-100)
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_calculate_performance_efficiency(const PerformanceBenchmarkResult* performance_result,
                                                                          float* efficiency_score);

/**
 * Calculate memory usage efficiency score
 * @param memory_result Memory benchmark result
 * @param context_length Context length used
 * @param efficiency_score Output efficiency score (0-100)
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_calculate_memory_efficiency(const MemoryBenchmarkResult* memory_result,
                                                                     uint32_t context_length,
                                                                     float* efficiency_score);

/**
 * Find optimal context length for given criteria
 * @param benchmark_results Array of benchmark results
 * @param num_results Number of benchmark results
 * @param optimization_criteria Optimization criteria (compression, speed, balance)
 * @param optimal_length Output optimal context length
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_find_optimal_context_length(const ContextBenchmarkResult* benchmark_results,
                                                                     uint32_t num_results,
                                                                     const char* optimization_criteria,
                                                                     uint32_t* optimal_length);

// Configuration Functions

/**
 * Create default extended context benchmark configuration
 * @param config Output default configuration
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_create_default_config(ExtendedContextBenchmarkConfig* config);

/**
 * Create phase 2 validation configuration
 * @param config Output Phase 2 validation configuration
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_create_phase2_config(ExtendedContextBenchmarkConfig* config);

/**
 * Create comprehensive benchmark configuration
 * @param config Output comprehensive benchmark configuration
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_create_comprehensive_config(ExtendedContextBenchmarkConfig* config);

/**
 * Validate extended context benchmark configuration
 * @param config Configuration to validate
 * @param is_valid Output boolean for validity
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_validate_config(const ExtendedContextBenchmarkConfig* config,
                                                         bool* is_valid);

// Utility Functions

/**
 * Get error string for extended benchmark error code
 * @param error_code ExtendedBenchmarkError code
 * @return Human-readable error message
 */
const char* extended_benchmark_get_error_string(ExtendedBenchmarkError error_code);

/**
 * Convert context test level to string
 * @param test_level Context test level
 * @return Human-readable test level name
 */
const char* extended_benchmark_context_test_level_to_string(ContextTestLevel test_level);

/**
 * Convert benchmark type to string
 * @param benchmark_type Benchmark type
 * @return Human-readable benchmark type name
 */
const char* extended_benchmark_type_to_string(BenchmarkType benchmark_type);

/**
 * Convert test content category to string
 * @param content_category Test content category
 * @return Human-readable content category name
 */
const char* extended_benchmark_content_category_to_string(TestContentCategory content_category);

/**
 * Get context length for test level
 * @param test_level Context test level
 * @return Context length in tokens
 */
uint32_t extended_benchmark_get_context_length_for_level(ContextTestLevel test_level);

/**
 * Estimate benchmark duration
 * @param config Benchmark configuration
 * @param content_category Content category to test
 * @param test_level Context test level
 * @param estimated_duration_us Output estimated duration (microseconds)
 * @return EXTENDED_BENCHMARK_SUCCESS on success, error code on failure
 */
ExtendedBenchmarkError extended_benchmark_estimate_duration(const ExtendedContextBenchmarkConfig* config,
                                                           TestContentCategory content_category,
                                                           ContextTestLevel test_level,
                                                           uint64_t* estimated_duration_us);

// Constants for extended context benchmarking
#define EXTENDED_BENCHMARK_CONTEXT_LENGTHS {64, 256, 512, 1024, 2048} // Standard context lengths
#define EXTENDED_BENCHMARK_NUM_CONTEXT_LENGTHS 5                      // Number of context lengths
#define EXTENDED_BENCHMARK_DEFAULT_ITERATIONS 10                      // Default test iterations
#define EXTENDED_BENCHMARK_DEFAULT_WARMUP 3                          // Default warmup iterations
#define EXTENDED_BENCHMARK_DEFAULT_DATA_SIZE 65536                   // Default test data size (64KB)
#define EXTENDED_BENCHMARK_MAX_BENCHMARK_TIME 3600000000            // Max benchmark time (1 hour)

// Phase 2 completion criteria
#define PHASE2_TARGET_COMPRESSION_MIN 0.30f                         // 30% minimum compression
#define PHASE2_TARGET_COMPRESSION_MAX 0.40f                         // 40% maximum compression  
#define PHASE2_QUALITY_THRESHOLD 0.95f                              // 95% quality threshold
#define PHASE2_PERFORMANCE_THRESHOLD 0.80f                          // 80% performance threshold
#define PHASE2_MEMORY_EFFICIENCY_THRESHOLD 0.75f                    // 75% memory efficiency

// Scoring and analysis constants
#define BENCHMARK_SCORE_WEIGHT_COMPRESSION 0.4f                     // Compression score weight
#define BENCHMARK_SCORE_WEIGHT_PERFORMANCE 0.3f                     // Performance score weight
#define BENCHMARK_SCORE_WEIGHT_MEMORY 0.2f                          // Memory score weight
#define BENCHMARK_SCORE_WEIGHT_QUALITY 0.1f                         // Quality score weight
#define BENCHMARK_TRADEOFF_BALANCE_THRESHOLD 0.8f                   // Tradeoff balance threshold

#ifdef __cplusplus
}
#endif

#endif // EXTENDED_CONTEXT_BENCHMARK_H
