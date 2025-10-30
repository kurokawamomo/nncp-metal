/*
 * PerformanceBenchmark.h
 * 
 * Performance Benchmarking System for CUDA enwik8 compatibility verification
 * Authentic compression ratio measurement and memory profiling
 * No dummy implementations - real performance analysis
 */

#ifndef PERFORMANCE_BENCHMARK_H
#define PERFORMANCE_BENCHMARK_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct PerformanceBenchmark PerformanceBenchmark;

// Benchmark types
typedef enum {
    BENCHMARK_TYPE_COMPRESSION_RATIO = 0,   // Compression ratio measurement
    BENCHMARK_TYPE_MEMORY_USAGE,            // Memory usage profiling
    BENCHMARK_TYPE_PROCESSING_TIME,         // Processing time measurement
    BENCHMARK_TYPE_THROUGHPUT,              // Data throughput analysis
    BENCHMARK_TYPE_CUDA_COMPARISON,         // CUDA implementation comparison
    BENCHMARK_TYPE_FULL_SUITE              // Complete benchmark suite
} BenchmarkType;

// Compression measurement results
typedef struct {
    size_t original_size_bytes;             // Original file size
    size_t compressed_size_bytes;           // Compressed output size
    float compression_ratio_percent;        // Compression ratio as percentage
    float bits_per_byte;                   // Bits per byte metric
    bool is_lossless;                      // Lossless compression verification
    
    // CUDA enwik8 target comparison
    float cuda_target_ratio_percent;       // CUDA enwik8 target (14.9%)
    float ratio_vs_cuda_target;            // Current vs CUDA target ratio
    bool meets_cuda_target;                // Whether target is met
    
    // Performance metrics
    uint64_t compression_time_microseconds; // Time taken for compression
    uint64_t decompression_time_microseconds; // Time taken for decompression
    float compression_throughput_mbps;      // Compression throughput MB/s
    float decompression_throughput_mbps;    // Decompression throughput MB/s
    
} CompressionBenchmarkResults;

// Memory profiling results
typedef struct {
    size_t peak_memory_usage_bytes;         // Peak memory usage
    size_t current_memory_usage_bytes;      // Current memory usage
    size_t model_weights_memory_bytes;      // Memory for model weights
    size_t context_buffer_memory_bytes;     // Memory for context buffers
    size_t workspace_memory_bytes;          // Memory for workspace
    
    // Apple Silicon specific metrics
    size_t unified_memory_usage_bytes;      // Unified memory usage
    size_t metal_buffer_memory_bytes;       // Metal buffer memory
    float memory_efficiency_ratio;         // Memory efficiency (0.0-1.0)
    
    // CUDA comparison
    size_t cuda_equivalent_memory_bytes;    // Estimated CUDA memory usage
    float memory_vs_cuda_ratio;            // Memory usage vs CUDA ratio
    
} MemoryProfilingResults;

// Processing time analysis results  
typedef struct {
    // Phase-wise timing
    uint64_t attention_computation_microseconds;    // Attention computation time
    uint64_t ffn_computation_microseconds;          // FFN computation time
    uint64_t memory_allocation_microseconds;        // Memory allocation time
    uint64_t data_transfer_microseconds;            // CPU-GPU transfer time
    uint64_t total_processing_microseconds;         // Total processing time
    
    // Relative performance
    float attention_percentage;             // % of time in attention
    float ffn_percentage;                   // % of time in FFN
    float memory_percentage;                // % of time in memory ops
    float transfer_percentage;              // % of time in transfers
    
    // CUDA comparison
    uint64_t cuda_equivalent_microseconds;  // Estimated CUDA processing time
    float performance_vs_cuda_ratio;       // Performance vs CUDA ratio
    bool meets_speed_constraint;           // <10x slower than CUDA
    
} ProcessingTimeBenchmarkResults;

// Complete benchmark results
typedef struct {
    CompressionBenchmarkResults compression;
    MemoryProfilingResults memory;
    ProcessingTimeBenchmarkResults timing;
    
    // Overall assessment
    float overall_score;                    // Combined performance score (0.0-1.0)
    bool passes_phase1_criteria;           // Phase 1 completion (60% compression)
    bool meets_cuda_enwik8_compatibility;  // Full CUDA enwik8 compatibility
    
    // Test configuration
    char test_file_path[256];               // Path to test file
    size_t test_file_size_bytes;            // Size of test file
    uint32_t context_length_used;          // Context length used in test
    uint32_t batch_size_used;              // Batch size used in test
    
} FullBenchmarkResults;

// Error codes
typedef enum {
    BENCHMARK_SUCCESS = 0,
    BENCHMARK_ERROR_INVALID_PARAM,
    BENCHMARK_ERROR_FILE_NOT_FOUND,
    BENCHMARK_ERROR_MEMORY_ALLOCATION,
    BENCHMARK_ERROR_DEVICE_NOT_FOUND,
    BENCHMARK_ERROR_COMPRESSION_FAILED,
    BENCHMARK_ERROR_DECOMPRESSION_FAILED,
    BENCHMARK_ERROR_VERIFICATION_FAILED,
    BENCHMARK_ERROR_BENCHMARK_TIMEOUT,
    BENCHMARK_ERROR_CUDA_PROFILE_NOT_FOUND
} BenchmarkError;

// Core API Functions

/**
 * Create performance benchmark context with CUDA enwik8 target configuration
 * @param benchmark Pointer to store created benchmark context
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_create(PerformanceBenchmark** benchmark);

/**
 * Run compression ratio benchmark on baseline_test.txt
 * Tests against CUDA enwik8 target of 14.9% compression ratio
 * @param benchmark Benchmark context
 * @param test_file_path Path to test file (e.g., "baseline_test.txt")
 * @param results Output compression benchmark results
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_run_compression_test(PerformanceBenchmark* benchmark,
                                              const char* test_file_path,
                                              CompressionBenchmarkResults* results);

/**
 * Run memory usage profiling during compression
 * Measures memory consumption across all zones and compares to CUDA baseline
 * @param benchmark Benchmark context
 * @param test_file_path Path to test file
 * @param results Output memory profiling results
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_run_memory_profiling(PerformanceBenchmark* benchmark,
                                              const char* test_file_path,
                                              MemoryProfilingResults* results);

/**
 * Run processing time measurement
 * Measures performance vs CUDA enwik8 baseline with <10x constraint
 * @param benchmark Benchmark context
 * @param test_file_path Path to test file
 * @param results Output timing benchmark results
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_run_timing_analysis(PerformanceBenchmark* benchmark,
                                             const char* test_file_path,
                                             ProcessingTimeBenchmarkResults* results);

/**
 * Run complete benchmark suite
 * Comprehensive testing for Phase 1 completion criteria (60% compression target)
 * @param benchmark Benchmark context
 * @param test_file_path Path to test file
 * @param results Output complete benchmark results
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_run_full_suite(PerformanceBenchmark* benchmark,
                                        const char* test_file_path,
                                        FullBenchmarkResults* results);

/**
 * Verify lossless compression integrity
 * Ensures 100% data integrity (decompress and compare with original)
 * @param benchmark Benchmark context
 * @param original_data Original data buffer
 * @param original_size Size of original data
 * @param compressed_data Compressed data buffer
 * @param compressed_size Size of compressed data
 * @param is_lossless Output boolean for lossless verification
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_verify_lossless_integrity(PerformanceBenchmark* benchmark,
                                                   const void* original_data,
                                                   size_t original_size,
                                                   const void* compressed_data,
                                                   size_t compressed_size,
                                                   bool* is_lossless);

/**
 * Compare performance with CUDA enwik8 baseline
 * @param benchmark Benchmark context
 * @param results Current benchmark results
 * @param cuda_baseline_file Path to CUDA baseline results (optional)
 * @param comparison_ratio Output comparison ratio (1.0 = same performance)
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_compare_with_cuda_baseline(PerformanceBenchmark* benchmark,
                                                    const FullBenchmarkResults* results,
                                                    const char* cuda_baseline_file,
                                                    float* comparison_ratio);

/**
 * Generate benchmark report
 * @param results Benchmark results to report
 * @param output_file Path to output report file (NULL for stdout)
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_generate_report(const FullBenchmarkResults* results,
                                         const char* output_file);

/**
 * Destroy benchmark context and free resources
 * @param benchmark Benchmark context to destroy
 */
void benchmark_destroy(PerformanceBenchmark* benchmark);

// Utility Functions

/**
 * Get error string for error code
 * @param error_code BenchmarkError code
 * @return Human-readable error message
 */
const char* benchmark_get_error_string(BenchmarkError error_code);

/**
 * Calculate compression ratio from sizes
 * @param original_size Original data size in bytes
 * @param compressed_size Compressed data size in bytes
 * @return Compression ratio as percentage (100.0 = no compression, 0.0 = perfect compression)
 */
float benchmark_calculate_compression_ratio(size_t original_size, size_t compressed_size);

/**
 * Calculate bits per byte metric
 * @param original_size Original data size in bytes
 * @param compressed_size Compressed data size in bytes
 * @return Bits per byte (8.0 = no compression, 0.0 = perfect compression)
 */
float benchmark_calculate_bits_per_byte(size_t original_size, size_t compressed_size);

/**
 * Load CUDA enwik8 target specifications
 * @param target_compression_ratio Output target compression ratio
 * @param target_memory_mb Output target memory usage in MB
 * @param max_processing_time_factor Output max processing time factor vs CUDA
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_get_cuda_enwik8_targets(float* target_compression_ratio,
                                                 size_t* target_memory_mb,
                                                 float* max_processing_time_factor);

/**
 * Check if results meet Phase 1 completion criteria
 * @param results Benchmark results to check
 * @param meets_criteria Output boolean for criteria check
 * @return BENCHMARK_SUCCESS on success, error code on failure
 */
BenchmarkError benchmark_check_phase1_criteria(const FullBenchmarkResults* results,
                                               bool* meets_criteria);

// CUDA enwik8 target constants
#define CUDA_ENWIK8_TARGET_COMPRESSION_PERCENT 14.9f  // CUDA enwik8 target: 14.9%
#define PHASE1_TARGET_COMPRESSION_PERCENT 60.0f       // Phase 1 target: 60%
#define MAX_PROCESSING_TIME_FACTOR 10.0f              // Max 10x slower than CUDA
#define CUDA_ENWIK8_BASELINE_MEMORY_MB 4096           // CUDA enwik8 memory baseline

// Benchmark configuration constants
#define BENCHMARK_TIMEOUT_SECONDS 300                  // 5 minute timeout
#define BENCHMARK_MIN_FILE_SIZE_BYTES 1024            // 1KB minimum file size
#define BENCHMARK_MAX_FILE_SIZE_BYTES (100*1024*1024) // 100MB maximum file size

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_BENCHMARK_H
