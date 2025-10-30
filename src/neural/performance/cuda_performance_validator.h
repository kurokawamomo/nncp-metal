/*
 * CUDA Performance Validation and Optimization
 * 
 * Comprehensive performance validation tools to ensure Metal implementation
 * meets CUDA performance requirements while maintaining accuracy.
 */

#ifndef CUDA_PERFORMANCE_VALIDATOR_H
#define CUDA_PERFORMANCE_VALIDATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Performance requirement thresholds (based on original CUDA implementation)
typedef struct {
    // Throughput requirements (operations per second)
    double min_compression_throughput_mbps;    // Minimum compression MB/s
    double min_decompression_throughput_mbps;  // Minimum decompression MB/s
    double min_matrix_ops_gflops;              // Minimum GFLOPS for matrix ops
    
    // Latency requirements (milliseconds)
    double max_compression_latency_ms;         // Maximum latency for compression
    double max_lstm_inference_latency_ms;      // Maximum LSTM inference latency
    double max_memory_allocation_latency_ms;   // Maximum memory allocation latency
    
    // Memory requirements
    size_t max_memory_usage_bytes;             // Maximum memory footprint
    double max_memory_overhead_ratio;          // Maximum overhead vs input size
    
    // Accuracy requirements
    float max_precision_error;                 // Maximum numerical error vs CUDA
    float min_compression_ratio;               // Minimum compression ratio
    
    // Energy efficiency (for mobile/battery scenarios)
    double max_energy_per_operation_mj;        // Maximum millijoules per operation
} CUDAPerformanceRequirements;

// Performance measurement results
typedef struct {
    // Measured throughput
    double measured_compression_throughput_mbps;
    double measured_decompression_throughput_mbps;
    double measured_matrix_ops_gflops;
    
    // Measured latency
    double measured_compression_latency_ms;
    double measured_lstm_inference_latency_ms;
    double measured_memory_allocation_latency_ms;
    
    // Measured memory usage
    size_t measured_memory_usage_bytes;
    double measured_memory_overhead_ratio;
    
    // Measured accuracy
    float measured_precision_error;
    float measured_compression_ratio;
    
    // Measured energy
    double measured_energy_per_operation_mj;
    
    // Comparison with CUDA baseline
    double cuda_performance_ratio;             // Performance relative to CUDA
    bool meets_cuda_requirements;              // Whether all requirements are met
    
    // Timing breakdown
    double initialization_time_ms;
    double computation_time_ms;
    double memory_transfer_time_ms;
    double cleanup_time_ms;
} CUDAPerformanceResults;

// Performance validation status
typedef enum {
    CUDA_PERF_VALIDATION_PASS = 0,
    CUDA_PERF_VALIDATION_FAIL_THROUGHPUT,
    CUDA_PERF_VALIDATION_FAIL_LATENCY,
    CUDA_PERF_VALIDATION_FAIL_MEMORY,
    CUDA_PERF_VALIDATION_FAIL_ACCURACY,
    CUDA_PERF_VALIDATION_FAIL_ENERGY,
    CUDA_PERF_VALIDATION_ERROR
} CUDAPerformanceValidationStatus;

// Test scenarios for performance validation
typedef enum {
    CUDA_PERF_SCENARIO_SMALL_FILES = 0,        // < 1KB files
    CUDA_PERF_SCENARIO_MEDIUM_FILES,           // 1KB - 1MB files
    CUDA_PERF_SCENARIO_LARGE_FILES,            // > 1MB files
    CUDA_PERF_SCENARIO_ENWIK8,                 // Standard enwik8 benchmark
    CUDA_PERF_SCENARIO_ENWIK9,                 // Standard enwik9 benchmark
    CUDA_PERF_SCENARIO_LSTM_FOCUSED,           // LSTM-optimized workloads
    CUDA_PERF_SCENARIO_TRANSFORMER_FOCUSED,    // Transformer-optimized workloads
    CUDA_PERF_SCENARIO_MIXED_WORKLOAD,         // Mixed compression scenarios
    CUDA_PERF_SCENARIO_STRESS_TEST,            // High-load stress testing
    CUDA_PERF_SCENARIO_COUNT
} CUDAPerformanceScenario;

// Performance optimization recommendations
typedef struct {
    bool enable_metal_performance_shaders;     // Use MPS optimizations
    bool enable_unified_memory;                // Use unified memory architecture
    bool enable_async_operations;              // Enable asynchronous operations
    bool enable_memory_prefetching;            // Prefetch memory for better latency
    bool enable_batch_processing;              // Process multiple items in batches
    
    // Profile-specific optimizations
    const char* recommended_cuda_profile;      // Best CUDA profile for workload
    size_t recommended_batch_size;             // Optimal batch size
    size_t recommended_memory_budget_mb;       // Optimal memory allocation
    
    // Algorithm-specific optimizations
    bool prefer_lstm_over_transformer;        // Algorithm preference
    bool enable_quantization;                 // Use quantized models for speed
    bool enable_pruning;                      // Use pruned models
    
    // System-level optimizations
    int recommended_thread_count;             // Optimal thread configuration
    bool enable_gpu_scheduling;               // Enable GPU command scheduling
} CUDAPerformanceOptimizations;

// Performance validation functions
CUDAPerformanceRequirements* cuda_perf_get_default_requirements(void);
CUDAPerformanceRequirements* cuda_perf_get_requirements_for_scenario(CUDAPerformanceScenario scenario);

// Performance measurement
CUDAPerformanceResults* cuda_perf_measure_scenario(CUDAPerformanceScenario scenario,
                                                   const void* test_data,
                                                   size_t data_size);
CUDAPerformanceResults* cuda_perf_measure_compression(const void* input_data,
                                                     size_t input_size,
                                                     const char* cuda_profile);
CUDAPerformanceResults* cuda_perf_measure_lstm_inference(const float* input_sequence,
                                                        size_t sequence_length,
                                                        size_t hidden_size);

// Performance validation
CUDAPerformanceValidationStatus cuda_perf_validate_requirements(
    const CUDAPerformanceResults* results,
    const CUDAPerformanceRequirements* requirements);

bool cuda_perf_meets_cuda_compatibility(const CUDAPerformanceResults* results);
void cuda_perf_print_validation_report(const CUDAPerformanceResults* results,
                                       const CUDAPerformanceRequirements* requirements);

// Performance optimization
CUDAPerformanceOptimizations* cuda_perf_analyze_and_optimize(
    const CUDAPerformanceResults* results,
    CUDAPerformanceScenario scenario);
bool cuda_perf_apply_optimizations(const CUDAPerformanceOptimizations* optimizations);

// Benchmarking utilities
typedef struct {
    double min_time_ms;
    double max_time_ms;
    double avg_time_ms;
    double std_dev_ms;
    size_t num_iterations;
    bool is_stable;                           // Whether measurements are stable
} CUDABenchmarkStats;

CUDABenchmarkStats* cuda_perf_benchmark_operation(void (*operation)(void*), 
                                                  void* context,
                                                  size_t num_iterations);
void cuda_perf_benchmark_print_stats(const CUDABenchmarkStats* stats, const char* operation_name);

// Memory profiling
typedef struct {
    size_t peak_memory_usage;
    size_t current_memory_usage;
    size_t allocation_count;
    size_t deallocation_count;
    size_t fragmentation_ratio;
    double avg_allocation_size;
} CUDAMemoryProfile;

void cuda_perf_memory_profiling_start(void);
void cuda_perf_memory_profiling_stop(void);
CUDAMemoryProfile* cuda_perf_get_memory_profile(void);

// Resource cleanup
void cuda_perf_free_requirements(CUDAPerformanceRequirements* requirements);
void cuda_perf_free_results(CUDAPerformanceResults* results);
void cuda_perf_free_optimizations(CUDAPerformanceOptimizations* optimizations);
void cuda_perf_free_benchmark_stats(CUDABenchmarkStats* stats);
void cuda_perf_free_memory_profile(CUDAMemoryProfile* profile);

#ifdef __cplusplus
}
#endif

#endif // CUDA_PERFORMANCE_VALIDATOR_H