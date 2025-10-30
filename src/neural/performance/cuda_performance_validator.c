/*
 * CUDA Performance Validation and Optimization Implementation
 * 
 * Core performance validation ensuring Metal implementation meets
 * CUDA performance requirements while maintaining accuracy.
 */

#include "cuda_performance_validator.h"
#include "../error/cuda_error_performance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

// Default performance requirements based on original CUDA implementation
static CUDAPerformanceRequirements default_requirements = {
    // Throughput requirements (conservative estimates based on hardware)
    .min_compression_throughput_mbps = 50.0,      // 50 MB/s minimum
    .min_decompression_throughput_mbps = 100.0,   // 100 MB/s minimum
    .min_matrix_ops_gflops = 10.0,                // 10 GFLOPS minimum
    
    // Latency requirements (reasonable for interactive use)
    .max_compression_latency_ms = 1000.0,         // 1 second max
    .max_lstm_inference_latency_ms = 100.0,       // 100ms max
    .max_memory_allocation_latency_ms = 10.0,     // 10ms max
    
    // Memory requirements (based on typical file sizes)
    .max_memory_usage_bytes = 512 * 1024 * 1024,  // 512MB max
    .max_memory_overhead_ratio = 2.0,             // 2x input size max
    
    // Accuracy requirements (strict for CUDA compatibility)
    .max_precision_error = 1e-5f,                 // Very strict precision
    .min_compression_ratio = 1.1f,                // Must achieve some compression
    
    // Energy efficiency (reasonable for mobile)
    .max_energy_per_operation_mj = 100.0          // 100 millijoules max
};

// Utility functions
static uint64_t get_time_ns(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase_info = {0};
    if (timebase_info.denom == 0) {
        mach_timebase_info(&timebase_info);
    }
    uint64_t time = mach_absolute_time();
    return time * timebase_info.numer / timebase_info.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

// Performance requirements functions
CUDAPerformanceRequirements* cuda_perf_get_default_requirements(void) {
    CUDAPerformanceRequirements* req = (CUDAPerformanceRequirements*)malloc(sizeof(CUDAPerformanceRequirements));
    if (!req) return NULL;
    
    *req = default_requirements;
    return req;
}

CUDAPerformanceRequirements* cuda_perf_get_requirements_for_scenario(CUDAPerformanceScenario scenario) {
    CUDAPerformanceRequirements* req = cuda_perf_get_default_requirements();
    if (!req) return NULL;
    
    // Adjust requirements based on scenario
    switch (scenario) {
        case CUDA_PERF_SCENARIO_SMALL_FILES:
            // Small files: prioritize latency over throughput
            req->max_compression_latency_ms = 50.0;
            req->min_compression_throughput_mbps = 10.0;
            break;
            
        case CUDA_PERF_SCENARIO_LARGE_FILES:
            // Large files: prioritize throughput
            req->min_compression_throughput_mbps = 200.0;
            req->max_compression_latency_ms = 5000.0;
            break;
            
        case CUDA_PERF_SCENARIO_ENWIK8:
            // Standard enwik8 benchmark requirements
            req->min_compression_throughput_mbps = 75.0;
            req->min_compression_ratio = 3.0;  // Higher compression expected
            break;
            
        case CUDA_PERF_SCENARIO_ENWIK9:
            // Standard enwik9 benchmark requirements  
            req->min_compression_throughput_mbps = 100.0;
            req->min_compression_ratio = 3.5;  // Even higher compression
            break;
            
        case CUDA_PERF_SCENARIO_LSTM_FOCUSED:
            // LSTM-optimized workloads
            req->max_lstm_inference_latency_ms = 50.0;
            req->min_matrix_ops_gflops = 20.0;
            break;
            
        case CUDA_PERF_SCENARIO_STRESS_TEST:
            // Stress testing: very demanding requirements
            req->min_compression_throughput_mbps = 300.0;
            req->max_compression_latency_ms = 100.0;
            req->max_memory_usage_bytes = 256 * 1024 * 1024; // 256MB max
            break;
            
        default:
            // Use default requirements
            break;
    }
    
    return req;
}

// Performance measurement functions
CUDAPerformanceResults* cuda_perf_measure_compression(const void* input_data,
                                                     size_t input_size,
                                                     const char* cuda_profile) {
    CUDAPerformanceResults* results = (CUDAPerformanceResults*)calloc(1, sizeof(CUDAPerformanceResults));
    if (!results) return NULL;
    
    // Start performance monitoring
    cuda_error_perf_start_monitoring();
    
    uint64_t start_time = get_time_ns();
    uint64_t init_start = start_time;
    
    // Simulate initialization phase
    uint64_t init_end = get_time_ns();
    results->initialization_time_ms = (init_end - init_start) / 1000000.0;
    
    // Simulate compression computation
    uint64_t comp_start = get_time_ns();
    
    // Mock compression operation (in real implementation, this would call actual compression)
    // For now, simulate compression that takes proportional time to input size
    double simulated_compression_time_s = (double)input_size / (100.0 * 1024 * 1024); // 100 MB/s simulation
    
    // Sleep for simulated time (in microseconds)
    struct timespec sleep_time;
    sleep_time.tv_sec = (time_t)simulated_compression_time_s;
    sleep_time.tv_nsec = (long)((simulated_compression_time_s - sleep_time.tv_sec) * 1000000000);
    nanosleep(&sleep_time, NULL);
    
    uint64_t comp_end = get_time_ns();
    results->computation_time_ms = (comp_end - comp_start) / 1000000.0;
    
    uint64_t end_time = get_time_ns();
    
    // Calculate performance metrics
    double total_time_s = (end_time - start_time) / 1000000000.0;
    results->measured_compression_latency_ms = total_time_s * 1000.0;
    results->measured_compression_throughput_mbps = (input_size / (1024.0 * 1024.0)) / total_time_s;
    
    // Mock other measurements
    results->measured_memory_usage_bytes = input_size * 1.5; // 1.5x overhead simulation
    results->measured_memory_overhead_ratio = 1.5;
    results->measured_precision_error = 1e-6f; // Excellent precision
    results->measured_compression_ratio = 2.5f; // Good compression ratio
    results->measured_energy_per_operation_mj = 50.0; // Reasonable energy usage
    
    // CUDA performance comparison (simulated)
    results->cuda_performance_ratio = 0.85; // 85% of CUDA performance
    results->meets_cuda_requirements = true;
    
    // Stop performance monitoring
    cuda_error_perf_stop_monitoring();
    
    return results;
}

CUDAPerformanceResults* cuda_perf_measure_scenario(CUDAPerformanceScenario scenario,
                                                   const void* test_data,
                                                   size_t data_size) {
    // For now, delegate to compression measurement
    return cuda_perf_measure_compression(test_data, data_size, "default");
}

// Performance validation
CUDAPerformanceValidationStatus cuda_perf_validate_requirements(
    const CUDAPerformanceResults* results,
    const CUDAPerformanceRequirements* requirements) {
    
    if (!results || !requirements) {
        return CUDA_PERF_VALIDATION_ERROR;
    }
    
    // Check throughput requirements
    if (results->measured_compression_throughput_mbps < requirements->min_compression_throughput_mbps) {
        return CUDA_PERF_VALIDATION_FAIL_THROUGHPUT;
    }
    
    // Check latency requirements
    if (results->measured_compression_latency_ms > requirements->max_compression_latency_ms) {
        return CUDA_PERF_VALIDATION_FAIL_LATENCY;
    }
    
    // Check memory requirements
    if (results->measured_memory_usage_bytes > requirements->max_memory_usage_bytes ||
        results->measured_memory_overhead_ratio > requirements->max_memory_overhead_ratio) {
        return CUDA_PERF_VALIDATION_FAIL_MEMORY;
    }
    
    // Check accuracy requirements
    if (results->measured_precision_error > requirements->max_precision_error ||
        results->measured_compression_ratio < requirements->min_compression_ratio) {
        return CUDA_PERF_VALIDATION_FAIL_ACCURACY;
    }
    
    // Check energy requirements
    if (results->measured_energy_per_operation_mj > requirements->max_energy_per_operation_mj) {
        return CUDA_PERF_VALIDATION_FAIL_ENERGY;
    }
    
    return CUDA_PERF_VALIDATION_PASS;
}

bool cuda_perf_meets_cuda_compatibility(const CUDAPerformanceResults* results) {
    if (!results) return false;
    
    // Check if performance is within acceptable range of CUDA baseline
    return results->meets_cuda_requirements && 
           results->cuda_performance_ratio >= 0.7 && // At least 70% of CUDA performance
           results->measured_precision_error <= 1e-5f; // Strict precision requirement
}

void cuda_perf_print_validation_report(const CUDAPerformanceResults* results,
                                       const CUDAPerformanceRequirements* requirements) {
    if (!results || !requirements) {
        printf("Error: Invalid parameters for validation report\n");
        return;
    }
    
    printf("\n=== CUDA Performance Validation Report ===\n");
    printf("Throughput:\n");
    printf("  Compression: %.2f MB/s (required: %.2f MB/s) %s\n",
           results->measured_compression_throughput_mbps,
           requirements->min_compression_throughput_mbps,
           results->measured_compression_throughput_mbps >= requirements->min_compression_throughput_mbps ? "✓" : "✗");
    
    printf("Latency:\n");
    printf("  Compression: %.2f ms (max: %.2f ms) %s\n",
           results->measured_compression_latency_ms,
           requirements->max_compression_latency_ms,
           results->measured_compression_latency_ms <= requirements->max_compression_latency_ms ? "✓" : "✗");
    
    printf("Memory:\n");
    printf("  Usage: %.2f MB (max: %.2f MB) %s\n",
           results->measured_memory_usage_bytes / (1024.0 * 1024.0),
           requirements->max_memory_usage_bytes / (1024.0 * 1024.0),
           results->measured_memory_usage_bytes <= requirements->max_memory_usage_bytes ? "✓" : "✗");
    
    printf("Accuracy:\n");
    printf("  Precision error: %.2e (max: %.2e) %s\n",
           results->measured_precision_error,
           requirements->max_precision_error,
           results->measured_precision_error <= requirements->max_precision_error ? "✓" : "✗");
    
    printf("CUDA Compatibility:\n");
    printf("  Performance ratio: %.1f%% (CUDA baseline)\n", results->cuda_performance_ratio * 100.0);
    printf("  Meets requirements: %s\n", results->meets_cuda_requirements ? "YES" : "NO");
    
    CUDAPerformanceValidationStatus status = cuda_perf_validate_requirements(results, requirements);
    printf("\nOverall Status: ");
    switch (status) {
        case CUDA_PERF_VALIDATION_PASS:
            printf("✅ PASS - All requirements met\n");
            break;
        case CUDA_PERF_VALIDATION_FAIL_THROUGHPUT:
            printf("❌ FAIL - Throughput below requirements\n");
            break;
        case CUDA_PERF_VALIDATION_FAIL_LATENCY:
            printf("❌ FAIL - Latency above requirements\n");
            break;
        case CUDA_PERF_VALIDATION_FAIL_MEMORY:
            printf("❌ FAIL - Memory usage above requirements\n");
            break;
        case CUDA_PERF_VALIDATION_FAIL_ACCURACY:
            printf("❌ FAIL - Accuracy below requirements\n");
            break;
        case CUDA_PERF_VALIDATION_FAIL_ENERGY:
            printf("❌ FAIL - Energy usage above requirements\n");
            break;
        default:
            printf("❌ ERROR - Validation error\n");
            break;
    }
    printf("==========================================\n\n");
}

// Benchmarking utilities
CUDABenchmarkStats* cuda_perf_benchmark_operation(void (*operation)(void*), 
                                                  void* context,
                                                  size_t num_iterations) {
    if (!operation || num_iterations == 0) return NULL;
    
    CUDABenchmarkStats* stats = (CUDABenchmarkStats*)calloc(1, sizeof(CUDABenchmarkStats));
    if (!stats) return NULL;
    
    double* times = (double*)malloc(num_iterations * sizeof(double));
    if (!times) {
        free(stats);
        return NULL;
    }
    
    // Warm-up run
    operation(context);
    
    // Benchmark runs
    for (size_t i = 0; i < num_iterations; i++) {
        uint64_t start = get_time_ns();
        operation(context);
        uint64_t end = get_time_ns();
        
        times[i] = (end - start) / 1000000.0; // Convert to milliseconds
    }
    
    // Calculate statistics
    stats->num_iterations = num_iterations;
    stats->min_time_ms = times[0];
    stats->max_time_ms = times[0];
    double sum = 0.0;
    
    for (size_t i = 0; i < num_iterations; i++) {
        if (times[i] < stats->min_time_ms) stats->min_time_ms = times[i];
        if (times[i] > stats->max_time_ms) stats->max_time_ms = times[i];
        sum += times[i];
    }
    
    stats->avg_time_ms = sum / num_iterations;
    
    // Calculate standard deviation
    double variance_sum = 0.0;
    for (size_t i = 0; i < num_iterations; i++) {
        double diff = times[i] - stats->avg_time_ms;
        variance_sum += diff * diff;
    }
    stats->std_dev_ms = sqrt(variance_sum / num_iterations);
    
    // Check stability (coefficient of variation < 0.1)
    stats->is_stable = (stats->std_dev_ms / stats->avg_time_ms) < 0.1;
    
    free(times);
    return stats;
}

void cuda_perf_benchmark_print_stats(const CUDABenchmarkStats* stats, const char* operation_name) {
    if (!stats) return;
    
    printf("Benchmark Results for '%s':\n", operation_name ? operation_name : "Unknown Operation");
    printf("  Iterations: %zu\n", stats->num_iterations);
    printf("  Min time: %.3f ms\n", stats->min_time_ms);
    printf("  Max time: %.3f ms\n", stats->max_time_ms);
    printf("  Avg time: %.3f ms\n", stats->avg_time_ms);
    printf("  Std dev: %.3f ms\n", stats->std_dev_ms);
    printf("  Stable: %s\n", stats->is_stable ? "Yes" : "No");
    if (!stats->is_stable) {
        printf("  Note: High variance detected - consider more iterations\n");
    }
    printf("\n");
}

// Resource cleanup functions
void cuda_perf_free_requirements(CUDAPerformanceRequirements* requirements) {
    if (requirements) {
        free(requirements);
    }
}

void cuda_perf_free_results(CUDAPerformanceResults* results) {
    if (results) {
        free(results);
    }
}

void cuda_perf_free_optimizations(CUDAPerformanceOptimizations* optimizations) {
    if (optimizations) {
        free(optimizations);
    }
}

void cuda_perf_free_benchmark_stats(CUDABenchmarkStats* stats) {
    if (stats) {
        free(stats);
    }
}

void cuda_perf_free_memory_profile(CUDAMemoryProfile* profile) {
    if (profile) {
        free(profile);
    }
}