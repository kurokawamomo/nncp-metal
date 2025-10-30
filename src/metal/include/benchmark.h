#ifndef NNCP_BENCHMARK_H
#define NNCP_BENCHMARK_H

#include "metal_context.h"
#include "compute_kernels.h"
#include "memory_manager.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Benchmark operation types
typedef enum {
    BENCH_MATRIX_MULTIPLY = 0,
    BENCH_CONVOLUTION = 1,
    BENCH_ACTIVATION = 2,
    BENCH_ATTENTION = 3,
    BENCH_MEMORY_COPY = 4,
    BENCH_LAYER_NORM = 5,
    BENCH_SOFTMAX = 6,
    BENCH_OP_COUNT
} BenchmarkOperation;

// Benchmark test sizes
typedef enum {
    BENCH_SIZE_SMALL = 0,    // 32x32, 64x64
    BENCH_SIZE_MEDIUM = 1,   // 128x128, 256x256
    BENCH_SIZE_LARGE = 2,    // 512x512, 1024x1024
    BENCH_SIZE_XLARGE = 3,   // 2048x2048, 4096x4096
    BENCH_SIZE_COUNT
} BenchmarkSize;

// Performance metrics structure
typedef struct {
    double execution_time_ms;        // Average execution time
    double min_time_ms;             // Minimum execution time
    double max_time_ms;             // Maximum execution time
    double std_deviation_ms;        // Standard deviation
    double memory_bandwidth_gbps;   // Memory bandwidth utilization
    double gflops;                  // Giga floating-point operations per second
    double efficiency_percent;      // Hardware efficiency percentage
    uint32_t iterations;            // Number of iterations run
    size_t memory_used_bytes;       // Peak memory usage
    double power_estimate_watts;    // Estimated power consumption
} BenchmarkResult;

// Benchmark configuration
typedef struct {
    uint32_t warmup_iterations;     // Warm-up runs before timing
    uint32_t measurement_iterations; // Actual measured iterations
    bool measure_power;             // Enable power measurement
    bool measure_memory;            // Enable memory usage tracking
    bool verbose_output;            // Detailed output
    double timeout_seconds;         // Maximum time per test
} BenchmarkConfig;

// Comprehensive benchmark suite
typedef struct {
    BenchmarkResult results[BENCH_OP_COUNT][BENCH_SIZE_COUNT];
    double total_score;             // Overall performance score
    char device_name[256];          // Metal device name
    uint64_t total_memory_gb;       // Total device memory
    bool neural_engine_available;   // Neural Engine availability
    char timestamp[64];             // Benchmark timestamp
} BenchmarkSuite;

// Benchmark context
typedef struct BenchmarkContext BenchmarkContext;

// Core benchmark functions
MetalError benchmark_context_create(BenchmarkContext** context, 
                                   MetalContext* metal_ctx, 
                                   MMManager* memory_manager,
                                   ComputeKernelContext* kernel_ctx);

void benchmark_context_destroy(BenchmarkContext* context);

// Configuration
MetalError benchmark_set_config(BenchmarkContext* context, const BenchmarkConfig* config);
MetalError benchmark_get_config(BenchmarkContext* context, BenchmarkConfig* config);

// Individual operation benchmarks
MetalError benchmark_matrix_multiply(BenchmarkContext* context, 
                                   BenchmarkSize size, 
                                   BenchmarkResult* result);

MetalError benchmark_activation_functions(BenchmarkContext* context, 
                                        BenchmarkSize size, 
                                        BenchmarkResult* result);

MetalError benchmark_memory_operations(BenchmarkContext* context, 
                                     BenchmarkSize size, 
                                     BenchmarkResult* result);

MetalError benchmark_convolution(BenchmarkContext* context, 
                               BenchmarkSize size, 
                               BenchmarkResult* result);

MetalError benchmark_attention(BenchmarkContext* context, 
                             BenchmarkSize size, 
                             BenchmarkResult* result);

// Comprehensive benchmarking
MetalError benchmark_run_full_suite(BenchmarkContext* context, BenchmarkSuite* suite);

MetalError benchmark_run_operation(BenchmarkContext* context, 
                                  BenchmarkOperation operation, 
                                  BenchmarkSize size, 
                                  BenchmarkResult* result);

// Comparison and analysis
MetalError benchmark_compare_backends(BenchmarkContext* context, 
                                    BenchmarkSuite* metal_results,
                                    BenchmarkSuite* cpu_results);

// Reporting and export
MetalError benchmark_print_results(const BenchmarkSuite* suite, bool detailed);
MetalError benchmark_export_json(const BenchmarkSuite* suite, const char* filename);
MetalError benchmark_export_csv(const BenchmarkSuite* suite, const char* filename);

// Performance profiling
typedef struct {
    uint64_t gpu_time_ns;          // GPU execution time
    uint64_t cpu_time_ns;          // CPU preparation time
    size_t bytes_read;             // Bytes read from memory
    size_t bytes_written;          // Bytes written to memory
    uint32_t draw_calls;           // Number of Metal draw calls
    uint32_t compute_dispatches;   // Number of compute dispatches
    double gpu_utilization;        // GPU utilization percentage
    double memory_utilization;     // Memory bandwidth utilization
} ProfileResult;

MetalError benchmark_profile_operation(BenchmarkContext* context, 
                                     BenchmarkOperation operation,
                                     BenchmarkSize size,
                                     ProfileResult* profile);

// Optimization suggestions
typedef struct {
    char suggestion[512];
    double potential_improvement;   // Estimated improvement percentage
    int priority;                  // 1=high, 2=medium, 3=low
} OptimizationSuggestion;

MetalError benchmark_analyze_performance(const BenchmarkSuite* suite, 
                                       OptimizationSuggestion* suggestions, 
                                       uint32_t max_suggestions,
                                       uint32_t* num_suggestions);

// Utility functions
const char* benchmark_operation_name(BenchmarkOperation op);
const char* benchmark_size_name(BenchmarkSize size);
uint32_t benchmark_get_matrix_dimension(BenchmarkSize size);
double benchmark_calculate_theoretical_gflops(BenchmarkOperation op, BenchmarkSize size);

#ifdef __cplusplus
}
#endif

#endif /* NNCP_BENCHMARK_H */