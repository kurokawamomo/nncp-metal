#import "benchmark.h"
#import "metal_context.h"
#import "neural_engine.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <sys/sysctl.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

// Define missing error constant
#ifndef METAL_ERROR_EXECUTION_FAILED
#define METAL_ERROR_EXECUTION_FAILED ((MetalError)3)
#endif

// Internal benchmark context structure
struct BenchmarkContext {
    MetalContext* metal_ctx;
    MMManager* memory_manager;
    ComputeKernelContext* kernel_ctx;
    BenchmarkConfig config;
    
    // Timing infrastructure
    mach_timebase_info_data_t timebase_info;
    uint64_t start_time;
    uint64_t end_time;
    
    // Memory tracking
    size_t peak_memory_usage;
    size_t current_memory_usage;
    
    // Statistics collection
    double* execution_times;
    uint32_t max_iterations;
};

// Helper functions
static double nanoseconds_to_milliseconds(uint64_t nanoseconds) {
    return nanoseconds / 1000000.0;
}

static uint64_t get_absolute_time_ns(BenchmarkContext* context) {
    uint64_t absolute_time = mach_absolute_time();
    return absolute_time * context->timebase_info.numer / context->timebase_info.denom;
}

static void get_device_info(BenchmarkContext* context, BenchmarkSuite* suite) {
    @autoreleasepool {
        // Access device through Metal context properly  
        id<MTLDevice> device = context->metal_ctx->device;
        
        // Get device name
        NSString* deviceName = device.name;
        strncpy(suite->device_name, [deviceName UTF8String], sizeof(suite->device_name) - 1);
        suite->device_name[sizeof(suite->device_name) - 1] = '\0';
        
        // Get memory info
        if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
            suite->total_memory_gb = device.recommendedMaxWorkingSetSize / (1024 * 1024 * 1024);
        } else {
            // Fallback estimation for older devices
            suite->total_memory_gb = 8; // Common Apple Silicon memory
        }
        
        // Get timestamp
        time_t rawtime;
        struct tm* timeinfo;
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(suite->timestamp, sizeof(suite->timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);
    }
}

// Core benchmark functions
MetalError benchmark_context_create(BenchmarkContext** context, 
                                   MetalContext* metal_ctx, 
                                   MMManager* memory_manager,
                                   ComputeKernelContext* kernel_ctx) {
    if (!context || !metal_ctx || !memory_manager || !kernel_ctx) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    BenchmarkContext* ctx = (BenchmarkContext*)calloc(1, sizeof(BenchmarkContext));
    if (!ctx) {
        return METAL_ERROR_OUT_OF_MEMORY;
    }
    
    ctx->metal_ctx = metal_ctx;
    ctx->memory_manager = memory_manager;
    ctx->kernel_ctx = kernel_ctx;
    
    // Initialize timing
    mach_timebase_info(&ctx->timebase_info);
    
    // Set default configuration
    ctx->config.warmup_iterations = 5;
    ctx->config.measurement_iterations = 10;
    ctx->config.measure_power = false;
    ctx->config.measure_memory = true;
    ctx->config.verbose_output = false;
    ctx->config.timeout_seconds = 30.0;
    
    // Allocate timing arrays
    ctx->max_iterations = 1000;
    ctx->execution_times = (double*)malloc(ctx->max_iterations * sizeof(double));
    if (!ctx->execution_times) {
        free(ctx);
        return METAL_ERROR_OUT_OF_MEMORY;
    }
    
    *context = ctx;
    return METAL_SUCCESS;
}

void benchmark_context_destroy(BenchmarkContext* context) {
    if (context) {
        free(context->execution_times);
        free(context);
    }
}

MetalError benchmark_set_config(BenchmarkContext* context, const BenchmarkConfig* config) {
    if (!context || !config) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    context->config = *config;
    return METAL_SUCCESS;
}

MetalError benchmark_get_config(BenchmarkContext* context, BenchmarkConfig* config) {
    if (!context || !config) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    *config = context->config;
    return METAL_SUCCESS;
}

// Benchmark individual operations
MetalError benchmark_matrix_multiply(BenchmarkContext* context, 
                                   BenchmarkSize size, 
                                   BenchmarkResult* result) {
    if (!context || !result) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    memset(result, 0, sizeof(BenchmarkResult));
    
    uint32_t dim = benchmark_get_matrix_dimension(size);
    size_t matrix_size = dim * dim * sizeof(float);
    
    // Allocate buffers
    MMBuffer* buffer_a = NULL;
    MMBuffer* buffer_b = NULL;
    MMBuffer* buffer_c = NULL;
    
    MetalError alloc_result = mm_buffer_alloc(context->memory_manager, matrix_size, MM_ACCESS_READ_ONLY, &buffer_a);
    if (alloc_result != METAL_SUCCESS) return alloc_result;
    
    alloc_result = mm_buffer_alloc(context->memory_manager, matrix_size, MM_ACCESS_READ_ONLY, &buffer_b);
    if (alloc_result != METAL_SUCCESS) {
        mm_buffer_release(buffer_a);
        return alloc_result;
    }
    
    alloc_result = mm_buffer_alloc(context->memory_manager, matrix_size, MM_ACCESS_WRITE_ONLY, &buffer_c);
    if (alloc_result != METAL_SUCCESS) {
        mm_buffer_release(buffer_a);
        mm_buffer_release(buffer_b);
        return alloc_result;
    }
    
    // Initialize data
    float* data_a = (float*)mm_buffer_map_write(buffer_a);
    float* data_b = (float*)mm_buffer_map_write(buffer_b);
    
    for (uint32_t i = 0; i < dim * dim; i++) {
        data_a[i] = (float)(rand()) / RAND_MAX;
        data_b[i] = (float)(rand()) / RAND_MAX;
    }
    
    mm_buffer_unmap(buffer_a);
    mm_buffer_unmap(buffer_b);
    
    // Setup parameters
    TensorDescriptor desc_a, desc_b, desc_c;
    uint32_t dims[] = {dim, dim};
    ck_create_tensor_desc(&desc_a, dims, 2);
    ck_create_tensor_desc(&desc_b, dims, 2);
    ck_create_tensor_desc(&desc_c, dims, 2);
    
    MatMulParams params = {
        .a_desc = desc_a,
        .b_desc = desc_b,
        .c_desc = desc_c,
        .transpose_a = false,
        .transpose_b = false,
        .alpha = 1.0f,
        .beta = 0.0f
    };
    
    // Warm-up iterations
    for (uint32_t i = 0; i < context->config.warmup_iterations; i++) {
        ck_matrix_multiply(context->kernel_ctx, buffer_a, buffer_b, buffer_c, &params);
    }
    
    // Measurement iterations
    double total_time = 0.0;
    double min_time = DBL_MAX;
    double max_time = 0.0;
    uint32_t successful_iterations = 0;
    
    for (uint32_t i = 0; i < context->config.measurement_iterations; i++) {
        uint64_t start_time = get_absolute_time_ns(context);
        
        MetalError exec_result = ck_matrix_multiply(context->kernel_ctx, buffer_a, buffer_b, buffer_c, &params);
        
        uint64_t end_time = get_absolute_time_ns(context);
        
        if (exec_result == METAL_SUCCESS) {
            double execution_time = nanoseconds_to_milliseconds(end_time - start_time);
            context->execution_times[successful_iterations] = execution_time;
            
            total_time += execution_time;
            min_time = fmin(min_time, execution_time);
            max_time = fmax(max_time, execution_time);
            successful_iterations++;
        }
    }
    
    if (successful_iterations == 0) {
        mm_buffer_release(buffer_a);
        mm_buffer_release(buffer_b);
        mm_buffer_release(buffer_c);
        return METAL_ERROR_EXECUTION_FAILED;
    }
    
    // Calculate statistics
    result->execution_time_ms = total_time / successful_iterations;
    result->min_time_ms = min_time;
    result->max_time_ms = max_time;
    result->iterations = successful_iterations;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (uint32_t i = 0; i < successful_iterations; i++) {
        double diff = context->execution_times[i] - result->execution_time_ms;
        variance += diff * diff;
    }
    result->std_deviation_ms = sqrt(variance / successful_iterations);
    
    // Calculate performance metrics
    double gflops = benchmark_calculate_theoretical_gflops(BENCH_MATRIX_MULTIPLY, size);
    result->gflops = gflops / (result->execution_time_ms / 1000.0);
    
    size_t total_bytes = 3 * matrix_size; // A + B + C
    result->memory_bandwidth_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (result->execution_time_ms / 1000.0);
    result->memory_used_bytes = total_bytes;
    
    // Cleanup
    mm_buffer_release(buffer_a);
    mm_buffer_release(buffer_b);
    mm_buffer_release(buffer_c);
    
    return METAL_SUCCESS;
}

MetalError benchmark_activation_functions(BenchmarkContext* context, 
                                        BenchmarkSize size, 
                                        BenchmarkResult* result) {
    if (!context || !result) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    memset(result, 0, sizeof(BenchmarkResult));
    
    uint32_t elements = benchmark_get_matrix_dimension(size) * benchmark_get_matrix_dimension(size);
    size_t buffer_size = elements * sizeof(float);
    
    // Allocate buffers
    MMBuffer* input_buffer = NULL;
    MMBuffer* output_buffer = NULL;
    
    MetalError alloc_result = mm_buffer_alloc(context->memory_manager, buffer_size, MM_ACCESS_READ_WRITE, &input_buffer);
    if (alloc_result != METAL_SUCCESS) return alloc_result;
    
    alloc_result = mm_buffer_alloc(context->memory_manager, buffer_size, MM_ACCESS_WRITE_ONLY, &output_buffer);
    if (alloc_result != METAL_SUCCESS) {
        mm_buffer_release(input_buffer);
        return alloc_result;
    }
    
    // Initialize input data
    float* input_data = (float*)mm_buffer_map_write(input_buffer);
    for (uint32_t i = 0; i < elements; i++) {
        input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Range [-1, 1]
    }
    mm_buffer_unmap(input_buffer);
    
    // Setup tensor descriptor
    TensorDescriptor desc;
    uint32_t dim = benchmark_get_matrix_dimension(size);
    uint32_t dims[] = {1, 1, dim, dim};
    ck_create_tensor_desc(&desc, dims, 4);
    
    // Benchmark different activation functions
    ActivationType activations[] = {CK_ACTIVATION_RELU, CK_ACTIVATION_SIGMOID, CK_ACTIVATION_TANH};
    uint32_t num_activations = sizeof(activations) / sizeof(activations[0]);
    
    double total_time = 0.0;
    uint32_t successful_iterations = 0;
    
    for (uint32_t act = 0; act < num_activations; act++) {
        // Warm-up
        for (uint32_t i = 0; i < context->config.warmup_iterations; i++) {
            ck_activation(context->kernel_ctx, input_buffer, output_buffer, activations[act], &desc);
        }
        
        // Measurement
        for (uint32_t i = 0; i < context->config.measurement_iterations; i++) {
            uint64_t start_time = get_absolute_time_ns(context);
            
            MetalError exec_result = ck_activation(context->kernel_ctx, input_buffer, output_buffer, activations[act], &desc);
            
            uint64_t end_time = get_absolute_time_ns(context);
            
            if (exec_result == METAL_SUCCESS) {
                double execution_time = nanoseconds_to_milliseconds(end_time - start_time);
                total_time += execution_time;
                successful_iterations++;
            }
        }
    }
    
    if (successful_iterations == 0) {
        mm_buffer_release(input_buffer);
        mm_buffer_release(output_buffer);
        return METAL_ERROR_EXECUTION_FAILED;
    }
    
    // Calculate results
    result->execution_time_ms = total_time / successful_iterations;
    result->iterations = successful_iterations;
    result->memory_used_bytes = 2 * buffer_size;
    result->memory_bandwidth_gbps = (result->memory_used_bytes / (1024.0 * 1024.0 * 1024.0)) / (result->execution_time_ms / 1000.0);
    
    // Cleanup
    mm_buffer_release(input_buffer);
    mm_buffer_release(output_buffer);
    
    return METAL_SUCCESS;
}

MetalError benchmark_memory_operations(BenchmarkContext* context, 
                                     BenchmarkSize size, 
                                     BenchmarkResult* result) {
    if (!context || !result) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    memset(result, 0, sizeof(BenchmarkResult));
    
    uint32_t dim = benchmark_get_matrix_dimension(size);
    size_t buffer_size = dim * dim * sizeof(float);
    
    // Test memory allocation/deallocation performance
    double total_time = 0.0;
    uint32_t successful_iterations = 0;
    
    for (uint32_t i = 0; i < context->config.measurement_iterations; i++) {
        uint64_t start_time = get_absolute_time_ns(context);
        
        // Allocate buffer
        MMBuffer* buffer = NULL;
        MetalError alloc_result = mm_buffer_alloc(context->memory_manager, buffer_size, MM_ACCESS_READ_WRITE, &buffer);
        
        if (alloc_result == METAL_SUCCESS) {
            // Perform memory operations
            float* data = (float*)mm_buffer_map_write(buffer);
            if (data) {
                memset(data, 0, buffer_size); // Write test
                mm_buffer_unmap(buffer);
                
                data = (float*)mm_buffer_map_read(buffer);
                if (data) {
                    volatile float sum = 0.0f; // Read test
                    for (uint32_t j = 0; j < dim * dim; j++) {
                        sum += data[j];
                    }
                    mm_buffer_unmap(buffer);
                }
            }
            
            // Release buffer
            mm_buffer_release(buffer);
        }
        
        uint64_t end_time = get_absolute_time_ns(context);
        
        if (alloc_result == METAL_SUCCESS) {
            double execution_time = nanoseconds_to_milliseconds(end_time - start_time);
            total_time += execution_time;
            successful_iterations++;
        }
    }
    
    if (successful_iterations == 0) {
        return METAL_ERROR_EXECUTION_FAILED;
    }
    
    result->execution_time_ms = total_time / successful_iterations;
    result->iterations = successful_iterations;
    result->memory_used_bytes = buffer_size;
    result->memory_bandwidth_gbps = (buffer_size * 2 / (1024.0 * 1024.0 * 1024.0)) / (result->execution_time_ms / 1000.0); // Read + Write
    
    return METAL_SUCCESS;
}

// Comprehensive benchmark suite
MetalError benchmark_run_full_suite(BenchmarkContext* context, BenchmarkSuite* suite) {
    if (!context || !suite) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    memset(suite, 0, sizeof(BenchmarkSuite));
    
    // Get device information
    get_device_info(context, suite);
    
    // Check Neural Engine availability
    NESystemInfo ne_info;
    if (ne_get_system_info(&ne_info) == 0) {
        suite->neural_engine_available = ne_info.neural_engine_available;
    }
    
    printf("Running comprehensive benchmark suite on %s...\n", suite->device_name);
    printf("Total Memory: %llu GB, Neural Engine: %s\n", 
           suite->total_memory_gb, 
           suite->neural_engine_available ? "YES" : "NO");
    
    double total_score = 0.0;
    uint32_t completed_tests = 0;
    
    // Run benchmarks for each operation and size
    BenchmarkOperation operations[] = {BENCH_MATRIX_MULTIPLY, BENCH_ACTIVATION, BENCH_MEMORY_COPY};
    uint32_t num_operations = sizeof(operations) / sizeof(operations[0]);
    
    for (uint32_t op = 0; op < num_operations; op++) {
        for (uint32_t sz = 0; sz < BENCH_SIZE_COUNT; sz++) {
            printf("Testing %s at %s size... ", 
                   benchmark_operation_name(operations[op]), 
                   benchmark_size_name((BenchmarkSize)sz));
            fflush(stdout);
            
            MetalError result = benchmark_run_operation(context, operations[op], (BenchmarkSize)sz, 
                                                       &suite->results[operations[op]][sz]);
            
            if (result == METAL_SUCCESS) {
                printf("PASS (%.2f ms)\n", suite->results[operations[op]][sz].execution_time_ms);
                total_score += suite->results[operations[op]][sz].gflops;
                completed_tests++;
            } else {
                printf("FAIL\n");
            }
        }
    }
    
    suite->total_score = total_score / completed_tests;
    
    printf("\nBenchmark Suite Complete!\n");
    printf("Overall Performance Score: %.2f GFLOPS\n", suite->total_score);
    
    return METAL_SUCCESS;
}

MetalError benchmark_run_operation(BenchmarkContext* context, 
                                  BenchmarkOperation operation, 
                                  BenchmarkSize size, 
                                  BenchmarkResult* result) {
    switch (operation) {
        case BENCH_MATRIX_MULTIPLY:
            return benchmark_matrix_multiply(context, size, result);
        case BENCH_ACTIVATION:
            return benchmark_activation_functions(context, size, result);
        case BENCH_MEMORY_COPY:
            return benchmark_memory_operations(context, size, result);
        default:
            return METAL_ERROR_INVALID_PARAMETER;
    }
}

// Reporting functions
MetalError benchmark_print_results(const BenchmarkSuite* suite, bool detailed) {
    if (!suite) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    printf("\n=== NNCP Metal Performance Benchmark Results ===\n");
    printf("Device: %s\n", suite->device_name);
    printf("Memory: %llu GB\n", suite->total_memory_gb);
    printf("Neural Engine: %s\n", suite->neural_engine_available ? "Available" : "Not Available");
    printf("Timestamp: %s\n", suite->timestamp);
    printf("Overall Score: %.2f GFLOPS\n\n", suite->total_score);
    
    if (detailed) {
        const char* size_names[] = {"Small", "Medium", "Large", "XLarge"};
        const char* op_names[] = {"MatMul", "Activation", "Memory", "Conv", "Attention", "LayerNorm", "Softmax"};
        
        printf("Detailed Results:\n");
        printf("%-12s", "Operation");
        for (int sz = 0; sz < BENCH_SIZE_COUNT; sz++) {
            printf("%12s", size_names[sz]);
        }
        printf("\n");
        
        for (int op = 0; op < BENCH_OP_COUNT; op++) {
            bool has_data = false;
            for (int sz = 0; sz < BENCH_SIZE_COUNT; sz++) {
                if (suite->results[op][sz].iterations > 0) {
                    has_data = true;
                    break;
                }
            }
            
            if (has_data) {
                printf("%-12s", op_names[op]);
                for (int sz = 0; sz < BENCH_SIZE_COUNT; sz++) {
                    if (suite->results[op][sz].iterations > 0) {
                        printf("%9.2f ms", suite->results[op][sz].execution_time_ms);
                    } else {
                        printf("%12s", "N/A");
                    }
                }
                printf("\n");
            }
        }
    }
    
    return METAL_SUCCESS;
}

// Utility functions
const char* benchmark_operation_name(BenchmarkOperation op) {
    switch (op) {
        case BENCH_MATRIX_MULTIPLY: return "Matrix Multiply";
        case BENCH_CONVOLUTION: return "Convolution";
        case BENCH_ACTIVATION: return "Activation";
        case BENCH_ATTENTION: return "Attention";
        case BENCH_MEMORY_COPY: return "Memory Copy";
        case BENCH_LAYER_NORM: return "Layer Norm";
        case BENCH_SOFTMAX: return "Softmax";
        default: return "Unknown";
    }
}

const char* benchmark_size_name(BenchmarkSize size) {
    switch (size) {
        case BENCH_SIZE_SMALL: return "Small";
        case BENCH_SIZE_MEDIUM: return "Medium";
        case BENCH_SIZE_LARGE: return "Large";
        case BENCH_SIZE_XLARGE: return "XLarge";
        default: return "Unknown";
    }
}

uint32_t benchmark_get_matrix_dimension(BenchmarkSize size) {
    switch (size) {
        case BENCH_SIZE_SMALL: return 64;
        case BENCH_SIZE_MEDIUM: return 128;
        case BENCH_SIZE_LARGE: return 256;
        case BENCH_SIZE_XLARGE: return 512;
        default: return 64;
    }
}

double benchmark_calculate_theoretical_gflops(BenchmarkOperation op, BenchmarkSize size) {
    uint32_t dim = benchmark_get_matrix_dimension(size);
    double operations = 0.0;
    
    switch (op) {
        case BENCH_MATRIX_MULTIPLY:
            // Matrix multiplication: 2 * M * N * K operations
            operations = 2.0 * dim * dim * dim;
            break;
        case BENCH_ACTIVATION:
            // Activation: 1 operation per element
            operations = dim * dim;
            break;
        case BENCH_CONVOLUTION:
            // Simplified convolution estimate
            operations = dim * dim * 9; // 3x3 kernel
            break;
        default:
            operations = dim * dim;
            break;
    }
    
    return operations / 1e9; // Convert to GFLOPS
}