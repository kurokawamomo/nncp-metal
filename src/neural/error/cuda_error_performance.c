/*
 * CUDA Error Handler Performance Optimizations Implementation
 * 
 * Performance-optimized error handling for production environments
 * while maintaining full CUDA compatibility.
 */

#include "cuda_error_performance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

// Global performance state
static CUDAErrorPerfProfile g_perf_profile = CUDA_ERROR_PERF_DEVELOPMENT;
static bool g_performance_mode = false;
static CUDAErrorPerfMetrics g_perf_metrics = {0};
static uint64_t g_monitoring_start_time = 0;
static bool g_monitoring_active = false;

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

// Performance-optimized configurations
CUDAErrorConfig* cuda_error_config_create_production_optimized(void) {
    CUDAErrorConfig* config = (CUDAErrorConfig*)calloc(1, sizeof(CUDAErrorConfig));
    if (!config) return NULL;
    
    // Minimal overhead configuration
    config->enable_error_logging = true;
    config->enable_stack_trace = false;
    config->abort_on_fatal_error = true;
    config->chain_errors = false;           // Reduced memory usage
    config->error_callback = NULL;
    config->callback_user_data = NULL;
    config->log_file_path = NULL;
    config->log_to_console = false;         // No console output in production
    config->min_log_severity = CUDA_ERROR_SEVERITY_FATAL;  // Only fatal errors
    config->max_error_chain_length = 1;    // Minimal chain
    config->enable_error_caching = false;   // No caching overhead
    
    return config;
}

CUDAErrorConfig* cuda_error_config_create_testing_optimized(void) {
    CUDAErrorConfig* config = (CUDAErrorConfig*)calloc(1, sizeof(CUDAErrorConfig));
    if (!config) return NULL;
    
    // Balanced configuration for automated testing
    config->enable_error_logging = true;
    config->enable_stack_trace = false;
    config->abort_on_fatal_error = false;   // Don't abort during tests
    config->chain_errors = true;
    config->error_callback = NULL;
    config->callback_user_data = NULL;
    config->log_file_path = NULL;
    config->log_to_console = false;         // Quiet testing
    config->min_log_severity = CUDA_ERROR_SEVERITY_ERROR;
    config->max_error_chain_length = 5;
    config->enable_error_caching = true;
    
    return config;
}

CUDAErrorConfig* cuda_error_config_create_benchmark_optimized(void) {
    CUDAErrorConfig* config = (CUDAErrorConfig*)calloc(1, sizeof(CUDAErrorConfig));
    if (!config) return NULL;
    
    // Ultra-minimal configuration for benchmarking
    config->enable_error_logging = false;   // No logging overhead
    config->enable_stack_trace = false;
    config->abort_on_fatal_error = false;   // Let benchmarks handle errors
    config->chain_errors = false;
    config->error_callback = NULL;
    config->callback_user_data = NULL;
    config->log_file_path = NULL;
    config->log_to_console = false;
    config->min_log_severity = CUDA_ERROR_SEVERITY_FATAL;
    config->max_error_chain_length = 0;    // No error chain
    config->enable_error_caching = false;
    
    return config;
}

CUDAErrorConfig* cuda_error_config_create_profile(CUDAErrorPerfProfile profile) {
    switch (profile) {
        case CUDA_ERROR_PERF_PRODUCTION:
            return cuda_error_config_create_production_optimized();
        case CUDA_ERROR_PERF_TESTING:
            return cuda_error_config_create_testing_optimized();
        case CUDA_ERROR_PERF_DEVELOPMENT:
            return cuda_error_config_create_debug();
        case CUDA_ERROR_PERF_BENCHMARKING:
            return cuda_error_config_create_benchmark_optimized();
        default:
            return cuda_error_config_create_default();
    }
}

// Performance monitoring
void cuda_error_perf_start_monitoring(void) {
    g_monitoring_start_time = get_time_ns();
    g_monitoring_active = true;
    memset(&g_perf_metrics, 0, sizeof(CUDAErrorPerfMetrics));
}

void cuda_error_perf_stop_monitoring(void) {
    if (g_monitoring_active) {
        uint64_t total_time = get_time_ns() - g_monitoring_start_time;
        g_perf_metrics.error_handler_overhead_ns = total_time;
        
        if (g_perf_metrics.total_error_checks > 0) {
            g_perf_metrics.avg_error_check_time_ns = 
                (float)g_perf_metrics.error_handler_overhead_ns / g_perf_metrics.total_error_checks;
        }
        
        // Detect if error handling is impacting performance
        // Consider it impactful if error handling takes >1% of total time
        g_perf_metrics.performance_impact_detected = 
            (g_perf_metrics.error_handler_overhead_ns > total_time / 100);
        
        g_monitoring_active = false;
    }
}

CUDAErrorPerfMetrics* cuda_error_perf_get_metrics(void) {
    return &g_perf_metrics;
}

void cuda_error_perf_reset_metrics(void) {
    memset(&g_perf_metrics, 0, sizeof(CUDAErrorPerfMetrics));
}

void cuda_error_perf_print_report(void) {
    printf("CUDA Error Handler Performance Report:\n");
    printf("=====================================\n");
    printf("Total error checks: %llu\n", (unsigned long long)g_perf_metrics.total_error_checks);
    printf("Failed error checks: %llu\n", (unsigned long long)g_perf_metrics.failed_error_checks);
    printf("Error handler overhead: %.2f ms\n", g_perf_metrics.error_handler_overhead_ns / 1000000.0);
    printf("Average error check time: %.2f ns\n", g_perf_metrics.avg_error_check_time_ns);
    printf("Memory used: %llu bytes\n", (unsigned long long)g_perf_metrics.memory_used_bytes);
    printf("Performance impact detected: %s\n", 
           g_perf_metrics.performance_impact_detected ? "YES" : "NO");
    
    if (g_perf_metrics.performance_impact_detected) {
        printf("\nOptimization recommendations:\n");
        printf("- Consider using CUDA_ERROR_PERF_PRODUCTION profile\n");
        printf("- Disable error chaining in performance-critical paths\n");
        printf("- Use CUDA_CHECK_FAST() in hot loops\n");
    }
    printf("=====================================\n");
}

// Runtime performance mode control
bool cuda_error_is_performance_mode(void) {
    return g_performance_mode;
}

void cuda_error_set_performance_mode(bool enabled) {
    g_performance_mode = enabled;
}

CUDAErrorPerfProfile cuda_error_get_perf_profile(void) {
    return g_perf_profile;
}

void cuda_error_set_perf_profile(CUDAErrorPerfProfile profile) {
    g_perf_profile = profile;
    
    // Automatically enable performance mode for optimized profiles
    switch (profile) {
        case CUDA_ERROR_PERF_PRODUCTION:
        case CUDA_ERROR_PERF_BENCHMARKING:
            g_performance_mode = true;
            break;
        case CUDA_ERROR_PERF_TESTING:
        case CUDA_ERROR_PERF_DEVELOPMENT:
            g_performance_mode = false;
            break;
    }
}

// Error handling optimization hints
void cuda_error_hint_performance_critical_section_start(void) {
    // Could implement thread-local performance mode switching
    // For now, just track that we're in a critical section
    if (g_monitoring_active) {
        // Performance-critical sections should have minimal error handling
    }
}

void cuda_error_hint_performance_critical_section_end(void) {
    // End of critical section
}

void cuda_error_hint_expected_error_rate(float error_rate) {
    // Hint about expected error rate to optimize error handling
    // If error_rate is very low, we can optimize for the success case
    // If error_rate is high, we should optimize error handling paths
    (void)error_rate; // Unused for now
}

// Memory and resource optimization
void cuda_error_optimize_memory_usage(void) {
    // Compact error chains, free unused memory, etc.
    cuda_error_compact_error_chain();
}

void cuda_error_compact_error_chain(void) {
    // Implementation would compact the error chain to reduce memory usage
    // This is a placeholder for actual implementation
}

size_t cuda_error_get_memory_footprint(void) {
    // Calculate approximate memory footprint of error handling system
    size_t footprint = sizeof(CUDAErrorPerfMetrics);
    
    // Add error chain memory
    size_t chain_length = cuda_error_get_chain_length();
    footprint += chain_length * sizeof(CUDAErrorInfo);
    
    g_perf_metrics.memory_used_bytes = footprint;
    return footprint;
}