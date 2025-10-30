/*
 * CUDA Error Handler Performance Optimizations
 * 
 * Performance-optimized error handling configurations for production
 * and testing environments while maintaining CUDA compatibility.
 */

#ifndef CUDA_ERROR_PERFORMANCE_H
#define CUDA_ERROR_PERFORMANCE_H

#include "cuda_error_handler.h"

#ifdef __cplusplus
extern "C" {
#endif

// Performance-optimized error configurations
typedef enum {
    CUDA_ERROR_PERF_PRODUCTION = 0,  // Minimal overhead, critical errors only
    CUDA_ERROR_PERF_TESTING,         // Balanced for automated testing
    CUDA_ERROR_PERF_DEVELOPMENT,     // Full debugging, maximum verbosity
    CUDA_ERROR_PERF_BENCHMARKING     // Ultra-minimal for performance testing
} CUDAErrorPerfProfile;

// Performance metrics for error handling
typedef struct {
    uint64_t error_handler_overhead_ns;  // Time spent in error handling
    uint64_t total_error_checks;         // Number of error checks performed
    uint64_t failed_error_checks;       // Number of failed checks
    uint64_t memory_used_bytes;          // Memory used by error system
    float avg_error_check_time_ns;       // Average time per error check
    bool performance_impact_detected;    // Whether error handling impacts perf
} CUDAErrorPerfMetrics;

// Performance-optimized configurations
CUDAErrorConfig* cuda_error_config_create_production_optimized(void);
CUDAErrorConfig* cuda_error_config_create_testing_optimized(void);
CUDAErrorConfig* cuda_error_config_create_benchmark_optimized(void);
CUDAErrorConfig* cuda_error_config_create_profile(CUDAErrorPerfProfile profile);

// Performance monitoring
void cuda_error_perf_start_monitoring(void);
void cuda_error_perf_stop_monitoring(void);
CUDAErrorPerfMetrics* cuda_error_perf_get_metrics(void);
void cuda_error_perf_reset_metrics(void);
void cuda_error_perf_print_report(void);

// Optimized error checking macros for performance-critical paths
#define CUDA_CHECK_FAST(call) do { \
    if (cuda_error_is_performance_mode()) { \
        (call); \
    } else { \
        CUDA_CHECK(call); \
    } \
} while(0)

#define CUDA_CHECK_PERF_CRITICAL(call) do { \
    if (cuda_error_get_perf_profile() == CUDA_ERROR_PERF_BENCHMARKING) { \
        (call); \
    } else { \
        CUDA_CHECK(call); \
    } \
} while(0)

// Runtime performance mode control
bool cuda_error_is_performance_mode(void);
void cuda_error_set_performance_mode(bool enabled);
CUDAErrorPerfProfile cuda_error_get_perf_profile(void);
void cuda_error_set_perf_profile(CUDAErrorPerfProfile profile);

// Error handling optimization hints
void cuda_error_hint_performance_critical_section_start(void);
void cuda_error_hint_performance_critical_section_end(void);
void cuda_error_hint_expected_error_rate(float error_rate);

// Memory and resource optimization
void cuda_error_optimize_memory_usage(void);
void cuda_error_compact_error_chain(void);
size_t cuda_error_get_memory_footprint(void);

#ifdef __cplusplus
}
#endif

#endif // CUDA_ERROR_PERFORMANCE_H