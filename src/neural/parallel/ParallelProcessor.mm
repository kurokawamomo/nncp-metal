/*
 * ParallelProcessor.mm
 * 
 * Advanced Parallel Processing Engine Implementation for Apple Silicon
 * Batch processing parallelization, layer pipelining, async GPU compute,
 * and intelligent CPU-GPU load balancing
 */

#include "ParallelProcessor.h"
#include "../acceleration/MetalComputeAccelerator.h"
#include "../optimization/CacheOptimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/sysctl.h>
#include <pthread.h>
#include <unistd.h>
#include <mach/mach.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Dispatch/Dispatch.h>
#import <Foundation/Foundation.h>
#endif

// Thread synchronization structures
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    volatile bool ready;
    volatile bool completed;
} ThreadSyncPoint;

// CPU thread pool worker
typedef struct {
    pthread_t thread;
    uint32_t worker_id;
    bool is_active;
    bool is_performance_core;
    void* work_queue;
    ThreadSyncPoint* sync_point;
    struct ParallelProcessor* processor;
} CPUWorkerThread;

// GPU command buffer context
typedef struct {
    uint32_t buffer_id;
    bool is_active;
    bool is_completed;
    uint64_t submission_time;
    uint64_t completion_time;
    size_t memory_usage_bytes;
#ifdef __OBJC__
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> compute_encoder;
#endif
} GPUCommandContext;

// Pipeline stage execution context
typedef struct {
    uint32_t stage_id;
    PipelineStageConfig config;
    bool is_executing;
    bool is_completed;
    uint64_t start_time;
    uint64_t end_time;
    void* input_data;
    void* output_data;
    size_t data_size;
    ThreadSyncPoint sync_point;
} PipelineStageContext;

// Internal structure for ParallelProcessor
struct ParallelProcessor {
    ParallelProcessingConfig config;        // Configuration settings
    AppleSiliconSystemInfo system_info;     // System hardware information
    ParallelProcessingMetrics metrics;      // Performance metrics
    LoadBalancingState load_balancing;      // Load balancing state
    
    // Hardware integration
    MetalComputeAccelerator* metal_accelerator; // Metal GPU accelerator
    CacheOptimizer* cache_optimizer;        // Cache optimizer
    
    // CPU thread management
    CPUWorkerThread* cpu_workers;           // CPU worker threads
    uint32_t active_cpu_workers;            // Number of active CPU workers
    pthread_mutex_t cpu_work_mutex;         // CPU work queue mutex
    
    // GPU command management
    GPUCommandContext* gpu_contexts;        // GPU command contexts
    uint32_t active_gpu_contexts;           // Number of active GPU contexts
    uint32_t max_gpu_contexts;              // Maximum GPU contexts
    
    // Pipeline management
    PipelineStageContext* pipeline_stages;  // Pipeline stage contexts
    uint32_t pipeline_stage_count;          // Number of pipeline stages
    uint32_t pipeline_depth;                // Current pipeline depth
    
    // Async execution tracking
    AsyncExecutionContext* async_contexts;  // Async execution contexts
    uint32_t max_async_contexts;            // Maximum async contexts
    uint32_t active_async_contexts;         // Active async contexts
    pthread_mutex_t async_mutex;            // Async context mutex
    
    // Performance monitoring
    bool monitoring_active;                 // Monitoring active flag
    uint64_t monitoring_start_time;        // Monitoring start timestamp
    uint32_t monitoring_interval_ms;       // Monitoring interval
    pthread_t monitoring_thread;           // Monitoring thread
    
    // Batch processing state
    BatchProcessingConfig batch_config;     // Batch processing configuration
    uint32_t optimal_cpu_batch_size;       // Optimal CPU batch size
    uint32_t optimal_gpu_batch_size;       // Optimal GPU batch size
    
    // Memory management
    void* shared_memory_pool;               // Shared memory pool
    size_t memory_pool_size;                // Memory pool size
    size_t memory_pool_used;                // Used memory in pool
    pthread_mutex_t memory_mutex;           // Memory allocation mutex
    
    // Thermal and power management
    float current_temperature;              // Current system temperature
    float current_power_draw;               // Current power draw
    bool thermal_throttle_active;           // Thermal throttling active
    
    bool initialized;                       // Initialization state
};

// Utility functions
static uint64_t get_current_time_microseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

static uint32_t get_cpu_core_count(void) {
    uint32_t core_count = 0;
    size_t size = sizeof(core_count);
    if (sysctlbyname("hw.ncpu", &core_count, &size, NULL, 0) != 0) {
        return 8; // Default fallback
    }
    return core_count;
}

static uint32_t get_performance_core_count(void) {
    uint32_t perf_core_count = 0;
    size_t size = sizeof(perf_core_count);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_core_count, &size, NULL, 0) != 0) {
        return get_cpu_core_count() / 2; // Estimate: half are performance cores
    }
    return perf_core_count;
}

static uint64_t get_system_memory_size(void) {
    uint64_t memory_size = 0;
    size_t size = sizeof(memory_size);
    if (sysctlbyname("hw.memsize", &memory_size, &size, NULL, 0) != 0) {
        return 16ULL * 1024 * 1024 * 1024; // Default to 16GB
    }
    return memory_size;
}

static float get_system_temperature(void) {
    // Simplified thermal monitoring - in real implementation would use IOKit
    // For now, return a simulated temperature based on system load
    return 45.0f + (rand() % 20); // 45-65°C range
}

// Thread synchronization utilities
static void init_sync_point(ThreadSyncPoint* sync) {
    pthread_mutex_init(&sync->mutex, NULL);
    pthread_cond_init(&sync->condition, NULL);
    sync->ready = false;
    sync->completed = false;
}

static void destroy_sync_point(ThreadSyncPoint* sync) {
    pthread_mutex_destroy(&sync->mutex);
    pthread_cond_destroy(&sync->condition);
}

static void signal_sync_point(ThreadSyncPoint* sync, bool is_completed) {
    pthread_mutex_lock(&sync->mutex);
    if (is_completed) {
        sync->completed = true;
    } else {
        sync->ready = true;
    }
    pthread_cond_broadcast(&sync->condition);
    pthread_mutex_unlock(&sync->mutex);
}

static void wait_sync_point(ThreadSyncPoint* sync, bool wait_for_completion) {
    pthread_mutex_lock(&sync->mutex);
    if (wait_for_completion) {
        while (!sync->completed) {
            pthread_cond_wait(&sync->condition, &sync->mutex);
        }
    } else {
        while (!sync->ready) {
            pthread_cond_wait(&sync->condition, &sync->mutex);
        }
    }
    pthread_mutex_unlock(&sync->mutex);
}

// CPU worker thread function
static void* cpu_worker_thread_function(void* arg) {
    CPUWorkerThread* worker = (CPUWorkerThread*)arg;
    ParallelProcessor* processor = worker->processor;
    
    // Set thread affinity based on core type
    if (worker->is_performance_core) {
        // Bind to performance core (simplified - actual implementation would use thread_policy_set)
        pthread_setname_np("NNCP-CPU-P-Worker");
    } else {
        // Bind to efficiency core
        pthread_setname_np("NNCP-CPU-E-Worker");
    }
    
    while (worker->is_active) {
        // Wait for work assignment
        wait_sync_point(worker->sync_point, false);
        
        if (!worker->is_active) break;
        
        // Process assigned work
        // (Work queue processing would be implemented here)
        
        // Signal completion
        signal_sync_point(worker->sync_point, true);
        
        // Small sleep to prevent busy waiting
        usleep(100);
    }
    
    return NULL;
}

// Performance monitoring thread function
static void* monitoring_thread_function(void* arg) {
    ParallelProcessor* processor = (ParallelProcessor*)arg;
    
    pthread_setname_np("NNCP-Monitor");
    
    while (processor->monitoring_active) {
        uint64_t current_time = get_current_time_microseconds();
        
        // Update thermal state
        processor->current_temperature = get_system_temperature();
        if (processor->current_temperature > THERMAL_THROTTLE_TEMPERATURE) {
            processor->thermal_throttle_active = true;
        } else if (processor->current_temperature < THERMAL_THROTTLE_TEMPERATURE - 5.0f) {
            processor->thermal_throttle_active = false;
        }
        
        // Update performance metrics
        processor->metrics.thermal_throttle_events = 
            processor->thermal_throttle_active ? processor->metrics.thermal_throttle_events + 1 : 
                                                processor->metrics.thermal_throttle_events;
        
        // Sleep for monitoring interval
        usleep(processor->monitoring_interval_ms * 1000);
    }
    
    return NULL;
}

// Core API Implementation

ParallelProcessorError parallel_processor_create(ParallelProcessor** processor,
                                                 const ParallelProcessingConfig* config,
                                                 MetalComputeAccelerator* metal_accelerator,
                                                 CacheOptimizer* cache_optimizer) {
    if (!processor || !metal_accelerator || !cache_optimizer) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    *processor = (ParallelProcessor*)calloc(1, sizeof(ParallelProcessor));
    if (!*processor) {
        return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    ParallelProcessor* proc = *processor;
    
    // Use provided config or create default
    if (config) {
        proc->config = *config;
    } else {
        parallel_processor_create_default_config(&proc->config);
    }
    
    proc->metal_accelerator = metal_accelerator;
    proc->cache_optimizer = cache_optimizer;
    
    // Initialize synchronization primitives
    pthread_mutex_init(&proc->cpu_work_mutex, NULL);
    pthread_mutex_init(&proc->async_mutex, NULL);
    pthread_mutex_init(&proc->memory_mutex, NULL);
    
    // Allocate CPU worker threads
    proc->active_cpu_workers = proc->config.max_cpu_threads;
    proc->cpu_workers = (CPUWorkerThread*)calloc(proc->active_cpu_workers, sizeof(CPUWorkerThread));
    if (!proc->cpu_workers) {
        free(proc);
        *processor = NULL;
        return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate GPU command contexts
    proc->max_gpu_contexts = proc->config.max_gpu_command_buffers;
    proc->gpu_contexts = (GPUCommandContext*)calloc(proc->max_gpu_contexts, sizeof(GPUCommandContext));
    if (!proc->gpu_contexts) {
        free(proc->cpu_workers);
        free(proc);
        *processor = NULL;
        return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate async execution contexts
    proc->max_async_contexts = PARALLEL_PROCESSOR_DEFAULT_ASYNC_CONTEXTS;
    proc->async_contexts = (AsyncExecutionContext*)calloc(proc->max_async_contexts, sizeof(AsyncExecutionContext));
    if (!proc->async_contexts) {
        free(proc->gpu_contexts);
        free(proc->cpu_workers);
        free(proc);
        *processor = NULL;
        return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate shared memory pool
    proc->memory_pool_size = 256 * 1024 * 1024; // 256MB shared pool
    proc->shared_memory_pool = malloc(proc->memory_pool_size);
    if (!proc->shared_memory_pool) {
        free(proc->async_contexts);
        free(proc->gpu_contexts);
        free(proc->cpu_workers);
        free(proc);
        *processor = NULL;
        return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    proc->memory_pool_used = 0;
    proc->initialized = false;
    proc->monitoring_active = false;
    proc->thermal_throttle_active = false;
    proc->current_temperature = 50.0f;
    proc->active_gpu_contexts = 0;
    proc->active_async_contexts = 0;
    
    // Initialize performance metrics
    memset(&proc->metrics, 0, sizeof(ParallelProcessingMetrics));
    proc->metrics.parallel_efficiency = 0.5f;
    proc->metrics.power_efficiency_score = 0.7f;
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_initialize(ParallelProcessor* processor) {
    if (!processor) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    // Detect Apple Silicon system characteristics
    processor->system_info.total_cpu_cores = get_cpu_core_count();
    processor->system_info.performance_cpu_cores = get_performance_core_count();
    processor->system_info.efficiency_cpu_cores = 
        processor->system_info.total_cpu_cores - processor->system_info.performance_cpu_cores;
    processor->system_info.unified_memory_size_gb = get_system_memory_size() / (1024 * 1024 * 1024);
    processor->system_info.memory_bandwidth_gbps = APPLE_SILICON_MEMORY_BANDWIDTH_GBPS;
    processor->system_info.max_threads_per_core = 2; // Apple Silicon supports 2 threads per core
    processor->system_info.supports_metal_gpu = true;
    processor->system_info.supports_neural_engine = true;
    processor->system_info.supports_simd_instructions = true;
    
    // Get GPU information from Metal accelerator
    MetalDeviceCapabilities gpu_caps;
    MetalComputeError metal_error = metal_compute_get_device_capabilities(processor->metal_accelerator, &gpu_caps);
    if (metal_error == METAL_COMPUTE_SUCCESS) {
        processor->system_info.gpu_compute_units = gpu_caps.max_compute_units;
    } else {
        processor->system_info.gpu_compute_units = 32; // Default estimate
    }
    
    processor->system_info.neural_engine_cores = 16; // Typical for Apple Silicon
    
    // Initialize CPU worker threads
    uint32_t perf_core_workers = fminf(processor->system_info.performance_cpu_cores, 
                                      processor->active_cpu_workers / 2);
    
    for (uint32_t i = 0; i < processor->active_cpu_workers; i++) {
        CPUWorkerThread* worker = &processor->cpu_workers[i];
        worker->worker_id = i;
        worker->is_performance_core = (i < perf_core_workers);
        worker->is_active = true;
        worker->processor = processor;
        
        // Initialize synchronization
        worker->sync_point = (ThreadSyncPoint*)malloc(sizeof(ThreadSyncPoint));
        if (!worker->sync_point) {
            return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
        }
        init_sync_point(worker->sync_point);
        
        // Create worker thread
        int thread_result = pthread_create(&worker->thread, NULL, cpu_worker_thread_function, worker);
        if (thread_result != 0) {
            return PARALLEL_PROCESSOR_ERROR_THREAD_CREATION;
        }
    }
    
    // Initialize GPU command contexts
    for (uint32_t i = 0; i < processor->max_gpu_contexts; i++) {
        GPUCommandContext* context = &processor->gpu_contexts[i];
        context->buffer_id = i;
        context->is_active = false;
        context->is_completed = true;
        context->memory_usage_bytes = 0;
    }
    
    // Initialize async execution contexts
    for (uint32_t i = 0; i < processor->max_async_contexts; i++) {
        AsyncExecutionContext* context = &processor->async_contexts[i];
        context->context_id = i;
        context->is_active = false;
        context->is_gpu_context = false;
        context->progress_percent = 0.0f;
    }
    
    // Initialize load balancing state
    processor->load_balancing.current_cpu_load = 0.0f;
    processor->load_balancing.current_gpu_load = 0.0f;
    processor->load_balancing.current_memory_pressure = 0.0f;
    processor->load_balancing.current_thermal_state = 0.0f;
    processor->load_balancing.needs_rebalancing = false;
    processor->load_balancing.last_rebalance_time_us = get_current_time_microseconds();
    
    // Initialize optimal workload assignment
    processor->load_balancing.optimal_assignment.cpu_workload_fraction = 0.5f;
    processor->load_balancing.optimal_assignment.gpu_workload_fraction = 0.5f;
    processor->load_balancing.optimal_assignment.cpu_thread_count = 
        processor->system_info.performance_cpu_cores;
    processor->load_balancing.optimal_assignment.gpu_compute_units_used = 
        processor->system_info.gpu_compute_units / 2;
    
    // Initialize batch processing configuration
    parallel_processor_create_default_config(&processor->config);
    processor->batch_config.total_batch_size = PARALLEL_PROCESSOR_DEFAULT_BATCH_SIZE;
    processor->batch_config.max_sub_batch_size = 
        PARALLEL_PROCESSOR_DEFAULT_BATCH_SIZE / processor->active_cpu_workers;
    processor->batch_config.min_sub_batch_size = PARALLEL_PROCESSOR_MIN_BATCH_SIZE;
    processor->batch_config.enable_dynamic_batching = true;
    processor->batch_config.enable_batch_prefetching = true;
    processor->batch_config.optimize_for_throughput = true;
    processor->batch_config.cpu_batch_count = processor->active_cpu_workers;
    processor->batch_config.gpu_batch_count = 4;
    processor->batch_config.memory_efficiency_target = 0.8f;
    
    processor->optimal_cpu_batch_size = processor->batch_config.max_sub_batch_size;
    processor->optimal_gpu_batch_size = processor->batch_config.total_batch_size / 4;
    
    processor->initialized = true;
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_get_system_info(ParallelProcessor* processor,
                                                          AppleSiliconSystemInfo* system_info) {
    if (!processor || !system_info || !processor->initialized) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    *system_info = processor->system_info;
    return PARALLEL_PROCESSOR_SUCCESS;
}

// Batch Processing Parallelization

ParallelProcessorError parallel_processor_configure_batch_processing(ParallelProcessor* processor,
                                                                     const BatchProcessingConfig* batch_config) {
    if (!processor || !batch_config) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    processor->batch_config = *batch_config;
    
    // Validate and adjust batch configuration
    if (processor->batch_config.max_sub_batch_size > PARALLEL_PROCESSOR_MAX_BATCH_SIZE) {
        processor->batch_config.max_sub_batch_size = PARALLEL_PROCESSOR_MAX_BATCH_SIZE;
    }
    if (processor->batch_config.min_sub_batch_size < PARALLEL_PROCESSOR_MIN_BATCH_SIZE) {
        processor->batch_config.min_sub_batch_size = PARALLEL_PROCESSOR_MIN_BATCH_SIZE;
    }
    if (processor->batch_config.cpu_batch_count > processor->active_cpu_workers) {
        processor->batch_config.cpu_batch_count = processor->active_cpu_workers;
    }
    if (processor->batch_config.gpu_batch_count > processor->max_gpu_contexts) {
        processor->batch_config.gpu_batch_count = processor->max_gpu_contexts;
    }
    
    // Recalculate optimal batch sizes
    processor->optimal_cpu_batch_size = 
        processor->batch_config.total_batch_size / processor->batch_config.cpu_batch_count;
    processor->optimal_gpu_batch_size = 
        processor->batch_config.total_batch_size / processor->batch_config.gpu_batch_count;
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_execute_batch_parallel(ParallelProcessor* processor,
                                                                 void* const* input_data,
                                                                 void** output_data,
                                                                 uint32_t batch_count,
                                                                 void* processing_function,
                                                                 void* user_context) {
    if (!processor || !input_data || !output_data || !processing_function || 
        batch_count == 0 || !processor->initialized) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_current_time_microseconds();
    
    // Determine optimal distribution between CPU and GPU
    uint32_t cpu_batches = (uint32_t)(batch_count * processor->load_balancing.optimal_assignment.cpu_workload_fraction);
    uint32_t gpu_batches = batch_count - cpu_batches;
    
    // Ensure at least some work for each processor type if enabled
    if (processor->config.enable_cpu_parallelization && cpu_batches == 0 && batch_count > 1) {
        cpu_batches = 1;
        gpu_batches--;
    }
    if (processor->config.enable_gpu_acceleration && gpu_batches == 0 && batch_count > 1) {
        gpu_batches = 1;
        cpu_batches--;
    }
    
    // Create synchronization points for batch completion tracking
    ThreadSyncPoint* batch_sync_points = (ThreadSyncPoint*)malloc(batch_count * sizeof(ThreadSyncPoint));
    if (!batch_sync_points) {
        return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    for (uint32_t i = 0; i < batch_count; i++) {
        init_sync_point(&batch_sync_points[i]);
    }
    
    // Submit CPU batches to worker threads
    for (uint32_t i = 0; i < cpu_batches && i < processor->active_cpu_workers; i++) {
        CPUWorkerThread* worker = &processor->cpu_workers[i];
        
        // Assign work to worker
        // (In a full implementation, this would involve setting up work queue items)
        
        // Signal worker to start processing
        signal_sync_point(worker->sync_point, false);
    }
    
    // Submit GPU batches
    for (uint32_t i = 0; i < gpu_batches && i < processor->max_gpu_contexts; i++) {
        GPUCommandContext* context = &processor->gpu_contexts[i];
        
        if (!context->is_active) {
            context->is_active = true;
            context->is_completed = false;
            context->submission_time = get_current_time_microseconds();
            
            // Submit GPU work
            // (In a full implementation, this would set up Metal compute commands)
            
            processor->active_gpu_contexts++;
        }
    }
    
    // Wait for all batches to complete
    bool all_completed = false;
    uint32_t completed_count = 0;
    
    while (!all_completed && completed_count < batch_count) {
        usleep(1000); // 1ms polling interval
        
        completed_count = 0;
        
        // Check CPU worker completion
        for (uint32_t i = 0; i < cpu_batches && i < processor->active_cpu_workers; i++) {
            CPUWorkerThread* worker = &processor->cpu_workers[i];
            if (worker->sync_point->completed) {
                completed_count++;
            }
        }
        
        // Check GPU completion
        for (uint32_t i = 0; i < gpu_batches && i < processor->max_gpu_contexts; i++) {
            GPUCommandContext* context = &processor->gpu_contexts[i];
            if (context->is_completed) {
                completed_count++;
            }
        }
        
        all_completed = (completed_count >= batch_count);
    }
    
    uint64_t end_time = get_current_time_microseconds();
    
    // Update performance metrics
    processor->metrics.total_processing_time_us += (end_time - start_time);
    
    // Calculate parallel efficiency
    uint64_t sequential_estimate = (end_time - start_time) * batch_count;
    uint64_t parallel_time = end_time - start_time;
    if (parallel_time > 0) {
        processor->metrics.parallel_efficiency = 
            fminf(1.0f, (float)sequential_estimate / (parallel_time * batch_count));
    }
    
    // Clean up synchronization points
    for (uint32_t i = 0; i < batch_count; i++) {
        destroy_sync_point(&batch_sync_points[i]);
    }
    free(batch_sync_points);
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_optimize_batch_size(ParallelProcessor* processor,
                                                             size_t total_data_size,
                                                             size_t element_size,
                                                             float compute_intensity,
                                                             uint32_t* optimal_batch_size,
                                                             WorkloadAssignment* recommended_split) {
    if (!processor || !optimal_batch_size || !recommended_split || 
        total_data_size == 0 || element_size == 0) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    size_t total_elements = total_data_size / element_size;
    size_t cache_friendly_size = 0;
    
    // Get cache-friendly batch size from cache optimizer
    CacheOptimizerError cache_error = cache_optimizer_calculate_aligned_size(
        element_size * 32, // Base batch of 32 elements
        64 // Cache line size
    );
    cache_friendly_size = 32; // Fallback if cache optimizer unavailable
    
    // Calculate optimal batch size based on multiple factors
    uint32_t cpu_optimal = processor->system_info.total_cpu_cores * 4; // 4 batches per core
    uint32_t gpu_optimal = processor->system_info.gpu_compute_units / 4; // Conservative GPU batching
    uint32_t memory_optimal = (uint32_t)(processor->system_info.unified_memory_size_gb * 1024 / 
                                         (element_size * 8)); // Memory-constrained batching
    
    // Choose optimal based on compute intensity
    if (compute_intensity >= 2.0f) {
        // Compute-intensive: favor larger batches for GPU
        *optimal_batch_size = fminf(gpu_optimal * 4, memory_optimal);
        recommended_split->gpu_workload_fraction = 0.7f;
        recommended_split->cpu_workload_fraction = 0.3f;
    } else if (compute_intensity >= 1.0f) {
        // Balanced: use moderate batch sizes
        *optimal_batch_size = fminf((cpu_optimal + gpu_optimal) / 2, memory_optimal);
        recommended_split->gpu_workload_fraction = 0.5f;
        recommended_split->cpu_workload_fraction = 0.5f;
    } else {
        // Memory-intensive: favor smaller, cache-friendly batches
        *optimal_batch_size = fminf(cpu_optimal, (uint32_t)cache_friendly_size);
        recommended_split->gpu_workload_fraction = 0.3f;
        recommended_split->cpu_workload_fraction = 0.7f;
    }
    
    // Ensure reasonable bounds
    *optimal_batch_size = fmaxf(PARALLEL_PROCESSOR_MIN_BATCH_SIZE, 
                               fminf(*optimal_batch_size, PARALLEL_PROCESSOR_MAX_BATCH_SIZE));
    
    // Set remaining workload assignment parameters
    recommended_split->cpu_thread_count = 
        (uint32_t)(processor->system_info.total_cpu_cores * recommended_split->cpu_workload_fraction);
    recommended_split->gpu_compute_units_used = 
        (uint32_t)(processor->system_info.gpu_compute_units * recommended_split->gpu_workload_fraction);
    
    // Estimate memory requirements
    size_t batch_memory_mb = (*optimal_batch_size * element_size) / (1024 * 1024);
    recommended_split->cpu_memory_allocated_mb = 
        (size_t)(batch_memory_mb * recommended_split->cpu_workload_fraction * recommended_split->cpu_thread_count);
    recommended_split->gpu_memory_allocated_mb = 
        (size_t)(batch_memory_mb * recommended_split->gpu_workload_fraction * 4); // GPU needs buffer space
    
    recommended_split->requires_cpu_gpu_synchronization = 
        (recommended_split->cpu_workload_fraction > 0.0f && recommended_split->gpu_workload_fraction > 0.0f);
    recommended_split->synchronization_points = recommended_split->requires_cpu_gpu_synchronization ? 2 : 0;
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

// Layer Pipeline Processing

ParallelProcessorError parallel_processor_configure_pipeline(ParallelProcessor* processor,
                                                            const PipelineStageConfig* pipeline_stages,
                                                            uint32_t stage_count) {
    if (!processor || !pipeline_stages || stage_count == 0 || 
        stage_count > PARALLEL_PROCESSOR_MAX_PIPELINE_DEPTH) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    // Free existing pipeline stages
    if (processor->pipeline_stages) {
        for (uint32_t i = 0; i < processor->pipeline_stage_count; i++) {
            destroy_sync_point(&processor->pipeline_stages[i].sync_point);
        }
        free(processor->pipeline_stages);
    }
    
    // Allocate new pipeline stages
    processor->pipeline_stage_count = stage_count;
    processor->pipeline_stages = (PipelineStageContext*)calloc(stage_count, sizeof(PipelineStageContext));
    if (!processor->pipeline_stages) {
        return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize pipeline stages
    for (uint32_t i = 0; i < stage_count; i++) {
        PipelineStageContext* stage = &processor->pipeline_stages[i];
        stage->stage_id = pipeline_stages[i].stage_id;
        stage->config = pipeline_stages[i];
        stage->is_executing = false;
        stage->is_completed = false;
        init_sync_point(&stage->sync_point);
    }
    
    // Validate pipeline dependencies
    for (uint32_t i = 0; i < stage_count; i++) {
        const PipelineStageConfig* stage = &pipeline_stages[i];
        for (uint32_t j = 0; j < stage->input_dependencies; j++) {
            bool dependency_found = false;
            for (uint32_t k = 0; k < stage_count; k++) {
                if (pipeline_stages[k].stage_id == stage->dependency_stage_ids[j]) {
                    dependency_found = true;
                    break;
                }
            }
            if (!dependency_found) {
                return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
            }
        }
    }
    
    processor->pipeline_depth = stage_count;
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_execute_pipeline(ParallelProcessor* processor,
                                                          const void* input_data,
                                                          void** output_data,
                                                          size_t data_size,
                                                          bool enable_overlapping) {
    if (!processor || !input_data || !output_data || data_size == 0 || 
        !processor->pipeline_stages || processor->pipeline_stage_count == 0) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    uint64_t pipeline_start_time = get_current_time_microseconds();
    
    // Allocate intermediate data buffers
    void** stage_buffers = (void**)malloc(processor->pipeline_stage_count * sizeof(void*));
    if (!stage_buffers) {
        return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    for (uint32_t i = 0; i < processor->pipeline_stage_count; i++) {
        stage_buffers[i] = malloc(data_size);
        if (!stage_buffers[i]) {
            // Clean up allocated buffers
            for (uint32_t j = 0; j < i; j++) {
                free(stage_buffers[j]);
            }
            free(stage_buffers);
            return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    // Set up first stage input
    stage_buffers[0] = (void*)input_data;
    
    if (enable_overlapping) {
        // Execute pipeline with overlapping stages
        bool pipeline_active = true;
        uint32_t completed_stages = 0;
        
        while (pipeline_active && completed_stages < processor->pipeline_stage_count) {
            for (uint32_t i = 0; i < processor->pipeline_stage_count; i++) {
                PipelineStageContext* stage = &processor->pipeline_stages[i];
                
                if (stage->is_completed) {
                    continue;
                }
                
                // Check if dependencies are satisfied
                bool dependencies_satisfied = true;
                for (uint32_t j = 0; j < stage->config.input_dependencies; j++) {
                    uint32_t dep_id = stage->config.dependency_stage_ids[j];
                    bool dep_completed = false;
                    
                    for (uint32_t k = 0; k < processor->pipeline_stage_count; k++) {
                        if (processor->pipeline_stages[k].stage_id == dep_id && 
                            processor->pipeline_stages[k].is_completed) {
                            dep_completed = true;
                            break;
                        }
                    }
                    
                    if (!dep_completed) {
                        dependencies_satisfied = false;
                        break;
                    }
                }
                
                if (dependencies_satisfied && !stage->is_executing) {
                    // Start stage execution
                    stage->is_executing = true;
                    stage->start_time = get_current_time_microseconds();
                    stage->input_data = (i == 0) ? (void*)input_data : stage_buffers[i - 1];
                    stage->output_data = (i == processor->pipeline_stage_count - 1) ? 
                                        stage_buffers[i] : stage_buffers[i + 1];
                    stage->data_size = data_size;
                    
                    // Execute stage (simplified - would dispatch to GPU or CPU)
                    if (stage->config.is_gpu_accelerated) {
                        // GPU execution
                        stage->end_time = get_current_time_microseconds() + 1000; // Simulate GPU delay
                    } else {
                        // CPU execution
                        memcpy(stage->output_data, stage->input_data, data_size);
                        stage->end_time = get_current_time_microseconds();
                    }
                    
                    stage->is_executing = false;
                    stage->is_completed = true;
                    completed_stages++;
                    
                    signal_sync_point(&stage->sync_point, true);
                }
            }
            
            pipeline_active = (completed_stages < processor->pipeline_stage_count);
            usleep(100); // Small delay to prevent busy waiting
        }
    } else {
        // Execute pipeline sequentially
        for (uint32_t i = 0; i < processor->pipeline_stage_count; i++) {
            PipelineStageContext* stage = &processor->pipeline_stages[i];
            
            stage->start_time = get_current_time_microseconds();
            stage->input_data = (i == 0) ? (void*)input_data : stage_buffers[i - 1];
            stage->output_data = (i == processor->pipeline_stage_count - 1) ? 
                                stage_buffers[i] : stage_buffers[i + 1];
            stage->data_size = data_size;
            
            // Execute stage
            if (stage->config.is_gpu_accelerated) {
                // GPU execution simulation
                usleep(1000); // Simulate GPU computation time
            } else {
                // CPU execution
                memcpy(stage->output_data, stage->input_data, data_size);
            }
            
            stage->end_time = get_current_time_microseconds();
            stage->is_completed = true;
            signal_sync_point(&stage->sync_point, true);
        }
    }
    
    // Set output data to final stage result
    *output_data = stage_buffers[processor->pipeline_stage_count - 1];
    
    uint64_t pipeline_end_time = get_current_time_microseconds();
    
    // Update performance metrics
    processor->metrics.total_processing_time_us += (pipeline_end_time - pipeline_start_time);
    
    // Clean up intermediate buffers (except input and output)
    for (uint32_t i = 1; i < processor->pipeline_stage_count - 1; i++) {
        free(stage_buffers[i]);
    }
    free(stage_buffers);
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_analyze_pipeline_bottlenecks(ParallelProcessor* processor,
                                                                       const ParallelProcessingMetrics* stage_performance,
                                                                       uint32_t stage_count,
                                                                       uint32_t* bottleneck_stage_id,
                                                                       char* optimization_suggestions) {
    if (!processor || !stage_performance || !bottleneck_stage_id || !optimization_suggestions ||
        stage_count == 0) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    *bottleneck_stage_id = 0;
    float max_processing_time = 0.0f;
    float total_pipeline_time = 0.0f;
    
    // Find the stage with maximum processing time
    for (uint32_t i = 0; i < stage_count; i++) {
        float stage_time = (float)stage_performance[i].total_processing_time_us;
        total_pipeline_time += stage_time;
        
        if (stage_time > max_processing_time) {
            max_processing_time = stage_time;
            *bottleneck_stage_id = i;
        }
    }
    
    // Analyze bottleneck characteristics
    const ParallelProcessingMetrics* bottleneck = &stage_performance[*bottleneck_stage_id];
    float bottleneck_ratio = max_processing_time / total_pipeline_time;
    
    // Generate optimization suggestions
    char* suggestions = optimization_suggestions;
    int written = 0;
    
    written += sprintf(suggestions + written, "Pipeline Bottleneck Analysis:\n");
    written += sprintf(suggestions + written, "- Bottleneck Stage: %u (%.1f%% of pipeline time)\n", 
                      *bottleneck_stage_id, bottleneck_ratio * 100.0f);
    
    if (bottleneck->cpu_utilization_percent < 50.0f && bottleneck->gpu_utilization_percent < 50.0f) {
        written += sprintf(suggestions + written, "- Low utilization detected. Consider increasing parallelism.\n");
    }
    
    if (bottleneck->memory_bandwidth_utilization > 0.8f) {
        written += sprintf(suggestions + written, "- Memory bandwidth bottleneck. Consider data compression or tiling.\n");
    }
    
    if (bottleneck->cache_misses > bottleneck->total_memory_accesses * 0.2f) {
        written += sprintf(suggestions + written, "- High cache miss rate. Consider cache-friendly data layouts.\n");
    }
    
    if (bottleneck->synchronization_time_us > bottleneck->total_processing_time_us * 0.1f) {
        written += sprintf(suggestions + written, "- High synchronization overhead. Consider reducing sync points.\n");
    }
    
    if (bottleneck_ratio > 0.5f) {
        written += sprintf(suggestions + written, "- Consider splitting bottleneck stage into multiple parallel sub-stages.\n");
    }
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

// Asynchronous GPU Computation

ParallelProcessorError parallel_processor_submit_async_gpu(ParallelProcessor* processor,
                                                          void* computation_function,
                                                          const void* input_data,
                                                          size_t input_size,
                                                          void* output_data,
                                                          size_t output_size,
                                                          AsyncExecutionContext* execution_context) {
    if (!processor || !computation_function || !input_data || !output_data || 
        !execution_context || input_size == 0 || output_size == 0) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&processor->async_mutex);
    
    // Find available async context
    uint32_t context_index = UINT32_MAX;
    for (uint32_t i = 0; i < processor->max_async_contexts; i++) {
        if (!processor->async_contexts[i].is_active) {
            context_index = i;
            break;
        }
    }
    
    if (context_index == UINT32_MAX) {
        pthread_mutex_unlock(&processor->async_mutex);
        return PARALLEL_PROCESSOR_ERROR_INSUFFICIENT_MEMORY;
    }
    
    // Initialize execution context
    AsyncExecutionContext* ctx = &processor->async_contexts[context_index];
    ctx->context_id = context_index;
    ctx->is_active = true;
    ctx->is_gpu_context = true;
    ctx->start_time_us = get_current_time_microseconds();
    ctx->timeout_ms = 10000; // 10 second default timeout
    ctx->progress_percent = 0.0f;
    ctx->completion_callback = NULL;
    ctx->user_data = NULL;
    
    processor->active_async_contexts++;
    
    pthread_mutex_unlock(&processor->async_mutex);
    
    // Submit GPU computation (simplified implementation)
    // In a full implementation, this would create Metal compute commands
    
#ifdef __OBJC__
    @autoreleasepool {
        // Find available GPU command context
        GPUCommandContext* gpu_context = NULL;
        for (uint32_t i = 0; i < processor->max_gpu_contexts; i++) {
            if (!processor->gpu_contexts[i].is_active) {
                gpu_context = &processor->gpu_contexts[i];
                break;
            }
        }
        
        if (gpu_context) {
            gpu_context->is_active = true;
            gpu_context->is_completed = false;
            gpu_context->submission_time = get_current_time_microseconds();
            gpu_context->memory_usage_bytes = input_size + output_size;
            
            // Create Metal command buffer (simplified)
            MetalDeviceCapabilities caps;
            metal_compute_get_device_capabilities(processor->metal_accelerator, &caps);
            
            // Simulate async GPU work
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
                // Simulate GPU computation time
                usleep(5000); // 5ms simulation
                
                // Mark as completed
                gpu_context->is_completed = true;
                gpu_context->completion_time = get_current_time_microseconds();
                ctx->progress_percent = 100.0f;
                
                // Copy results (simplified)
                memcpy(output_data, input_data, fminf(input_size, output_size));
            });
            
            processor->active_gpu_contexts++;
        }
    }
#else
    // CPU fallback for non-Objective-C compilation
    memcpy(output_data, input_data, fminf(input_size, output_size));
    ctx->progress_percent = 100.0f;
#endif
    
    // Copy context to output
    *execution_context = *ctx;
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_check_async_status(ParallelProcessor* processor,
                                                           const AsyncExecutionContext* execution_context,
                                                           bool* is_complete,
                                                           float* progress_percent) {
    if (!processor || !execution_context || !is_complete || !progress_percent) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    if (execution_context->context_id >= processor->max_async_contexts) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    AsyncExecutionContext* ctx = &processor->async_contexts[execution_context->context_id];
    
    *is_complete = (ctx->progress_percent >= 100.0f);
    *progress_percent = ctx->progress_percent;
    
    // Check for timeout
    uint64_t current_time = get_current_time_microseconds();
    if ((current_time - ctx->start_time_us) > (ctx->timeout_ms * 1000)) {
        *is_complete = true; // Mark as complete due to timeout
        return PARALLEL_PROCESSOR_ERROR_EXECUTION_TIMEOUT;
    }
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_wait_async_completion(ParallelProcessor* processor,
                                                              const AsyncExecutionContext* execution_context,
                                                              uint32_t timeout_ms) {
    if (!processor || !execution_context) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    uint64_t wait_start = get_current_time_microseconds();
    uint64_t timeout_us = timeout_ms * 1000;
    
    bool is_complete = false;
    float progress = 0.0f;
    
    while (!is_complete) {
        ParallelProcessorError status = parallel_processor_check_async_status(processor, execution_context,
                                                                              &is_complete, &progress);
        if (status != PARALLEL_PROCESSOR_SUCCESS) {
            return status;
        }
        
        if (is_complete) break;
        
        // Check timeout
        if (timeout_ms > 0) {
            uint64_t elapsed = get_current_time_microseconds() - wait_start;
            if (elapsed > timeout_us) {
                return PARALLEL_PROCESSOR_ERROR_SYNCHRONIZATION_TIMEOUT;
            }
        }
        
        usleep(1000); // 1ms polling interval
    }
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_cancel_async(ParallelProcessor* processor,
                                                      AsyncExecutionContext* execution_context) {
    if (!processor || !execution_context) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    if (execution_context->context_id >= processor->max_async_contexts) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&processor->async_mutex);
    
    AsyncExecutionContext* ctx = &processor->async_contexts[execution_context->context_id];
    if (ctx->is_active) {
        ctx->is_active = false;
        ctx->progress_percent = 0.0f;
        processor->active_async_contexts--;
    }
    
    pthread_mutex_unlock(&processor->async_mutex);
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

// CPU-GPU Load Balancing

ParallelProcessorError parallel_processor_analyze_system_load(ParallelProcessor* processor,
                                                             LoadBalancingState* load_balancing_state) {
    if (!processor || !load_balancing_state) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    // Get current system load metrics
    processor->load_balancing.current_cpu_load = 
        processor->metrics.cpu_utilization_percent / 100.0f;
    processor->load_balancing.current_gpu_load = 
        processor->metrics.gpu_utilization_percent / 100.0f;
    
    // Calculate memory pressure
    size_t used_memory_mb = processor->memory_pool_used / (1024 * 1024);
    size_t total_memory_mb = processor->memory_pool_size / (1024 * 1024);
    processor->load_balancing.current_memory_pressure = 
        (float)used_memory_mb / total_memory_mb;
    
    // Update thermal state
    processor->load_balancing.current_thermal_state = 
        processor->current_temperature / THERMAL_THROTTLE_TEMPERATURE;
    
    // Determine if rebalancing is needed
    uint64_t current_time = get_current_time_microseconds();
    uint64_t time_since_last_rebalance = 
        current_time - processor->load_balancing.last_rebalance_time_us;
    
    bool significant_load_imbalance = 
        fabs(processor->load_balancing.current_cpu_load - processor->load_balancing.current_gpu_load) > 0.3f;
    bool high_memory_pressure = 
        processor->load_balancing.current_memory_pressure > MEMORY_PRESSURE_HIGH_THRESHOLD;
    bool thermal_issues = 
        processor->load_balancing.current_thermal_state > 0.9f;
    bool time_for_rebalance = 
        time_since_last_rebalance > 5000000; // 5 seconds
    
    processor->load_balancing.needs_rebalancing = 
        (significant_load_imbalance || high_memory_pressure || thermal_issues) && time_for_rebalance;
    
    // Calculate performance trend
    static float last_efficiency = 0.5f;
    float current_efficiency = processor->metrics.parallel_efficiency;
    processor->load_balancing.performance_trend = current_efficiency - last_efficiency;
    last_efficiency = current_efficiency;
    
    *load_balancing_state = processor->load_balancing;
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_calculate_optimal_distribution(ParallelProcessor* processor,
                                                                        const void* workload_characteristics,
                                                                        const LoadBalancingState* current_system_state,
                                                                        WorkloadAssignment* optimal_assignment) {
    if (!processor || !current_system_state || !optimal_assignment) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    // Start with current assignment
    *optimal_assignment = current_system_state->optimal_assignment;
    
    // Adjust based on current load
    if (current_system_state->current_cpu_load > 0.8f && current_system_state->current_gpu_load < 0.5f) {
        // CPU overloaded, shift work to GPU
        optimal_assignment->gpu_workload_fraction += 0.1f;
        optimal_assignment->cpu_workload_fraction -= 0.1f;
    } else if (current_system_state->current_gpu_load > 0.8f && current_system_state->current_cpu_load < 0.5f) {
        // GPU overloaded, shift work to CPU
        optimal_assignment->cpu_workload_fraction += 0.1f;
        optimal_assignment->gpu_workload_fraction -= 0.1f;
    }
    
    // Adjust for thermal throttling
    if (current_system_state->current_thermal_state > 0.9f) {
        // Reduce overall intensity to manage thermal state
        optimal_assignment->cpu_workload_fraction *= 0.8f;
        optimal_assignment->gpu_workload_fraction *= 0.8f;
    }
    
    // Adjust for memory pressure
    if (current_system_state->current_memory_pressure > MEMORY_PRESSURE_HIGH_THRESHOLD) {
        // Favor CPU processing to reduce GPU memory pressure
        optimal_assignment->cpu_workload_fraction += 0.05f;
        optimal_assignment->gpu_workload_fraction -= 0.05f;
    }
    
    // Ensure fractions sum to 1.0 and stay within bounds
    float total_fraction = optimal_assignment->cpu_workload_fraction + optimal_assignment->gpu_workload_fraction;
    if (total_fraction > 0.0f) {
        optimal_assignment->cpu_workload_fraction /= total_fraction;
        optimal_assignment->gpu_workload_fraction /= total_fraction;
    }
    
    // Clamp fractions to reasonable bounds
    optimal_assignment->cpu_workload_fraction = 
        fmaxf(0.1f, fminf(0.9f, optimal_assignment->cpu_workload_fraction));
    optimal_assignment->gpu_workload_fraction = 
        fmaxf(0.1f, fminf(0.9f, optimal_assignment->gpu_workload_fraction));
    
    // Update resource allocations
    optimal_assignment->cpu_thread_count = 
        (uint32_t)(processor->system_info.performance_cpu_cores * optimal_assignment->cpu_workload_fraction);
    optimal_assignment->gpu_compute_units_used = 
        (uint32_t)(processor->system_info.gpu_compute_units * optimal_assignment->gpu_workload_fraction);
    
    // Determine synchronization needs
    optimal_assignment->requires_cpu_gpu_synchronization = 
        (optimal_assignment->cpu_workload_fraction > 0.1f && optimal_assignment->gpu_workload_fraction > 0.1f);
    optimal_assignment->synchronization_points = 
        optimal_assignment->requires_cpu_gpu_synchronization ? 
        (uint32_t)(2.0f / fminf(optimal_assignment->cpu_workload_fraction, optimal_assignment->gpu_workload_fraction)) : 0;
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_execute_load_balanced(ParallelProcessor* processor,
                                                               void* cpu_workload_function,
                                                               void* gpu_workload_function,
                                                               const void* input_data,
                                                               void* output_data,
                                                               size_t data_size,
                                                               WorkloadDistributionPattern distribution_pattern) {
    if (!processor || !input_data || !output_data || data_size == 0) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    // Get current optimal assignment
    LoadBalancingState current_state;
    ParallelProcessorError error = parallel_processor_analyze_system_load(processor, &current_state);
    if (error != PARALLEL_PROCESSOR_SUCCESS) {
        return error;
    }
    
    WorkloadAssignment optimal_assignment;
    error = parallel_processor_calculate_optimal_distribution(processor, NULL, 
                                                            &current_state, &optimal_assignment);
    if (error != PARALLEL_PROCESSOR_SUCCESS) {
        return error;
    }
    
    uint64_t execution_start = get_current_time_microseconds();
    
    // Split data according to optimal assignment
    size_t cpu_data_size = (size_t)(data_size * optimal_assignment.cpu_workload_fraction);
    size_t gpu_data_size = data_size - cpu_data_size;
    
    // Allocate output buffers
    void* cpu_output = malloc(cpu_data_size);
    void* gpu_output = malloc(gpu_data_size);
    
    if (!cpu_output || !gpu_output) {
        free(cpu_output);
        free(gpu_output);
        return PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Execute CPU portion
    ThreadSyncPoint cpu_sync, gpu_sync;
    init_sync_point(&cpu_sync);
    init_sync_point(&gpu_sync);
    
    // Launch CPU work
    if (cpu_data_size > 0 && cpu_workload_function) {
        // In a full implementation, this would dispatch CPU work
        memcpy(cpu_output, input_data, cpu_data_size);
        signal_sync_point(&cpu_sync, true);
    } else {
        signal_sync_point(&cpu_sync, true);
    }
    
    // Launch GPU work
    if (gpu_data_size > 0 && gpu_workload_function) {
        // In a full implementation, this would dispatch GPU work
        const char* gpu_input = (const char*)input_data + cpu_data_size;
        memcpy(gpu_output, gpu_input, gpu_data_size);
        signal_sync_point(&gpu_sync, true);
    } else {
        signal_sync_point(&gpu_sync, true);
    }
    
    // Wait for both CPU and GPU to complete
    wait_sync_point(&cpu_sync, true);
    wait_sync_point(&gpu_sync, true);
    
    // Merge results
    if (cpu_data_size > 0) {
        memcpy(output_data, cpu_output, cpu_data_size);
    }
    if (gpu_data_size > 0) {
        memcpy((char*)output_data + cpu_data_size, gpu_output, gpu_data_size);
    }
    
    uint64_t execution_end = get_current_time_microseconds();
    
    // Update performance metrics
    processor->metrics.total_processing_time_us += (execution_end - execution_start);
    processor->metrics.cpu_processing_time_us += (execution_end - execution_start) / 2; // Approximation
    processor->metrics.gpu_processing_time_us += (execution_end - execution_start) / 2; // Approximation
    
    // Clean up
    destroy_sync_point(&cpu_sync);
    destroy_sync_point(&gpu_sync);
    free(cpu_output);
    free(gpu_output);
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_adjust_load_balancing(ParallelProcessor* processor,
                                                               const ParallelProcessingMetrics* current_metrics,
                                                               float adjustment_aggressiveness,
                                                               WorkloadAssignment* new_assignment) {
    if (!processor || !current_metrics || !new_assignment || 
        adjustment_aggressiveness < 0.0f || adjustment_aggressiveness > 1.0f) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    *new_assignment = processor->load_balancing.optimal_assignment;
    
    // Calculate efficiency metrics
    float cpu_efficiency = current_metrics->cpu_utilization_percent / 100.0f;
    float gpu_efficiency = current_metrics->gpu_utilization_percent / 100.0f;
    float overall_efficiency = current_metrics->parallel_efficiency;
    
    // Adjust based on utilization imbalance
    float efficiency_diff = cpu_efficiency - gpu_efficiency;
    float adjustment = efficiency_diff * adjustment_aggressiveness * 0.1f; // Max 10% shift
    
    if (efficiency_diff > 0.2f) {
        // CPU overutilized, shift work to GPU
        new_assignment->gpu_workload_fraction += adjustment;
        new_assignment->cpu_workload_fraction -= adjustment;
    } else if (efficiency_diff < -0.2f) {
        // GPU overutilized, shift work to CPU
        new_assignment->cpu_workload_fraction += adjustment;
        new_assignment->gpu_workload_fraction -= adjustment;
    }
    
    // Adjust based on overall efficiency
    if (overall_efficiency < PARALLEL_EFFICIENCY_ACCEPTABLE) {
        // Poor parallel efficiency, try more balanced approach
        float target_cpu_fraction = 0.5f;
        float target_gpu_fraction = 0.5f;
        
        new_assignment->cpu_workload_fraction = 
            new_assignment->cpu_workload_fraction * (1.0f - adjustment_aggressiveness) +
            target_cpu_fraction * adjustment_aggressiveness;
        new_assignment->gpu_workload_fraction = 
            new_assignment->gpu_workload_fraction * (1.0f - adjustment_aggressiveness) +
            target_gpu_fraction * adjustment_aggressiveness;
    }
    
    // Ensure valid fractions
    float total_fraction = new_assignment->cpu_workload_fraction + new_assignment->gpu_workload_fraction;
    if (total_fraction > 0.0f) {
        new_assignment->cpu_workload_fraction /= total_fraction;
        new_assignment->gpu_workload_fraction /= total_fraction;
    }
    
    // Update processor's optimal assignment
    processor->load_balancing.optimal_assignment = *new_assignment;
    processor->load_balancing.last_rebalance_time_us = get_current_time_microseconds();
    processor->load_balancing.needs_rebalancing = false;
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

// Performance Monitoring

ParallelProcessorError parallel_processor_start_monitoring(ParallelProcessor* processor,
                                                          uint32_t monitoring_interval_ms) {
    if (!processor || monitoring_interval_ms == 0) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    if (processor->monitoring_active) {
        return PARALLEL_PROCESSOR_SUCCESS; // Already monitoring
    }
    
    processor->monitoring_active = true;
    processor->monitoring_start_time = get_current_time_microseconds();
    processor->monitoring_interval_ms = monitoring_interval_ms;
    
    // Reset performance metrics
    memset(&processor->metrics, 0, sizeof(ParallelProcessingMetrics));
    processor->metrics.parallel_efficiency = 0.5f;
    processor->metrics.power_efficiency_score = 0.7f;
    
    // Start monitoring thread
    int thread_result = pthread_create(&processor->monitoring_thread, NULL, 
                                     monitoring_thread_function, processor);
    if (thread_result != 0) {
        processor->monitoring_active = false;
        return PARALLEL_PROCESSOR_ERROR_THREAD_CREATION;
    }
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_get_performance_metrics(ParallelProcessor* processor,
                                                                 ParallelProcessingMetrics* metrics) {
    if (!processor || !metrics) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    *metrics = processor->metrics;
    
    // Calculate derived metrics if monitoring is active
    if (processor->monitoring_active) {
        uint64_t current_time = get_current_time_microseconds();
        uint64_t monitoring_duration = current_time - processor->monitoring_start_time;
        
        // Update utilization percentages
        if (monitoring_duration > 0) {
            metrics->cpu_utilization_percent = 
                fminf(100.0f, (float)processor->metrics.cpu_processing_time_us / monitoring_duration * 100.0f);
            metrics->gpu_utilization_percent = 
                fminf(100.0f, (float)processor->metrics.gpu_processing_time_us / monitoring_duration * 100.0f);
        }
        
        // Calculate memory bandwidth utilization
        if (processor->metrics.total_processing_time_us > 0) {
            float memory_transfer_mbps = 
                (float)processor->metrics.memory_transfer_time_us / processor->metrics.total_processing_time_us;
            metrics->memory_bandwidth_utilization = 
                fminf(1.0f, memory_transfer_mbps / processor->system_info.memory_bandwidth_gbps);
        }
        
        // Update parallel efficiency
        if (processor->metrics.total_processing_time_us > 0) {
            float sequential_time_estimate = 
                processor->metrics.cpu_processing_time_us + processor->metrics.gpu_processing_time_us;
            if (sequential_time_estimate > 0) {
                metrics->parallel_efficiency = 
                    fminf(1.0f, sequential_time_estimate / processor->metrics.total_processing_time_us);
            }
        }
        
        // Update power efficiency score
        float thermal_penalty = fmaxf(0.0f, processor->current_temperature - 70.0f) / 20.0f; // 70-90°C range
        metrics->power_efficiency_score = 
            fmaxf(0.0f, 1.0f - thermal_penalty) * metrics->parallel_efficiency;
    }
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_stop_monitoring(ParallelProcessor* processor,
                                                         ParallelProcessingMetrics* final_metrics) {
    if (!processor) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    if (!processor->monitoring_active) {
        return PARALLEL_PROCESSOR_SUCCESS; // Not monitoring
    }
    
    // Stop monitoring
    processor->monitoring_active = false;
    
    // Wait for monitoring thread to complete
    pthread_join(processor->monitoring_thread, NULL);
    
    // Get final metrics
    if (final_metrics) {
        parallel_processor_get_performance_metrics(processor, final_metrics);
    }
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

void parallel_processor_reset_counters(ParallelProcessor* processor) {
    if (!processor) {
        return;
    }
    
    memset(&processor->metrics, 0, sizeof(ParallelProcessingMetrics));
    processor->metrics.parallel_efficiency = 0.5f;
    processor->metrics.power_efficiency_score = 0.7f;
    
    processor->load_balancing.recent_task_count = 0;
    processor->load_balancing.last_rebalance_time_us = get_current_time_microseconds();
}

// Configuration Functions

ParallelProcessorError parallel_processor_create_default_config(ParallelProcessingConfig* config) {
    if (!config) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(ParallelProcessingConfig));
    
    config->enable_cpu_parallelization = true;
    config->enable_gpu_acceleration = true;
    config->enable_async_execution = true;
    config->enable_pipeline_processing = true;
    config->enable_batch_parallelization = true;
    config->enable_dynamic_load_balancing = true;
    config->enable_numa_awareness = false; // Not typically needed on Apple Silicon
    config->prefer_performance_cores = true;
    config->max_cpu_threads = get_cpu_core_count();
    config->max_gpu_command_buffers = 8;
    config->pipeline_depth = PARALLEL_PROCESSOR_DEFAULT_PIPELINE_DEPTH;
    config->cpu_gpu_balance_ratio = CPU_GPU_BALANCE_RATIO_BALANCED;
    config->batch_size_threshold = PARALLEL_PROCESSOR_DEFAULT_BATCH_SIZE;
    config->memory_pressure_threshold = 80; // 80%
    config->thermal_throttle_threshold = (uint32_t)THERMAL_THROTTLE_TEMPERATURE;
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_create_apple_silicon_config(ParallelProcessingConfig* config,
                                                                      const AppleSiliconSystemInfo* system_info) {
    if (!config || !system_info) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    parallel_processor_create_default_config(config);
    
    // Apple Silicon specific optimizations
    config->max_cpu_threads = system_info->performance_cpu_cores + system_info->efficiency_cpu_cores / 2;
    config->prefer_performance_cores = true;
    config->cpu_gpu_balance_ratio = 0.6f; // Slightly favor GPU on Apple Silicon
    config->enable_numa_awareness = false; // Unified memory architecture
    config->memory_pressure_threshold = 85; // Higher threshold due to unified memory
    
    // Adjust based on memory size
    if (system_info->unified_memory_size_gb >= 32) {
        config->batch_size_threshold = PARALLEL_PROCESSOR_DEFAULT_BATCH_SIZE * 2;
        config->max_gpu_command_buffers = 16;
    } else if (system_info->unified_memory_size_gb <= 8) {
        config->batch_size_threshold = PARALLEL_PROCESSOR_DEFAULT_BATCH_SIZE / 2;
        config->max_gpu_command_buffers = 4;
    }
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

void parallel_processor_destroy(ParallelProcessor* processor) {
    if (!processor) {
        return;
    }
    
    // Stop monitoring if active
    if (processor->monitoring_active) {
        parallel_processor_stop_monitoring(processor, NULL);
    }
    
    // Stop and clean up CPU worker threads
    for (uint32_t i = 0; i < processor->active_cpu_workers; i++) {
        CPUWorkerThread* worker = &processor->cpu_workers[i];
        worker->is_active = false;
        signal_sync_point(worker->sync_point, false);
        pthread_join(worker->thread, NULL);
        destroy_sync_point(worker->sync_point);
        free(worker->sync_point);
    }
    
    // Clean up pipeline stages
    if (processor->pipeline_stages) {
        for (uint32_t i = 0; i < processor->pipeline_stage_count; i++) {
            destroy_sync_point(&processor->pipeline_stages[i].sync_point);
        }
        free(processor->pipeline_stages);
    }
    
    // Destroy synchronization primitives
    pthread_mutex_destroy(&processor->cpu_work_mutex);
    pthread_mutex_destroy(&processor->async_mutex);
    pthread_mutex_destroy(&processor->memory_mutex);
    
    // Free allocated arrays
    free(processor->cpu_workers);
    free(processor->gpu_contexts);
    free(processor->async_contexts);
    free(processor->shared_memory_pool);
    
    // Free main structure
    free(processor);
}

// Utility Functions

const char* parallel_processor_get_error_string(ParallelProcessorError error_code) {
    switch (error_code) {
        case PARALLEL_PROCESSOR_SUCCESS:
            return "Operation completed successfully";
        case PARALLEL_PROCESSOR_ERROR_INVALID_PARAM:
            return "Invalid parameter provided";
        case PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case PARALLEL_PROCESSOR_ERROR_THREAD_CREATION:
            return "Thread creation failed";
        case PARALLEL_PROCESSOR_ERROR_GPU_INITIALIZATION:
            return "GPU initialization failed";
        case PARALLEL_PROCESSOR_ERROR_PIPELINE_STALL:
            return "Pipeline stall detected";
        case PARALLEL_PROCESSOR_ERROR_LOAD_BALANCING_FAILED:
            return "Load balancing failed";
        case PARALLEL_PROCESSOR_ERROR_SYNCHRONIZATION_TIMEOUT:
            return "Synchronization timeout";
        case PARALLEL_PROCESSOR_ERROR_INSUFFICIENT_MEMORY:
            return "Insufficient memory for operation";
        case PARALLEL_PROCESSOR_ERROR_THERMAL_THROTTLE:
            return "Thermal throttling active";
        case PARALLEL_PROCESSOR_ERROR_EXECUTION_TIMEOUT:
            return "Execution timeout";
        case PARALLEL_PROCESSOR_ERROR_INVALID_WORKLOAD:
            return "Invalid workload specification";
        case PARALLEL_PROCESSOR_ERROR_HARDWARE_INCOMPATIBLE:
            return "Hardware incompatible with parallel processing";
        default:
            return "Unknown error";
    }
}

const char* parallel_processor_get_strategy_string(ParallelProcessingStrategy strategy) {
    switch (strategy) {
        case PARALLEL_STRATEGY_CPU_ONLY:
            return "CPU-Only Processing";
        case PARALLEL_STRATEGY_GPU_ONLY:
            return "GPU-Only Processing";
        case PARALLEL_STRATEGY_CPU_GPU_BALANCED:
            return "Balanced CPU-GPU Processing";
        case PARALLEL_STRATEGY_GPU_DOMINANT:
            return "GPU-Dominant Processing";
        case PARALLEL_STRATEGY_CPU_DOMINANT:
            return "CPU-Dominant Processing";
        case PARALLEL_STRATEGY_ADAPTIVE:
            return "Adaptive Strategy Selection";
        case PARALLEL_STRATEGY_PIPELINE:
            return "Pipelined Processing";
        case PARALLEL_STRATEGY_NEURAL_ENGINE:
            return "Neural Engine Acceleration";
        default:
            return "Unknown Strategy";
    }
}

const char* parallel_processor_get_distribution_pattern_string(WorkloadDistributionPattern pattern) {
    switch (pattern) {
        case WORKLOAD_DISTRIBUTION_EVEN:
            return "Even Distribution";
        case WORKLOAD_DISTRIBUTION_WEIGHTED:
            return "Weighted Distribution";
        case WORKLOAD_DISTRIBUTION_ADAPTIVE:
            return "Adaptive Distribution";
        case WORKLOAD_DISTRIBUTION_BATCH_SPLIT:
            return "Batch Dimension Split";
        case WORKLOAD_DISTRIBUTION_LAYER_SPLIT:
            return "Layer Dimension Split";
        case WORKLOAD_DISTRIBUTION_FEATURE_SPLIT:
            return "Feature Dimension Split";
        case WORKLOAD_DISTRIBUTION_HYBRID:
            return "Hybrid Distribution";
        default:
            return "Unknown Pattern";
    }
}

uint32_t parallel_processor_calculate_optimal_thread_count(const char* workload_type,
                                                          uint32_t available_cores,
                                                          float system_load) {
    if (!workload_type || available_cores == 0) {
        return 1;
    }
    
    uint32_t base_threads = available_cores;
    
    // Adjust based on workload type
    if (strcmp(workload_type, "compute_intensive") == 0) {
        // CPU-bound work: use all cores but don't oversubscribe
        base_threads = available_cores;
    } else if (strcmp(workload_type, "memory_intensive") == 0) {
        // Memory-bound work: fewer threads to reduce memory pressure
        base_threads = available_cores / 2;
    } else if (strcmp(workload_type, "io_intensive") == 0) {
        // I/O-bound work: can oversubscribe
        base_threads = available_cores * 2;
    }
    
    // Adjust for current system load
    float load_factor = 1.0f - system_load;
    base_threads = (uint32_t)(base_threads * load_factor);
    
    // Ensure minimum of 1 thread
    return fmaxf(1, base_threads);
}

ParallelProcessorError parallel_processor_calculate_memory_requirements(uint32_t batch_size,
                                                                       size_t data_element_size,
                                                                       uint32_t num_parallel_streams,
                                                                       size_t* cpu_memory_mb,
                                                                       size_t* gpu_memory_mb) {
    if (!cpu_memory_mb || !gpu_memory_mb || batch_size == 0 || 
        data_element_size == 0 || num_parallel_streams == 0) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    // Calculate per-stream memory requirement
    size_t per_stream_bytes = batch_size * data_element_size;
    
    // CPU memory: input + output + intermediate buffers
    size_t cpu_total_bytes = per_stream_bytes * num_parallel_streams * 3; // 3x for buffers
    *cpu_memory_mb = cpu_total_bytes / (1024 * 1024);
    
    // GPU memory: input + output + Metal buffers + command overhead
    size_t gpu_total_bytes = per_stream_bytes * num_parallel_streams * 4; // 4x for GPU buffers
    *gpu_memory_mb = gpu_total_bytes / (1024 * 1024);
    
    return PARALLEL_PROCESSOR_SUCCESS;
}

ParallelProcessorError parallel_processor_check_thermal_state(ParallelProcessor* processor,
                                                             float current_temperature,
                                                             bool* thermal_throttle_needed,
                                                             float* recommended_intensity) {
    if (!processor || !thermal_throttle_needed || !recommended_intensity) {
        return PARALLEL_PROCESSOR_ERROR_INVALID_PARAM;
    }
    
    processor->current_temperature = current_temperature;
    
    // Determine thermal throttling need
    *thermal_throttle_needed = (current_temperature > THERMAL_THROTTLE_TEMPERATURE);
    
    // Calculate recommended processing intensity
    if (current_temperature <= 70.0f) {
        *recommended_intensity = 1.0f; // Full intensity
    } else if (current_temperature <= 80.0f) {
        *recommended_intensity = 0.8f; // Slight reduction
    } else if (current_temperature <= 85.0f) {
        *recommended_intensity = 0.6f; // Moderate reduction
    } else if (current_temperature <= 90.0f) {
        *recommended_intensity = 0.4f; // Significant reduction
    } else {
        *recommended_intensity = 0.2f; // Emergency throttling
    }
    
    processor->thermal_throttle_active = *thermal_throttle_needed;
    
    return PARALLEL_PROCESSOR_SUCCESS;
}
