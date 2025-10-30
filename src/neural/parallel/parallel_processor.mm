#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "parallel_processor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>

// Helper functions
static uint64_t get_current_time_us(void);
static void* worker_thread_main(void* arg);
static ProcessingTask* dequeue_task(ParallelProcessor* processor);
static ParallelProcessorError enqueue_task(ParallelProcessor* processor, ProcessingTask* task);
static void update_metrics(ParallelProcessor* processor, ProcessingTask* task);
static ParallelProcessorError initialize_gpu_context(ParallelProcessor* processor);
static void cleanup_gpu_context(GPUContext* context);

ParallelProcessorError parallel_processor_create(ParallelProcessor** processor,
                                               const ParallelProcessorConfig* config) {
    if (!processor || !config) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    // Allocate processor
    ParallelProcessor* pp = (ParallelProcessor*)calloc(1, sizeof(ParallelProcessor));
    if (!pp) {
        return PP_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy configuration
    memcpy(&pp->config, config, sizeof(ParallelProcessorConfig));
    
    // Initialize mutexes and conditions
    if (pthread_mutex_init(&pp->queue_mutex, NULL) != 0) {
        free(pp);
        return PP_ERROR_SYNCHRONIZATION;
    }
    
    if (pthread_cond_init(&pp->queue_cond, NULL) != 0) {
        pthread_mutex_destroy(&pp->queue_mutex);
        free(pp);
        return PP_ERROR_SYNCHRONIZATION;
    }
    
    if (pthread_mutex_init(&pp->id_mutex, NULL) != 0) {
        pthread_cond_destroy(&pp->queue_cond);
        pthread_mutex_destroy(&pp->queue_mutex);
        free(pp);
        return PP_ERROR_SYNCHRONIZATION;
    }
    
    if (pthread_mutex_init(&pp->metrics_mutex, NULL) != 0) {
        pthread_mutex_destroy(&pp->id_mutex);
        pthread_cond_destroy(&pp->queue_cond);
        pthread_mutex_destroy(&pp->queue_mutex);
        free(pp);
        return PP_ERROR_SYNCHRONIZATION;
    }
    
    if (pthread_rwlock_init(&pp->cache_lock, NULL) != 0) {
        pthread_mutex_destroy(&pp->metrics_mutex);
        pthread_mutex_destroy(&pp->id_mutex);
        pthread_cond_destroy(&pp->queue_cond);
        pthread_mutex_destroy(&pp->queue_mutex);
        free(pp);
        return PP_ERROR_SYNCHRONIZATION;
    }
    
    // Initialize GPU context if enabled
    if (config->gpu_config.enable_gpu && (config->mode == PP_MODE_GPU_ONLY || 
                                          config->mode == PP_MODE_HYBRID ||
                                          config->mode == PP_MODE_ADAPTIVE)) {
        ParallelProcessorError error = initialize_gpu_context(pp);
        if (error != PP_SUCCESS) {
            // GPU initialization failed, fall back to CPU-only if hybrid/adaptive
            if (config->mode == PP_MODE_GPU_ONLY) {
                pthread_rwlock_destroy(&pp->cache_lock);
                pthread_mutex_destroy(&pp->metrics_mutex);
                pthread_mutex_destroy(&pp->id_mutex);
                pthread_cond_destroy(&pp->queue_cond);
                pthread_mutex_destroy(&pp->queue_mutex);
                free(pp);
                return error;
            }
            pp->gpu_available = false;
        } else {
            pp->gpu_available = true;
        }
    }
    
    // Allocate worker threads
    uint32_t num_threads = config->thread_config.num_threads;
    if (num_threads == 0) {
        // Auto-detect optimal thread count
        parallel_processor_get_optimal_threads(&num_threads);
    }
    
    pp->workers = (WorkerThread*)calloc(num_threads, sizeof(WorkerThread));
    if (!pp->workers) {
        if (pp->gpu_context) {
            cleanup_gpu_context(pp->gpu_context);
        }
        pthread_rwlock_destroy(&pp->cache_lock);
        pthread_mutex_destroy(&pp->metrics_mutex);
        pthread_mutex_destroy(&pp->id_mutex);
        pthread_cond_destroy(&pp->queue_cond);
        pthread_mutex_destroy(&pp->queue_mutex);
        free(pp);
        return PP_ERROR_MEMORY_ALLOCATION;
    }
    
    pp->num_workers = num_threads;
    
    // Initialize workers
    for (uint32_t i = 0; i < num_threads; i++) {
        pp->workers[i].thread_id = i;
        pp->workers[i].processor = pp;
        pthread_mutex_init(&pp->workers[i].queue_mutex, NULL);
    }
    
    // Initialize cache if enabled
    if (config->enable_caching && config->cache_size_mb > 0) {
        pp->cache_capacity = config->cache_size_mb * 1024 * 1024;
        pp->cache = malloc(pp->cache_capacity);
        if (!pp->cache) {
            pp->config.enable_caching = false;
        }
    }
    
    pp->is_initialized = true;
    pp->next_task_id = 1;
    
    *processor = pp;
    return PP_SUCCESS;
}

ParallelProcessorError parallel_processor_start(ParallelProcessor* processor) {
    if (!processor || !processor->is_initialized) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    if (processor->is_running) {
        return PP_ERROR_ALREADY_RUNNING;
    }
    
    processor->is_running = true;
    
    // Start worker threads
    for (uint32_t i = 0; i < processor->num_workers; i++) {
        processor->workers[i].should_stop = false;
        processor->workers[i].is_active = true;
        
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        
        if (processor->config.thread_config.thread_stack_size > 0) {
            pthread_attr_setstacksize(&attr, processor->config.thread_config.thread_stack_size);
        }
        
        int result = pthread_create(&processor->workers[i].thread, &attr, 
                                   worker_thread_main, &processor->workers[i]);
        pthread_attr_destroy(&attr);
        
        if (result != 0) {
            // Stop already started threads
            processor->is_running = false;
            for (uint32_t j = 0; j < i; j++) {
                processor->workers[j].should_stop = true;
            }
            pthread_cond_broadcast(&processor->queue_cond);
            
            for (uint32_t j = 0; j < i; j++) {
                pthread_join(processor->workers[j].thread, NULL);
            }
            
            return PP_ERROR_THREAD_CREATION;
        }
        
        // Set thread affinity if configured
        if (processor->config.thread_config.pin_threads_to_cores && 
            processor->config.thread_config.core_affinity_map) {
            // macOS doesn't support pthread_setaffinity_np
            // Would need to use thread_policy_set() with mach threads
        }
    }
    
    return PP_SUCCESS;
}

ParallelProcessorError parallel_processor_stop(ParallelProcessor* processor, bool force_stop) {
    if (!processor || !processor->is_running) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    processor->is_running = false;
    
    // Signal all threads to stop
    pthread_mutex_lock(&processor->queue_mutex);
    for (uint32_t i = 0; i < processor->num_workers; i++) {
        processor->workers[i].should_stop = true;
    }
    pthread_cond_broadcast(&processor->queue_cond);
    pthread_mutex_unlock(&processor->queue_mutex);
    
    // Wait for threads to finish
    for (uint32_t i = 0; i < processor->num_workers; i++) {
        if (processor->workers[i].is_active) {
            pthread_join(processor->workers[i].thread, NULL);
            processor->workers[i].is_active = false;
        }
    }
    
    // Cancel pending tasks if force stop
    if (force_stop) {
        pthread_mutex_lock(&processor->queue_mutex);
        ProcessingTask* task = processor->task_queue_head;
        while (task) {
            if (task->status == PP_TASK_PENDING || task->status == PP_TASK_SCHEDULED) {
                task->status = PP_TASK_CANCELLED;
            }
            task = task->next;
        }
        pthread_mutex_unlock(&processor->queue_mutex);
    }
    
    return PP_SUCCESS;
}

void parallel_processor_destroy(ParallelProcessor* processor) {
    if (!processor) {
        return;
    }
    
    // Stop if still running
    if (processor->is_running) {
        parallel_processor_stop(processor, true);
    }
    
    // Clean up tasks
    pthread_mutex_lock(&processor->queue_mutex);
    ProcessingTask* task = processor->task_queue_head;
    while (task) {
        ProcessingTask* next = task->next;
        if (task->output_data && task->allocated_output_size > 0) {
            free(task->output_data);
        }
        free(task);
        task = next;
    }
    pthread_mutex_unlock(&processor->queue_mutex);
    
    // Clean up workers
    if (processor->workers) {
        for (uint32_t i = 0; i < processor->num_workers; i++) {
            pthread_mutex_destroy(&processor->workers[i].queue_mutex);
            if (processor->workers[i].thread_local_storage) {
                free(processor->workers[i].thread_local_storage);
            }
        }
        free(processor->workers);
    }
    
    // Clean up GPU context
    if (processor->gpu_context) {
        cleanup_gpu_context(processor->gpu_context);
    }
    
    // Clean up cache
    if (processor->cache) {
        free(processor->cache);
    }
    
    // Destroy synchronization primitives
    pthread_rwlock_destroy(&processor->cache_lock);
    pthread_mutex_destroy(&processor->metrics_mutex);
    pthread_mutex_destroy(&processor->id_mutex);
    pthread_cond_destroy(&processor->queue_cond);
    pthread_mutex_destroy(&processor->queue_mutex);
    
    free(processor);
}

ParallelProcessorError parallel_processor_submit_task(ParallelProcessor* processor,
                                                    const void* input_data,
                                                    size_t input_size,
                                                    ProcessingFunction cpu_function,
                                                    GPUKernelFunction gpu_function,
                                                    TaskPriority priority,
                                                    uint64_t* task_id) {
    if (!processor || !input_data || input_size == 0 || (!cpu_function && !gpu_function)) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    if (!processor->is_running) {
        return PP_ERROR_NOT_INITIALIZED;
    }
    
    // Check queue size limit
    pthread_mutex_lock(&processor->queue_mutex);
    if (processor->queue_size >= processor->config.max_queue_size) {
        pthread_mutex_unlock(&processor->queue_mutex);
        return PP_ERROR_QUEUE_FULL;
    }
    pthread_mutex_unlock(&processor->queue_mutex);
    
    // Allocate task
    ProcessingTask* task = (ProcessingTask*)calloc(1, sizeof(ProcessingTask));
    if (!task) {
        return PP_ERROR_MEMORY_ALLOCATION;
    }
    
    // Generate unique task ID
    pthread_mutex_lock(&processor->id_mutex);
    task->task_id = processor->next_task_id++;
    pthread_mutex_unlock(&processor->id_mutex);
    
    // Initialize task
    task->status = PP_TASK_PENDING;
    task->priority = priority;
    task->input_data = (void*)input_data;
    task->input_size = input_size;
    task->cpu_function = cpu_function;
    task->gpu_function = gpu_function;
    task->submit_time = get_current_time_us();
    
    // Allocate output buffer with some reasonable default size
    task->allocated_output_size = input_size * 2; // Assume 2x expansion max
    task->output_data = malloc(task->allocated_output_size);
    if (!task->output_data) {
        free(task);
        return PP_ERROR_MEMORY_ALLOCATION;
    }
    
    // Enqueue task
    ParallelProcessorError error = enqueue_task(processor, task);
    if (error != PP_SUCCESS) {
        free(task->output_data);
        free(task);
        return error;
    }
    
    if (task_id) {
        *task_id = task->task_id;
    }
    
    // Signal worker threads
    pthread_cond_signal(&processor->queue_cond);
    
    return PP_SUCCESS;
}

ParallelProcessorError parallel_processor_config_default(ParallelProcessorConfig* config,
                                                       ProcessingMode mode) {
    if (!config) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(ParallelProcessorConfig));
    
    config->mode = mode;
    
    // Thread configuration
    config->thread_config.num_threads = 0; // Auto-detect
    config->thread_config.thread_stack_size = 0; // Default
    config->thread_config.pin_threads_to_cores = false;
    config->thread_config.load_balancing = PP_BALANCE_LEAST_LOADED;
    config->thread_config.enable_work_stealing = true;
    config->thread_config.steal_threshold = 4;
    
    // GPU configuration
    config->gpu_config.enable_gpu = (mode != PP_MODE_CPU_ONLY);
    config->gpu_config.num_gpu_queues = 2;
    config->gpu_config.scheduling_policy = PP_GPU_SCHEDULE_DYNAMIC;
    config->gpu_config.batch_size = 32;
    config->gpu_config.max_concurrent_kernels = 8;
    config->gpu_config.enable_gpu_memory_pool = true;
    config->gpu_config.gpu_memory_pool_size = 256 * 1024 * 1024; // 256MB
    config->gpu_config.gpu_utilization_target = 0.8f;
    
    // General configuration
    config->max_queue_size = 1000;
    config->chunk_size = PP_DEFAULT_CHUNK_SIZE;
    config->enable_auto_chunking = true;
    config->enable_caching = false;
    config->cache_size_mb = 64;
    config->enable_compression = false;
    config->compression_level = 6;
    config->timeout_seconds = 300; // 5 minutes
    config->enable_progress_tracking = true;
    config->enable_performance_monitoring = true;
    
    return PP_SUCCESS;
}

ParallelProcessorError parallel_processor_get_metrics(ParallelProcessor* processor,
                                                     PerformanceMetrics* metrics) {
    if (!processor || !metrics) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&processor->metrics_mutex);
    memcpy(metrics, &processor->metrics, sizeof(PerformanceMetrics));
    
    // Calculate derived metrics
    if (metrics->total_tasks_processed > 0) {
        metrics->average_task_time = metrics->total_processing_time / metrics->total_tasks_processed;
    }
    
    // Calculate utilization
    uint64_t current_time = get_current_time_us();
    uint64_t total_time = current_time / 1000000.0; // Convert to seconds
    
    if (total_time > 0) {
        metrics->cpu_utilization_percent = (uint32_t)((metrics->cpu_processing_time / total_time) * 100);
        metrics->gpu_utilization_percent = (uint32_t)((metrics->gpu_processing_time / total_time) * 100);
    }
    
    // Count active threads
    metrics->active_threads = 0;
    for (uint32_t i = 0; i < processor->num_workers; i++) {
        if (processor->workers[i].is_active) {
            metrics->active_threads++;
        }
    }
    
    metrics->pending_tasks = processor->queue_size;
    
    pthread_mutex_unlock(&processor->metrics_mutex);
    
    return PP_SUCCESS;
}

// Worker thread main function
static void* worker_thread_main(void* arg) {
    WorkerThread* worker = (WorkerThread*)arg;
    ParallelProcessor* processor = worker->processor;
    
    while (!worker->should_stop) {
        // Get next task
        ProcessingTask* task = dequeue_task(processor);
        
        if (!task) {
            // No task available, wait
            pthread_mutex_lock(&processor->queue_mutex);
            if (!worker->should_stop && processor->queue_size == 0) {
                pthread_cond_wait(&processor->queue_cond, &processor->queue_mutex);
            }
            pthread_mutex_unlock(&processor->queue_mutex);
            continue;
        }
        
        // Process task
        task->status = PP_TASK_RUNNING;
        task->start_time = get_current_time_us();
        
        ParallelProcessorError error = PP_SUCCESS;
        size_t output_size = task->allocated_output_size;
        
        // Decide whether to use CPU or GPU
        bool use_gpu = false;
        if (processor->gpu_available && task->gpu_function) {
            if (processor->config.mode == PP_MODE_GPU_ONLY) {
                use_gpu = true;
            } else if (processor->config.mode == PP_MODE_HYBRID) {
                // Simple heuristic: use GPU for larger tasks
                use_gpu = (task->input_size >= 100 * 1024); // 100KB threshold
            } else if (processor->config.mode == PP_MODE_ADAPTIVE) {
                // Adaptive: check GPU load
                uint32_t gpu_util;
                parallel_processor_get_gpu_utilization(processor, &gpu_util);
                use_gpu = (gpu_util < 80) && (task->input_size >= 50 * 1024);
            }
        }
        
        if (use_gpu && task->gpu_function) {
            // GPU processing
            pthread_mutex_lock(&processor->gpu_context->gpu_mutex);
            error = task->gpu_function(processor->gpu_context,
                                     task->input_data, task->input_size,
                                     task->output_data, &output_size);
            pthread_mutex_unlock(&processor->gpu_context->gpu_mutex);
        } else if (task->cpu_function) {
            // CPU processing
            error = task->cpu_function(task->input_data, task->input_size,
                                     task->output_data, &output_size,
                                     task->function_context);
        } else {
            error = PP_ERROR_INVALID_PARAM;
        }
        
        task->end_time = get_current_time_us();
        task->output_size = output_size;
        task->error_code = error;
        
        if (error == PP_SUCCESS) {
            task->status = PP_TASK_COMPLETED;
        } else {
            task->status = PP_TASK_FAILED;
        }
        
        // Update metrics
        update_metrics(processor, task);
        
        // Call completion callback if set
        if (task->callback) {
            task->callback(task, task->callback_data);
        }
        
        // Update worker statistics
        worker->tasks_processed++;
        worker->processing_time_us += (task->end_time - task->start_time);
    }
    
    return NULL;
}

// Helper function implementations

static uint64_t get_current_time_us(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    
    uint64_t time = mach_absolute_time();
    return time * timebase.numer / timebase.denom / 1000; // Convert to microseconds
}

static ProcessingTask* dequeue_task(ParallelProcessor* processor) {
    ProcessingTask* task = NULL;
    
    pthread_mutex_lock(&processor->queue_mutex);
    
    if (processor->queue_size > 0) {
        // Priority-based dequeue (simple implementation)
        ProcessingTask* prev = NULL;
        ProcessingTask* current = processor->task_queue_head;
        ProcessingTask* best_prev = NULL;
        ProcessingTask* best_task = current;
        
        // Find highest priority task
        while (current) {
            if (current->status == PP_TASK_PENDING && 
                current->priority > best_task->priority) {
                best_prev = prev;
                best_task = current;
            }
            prev = current;
            current = current->next;
        }
        
        if (best_task && best_task->status == PP_TASK_PENDING) {
            task = best_task;
            task->status = PP_TASK_SCHEDULED;
            
            // Remove from queue
            if (best_prev) {
                best_prev->next = task->next;
            } else {
                processor->task_queue_head = task->next;
            }
            
            if (processor->task_queue_tail == task) {
                processor->task_queue_tail = best_prev;
            }
            
            processor->queue_size--;
            task->next = NULL;
        }
    }
    
    pthread_mutex_unlock(&processor->queue_mutex);
    
    return task;
}

static ParallelProcessorError enqueue_task(ParallelProcessor* processor, ProcessingTask* task) {
    pthread_mutex_lock(&processor->queue_mutex);
    
    task->next = NULL;
    
    if (processor->task_queue_tail) {
        processor->task_queue_tail->next = task;
    } else {
        processor->task_queue_head = task;
    }
    
    processor->task_queue_tail = task;
    processor->queue_size++;
    
    pthread_mutex_unlock(&processor->queue_mutex);
    
    return PP_SUCCESS;
}

static void update_metrics(ParallelProcessor* processor, ProcessingTask* task) {
    pthread_mutex_lock(&processor->metrics_mutex);
    
    processor->metrics.total_tasks_processed++;
    
    if (task->status == PP_TASK_COMPLETED) {
        processor->metrics.tasks_succeeded++;
    } else if (task->status == PP_TASK_FAILED) {
        processor->metrics.tasks_failed++;
    } else if (task->status == PP_TASK_CANCELLED) {
        processor->metrics.tasks_cancelled++;
    }
    
    double task_time = (task->end_time - task->start_time) / 1000000.0; // Convert to seconds
    processor->metrics.total_processing_time += task_time;
    
    if (task->gpu_function) {
        processor->metrics.gpu_processing_time += task_time;
    } else {
        processor->metrics.cpu_processing_time += task_time;
    }
    
    // Calculate throughput
    if (task_time > 0) {
        double mbps = (task->input_size / (1024.0 * 1024.0)) / task_time;
        processor->metrics.throughput_mbps = 
            (processor->metrics.throughput_mbps * 0.9) + (mbps * 0.1); // Exponential moving average
    }
    
    pthread_mutex_unlock(&processor->metrics_mutex);
}

static ParallelProcessorError initialize_gpu_context(ParallelProcessor* processor) {
    @autoreleasepool {
        GPUContext* context = (GPUContext*)calloc(1, sizeof(GPUContext));
        if (!context) {
            return PP_ERROR_MEMORY_ALLOCATION;
        }
        
        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            free(context);
            return PP_ERROR_DEVICE_NOT_FOUND;
        }
        
        context->device = (__bridge_retained void*)device;
        
        // Create command queues
        uint32_t num_queues = processor->config.gpu_config.num_gpu_queues;
        if (num_queues > PP_MAX_GPU_QUEUES) {
            num_queues = PP_MAX_GPU_QUEUES;
        }
        
        for (uint32_t i = 0; i < num_queues; i++) {
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!queue) {
                cleanup_gpu_context(context);
                return PP_ERROR_GPU_ERROR;
            }
            context->command_queues[i] = (__bridge_retained void*)queue;
        }
        
        context->num_queues = num_queues;
        
        // Initialize GPU mutex
        if (pthread_mutex_init(&context->gpu_mutex, NULL) != 0) {
            cleanup_gpu_context(context);
            return PP_ERROR_SYNCHRONIZATION;
        }
        
        // Allocate memory pool if enabled
        if (processor->config.gpu_config.enable_gpu_memory_pool) {
            size_t pool_size = processor->config.gpu_config.gpu_memory_pool_size;
            id<MTLBuffer> buffer = [device newBufferWithLength:pool_size
                                                      options:MTLResourceStorageModeShared];
            if (buffer) {
                context->memory_pool = (__bridge_retained void*)buffer;
                context->pool_size = pool_size;
            }
        }
        
        processor->gpu_context = context;
        return PP_SUCCESS;
    }
}

static void cleanup_gpu_context(GPUContext* context) {
    if (!context) {
        return;
    }
    
    @autoreleasepool {
        // Release command queues
        for (uint32_t i = 0; i < context->num_queues; i++) {
            if (context->command_queues[i]) {
                CFRelease(context->command_queues[i]);
            }
        }
        
        // Release memory pool
        if (context->memory_pool) {
            CFRelease(context->memory_pool);
        }
        
        // Release compute pipelines
        for (uint32_t i = 0; i < context->num_pipelines; i++) {
            if (context->compute_pipelines[i]) {
                CFRelease(context->compute_pipelines[i]);
            }
        }
        
        // Release device
        if (context->device) {
            CFRelease(context->device);
        }
        
        // Destroy mutex
        pthread_mutex_destroy(&context->gpu_mutex);
        
        free(context);
    }
}

// Additional function implementations

ParallelProcessorError parallel_processor_get_optimal_threads(uint32_t* num_threads) {
    if (!num_threads) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    // Get number of CPU cores
    int mib[2] = {CTL_HW, HW_NCPU};
    size_t len = sizeof(int);
    int num_cpus;
    
    if (sysctl(mib, 2, &num_cpus, &len, NULL, 0) == 0) {
        // Use number of cores minus 1 (leave one for system)
        *num_threads = (num_cpus > 1) ? num_cpus - 1 : 1;
    } else {
        // Default to 4 threads
        *num_threads = 4;
    }
    
    // Cap at maximum
    if (*num_threads > PP_MAX_THREADS) {
        *num_threads = PP_MAX_THREADS;
    }
    
    return PP_SUCCESS;
}

ParallelProcessorError parallel_processor_check_gpu_available(bool* is_available) {
    if (!is_available) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        *is_available = (device != nil);
    }
    
    return PP_SUCCESS;
}

ParallelProcessorError parallel_processor_get_gpu_utilization(ParallelProcessor* processor,
                                                             uint32_t* utilization) {
    if (!processor || !utilization) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    if (!processor->gpu_context) {
        *utilization = 0;
        return PP_SUCCESS;
    }
    
    // Simple estimation based on kernel execution time
    pthread_mutex_lock(&processor->metrics_mutex);
    
    uint64_t current_time = get_current_time_us();
    double time_window = 1.0; // 1 second window
    double gpu_time = processor->metrics.gpu_processing_time;
    double total_time = current_time / 1000000.0;
    
    if (total_time > 0 && total_time >= time_window) {
        *utilization = (uint32_t)((gpu_time / total_time) * 100);
        if (*utilization > 100) {
            *utilization = 100;
        }
    } else {
        *utilization = 0;
    }
    
    pthread_mutex_unlock(&processor->metrics_mutex);
    
    return PP_SUCCESS;
}

const char* parallel_processor_error_string(ParallelProcessorError error) {
    switch (error) {
        case PP_SUCCESS: return "Success";
        case PP_ERROR_INVALID_PARAM: return "Invalid parameter";
        case PP_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case PP_ERROR_THREAD_CREATION: return "Thread creation failed";
        case PP_ERROR_DEVICE_NOT_FOUND: return "GPU device not found";
        case PP_ERROR_QUEUE_FULL: return "Task queue is full";
        case PP_ERROR_TASK_FAILED: return "Task processing failed";
        case PP_ERROR_TIMEOUT: return "Operation timed out";
        case PP_ERROR_CANCELLED: return "Operation cancelled";
        case PP_ERROR_DEPENDENCY_FAILED: return "Task dependency failed";
        case PP_ERROR_GPU_ERROR: return "GPU processing error";
        case PP_ERROR_ALREADY_RUNNING: return "Processor already running";
        case PP_ERROR_NOT_INITIALIZED: return "Processor not initialized";
        case PP_ERROR_RESOURCE_EXHAUSTED: return "Resource exhausted";
        case PP_ERROR_SYNCHRONIZATION: return "Synchronization error";
        default: return "Unknown error";
    }
}

ParallelProcessorError parallel_processor_estimate_chunk_size(size_t data_size,
                                                            uint32_t num_workers,
                                                            size_t* chunk_size) {
    if (!chunk_size || num_workers == 0) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    // Simple heuristic: divide evenly with minimum chunk size
    size_t min_chunk = 64 * 1024; // 64KB minimum
    size_t max_chunk = 16 * 1024 * 1024; // 16MB maximum
    
    size_t optimal = data_size / (num_workers * 4); // Aim for 4 chunks per worker
    
    if (optimal < min_chunk) {
        *chunk_size = min_chunk;
    } else if (optimal > max_chunk) {
        *chunk_size = max_chunk;
    } else {
        *chunk_size = optimal;
    }
    
    // Align to 4KB boundary
    *chunk_size = (*chunk_size + 4095) & ~4095;
    
    return PP_SUCCESS;
}
