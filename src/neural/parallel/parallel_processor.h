#ifndef PARALLEL_PROCESSOR_H
#define PARALLEL_PROCESSOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __OBJC__
@class MPSCommandBuffer;
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLBuffer;
@protocol MTLComputePipelineState;
#else
typedef void MPSCommandBuffer;
typedef void* MTLDevice_t;
typedef void* MTLCommandQueue_t;
typedef void* MTLBuffer_t;
typedef void* MTLComputePipelineState_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Maximum limits
#define PP_MAX_THREADS              64
#define PP_MAX_GPU_QUEUES           8
#define PP_MAX_BATCH_SIZE           1024
#define PP_MAX_WORK_ITEMS           10000
#define PP_MAX_DEPENDENCIES         16
#define PP_DEFAULT_CHUNK_SIZE       (1 * 1024 * 1024)  // 1MB default chunk

// Error codes
typedef enum {
    PP_SUCCESS = 0,
    PP_ERROR_INVALID_PARAM,
    PP_ERROR_MEMORY_ALLOCATION,
    PP_ERROR_THREAD_CREATION,
    PP_ERROR_DEVICE_NOT_FOUND,
    PP_ERROR_QUEUE_FULL,
    PP_ERROR_TASK_FAILED,
    PP_ERROR_TIMEOUT,
    PP_ERROR_CANCELLED,
    PP_ERROR_DEPENDENCY_FAILED,
    PP_ERROR_GPU_ERROR,
    PP_ERROR_ALREADY_RUNNING,
    PP_ERROR_NOT_INITIALIZED,
    PP_ERROR_RESOURCE_EXHAUSTED,
    PP_ERROR_SYNCHRONIZATION
} ParallelProcessorError;

// Processing modes
typedef enum {
    PP_MODE_CPU_ONLY = 0,           // CPU-only processing
    PP_MODE_GPU_ONLY,               // GPU-only processing
    PP_MODE_HYBRID,                 // CPU+GPU hybrid processing
    PP_MODE_ADAPTIVE,               // Adaptive mode selection
    PP_MODE_STREAMING               // Streaming processing mode
} ProcessingMode;

// Task priorities
typedef enum {
    PP_PRIORITY_LOW = 0,
    PP_PRIORITY_NORMAL,
    PP_PRIORITY_HIGH,
    PP_PRIORITY_CRITICAL
} TaskPriority;

// Task status
typedef enum {
    PP_TASK_PENDING = 0,
    PP_TASK_SCHEDULED,
    PP_TASK_RUNNING,
    PP_TASK_COMPLETED,
    PP_TASK_FAILED,
    PP_TASK_CANCELLED
} TaskStatus;

// Load balancing strategies
typedef enum {
    PP_BALANCE_ROUND_ROBIN = 0,     // Simple round-robin
    PP_BALANCE_LEAST_LOADED,        // Assign to least loaded worker
    PP_BALANCE_WORK_STEALING,       // Work stealing between threads
    PP_BALANCE_PREDICTIVE,          // Predictive load balancing
    PP_BALANCE_ADAPTIVE             // Adaptive strategy selection
} LoadBalancingStrategy;

// GPU scheduling policies
typedef enum {
    PP_GPU_SCHEDULE_FIFO = 0,       // First-in-first-out
    PP_GPU_SCHEDULE_PRIORITY,       // Priority-based scheduling
    PP_GPU_SCHEDULE_BATCH,          // Batch processing
    PP_GPU_SCHEDULE_DYNAMIC         // Dynamic scheduling
} GPUSchedulingPolicy;

// Forward declarations
typedef struct ParallelProcessor ParallelProcessor;
typedef struct ProcessingTask ProcessingTask;
typedef struct WorkerThread WorkerThread;
typedef struct GPUContext GPUContext;

// Task completion callback
typedef void (*TaskCompletionCallback)(ProcessingTask* task, void* user_data);

// Custom processing function
typedef ParallelProcessorError (*ProcessingFunction)(const void* input, size_t input_size,
                                                    void* output, size_t* output_size,
                                                    void* context);

// GPU kernel function
typedef ParallelProcessorError (*GPUKernelFunction)(void* gpu_context, 
                                                    const void* input, size_t input_size,
                                                    void* output, size_t* output_size);

// Performance metrics
typedef struct {
    uint64_t total_tasks_processed;         // Total number of tasks processed
    uint64_t tasks_succeeded;               // Number of successful tasks
    uint64_t tasks_failed;                  // Number of failed tasks
    uint64_t tasks_cancelled;               // Number of cancelled tasks
    
    double total_processing_time;           // Total processing time (seconds)
    double cpu_processing_time;             // CPU processing time
    double gpu_processing_time;             // GPU processing time
    double idle_time;                       // Idle time
    
    double average_task_time;               // Average task processing time
    double throughput_mbps;                 // Throughput in MB/s
    
    uint32_t cpu_utilization_percent;       // CPU utilization percentage
    uint32_t gpu_utilization_percent;       // GPU utilization percentage
    uint32_t memory_usage_mb;               // Memory usage in MB
    
    uint64_t cache_hits;                    // Number of cache hits
    uint64_t cache_misses;                  // Number of cache misses
    double cache_hit_rate;                  // Cache hit rate
    
    uint32_t active_threads;                // Number of active threads
    uint32_t active_gpu_queues;            // Number of active GPU queues
    uint32_t pending_tasks;                // Number of pending tasks
} PerformanceMetrics;

// Thread pool configuration
typedef struct {
    uint32_t num_threads;                   // Number of worker threads
    uint32_t thread_stack_size;             // Stack size per thread
    bool pin_threads_to_cores;              // Pin threads to CPU cores
    uint32_t* core_affinity_map;            // Core affinity mapping
    LoadBalancingStrategy load_balancing;   // Load balancing strategy
    bool enable_work_stealing;              // Enable work stealing
    uint32_t steal_threshold;               // Work stealing threshold
} ThreadPoolConfig;

// GPU configuration
typedef struct {
    bool enable_gpu;                        // Enable GPU processing
    uint32_t num_gpu_queues;                // Number of GPU command queues
    GPUSchedulingPolicy scheduling_policy;  // GPU scheduling policy
    uint32_t batch_size;                    // GPU batch size
    uint32_t max_concurrent_kernels;        // Max concurrent GPU kernels
    bool enable_gpu_memory_pool;            // Enable GPU memory pooling
    size_t gpu_memory_pool_size;            // GPU memory pool size
    float gpu_utilization_target;           // Target GPU utilization (0-1)
} GPUConfig;

// Parallel processor configuration
typedef struct {
    ProcessingMode mode;                    // Processing mode
    ThreadPoolConfig thread_config;         // Thread pool configuration
    GPUConfig gpu_config;                   // GPU configuration
    
    uint32_t max_queue_size;                // Maximum task queue size
    uint32_t chunk_size;                    // Default chunk size for data
    bool enable_auto_chunking;              // Enable automatic chunking
    
    bool enable_caching;                    // Enable result caching
    size_t cache_size_mb;                   // Cache size in MB
    
    bool enable_compression;                // Enable data compression
    uint32_t compression_level;             // Compression level (0-9)
    
    uint32_t timeout_seconds;               // Task timeout in seconds
    bool enable_progress_tracking;          // Enable progress tracking
    bool enable_performance_monitoring;     // Enable performance monitoring
    
    void* custom_context;                   // Custom user context
} ParallelProcessorConfig;

// Processing task
struct ProcessingTask {
    uint64_t task_id;                       // Unique task identifier
    TaskStatus status;                      // Current task status
    TaskPriority priority;                  // Task priority
    
    void* input_data;                       // Input data pointer
    size_t input_size;                      // Input data size
    void* output_data;                      // Output data pointer
    size_t output_size;                     // Output data size
    size_t allocated_output_size;           // Allocated output buffer size
    
    ProcessingFunction cpu_function;        // CPU processing function
    GPUKernelFunction gpu_function;         // GPU kernel function
    void* function_context;                 // Function-specific context
    
    TaskCompletionCallback callback;        // Completion callback
    void* callback_data;                    // Callback user data
    
    uint64_t dependencies[PP_MAX_DEPENDENCIES]; // Task dependencies
    uint32_t num_dependencies;              // Number of dependencies
    
    uint64_t submit_time;                   // Task submission time
    uint64_t start_time;                    // Task start time
    uint64_t end_time;                      // Task completion time
    
    ParallelProcessorError error_code;      // Error code if failed
    char error_message[256];                // Error message
    
    struct ProcessingTask* next;            // Next task in queue
};

// Worker thread structure
struct WorkerThread {
    pthread_t thread;                       // Thread handle
    uint32_t thread_id;                     // Thread identifier
    bool is_active;                         // Whether thread is active
    bool should_stop;                       // Stop flag
    
    ParallelProcessor* processor;           // Parent processor
    
    uint64_t tasks_processed;               // Tasks processed by this thread
    uint64_t processing_time_us;            // Total processing time
    
    void* thread_local_storage;             // Thread-local storage
    size_t tls_size;                        // TLS size
    
    // Work stealing
    ProcessingTask* local_queue;            // Local task queue
    uint32_t local_queue_size;              // Local queue size
    pthread_mutex_t queue_mutex;            // Queue mutex
};

// GPU context
struct GPUContext {
    void* device;                           // Metal device
    void* command_queues[PP_MAX_GPU_QUEUES]; // Command queues
    uint32_t num_queues;                    // Number of queues
    uint32_t current_queue;                 // Current queue index
    
    void* memory_pool;                      // GPU memory pool
    size_t pool_size;                       // Pool size
    size_t pool_used;                       // Pool used bytes
    
    void* compute_pipelines[16];            // Compute pipeline states
    uint32_t num_pipelines;                 // Number of pipelines
    
    pthread_mutex_t gpu_mutex;              // GPU access mutex
    
    uint64_t kernels_executed;              // Number of kernels executed
    uint64_t gpu_time_us;                   // Total GPU time
};

// Main parallel processor structure
struct ParallelProcessor {
    ParallelProcessorConfig config;         // Configuration
    bool is_initialized;                    // Initialization flag
    bool is_running;                        // Running flag
    
    // Thread pool
    WorkerThread* workers;                  // Worker threads
    uint32_t num_workers;                   // Number of workers
    
    // Task management
    ProcessingTask* task_queue_head;        // Task queue head
    ProcessingTask* task_queue_tail;        // Task queue tail
    uint32_t queue_size;                    // Current queue size
    pthread_mutex_t queue_mutex;            // Queue mutex
    pthread_cond_t queue_cond;              // Queue condition variable
    
    uint64_t next_task_id;                  // Next task ID
    pthread_mutex_t id_mutex;               // ID generation mutex
    
    // GPU context
    GPUContext* gpu_context;                // GPU context
    bool gpu_available;                     // GPU availability flag
    
    // Performance metrics
    PerformanceMetrics metrics;             // Performance metrics
    pthread_mutex_t metrics_mutex;          // Metrics mutex
    
    // Cache
    void* cache;                           // Result cache
    size_t cache_capacity;                 // Cache capacity
    pthread_rwlock_t cache_lock;           // Cache lock
};

// Batch processing request
typedef struct {
    ProcessingTask** tasks;                 // Array of tasks
    uint32_t num_tasks;                     // Number of tasks
    bool wait_for_completion;               // Wait for all tasks
    uint32_t* completed_count;              // Completed task count
    pthread_cond_t* completion_cond;        // Completion condition
    pthread_mutex_t* completion_mutex;      // Completion mutex
} BatchRequest;

// Core API Functions

/**
 * Create and initialize parallel processor
 * @param processor Pointer to store created processor
 * @param config Processor configuration
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_create(ParallelProcessor** processor,
                                               const ParallelProcessorConfig* config);

/**
 * Start the parallel processor
 * @param processor Parallel processor instance
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_start(ParallelProcessor* processor);

/**
 * Stop the parallel processor
 * @param processor Parallel processor instance
 * @param force_stop Force stop without waiting for tasks
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_stop(ParallelProcessor* processor, bool force_stop);

/**
 * Destroy parallel processor and free resources
 * @param processor Processor to destroy
 */
void parallel_processor_destroy(ParallelProcessor* processor);

// Task Management Functions

/**
 * Submit a processing task
 * @param processor Parallel processor instance
 * @param input_data Input data
 * @param input_size Input data size
 * @param cpu_function CPU processing function
 * @param gpu_function GPU kernel function (optional)
 * @param priority Task priority
 * @param task_id Output task ID
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_submit_task(ParallelProcessor* processor,
                                                    const void* input_data,
                                                    size_t input_size,
                                                    ProcessingFunction cpu_function,
                                                    GPUKernelFunction gpu_function,
                                                    TaskPriority priority,
                                                    uint64_t* task_id);

/**
 * Submit a batch of tasks
 * @param processor Parallel processor instance
 * @param batch Batch request
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_submit_batch(ParallelProcessor* processor,
                                                     BatchRequest* batch);

/**
 * Wait for task completion
 * @param processor Parallel processor instance
 * @param task_id Task ID to wait for
 * @param timeout_ms Timeout in milliseconds (0 = infinite)
 * @param result Output task result
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_wait_for_task(ParallelProcessor* processor,
                                                       uint64_t task_id,
                                                       uint32_t timeout_ms,
                                                       ProcessingTask** result);

/**
 * Cancel a pending task
 * @param processor Parallel processor instance
 * @param task_id Task ID to cancel
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_cancel_task(ParallelProcessor* processor,
                                                     uint64_t task_id);

/**
 * Get task status
 * @param processor Parallel processor instance
 * @param task_id Task ID
 * @param status Output task status
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_get_task_status(ParallelProcessor* processor,
                                                         uint64_t task_id,
                                                         TaskStatus* status);

// Configuration Functions

/**
 * Create default processor configuration
 * @param config Output configuration
 * @param mode Processing mode
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_config_default(ParallelProcessorConfig* config,
                                                       ProcessingMode mode);

/**
 * Update processor configuration
 * @param processor Parallel processor instance
 * @param config New configuration
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_update_config(ParallelProcessor* processor,
                                                       const ParallelProcessorConfig* config);

/**
 * Set number of worker threads
 * @param processor Parallel processor instance
 * @param num_threads Number of threads
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_set_thread_count(ParallelProcessor* processor,
                                                          uint32_t num_threads);

/**
 * Set GPU batch size
 * @param processor Parallel processor instance
 * @param batch_size Batch size
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_set_gpu_batch_size(ParallelProcessor* processor,
                                                            uint32_t batch_size);

// Performance and Monitoring Functions

/**
 * Get performance metrics
 * @param processor Parallel processor instance
 * @param metrics Output performance metrics
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_get_metrics(ParallelProcessor* processor,
                                                     PerformanceMetrics* metrics);

/**
 * Reset performance metrics
 * @param processor Parallel processor instance
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_reset_metrics(ParallelProcessor* processor);

/**
 * Get current queue size
 * @param processor Parallel processor instance
 * @param size Output queue size
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_get_queue_size(ParallelProcessor* processor,
                                                        uint32_t* size);

/**
 * Get GPU utilization
 * @param processor Parallel processor instance
 * @param utilization Output GPU utilization (0-100)
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_get_gpu_utilization(ParallelProcessor* processor,
                                                             uint32_t* utilization);

// Utility Functions

/**
 * Estimate optimal chunk size for data
 * @param data_size Total data size
 * @param num_workers Number of workers
 * @param chunk_size Output optimal chunk size
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_estimate_chunk_size(size_t data_size,
                                                            uint32_t num_workers,
                                                            size_t* chunk_size);

/**
 * Get error string for error code
 * @param error Error code
 * @return Human-readable error string
 */
const char* parallel_processor_error_string(ParallelProcessorError error);

/**
 * Get optimal number of threads for system
 * @param num_threads Output optimal thread count
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_get_optimal_threads(uint32_t* num_threads);

/**
 * Check if GPU is available
 * @param is_available Output GPU availability
 * @return PP_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_check_gpu_available(bool* is_available);

#ifdef __cplusplus
}
#endif

#endif // PARALLEL_PROCESSOR_H
