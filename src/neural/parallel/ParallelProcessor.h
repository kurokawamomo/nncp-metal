/*
 * ParallelProcessor.h
 * 
 * Advanced Parallel Processing Engine for Apple Silicon
 * Batch processing parallelization, layer pipelining, async GPU compute,
 * and intelligent CPU-GPU load balancing
 */

#ifndef PARALLEL_PROCESSOR_H
#define PARALLEL_PROCESSOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../acceleration/MetalComputeAccelerator.h"
#include "../optimization/CacheOptimizer.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Dispatch/Dispatch.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct ParallelProcessor ParallelProcessor;

// Apple Silicon system architecture information
typedef struct {
    uint32_t total_cpu_cores;              // Total CPU cores
    uint32_t performance_cpu_cores;        // Performance CPU cores (P-cores)
    uint32_t efficiency_cpu_cores;         // Efficiency CPU cores (E-cores)
    uint32_t gpu_compute_units;            // GPU compute units
    uint32_t neural_engine_cores;          // Neural Engine cores (if available)
    uint64_t unified_memory_size_gb;       // Unified memory size in GB
    uint32_t memory_bandwidth_gbps;        // Memory bandwidth in GB/s
    uint32_t max_threads_per_core;         // Maximum threads per CPU core
    bool supports_metal_gpu;               // Metal GPU support
    bool supports_neural_engine;           // Neural Engine support
    bool supports_simd_instructions;       // SIMD instruction support
} AppleSiliconSystemInfo;

// Parallel processing strategy types
typedef enum {
    PARALLEL_STRATEGY_CPU_ONLY = 0,       // CPU-only processing
    PARALLEL_STRATEGY_GPU_ONLY,           // GPU-only processing
    PARALLEL_STRATEGY_CPU_GPU_BALANCED,   // Balanced CPU-GPU processing
    PARALLEL_STRATEGY_GPU_DOMINANT,       // GPU-dominant with CPU assist
    PARALLEL_STRATEGY_CPU_DOMINANT,       // CPU-dominant with GPU assist
    PARALLEL_STRATEGY_ADAPTIVE,           // Adaptive strategy selection
    PARALLEL_STRATEGY_PIPELINE,           // Pipelined processing
    PARALLEL_STRATEGY_NEURAL_ENGINE       // Neural Engine acceleration
} ParallelProcessingStrategy;

// Workload distribution patterns
typedef enum {
    WORKLOAD_DISTRIBUTION_EVEN = 0,       // Even distribution across cores
    WORKLOAD_DISTRIBUTION_WEIGHTED,       // Weighted by core performance
    WORKLOAD_DISTRIBUTION_ADAPTIVE,       // Adaptive based on load
    WORKLOAD_DISTRIBUTION_BATCH_SPLIT,    // Split by batch dimension
    WORKLOAD_DISTRIBUTION_LAYER_SPLIT,    // Split by layer dimension
    WORKLOAD_DISTRIBUTION_FEATURE_SPLIT,  // Split by feature dimension
    WORKLOAD_DISTRIBUTION_HYBRID          // Hybrid distribution strategy
} WorkloadDistributionPattern;

// Parallel processing configuration
typedef struct {
    bool enable_cpu_parallelization;      // Enable CPU-based parallelization
    bool enable_gpu_acceleration;         // Enable GPU acceleration
    bool enable_async_execution;          // Enable asynchronous execution
    bool enable_pipeline_processing;      // Enable layer pipelining
    bool enable_batch_parallelization;    // Enable batch parallelization
    bool enable_dynamic_load_balancing;   // Enable dynamic load balancing
    bool enable_numa_awareness;           // Enable NUMA-aware scheduling
    bool prefer_performance_cores;        // Prefer P-cores over E-cores
    uint32_t max_cpu_threads;             // Maximum CPU threads to use
    uint32_t max_gpu_command_buffers;     // Maximum GPU command buffers
    uint32_t pipeline_depth;              // Pipeline depth (layers)
    float cpu_gpu_balance_ratio;          // CPU:GPU work ratio (0.0-1.0)
    uint32_t batch_size_threshold;        // Threshold for batch parallelization
    uint32_t memory_pressure_threshold;   // Memory pressure threshold (%)
    uint32_t thermal_throttle_threshold;  // Thermal throttle threshold (°C)
} ParallelProcessingConfig;

// Batch processing configuration
typedef struct {
    uint32_t total_batch_size;            // Total batch size to process
    uint32_t max_sub_batch_size;          // Maximum sub-batch size
    uint32_t min_sub_batch_size;          // Minimum sub-batch size
    bool enable_dynamic_batching;         // Enable dynamic batch sizing
    bool enable_batch_prefetching;        // Enable batch data prefetching
    bool optimize_for_throughput;         // Optimize for throughput vs latency
    uint32_t cpu_batch_count;             // Number of CPU batch chunks
    uint32_t gpu_batch_count;             // Number of GPU batch chunks
    float memory_efficiency_target;       // Target memory efficiency (0.0-1.0)
} BatchProcessingConfig;

// Pipeline stage configuration
typedef struct {
    uint32_t stage_id;                    // Pipeline stage identifier
    const char* stage_name;               // Human-readable stage name
    uint32_t input_dependencies;          // Number of input dependencies
    uint32_t* dependency_stage_ids;       // Array of dependency stage IDs
    bool is_gpu_accelerated;              // Stage uses GPU acceleration
    bool is_memory_intensive;             // Stage is memory-intensive
    bool is_compute_intensive;            // Stage is compute-intensive
    uint32_t estimated_compute_cycles;    // Estimated compute cycles
    uint32_t estimated_memory_accesses;   // Estimated memory accesses
    float stage_priority;                 // Stage priority (0.0-1.0)
} PipelineStageConfig;

// CPU-GPU workload assignment
typedef struct {
    float cpu_workload_fraction;          // Fraction of work for CPU (0.0-1.0)
    float gpu_workload_fraction;          // Fraction of work for GPU (0.0-1.0)
    uint32_t cpu_thread_count;            // Number of CPU threads assigned
    uint32_t gpu_compute_units_used;      // Number of GPU compute units used
    size_t cpu_memory_allocated_mb;       // CPU memory allocated (MB)
    size_t gpu_memory_allocated_mb;       // GPU memory allocated (MB)
    bool requires_cpu_gpu_synchronization; // Requires CPU-GPU sync points
    uint32_t synchronization_points;      // Number of sync points
} WorkloadAssignment;

// Performance monitoring metrics
typedef struct {
    uint64_t total_processing_time_us;    // Total processing time
    uint64_t cpu_processing_time_us;      // CPU processing time
    uint64_t gpu_processing_time_us;      // GPU processing time
    uint64_t memory_transfer_time_us;     // Memory transfer time
    uint64_t synchronization_time_us;     // Synchronization overhead time
    float cpu_utilization_percent;        // CPU utilization percentage
    float gpu_utilization_percent;        // GPU utilization percentage
    float memory_bandwidth_utilization;   // Memory bandwidth utilization
    uint32_t pipeline_stalls;             // Number of pipeline stalls
    uint32_t load_balancing_adjustments;  // Load balancing adjustments made
    float thermal_throttle_events;        // Thermal throttling events
    float power_efficiency_score;         // Power efficiency score
    uint64_t cache_misses;                // Cache miss count
    float parallel_efficiency;            // Parallel efficiency score (0.0-1.0)
} ParallelProcessingMetrics;

// Asynchronous execution context
typedef struct {
    uint32_t context_id;                  // Execution context ID
    bool is_active;                       // Context is currently active
    bool is_gpu_context;                  // GPU execution context
    uint32_t associated_thread_id;        // Associated thread ID
    void* completion_callback;            // Completion callback function
    void* user_data;                      // User data for callback
    uint64_t start_time_us;               // Execution start time
    uint32_t timeout_ms;                  // Execution timeout
    float progress_percent;               // Execution progress (0.0-100.0)
} AsyncExecutionContext;

// Load balancing state
typedef struct {
    float current_cpu_load;               // Current CPU load (0.0-1.0)
    float current_gpu_load;               // Current GPU load (0.0-1.0)
    float current_memory_pressure;        // Current memory pressure (0.0-1.0)
    float current_thermal_state;          // Current thermal state (0.0-1.0)
    uint32_t recent_task_count;           // Recent task count
    uint64_t last_rebalance_time_us;      // Last rebalance timestamp
    bool needs_rebalancing;               // Needs load rebalancing
    WorkloadAssignment optimal_assignment; // Optimal workload assignment
    float performance_trend;              // Performance trend indicator
} LoadBalancingState;

// Error codes for parallel processing
typedef enum {
    PARALLEL_PROCESSOR_SUCCESS = 0,
    PARALLEL_PROCESSOR_ERROR_INVALID_PARAM,
    PARALLEL_PROCESSOR_ERROR_MEMORY_ALLOCATION,
    PARALLEL_PROCESSOR_ERROR_THREAD_CREATION,
    PARALLEL_PROCESSOR_ERROR_GPU_INITIALIZATION,
    PARALLEL_PROCESSOR_ERROR_PIPELINE_STALL,
    PARALLEL_PROCESSOR_ERROR_LOAD_BALANCING_FAILED,
    PARALLEL_PROCESSOR_ERROR_SYNCHRONIZATION_TIMEOUT,
    PARALLEL_PROCESSOR_ERROR_INSUFFICIENT_MEMORY,
    PARALLEL_PROCESSOR_ERROR_THERMAL_THROTTLE,
    PARALLEL_PROCESSOR_ERROR_EXECUTION_TIMEOUT,
    PARALLEL_PROCESSOR_ERROR_INVALID_WORKLOAD,
    PARALLEL_PROCESSOR_ERROR_HARDWARE_INCOMPATIBLE
} ParallelProcessorError;

// Core API Functions

/**
 * Create parallel processor instance
 * @param processor Pointer to store created processor
 * @param config Parallel processing configuration
 * @param metal_accelerator Metal GPU accelerator for GPU operations
 * @param cache_optimizer Cache optimizer for memory optimization
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_create(ParallelProcessor** processor,
                                                 const ParallelProcessingConfig* config,
                                                 MetalComputeAccelerator* metal_accelerator,
                                                 CacheOptimizer* cache_optimizer);

/**
 * Initialize parallel processor with system detection
 * @param processor Parallel processor instance
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_initialize(ParallelProcessor* processor);

/**
 * Get Apple Silicon system information
 * @param processor Parallel processor instance
 * @param system_info Output system information
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_get_system_info(ParallelProcessor* processor,
                                                          AppleSiliconSystemInfo* system_info);

// Batch Processing Parallelization

/**
 * Configure batch processing parallelization
 * @param processor Parallel processor instance
 * @param batch_config Batch processing configuration
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_configure_batch_processing(ParallelProcessor* processor,
                                                                     const BatchProcessingConfig* batch_config);

/**
 * Execute parallel batch processing
 * @param processor Parallel processor instance
 * @param input_data Array of input data batches
 * @param output_data Array of output data batches
 * @param batch_count Number of batches to process
 * @param processing_function Function to process each batch
 * @param user_context User context passed to processing function
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_execute_batch_parallel(ParallelProcessor* processor,
                                                                 void* const* input_data,
                                                                 void** output_data,
                                                                 uint32_t batch_count,
                                                                 void* processing_function,
                                                                 void* user_context);

/**
 * Optimize batch size for parallel processing
 * @param processor Parallel processor instance
 * @param total_data_size Total size of data to process
 * @param element_size Size of individual data elements
 * @param compute_intensity Compute intensity factor (1.0 = balanced)
 * @param optimal_batch_size Output optimal batch size
 * @param recommended_split Output recommended CPU/GPU split
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_optimize_batch_size(ParallelProcessor* processor,
                                                             size_t total_data_size,
                                                             size_t element_size,
                                                             float compute_intensity,
                                                             uint32_t* optimal_batch_size,
                                                             WorkloadAssignment* recommended_split);

// Layer Pipeline Processing

/**
 * Configure layer pipeline processing
 * @param processor Parallel processor instance
 * @param pipeline_stages Array of pipeline stage configurations
 * @param stage_count Number of pipeline stages
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_configure_pipeline(ParallelProcessor* processor,
                                                            const PipelineStageConfig* pipeline_stages,
                                                            uint32_t stage_count);

/**
 * Execute pipelined layer processing
 * @param processor Parallel processor instance
 * @param input_data Input data for first pipeline stage
 * @param output_data Output data from last pipeline stage
 * @param data_size Size of data flowing through pipeline
 * @param enable_overlapping Enable stage overlapping
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_execute_pipeline(ParallelProcessor* processor,
                                                          const void* input_data,
                                                          void** output_data,
                                                          size_t data_size,
                                                          bool enable_overlapping);

/**
 * Analyze pipeline bottlenecks and optimize
 * @param processor Parallel processor instance
 * @param stage_performance Array of per-stage performance metrics
 * @param stage_count Number of pipeline stages
 * @param bottleneck_stage_id Output ID of bottleneck stage
 * @param optimization_suggestions Output optimization suggestions
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_analyze_pipeline_bottlenecks(ParallelProcessor* processor,
                                                                       const ParallelProcessingMetrics* stage_performance,
                                                                       uint32_t stage_count,
                                                                       uint32_t* bottleneck_stage_id,
                                                                       char* optimization_suggestions);

// Asynchronous GPU Computation

/**
 * Submit asynchronous GPU computation
 * @param processor Parallel processor instance
 * @param computation_function GPU computation function
 * @param input_data Input data for computation
 * @param input_size Size of input data
 * @param output_data Output data buffer
 * @param output_size Size of output buffer
 * @param execution_context Output execution context for tracking
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_submit_async_gpu(ParallelProcessor* processor,
                                                          void* computation_function,
                                                          const void* input_data,
                                                          size_t input_size,
                                                          void* output_data,
                                                          size_t output_size,
                                                          AsyncExecutionContext* execution_context);

/**
 * Check status of asynchronous GPU computation
 * @param processor Parallel processor instance
 * @param execution_context Execution context to check
 * @param is_complete Output boolean for completion status
 * @param progress_percent Output progress percentage
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_check_async_status(ParallelProcessor* processor,
                                                           const AsyncExecutionContext* execution_context,
                                                           bool* is_complete,
                                                           float* progress_percent);

/**
 * Wait for asynchronous GPU computation to complete
 * @param processor Parallel processor instance
 * @param execution_context Execution context to wait for
 * @param timeout_ms Maximum wait time in milliseconds (0 = infinite)
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_wait_async_completion(ParallelProcessor* processor,
                                                              const AsyncExecutionContext* execution_context,
                                                              uint32_t timeout_ms);

/**
 * Cancel asynchronous GPU computation
 * @param processor Parallel processor instance
 * @param execution_context Execution context to cancel
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_cancel_async(ParallelProcessor* processor,
                                                      AsyncExecutionContext* execution_context);

// CPU-GPU Load Balancing

/**
 * Analyze current system load and performance
 * @param processor Parallel processor instance
 * @param load_balancing_state Output current load balancing state
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_analyze_system_load(ParallelProcessor* processor,
                                                             LoadBalancingState* load_balancing_state);

/**
 * Calculate optimal CPU-GPU workload distribution
 * @param processor Parallel processor instance
 * @param workload_characteristics Characteristics of the workload
 * @param current_system_state Current system load state
 * @param optimal_assignment Output optimal workload assignment
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_calculate_optimal_distribution(ParallelProcessor* processor,
                                                                        const void* workload_characteristics,
                                                                        const LoadBalancingState* current_system_state,
                                                                        WorkloadAssignment* optimal_assignment);

/**
 * Execute workload with dynamic load balancing
 * @param processor Parallel processor instance
 * @param cpu_workload_function CPU workload function
 * @param gpu_workload_function GPU workload function
 * @param input_data Input data to process
 * @param output_data Output data buffer
 * @param data_size Size of data to process
 * @param distribution_pattern Workload distribution pattern
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_execute_load_balanced(ParallelProcessor* processor,
                                                               void* cpu_workload_function,
                                                               void* gpu_workload_function,
                                                               const void* input_data,
                                                               void* output_data,
                                                               size_t data_size,
                                                               WorkloadDistributionPattern distribution_pattern);

/**
 * Adjust load balancing based on runtime performance
 * @param processor Parallel processor instance
 * @param current_metrics Current performance metrics
 * @param adjustment_aggressiveness Adjustment aggressiveness (0.0-1.0)
 * @param new_assignment Output new workload assignment
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_adjust_load_balancing(ParallelProcessor* processor,
                                                               const ParallelProcessingMetrics* current_metrics,
                                                               float adjustment_aggressiveness,
                                                               WorkloadAssignment* new_assignment);

// Neural Network Specific Parallel Operations

/**
 * Execute parallel transformer attention computation
 * @param processor Parallel processor instance
 * @param query_matrices Array of query matrices for parallel processing
 * @param key_matrices Array of key matrices
 * @param value_matrices Array of value matrices
 * @param attention_outputs Array of attention output matrices
 * @param sequence_length Sequence length
 * @param d_model Model dimension
 * @param num_heads Number of attention heads
 * @param batch_count Number of batches to process in parallel
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_execute_parallel_attention(ParallelProcessor* processor,
                                                                     const float* const* query_matrices,
                                                                     const float* const* key_matrices,
                                                                     const float* const* value_matrices,
                                                                     float** attention_outputs,
                                                                     uint32_t sequence_length,
                                                                     uint32_t d_model,
                                                                     uint32_t num_heads,
                                                                     uint32_t batch_count);

/**
 * Execute parallel feed-forward network computation
 * @param processor Parallel processor instance
 * @param input_matrices Array of input matrices for parallel processing
 * @param weight_matrix Shared weight matrix
 * @param bias_vector Shared bias vector
 * @param output_matrices Array of output matrices
 * @param input_dimension Input dimension
 * @param hidden_dimension Hidden dimension
 * @param batch_count Number of batches to process in parallel
 * @param activation_function Activation function to use
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_execute_parallel_ffn(ParallelProcessor* processor,
                                                               const float* const* input_matrices,
                                                               const float* weight_matrix,
                                                               const float* bias_vector,
                                                               float** output_matrices,
                                                               uint32_t input_dimension,
                                                               uint32_t hidden_dimension,
                                                               uint32_t batch_count,
                                                               const char* activation_function);

/**
 * Execute parallel layer normalization
 * @param processor Parallel processor instance
 * @param input_vectors Array of input vectors for parallel processing
 * @param gamma_weights Gamma scaling weights
 * @param beta_weights Beta bias weights
 * @param output_vectors Array of output normalized vectors
 * @param vector_length Length of each vector
 * @param batch_count Number of vectors to normalize in parallel
 * @param epsilon Normalization epsilon
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_execute_parallel_layer_norm(ParallelProcessor* processor,
                                                                      const float* const* input_vectors,
                                                                      const float* gamma_weights,
                                                                      const float* beta_weights,
                                                                      float** output_vectors,
                                                                      uint32_t vector_length,
                                                                      uint32_t batch_count,
                                                                      float epsilon);

// Performance Monitoring and Analysis

/**
 * Start parallel processing performance monitoring
 * @param processor Parallel processor instance
 * @param monitoring_interval_ms Monitoring interval in milliseconds
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_start_monitoring(ParallelProcessor* processor,
                                                          uint32_t monitoring_interval_ms);

/**
 * Get current parallel processing performance metrics
 * @param processor Parallel processor instance
 * @param metrics Output performance metrics
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_get_performance_metrics(ParallelProcessor* processor,
                                                                 ParallelProcessingMetrics* metrics);

/**
 * Stop parallel processing performance monitoring
 * @param processor Parallel processor instance
 * @param final_metrics Output final performance metrics
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_stop_monitoring(ParallelProcessor* processor,
                                                         ParallelProcessingMetrics* final_metrics);

/**
 * Reset parallel processing performance counters
 * @param processor Parallel processor instance
 */
void parallel_processor_reset_counters(ParallelProcessor* processor);

/**
 * Generate parallel processing performance report
 * @param processor Parallel processor instance
 * @param metrics Performance metrics to analyze
 * @param report_buffer Output buffer for report text
 * @param buffer_size Size of report buffer
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_generate_performance_report(ParallelProcessor* processor,
                                                                      const ParallelProcessingMetrics* metrics,
                                                                      char* report_buffer,
                                                                      size_t buffer_size);

// Configuration and Utility Functions

/**
 * Create default parallel processing configuration
 * @param config Output default configuration
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_create_default_config(ParallelProcessingConfig* config);

/**
 * Create Apple Silicon optimized parallel processing configuration
 * @param config Output Apple Silicon optimized configuration
 * @param system_info System information for optimization
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_create_apple_silicon_config(ParallelProcessingConfig* config,
                                                                      const AppleSiliconSystemInfo* system_info);

/**
 * Create throughput-optimized parallel processing configuration
 * @param config Output throughput-optimized configuration
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_create_throughput_config(ParallelProcessingConfig* config);

/**
 * Create latency-optimized parallel processing configuration
 * @param config Output latency-optimized configuration
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_create_latency_config(ParallelProcessingConfig* config);

/**
 * Validate parallel processing configuration
 * @param config Configuration to validate
 * @param system_info System information for validation
 * @param is_valid Output boolean for validity
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_validate_config(const ParallelProcessingConfig* config,
                                                          const AppleSiliconSystemInfo* system_info,
                                                          bool* is_valid);

/**
 * Estimate parallel processing performance improvement
 * @param processor Parallel processor instance
 * @param workload_characteristics Workload characteristics
 * @param target_configuration Target parallel configuration
 * @param estimated_speedup Output estimated speedup factor
 * @param estimated_efficiency Output estimated parallel efficiency
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_estimate_performance(ParallelProcessor* processor,
                                                              const void* workload_characteristics,
                                                              const ParallelProcessingConfig* target_configuration,
                                                              float* estimated_speedup,
                                                              float* estimated_efficiency);

/**
 * Destroy parallel processor and free resources
 * @param processor Parallel processor instance to destroy
 */
void parallel_processor_destroy(ParallelProcessor* processor);

// Utility Functions

/**
 * Get error string for parallel processor error code
 * @param error_code ParallelProcessorError code
 * @return Human-readable error message
 */
const char* parallel_processor_get_error_string(ParallelProcessorError error_code);

/**
 * Get parallel processing strategy string
 * @param strategy Parallel processing strategy
 * @return Human-readable strategy name
 */
const char* parallel_processor_get_strategy_string(ParallelProcessingStrategy strategy);

/**
 * Get workload distribution pattern string
 * @param pattern Workload distribution pattern
 * @return Human-readable pattern name
 */
const char* parallel_processor_get_distribution_pattern_string(WorkloadDistributionPattern pattern);

/**
 * Calculate optimal thread count for CPU workload
 * @param workload_type Type of workload (compute/memory intensive)
 * @param available_cores Number of available CPU cores
 * @param system_load Current system load (0.0-1.0)
 * @return Optimal thread count
 */
uint32_t parallel_processor_calculate_optimal_thread_count(const char* workload_type,
                                                          uint32_t available_cores,
                                                          float system_load);

/**
 * Calculate memory requirements for parallel processing
 * @param batch_size Batch size to process
 * @param data_element_size Size of individual data elements
 * @param num_parallel_streams Number of parallel processing streams
 * @param cpu_memory_mb Output CPU memory requirement in MB
 * @param gpu_memory_mb Output GPU memory requirement in MB
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_calculate_memory_requirements(uint32_t batch_size,
                                                                       size_t data_element_size,
                                                                       uint32_t num_parallel_streams,
                                                                       size_t* cpu_memory_mb,
                                                                       size_t* gpu_memory_mb);

/**
 * Check thermal state and adjust processing intensity
 * @param processor Parallel processor instance
 * @param current_temperature Current system temperature (°C)
 * @param thermal_throttle_needed Output boolean for thermal throttling
 * @param recommended_intensity Output recommended processing intensity (0.0-1.0)
 * @return PARALLEL_PROCESSOR_SUCCESS on success, error code on failure
 */
ParallelProcessorError parallel_processor_check_thermal_state(ParallelProcessor* processor,
                                                             float current_temperature,
                                                             bool* thermal_throttle_needed,
                                                             float* recommended_intensity);

// Constants for Apple Silicon parallel processing
#define APPLE_SILICON_MAX_CPU_CORES 24              // Maximum CPU cores (M3 Max)
#define APPLE_SILICON_MAX_PERFORMANCE_CORES 12      // Maximum P-cores
#define APPLE_SILICON_MAX_EFFICIENCY_CORES 12       // Maximum E-cores
#define APPLE_SILICON_MAX_GPU_CORES 40              // Maximum GPU cores (M3 Max)
#define APPLE_SILICON_MAX_UNIFIED_MEMORY_GB 128     // Maximum unified memory

// Parallel processing constants
#define PARALLEL_PROCESSOR_DEFAULT_BATCH_SIZE 32        // Default batch size
#define PARALLEL_PROCESSOR_MIN_BATCH_SIZE 4             // Minimum batch size
#define PARALLEL_PROCESSOR_MAX_BATCH_SIZE 1024          // Maximum batch size
#define PARALLEL_PROCESSOR_DEFAULT_PIPELINE_DEPTH 4     // Default pipeline depth
#define PARALLEL_PROCESSOR_MAX_PIPELINE_DEPTH 16        // Maximum pipeline depth
#define PARALLEL_PROCESSOR_DEFAULT_ASYNC_CONTEXTS 8     // Default async contexts

// Performance thresholds
#define PARALLEL_EFFICIENCY_EXCELLENT 0.9f              // Excellent parallel efficiency
#define PARALLEL_EFFICIENCY_GOOD 0.7f                   // Good parallel efficiency
#define PARALLEL_EFFICIENCY_ACCEPTABLE 0.5f             // Acceptable parallel efficiency
#define CPU_GPU_BALANCE_RATIO_BALANCED 0.5f             // Balanced CPU:GPU ratio
#define THERMAL_THROTTLE_TEMPERATURE 85.0f              // Thermal throttle threshold (°C)
#define MEMORY_PRESSURE_HIGH_THRESHOLD 0.8f             // High memory pressure threshold

#ifdef __cplusplus
}
#endif

#endif // PARALLEL_PROCESSOR_H
