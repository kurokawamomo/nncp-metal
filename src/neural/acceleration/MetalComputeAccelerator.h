/*
 * MetalComputeAccelerator.h
 * 
 * Metal GPU Acceleration Module for Apple Silicon
 * High-performance matrix operations, self-attention, and vectorized operations
 * Optimized for M1/M2/M3/M4 GPU architectures with unified memory
 */

#ifndef METAL_COMPUTE_ACCELERATOR_H
#define METAL_COMPUTE_ACCELERATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct MetalComputeAccelerator MetalComputeAccelerator;

// Metal GPU device capabilities
typedef struct {
    bool supports_unified_memory;       // Apple Silicon unified memory support
    bool supports_float16;              // Half-precision floating point
    bool supports_bfloat16;             // BFloat16 support (M2+)
    bool supports_simd_shuffle;         // SIMD shuffle operations
    bool supports_atomic_operations;     // Atomic operations support
    bool supports_indirect_command_buffers; // Indirect command buffer support
    uint32_t max_compute_units;         // Maximum compute units
    size_t max_buffer_size_mb;          // Maximum buffer size in MB
    uint32_t max_threads_per_group;     // Maximum threads per threadgroup
    uint32_t simd_width;                // SIMD width (32 for Apple GPU)
    uint32_t memory_bandwidth_gbps;     // Memory bandwidth in GB/s
    const char* device_name;            // GPU device name
} MetalDeviceCapabilities;

// Matrix operation configurations
typedef struct {
    uint32_t rows_a;                    // Matrix A rows
    uint32_t cols_a;                    // Matrix A columns
    uint32_t cols_b;                    // Matrix B columns
    bool transpose_a;                   // Transpose matrix A
    bool transpose_b;                   // Transpose matrix B
    bool use_mixed_precision;           // Use mixed precision (FP16/FP32)
    bool use_simd_optimization;         // Enable SIMD optimizations
    uint32_t tile_size;                 // Tile size for blocked operations
    float alpha;                        // Scaling factor alpha
    float beta;                         // Scaling factor beta
} MatrixOperationConfig;

// Self-attention configuration for transformers
typedef struct {
    uint32_t sequence_length;           // Input sequence length
    uint32_t d_model;                   // Model dimension (768)
    uint32_t num_heads;                 // Number of attention heads (16)
    uint32_t head_dimension;            // Dimension per head (48)
    bool use_causal_mask;               // Apply causal masking
    bool use_flash_attention;           // Use Flash Attention optimization
    bool use_sparse_attention;          // Use sparse attention patterns
    float attention_dropout;            // Attention dropout rate
    float softmax_scale;                // Softmax scaling factor
    uint32_t kv_cache_size;             // Key-value cache size
    bool enable_gradient_checkpointing; // Enable gradient checkpointing
} SelfAttentionConfig;

// Vectorized operation configurations
typedef struct {
    uint32_t vector_length;             // Length of input vectors
    uint32_t batch_size;                // Number of vectors to process
    bool use_simd_instructions;         // Use SIMD instructions
    bool use_half_precision;            // Use half precision
    uint32_t threads_per_group;         // Threads per compute group
    uint32_t elements_per_thread;       // Elements processed per thread
    bool enable_memory_coalescing;      // Enable memory coalescing
} VectorOperationConfig;

// Performance metrics and statistics
typedef struct {
    uint64_t total_operations;          // Total GPU operations performed
    uint64_t total_gpu_time_us;         // Total GPU execution time
    uint64_t total_memory_transfers_mb; // Total memory transferred
    float average_gpu_utilization;      // Average GPU utilization
    float peak_memory_bandwidth_gbps;   // Peak memory bandwidth achieved
    uint32_t cache_hits;                // Shader cache hits
    uint32_t cache_misses;              // Shader cache misses
    size_t peak_memory_usage_mb;        // Peak GPU memory usage
    float thermal_throttle_events;      // Thermal throttling events
    float power_efficiency_score;       // Power efficiency score
} MetalPerformanceStats;

// Metal compute accelerator configuration
typedef struct {
    bool enable_unified_memory_optimization; // Enable unified memory optimizations
    bool enable_shader_cache;            // Enable compute shader caching
    bool enable_performance_monitoring;  // Enable performance monitoring
    bool enable_thermal_management;      // Enable thermal management
    bool enable_power_optimization;      // Enable power optimization
    uint32_t command_buffer_pool_size;   // Command buffer pool size
    uint32_t max_concurrent_operations;  // Maximum concurrent operations
    size_t memory_pool_size_mb;          // Memory pool size in MB
    float gpu_memory_fraction;           // Fraction of GPU memory to use
    uint32_t shader_optimization_level;  // Shader optimization level (0-3)
} MetalComputeConfig;

// Error codes for Metal compute acceleration
typedef enum {
    METAL_COMPUTE_SUCCESS = 0,
    METAL_COMPUTE_ERROR_INVALID_PARAM,
    METAL_COMPUTE_ERROR_DEVICE_NOT_FOUND,
    METAL_COMPUTE_ERROR_SHADER_COMPILATION_FAILED,
    METAL_COMPUTE_ERROR_BUFFER_ALLOCATION_FAILED,
    METAL_COMPUTE_ERROR_COMMAND_ENCODING_FAILED,
    METAL_COMPUTE_ERROR_EXECUTION_FAILED,
    METAL_COMPUTE_ERROR_MEMORY_INSUFFICIENT,
    METAL_COMPUTE_ERROR_THERMAL_THROTTLE,
    METAL_COMPUTE_ERROR_TIMEOUT,
    METAL_COMPUTE_ERROR_UNSUPPORTED_OPERATION,
    METAL_COMPUTE_ERROR_DEVICE_LOST
} MetalComputeError;

// Core API Functions

/**
 * Create Metal compute accelerator
 * @param accelerator Pointer to store created accelerator
 * @param config Metal compute configuration
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_create(MetalComputeAccelerator** accelerator,
                                      const MetalComputeConfig* config);

/**
 * Initialize Metal compute accelerator with device selection
 * @param accelerator Metal compute accelerator instance
 * @param prefer_unified_memory Prefer devices with unified memory
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_initialize(MetalComputeAccelerator* accelerator,
                                          bool prefer_unified_memory);

/**
 * Get Metal device capabilities
 * @param accelerator Metal compute accelerator instance
 * @param capabilities Output device capabilities
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_get_device_capabilities(MetalComputeAccelerator* accelerator,
                                                       MetalDeviceCapabilities* capabilities);

// Matrix Operations

/**
 * Perform large-scale matrix multiplication (GEMM)
 * @param accelerator Metal compute accelerator instance
 * @param config Matrix operation configuration
 * @param matrix_a Input matrix A (row-major)
 * @param matrix_b Input matrix B (row-major)
 * @param result_matrix Output result matrix
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_matrix_multiply(MetalComputeAccelerator* accelerator,
                                               const MatrixOperationConfig* config,
                                               const float* matrix_a,
                                               const float* matrix_b,
                                               float* result_matrix);

/**
 * Perform batched matrix multiplication
 * @param accelerator Metal compute accelerator instance
 * @param batch_size Number of matrix pairs to multiply
 * @param config Matrix operation configuration
 * @param matrix_a_batch Array of input matrices A
 * @param matrix_b_batch Array of input matrices B  
 * @param result_batch Array of output result matrices
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_batched_matrix_multiply(MetalComputeAccelerator* accelerator,
                                                       uint32_t batch_size,
                                                       const MatrixOperationConfig* config,
                                                       const float* const* matrix_a_batch,
                                                       const float* const* matrix_b_batch,
                                                       float* const* result_batch);

/**
 * Perform matrix transpose operation
 * @param accelerator Metal compute accelerator instance
 * @param rows Number of rows in input matrix
 * @param cols Number of columns in input matrix
 * @param input_matrix Input matrix to transpose
 * @param output_matrix Output transposed matrix
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_matrix_transpose(MetalComputeAccelerator* accelerator,
                                                uint32_t rows,
                                                uint32_t cols,
                                                const float* input_matrix,
                                                float* output_matrix);

// Self-Attention Operations

/**
 * Compute high-efficiency self-attention
 * @param accelerator Metal compute accelerator instance
 * @param config Self-attention configuration
 * @param query_matrix Query matrix (seq_len x d_model)
 * @param key_matrix Key matrix (seq_len x d_model)
 * @param value_matrix Value matrix (seq_len x d_model)
 * @param attention_output Output attention matrix
 * @param attention_weights Optional output attention weights
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_self_attention(MetalComputeAccelerator* accelerator,
                                              const SelfAttentionConfig* config,
                                              const float* query_matrix,
                                              const float* key_matrix,
                                              const float* value_matrix,
                                              float* attention_output,
                                              float* attention_weights);

/**
 * Compute Flash Attention (memory-efficient attention)
 * @param accelerator Metal compute accelerator instance
 * @param config Self-attention configuration
 * @param query_matrix Query matrix
 * @param key_matrix Key matrix
 * @param value_matrix Value matrix
 * @param attention_output Output attention matrix
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_flash_attention(MetalComputeAccelerator* accelerator,
                                               const SelfAttentionConfig* config,
                                               const float* query_matrix,
                                               const float* key_matrix,
                                               const float* value_matrix,
                                               float* attention_output);

/**
 * Compute sparse attention with custom patterns
 * @param accelerator Metal compute accelerator instance
 * @param config Self-attention configuration
 * @param attention_mask Sparse attention mask (1=attend, 0=ignore)
 * @param query_matrix Query matrix
 * @param key_matrix Key matrix
 * @param value_matrix Value matrix
 * @param attention_output Output attention matrix
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_sparse_attention(MetalComputeAccelerator* accelerator,
                                                const SelfAttentionConfig* config,
                                                const bool* attention_mask,
                                                const float* query_matrix,
                                                const float* key_matrix,
                                                const float* value_matrix,
                                                float* attention_output);

// Vectorized Operations

/**
 * Compute vectorized softmax operation
 * @param accelerator Metal compute accelerator instance
 * @param config Vector operation configuration
 * @param input_vectors Input vectors to apply softmax
 * @param output_vectors Output softmax vectors
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_vectorized_softmax(MetalComputeAccelerator* accelerator,
                                                  const VectorOperationConfig* config,
                                                  const float* input_vectors,
                                                  float* output_vectors);

/**
 * Compute vectorized layer normalization
 * @param accelerator Metal compute accelerator instance
 * @param config Vector operation configuration
 * @param input_vectors Input vectors to normalize
 * @param gamma_weights Gamma scaling weights
 * @param beta_weights Beta bias weights
 * @param output_vectors Output normalized vectors
 * @param epsilon Normalization epsilon
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_vectorized_layer_norm(MetalComputeAccelerator* accelerator,
                                                     const VectorOperationConfig* config,
                                                     const float* input_vectors,
                                                     const float* gamma_weights,
                                                     const float* beta_weights,
                                                     float* output_vectors,
                                                     float epsilon);

/**
 * Compute vectorized GELU activation
 * @param accelerator Metal compute accelerator instance
 * @param config Vector operation configuration
 * @param input_vectors Input vectors
 * @param output_vectors Output GELU-activated vectors
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_vectorized_gelu(MetalComputeAccelerator* accelerator,
                                               const VectorOperationConfig* config,
                                               const float* input_vectors,
                                               float* output_vectors);

/**
 * Compute vectorized SwiGLU activation
 * @param accelerator Metal compute accelerator instance
 * @param config Vector operation configuration
 * @param input_vectors Input vectors
 * @param gate_vectors Gate vectors for SwiGLU
 * @param output_vectors Output SwiGLU-activated vectors
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_vectorized_swiglu(MetalComputeAccelerator* accelerator,
                                                 const VectorOperationConfig* config,
                                                 const float* input_vectors,
                                                 const float* gate_vectors,
                                                 float* output_vectors);

// Memory Management and Optimization

/**
 * Optimize memory layout for unified memory architecture
 * @param accelerator Metal compute accelerator instance
 * @param data_size Size of data in bytes
 * @param data_ptr Pointer to data to optimize
 * @param optimized_ptr Output optimized data pointer
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_optimize_memory_layout(MetalComputeAccelerator* accelerator,
                                                      size_t data_size,
                                                      const void* data_ptr,
                                                      void** optimized_ptr);

/**
 * Perform asynchronous memory transfer
 * @param accelerator Metal compute accelerator instance
 * @param src_ptr Source data pointer
 * @param dst_ptr Destination data pointer
 * @param size Number of bytes to transfer
 * @param wait_for_completion Wait for transfer completion
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_async_memory_transfer(MetalComputeAccelerator* accelerator,
                                                     const void* src_ptr,
                                                     void* dst_ptr,
                                                     size_t size,
                                                     bool wait_for_completion);

/**
 * Synchronize all pending GPU operations
 * @param accelerator Metal compute accelerator instance
 * @param timeout_ms Timeout in milliseconds (0 = infinite)
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_synchronize(MetalComputeAccelerator* accelerator,
                                           uint32_t timeout_ms);

// Performance Monitoring and Optimization

/**
 * Get Metal compute performance statistics
 * @param accelerator Metal compute accelerator instance
 * @param stats Output performance statistics
 */
void metal_compute_get_performance_stats(MetalComputeAccelerator* accelerator,
                                        MetalPerformanceStats* stats);

/**
 * Reset performance counters
 * @param accelerator Metal compute accelerator instance
 */
void metal_compute_reset_performance_counters(MetalComputeAccelerator* accelerator);

/**
 * Enable thermal monitoring and management
 * @param accelerator Metal compute accelerator instance
 * @param enable_throttling Enable automatic thermal throttling
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_enable_thermal_management(MetalComputeAccelerator* accelerator,
                                                         bool enable_throttling);

/**
 * Optimize shader pipeline for current workload
 * @param accelerator Metal compute accelerator instance
 * @param workload_type Type of workload to optimize for
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_optimize_shader_pipeline(MetalComputeAccelerator* accelerator,
                                                        const char* workload_type);

// Configuration and Utility Functions

/**
 * Create default Metal compute configuration
 * @param config Output default configuration
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_create_default_config(MetalComputeConfig* config);

/**
 * Create Apple Silicon optimized configuration
 * @param config Output Apple Silicon optimized configuration
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_create_apple_silicon_config(MetalComputeConfig* config);

/**
 * Create matrix operation configuration
 * @param rows_a Matrix A rows
 * @param cols_a Matrix A columns
 * @param cols_b Matrix B columns
 * @param config Output matrix operation configuration
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_create_matrix_config(uint32_t rows_a,
                                                    uint32_t cols_a,
                                                    uint32_t cols_b,
                                                    MatrixOperationConfig* config);

/**
 * Create self-attention configuration for transformer
 * @param sequence_length Input sequence length
 * @param d_model Model dimension
 * @param num_heads Number of attention heads
 * @param config Output self-attention configuration
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_create_attention_config(uint32_t sequence_length,
                                                       uint32_t d_model,
                                                       uint32_t num_heads,
                                                       SelfAttentionConfig* config);

/**
 * Validate Metal compute configuration
 * @param config Configuration to validate
 * @param device_capabilities Device capabilities for validation
 * @param is_valid Output boolean for validity
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_validate_config(const MetalComputeConfig* config,
                                               const MetalDeviceCapabilities* device_capabilities,
                                               bool* is_valid);

/**
 * Estimate compute requirements for operation
 * @param operation_type Type of operation (gemm, attention, etc.)
 * @param input_size Input data size
 * @param estimated_compute_units Output estimated compute units needed
 * @param estimated_memory_mb Output estimated memory requirement
 * @param estimated_time_us Output estimated execution time
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_estimate_requirements(const char* operation_type,
                                                     size_t input_size,
                                                     uint32_t* estimated_compute_units,
                                                     size_t* estimated_memory_mb,
                                                     uint64_t* estimated_time_us);

/**
 * Destroy Metal compute accelerator and free resources
 * @param accelerator Metal compute accelerator instance to destroy
 */
void metal_compute_destroy(MetalComputeAccelerator* accelerator);

// Utility Functions

/**
 * Get error string for Metal compute error code
 * @param error_code MetalComputeError code
 * @return Human-readable error message
 */
const char* metal_compute_get_error_string(MetalComputeError error_code);

/**
 * Get device name string
 * @param accelerator Metal compute accelerator instance
 * @return Device name string
 */
const char* metal_compute_get_device_name(MetalComputeAccelerator* accelerator);

/**
 * Check if operation is supported on current device
 * @param accelerator Metal compute accelerator instance
 * @param operation_name Name of operation to check
 * @return true if supported, false otherwise
 */
bool metal_compute_is_operation_supported(MetalComputeAccelerator* accelerator,
                                         const char* operation_name);

/**
 * Get optimal tile size for matrix operations
 * @param accelerator Metal compute accelerator instance
 * @param matrix_rows Number of matrix rows
 * @param matrix_cols Number of matrix columns
 * @return Optimal tile size
 */
uint32_t metal_compute_get_optimal_tile_size(MetalComputeAccelerator* accelerator,
                                            uint32_t matrix_rows,
                                            uint32_t matrix_cols);

/**
 * Get recommended thread configuration for vector operations
 * @param accelerator Metal compute accelerator instance
 * @param vector_length Length of vectors to process
 * @param threads_per_group Output recommended threads per group
 * @param elements_per_thread Output recommended elements per thread
 * @return METAL_COMPUTE_SUCCESS on success, error code on failure
 */
MetalComputeError metal_compute_get_vector_thread_config(MetalComputeAccelerator* accelerator,
                                                        uint32_t vector_length,
                                                        uint32_t* threads_per_group,
                                                        uint32_t* elements_per_thread);

// Constants for Metal compute acceleration
#define METAL_COMPUTE_DEFAULT_TILE_SIZE 64              // Default matrix tile size
#define METAL_COMPUTE_MAX_THREADS_PER_GROUP 1024        // Maximum threads per group
#define METAL_COMPUTE_SIMD_WIDTH 32                     // Apple GPU SIMD width
#define METAL_COMPUTE_MEMORY_ALIGNMENT 64               // Memory alignment requirement
#define METAL_COMPUTE_DEFAULT_TIMEOUT_MS 5000           // Default operation timeout

// Apple Silicon specific constants
#define APPLE_SILICON_UNIFIED_MEMORY_BANDWIDTH 400      // GB/s for M3 Max
#define APPLE_SILICON_MAX_COMPUTE_UNITS 40              // M3 Max compute units
#define APPLE_SILICON_MAX_BUFFER_SIZE_GB 96              // Maximum buffer size
#define APPLE_SILICON_THERMAL_THRESHOLD 85              // Thermal threshold (°C)

// Performance optimization constants
#define METAL_MATRIX_MULTIPLY_THRESHOLD 256             // Use GPU for matrices >= 256x256
#define METAL_ATTENTION_SEQUENCE_THRESHOLD 64           // Use GPU for sequences >= 64 tokens
#define METAL_VECTOR_BATCH_THRESHOLD 1024               // Use GPU for vectors >= 1024 elements
#define METAL_MEMORY_POOL_SIZE_MB 512                   // Default memory pool size

#ifdef __cplusplus
}
#endif

#endif // METAL_COMPUTE_ACCELERATOR_H
