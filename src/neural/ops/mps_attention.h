#ifndef MPS_ATTENTION_H
#define MPS_ATTENTION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __OBJC__
@class MPSGraph;
@class MPSGraphTensor;
@class MPSGraphExecutable;
@class MPSCommandBuffer;
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLBuffer;
#else
typedef void MPSGraph;
typedef void MPSGraphTensor;
typedef void MPSGraphExecutable;
typedef void MPSCommandBuffer;
typedef void* MTLDevice_t;
typedef void* MTLCommandQueue_t;
typedef void* MTLBuffer_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for MPS Attention operations
typedef enum {
    MPS_ATTENTION_SUCCESS = 0,
    MPS_ATTENTION_ERROR_INVALID_PARAM,
    MPS_ATTENTION_ERROR_MEMORY_ALLOCATION,
    MPS_ATTENTION_ERROR_DEVICE_NOT_FOUND,
    MPS_ATTENTION_ERROR_COMPUTE_FAILED,
    MPS_ATTENTION_ERROR_INVALID_DIMENSIONS,
    MPS_ATTENTION_ERROR_BUFFER_ALLOCATION,
    MPS_ATTENTION_ERROR_GRAPH_COMPILATION,
    MPS_ATTENTION_ERROR_EXECUTION_FAILED,
    MPS_ATTENTION_ERROR_UNSUPPORTED_OPERATION
} MPSAttentionError;

// Attention types
typedef enum {
    MPS_ATTENTION_SCALED_DOT_PRODUCT = 0,
    MPS_ATTENTION_MULTI_HEAD,
    MPS_ATTENTION_SELF_ATTENTION,
    MPS_ATTENTION_CROSS_ATTENTION,
    MPS_ATTENTION_CAUSAL,
    MPS_ATTENTION_SPARSE
} MPSAttentionType;

// Attention configuration
typedef struct {
    uint32_t sequence_length;          // Input sequence length
    uint32_t hidden_size;              // Hidden dimension size
    uint32_t num_heads;                // Number of attention heads
    uint32_t head_dim;                 // Dimension per head (hidden_size / num_heads)
    uint32_t batch_size;               // Batch size
    uint32_t max_sequence_length;      // Maximum supported sequence length
    float scale_factor;                // Attention scale factor (1/sqrt(head_dim))
    bool use_bias;                     // Whether to use bias in linear projections
    bool use_causal_mask;              // Whether to apply causal masking
    bool use_key_padding_mask;         // Whether to use key padding mask
    float dropout_rate;                // Attention dropout rate
    MPSAttentionType attention_type;   // Type of attention mechanism
} MPSAttentionConfig;

// Attention tensor shapes
typedef struct {
    uint32_t batch_size;               // Batch dimension
    uint32_t sequence_length;          // Sequence dimension  
    uint32_t hidden_size;              // Hidden dimension
    uint32_t num_heads;                // Number of heads
    uint32_t head_dim;                 // Head dimension
} MPSAttentionShape;

// Attention weights and biases
typedef struct {
    void* query_weights;               // Query projection weights [hidden_size, hidden_size]
    void* key_weights;                 // Key projection weights [hidden_size, hidden_size]
    void* value_weights;               // Value projection weights [hidden_size, hidden_size]
    void* output_weights;              // Output projection weights [hidden_size, hidden_size]
    void* query_bias;                  // Query projection bias [hidden_size] (optional)
    void* key_bias;                    // Key projection bias [hidden_size] (optional)
    void* value_bias;                  // Value projection bias [hidden_size] (optional)
    void* output_bias;                 // Output projection bias [hidden_size] (optional)
    size_t weights_size;               // Size of each weight matrix in bytes
    size_t bias_size;                  // Size of each bias vector in bytes
} MPSAttentionWeights;

// Metal buffers for attention computation
typedef struct {
    void* query_buffer;         // Query tensor buffer
    void* key_buffer;           // Key tensor buffer  
    void* value_buffer;         // Value tensor buffer
    void* output_buffer;        // Output tensor buffer
    void* weights_buffer;       // Weights buffer
    void* mask_buffer;          // Attention mask buffer (optional)
    void* temp_buffers[4];      // Temporary computation buffers
    size_t buffer_sizes[8];            // Sizes of all buffers
} MPSAttentionBuffers;

// Performance statistics
typedef struct {
    uint64_t total_operations;         // Total attention operations performed
    uint64_t total_compute_time_ns;    // Total compute time in nanoseconds
    uint64_t total_memory_transferred; // Total memory transferred in bytes
    uint32_t batch_size_sum;           // Sum of batch sizes processed
    uint32_t sequence_length_sum;      // Sum of sequence lengths processed
    float average_compute_time_ms;     // Average compute time per operation
    float gflops_achieved;             // GFLOPS performance achieved
    uint32_t memory_usage_mb;          // Current memory usage in MB
    uint32_t peak_memory_usage_mb;     // Peak memory usage in MB
} MPSAttentionStats;

// Main MPS Attention context
typedef struct {
    MPSAttentionConfig config;         // Attention configuration
    MPSAttentionStats stats;           // Performance statistics
    MPSAttentionBuffers buffers;       // Metal buffers
    MPSAttentionWeights weights;       // Model weights and biases
    
    // Metal/MPS objects
    void* device;               // Metal device
    void* command_queue;        // Metal command queue
    void* mps_graph;                   // MPS computation graph
    void* graph_executable;            // Compiled graph executable
    
    // Internal state
    bool is_initialized;               // Whether context is initialized
    uint32_t current_batch_size;      // Current batch size being processed
    uint32_t current_sequence_length; // Current sequence length being processed
    void* platform_context;           // Platform-specific context
} MPSAttentionContext;

// Attention input/output tensors
typedef struct {
    float* query;                      // Query tensor [batch, seq_len, hidden_size]
    float* key;                        // Key tensor [batch, seq_len, hidden_size]  
    float* value;                      // Value tensor [batch, seq_len, hidden_size]
    float* output;                     // Output tensor [batch, seq_len, hidden_size]
    float* attention_mask;             // Attention mask [batch, seq_len, seq_len] (optional)
    float* key_padding_mask;           // Key padding mask [batch, seq_len] (optional)
    MPSAttentionShape shape;           // Tensor shapes
    bool owns_memory;                  // Whether this structure owns the memory
} MPSAttentionTensors;

// Core API Functions

/**
 * Create and initialize MPS attention context
 * @param context Pointer to store created context
 * @param config Attention configuration
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_create(MPSAttentionContext** context, 
                                      const MPSAttentionConfig* config);

/**
 * Load attention weights from memory
 * @param context MPS attention context
 * @param weights Attention weights and biases
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_load_weights(MPSAttentionContext* context,
                                            const MPSAttentionWeights* weights);

/**
 * Perform forward pass of attention mechanism
 * @param context MPS attention context
 * @param tensors Input/output tensors
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_forward(MPSAttentionContext* context,
                                       MPSAttentionTensors* tensors);

/**
 * Perform multi-head self-attention
 * @param context MPS attention context
 * @param input Input tensor [batch, seq_len, hidden_size]
 * @param output Output tensor [batch, seq_len, hidden_size]
 * @param attention_mask Optional attention mask [batch, seq_len, seq_len]
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_self_attention(MPSAttentionContext* context,
                                              const float* input,
                                              float* output,
                                              const float* attention_mask);

/**
 * Perform cross-attention between query and key-value pairs
 * @param context MPS attention context
 * @param query Query tensor [batch, seq_len_q, hidden_size]
 * @param key Key tensor [batch, seq_len_kv, hidden_size]
 * @param value Value tensor [batch, seq_len_kv, hidden_size]
 * @param output Output tensor [batch, seq_len_q, hidden_size]
 * @param attention_mask Optional attention mask [batch, seq_len_q, seq_len_kv]
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_cross_attention(MPSAttentionContext* context,
                                               const float* query,
                                               const float* key,
                                               const float* value,
                                               float* output,
                                               const float* attention_mask);

// Configuration Functions

/**
 * Create default attention configuration
 * @param config Pointer to store default configuration
 * @param sequence_length Input sequence length
 * @param hidden_size Hidden dimension size
 * @param num_heads Number of attention heads
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_config_create_default(MPSAttentionConfig* config,
                                                     uint32_t sequence_length,
                                                     uint32_t hidden_size,
                                                     uint32_t num_heads);

/**
 * Validate attention configuration
 * @param config Configuration to validate
 * @return MPS_ATTENTION_SUCCESS if valid, error code otherwise
 */
MPSAttentionError mps_attention_config_validate(const MPSAttentionConfig* config);

/**
 * Calculate memory requirements for attention operation
 * @param config Attention configuration
 * @param memory_mb Pointer to store memory requirement in MB
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_calculate_memory_requirements(const MPSAttentionConfig* config,
                                                             uint32_t* memory_mb);

// Buffer Management Functions

/**
 * Allocate Metal buffers for attention computation
 * @param context MPS attention context
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_allocate_buffers(MPSAttentionContext* context);

/**
 * Free Metal buffers
 * @param context MPS attention context
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_free_buffers(MPSAttentionContext* context);

/**
 * Copy data to Metal buffers
 * @param context MPS attention context
 * @param tensors Input tensors to copy
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_upload_tensors(MPSAttentionContext* context,
                                              const MPSAttentionTensors* tensors);

/**
 * Copy results from Metal buffers
 * @param context MPS attention context
 * @param tensors Output tensors to copy to
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_download_tensors(MPSAttentionContext* context,
                                                MPSAttentionTensors* tensors);

// Graph Compilation Functions

/**
 * Build MPS computation graph for attention
 * @param context MPS attention context
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_build_graph(MPSAttentionContext* context);

/**
 * Compile MPS graph for execution
 * @param context MPS attention context
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_compile_graph(MPSAttentionContext* context);

// Optimization Functions

/**
 * Optimize attention for specific input sizes
 * @param context MPS attention context
 * @param batch_size Target batch size
 * @param sequence_length Target sequence length
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_optimize_for_size(MPSAttentionContext* context,
                                                  uint32_t batch_size,
                                                  uint32_t sequence_length);

/**
 * Enable/disable attention optimizations
 * @param context MPS attention context
 * @param enable_flash_attention Enable Flash Attention optimization
 * @param enable_memory_efficient Enable memory-efficient attention
 * @param enable_gradient_checkpointing Enable gradient checkpointing
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_set_optimizations(MPSAttentionContext* context,
                                                  bool enable_flash_attention,
                                                  bool enable_memory_efficient,
                                                  bool enable_gradient_checkpointing);

// Statistics and Monitoring Functions

/**
 * Get attention performance statistics
 * @param context MPS attention context
 * @param stats Pointer to store statistics
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_get_stats(MPSAttentionContext* context,
                                         MPSAttentionStats* stats);

/**
 * Reset performance statistics
 * @param context MPS attention context
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_reset_stats(MPSAttentionContext* context);

/**
 * Get current memory usage
 * @param context MPS attention context
 * @param memory_usage_mb Pointer to store memory usage in MB
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_get_memory_usage(MPSAttentionContext* context,
                                                uint32_t* memory_usage_mb);

// Tensor Management Functions

/**
 * Create attention tensors structure
 * @param tensors Pointer to store created tensors
 * @param shape Tensor shapes
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_tensors_create(MPSAttentionTensors** tensors,
                                              const MPSAttentionShape* shape);

/**
 * Destroy attention tensors structure
 * @param tensors Tensors to destroy
 */
void mps_attention_tensors_destroy(MPSAttentionTensors* tensors);

/**
 * Validate tensor dimensions
 * @param tensors Tensors to validate
 * @param config Attention configuration
 * @return MPS_ATTENTION_SUCCESS if valid, error code otherwise
 */
MPSAttentionError mps_attention_tensors_validate(const MPSAttentionTensors* tensors,
                                                const MPSAttentionConfig* config);

// Utility Functions

/**
 * Get error message string
 * @param error_code MPS attention error code
 * @return Human-readable error message
 */
const char* mps_attention_get_error_string(MPSAttentionError error_code);

/**
 * Check if MPS is available on current device
 * @return true if MPS is available, false otherwise
 */
bool mps_attention_is_available(void);

/**
 * Get MPS device information
 * @param device_name Buffer to store device name
 * @param buffer_size Size of device name buffer
 * @param compute_units Pointer to store number of compute units
 * @param max_memory_mb Pointer to store maximum memory in MB
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_get_device_info(char* device_name,
                                               size_t buffer_size,
                                               uint32_t* compute_units,
                                               uint32_t* max_memory_mb);

/**
 * Create attention weights structure
 * @param weights Pointer to store created weights
 * @param config Attention configuration
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_weights_create(MPSAttentionWeights** weights,
                                              const MPSAttentionConfig* config);

/**
 * Initialize weights with random values
 * @param weights Weights structure to initialize
 * @param config Attention configuration
 * @param seed Random seed
 * @return MPS_ATTENTION_SUCCESS on success, error code on failure
 */
MPSAttentionError mps_attention_weights_init_random(MPSAttentionWeights* weights,
                                                   const MPSAttentionConfig* config,
                                                   uint32_t seed);

/**
 * Destroy attention weights structure
 * @param weights Weights to destroy
 */
void mps_attention_weights_destroy(MPSAttentionWeights* weights);

/**
 * Destroy MPS attention context
 * @param context Context to destroy
 */
void mps_attention_destroy(MPSAttentionContext* context);

#ifdef __cplusplus
}
#endif

#endif // MPS_ATTENTION_H