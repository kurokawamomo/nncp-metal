#ifndef NNCP_LSTM_METAL_H
#define NNCP_LSTM_METAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __APPLE__
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLBuffer;
@protocol MTLComputePipelineState;
@protocol MTLCommandBuffer;
@protocol MTLComputeCommandEncoder;
#else
// C compatibility - forward declare as opaque pointers
typedef struct MTLDevice_t* MTLDevice_t;
typedef struct MTLCommandQueue_t* MTLCommandQueue_t;
typedef struct MTLBuffer_t* MTLBuffer_t;
typedef struct MTLComputePipelineState_t* MTLComputePipelineState_t;
typedef struct MTLCommandBuffer_t* MTLCommandBuffer_t;
typedef struct MTLComputeCommandEncoder_t* MTLComputeCommandEncoder_t;
#endif
#else
typedef void MTLDevice_t;
typedef void MTLCommandQueue_t;
typedef void MTLBuffer_t;
typedef void MTLComputePipelineState_t;
typedef void MTLCommandBuffer_t;
typedef void MTLComputeCommandEncoder_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for NNCP LSTM Metal operations
typedef enum {
    NNCP_LSTM_SUCCESS = 0,
    NNCP_LSTM_ERROR_INVALID_PARAM,
    NNCP_LSTM_ERROR_MEMORY_ALLOCATION,
    NNCP_LSTM_ERROR_DEVICE_NOT_FOUND,
    NNCP_LSTM_ERROR_COMPUTE_FAILED,
    NNCP_LSTM_ERROR_INVALID_DIMENSIONS,
    NNCP_LSTM_ERROR_BUFFER_ALLOCATION,
    NNCP_LSTM_ERROR_SHADER_COMPILATION,
    NNCP_LSTM_ERROR_EXECUTION_FAILED,
    NNCP_LSTM_ERROR_UNSUPPORTED_OPERATION
} NNCPLSTMError;

// NNCP LSTM Configuration (matching CUDA implementation)
typedef struct {
    // Architecture parameters (from CUDA reference)
    uint32_t n_layers;              // Number of LSTM layers (default: 4)
    uint32_t n_cells;               // Hidden state dimension (default: 352)
    uint32_t n_cells2;              // Cell state dimension (default: 352, no projection)
    uint32_t n_symbols;             // Vocabulary size (default: 256)
    uint32_t mat_count;             // Number of gate matrices (default: 3 for CLAMPED)
    
    // Sequence processing parameters
    uint32_t batch_size;            // Batch processing size (default: 32)
    uint32_t seg_len;               // Sequence segment length (default: 20)
    uint32_t n_embed_out;           // Output embedding layers (default: 4)
    
    // LSTM variant and features
    bool use_layer_norm;            // Enable RMS layer normalization (default: true)
    bool full_connect;              // Connect all previous layers (default: true)
    float forget_bias;              // Additional forget gate bias (default: 0.0)
    float layer_norm_eps;           // Layer norm epsilon (default: 1e-5)
    
    // Memory and performance
    uint32_t max_memory_mb;         // Maximum memory usage limit
    bool verbose_logging;           // Enable detailed logging
} NNCPLSTMConfig;

// NNCP LSTM Weight Initialization (matching CUDA scheme)
typedef struct {
    float variance_u;               // Recurrent weight variance (1.0/sqrt(n_cells))
    float variance_w;               // Dense input weight variance (1.0/sqrt(n_inputs))
    float variance_ws;              // Sparse embedding variance (0.75)
    float variance_fc;              // Output projection variance
    uint32_t random_seed;           // Random seed for reproducibility
} NNCPLSTMWeightInit;

// NNCP LSTM Buffer Set
typedef struct {
    // Weight matrices (concatenated for efficiency)
    void* u_weights;                // Recurrent weights [n_cells2*mat_count, n_cells]
    void* w_weights;                // Dense input weights [n_cells2*mat_count, n_inputs]
    void* ws_weights;               // Sparse embedding [n_cells2*mat_count, n_symbols]
    
    // Bias vectors (per gate, per layer)
    void* b_forget;                 // Forget gate bias [n_cells2]
    void* b_input;                  // Input gate bias [n_cells2]
    void* b_output;                 // Output gate bias [n_cells2]
    
    // Layer normalization parameters
    void* g_forget;                 // Forget gate scale [n_cells2]
    void* g_input;                  // Input gate scale [n_cells2]
    void* g_output;                 // Output gate scale [n_cells2]
    
    // State buffers
    void* h_states;                 // Hidden states [n_cells, n_streams]
    void* c_states;                 // Cell states [n_cells2, n_streams]
    void* h0_states;                // Initial hidden states [n_cells, n_streams]
    void* c0_states;                // Initial cell states [n_cells2, n_streams]
    
    // Input/Output buffers
    void* input_symbols;            // Input symbol sequence [n_streams, seg_len]
    void* output_logits;            // Output logits [n_streams*seg_len, n_symbols]
    
    // Temporary computation buffers
    void* gate_inputs;              // Gate input buffer [n_cells2*mat_count, n_streams]
    void* gate_outputs;             // Gate output buffer [n_cells2*mat_count, n_streams]
    void* temp_buffers[8];          // Additional temporary buffers
    
    // Buffer metadata
    size_t buffer_sizes[32];        // Sizes of all buffers
    bool buffers_allocated;         // Allocation status
} NNCPLSTMBuffers;

// NNCP LSTM Compute Pipeline States
typedef struct {
    void* sparse_lookup_pipeline;      // Sparse embedding lookup shader
    void* matrix_multiply_pipeline;     // Matrix multiplication shader
    void* gate_computation_pipeline;    // Gate computation shader
    void* layer_norm_pipeline;          // RMS layer normalization shader
    void* lstm_clamped_pipeline;        // nc_lstm_clamped operation shader
    void* output_projection_pipeline;   // Output projection shader
    
    bool pipelines_compiled;            // Compilation status
} NNCPLSTMPipelines;

// NNCP LSTM Performance Statistics
typedef struct {
    uint64_t total_operations;          // Total forward passes
    uint64_t total_compute_time_ns;     // Total computation time in nanoseconds
    uint32_t memory_usage_mb;           // Current memory usage in MB
    uint64_t average_operation_time_ns; // Average time per operation
    float gflops_achieved;              // GFLOPS performance
    float memory_bandwidth_gbps;        // Memory bandwidth utilization
    uint32_t cache_hit_rate_percent;    // Cache hit rate percentage
} NNCPLSTMPerformanceStats;

// Main NNCP LSTM Context
typedef struct {
    NNCPLSTMConfig config;              // LSTM configuration
    NNCPLSTMBuffers buffers;            // Metal buffers
    NNCPLSTMPipelines pipelines;        // Compute pipelines
    
    // Metal objects
    void* device;                       // MTLDevice
    void* command_queue;                // MTLCommandQueue
    
    // State management
    bool is_initialized;                // Initialization status
    bool has_weights;                   // Whether weights are loaded
    bool has_state;                     // Whether state is maintained
    
    // Performance tracking
    NNCPLSTMPerformanceStats performance; // Performance statistics
} NNCPLSTMContext;

// Tensor shape for NNCP LSTM operations
typedef struct {
    uint32_t n_streams;                 // Batch size (number of parallel sequences)
    uint32_t seq_len;                   // Sequence length
    uint32_t n_layers;                  // Number of LSTM layers
    uint32_t n_cells;                   // Hidden dimension
    uint32_t n_symbols;                 // Vocabulary size
} NNCPLSTMShape;

// NNCP LSTM Metal Compute Parameters
typedef struct {
    uint32_t threads_per_threadgroup;   // Threads per threadgroup
    uint32_t threadgroups_per_grid_x;   // Threadgroups in X dimension
    uint32_t threadgroups_per_grid_y;   // Threadgroups in Y dimension
    uint32_t threadgroups_per_grid_z;   // Threadgroups in Z dimension
} NNCPLSTMComputeParams;

// Core API Functions

/**
 * Create and initialize NNCP LSTM Metal context
 * @param context Pointer to store created context
 * @param config NNCP LSTM configuration
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_create(NNCPLSTMContext** context, const NNCPLSTMConfig* config);

/**
 * Initialize weights with CUDA-compatible scheme
 * @param context NNCP LSTM context
 * @param weight_init Weight initialization parameters
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_init_weights(NNCPLSTMContext* context, 
                                          const NNCPLSTMWeightInit* weight_init);

/**
 * Load pre-trained weights (CUDA format compatibility)
 * @param context NNCP LSTM context
 * @param layer Layer index
 * @param u_weights Recurrent weights
 * @param w_weights Dense input weights (NULL for layer 0)
 * @param ws_weights Sparse embedding weights (layer 0 only)
 * @param biases Bias vectors
 * @param layer_norm_params Layer normalization parameters
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_load_layer_weights(NNCPLSTMContext* context,
                                                 uint32_t layer,
                                                 const float* u_weights,
                                                 const float* w_weights,
                                                 const float* ws_weights,
                                                 const float* biases,
                                                 const float* layer_norm_params);

/**
 * Forward pass through NNCP LSTM (CUDA-compatible)
 * @param context NNCP LSTM context
 * @param input_symbols Input symbol sequence [n_streams, seg_len]
 * @param output_logits Output logits [n_streams*seg_len, n_symbols]
 * @param n_streams Number of parallel sequences
 * @param seg_len Sequence length
 * @param reset_state Whether to reset hidden/cell states
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_forward(NNCPLSTMContext* context,
                                     const uint32_t* input_symbols,
                                     float* output_logits,
                                     uint32_t n_streams,
                                     uint32_t seg_len,
                                     bool reset_state);

/**
 * Process single timestep through LSTM layers
 * @param context NNCP LSTM context
 * @param layer Layer index
 * @param input_symbols Current timestep input symbols [n_streams]
 * @param dense_input Dense input from previous layer [n_cells*n_streams] (NULL for layer 0)
 * @param hidden_output Output hidden states [n_cells*n_streams]
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_process_timestep(NNCPLSTMContext* context,
                                              uint32_t layer,
                                              const uint32_t* input_symbols,
                                              const float* dense_input,
                                              float* hidden_output);

/**
 * Reset LSTM hidden and cell states to initial values
 * @param context NNCP LSTM context
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_reset_state(NNCPLSTMContext* context);

/**
 * Get current LSTM states
 * @param context NNCP LSTM context
 * @param layer Layer index
 * @param hidden_state Output hidden state [n_cells*n_streams]
 * @param cell_state Output cell state [n_cells2*n_streams]
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_get_state(NNCPLSTMContext* context,
                                       uint32_t layer,
                                       float* hidden_state,
                                       float* cell_state);

/**
 * Set LSTM states
 * @param context NNCP LSTM context  
 * @param layer Layer index
 * @param hidden_state Input hidden state [n_cells*n_streams]
 * @param cell_state Input cell state [n_cells2*n_streams]
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_set_state(NNCPLSTMContext* context,
                                       uint32_t layer,
                                       const float* hidden_state,
                                       const float* cell_state);

// Configuration Functions

/**
 * Create default NNCP LSTM configuration (CUDA-compatible)
 * @param config Pointer to store default configuration
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_config_create_default(NNCPLSTMConfig* config);

/**
 * Create LSTM model profile configuration
 * @param config Pointer to store model configuration
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_config_create_lstm_model(NNCPLSTMConfig* config);

/**
 * Create fast LSTM profile configuration
 * @param config Pointer to store fast configuration
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_config_create_fast_lstm(NNCPLSTMConfig* config);

/**
 * Validate NNCP LSTM configuration
 * @param config LSTM configuration to validate
 * @return NNCP_LSTM_SUCCESS if valid, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_config_validate(const NNCPLSTMConfig* config);

// Weight Initialization Functions

/**
 * Create default weight initialization parameters (CUDA-compatible)
 * @param weight_init Pointer to store weight initialization
 * @param config LSTM configuration
 * @param seed Random seed
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_weight_init_create_default(NNCPLSTMWeightInit* weight_init,
                                                        const NNCPLSTMConfig* config,
                                                        uint32_t seed);

// Utility Functions

/**
 * Get error message string
 * @param error_code NNCP LSTM error code
 * @return Human-readable error message
 */
const char* nncp_lstm_metal_get_error_string(NNCPLSTMError error_code);

/**
 * Check if NNCP LSTM Metal is available on current platform
 * @return true if available, false otherwise
 */
bool nncp_lstm_metal_is_available(void);

/**
 * Get Metal device information
 * @param device_name Buffer to store device name
 * @param buffer_size Size of device name buffer
 * @param max_memory_mb Pointer to store maximum memory in MB
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_get_device_info(char* device_name,
                                             size_t buffer_size,
                                             uint32_t* max_memory_mb);

/**
 * Calculate memory requirements for NNCP LSTM configuration
 * @param config LSTM configuration
 * @param memory_mb Pointer to store memory requirement in MB
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_calculate_memory_requirements(const NNCPLSTMConfig* config,
                                                           uint32_t* memory_mb);

/**
 * Get performance statistics
 * @param context NNCP LSTM context
 * @param avg_latency_ms Average latency per forward pass in milliseconds
 * @param gflops_achieved GFLOPS performance achieved
 * @param memory_usage_mb Current memory usage in MB
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_get_performance_stats(NNCPLSTMContext* context,
                                                   float* avg_latency_ms,
                                                   float* gflops_achieved,
                                                   uint32_t* memory_usage_mb);

/**
 * Destroy NNCP LSTM context
 * @param context NNCP LSTM context to destroy
 */
void nncp_lstm_metal_destroy(NNCPLSTMContext* context);

// Advanced Functions

/**
 * Compile Metal compute shaders for NNCP LSTM
 * @param context NNCP LSTM context
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_compile_shaders(NNCPLSTMContext* context);

/**
 * Validate Metal compute pipeline functionality
 * @param context NNCP LSTM context
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_validate_pipelines(NNCPLSTMContext* context);

/**
 * Benchmark NNCP LSTM Metal performance
 * @param context NNCP LSTM context
 * @param iterations Number of benchmark iterations
 * @param test_data_size Size of test data
 * @return NNCP_LSTM_SUCCESS on success, error code on failure
 */
NNCPLSTMError nncp_lstm_metal_benchmark(NNCPLSTMContext* context,
                                       uint32_t iterations,
                                       uint32_t test_data_size);

#ifdef __cplusplus
}
#endif

#endif // NNCP_LSTM_METAL_H
