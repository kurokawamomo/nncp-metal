#ifndef MPS_LSTM_H
#define MPS_LSTM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __APPLE__
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLBuffer;
@class MPSGraph;
@class MPSGraphExecutable;
@class MPSCommandBuffer;
#else
// C compatibility - forward declare as opaque pointers
typedef struct MTLDevice_t* MTLDevice_t;
typedef struct MTLCommandQueue_t* MTLCommandQueue_t;
typedef struct MTLBuffer_t* MTLBuffer_t;
typedef struct MPSGraph* MPSGraph;
typedef struct MPSGraphExecutable* MPSGraphExecutable;
typedef struct MPSCommandBuffer* MPSCommandBuffer;
#endif
#else
typedef void MTLDevice_t;
typedef void MTLCommandQueue_t;
typedef void MTLBuffer_t;
typedef void MPSGraph;
typedef void MPSGraphExecutable;
typedef void MPSCommandBuffer;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for MPS LSTM operations
typedef enum {
    MPS_LSTM_SUCCESS = 0,
    MPS_LSTM_ERROR_INVALID_PARAM,
    MPS_LSTM_ERROR_MEMORY_ALLOCATION,
    MPS_LSTM_ERROR_DEVICE_NOT_FOUND,
    MPS_LSTM_ERROR_COMPUTE_FAILED,
    MPS_LSTM_ERROR_INVALID_DIMENSIONS,
    MPS_LSTM_ERROR_BUFFER_ALLOCATION,
    MPS_LSTM_ERROR_GRAPH_COMPILATION,
    MPS_LSTM_ERROR_EXECUTION_FAILED,
    MPS_LSTM_ERROR_UNSUPPORTED_OPERATION
} MPSLSTMError;

// LSTM activation functions
typedef enum {
    MPS_LSTM_ACTIVATION_TANH = 0,
    MPS_LSTM_ACTIVATION_SIGMOID,
    MPS_LSTM_ACTIVATION_RELU,
    MPS_LSTM_ACTIVATION_HARD_SIGMOID,
    MPS_LSTM_ACTIVATION_CUSTOM
} MPSLSTMActivation;

// LSTM direction configuration
typedef enum {
    MPS_LSTM_FORWARD = 0,
    MPS_LSTM_BACKWARD,
    MPS_LSTM_BIDIRECTIONAL
} MPSLSTMDirection;

// LSTM cell type
typedef enum {
    MPS_LSTM_VANILLA = 0,
    MPS_LSTM_PEEPHOLE,
    MPS_LSTM_COUPLED_INPUT_FORGET,
    MPS_LSTM_GRU_STYLE
} MPSLSTMCellType;

// LSTM configuration
typedef struct {
    uint32_t input_size;           // Input feature dimension
    uint32_t hidden_size;          // Hidden state dimension
    uint32_t num_layers;           // Number of stacked LSTM layers
    uint32_t batch_size;           // Batch size for processing
    uint32_t sequence_length;      // Maximum sequence length
    MPSLSTMDirection direction;    // Forward/Backward/Bidirectional
    MPSLSTMCellType cell_type;     // LSTM cell variant
    
    // Activation functions
    MPSLSTMActivation input_activation;   // Input gate activation
    MPSLSTMActivation forget_activation;  // Forget gate activation
    MPSLSTMActivation output_activation;  // Output gate activation
    MPSLSTMActivation cell_activation;    // Cell state activation
    
    // Dropout configuration
    float dropout_rate;            // Dropout probability (0.0 = disabled)
    bool use_dropout_on_input;     // Apply dropout to input
    bool use_dropout_on_hidden;    // Apply dropout to hidden states
    bool use_dropout_on_output;    // Apply dropout to output
    
    // Performance options
    bool use_bias;                 // Enable bias terms
    bool use_peepholes;            // Enable peephole connections
    bool use_layer_norm;           // Enable layer normalization
    bool use_residual_connections; // Enable residual connections
    bool bidirectional_merge_mode; // For bidirectional: concat or sum
    
    // Memory optimization
    bool stateful;                 // Maintain state between calls
    bool return_sequences;         // Return full sequence or last output
    bool return_state;             // Return final hidden/cell states
    uint32_t max_memory_mb;        // Maximum memory usage limit
} MPSLSTMConfig;

// LSTM tensor dimensions
typedef struct {
    uint32_t batch_size;           // Batch dimension
    uint32_t sequence_length;      // Sequence dimension
    uint32_t input_size;           // Input feature dimension
    uint32_t hidden_size;          // Hidden state dimension
    uint32_t num_layers;           // Number of layers
    uint32_t num_directions;       // 1 for unidirectional, 2 for bidirectional
} MPSLSTMShape;

// LSTM weights structure
typedef struct {
    // Input-to-hidden weights [input_size x hidden_size * 4]
    void* input_weights;           // Combined i,f,g,o gates
    
    // Hidden-to-hidden weights [hidden_size x hidden_size * 4]
    void* hidden_weights;          // Combined i,f,g,o gates
    
    // Bias terms [hidden_size * 4] (optional)
    void* input_bias;              // Input bias
    void* hidden_bias;             // Hidden bias
    
    // Peephole weights [hidden_size] (optional)
    void* input_peephole;          // Input gate peephole
    void* forget_peephole;         // Forget gate peephole
    void* output_peephole;         // Output gate peephole
    
    // Layer normalization parameters (optional)
    void* layer_norm_weights;      // Layer norm weights
    void* layer_norm_bias;         // Layer norm bias
    
    // Weight dimensions and memory info
    size_t input_weights_size;     // Size of input weights in bytes
    size_t hidden_weights_size;    // Size of hidden weights in bytes
    size_t bias_size;              // Size of bias in bytes
    size_t peephole_size;          // Size of peephole weights in bytes
    size_t layer_norm_size;        // Size of layer norm params in bytes
    bool owns_memory;              // Whether this structure owns the memory
} MPSLSTMWeights;

// LSTM tensors for computation
typedef struct {
    // Input data [batch_size x sequence_length x input_size]
    float* input;                  // Input sequence
    
    // Initial states [batch_size x num_layers * num_directions x hidden_size]
    float* initial_hidden;         // Initial hidden state (optional)
    float* initial_cell;           // Initial cell state (optional)
    
    // Output data [batch_size x sequence_length x hidden_size * num_directions]
    float* output;                 // Output sequence
    
    // Final states [batch_size x num_layers * num_directions x hidden_size]
    float* final_hidden;           // Final hidden state (optional)
    float* final_cell;             // Final cell state (optional)
    
    // Mask for variable length sequences (optional)
    float* sequence_mask;          // Mask for padding tokens
    
    MPSLSTMShape shape;            // Tensor dimensions
    bool owns_memory;              // Whether tensors own their memory
} MPSLSTMTensors;

// LSTM performance statistics
typedef struct {
    uint64_t total_operations;     // Total number of forward passes
    uint64_t total_compute_time_ns; // Total computation time in nanoseconds
    float average_compute_time_ms; // Average computation time in milliseconds
    float gflops_achieved;         // GFLOPS performance
    uint32_t memory_usage_mb;      // Current memory usage
    uint32_t peak_memory_usage_mb; // Peak memory usage
    uint64_t sequence_length_sum;  // Sum of all processed sequence lengths
    uint64_t batch_size_sum;       // Sum of all processed batch sizes
} MPSLSTMStats;

// Metal buffers for LSTM computation
typedef struct {
    void* input_buffer;            // Input tensor buffer
    void* hidden_buffer;           // Hidden state buffer
    void* cell_buffer;             // Cell state buffer
    void* output_buffer;           // Output tensor buffer
    void* weights_buffer;          // Weights buffer
    void* bias_buffer;             // Bias buffer
    void* temp_buffers[8];         // Temporary computation buffers
    size_t buffer_sizes[16];       // Sizes of all buffers
} MPSLSTMBuffers;

// Main LSTM context
typedef struct {
    MPSLSTMConfig config;          // LSTM configuration
    MPSLSTMStats stats;            // Performance statistics
    MPSLSTMBuffers buffers;        // Metal buffers
    MPSLSTMWeights weights;        // LSTM weights
    
    // Metal objects
    void* device;                  // Metal device
    void* command_queue;           // Metal command queue
    void* mps_graph;               // MPS computation graph
    void* graph_executable;        // Compiled graph executable
    
    // State management
    bool is_initialized;           // Initialization status
    bool has_weights;              // Whether weights are loaded
    bool has_state;                // Whether state is maintained
    void* platform_context;       // Platform-specific context
} MPSLSTMContext;

// Core API Functions

/**
 * Create and initialize MPS LSTM context
 * @param context Pointer to store created context
 * @param config LSTM configuration
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_create(MPSLSTMContext** context, const MPSLSTMConfig* config);

/**
 * Load weights into LSTM context
 * @param context LSTM context
 * @param weights LSTM weights structure
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_load_weights(MPSLSTMContext* context, const MPSLSTMWeights* weights);

/**
 * Forward pass through LSTM
 * @param context LSTM context
 * @param tensors Input/output tensors
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_forward(MPSLSTMContext* context, MPSLSTMTensors* tensors);

/**
 * Process single sequence through LSTM
 * @param context LSTM context
 * @param input Input sequence [sequence_length x input_size]
 * @param output Output sequence [sequence_length x hidden_size * num_directions]
 * @param sequence_length Actual sequence length
 * @param initial_hidden Initial hidden state (optional)
 * @param initial_cell Initial cell state (optional)
 * @param final_hidden Final hidden state output (optional)
 * @param final_cell Final cell state output (optional)
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_process_sequence(MPSLSTMContext* context,
                                      const float* input,
                                      float* output,
                                      uint32_t sequence_length,
                                      const float* initial_hidden,
                                      const float* initial_cell,
                                      float* final_hidden,
                                      float* final_cell);

/**
 * Process batch of sequences through LSTM
 * @param context LSTM context
 * @param input Input batch [batch_size x sequence_length x input_size]
 * @param output Output batch [batch_size x sequence_length x hidden_size * num_directions]
 * @param sequence_lengths Array of actual sequence lengths for each batch item
 * @param batch_size Number of sequences in batch
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_process_batch(MPSLSTMContext* context,
                                   const float* input,
                                   float* output,
                                   const uint32_t* sequence_lengths,
                                   uint32_t batch_size);

/**
 * Reset LSTM hidden and cell states
 * @param context LSTM context
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_reset_state(MPSLSTMContext* context);

/**
 * Get current LSTM hidden and cell states
 * @param context LSTM context
 * @param hidden_state Output hidden state buffer
 * @param cell_state Output cell state buffer
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_get_state(MPSLSTMContext* context,
                               float* hidden_state,
                               float* cell_state);

/**
 * Set LSTM hidden and cell states
 * @param context LSTM context
 * @param hidden_state Input hidden state
 * @param cell_state Input cell state
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_set_state(MPSLSTMContext* context,
                               const float* hidden_state,
                               const float* cell_state);

// Configuration Functions

/**
 * Create default LSTM configuration
 * @param config Pointer to store default configuration
 * @param input_size Input feature dimension
 * @param hidden_size Hidden state dimension
 * @param num_layers Number of LSTM layers
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_config_create_default(MPSLSTMConfig* config,
                                           uint32_t input_size,
                                           uint32_t hidden_size,
                                           uint32_t num_layers);

/**
 * Validate LSTM configuration
 * @param config LSTM configuration to validate
 * @return MPS_LSTM_SUCCESS if valid, error code on failure
 */
MPSLSTMError mps_lstm_config_validate(const MPSLSTMConfig* config);

/**
 * Calculate memory requirements for LSTM configuration
 * @param config LSTM configuration
 * @param memory_mb Pointer to store memory requirement in MB
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_calculate_memory_requirements(const MPSLSTMConfig* config,
                                                   uint32_t* memory_mb);

// Utility Functions

/**
 * Get error message string
 * @param error_code LSTM error code
 * @return Human-readable error message
 */
const char* mps_lstm_get_error_string(MPSLSTMError error_code);

/**
 * Check if MPS LSTM is available on current platform
 * @return true if available, false otherwise
 */
bool mps_lstm_is_available(void);

/**
 * Get Metal device information
 * @param device_name Buffer to store device name
 * @param buffer_size Size of device name buffer
 * @param compute_units Pointer to store number of compute units
 * @param max_memory_mb Pointer to store maximum memory in MB
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_get_device_info(char* device_name,
                                     size_t buffer_size,
                                     uint32_t* compute_units,
                                     uint32_t* max_memory_mb);

// Tensor Management Functions

/**
 * Create LSTM tensors structure
 * @param tensors Pointer to store created tensors
 * @param shape Tensor dimensions
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_tensors_create(MPSLSTMTensors** tensors, const MPSLSTMShape* shape);

/**
 * Destroy LSTM tensors structure
 * @param tensors Tensors to destroy
 */
void mps_lstm_tensors_destroy(MPSLSTMTensors* tensors);

/**
 * Validate LSTM tensors against configuration
 * @param tensors LSTM tensors
 * @param config LSTM configuration
 * @return MPS_LSTM_SUCCESS if valid, error code on failure
 */
MPSLSTMError mps_lstm_tensors_validate(const MPSLSTMTensors* tensors,
                                      const MPSLSTMConfig* config);

// Statistics Functions

/**
 * Get LSTM performance statistics
 * @param context LSTM context
 * @param stats Pointer to store statistics
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_get_stats(MPSLSTMContext* context, MPSLSTMStats* stats);

/**
 * Reset performance statistics
 * @param context LSTM context
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_reset_stats(MPSLSTMContext* context);

/**
 * Get current memory usage
 * @param context LSTM context
 * @param memory_usage_mb Pointer to store memory usage in MB
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_get_memory_usage(MPSLSTMContext* context, uint32_t* memory_usage_mb);

// Weights Management Functions

/**
 * Create LSTM weights structure
 * @param weights Pointer to store created weights
 * @param config LSTM configuration
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_weights_create(MPSLSTMWeights** weights, const MPSLSTMConfig* config);

/**
 * Initialize LSTM weights with random values
 * @param weights LSTM weights structure
 * @param config LSTM configuration
 * @param seed Random seed
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_weights_init_random(MPSLSTMWeights* weights,
                                         const MPSLSTMConfig* config,
                                         uint32_t seed);

/**
 * Initialize LSTM weights with Xavier/Glorot initialization
 * @param weights LSTM weights structure
 * @param config LSTM configuration
 * @param seed Random seed
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_weights_init_xavier(MPSLSTMWeights* weights,
                                         const MPSLSTMConfig* config,
                                         uint32_t seed);

/**
 * Destroy LSTM weights structure
 * @param weights LSTM weights to destroy
 */
void mps_lstm_weights_destroy(MPSLSTMWeights* weights);

// Buffer Management Functions

/**
 * Free all Metal buffers
 * @param context LSTM context
 * @return MPS_LSTM_SUCCESS on success, error code on failure
 */
MPSLSTMError mps_lstm_free_buffers(MPSLSTMContext* context);

/**
 * Destroy LSTM context
 * @param context LSTM context to destroy
 */
void mps_lstm_destroy(MPSLSTMContext* context);

#ifdef __cplusplus
}
#endif

#endif // MPS_LSTM_H