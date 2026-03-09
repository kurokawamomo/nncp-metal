#ifndef NNCP_LSTM_STATE_MANAGER_H
#define NNCP_LSTM_STATE_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __APPLE__
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLBuffer;
#else
typedef struct MTLDevice_t* MTLDevice_t;
typedef struct MTLCommandQueue_t* MTLCommandQueue_t;
typedef struct MTLBuffer_t* MTLBuffer_t;
#endif
#else
typedef void MTLDevice_t;
typedef void MTLCommandQueue_t;
typedef void MTLBuffer_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration - see nncp_lstm_metal.h for complete definition

// LSTM State Management Error Codes
typedef enum {
    NNCP_STATE_SUCCESS = 0,
    NNCP_STATE_ERROR_INVALID_PARAM,
    NNCP_STATE_ERROR_MEMORY_ALLOCATION,
    NNCP_STATE_ERROR_BUFFER_ACCESS,
    NNCP_STATE_ERROR_DIMENSION_MISMATCH,
    NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS,
    NNCP_STATE_ERROR_STATE_NOT_INITIALIZED,
    NNCP_STATE_ERROR_METAL_OPERATION_FAILED
} NNCPLSTMStateError;

// LSTM Layer State Structure (matching CUDA implementation)
typedef struct {
    // Current states [n_cells/n_cells2, n_streams]
    float* h_current;               // Hidden state
    float* c_current;               // Cell state
    
    // Initial states (preserved across segments)
    float* h0_initial;              // Initial hidden state
    float* c0_initial;              // Initial cell state
    
    // State dimensions
    uint32_t n_cells;               // Hidden state dimension
    uint32_t n_cells2;              // Cell state dimension
    uint32_t n_streams;             // Number of parallel sequences
    
    // Metal buffers for GPU operations
    void* h_buffer;                 // Hidden state Metal buffer
    void* c_buffer;                 // Cell state Metal buffer
    void* h0_buffer;                // Initial hidden state Metal buffer
    void* c0_buffer;                // Initial cell state Metal buffer
    
    // State metadata
    bool states_initialized;        // Whether states have been initialized
    bool has_initial_states;        // Whether h0/c0 are set
    uint64_t last_update_time;      // Timestamp of last state update
    uint32_t sequence_position;     // Current position in sequence processing
} NNCPLSTMLayerState;

// Multi-Layer LSTM State Manager
typedef struct {
    // Layer states
    NNCPLSTMLayerState* layer_states;   // Array of layer states
    uint32_t n_layers;                  // Number of LSTM layers
    
    // Global state management
    uint32_t current_segment;           // Current segment being processed
    uint32_t total_segments;            // Total segments in sequence
    bool segment_boundaries_enabled;    // Whether to preserve states across segments
    bool auto_reset_enabled;            // Whether to auto-reset states
    
    // Sequence processing state
    uint32_t current_timestep;          // Current timestep within segment
    uint32_t seg_len;                   // Segment length
    uint32_t n_streams;                 // Number of parallel streams
    
    // Metal device context
    void* device;                       // MTLDevice
    void* command_queue;                // MTLCommandQueue
    
    // Performance and debugging
    uint64_t total_state_updates;       // Total number of state updates
    uint64_t total_state_resets;        // Total number of state resets
    uint64_t total_state_preservations; // Total state preservations
    bool verbose_logging;               // Enable detailed logging
} NNCPLSTMStateManager;

// State Management Configuration
typedef struct {
    uint32_t n_layers;                  // Number of LSTM layers
    uint32_t n_cells;                   // Hidden state dimension
    uint32_t n_cells2;                  // Cell state dimension (usually == n_cells)
    uint32_t n_streams;                 // Number of parallel sequences
    uint32_t seg_len;                   // Segment length
    
    // State management options
    bool preserve_states_across_segments;  // Preserve h0/c0 between segments
    bool auto_reset_on_sequence_start;     // Auto-reset states at sequence start
    bool enable_state_checkpointing;       // Enable state checkpointing
    bool use_metal_buffers;                // Use Metal GPU buffers
    
    // Memory management
    uint32_t max_memory_mb;             // Maximum memory for state management
    bool optimize_for_memory;           // Optimize for memory usage vs performance
    
    // Debugging and validation
    bool verbose_logging;               // Enable detailed logging
    bool validate_state_integrity;      // Validate state integrity after operations
    bool track_performance_stats;       // Track performance statistics
} NNCPLSTMStateConfig;

// Core State Management API

/**
 * Create LSTM state manager
 * @param manager Pointer to store created state manager
 * @param config State management configuration
 * @param device Metal device (NULL for CPU-only)
 * @param command_queue Metal command queue (NULL for CPU-only)
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_create(NNCPLSTMStateManager** manager,
                                                  const NNCPLSTMStateConfig* config,
                                                  void* device,
                                                  void* command_queue);

/**
 * Initialize all layer states (equivalent to CUDA reset)
 * @param manager State manager
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_initialize(NNCPLSTMStateManager* manager);

/**
 * Reset all states to zero (CUDA lstm_reset equivalent)
 * @param manager State manager
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_reset_all(NNCPLSTMStateManager* manager);

/**
 * Reset specific layer states
 * @param manager State manager
 * @param layer Layer index
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_reset_layer(NNCPLSTMStateManager* manager,
                                                       uint32_t layer);

/**
 * Preserve current states as initial states (h0 = h, c0 = c)
 * @param manager State manager
 * @param layer Layer index
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_preserve_states(NNCPLSTMStateManager* manager,
                                                          uint32_t layer);

/**
 * Restore states from initial states (h = h0, c = c0)
 * @param manager State manager
 * @param layer Layer index
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_restore_states(NNCPLSTMStateManager* manager,
                                                         uint32_t layer);

/**
 * Get current hidden state for a layer
 * @param manager State manager
 * @param layer Layer index
 * @param hidden_state Output buffer [n_cells * n_streams]
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_get_hidden_state(NNCPLSTMStateManager* manager,
                                                           uint32_t layer,
                                                           float* hidden_state);

/**
 * Get current cell state for a layer
 * @param manager State manager
 * @param layer Layer index
 * @param cell_state Output buffer [n_cells2 * n_streams]
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_get_cell_state(NNCPLSTMStateManager* manager,
                                                         uint32_t layer,
                                                         float* cell_state);

/**
 * Set hidden state for a layer
 * @param manager State manager
 * @param layer Layer index
 * @param hidden_state Input buffer [n_cells * n_streams]
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_set_hidden_state(NNCPLSTMStateManager* manager,
                                                           uint32_t layer,
                                                           const float* hidden_state);

/**
 * Set cell state for a layer
 * @param manager State manager
 * @param layer Layer index
 * @param cell_state Input buffer [n_cells2 * n_streams]
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_set_cell_state(NNCPLSTMStateManager* manager,
                                                         uint32_t layer,
                                                         const float* cell_state);

/**
 * Update states after LSTM computation (h_new, c_new -> h, c)
 * @param manager State manager
 * @param layer Layer index
 * @param new_hidden_state New hidden state [n_cells * n_streams]
 * @param new_cell_state New cell state [n_cells2 * n_streams]
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_update_states(NNCPLSTMStateManager* manager,
                                                        uint32_t layer,
                                                        const float* new_hidden_state,
                                                        const float* new_cell_state);

// Sequence Processing State Management

/**
 * Begin new sequence segment (CUDA segment processing)
 * @param manager State manager
 * @param segment_index Segment index
 * @param preserve_from_previous Whether to preserve states from previous segment
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_begin_segment(NNCPLSTMStateManager* manager,
                                                        uint32_t segment_index,
                                                        bool preserve_from_previous);

/**
 * End current sequence segment
 * @param manager State manager
 * @param preserve_for_next Whether to preserve states for next segment
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_end_segment(NNCPLSTMStateManager* manager,
                                                      bool preserve_for_next);

/**
 * Begin new timestep within segment
 * @param manager State manager
 * @param timestep Timestep index within segment
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_begin_timestep(NNCPLSTMStateManager* manager,
                                                         uint32_t timestep);

/**
 * End current timestep
 * @param manager State manager
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_end_timestep(NNCPLSTMStateManager* manager);

// Metal GPU Integration

/**
 * Get Metal buffer for hidden state
 * @param manager State manager
 * @param layer Layer index
 * @param buffer_ptr Pointer to store Metal buffer reference
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
// Batch Processing API

/**
 * Process batch sequences with CUDA-compatible state management
 * @param manager State manager
 * @param batch_size Number of parallel sequences
 * @param sequence_length Length of each sequence
 * @param preserve_states_across_batches Whether to preserve states between batches
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_process_batch(NNCPLSTMStateManager* manager,
                                                        uint32_t batch_size,
                                                        uint32_t sequence_length,
                                                        bool preserve_states_across_batches);

/**
 * Process parallel sequences with independent or continuation handling
 * @param manager State manager
 * @param batch_size Number of parallel sequences
 * @param sequence_length Length of each sequence
 * @param independent_sequences Whether sequences are independent (reset states)
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_process_sequence_parallel(NNCPLSTMStateManager* manager,
                                                                   uint32_t batch_size,
                                                                   uint32_t sequence_length,
                                                                   bool independent_sequences);

NNCPLSTMStateError nncp_lstm_state_manager_get_hidden_buffer(NNCPLSTMStateManager* manager,
                                                           uint32_t layer,
                                                           void** buffer_ptr);

/**
 * Get Metal buffer for cell state
 * @param manager State manager
 * @param layer Layer index
 * @param buffer_ptr Pointer to store Metal buffer reference
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_get_cell_buffer(NNCPLSTMStateManager* manager,
                                                         uint32_t layer,
                                                         void** buffer_ptr);

/**
 * Synchronize CPU and GPU states
 * @param manager State manager
 * @param layer Layer index
 * @param direction 0 = CPU to GPU, 1 = GPU to CPU
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_sync_states(NNCPLSTMStateManager* manager,
                                                      uint32_t layer,
                                                      int direction);

// State Validation and Debugging

/**
 * Validate state integrity
 * @param manager State manager
 * @param layer Layer index (-1 for all layers)
 * @param error_flags Output error flags
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_validate_states(NNCPLSTMStateManager* manager,
                                                          int layer,
                                                          uint32_t* error_flags);

/**
 * Get state statistics
 * @param manager State manager
 * @param total_updates Pointer to store total updates
 * @param total_resets Pointer to store total resets
 * @param total_preservations Pointer to store total preservations
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_get_stats(NNCPLSTMStateManager* manager,
                                                    uint64_t* total_updates,
                                                    uint64_t* total_resets,
                                                    uint64_t* total_preservations);

/**
 * Print state manager debug information
 * @param manager State manager
 * @param layer Layer index (-1 for all layers)
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_manager_print_debug_info(NNCPLSTMStateManager* manager,
                                                           int layer);

// Utility Functions

/**
 * Get error message string
 * @param error_code State management error code
 * @return Human-readable error message
 */
const char* nncp_lstm_state_get_error_string(NNCPLSTMStateError error_code);

/**
 * Calculate memory requirements for state management
 * @param config State management configuration
 * @param memory_mb Pointer to store memory requirement in MB
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_calculate_memory_requirements(const NNCPLSTMStateConfig* config,
                                                               uint32_t* memory_mb);

/**
 * Create default state management configuration
 * @param config Pointer to store default configuration
 * @param n_layers Number of LSTM layers
 * @param n_cells Hidden state dimension
 * @param n_streams Number of parallel sequences
 * @param seg_len Segment length
 * @return NNCP_STATE_SUCCESS on success, error code on failure
 */
NNCPLSTMStateError nncp_lstm_state_config_create_default(NNCPLSTMStateConfig* config,
                                                        uint32_t n_layers,
                                                        uint32_t n_cells,
                                                        uint32_t n_streams,
                                                        uint32_t seg_len);

/**
 * Destroy state manager and free resources
 * @param manager State manager to destroy
 */
void nncp_lstm_state_manager_destroy(NNCPLSTMStateManager* manager);

#ifdef __cplusplus
}
#endif

#endif // NNCP_LSTM_STATE_MANAGER_H
