#include "nncp_lstm_state_manager.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#import <Metal/Metal.h>
#define NNCP_METAL_AVAILABLE 1
#else
#define NNCP_METAL_AVAILABLE 0
#endif

// Internal helper functions
static NNCPLSTMStateError allocate_layer_state_buffers(NNCPLSTMLayerState* layer_state,
                                                      const NNCPLSTMStateConfig* config,
                                                      uint32_t layer_index,
                                                      void* device);
static NNCPLSTMStateError sync_metal_buffers(NNCPLSTMLayerState* layer_state, int direction);
static uint64_t get_timestamp_ns(void);
static void validate_state_values(const float* data, size_t size, const char* state_name);

// Error messages
static const char* state_error_messages[] = {
    "Success",
    "Invalid parameter",
    "Memory allocation failed",
    "Buffer access failed",
    "Dimension mismatch",
    "Layer index out of bounds",
    "State not initialized",
    "Metal operation failed"
};

// Core API Implementation

NNCPLSTMStateError nncp_lstm_state_manager_create(NNCPLSTMStateManager** manager,
                                                  const NNCPLSTMStateConfig* config,
                                                  void* device,
                                                  void* command_queue) {
    if (!manager || !config) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (config->n_layers == 0 || config->n_cells == 0 || config->n_streams == 0) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    *manager = (NNCPLSTMStateManager*)calloc(1, sizeof(NNCPLSTMStateManager));
    if (!*manager) {
        return NNCP_STATE_ERROR_MEMORY_ALLOCATION;
    }
    
    NNCPLSTMStateManager* mgr = *manager;
    
    // Copy configuration
    mgr->n_layers = config->n_layers;
    mgr->seg_len = config->seg_len;
    mgr->n_streams = config->n_streams;
    mgr->segment_boundaries_enabled = config->preserve_states_across_segments;
    mgr->auto_reset_enabled = config->auto_reset_on_sequence_start;
    mgr->verbose_logging = config->verbose_logging;
    
    // Initialize sequence processing state
    mgr->current_segment = 0;
    mgr->total_segments = 0;
    mgr->current_timestep = 0;
    
    // Initialize performance counters
    mgr->total_state_updates = 0;
    mgr->total_state_resets = 0;
    mgr->total_state_preservations = 0;
    
    // Store Metal context
    mgr->device = device;
    mgr->command_queue = command_queue;
    
    // Allocate layer states
    mgr->layer_states = (NNCPLSTMLayerState*)calloc(config->n_layers, sizeof(NNCPLSTMLayerState));
    if (!mgr->layer_states) {
        free(mgr);
        *manager = NULL;
        return NNCP_STATE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize each layer state
    for (uint32_t layer = 0; layer < config->n_layers; layer++) {
        NNCPLSTMStateError error = allocate_layer_state_buffers(&mgr->layer_states[layer],
                                                               config, layer, device);
        if (error != NNCP_STATE_SUCCESS) {
            nncp_lstm_state_manager_destroy(mgr);
            *manager = NULL;
            return error;
        }
    }
    
    if (config->verbose_logging) {
        printf("NNCP LSTM State Manager created:\\n");
        printf("  Layers: %u\\n", config->n_layers);
        printf("  Hidden cells: %u, Cell state: %u\\n", config->n_cells, config->n_cells2);
        printf("  Streams: %u, Segment length: %u\\n", config->n_streams, config->seg_len);
        printf("  Preserve across segments: %s\\n", 
               config->preserve_states_across_segments ? "enabled" : "disabled");
        printf("  Metal acceleration: %s\\n", device ? "enabled" : "disabled");
    }
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_initialize(NNCPLSTMStateManager* manager) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    // Initialize all layer states to zero (matching CUDA behavior)
    for (uint32_t layer = 0; layer < manager->n_layers; layer++) {
        NNCPLSTMLayerState* state = &manager->layer_states[layer];
        
        // Clear current states
        memset(state->h_current, 0, state->n_cells * state->n_streams * sizeof(float));
        memset(state->c_current, 0, state->n_cells2 * state->n_streams * sizeof(float));
        
        // Clear initial states
        memset(state->h0_initial, 0, state->n_cells * state->n_streams * sizeof(float));
        memset(state->c0_initial, 0, state->n_cells2 * state->n_streams * sizeof(float));
        
        // Sync to Metal buffers if available
        if (state->h_buffer) {
            sync_metal_buffers(state, 0); // CPU to GPU
        }
        
        state->states_initialized = true;
        state->has_initial_states = false;
        state->last_update_time = get_timestamp_ns();
        state->sequence_position = 0;
        
        if (manager->verbose_logging) {
            printf("Initialized layer %u states\\n", layer);
        }
    }
    
    // Reset global state
    manager->current_segment = 0;
    manager->current_timestep = 0;
    manager->total_state_resets++;
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_reset_all(NNCPLSTMStateManager* manager) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    for (uint32_t layer = 0; layer < manager->n_layers; layer++) {
        NNCPLSTMStateError error = nncp_lstm_state_manager_reset_layer(manager, layer);
        if (error != NNCP_STATE_SUCCESS) {
            return error;
        }
    }
    
    if (manager->verbose_logging) {
        printf("Reset all LSTM states (total resets: %llu)\\n", 
               (unsigned long long)manager->total_state_resets);
    }
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_reset_layer(NNCPLSTMStateManager* manager,
                                                       uint32_t layer) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    NNCPLSTMLayerState* state = &manager->layer_states[layer];
    
    // Clear current states (CUDA lstm_reset behavior)
    memset(state->h_current, 0, state->n_cells * state->n_streams * sizeof(float));
    memset(state->c_current, 0, state->n_cells2 * state->n_streams * sizeof(float));
    
    // Clear initial states as well (complete reset)
    memset(state->h0_initial, 0, state->n_cells * state->n_streams * sizeof(float));
    memset(state->c0_initial, 0, state->n_cells2 * state->n_streams * sizeof(float));
    
    // Sync to Metal buffers
    if (state->h_buffer) {
        sync_metal_buffers(state, 0); // CPU to GPU
    }
    
    state->has_initial_states = false;
    state->last_update_time = get_timestamp_ns();
    state->sequence_position = 0;
    
    manager->total_state_resets++;
    
    if (manager->verbose_logging) {
        printf("Reset layer %u states\\n", layer);
    }
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_preserve_states(NNCPLSTMStateManager* manager,
                                                          uint32_t layer) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    NNCPLSTMLayerState* state = &manager->layer_states[layer];
    
    if (!state->states_initialized) {
        return NNCP_STATE_ERROR_STATE_NOT_INITIALIZED;
    }
    
    // CUDA behavior: h0 = h, c0 = c (preserve current states as initial)
    memcpy(state->h0_initial, state->h_current, 
           state->n_cells * state->n_streams * sizeof(float));
    memcpy(state->c0_initial, state->c_current, 
           state->n_cells2 * state->n_streams * sizeof(float));
    
    state->has_initial_states = true;
    state->last_update_time = get_timestamp_ns();
    
    manager->total_state_preservations++;
    
    if (manager->verbose_logging) {
        printf("Preserved states for layer %u (h0=h, c0=c)\\n", layer);
    }
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_restore_states(NNCPLSTMStateManager* manager,
                                                         uint32_t layer) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    NNCPLSTMLayerState* state = &manager->layer_states[layer];
    
    if (!state->has_initial_states) {
        return NNCP_STATE_ERROR_STATE_NOT_INITIALIZED;
    }
    
    // CUDA behavior: h = h0, c = c0 (restore from initial states)
    memcpy(state->h_current, state->h0_initial,
           state->n_cells * state->n_streams * sizeof(float));
    memcpy(state->c_current, state->c0_initial,
           state->n_cells2 * state->n_streams * sizeof(float));
    
    // Sync to Metal buffers
    if (state->h_buffer) {
        sync_metal_buffers(state, 0); // CPU to GPU
    }
    
    state->last_update_time = get_timestamp_ns();
    
    if (manager->verbose_logging) {
        printf("Restored states for layer %u (h=h0, c=c0)\\n", layer);
    }
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_get_hidden_state(NNCPLSTMStateManager* manager,
                                                           uint32_t layer,
                                                           float* hidden_state) {
    if (!manager || !hidden_state) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    NNCPLSTMLayerState* state = &manager->layer_states[layer];
    
    if (!state->states_initialized) {
        return NNCP_STATE_ERROR_STATE_NOT_INITIALIZED;
    }
    
    // Sync from Metal buffer if needed
    if (state->h_buffer) {
        NNCPLSTMStateError error = sync_metal_buffers(state, 1); // GPU to CPU
        if (error != NNCP_STATE_SUCCESS) {
            return error;
        }
    }
    
    memcpy(hidden_state, state->h_current,
           state->n_cells * state->n_streams * sizeof(float));
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_get_cell_state(NNCPLSTMStateManager* manager,
                                                         uint32_t layer,
                                                         float* cell_state) {
    if (!manager || !cell_state) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    NNCPLSTMLayerState* state = &manager->layer_states[layer];
    
    if (!state->states_initialized) {
        return NNCP_STATE_ERROR_STATE_NOT_INITIALIZED;
    }
    
    // Sync from Metal buffer if needed
    if (state->c_buffer) {
        NNCPLSTMStateError error = sync_metal_buffers(state, 1); // GPU to CPU
        if (error != NNCP_STATE_SUCCESS) {
            return error;
        }
    }
    
    memcpy(cell_state, state->c_current,
           state->n_cells2 * state->n_streams * sizeof(float));
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_set_hidden_state(NNCPLSTMStateManager* manager,
                                                           uint32_t layer,
                                                           const float* hidden_state) {
    if (!manager || !hidden_state) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    NNCPLSTMLayerState* state = &manager->layer_states[layer];
    
    memcpy(state->h_current, hidden_state,
           state->n_cells * state->n_streams * sizeof(float));
    
    // Validate state values
    if (manager->verbose_logging) {
        validate_state_values(state->h_current, 
                             state->n_cells * state->n_streams,
                             "hidden_state");
    }
    
    // Sync to Metal buffer
    if (state->h_buffer) {
        sync_metal_buffers(state, 0); // CPU to GPU
    }
    
    state->states_initialized = true;
    state->last_update_time = get_timestamp_ns();
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_set_cell_state(NNCPLSTMStateManager* manager,
                                                         uint32_t layer,
                                                         const float* cell_state) {
    if (!manager || !cell_state) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    NNCPLSTMLayerState* state = &manager->layer_states[layer];
    
    memcpy(state->c_current, cell_state,
           state->n_cells2 * state->n_streams * sizeof(float));
    
    // Validate state values
    if (manager->verbose_logging) {
        validate_state_values(state->c_current,
                             state->n_cells2 * state->n_streams,
                             "cell_state");
    }
    
    // Sync to Metal buffer
    if (state->c_buffer) {
        sync_metal_buffers(state, 0); // CPU to GPU
    }
    
    state->states_initialized = true;
    state->last_update_time = get_timestamp_ns();
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_update_states(NNCPLSTMStateManager* manager,
                                                        uint32_t layer,
                                                        const float* new_hidden_state,
                                                        const float* new_cell_state) {
    if (!manager || !new_hidden_state || !new_cell_state) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    NNCPLSTMLayerState* state = &manager->layer_states[layer];
    
    // Update current states (CUDA behavior: h = h_new, c = c_new)
    memcpy(state->h_current, new_hidden_state,
           state->n_cells * state->n_streams * sizeof(float));
    memcpy(state->c_current, new_cell_state,
           state->n_cells2 * state->n_streams * sizeof(float));
    
    // Validate updated states
    if (manager->verbose_logging) {
        validate_state_values(state->h_current,
                             state->n_cells * state->n_streams,
                             "updated_hidden");
        validate_state_values(state->c_current,
                             state->n_cells2 * state->n_streams,
                             "updated_cell");
    }
    
    // Sync to Metal buffers
    if (state->h_buffer) {
        sync_metal_buffers(state, 0); // CPU to GPU
    }
    
    state->states_initialized = true;
    state->last_update_time = get_timestamp_ns();
    state->sequence_position++;
    
    manager->total_state_updates++;
    
    return NNCP_STATE_SUCCESS;
}

// Sequence Processing State Management

NNCPLSTMStateError nncp_lstm_state_manager_begin_segment(NNCPLSTMStateManager* manager,
                                                        uint32_t segment_index,
                                                        bool preserve_from_previous) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    manager->current_segment = segment_index;
    manager->current_timestep = 0;
    
    if (preserve_from_previous && manager->segment_boundaries_enabled) {
        // Restore states from preserved h0/c0 (CUDA behavior)
        for (uint32_t layer = 0; layer < manager->n_layers; layer++) {
            if (manager->layer_states[layer].has_initial_states) {
                NNCPLSTMStateError error = nncp_lstm_state_manager_restore_states(manager, layer);
                if (error != NNCP_STATE_SUCCESS) {
                    return error;
                }
            }
        }
        
        if (manager->verbose_logging) {
            printf("Began segment %u with preserved states\\n", segment_index);
        }
    } else {
        // Reset all states for new segment
        NNCPLSTMStateError error = nncp_lstm_state_manager_reset_all(manager);
        if (error != NNCP_STATE_SUCCESS) {
            return error;
        }
        
        if (manager->verbose_logging) {
            printf("Began segment %u with reset states\\n", segment_index);
        }
    }
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_end_segment(NNCPLSTMStateManager* manager,
                                                      bool preserve_for_next) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (preserve_for_next && manager->segment_boundaries_enabled) {
        // Preserve current states as initial states for next segment
        for (uint32_t layer = 0; layer < manager->n_layers; layer++) {
            NNCPLSTMStateError error = nncp_lstm_state_manager_preserve_states(manager, layer);
            if (error != NNCP_STATE_SUCCESS) {
                return error;
            }
        }
        
        if (manager->verbose_logging) {
            printf("Ended segment %u with state preservation\\n", manager->current_segment);
        }
    } else {
        if (manager->verbose_logging) {
            printf("Ended segment %u without state preservation\\n", manager->current_segment);
        }
    }
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_begin_timestep(NNCPLSTMStateManager* manager,
                                                         uint32_t timestep) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (timestep >= manager->seg_len) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    manager->current_timestep = timestep;
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_end_timestep(NNCPLSTMStateManager* manager) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    manager->current_timestep++;
    
    return NNCP_STATE_SUCCESS;
}

// Metal GPU Integration

NNCPLSTMStateError nncp_lstm_state_manager_get_hidden_buffer(NNCPLSTMStateManager* manager,
                                                           uint32_t layer,
                                                           void** buffer_ptr) {
    if (!manager || !buffer_ptr) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    *buffer_ptr = manager->layer_states[layer].h_buffer;
    
    if (!*buffer_ptr) {
        return NNCP_STATE_ERROR_BUFFER_ACCESS;
    }
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_get_cell_buffer(NNCPLSTMStateManager* manager,
                                                         uint32_t layer,
                                                         void** buffer_ptr) {
    if (!manager || !buffer_ptr) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (layer >= manager->n_layers) {
        return NNCP_STATE_ERROR_LAYER_INDEX_OUT_OF_BOUNDS;
    }
    
    *buffer_ptr = manager->layer_states[layer].c_buffer;
    
    if (!*buffer_ptr) {
        return NNCP_STATE_ERROR_BUFFER_ACCESS;
    }
    
    return NNCP_STATE_SUCCESS;
}

// Utility Functions

const char* nncp_lstm_state_get_error_string(NNCPLSTMStateError error_code) {
    if (error_code >= sizeof(state_error_messages) / sizeof(state_error_messages[0])) {
        return "Unknown error";
    }
    return state_error_messages[error_code];
}

NNCPLSTMStateError nncp_lstm_state_calculate_memory_requirements(const NNCPLSTMStateConfig* config,
                                                               uint32_t* memory_mb) {
    if (!config || !memory_mb) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    // Calculate memory for each layer
    uint64_t memory_per_layer = 0;
    
    // Hidden states (current + initial)
    memory_per_layer += 2 * config->n_cells * config->n_streams * sizeof(float);
    
    // Cell states (current + initial)
    memory_per_layer += 2 * config->n_cells2 * config->n_streams * sizeof(float);
    
    // Total memory for all layers
    uint64_t total_memory = memory_per_layer * config->n_layers;
    
    // Add overhead for management structures
    total_memory += sizeof(NNCPLSTMStateManager) + 
                   config->n_layers * sizeof(NNCPLSTMLayerState);
    
    *memory_mb = (uint32_t)((total_memory + 1024 * 1024 - 1) / (1024 * 1024));
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_config_create_default(NNCPLSTMStateConfig* config,
                                                        uint32_t n_layers,
                                                        uint32_t n_cells,
                                                        uint32_t n_streams,
                                                        uint32_t seg_len) {
    if (!config) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(NNCPLSTMStateConfig));
    
    config->n_layers = n_layers;
    config->n_cells = n_cells;
    config->n_cells2 = n_cells; // Usually same as n_cells
    config->n_streams = n_streams;
    config->seg_len = seg_len;
    
    // Default state management options (matching CUDA behavior)
    config->preserve_states_across_segments = true;
    config->auto_reset_on_sequence_start = false;
    config->enable_state_checkpointing = false;
    config->use_metal_buffers = true;
    
    // Memory management
    config->max_memory_mb = 256;
    config->optimize_for_memory = false;
    
    // Debugging
    config->verbose_logging = false;
    config->validate_state_integrity = false;
    config->track_performance_stats = true;
    
    return NNCP_STATE_SUCCESS;
}

void nncp_lstm_state_manager_destroy(NNCPLSTMStateManager* manager) {
    if (!manager) {
        return;
    }
    
    if (manager->layer_states) {
        for (uint32_t layer = 0; layer < manager->n_layers; layer++) {
            NNCPLSTMLayerState* state = &manager->layer_states[layer];
            
            // Free CPU buffers
            free(state->h_current);
            free(state->c_current);
            free(state->h0_initial);
            free(state->c0_initial);
            
            // Metal buffers are automatically released
            state->h_buffer = NULL;
            state->c_buffer = NULL;
            state->h0_buffer = NULL;
            state->c0_buffer = NULL;
        }
        
        free(manager->layer_states);
    }
    
    if (manager->verbose_logging) {
        printf("NNCP LSTM State Manager destroyed\\n");
        printf("  Total state updates: %llu\\n", (unsigned long long)manager->total_state_updates);
        printf("  Total state resets: %llu\\n", (unsigned long long)manager->total_state_resets);
        printf("  Total state preservations: %llu\\n", (unsigned long long)manager->total_state_preservations);
    }
    
    free(manager);
}

// Internal helper function implementations

static NNCPLSTMStateError allocate_layer_state_buffers(NNCPLSTMLayerState* layer_state,
                                                      const NNCPLSTMStateConfig* config,
                                                      uint32_t layer_index,
                                                      void* device) {
    if (!layer_state || !config) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    // Set dimensions
    layer_state->n_cells = config->n_cells;
    layer_state->n_cells2 = config->n_cells2;
    layer_state->n_streams = config->n_streams;
    
    // Calculate buffer sizes
    size_t hidden_size = config->n_cells * config->n_streams * sizeof(float);
    size_t cell_size = config->n_cells2 * config->n_streams * sizeof(float);
    
    // Allocate CPU buffers
    layer_state->h_current = (float*)malloc(hidden_size);
    layer_state->c_current = (float*)malloc(cell_size);
    layer_state->h0_initial = (float*)malloc(hidden_size);
    layer_state->c0_initial = (float*)malloc(cell_size);
    
    if (!layer_state->h_current || !layer_state->c_current ||
        !layer_state->h0_initial || !layer_state->c0_initial) {
        return NNCP_STATE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize to zero
    memset(layer_state->h_current, 0, hidden_size);
    memset(layer_state->c_current, 0, cell_size);
    memset(layer_state->h0_initial, 0, hidden_size);
    memset(layer_state->c0_initial, 0, cell_size);
    
    // Allocate Metal buffers if device is available
    if (device && config->use_metal_buffers) {
#if NNCP_METAL_AVAILABLE
        @autoreleasepool {
            id<MTLDevice> metalDevice = (__bridge id<MTLDevice>)device;
            
            id<MTLBuffer> hBuffer = [metalDevice newBufferWithLength:hidden_size 
                                                             options:MTLResourceStorageModeShared];
            id<MTLBuffer> cBuffer = [metalDevice newBufferWithLength:cell_size 
                                                             options:MTLResourceStorageModeShared];
            id<MTLBuffer> h0Buffer = [metalDevice newBufferWithLength:hidden_size 
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> c0Buffer = [metalDevice newBufferWithLength:cell_size 
                                                              options:MTLResourceStorageModeShared];
            
            if (!hBuffer || !cBuffer || !h0Buffer || !c0Buffer) {
                return NNCP_STATE_ERROR_BUFFER_ACCESS;
            }
            
            layer_state->h_buffer = (__bridge void*)hBuffer;
            layer_state->c_buffer = (__bridge void*)cBuffer;
            layer_state->h0_buffer = (__bridge void*)h0Buffer;
            layer_state->c0_buffer = (__bridge void*)c0Buffer;
            
            // Initialize Metal buffers to zero
            memset([hBuffer contents], 0, hidden_size);
            memset([cBuffer contents], 0, cell_size);
            memset([h0Buffer contents], 0, hidden_size);
            memset([c0Buffer contents], 0, cell_size);
        }
#endif
    }
    
    // Initialize metadata
    layer_state->states_initialized = false;
    layer_state->has_initial_states = false;
    layer_state->last_update_time = 0;
    layer_state->sequence_position = 0;
    
    return NNCP_STATE_SUCCESS;
}

static NNCPLSTMStateError sync_metal_buffers(NNCPLSTMLayerState* layer_state, int direction) {
    if (!layer_state) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
#if NNCP_METAL_AVAILABLE
    if (layer_state->h_buffer && layer_state->c_buffer) {
        @autoreleasepool {
            id<MTLBuffer> hBuffer = (__bridge id<MTLBuffer>)layer_state->h_buffer;
            id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)layer_state->c_buffer;
            
            size_t hidden_size = layer_state->n_cells * layer_state->n_streams * sizeof(float);
            size_t cell_size = layer_state->n_cells2 * layer_state->n_streams * sizeof(float);
            
            if (direction == 0) {
                // CPU to GPU
                memcpy([hBuffer contents], layer_state->h_current, hidden_size);
                memcpy([cBuffer contents], layer_state->c_current, cell_size);
            } else {
                // GPU to CPU
                memcpy(layer_state->h_current, [hBuffer contents], hidden_size);
                memcpy(layer_state->c_current, [cBuffer contents], cell_size);
            }
        }
    }
#endif
    
    return NNCP_STATE_SUCCESS;
}

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

static void validate_state_values(const float* data, size_t size, const char* state_name) {
    if (!data || !state_name) {
        return;
    }
    
    bool has_nan = false;
    bool has_inf = false;
    float min_val = data[0];
    float max_val = data[0];
    
    for (size_t i = 0; i < size; i++) {
        float val = data[i];
        
        if (isnan(val)) {
            has_nan = true;
        }
        if (isinf(val)) {
            has_inf = true;
        }
        
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    
    if (has_nan || has_inf) {
        printf("WARNING: %s contains invalid values (NaN: %s, Inf: %s)\\n",
               state_name, has_nan ? "yes" : "no", has_inf ? "yes" : "no");
    }
    
    printf("State %s: min=%.6f, max=%.6f\\n", state_name, min_val, max_val);
}

// Batch Processing API - CUDA-compatible sequence processing

NNCPLSTMStateError nncp_lstm_state_manager_process_batch(NNCPLSTMStateManager* manager,
                                                        uint32_t batch_size,
                                                        uint32_t sequence_length,
                                                        bool preserve_states_across_batches) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (batch_size == 0 || sequence_length == 0) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    // Ensure batch size doesn't exceed configured n_streams
    if (batch_size > manager->n_streams) {
        if (manager->verbose_logging) {
            printf("WARNING: Batch size %u exceeds n_streams %u, clamping to n_streams\\n", 
                   batch_size, manager->n_streams);
        }
        batch_size = manager->n_streams;
    }
    
    // Process sequence in segments (CUDA behavior: seg_len chunks)
    uint32_t total_segments = (sequence_length + manager->seg_len - 1) / manager->seg_len;
    
    for (uint32_t segment = 0; segment < total_segments; segment++) {
        uint32_t segment_start = segment * manager->seg_len;
        uint32_t segment_end = segment_start + manager->seg_len;
        if (segment_end > sequence_length) {
            segment_end = sequence_length;
        }
        uint32_t segment_length = segment_end - segment_start;
        
        // Begin segment with state preservation logic
        bool preserve_from_previous = (segment > 0) && preserve_states_across_batches;
        NNCPLSTMStateError error = nncp_lstm_state_manager_begin_segment(manager, segment, preserve_from_previous);
        if (error != NNCP_STATE_SUCCESS) {
            return error;
        }
        
        // Process each timestep in the segment
        for (uint32_t timestep = 0; timestep < segment_length; timestep++) {
            error = nncp_lstm_state_manager_begin_timestep(manager, timestep);
            if (error != NNCP_STATE_SUCCESS) {
                return error;
            }
            
            // Note: Actual LSTM computation happens externally
            // This function just manages the state transitions
            
            error = nncp_lstm_state_manager_end_timestep(manager);
            if (error != NNCP_STATE_SUCCESS) {
                return error;
            }
        }
        
        // End segment with state preservation for next segment
        bool preserve_for_next = (segment < total_segments - 1) && preserve_states_across_batches;
        error = nncp_lstm_state_manager_end_segment(manager, preserve_for_next);
        if (error != NNCP_STATE_SUCCESS) {
            return error;
        }
    }
    
    manager->total_segments = total_segments;
    
    if (manager->verbose_logging) {
        printf("Processed batch: size=%u, seq_len=%u, segments=%u\\n", 
               batch_size, sequence_length, total_segments);
    }
    
    return NNCP_STATE_SUCCESS;
}

NNCPLSTMStateError nncp_lstm_state_manager_process_sequence_parallel(NNCPLSTMStateManager* manager,
                                                                   uint32_t batch_size,
                                                                   uint32_t sequence_length,
                                                                   bool independent_sequences) {
    if (!manager) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    if (batch_size > manager->n_streams) {
        return NNCP_STATE_ERROR_INVALID_PARAM;
    }
    
    // For independent sequences, reset states for each sequence in the batch
    if (independent_sequences) {
        NNCPLSTMStateError error = nncp_lstm_state_manager_reset_all(manager);
        if (error != NNCP_STATE_SUCCESS) {
            return error;
        }
        
        if (manager->verbose_logging) {
            printf("Processing %u independent parallel sequences of length %u\\n", 
                   batch_size, sequence_length);
        }
    } else {
        // For continuation sequences, preserve states from previous processing
        if (manager->verbose_logging) {
            printf("Processing %u continuation parallel sequences of length %u\\n", 
                   batch_size, sequence_length);
        }
    }
    
    // Process sequences using the standard batch processing logic
    return nncp_lstm_state_manager_process_batch(manager, batch_size, sequence_length, !independent_sequences);
}
