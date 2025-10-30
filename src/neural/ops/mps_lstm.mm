#include "mps_lstm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#define MPS_AVAILABLE 1
#else
#define MPS_AVAILABLE 0
#endif

// Internal helper functions
static MPSLSTMError create_metal_device(MPSLSTMContext* context);
static MPSLSTMError build_lstm_graph(MPSLSTMContext* context);
static MPSLSTMError allocate_metal_buffers(MPSLSTMContext* context);
static MPSLSTMError execute_lstm_graph(MPSLSTMContext* context, MPSLSTMTensors* tensors);
static uint64_t get_timestamp_ns(void);
static size_t calculate_tensor_size(const MPSLSTMShape* shape, bool include_directions);
static size_t calculate_weights_size(const MPSLSTMConfig* config);
static size_t calculate_state_size(const MPSLSTMConfig* config);

// Error messages
static const char* error_messages[] = {
    "Success",
    "Invalid parameter",
    "Memory allocation failed",
    "Metal device not found",
    "Compute operation failed",
    "Invalid tensor dimensions",
    "Buffer allocation failed",
    "Graph compilation failed",
    "Execution failed",
    "Unsupported operation"
};

// Core API Implementation

MPSLSTMError mps_lstm_create(MPSLSTMContext** context, const MPSLSTMConfig* config) {
    if (!context || !config) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    // Validate configuration
    MPSLSTMError error = mps_lstm_config_validate(config);
    if (error != MPS_LSTM_SUCCESS) {
        return error;
    }
    
    *context = (MPSLSTMContext*)calloc(1, sizeof(MPSLSTMContext));
    if (!*context) {
        return MPS_LSTM_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy configuration
    (*context)->config = *config;
    
    // Initialize statistics
    memset(&(*context)->stats, 0, sizeof(MPSLSTMStats));
    
    // Create Metal device
    error = create_metal_device(*context);
    if (error != MPS_LSTM_SUCCESS) {
        free(*context);
        *context = NULL;
        return error;
    }
    
    // Build computation graph
    error = build_lstm_graph(*context);
    if (error != MPS_LSTM_SUCCESS) {
        mps_lstm_destroy(*context);
        *context = NULL;
        return error;
    }
    
    // Allocate buffers
    error = allocate_metal_buffers(*context);
    if (error != MPS_LSTM_SUCCESS) {
        mps_lstm_destroy(*context);
        *context = NULL;
        return error;
    }
    
    (*context)->is_initialized = true;
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_load_weights(MPSLSTMContext* context, const MPSLSTMWeights* weights) {
    if (!context || !weights) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    // Copy weights structure
    context->weights = *weights;
    context->has_weights = true;
    
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        
        // Create weights buffer if not already created
        if (!context->buffers.weights_buffer) {
            size_t total_weights_size = calculate_weights_size(&context->config);
            id<MTLBuffer> weightsBuffer = [device newBufferWithLength:total_weights_size
                                                              options:MTLResourceStorageModeShared];
            if (!weightsBuffer) {
                return MPS_LSTM_ERROR_BUFFER_ALLOCATION;
            }
            context->buffers.weights_buffer = (__bridge void*)weightsBuffer;
        }
        
        // Copy weights data to Metal buffer
        id<MTLBuffer> weightsBuffer = (__bridge id<MTLBuffer>)context->buffers.weights_buffer;
        uint8_t* buffer_ptr = (uint8_t*)[weightsBuffer contents];
        size_t offset = 0;
        
        // Copy input-to-hidden weights
        if (weights->input_weights) {
            memcpy(buffer_ptr + offset, weights->input_weights, weights->input_weights_size);
            offset += weights->input_weights_size;
        }
        
        // Copy hidden-to-hidden weights
        if (weights->hidden_weights) {
            memcpy(buffer_ptr + offset, weights->hidden_weights, weights->hidden_weights_size);
            offset += weights->hidden_weights_size;
        }
        
        // Copy biases if enabled
        if (context->config.use_bias) {
            if (weights->input_bias) {
                memcpy(buffer_ptr + offset, weights->input_bias, weights->bias_size);
                offset += weights->bias_size;
            }
            if (weights->hidden_bias) {
                memcpy(buffer_ptr + offset, weights->hidden_bias, weights->bias_size);
                offset += weights->bias_size;
            }
        }
        
        // Copy peephole weights if enabled
        if (context->config.use_peepholes && weights->input_peephole) {
            memcpy(buffer_ptr + offset, weights->input_peephole, weights->peephole_size);
            offset += weights->peephole_size;
            memcpy(buffer_ptr + offset, weights->forget_peephole, weights->peephole_size);
            offset += weights->peephole_size;
            memcpy(buffer_ptr + offset, weights->output_peephole, weights->peephole_size);
            offset += weights->peephole_size;
        }
        
        // Copy layer norm parameters if enabled
        if (context->config.use_layer_norm && weights->layer_norm_weights) {
            memcpy(buffer_ptr + offset, weights->layer_norm_weights, weights->layer_norm_size);
            offset += weights->layer_norm_size;
            memcpy(buffer_ptr + offset, weights->layer_norm_bias, weights->layer_norm_size);
        }
    }
#endif
    
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_forward(MPSLSTMContext* context, MPSLSTMTensors* tensors) {
    if (!context || !tensors) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized || !context->has_weights) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    // Validate tensor dimensions
    MPSLSTMError error = mps_lstm_tensors_validate(tensors, &context->config);
    if (error != MPS_LSTM_SUCCESS) {
        return error;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    // Execute LSTM computation
    error = execute_lstm_graph(context, tensors);
    if (error != MPS_LSTM_SUCCESS) {
        return error;
    }
    
    // Update statistics
    uint64_t compute_time = get_timestamp_ns() - start_time;
    context->stats.total_operations++;
    context->stats.total_compute_time_ns += compute_time;
    context->stats.batch_size_sum += tensors->shape.batch_size;
    context->stats.sequence_length_sum += tensors->shape.sequence_length;
    
    // Calculate derived statistics
    context->stats.average_compute_time_ms = 
        (float)context->stats.total_compute_time_ns / (1000000.0f * context->stats.total_operations);
    
    // Estimate GFLOPS (approximate calculation for LSTM)
    uint64_t ops_per_lstm = 8ULL * tensors->shape.batch_size * 
                           tensors->shape.sequence_length * 
                           context->config.hidden_size * 
                           (context->config.input_size + context->config.hidden_size);
    context->stats.gflops_achieved = (float)ops_per_lstm / (compute_time / 1000.0f);
    
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_process_sequence(MPSLSTMContext* context,
                                      const float* input,
                                      float* output,
                                      uint32_t sequence_length,
                                      const float* initial_hidden,
                                      const float* initial_cell,
                                      float* final_hidden,
                                      float* final_cell) {
    if (!context || !input || !output) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    // Create tensors structure for single sequence
    MPSLSTMShape shape = {
        .batch_size = 1,
        .sequence_length = sequence_length,
        .input_size = context->config.input_size,
        .hidden_size = context->config.hidden_size,
        .num_layers = context->config.num_layers,
        .num_directions = (uint32_t)((context->config.direction == MPS_LSTM_BIDIRECTIONAL) ? 2 : 1)
    };
    
    MPSLSTMTensors* tensors = NULL;
    MPSLSTMError error = mps_lstm_tensors_create(&tensors, &shape);
    if (error != MPS_LSTM_SUCCESS) {
        return error;
    }
    
    // Set tensor data
    tensors->input = (float*)input;
    tensors->output = output;
    tensors->initial_hidden = (float*)initial_hidden;
    tensors->initial_cell = (float*)initial_cell;
    tensors->final_hidden = final_hidden;
    tensors->final_cell = final_cell;
    tensors->owns_memory = false;
    
    // Perform forward pass
    error = mps_lstm_forward(context, tensors);
    
    // Cleanup
    free(tensors);
    
    return error;
}

MPSLSTMError mps_lstm_process_batch(MPSLSTMContext* context,
                                   const float* input,
                                   float* output,
                                   const uint32_t* sequence_lengths,
                                   uint32_t batch_size) {
    if (!context || !input || !output || !sequence_lengths) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    // Find maximum sequence length in batch
    uint32_t max_sequence_length = 0;
    for (uint32_t i = 0; i < batch_size; i++) {
        if (sequence_lengths[i] > max_sequence_length) {
            max_sequence_length = sequence_lengths[i];
        }
    }
    
    // Create tensors structure for batch
    MPSLSTMShape shape = {
        .batch_size = batch_size,
        .sequence_length = max_sequence_length,
        .input_size = context->config.input_size,
        .hidden_size = context->config.hidden_size,
        .num_layers = context->config.num_layers,
        .num_directions = (uint32_t)((context->config.direction == MPS_LSTM_BIDIRECTIONAL) ? 2 : 1)
    };
    
    MPSLSTMTensors* tensors = NULL;
    MPSLSTMError error = mps_lstm_tensors_create(&tensors, &shape);
    if (error != MPS_LSTM_SUCCESS) {
        return error;
    }
    
    // Create sequence mask for variable length sequences
    size_t mask_size = batch_size * max_sequence_length * sizeof(float);
    tensors->sequence_mask = (float*)malloc(mask_size);
    if (!tensors->sequence_mask) {
        mps_lstm_tensors_destroy(tensors);
        return MPS_LSTM_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize mask (1.0 for valid tokens, 0.0 for padding)
    float* mask = tensors->sequence_mask;
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < max_sequence_length; s++) {
            mask[b * max_sequence_length + s] = (s < sequence_lengths[b]) ? 1.0f : 0.0f;
        }
    }
    
    // Set tensor data
    tensors->input = (float*)input;
    tensors->output = output;
    tensors->owns_memory = true; // We allocated the mask
    
    // Perform forward pass
    error = mps_lstm_forward(context, tensors);
    
    // Cleanup
    mps_lstm_tensors_destroy(tensors);
    
    return error;
}

MPSLSTMError mps_lstm_reset_state(MPSLSTMContext* context) {
    if (!context) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    @autoreleasepool {
        // Clear hidden and cell state buffers
        if (context->buffers.hidden_buffer) {
            id<MTLBuffer> hiddenBuffer = (__bridge id<MTLBuffer>)context->buffers.hidden_buffer;
            memset([hiddenBuffer contents], 0, [hiddenBuffer length]);
        }
        
        if (context->buffers.cell_buffer) {
            id<MTLBuffer> cellBuffer = (__bridge id<MTLBuffer>)context->buffers.cell_buffer;
            memset([cellBuffer contents], 0, [cellBuffer length]);
        }
    }
#endif
    
    context->has_state = false;
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_get_state(MPSLSTMContext* context,
                               float* hidden_state,
                               float* cell_state) {
    if (!context || !hidden_state || !cell_state) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    if (!context->has_state) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    @autoreleasepool {
        size_t state_size = calculate_state_size(&context->config);
        
        if (context->buffers.hidden_buffer) {
            id<MTLBuffer> hiddenBuffer = (__bridge id<MTLBuffer>)context->buffers.hidden_buffer;
            memcpy(hidden_state, [hiddenBuffer contents], state_size);
        }
        
        if (context->buffers.cell_buffer) {
            id<MTLBuffer> cellBuffer = (__bridge id<MTLBuffer>)context->buffers.cell_buffer;
            memcpy(cell_state, [cellBuffer contents], state_size);
        }
    }
#endif
    
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_set_state(MPSLSTMContext* context,
                               const float* hidden_state,
                               const float* cell_state) {
    if (!context || !hidden_state || !cell_state) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    @autoreleasepool {
        size_t state_size = calculate_state_size(&context->config);
        
        if (context->buffers.hidden_buffer) {
            id<MTLBuffer> hiddenBuffer = (__bridge id<MTLBuffer>)context->buffers.hidden_buffer;
            memcpy([hiddenBuffer contents], hidden_state, state_size);
        }
        
        if (context->buffers.cell_buffer) {
            id<MTLBuffer> cellBuffer = (__bridge id<MTLBuffer>)context->buffers.cell_buffer;
            memcpy([cellBuffer contents], cell_state, state_size);
        }
    }
#endif
    
    context->has_state = true;
    return MPS_LSTM_SUCCESS;
}

// Configuration Functions

MPSLSTMError mps_lstm_config_create_default(MPSLSTMConfig* config,
                                           uint32_t input_size,
                                           uint32_t hidden_size,
                                           uint32_t num_layers) {
    if (!config || input_size == 0 || hidden_size == 0 || num_layers == 0) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(MPSLSTMConfig));
    
    config->input_size = input_size;
    config->hidden_size = hidden_size;
    config->num_layers = num_layers;
    config->batch_size = 1;
    config->sequence_length = 128;
    config->direction = MPS_LSTM_FORWARD;
    config->cell_type = MPS_LSTM_VANILLA;
    
    // Default activation functions
    config->input_activation = MPS_LSTM_ACTIVATION_SIGMOID;
    config->forget_activation = MPS_LSTM_ACTIVATION_SIGMOID;
    config->output_activation = MPS_LSTM_ACTIVATION_SIGMOID;
    config->cell_activation = MPS_LSTM_ACTIVATION_TANH;
    
    // Default settings
    config->dropout_rate = 0.0f;
    config->use_dropout_on_input = false;
    config->use_dropout_on_hidden = false;
    config->use_dropout_on_output = false;
    config->use_bias = true;
    config->use_peepholes = false;
    config->use_layer_norm = false;
    config->use_residual_connections = false;
    config->bidirectional_merge_mode = true; // concat
    config->stateful = false;
    config->return_sequences = true;
    config->return_state = false;
    config->max_memory_mb = 512;
    
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_config_validate(const MPSLSTMConfig* config) {
    if (!config) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    if (config->input_size == 0 || config->input_size > 8192) {
        return MPS_LSTM_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->hidden_size == 0 || config->hidden_size > 4096) {
        return MPS_LSTM_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->num_layers == 0 || config->num_layers > 32) {
        return MPS_LSTM_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->batch_size == 0 || config->batch_size > 256) {
        return MPS_LSTM_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->sequence_length == 0 || config->sequence_length > 16384) {
        return MPS_LSTM_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->dropout_rate < 0.0f || config->dropout_rate >= 1.0f) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_calculate_memory_requirements(const MPSLSTMConfig* config,
                                                   uint32_t* memory_mb) {
    if (!config || !memory_mb) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    // Calculate memory for tensors
    uint64_t tensor_memory = 0;
    
    // Input/Output tensors
    uint32_t num_directions = (config->direction == MPS_LSTM_BIDIRECTIONAL) ? 2 : 1;
    uint64_t input_size = (uint64_t)config->batch_size * 
                         config->sequence_length * 
                         config->input_size * sizeof(float);
    uint64_t output_size = (uint64_t)config->batch_size * 
                          config->sequence_length * 
                          config->hidden_size * num_directions * sizeof(float);
    
    tensor_memory += input_size + output_size;
    
    // Hidden and cell state tensors
    uint64_t state_size = (uint64_t)config->batch_size * 
                         config->num_layers * num_directions * 
                         config->hidden_size * sizeof(float);
    tensor_memory += state_size * 2; // hidden + cell
    
    // Weights memory
    uint64_t weights_memory = calculate_weights_size(config);
    
    // Temporary buffers (estimate 4x the main tensors for intermediate computations)
    uint64_t temp_memory = tensor_memory * 4;
    
    uint64_t total_memory = tensor_memory + weights_memory + temp_memory;
    *memory_mb = (uint32_t)((total_memory + 1024 * 1024 - 1) / (1024 * 1024));
    
    return MPS_LSTM_SUCCESS;
}

// Utility Functions

const char* mps_lstm_get_error_string(MPSLSTMError error_code) {
    if (error_code < 0 || error_code >= sizeof(error_messages) / sizeof(error_messages[0])) {
        return "Unknown error";
    }
    return error_messages[error_code];
}

bool mps_lstm_is_available(void) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device && [device supportsFamily:MTLGPUFamilyApple4]) {
            return true;
        }
    }
#endif
    return false;
}

MPSLSTMError mps_lstm_get_device_info(char* device_name,
                                     size_t buffer_size,
                                     uint32_t* compute_units,
                                     uint32_t* max_memory_mb) {
    if (!device_name || !compute_units || !max_memory_mb) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return MPS_LSTM_ERROR_DEVICE_NOT_FOUND;
        }
        
        NSString* name = device.name;
        strncpy(device_name, [name UTF8String], buffer_size - 1);
        device_name[buffer_size - 1] = '\0';
        
        if (@available(macOS 10.15, *)) {
            *compute_units = (uint32_t)device.maxThreadsPerThreadgroup.width;
            *max_memory_mb = (uint32_t)(device.recommendedMaxWorkingSetSize / (1024 * 1024));
        } else {
            *compute_units = 256;
            *max_memory_mb = 1024;
        }
    }
#else
    strncpy(device_name, "MPS Not Available", buffer_size - 1);
    device_name[buffer_size - 1] = '\0';
    *compute_units = 0;
    *max_memory_mb = 0;
#endif
    
    return MPS_LSTM_SUCCESS;
}

// Tensor Management Functions

MPSLSTMError mps_lstm_tensors_create(MPSLSTMTensors** tensors, const MPSLSTMShape* shape) {
    if (!tensors || !shape) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    *tensors = (MPSLSTMTensors*)calloc(1, sizeof(MPSLSTMTensors));
    if (!*tensors) {
        return MPS_LSTM_ERROR_MEMORY_ALLOCATION;
    }
    
    (*tensors)->shape = *shape;
    (*tensors)->owns_memory = false;
    
    return MPS_LSTM_SUCCESS;
}

void mps_lstm_tensors_destroy(MPSLSTMTensors* tensors) {
    if (!tensors) {
        return;
    }
    
    if (tensors->owns_memory) {
        free(tensors->input);
        free(tensors->initial_hidden);
        free(tensors->initial_cell);
        free(tensors->output);
        free(tensors->final_hidden);
        free(tensors->final_cell);
        free(tensors->sequence_mask);
    }
    
    free(tensors);
}

MPSLSTMError mps_lstm_tensors_validate(const MPSLSTMTensors* tensors,
                                      const MPSLSTMConfig* config) {
    if (!tensors || !config) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    if (!tensors->input || !tensors->output) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    if (tensors->shape.input_size != config->input_size ||
        tensors->shape.hidden_size != config->hidden_size) {
        return MPS_LSTM_ERROR_INVALID_DIMENSIONS;
    }
    
    uint32_t expected_directions = (config->direction == MPS_LSTM_BIDIRECTIONAL) ? 2 : 1;
    if (tensors->shape.num_directions != expected_directions) {
        return MPS_LSTM_ERROR_INVALID_DIMENSIONS;
    }
    
    return MPS_LSTM_SUCCESS;
}

// Statistics Functions

MPSLSTMError mps_lstm_get_stats(MPSLSTMContext* context, MPSLSTMStats* stats) {
    if (!context || !stats) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    *stats = context->stats;
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_reset_stats(MPSLSTMContext* context) {
    if (!context) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    memset(&context->stats, 0, sizeof(MPSLSTMStats));
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_get_memory_usage(MPSLSTMContext* context, uint32_t* memory_usage_mb) {
    if (!context || !memory_usage_mb) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    // Calculate current memory usage from buffers
    uint64_t total_memory = 0;
    for (int i = 0; i < 16; i++) {
        total_memory += context->buffers.buffer_sizes[i];
    }
    
    *memory_usage_mb = (uint32_t)((total_memory + 1024 * 1024 - 1) / (1024 * 1024));
    context->stats.memory_usage_mb = *memory_usage_mb;
    
    if (*memory_usage_mb > context->stats.peak_memory_usage_mb) {
        context->stats.peak_memory_usage_mb = *memory_usage_mb;
    }
    
    return MPS_LSTM_SUCCESS;
}

// Weights Management Functions

MPSLSTMError mps_lstm_weights_create(MPSLSTMWeights** weights, const MPSLSTMConfig* config) {
    if (!weights || !config) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    *weights = (MPSLSTMWeights*)calloc(1, sizeof(MPSLSTMWeights));
    if (!*weights) {
        return MPS_LSTM_ERROR_MEMORY_ALLOCATION;
    }
    
    // Calculate sizes (4 gates: input, forget, cell, output)
    (*weights)->input_weights_size = config->input_size * config->hidden_size * 4 * sizeof(float);
    (*weights)->hidden_weights_size = config->hidden_size * config->hidden_size * 4 * sizeof(float);
    (*weights)->bias_size = config->hidden_size * 4 * sizeof(float);
    (*weights)->peephole_size = config->hidden_size * sizeof(float);
    (*weights)->layer_norm_size = config->hidden_size * 4 * sizeof(float);
    
    // Allocate weights
    (*weights)->input_weights = malloc((*weights)->input_weights_size);
    (*weights)->hidden_weights = malloc((*weights)->hidden_weights_size);
    
    if (!(*weights)->input_weights || !(*weights)->hidden_weights) {
        mps_lstm_weights_destroy(*weights);
        *weights = NULL;
        return MPS_LSTM_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate biases if needed
    if (config->use_bias) {
        (*weights)->input_bias = malloc((*weights)->bias_size);
        (*weights)->hidden_bias = malloc((*weights)->bias_size);
        
        if (!(*weights)->input_bias || !(*weights)->hidden_bias) {
            mps_lstm_weights_destroy(*weights);
            *weights = NULL;
            return MPS_LSTM_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    // Allocate peephole weights if needed
    if (config->use_peepholes) {
        (*weights)->input_peephole = malloc((*weights)->peephole_size);
        (*weights)->forget_peephole = malloc((*weights)->peephole_size);
        (*weights)->output_peephole = malloc((*weights)->peephole_size);
        
        if (!(*weights)->input_peephole || !(*weights)->forget_peephole || !(*weights)->output_peephole) {
            mps_lstm_weights_destroy(*weights);
            *weights = NULL;
            return MPS_LSTM_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    // Allocate layer norm parameters if needed
    if (config->use_layer_norm) {
        (*weights)->layer_norm_weights = malloc((*weights)->layer_norm_size);
        (*weights)->layer_norm_bias = malloc((*weights)->layer_norm_size);
        
        if (!(*weights)->layer_norm_weights || !(*weights)->layer_norm_bias) {
            mps_lstm_weights_destroy(*weights);
            *weights = NULL;
            return MPS_LSTM_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    (*weights)->owns_memory = true;
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_weights_init_random(MPSLSTMWeights* weights,
                                         const MPSLSTMConfig* config,
                                         uint32_t seed) {
    if (!weights || !config) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    srand(seed);
    
    // Uniform initialization for LSTM gates
    float input_scale = 1.0f / sqrtf((float)config->input_size);
    float hidden_scale = 1.0f / sqrtf((float)config->hidden_size);
    
    // Initialize input-to-hidden weights
    if (weights->input_weights) {
        float* input_w = (float*)weights->input_weights;
        size_t num_elements = config->input_size * config->hidden_size * 4;
        for (size_t i = 0; i < num_elements; i++) {
            input_w[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * input_scale;
        }
    }
    
    // Initialize hidden-to-hidden weights
    if (weights->hidden_weights) {
        float* hidden_w = (float*)weights->hidden_weights;
        size_t num_elements = config->hidden_size * config->hidden_size * 4;
        for (size_t i = 0; i < num_elements; i++) {
            hidden_w[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * hidden_scale;
        }
    }
    
    // Initialize biases
    if (config->use_bias && weights->input_bias) {
        float* input_bias = (float*)weights->input_bias;
        float* hidden_bias = (float*)weights->hidden_bias;
        size_t num_elements = config->hidden_size * 4;
        
        for (size_t i = 0; i < num_elements; i++) {
            input_bias[i] = 0.0f;
            hidden_bias[i] = 0.0f;
            
            // Set forget gate bias to 1.0 to help with vanishing gradients
            if (i >= config->hidden_size && i < config->hidden_size * 2) {
                input_bias[i] = 1.0f;
                hidden_bias[i] = 1.0f;
            }
        }
    }
    
    // Initialize peephole weights
    if (config->use_peepholes && weights->input_peephole) {
        float* peephole_arrays[] = {
            (float*)weights->input_peephole,
            (float*)weights->forget_peephole,
            (float*)weights->output_peephole
        };
        
        for (int i = 0; i < 3; i++) {
            for (uint32_t j = 0; j < config->hidden_size; j++) {
                peephole_arrays[i][j] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * hidden_scale;
            }
        }
    }
    
    // Initialize layer norm parameters
    if (config->use_layer_norm && weights->layer_norm_weights) {
        float* ln_weights = (float*)weights->layer_norm_weights;
        float* ln_bias = (float*)weights->layer_norm_bias;
        size_t num_elements = config->hidden_size * 4;
        
        for (size_t i = 0; i < num_elements; i++) {
            ln_weights[i] = 1.0f;
            ln_bias[i] = 0.0f;
        }
    }
    
    return MPS_LSTM_SUCCESS;
}

MPSLSTMError mps_lstm_weights_init_xavier(MPSLSTMWeights* weights,
                                         const MPSLSTMConfig* config,
                                         uint32_t seed) {
    if (!weights || !config) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
    srand(seed);
    
    // Xavier/Glorot initialization
    float input_scale = sqrtf(6.0f / (config->input_size + config->hidden_size * 4));
    float hidden_scale = sqrtf(6.0f / (config->hidden_size + config->hidden_size * 4));
    
    // Initialize input-to-hidden weights
    if (weights->input_weights) {
        float* input_w = (float*)weights->input_weights;
        size_t num_elements = config->input_size * config->hidden_size * 4;
        for (size_t i = 0; i < num_elements; i++) {
            input_w[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * input_scale;
        }
    }
    
    // Initialize hidden-to-hidden weights
    if (weights->hidden_weights) {
        float* hidden_w = (float*)weights->hidden_weights;
        size_t num_elements = config->hidden_size * config->hidden_size * 4;
        for (size_t i = 0; i < num_elements; i++) {
            hidden_w[i] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * hidden_scale;
        }
    }
    
    // Use same bias initialization as random
    return mps_lstm_weights_init_random(weights, config, seed);
}

void mps_lstm_weights_destroy(MPSLSTMWeights* weights) {
    if (!weights) {
        return;
    }
    
    if (weights->owns_memory) {
        free(weights->input_weights);
        free(weights->hidden_weights);
        free(weights->input_bias);
        free(weights->hidden_bias);
        free(weights->input_peephole);
        free(weights->forget_peephole);
        free(weights->output_peephole);
        free(weights->layer_norm_weights);
        free(weights->layer_norm_bias);
    }
    
    free(weights);
}

// Buffer Management Functions

MPSLSTMError mps_lstm_free_buffers(MPSLSTMContext* context) {
    if (!context) {
        return MPS_LSTM_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    // Clear buffer pointers (objects will be released automatically)
    context->buffers.input_buffer = NULL;
    context->buffers.hidden_buffer = NULL;
    context->buffers.cell_buffer = NULL;
    context->buffers.output_buffer = NULL;
    context->buffers.weights_buffer = NULL;
    context->buffers.bias_buffer = NULL;
    
    for (int i = 0; i < 8; i++) {
        context->buffers.temp_buffers[i] = NULL;
    }
#endif
    
    memset(&context->buffers, 0, sizeof(MPSLSTMBuffers));
    return MPS_LSTM_SUCCESS;
}

void mps_lstm_destroy(MPSLSTMContext* context) {
    if (!context) {
        return;
    }
    
    // Free Metal buffers
    mps_lstm_free_buffers(context);
    
    // Clear Metal objects (objects will be released automatically)
    context->device = NULL;
    context->command_queue = NULL;
    context->mps_graph = NULL;
    context->graph_executable = NULL;
    
    free(context);
}

// Internal helper function implementations

static MPSLSTMError create_metal_device(MPSLSTMContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return MPS_LSTM_ERROR_DEVICE_NOT_FOUND;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            return MPS_LSTM_ERROR_DEVICE_NOT_FOUND;
        }
        
        context->device = (__bridge void*)device;
        context->command_queue = (__bridge void*)commandQueue;
    }
    return MPS_LSTM_SUCCESS;
#else
    return MPS_LSTM_ERROR_DEVICE_NOT_FOUND;
#endif
}

static MPSLSTMError build_lstm_graph(MPSLSTMContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        MPSGraph* graph = [[MPSGraph alloc] init];
        if (!graph) {
            return MPS_LSTM_ERROR_GRAPH_COMPILATION;
        }
        
        context->mps_graph = (__bridge void*)graph;
        
        // For now, we'll build a placeholder graph structure
        // In a full implementation, this would create the complete LSTM computation graph
        // with all gates, activations, and state management
        
        return MPS_LSTM_SUCCESS;
    }
#else
    return MPS_LSTM_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static MPSLSTMError allocate_metal_buffers(MPSLSTMContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        
        // Calculate buffer sizes
        uint32_t num_directions = (context->config.direction == MPS_LSTM_BIDIRECTIONAL) ? 2 : 1;
        
        size_t input_size = context->config.batch_size * 
                           context->config.sequence_length * 
                           context->config.input_size * sizeof(float);
        
        size_t output_size = context->config.batch_size * 
                            context->config.sequence_length * 
                            context->config.hidden_size * num_directions * sizeof(float);
        
        size_t state_size = context->config.batch_size * 
                           context->config.num_layers * num_directions * 
                           context->config.hidden_size * sizeof(float);
        
        // Allocate main buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithLength:input_size 
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_size 
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> hiddenBuffer = [device newBufferWithLength:state_size 
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> cellBuffer = [device newBufferWithLength:state_size 
                                                        options:MTLResourceStorageModeShared];
        
        if (!inputBuffer || !outputBuffer || !hiddenBuffer || !cellBuffer) {
            return MPS_LSTM_ERROR_BUFFER_ALLOCATION;
        }
        
        context->buffers.input_buffer = (__bridge void*)inputBuffer;
        context->buffers.output_buffer = (__bridge void*)outputBuffer;
        context->buffers.hidden_buffer = (__bridge void*)hiddenBuffer;
        context->buffers.cell_buffer = (__bridge void*)cellBuffer;
        
        // Store buffer sizes
        context->buffers.buffer_sizes[0] = input_size;
        context->buffers.buffer_sizes[1] = output_size;
        context->buffers.buffer_sizes[2] = state_size;
        context->buffers.buffer_sizes[3] = state_size;
        
        // Allocate temporary buffers for intermediate computations
        size_t temp_size = context->config.batch_size * 
                          context->config.hidden_size * 4 * sizeof(float); // 4 gates
        
        for (int i = 0; i < 8; i++) {
            id<MTLBuffer> tempBuffer = [device newBufferWithLength:temp_size 
                                                           options:MTLResourceStorageModePrivate];
            if (!tempBuffer) {
                return MPS_LSTM_ERROR_BUFFER_ALLOCATION;
            }
            context->buffers.temp_buffers[i] = (__bridge void*)tempBuffer;
            context->buffers.buffer_sizes[4 + i] = temp_size;
        }
    }
    return MPS_LSTM_SUCCESS;
#else
    return MPS_LSTM_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static MPSLSTMError execute_lstm_graph(MPSLSTMContext* context, MPSLSTMTensors* tensors) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)context->command_queue;
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (!commandBuffer) {
            return MPS_LSTM_ERROR_EXECUTION_FAILED;
        }
        
        // Copy input data to Metal buffers
        id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)context->buffers.input_buffer;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)context->buffers.output_buffer;
        
        size_t input_size = calculate_tensor_size(&tensors->shape, false);
        size_t output_size = calculate_tensor_size(&tensors->shape, true);
        
        memcpy([inputBuffer contents], tensors->input, input_size);
        
        // For now, implement a simple copy operation as placeholder
        // In a full implementation, this would execute the complete LSTM computation
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder copyFromBuffer:inputBuffer 
                       sourceOffset:0 
                           toBuffer:outputBuffer 
                  destinationOffset:0 
                               size:MIN(input_size, output_size)];
        [blitEncoder endEncoding];
        
        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back
        memcpy(tensors->output, [outputBuffer contents], output_size);
        
        context->has_state = true;
    }
    return MPS_LSTM_SUCCESS;
#else
    // Fallback implementation for non-Metal platforms
    size_t input_size = calculate_tensor_size(&tensors->shape, false);
    size_t output_size = calculate_tensor_size(&tensors->shape, true);
    size_t copy_size = MIN(input_size, output_size);
    memcpy(tensors->output, tensors->input, copy_size);
    return MPS_LSTM_SUCCESS;
#endif
}

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

static size_t calculate_tensor_size(const MPSLSTMShape* shape, bool include_directions) {
    size_t base_size = (size_t)shape->batch_size * 
                      shape->sequence_length * 
                      sizeof(float);
    
    if (include_directions) {
        return base_size * shape->hidden_size * shape->num_directions;
    } else {
        return base_size * shape->input_size;
    }
}

static size_t calculate_weights_size(const MPSLSTMConfig* config) {
    size_t weights_size = 0;
    
    // Input-to-hidden and hidden-to-hidden weights (4 gates each)
    weights_size += config->input_size * config->hidden_size * 4 * sizeof(float);
    weights_size += config->hidden_size * config->hidden_size * 4 * sizeof(float);
    
    // Biases
    if (config->use_bias) {
        weights_size += config->hidden_size * 4 * 2 * sizeof(float); // input + hidden bias
    }
    
    // Peephole connections
    if (config->use_peepholes) {
        weights_size += config->hidden_size * 3 * sizeof(float); // input, forget, output
    }
    
    // Layer normalization
    if (config->use_layer_norm) {
        weights_size += config->hidden_size * 4 * 2 * sizeof(float); // weights + bias
    }
    
    // Multiply by number of layers and directions
    uint32_t num_directions = (config->direction == MPS_LSTM_BIDIRECTIONAL) ? 2 : 1;
    weights_size *= config->num_layers * num_directions;
    
    return weights_size;
}

static size_t calculate_state_size(const MPSLSTMConfig* config) {
    uint32_t num_directions = (config->direction == MPS_LSTM_BIDIRECTIONAL) ? 2 : 1;
    return (size_t)config->batch_size * 
           config->num_layers * num_directions * 
           config->hidden_size * sizeof(float);
}