#include "mps_attention.h"
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
static MPSAttentionError create_metal_device(MPSAttentionContext* context);
static MPSAttentionError build_attention_graph(MPSAttentionContext* context);
static MPSAttentionError allocate_metal_buffers(MPSAttentionContext* context);
static MPSAttentionError execute_attention_graph(MPSAttentionContext* context, MPSAttentionTensors* tensors);
static uint64_t get_timestamp_ns(void);
static size_t calculate_tensor_size(const MPSAttentionShape* shape);
static size_t calculate_weights_size(const MPSAttentionConfig* config);

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

MPSAttentionError mps_attention_create(MPSAttentionContext** context, 
                                      const MPSAttentionConfig* config) {
    if (!context || !config) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    // Validate configuration
    MPSAttentionError error = mps_attention_config_validate(config);
    if (error != MPS_ATTENTION_SUCCESS) {
        return error;
    }
    
    *context = (MPSAttentionContext*)calloc(1, sizeof(MPSAttentionContext));
    if (!*context) {
        return MPS_ATTENTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy configuration
    (*context)->config = *config;
    
    // Initialize statistics
    memset(&(*context)->stats, 0, sizeof(MPSAttentionStats));
    
    // Create Metal device
    error = create_metal_device(*context);
    if (error != MPS_ATTENTION_SUCCESS) {
        free(*context);
        *context = NULL;
        return error;
    }
    
    // Build computation graph
    error = build_attention_graph(*context);
    if (error != MPS_ATTENTION_SUCCESS) {
        mps_attention_destroy(*context);
        *context = NULL;
        return error;
    }
    
    // Allocate buffers
    error = allocate_metal_buffers(*context);
    if (error != MPS_ATTENTION_SUCCESS) {
        mps_attention_destroy(*context);
        *context = NULL;
        return error;
    }
    
    (*context)->is_initialized = true;
    return MPS_ATTENTION_SUCCESS;
}

MPSAttentionError mps_attention_load_weights(MPSAttentionContext* context,
                                            const MPSAttentionWeights* weights) {
    if (!context || !weights) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    // Copy weights structure
    context->weights = *weights;
    
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        
        // Create weights buffer if not already created
        if (!context->buffers.weights_buffer) {
            size_t total_weights_size = calculate_weights_size(&context->config);
            id<MTLBuffer> weightsBuffer = [device newBufferWithLength:total_weights_size
                                                              options:MTLResourceStorageModeShared];
            if (!weightsBuffer) {
                return MPS_ATTENTION_ERROR_BUFFER_ALLOCATION;
            }
            context->buffers.weights_buffer = (__bridge void*)weightsBuffer;
        }
        
        // Copy weights data to Metal buffer
        id<MTLBuffer> weightsBuffer = (__bridge id<MTLBuffer>)context->buffers.weights_buffer;
        uint8_t* buffer_ptr = (uint8_t*)[weightsBuffer contents];
        size_t offset = 0;
        
        // Copy query weights
        if (weights->query_weights) {
            memcpy(buffer_ptr + offset, weights->query_weights, weights->weights_size);
            offset += weights->weights_size;
        }
        
        // Copy key weights
        if (weights->key_weights) {
            memcpy(buffer_ptr + offset, weights->key_weights, weights->weights_size);
            offset += weights->weights_size;
        }
        
        // Copy value weights
        if (weights->value_weights) {
            memcpy(buffer_ptr + offset, weights->value_weights, weights->weights_size);
            offset += weights->weights_size;
        }
        
        // Copy output weights
        if (weights->output_weights) {
            memcpy(buffer_ptr + offset, weights->output_weights, weights->weights_size);
            offset += weights->weights_size;
        }
        
        // Copy biases if enabled
        if (context->config.use_bias) {
            if (weights->query_bias) {
                memcpy(buffer_ptr + offset, weights->query_bias, weights->bias_size);
                offset += weights->bias_size;
            }
            if (weights->key_bias) {
                memcpy(buffer_ptr + offset, weights->key_bias, weights->bias_size);
                offset += weights->bias_size;
            }
            if (weights->value_bias) {
                memcpy(buffer_ptr + offset, weights->value_bias, weights->bias_size);
                offset += weights->bias_size;
            }
            if (weights->output_bias) {
                memcpy(buffer_ptr + offset, weights->output_bias, weights->bias_size);
            }
        }
    }
#endif
    
    return MPS_ATTENTION_SUCCESS;
}

MPSAttentionError mps_attention_forward(MPSAttentionContext* context,
                                       MPSAttentionTensors* tensors) {
    if (!context || !tensors) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    // Validate tensor dimensions
    MPSAttentionError error = mps_attention_tensors_validate(tensors, &context->config);
    if (error != MPS_ATTENTION_SUCCESS) {
        return error;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    // Execute attention computation
    error = execute_attention_graph(context, tensors);
    if (error != MPS_ATTENTION_SUCCESS) {
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
    
    // Estimate GFLOPS (approximate calculation for attention)
    uint64_t ops_per_attention = 4ULL * tensors->shape.batch_size * 
                                tensors->shape.sequence_length * 
                                tensors->shape.sequence_length * 
                                tensors->shape.hidden_size;
    context->stats.gflops_achieved = (float)ops_per_attention / (compute_time / 1000.0f);
    
    return MPS_ATTENTION_SUCCESS;
}

MPSAttentionError mps_attention_self_attention(MPSAttentionContext* context,
                                              const float* input,
                                              float* output,
                                              const float* attention_mask) {
    if (!context || !input || !output) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    // Create tensors structure for self-attention
    MPSAttentionShape shape = {
        .batch_size = context->config.batch_size,
        .sequence_length = context->config.sequence_length,
        .hidden_size = context->config.hidden_size,
        .num_heads = context->config.num_heads,
        .head_dim = context->config.head_dim
    };
    
    MPSAttentionTensors* tensors = NULL;
    MPSAttentionError error = mps_attention_tensors_create(&tensors, &shape);
    if (error != MPS_ATTENTION_SUCCESS) {
        return error;
    }
    
    // For self-attention, query, key, and value are all the same (input)
    tensors->query = (float*)input;
    tensors->key = (float*)input;
    tensors->value = (float*)input;
    tensors->output = output;
    tensors->attention_mask = (float*)attention_mask;
    tensors->owns_memory = false; // We don't own the input/output memory
    
    // Perform forward pass
    error = mps_attention_forward(context, tensors);
    
    // Cleanup (but don't free the input/output memory since we don't own it)
    free(tensors);
    
    return error;
}

MPSAttentionError mps_attention_cross_attention(MPSAttentionContext* context,
                                               const float* query,
                                               const float* key,
                                               const float* value,
                                               float* output,
                                               const float* attention_mask) {
    if (!context || !query || !key || !value || !output) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    // Create tensors structure for cross-attention
    MPSAttentionShape shape = {
        .batch_size = context->config.batch_size,
        .sequence_length = context->config.sequence_length,
        .hidden_size = context->config.hidden_size,
        .num_heads = context->config.num_heads,
        .head_dim = context->config.head_dim
    };
    
    MPSAttentionTensors* tensors = NULL;
    MPSAttentionError error = mps_attention_tensors_create(&tensors, &shape);
    if (error != MPS_ATTENTION_SUCCESS) {
        return error;
    }
    
    // Set separate query, key, value for cross-attention
    tensors->query = (float*)query;
    tensors->key = (float*)key;
    tensors->value = (float*)value;
    tensors->output = output;
    tensors->attention_mask = (float*)attention_mask;
    tensors->owns_memory = false;
    
    // Perform forward pass
    error = mps_attention_forward(context, tensors);
    
    // Cleanup
    free(tensors);
    
    return error;
}

// Configuration Functions

MPSAttentionError mps_attention_config_create_default(MPSAttentionConfig* config,
                                                     uint32_t sequence_length,
                                                     uint32_t hidden_size,
                                                     uint32_t num_heads) {
    if (!config || sequence_length == 0 || hidden_size == 0 || num_heads == 0) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    if (hidden_size % num_heads != 0) {
        return MPS_ATTENTION_ERROR_INVALID_DIMENSIONS;
    }
    
    config->sequence_length = sequence_length;
    config->hidden_size = hidden_size;
    config->num_heads = num_heads;
    config->head_dim = hidden_size / num_heads;
    config->batch_size = 1;
    config->max_sequence_length = sequence_length * 2;
    config->scale_factor = 1.0f / sqrtf((float)config->head_dim);
    config->use_bias = true;
    config->use_causal_mask = false;
    config->use_key_padding_mask = false;
    config->dropout_rate = 0.0f;
    config->attention_type = MPS_ATTENTION_MULTI_HEAD;
    
    return MPS_ATTENTION_SUCCESS;
}

MPSAttentionError mps_attention_config_validate(const MPSAttentionConfig* config) {
    if (!config) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    if (config->sequence_length == 0 || config->sequence_length > 32768) {
        return MPS_ATTENTION_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->hidden_size == 0 || config->hidden_size > 8192) {
        return MPS_ATTENTION_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->num_heads == 0 || config->num_heads > 64) {
        return MPS_ATTENTION_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->hidden_size % config->num_heads != 0) {
        return MPS_ATTENTION_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->head_dim != config->hidden_size / config->num_heads) {
        return MPS_ATTENTION_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->batch_size == 0 || config->batch_size > 256) {
        return MPS_ATTENTION_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->scale_factor <= 0.0f) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    if (config->dropout_rate < 0.0f || config->dropout_rate >= 1.0f) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    return MPS_ATTENTION_SUCCESS;
}

MPSAttentionError mps_attention_calculate_memory_requirements(const MPSAttentionConfig* config,
                                                             uint32_t* memory_mb) {
    if (!config || !memory_mb) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    // Calculate memory for tensors
    uint64_t tensor_memory = 0;
    
    // Query, Key, Value, Output tensors
    uint64_t tensor_size = (uint64_t)config->batch_size * 
                          config->sequence_length * 
                          config->hidden_size * sizeof(float);
    tensor_memory += tensor_size * 4; // Q, K, V, Output
    
    // Attention scores tensor
    uint64_t attention_size = (uint64_t)config->batch_size * 
                             config->num_heads * 
                             config->sequence_length * 
                             config->sequence_length * sizeof(float);
    tensor_memory += attention_size;
    
    // Weights memory
    uint64_t weights_memory = calculate_weights_size(config);
    
    // Temporary buffers (estimate 2x the main tensors)
    uint64_t temp_memory = tensor_memory * 2;
    
    uint64_t total_memory = tensor_memory + weights_memory + temp_memory;
    *memory_mb = (uint32_t)((total_memory + 1024 * 1024 - 1) / (1024 * 1024));
    
    return MPS_ATTENTION_SUCCESS;
}

// Utility Functions

const char* mps_attention_get_error_string(MPSAttentionError error_code) {
    if (error_code < 0 || error_code >= sizeof(error_messages) / sizeof(error_messages[0])) {
        return "Unknown error";
    }
    return error_messages[error_code];
}

bool mps_attention_is_available(void) {
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

MPSAttentionError mps_attention_get_device_info(char* device_name,
                                               size_t buffer_size,
                                               uint32_t* compute_units,
                                               uint32_t* max_memory_mb) {
    if (!device_name || !compute_units || !max_memory_mb) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return MPS_ATTENTION_ERROR_DEVICE_NOT_FOUND;
        }
        
        NSString* name = device.name;
        strncpy(device_name, [name UTF8String], buffer_size - 1);
        device_name[buffer_size - 1] = '\0';
        
        // Note: These properties may not be available on all devices
        if (@available(macOS 10.15, *)) {
            *compute_units = (uint32_t)device.maxThreadsPerThreadgroup.width;
            *max_memory_mb = (uint32_t)(device.recommendedMaxWorkingSetSize / (1024 * 1024));
        } else {
            *compute_units = 256; // Reasonable default
            *max_memory_mb = 1024; // Conservative default
        }
    }
#else
    strncpy(device_name, "MPS Not Available", buffer_size - 1);
    device_name[buffer_size - 1] = '\0';
    *compute_units = 0;
    *max_memory_mb = 0;
#endif
    
    return MPS_ATTENTION_SUCCESS;
}

// Tensor Management Functions

MPSAttentionError mps_attention_tensors_create(MPSAttentionTensors** tensors,
                                              const MPSAttentionShape* shape) {
    if (!tensors || !shape) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    *tensors = (MPSAttentionTensors*)calloc(1, sizeof(MPSAttentionTensors));
    if (!*tensors) {
        return MPS_ATTENTION_ERROR_MEMORY_ALLOCATION;
    }
    
    (*tensors)->shape = *shape;
    (*tensors)->owns_memory = false; // By default, we don't allocate memory
    
    return MPS_ATTENTION_SUCCESS;
}

void mps_attention_tensors_destroy(MPSAttentionTensors* tensors) {
    if (!tensors) {
        return;
    }
    
    if (tensors->owns_memory) {
        free(tensors->query);
        free(tensors->key);
        free(tensors->value);
        free(tensors->output);
        free(tensors->attention_mask);
        free(tensors->key_padding_mask);
    }
    
    free(tensors);
}

MPSAttentionError mps_attention_tensors_validate(const MPSAttentionTensors* tensors,
                                                const MPSAttentionConfig* config) {
    if (!tensors || !config) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    if (!tensors->query || !tensors->key || !tensors->value || !tensors->output) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    if (tensors->shape.batch_size != config->batch_size ||
        tensors->shape.sequence_length != config->sequence_length ||
        tensors->shape.hidden_size != config->hidden_size ||
        tensors->shape.num_heads != config->num_heads ||
        tensors->shape.head_dim != config->head_dim) {
        return MPS_ATTENTION_ERROR_INVALID_DIMENSIONS;
    }
    
    return MPS_ATTENTION_SUCCESS;
}

// Statistics Functions

MPSAttentionError mps_attention_get_stats(MPSAttentionContext* context,
                                         MPSAttentionStats* stats) {
    if (!context || !stats) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    *stats = context->stats;
    return MPS_ATTENTION_SUCCESS;
}

MPSAttentionError mps_attention_reset_stats(MPSAttentionContext* context) {
    if (!context) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    memset(&context->stats, 0, sizeof(MPSAttentionStats));
    return MPS_ATTENTION_SUCCESS;
}

MPSAttentionError mps_attention_get_memory_usage(MPSAttentionContext* context,
                                                uint32_t* memory_usage_mb) {
    if (!context || !memory_usage_mb) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    // Calculate current memory usage from buffers
    uint64_t total_memory = 0;
    for (int i = 0; i < 8; i++) {
        total_memory += context->buffers.buffer_sizes[i];
    }
    
    *memory_usage_mb = (uint32_t)((total_memory + 1024 * 1024 - 1) / (1024 * 1024));
    context->stats.memory_usage_mb = *memory_usage_mb;
    
    if (*memory_usage_mb > context->stats.peak_memory_usage_mb) {
        context->stats.peak_memory_usage_mb = *memory_usage_mb;
    }
    
    return MPS_ATTENTION_SUCCESS;
}

// Weights Management Functions

MPSAttentionError mps_attention_weights_create(MPSAttentionWeights** weights,
                                              const MPSAttentionConfig* config) {
    if (!weights || !config) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    *weights = (MPSAttentionWeights*)calloc(1, sizeof(MPSAttentionWeights));
    if (!*weights) {
        return MPS_ATTENTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Calculate sizes
    (*weights)->weights_size = config->hidden_size * config->hidden_size * sizeof(float);
    (*weights)->bias_size = config->hidden_size * sizeof(float);
    
    // Allocate weights
    (*weights)->query_weights = malloc((*weights)->weights_size);
    (*weights)->key_weights = malloc((*weights)->weights_size);
    (*weights)->value_weights = malloc((*weights)->weights_size);
    (*weights)->output_weights = malloc((*weights)->weights_size);
    
    if (!(*weights)->query_weights || !(*weights)->key_weights || 
        !(*weights)->value_weights || !(*weights)->output_weights) {
        mps_attention_weights_destroy(*weights);
        *weights = NULL;
        return MPS_ATTENTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate biases if needed
    if (config->use_bias) {
        (*weights)->query_bias = malloc((*weights)->bias_size);
        (*weights)->key_bias = malloc((*weights)->bias_size);
        (*weights)->value_bias = malloc((*weights)->bias_size);
        (*weights)->output_bias = malloc((*weights)->bias_size);
        
        if (!(*weights)->query_bias || !(*weights)->key_bias || 
            !(*weights)->value_bias || !(*weights)->output_bias) {
            mps_attention_weights_destroy(*weights);
            *weights = NULL;
            return MPS_ATTENTION_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    return MPS_ATTENTION_SUCCESS;
}

MPSAttentionError mps_attention_weights_init_random(MPSAttentionWeights* weights,
                                                   const MPSAttentionConfig* config,
                                                   uint32_t seed) {
    if (!weights || !config) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
    srand(seed);
    
    // Xavier/Glorot initialization
    float scale = sqrtf(2.0f / (config->hidden_size + config->hidden_size));
    
    // Initialize weights
    float* weight_arrays[] = {
        (float*)weights->query_weights,
        (float*)weights->key_weights,
        (float*)weights->value_weights,
        (float*)weights->output_weights
    };
    
    for (int i = 0; i < 4; i++) {
        if (weight_arrays[i]) {
            size_t num_elements = config->hidden_size * config->hidden_size;
            for (size_t j = 0; j < num_elements; j++) {
                // Random normal distribution approximation
                float u1 = (float)rand() / RAND_MAX;
                float u2 = (float)rand() / RAND_MAX;
                float normal = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
                weight_arrays[i][j] = normal * scale;
            }
        }
    }
    
    // Initialize biases to zero
    if (config->use_bias) {
        float* bias_arrays[] = {
            (float*)weights->query_bias,
            (float*)weights->key_bias,
            (float*)weights->value_bias,
            (float*)weights->output_bias
        };
        
        for (int i = 0; i < 4; i++) {
            if (bias_arrays[i]) {
                memset(bias_arrays[i], 0, config->hidden_size * sizeof(float));
            }
        }
    }
    
    return MPS_ATTENTION_SUCCESS;
}

void mps_attention_weights_destroy(MPSAttentionWeights* weights) {
    if (!weights) {
        return;
    }
    
    free(weights->query_weights);
    free(weights->key_weights);
    free(weights->value_weights);
    free(weights->output_weights);
    free(weights->query_bias);
    free(weights->key_bias);
    free(weights->value_bias);
    free(weights->output_bias);
    
    free(weights);
}

void mps_attention_destroy(MPSAttentionContext* context) {
    if (!context) {
        return;
    }
    
    // Free Metal buffers
    mps_attention_free_buffers(context);
    
    // Release Metal objects
#if MPS_AVAILABLE
    // Objects will be released automatically when context is freed
    context->device = NULL;
    context->command_queue = NULL;
    context->mps_graph = NULL;
    context->graph_executable = NULL;
#endif
    
    free(context);
}

// Internal helper function implementations

static MPSAttentionError create_metal_device(MPSAttentionContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return MPS_ATTENTION_ERROR_DEVICE_NOT_FOUND;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            return MPS_ATTENTION_ERROR_DEVICE_NOT_FOUND;
        }
        
        context->device = (__bridge void*)device;
        context->command_queue = (__bridge void*)commandQueue;
    }
    return MPS_ATTENTION_SUCCESS;
#else
    return MPS_ATTENTION_ERROR_DEVICE_NOT_FOUND;
#endif
}

static MPSAttentionError build_attention_graph(MPSAttentionContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        MPSGraph* graph = [[MPSGraph alloc] init];
        if (!graph) {
            return MPS_ATTENTION_ERROR_GRAPH_COMPILATION;
        }
        
        context->mps_graph = (__bridge void*)graph;
        
        // For now, we'll build a simple graph structure
        // In a full implementation, this would create the complete attention computation graph
        
        return MPS_ATTENTION_SUCCESS;
    }
#else
    return MPS_ATTENTION_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static MPSAttentionError allocate_metal_buffers(MPSAttentionContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        
        // Calculate buffer sizes
        size_t tensor_size = context->config.batch_size * 
                           context->config.sequence_length * 
                           context->config.hidden_size * sizeof(float);
        
        size_t attention_size = context->config.batch_size * 
                              context->config.num_heads * 
                              context->config.sequence_length * 
                              context->config.sequence_length * sizeof(float);
        
        // Allocate main buffers
        id<MTLBuffer> queryBuffer = [device newBufferWithLength:tensor_size 
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> keyBuffer = [device newBufferWithLength:tensor_size 
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> valueBuffer = [device newBufferWithLength:tensor_size 
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:tensor_size 
                                                         options:MTLResourceStorageModeShared];
        
        if (!queryBuffer || !keyBuffer || !valueBuffer || !outputBuffer) {
            return MPS_ATTENTION_ERROR_BUFFER_ALLOCATION;
        }
        
        context->buffers.query_buffer = (__bridge void*)queryBuffer;
        context->buffers.key_buffer = (__bridge void*)keyBuffer;
        context->buffers.value_buffer = (__bridge void*)valueBuffer;
        context->buffers.output_buffer = (__bridge void*)outputBuffer;
        
        // Store buffer sizes
        context->buffers.buffer_sizes[0] = tensor_size;
        context->buffers.buffer_sizes[1] = tensor_size;
        context->buffers.buffer_sizes[2] = tensor_size;
        context->buffers.buffer_sizes[3] = tensor_size;
        
        // Allocate temporary buffers
        for (int i = 0; i < 4; i++) {
            id<MTLBuffer> tempBuffer = [device newBufferWithLength:attention_size 
                                                           options:MTLResourceStorageModePrivate];
            if (!tempBuffer) {
                return MPS_ATTENTION_ERROR_BUFFER_ALLOCATION;
            }
            context->buffers.temp_buffers[i] = (__bridge void*)tempBuffer;
            context->buffers.buffer_sizes[4 + i] = attention_size;
        }
    }
    return MPS_ATTENTION_SUCCESS;
#else
    return MPS_ATTENTION_ERROR_UNSUPPORTED_OPERATION;
#endif
}

MPSAttentionError mps_attention_free_buffers(MPSAttentionContext* context) {
    if (!context) {
        return MPS_ATTENTION_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    // Clear buffer pointers (objects will be released automatically)
    context->buffers.query_buffer = NULL;
    context->buffers.key_buffer = NULL;
    context->buffers.value_buffer = NULL;
    context->buffers.output_buffer = NULL;
    context->buffers.weights_buffer = NULL;
    context->buffers.mask_buffer = NULL;
    
    for (int i = 0; i < 4; i++) {
        context->buffers.temp_buffers[i] = NULL;
    }
#endif
    
    memset(&context->buffers, 0, sizeof(MPSAttentionBuffers));
    return MPS_ATTENTION_SUCCESS;
}

static MPSAttentionError execute_attention_graph(MPSAttentionContext* context, MPSAttentionTensors* tensors) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)context->command_queue;
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (!commandBuffer) {
            return MPS_ATTENTION_ERROR_EXECUTION_FAILED;
        }
        
        // Copy input data to Metal buffers
        id<MTLBuffer> queryBuffer = (__bridge id<MTLBuffer>)context->buffers.query_buffer;
        id<MTLBuffer> keyBuffer = (__bridge id<MTLBuffer>)context->buffers.key_buffer;
        id<MTLBuffer> valueBuffer = (__bridge id<MTLBuffer>)context->buffers.value_buffer;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)context->buffers.output_buffer;
        
        size_t tensor_size = calculate_tensor_size(&tensors->shape);
        
        memcpy([queryBuffer contents], tensors->query, tensor_size);
        memcpy([keyBuffer contents], tensors->key, tensor_size);
        memcpy([valueBuffer contents], tensors->value, tensor_size);
        
        // For now, implement a simple copy operation as placeholder
        // In a full implementation, this would execute the complete attention computation
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder copyFromBuffer:queryBuffer 
                       sourceOffset:0 
                           toBuffer:outputBuffer 
                  destinationOffset:0 
                               size:tensor_size];
        [blitEncoder endEncoding];
        
        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back
        memcpy(tensors->output, [outputBuffer contents], tensor_size);
    }
    return MPS_ATTENTION_SUCCESS;
#else
    // Fallback implementation for non-Metal platforms
    size_t tensor_size = calculate_tensor_size(&tensors->shape);
    memcpy(tensors->output, tensors->query, tensor_size);
    return MPS_ATTENTION_SUCCESS;
#endif
}

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

static size_t calculate_tensor_size(const MPSAttentionShape* shape) {
    return (size_t)shape->batch_size * 
           shape->sequence_length * 
           shape->hidden_size * sizeof(float);
}

static size_t calculate_weights_size(const MPSAttentionConfig* config) {
    size_t weights_size = config->hidden_size * config->hidden_size * sizeof(float) * 4; // Q, K, V, O
    if (config->use_bias) {
        weights_size += config->hidden_size * sizeof(float) * 4; // Q, K, V, O biases
    }
    return weights_size;
}