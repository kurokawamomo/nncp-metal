/*
 * MixedPrecisionEngine.mm
 * 
 * Mixed Precision Computing Engine Implementation for Apple Silicon
 * Intelligent precision selection with float32 for critical operations
 * and float16 acceleration for FFN intermediate layers
 */

#include "MixedPrecisionEngine.h"
#include "../acceleration/MetalComputeAccelerator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreFoundation/CoreFoundation.h>
#endif

// Internal structure for MixedPrecisionEngine
struct MixedPrecisionEngine {
    MixedPrecisionConfig config;                    // Engine configuration
    MetalComputeAccelerator* metal_accelerator;     // Metal GPU accelerator
    MixedPrecisionStats stats;                      // Performance statistics
    NumericalStabilityAnalysis stability_state;    // Current stability analysis
    
    // Metal-specific objects
#ifdef __OBJC__
    id<MTLDevice> device;                           // Metal device
    id<MTLCommandQueue> command_queue;              // Command queue
    id<MTLLibrary> shader_library;                  // Compute shader library
    
    // Compute pipeline states
    id<MTLComputePipelineState> float32_attention_pipeline;
    id<MTLComputePipelineState> float32_softmax_pipeline;
    id<MTLComputePipelineState> float32_layer_norm_pipeline;
    id<MTLComputePipelineState> float16_matrix_multiply_pipeline;
    id<MTLComputePipelineState> float16_ffn_pipeline;
    id<MTLComputePipelineState> float16_gelu_pipeline;
    id<MTLComputePipelineState> float16_swiglu_pipeline;
    id<MTLComputePipelineState> data_conversion_pipeline;
#endif
    
    // Internal state
    bool initialized;                               // Initialization state
    uint32_t stability_check_counter;              // Stability check counter
    uint32_t consecutive_instabilities;            // Consecutive instability count
    float last_gradient_norm;                      // Last gradient norm
    float last_loss_value;                         // Last loss value
    
    // Memory pools for conversion
    void* float32_buffer;                          // Float32 conversion buffer
    void* float16_buffer;                          // Float16 conversion buffer
    size_t buffer_size;                            // Buffer size
};

// Metal shader source code
#ifdef __OBJC__
static NSString* const kMixedPrecisionShaders = @R"(
#include <metal_stdlib>
using namespace metal;

// Float32 critical operations

kernel void float32_attention_computation(
    constant float* query [[buffer(0)]],
    constant float* key [[buffer(1)]],
    constant float* value [[buffer(2)]],
    device float* attention_output [[buffer(3)]],
    device float* attention_weights [[buffer(4)]],
    constant uint& sequence_length [[buffer(5)]],
    constant uint& d_model [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& head_dimension [[buffer(8)]],
    constant float& attention_scale [[buffer(9)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    uint head = threadgroup_position.x;
    uint seq_pos = threadgroup_position.y;
    uint dim = thread_position_in_threadgroup.x;
    
    if (head >= num_heads || seq_pos >= sequence_length || dim >= head_dimension) return;
    
    uint head_offset = head * head_dimension;
    uint seq_offset = seq_pos * d_model;
    
    // Compute attention scores (float32 for numerical stability)
    float attention_score = 0.0f;
    for (uint k = 0; k < head_dimension; k++) {
        float q_val = query[seq_offset + head_offset + k];
        float k_val = key[seq_offset + head_offset + k];
        attention_score += q_val * k_val;
    }
    attention_score *= attention_scale;
    
    // Store attention weights for softmax
    attention_weights[head * sequence_length * sequence_length + seq_pos * sequence_length + seq_pos] = attention_score;
    
    // Apply value transformation (will be done after softmax)
}

kernel void float32_softmax_computation(
    device float* input_vectors [[buffer(0)]],
    device float* output_vectors [[buffer(1)]],
    constant uint& vector_length [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    uint threadgroup_position [[threadgroup_position_in_grid]],
    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    uint batch_idx = threadgroup_position;
    uint elem_idx = thread_position_in_threadgroup;
    
    if (batch_idx >= batch_size || elem_idx >= vector_length) return;
    
    device float* input_vec = input_vectors + batch_idx * vector_length;
    device float* output_vec = output_vectors + batch_idx * vector_length;
    
    // Find maximum for numerical stability (float32)
    threadgroup float shared_max;
    if (thread_position_in_threadgroup == 0) {
        float max_val = input_vec[0];
        for (uint i = 1; i < vector_length; i++) {
            max_val = max(max_val, input_vec[i]);
        }
        shared_max = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute exp and sum (float32 for stability)
    threadgroup float shared_sum;
    if (thread_position_in_threadgroup == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < vector_length; i++) {
            sum += exp(input_vec[i] - shared_max);
        }
        shared_sum = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute final softmax values
    if (elem_idx < vector_length) {
        output_vec[elem_idx] = exp(input_vec[elem_idx] - shared_max) / shared_sum;
    }
}

kernel void float32_layer_norm_computation(
    constant float* input_vectors [[buffer(0)]],
    constant float* gamma_weights [[buffer(1)]],
    constant float* beta_weights [[buffer(2)]],
    device float* output_vectors [[buffer(3)]],
    constant uint& vector_length [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint threadgroup_position [[threadgroup_position_in_grid]],
    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    uint batch_idx = threadgroup_position;
    uint elem_idx = thread_position_in_threadgroup;
    
    if (batch_idx >= batch_size || elem_idx >= vector_length) return;
    
    constant float* input_vec = input_vectors + batch_idx * vector_length;
    device float* output_vec = output_vectors + batch_idx * vector_length;
    
    // Compute mean and variance (float32 for numerical precision)
    threadgroup float shared_mean, shared_variance;
    if (thread_position_in_threadgroup == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < vector_length; i++) {
            sum += input_vec[i];
        }
        shared_mean = sum / vector_length;
        
        float var_sum = 0.0f;
        for (uint i = 0; i < vector_length; i++) {
            float diff = input_vec[i] - shared_mean;
            var_sum += diff * diff;
        }
        shared_variance = var_sum / vector_length;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Apply normalization
    if (elem_idx < vector_length) {
        float normalized = (input_vec[elem_idx] - shared_mean) / sqrt(shared_variance + epsilon);
        output_vec[elem_idx] = gamma_weights[elem_idx] * normalized + beta_weights[elem_idx];
    }
}

// Float16 accelerated operations

kernel void float16_matrix_multiply(
    constant half* matrix_a [[buffer(0)]],
    constant half* matrix_b [[buffer(1)]],
    device half* result_matrix [[buffer(2)]],
    constant uint& rows_a [[buffer(3)]],
    constant uint& cols_a [[buffer(4)]],
    constant uint& cols_b [[buffer(5)]],
    uint2 threadgroup_position [[threadgroup_position_in_grid]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    uint row = threadgroup_position.y * 32 + thread_position_in_threadgroup.y;
    uint col = threadgroup_position.x * 32 + thread_position_in_threadgroup.x;
    
    if (row >= rows_a || col >= cols_b) return;
    
    // Use half precision for intermediate calculations (faster on Apple Silicon)
    half sum = 0.0h;
    for (uint k = 0; k < cols_a; k++) {
        sum += matrix_a[row * cols_a + k] * matrix_b[k * cols_b + col];
    }
    result_matrix[row * cols_b + col] = sum;
}

kernel void float16_ffn_computation(
    constant half* input_matrix [[buffer(0)]],
    constant half* weight_matrix [[buffer(1)]],
    constant half* bias_vector [[buffer(2)]],
    device half* output_matrix [[buffer(3)]],
    constant uint& input_dim [[buffer(4)]],
    constant uint& hidden_dim [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint2 threadgroup_position [[threadgroup_position_in_grid]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    uint batch_idx = threadgroup_position.y;
    uint hidden_idx = threadgroup_position.x * 32 + thread_position_in_threadgroup.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_dim) return;
    
    // FFN computation with float16 acceleration
    half sum = bias_vector[hidden_idx];
    for (uint i = 0; i < input_dim; i++) {
        sum += input_matrix[batch_idx * input_dim + i] * weight_matrix[i * hidden_dim + hidden_idx];
    }
    output_matrix[batch_idx * hidden_dim + hidden_idx] = sum;
}

kernel void float16_gelu_computation(
    constant half* input_vectors [[buffer(0)]],
    device half* output_vectors [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    uint thread_position [[thread_position_in_grid]]
) {
    uint idx = thread_position;
    if (idx >= total_elements) return;
    
    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    half x = input_vectors[idx];
    half x_cubed = x * x * x;
    half inner = sqrt(2.0h / M_PI_H) * (x + 0.044715h * x_cubed);
    half gelu_val = x * 0.5h * (1.0h + tanh(inner));
    output_vectors[idx] = gelu_val;
}

kernel void float16_swiglu_computation(
    constant half* input_vectors [[buffer(0)]],
    constant half* gate_vectors [[buffer(1)]],
    device half* output_vectors [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint thread_position [[thread_position_in_grid]]
) {
    uint idx = thread_position;
    if (idx >= total_elements) return;
    
    // SwiGLU: x * silu(gate) where silu(x) = x * sigmoid(x)
    half x = input_vectors[idx];
    half gate = gate_vectors[idx];
    half sigmoid_gate = 1.0h / (1.0h + exp(-gate));
    half silu_gate = gate * sigmoid_gate;
    output_vectors[idx] = x * silu_gate;
}

// Data conversion kernels

kernel void convert_float32_to_float16(
    constant float* input_data [[buffer(0)]],
    device half* output_data [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint thread_position [[thread_position_in_grid]]
) {
    uint idx = thread_position;
    if (idx >= element_count) return;
    output_data[idx] = half(input_data[idx]);
}

kernel void convert_float16_to_float32(
    constant half* input_data [[buffer(0)]],
    device float* output_data [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint thread_position [[thread_position_in_grid]]
) {
    uint idx = thread_position;
    if (idx >= element_count) return;
    output_data[idx] = float(input_data[idx]);
}
)";
#endif

// Utility functions
static uint64_t get_current_time_microseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

static float float16_to_float32(uint16_t half_value) {
    uint32_t sign = (half_value >> 15) & 0x1;
    uint32_t exp = (half_value >> 10) & 0x1f;
    uint32_t mant = half_value & 0x3ff;
    
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            // Denormal number
            float value = (float)mant / 1024.0f / 16384.0f;
            return sign ? -value : value;
        }
    } else if (exp == 31) {
        if (mant == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    } else {
        // Normal number
        uint32_t float32_bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
        return *(float*)&float32_bits;
    }
}

static uint16_t float32_to_float16(float value) {
    if (isnan(value)) {
        return 0x7fff;
    }
    if (isinf(value)) {
        return value < 0 ? 0xfc00 : 0x7c00;
    }
    
    uint32_t bits = *(uint32_t*)&value;
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp = (bits >> 23) & 0xff;
    uint32_t mant = bits & 0x7fffff;
    
    if (exp == 0) {
        return (uint16_t)(sign << 15);
    }
    
    int32_t new_exp = (int32_t)exp - 127 + 15;
    if (new_exp <= 0) {
        return (uint16_t)(sign << 15);
    }
    if (new_exp >= 31) {
        return (uint16_t)((sign << 15) | 0x7c00);
    }
    
    return (uint16_t)((sign << 15) | (new_exp << 10) | (mant >> 13));
}

// Core API Implementation

MixedPrecisionError mixed_precision_create(MixedPrecisionEngine** engine,
                                          const MixedPrecisionConfig* config,
                                          MetalComputeAccelerator* metal_accelerator) {
    if (!engine || !metal_accelerator) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    *engine = (MixedPrecisionEngine*)calloc(1, sizeof(MixedPrecisionEngine));
    if (!*engine) {
        return MIXED_PRECISION_ERROR_MEMORY_ALLOCATION;
    }
    
    MixedPrecisionEngine* eng = *engine;
    
    // Use provided config or create default
    if (config) {
        eng->config = *config;
    } else {
        mixed_precision_create_default_config(&eng->config);
    }
    
    eng->metal_accelerator = metal_accelerator;
    eng->initialized = false;
    eng->stability_check_counter = 0;
    eng->consecutive_instabilities = 0;
    eng->last_gradient_norm = 0.0f;
    eng->last_loss_value = 0.0f;
    
    // Initialize statistics
    memset(&eng->stats, 0, sizeof(MixedPrecisionStats));
    eng->stats.numerical_stability_score = 1.0f;
    
    // Initialize stability state
    eng->stability_state.status = STABILITY_STATUS_STABLE;
    eng->stability_state.stability_score = 1.0f;
    eng->stability_state.recommended_precision = PRECISION_TYPE_MIXED_ADAPTIVE;
    
    // Allocate conversion buffers (1MB each)
    eng->buffer_size = 1024 * 1024;
    eng->float32_buffer = malloc(eng->buffer_size);
    eng->float16_buffer = malloc(eng->buffer_size / 2); // Half the size for float16
    
    if (!eng->float32_buffer || !eng->float16_buffer) {
        free(eng->float32_buffer);
        free(eng->float16_buffer);
        free(eng);
        *engine = NULL;
        return MIXED_PRECISION_ERROR_MEMORY_ALLOCATION;
    }
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_initialize(MixedPrecisionEngine* engine) {
    if (!engine || !engine->metal_accelerator) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
#ifdef __OBJC__
    // Get Metal device capabilities
    MetalDeviceCapabilities device_caps;
    MetalComputeError metal_error = metal_compute_get_device_capabilities(engine->metal_accelerator, &device_caps);
    if (metal_error != METAL_COMPUTE_SUCCESS) {
        return MIXED_PRECISION_ERROR_METAL_DEVICE_ERROR;
    }
    
    // Validate float16 support
    if (!device_caps.supports_float16) {
        return MIXED_PRECISION_ERROR_UNSUPPORTED_PRECISION;
    }
    
    // Get Metal device from accelerator
    engine->device = (__bridge id<MTLDevice>)metal_compute_get_device_name(engine->metal_accelerator);
    if (!engine->device) {
        return MIXED_PRECISION_ERROR_DEVICE_NOT_FOUND;
    }
    
    // Create command queue
    engine->command_queue = [engine->device newCommandQueue];
    if (!engine->command_queue) {
        return MIXED_PRECISION_ERROR_DEVICE_NOT_FOUND;
    }
    
    // Compile shader library
    NSError* error = nil;
    engine->shader_library = [engine->device newLibraryWithSource:kMixedPrecisionShaders
                                                          options:nil
                                                            error:&error];
    if (!engine->shader_library || error) {
        return MIXED_PRECISION_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    // Create compute pipeline states
    id<MTLFunction> float32_attention_func = [engine->shader_library newFunctionWithName:@"float32_attention_computation"];
    engine->float32_attention_pipeline = [engine->device newComputePipelineStateWithFunction:float32_attention_func error:&error];
    if (!engine->float32_attention_pipeline || error) {
        return MIXED_PRECISION_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    id<MTLFunction> float32_softmax_func = [engine->shader_library newFunctionWithName:@"float32_softmax_computation"];
    engine->float32_softmax_pipeline = [engine->device newComputePipelineStateWithFunction:float32_softmax_func error:&error];
    if (!engine->float32_softmax_pipeline || error) {
        return MIXED_PRECISION_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    id<MTLFunction> float32_layer_norm_func = [engine->shader_library newFunctionWithName:@"float32_layer_norm_computation"];
    engine->float32_layer_norm_pipeline = [engine->device newComputePipelineStateWithFunction:float32_layer_norm_func error:&error];
    if (!engine->float32_layer_norm_pipeline || error) {
        return MIXED_PRECISION_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    id<MTLFunction> float16_matmul_func = [engine->shader_library newFunctionWithName:@"float16_matrix_multiply"];
    engine->float16_matrix_multiply_pipeline = [engine->device newComputePipelineStateWithFunction:float16_matmul_func error:&error];
    if (!engine->float16_matrix_multiply_pipeline || error) {
        return MIXED_PRECISION_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    id<MTLFunction> float16_ffn_func = [engine->shader_library newFunctionWithName:@"float16_ffn_computation"];
    engine->float16_ffn_pipeline = [engine->device newComputePipelineStateWithFunction:float16_ffn_func error:&error];
    if (!engine->float16_ffn_pipeline || error) {
        return MIXED_PRECISION_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    id<MTLFunction> float16_gelu_func = [engine->shader_library newFunctionWithName:@"float16_gelu_computation"];
    engine->float16_gelu_pipeline = [engine->device newComputePipelineStateWithFunction:float16_gelu_func error:&error];
    if (!engine->float16_gelu_pipeline || error) {
        return MIXED_PRECISION_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    id<MTLFunction> float16_swiglu_func = [engine->shader_library newFunctionWithName:@"float16_swiglu_computation"];
    engine->float16_swiglu_pipeline = [engine->device newComputePipelineStateWithFunction:float16_swiglu_func error:&error];
    if (!engine->float16_swiglu_pipeline || error) {
        return MIXED_PRECISION_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    id<MTLFunction> data_conv_func = [engine->shader_library newFunctionWithName:@"convert_float32_to_float16"];
    engine->data_conversion_pipeline = [engine->device newComputePipelineStateWithFunction:data_conv_func error:&error];
    if (!engine->data_conversion_pipeline || error) {
        return MIXED_PRECISION_ERROR_SHADER_COMPILATION_FAILED;
    }
#endif
    
    engine->initialized = true;
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_select_optimal_precision(MixedPrecisionEngine* engine,
                                                            const MixedPrecisionOpConfig* op_config,
                                                            const float* input_data,
                                                            size_t input_size,
                                                            PrecisionType* selected_precision) {
    if (!engine || !op_config || !input_data || !selected_precision || !engine->initialized) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    // Check forced precision
    if (op_config->forced_precision != PRECISION_TYPE_MIXED_ADAPTIVE) {
        *selected_precision = op_config->forced_precision;
        return MIXED_PRECISION_SUCCESS;
    }
    
    // Determine precision based on criticality level
    switch (op_config->criticality) {
        case OPERATION_CRITICAL_LEVEL_CRITICAL:
            // Must use float32 (attention, gradients)
            *selected_precision = PRECISION_TYPE_FLOAT32;
            break;
            
        case OPERATION_CRITICAL_LEVEL_IMPORTANT:
            // Should use float32 (layer norm, softmax)
            *selected_precision = PRECISION_TYPE_FLOAT32;
            break;
            
        case OPERATION_CRITICAL_LEVEL_STANDARD:
            // Can use float16 (FFN intermediate)
            *selected_precision = PRECISION_TYPE_FLOAT16;
            break;
            
        case OPERATION_CRITICAL_LEVEL_ACCELERATED:
            // Prioritize speed with float16
            *selected_precision = PRECISION_TYPE_FLOAT16;
            break;
            
        default:
            *selected_precision = PRECISION_TYPE_MIXED_ADAPTIVE;
            break;
    }
    
    // Check numerical stability if enabled
    if (engine->config.enable_stability_monitoring) {
        // Analyze input data for potential numerical issues
        uint32_t overflow_count = 0, underflow_count = 0, nan_count = 0, inf_count = 0;
        
        MixedPrecisionError check_error = mixed_precision_check_numerical_issues(engine,
                                                                               input_data, input_size,
                                                                               &overflow_count, &underflow_count,
                                                                               &nan_count, &inf_count);
        
        if (check_error == MIXED_PRECISION_SUCCESS) {
            float instability_ratio = (float)(overflow_count + underflow_count + nan_count + inf_count) / input_size;
            
            // Force float32 if instability detected
            if (instability_ratio > engine->config.stability_threshold) {
                *selected_precision = PRECISION_TYPE_FLOAT32;
                engine->consecutive_instabilities++;
            } else {
                engine->consecutive_instabilities = 0;
            }
        }
    }
    
    // Override if too many consecutive instabilities
    if (engine->consecutive_instabilities >= engine->config.fallback_threshold) {
        *selected_precision = PRECISION_TYPE_FLOAT32;
    }
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_analyze_stability(MixedPrecisionEngine* engine,
                                                     const float* data,
                                                     size_t data_size,
                                                     NumericalStabilityAnalysis* stability_analysis) {
    if (!engine || !data || !stability_analysis || data_size == 0) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    // Initialize analysis result
    memset(stability_analysis, 0, sizeof(NumericalStabilityAnalysis));
    stability_analysis->status = STABILITY_STATUS_STABLE;
    stability_analysis->stability_score = 1.0f;
    stability_analysis->recommended_precision = PRECISION_TYPE_MIXED_ADAPTIVE;
    
    // Analyze data for numerical issues
    uint32_t overflow_count = 0, underflow_count = 0, nan_count = 0, inf_count = 0;
    float sum = 0.0f, sum_squares = 0.0f;
    float min_val = FLT_MAX, max_val = -FLT_MAX;
    
    for (size_t i = 0; i < data_size; i++) {
        float val = data[i];
        
        if (isnan(val)) {
            nan_count++;
        } else if (isinf(val)) {
            inf_count++;
        } else if (fabs(val) > STABILITY_OVERFLOW_THRESHOLD) {
            overflow_count++;
        } else if (fabs(val) < STABILITY_UNDERFLOW_THRESHOLD && val != 0.0f) {
            underflow_count++;
        } else {
            // Normal value
            sum += val;
            sum_squares += val * val;
            min_val = fminf(min_val, val);
            max_val = fmaxf(max_val, val);
        }
    }
    
    size_t normal_count = data_size - overflow_count - underflow_count - nan_count - inf_count;
    
    // Calculate statistics
    if (normal_count > 0) {
        float mean = sum / normal_count;
        float variance = (sum_squares / normal_count) - (mean * mean);
        stability_analysis->parameter_norm = sqrtf(sum_squares / normal_count);
        stability_analysis->gradient_norm = stability_analysis->parameter_norm; // Approximation
    }
    
    // Store counts
    stability_analysis->overflow_count = overflow_count;
    stability_analysis->underflow_count = underflow_count;
    stability_analysis->nan_count = nan_count;
    stability_analysis->inf_count = inf_count;
    
    // Calculate stability score (0.0 = unstable, 1.0 = stable)
    float instability_ratio = (float)(overflow_count + underflow_count + nan_count + inf_count) / data_size;
    stability_analysis->stability_score = fmaxf(0.0f, 1.0f - instability_ratio * 10.0f);
    
    // Determine stability status
    if (instability_ratio == 0.0f) {
        stability_analysis->status = STABILITY_STATUS_STABLE;
    } else if (instability_ratio < 0.01f) {
        stability_analysis->status = STABILITY_STATUS_WARNING;
    } else if (instability_ratio < 0.1f) {
        stability_analysis->status = STABILITY_STATUS_UNSTABLE;
        stability_analysis->requires_precision_fallback = true;
    } else {
        stability_analysis->status = STABILITY_STATUS_CRITICAL;
        stability_analysis->requires_precision_fallback = true;
    }
    
    // Recommend precision type
    if (stability_analysis->requires_precision_fallback) {
        stability_analysis->recommended_precision = PRECISION_TYPE_FLOAT32;
    } else if (stability_analysis->status == STABILITY_STATUS_STABLE) {
        stability_analysis->recommended_precision = PRECISION_TYPE_MIXED_ADAPTIVE;
    } else {
        stability_analysis->recommended_precision = PRECISION_TYPE_FLOAT32;
    }
    
    // Update engine state
    engine->stability_state = *stability_analysis;
    
    return MIXED_PRECISION_SUCCESS;
}

// Critical Operations (Float32 Required)

MixedPrecisionError mixed_precision_compute_attention_float32(MixedPrecisionEngine* engine,
                                                             const AttentionPrecisionConfig* attention_config,
                                                             const float* query_matrix,
                                                             const float* key_matrix,
                                                             const float* value_matrix,
                                                             float* attention_output,
                                                             float* attention_weights,
                                                             MixedPrecisionOpResult* op_result) {
    if (!engine || !attention_config || !query_matrix || !key_matrix || !value_matrix || 
        !attention_output || !op_result || !engine->initialized) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_current_time_microseconds();
    
    // Initialize operation result
    memset(op_result, 0, sizeof(MixedPrecisionOpResult));
    op_result->used_precision = PRECISION_TYPE_FLOAT32;
    op_result->stability_status = STABILITY_STATUS_STABLE;
    
#ifdef __OBJC__
    @autoreleasepool {
        // Create Metal buffers for attention computation
        size_t seq_len = attention_config->sequence_length;
        size_t d_model = attention_config->d_model;
        size_t num_heads = attention_config->num_heads;
        size_t head_dim = attention_config->head_dimension;
        
        size_t matrix_size = seq_len * d_model * sizeof(float);
        size_t weights_size = num_heads * seq_len * seq_len * sizeof(float);
        
        id<MTLBuffer> query_buffer = [engine->device newBufferWithBytes:query_matrix
                                                                length:matrix_size
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> key_buffer = [engine->device newBufferWithBytes:key_matrix
                                                              length:matrix_size
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> value_buffer = [engine->device newBufferWithBytes:value_matrix
                                                                length:matrix_size
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [engine->device newBufferWithLength:matrix_size
                                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> weights_buffer = nil;
        
        if (attention_weights) {
            weights_buffer = [engine->device newBufferWithLength:weights_size
                                                         options:MTLResourceStorageModeShared];
        }
        
        if (!query_buffer || !key_buffer || !value_buffer || !output_buffer) {
            return MIXED_PRECISION_ERROR_BUFFER_ALLOCATION_FAILED;
        }
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> command_buffer = [engine->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:engine->float32_attention_pipeline];
        [encoder setBuffer:query_buffer offset:0 atIndex:0];
        [encoder setBuffer:key_buffer offset:0 atIndex:1];
        [encoder setBuffer:value_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        if (weights_buffer) {
            [encoder setBuffer:weights_buffer offset:0 atIndex:4];
        }
        
        // Set parameters
        uint32_t seq_len_param = (uint32_t)seq_len;
        uint32_t d_model_param = (uint32_t)d_model;
        uint32_t num_heads_param = (uint32_t)num_heads;
        uint32_t head_dim_param = (uint32_t)head_dim;
        float attention_scale = attention_config->attention_scale;
        
        [encoder setBytes:&seq_len_param length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&d_model_param length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&num_heads_param length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&head_dim_param length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&attention_scale length:sizeof(float) atIndex:9];
        
        // Configure thread groups
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize gridSize = MTLSizeMake((num_heads + 15) / 16, (seq_len + 15) / 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        // Execute computation
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        // Copy results back
        memcpy(attention_output, output_buffer.contents, matrix_size);
        if (attention_weights && weights_buffer) {
            memcpy(attention_weights, weights_buffer.contents, weights_size);
        }
        
        // Check for Metal errors
        if (command_buffer.error) {
            return MIXED_PRECISION_ERROR_EXECUTION_FAILED;
        }
    }
#else
    // Fallback CPU implementation for attention
    size_t seq_len = attention_config->sequence_length;
    size_t d_model = attention_config->d_model;
    size_t num_heads = attention_config->num_heads;
    size_t head_dim = attention_config->head_dimension;
    float scale = attention_config->attention_scale;
    
    // Simple CPU attention computation (placeholder)
    for (size_t h = 0; h < num_heads; h++) {
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                size_t idx = i * d_model + h * head_dim + j;
                attention_output[idx] = query_matrix[idx] * scale + 
                                       key_matrix[idx] * scale + 
                                       value_matrix[idx];
            }
        }
    }
#endif
    
    uint64_t end_time = get_current_time_microseconds();
    op_result->operation_time_us = end_time - start_time;
    op_result->precision_speedup = 1.0f; // Float32 baseline
    
    // Update statistics
    engine->stats.total_operations++;
    engine->stats.float32_operations++;
    engine->stats.total_compute_time_us += op_result->operation_time_us;
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_compute_softmax_float32(MixedPrecisionEngine* engine,
                                                           const float* input_vectors,
                                                           float* output_vectors,
                                                           uint32_t vector_length,
                                                           uint32_t batch_size,
                                                           MixedPrecisionOpResult* op_result) {
    if (!engine || !input_vectors || !output_vectors || !op_result || !engine->initialized ||
        vector_length == 0 || batch_size == 0) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_current_time_microseconds();
    
    // Initialize operation result
    memset(op_result, 0, sizeof(MixedPrecisionOpResult));
    op_result->used_precision = PRECISION_TYPE_FLOAT32;
    op_result->stability_status = STABILITY_STATUS_STABLE;
    
#ifdef __OBJC__
    @autoreleasepool {
        size_t buffer_size = vector_length * batch_size * sizeof(float);
        
        id<MTLBuffer> input_buffer = [engine->device newBufferWithBytes:input_vectors
                                                                length:buffer_size
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [engine->device newBufferWithLength:buffer_size
                                                                   options:MTLResourceStorageModeShared];
        
        if (!input_buffer || !output_buffer) {
            return MIXED_PRECISION_ERROR_BUFFER_ALLOCATION_FAILED;
        }
        
        id<MTLCommandBuffer> command_buffer = [engine->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:engine->float32_softmax_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBytes:&vector_length length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:3];
        
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        memcpy(output_vectors, output_buffer.contents, buffer_size);
        
        if (command_buffer.error) {
            return MIXED_PRECISION_ERROR_EXECUTION_FAILED;
        }
    }
#else
    // CPU fallback softmax
    for (uint32_t b = 0; b < batch_size; b++) {
        const float* input_vec = input_vectors + b * vector_length;
        float* output_vec = output_vectors + b * vector_length;
        
        // Find max for numerical stability
        float max_val = input_vec[0];
        for (uint32_t i = 1; i < vector_length; i++) {
            if (input_vec[i] > max_val) max_val = input_vec[i];
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (uint32_t i = 0; i < vector_length; i++) {
            output_vec[i] = expf(input_vec[i] - max_val);
            sum += output_vec[i];
        }
        
        // Normalize
        for (uint32_t i = 0; i < vector_length; i++) {
            output_vec[i] /= sum;
        }
    }
#endif
    
    uint64_t end_time = get_current_time_microseconds();
    op_result->operation_time_us = end_time - start_time;
    op_result->precision_speedup = 1.0f;
    
    // Update statistics
    engine->stats.total_operations++;
    engine->stats.float32_operations++;
    engine->stats.total_compute_time_us += op_result->operation_time_us;
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_compute_layer_norm_float32(MixedPrecisionEngine* engine,
                                                              const float* input_vectors,
                                                              const float* gamma_weights,
                                                              const float* beta_weights,
                                                              float* output_vectors,
                                                              uint32_t vector_length,
                                                              uint32_t batch_size,
                                                              float epsilon,
                                                              MixedPrecisionOpResult* op_result) {
    if (!engine || !input_vectors || !gamma_weights || !beta_weights || !output_vectors ||
        !op_result || !engine->initialized || vector_length == 0 || batch_size == 0) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_current_time_microseconds();
    
    // Initialize operation result
    memset(op_result, 0, sizeof(MixedPrecisionOpResult));
    op_result->used_precision = PRECISION_TYPE_FLOAT32;
    op_result->stability_status = STABILITY_STATUS_STABLE;
    
#ifdef __OBJC__
    @autoreleasepool {
        size_t vector_buffer_size = vector_length * batch_size * sizeof(float);
        size_t weights_buffer_size = vector_length * sizeof(float);
        
        id<MTLBuffer> input_buffer = [engine->device newBufferWithBytes:input_vectors
                                                                length:vector_buffer_size
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> gamma_buffer = [engine->device newBufferWithBytes:gamma_weights
                                                                length:weights_buffer_size
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> beta_buffer = [engine->device newBufferWithBytes:beta_weights
                                                               length:weights_buffer_size
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [engine->device newBufferWithLength:vector_buffer_size
                                                                   options:MTLResourceStorageModeShared];
        
        if (!input_buffer || !gamma_buffer || !beta_buffer || !output_buffer) {
            return MIXED_PRECISION_ERROR_BUFFER_ALLOCATION_FAILED;
        }
        
        id<MTLCommandBuffer> command_buffer = [engine->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:engine->float32_layer_norm_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:gamma_buffer offset:0 atIndex:1];
        [encoder setBuffer:beta_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setBytes:&vector_length length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&epsilon length:sizeof(float) atIndex:6];
        
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        memcpy(output_vectors, output_buffer.contents, vector_buffer_size);
        
        if (command_buffer.error) {
            return MIXED_PRECISION_ERROR_EXECUTION_FAILED;
        }
    }
#else
    // CPU fallback layer normalization
    for (uint32_t b = 0; b < batch_size; b++) {
        const float* input_vec = input_vectors + b * vector_length;
        float* output_vec = output_vectors + b * vector_length;
        
        // Compute mean
        float sum = 0.0f;
        for (uint32_t i = 0; i < vector_length; i++) {
            sum += input_vec[i];
        }
        float mean = sum / vector_length;
        
        // Compute variance
        float var_sum = 0.0f;
        for (uint32_t i = 0; i < vector_length; i++) {
            float diff = input_vec[i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / vector_length;
        float std_dev = sqrtf(variance + epsilon);
        
        // Normalize and scale
        for (uint32_t i = 0; i < vector_length; i++) {
            float normalized = (input_vec[i] - mean) / std_dev;
            output_vec[i] = gamma_weights[i] * normalized + beta_weights[i];
        }
    }
#endif
    
    uint64_t end_time = get_current_time_microseconds();
    op_result->operation_time_us = end_time - start_time;
    op_result->precision_speedup = 1.0f;
    
    // Update statistics
    engine->stats.total_operations++;
    engine->stats.float32_operations++;
    engine->stats.total_compute_time_us += op_result->operation_time_us;
    
    return MIXED_PRECISION_SUCCESS;
}

// Accelerated Operations (Float16 Optimized)

MixedPrecisionError mixed_precision_compute_ffn_float16(MixedPrecisionEngine* engine,
                                                       const FFNPrecisionConfig* ffn_config,
                                                       const float* input_matrix,
                                                       const float* weight_matrix,
                                                       const float* bias_vector,
                                                       float* output_matrix,
                                                       MixedPrecisionOpResult* op_result) {
    if (!engine || !ffn_config || !input_matrix || !weight_matrix || !bias_vector ||
        !output_matrix || !op_result || !engine->initialized) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_current_time_microseconds();
    
    // Initialize operation result
    memset(op_result, 0, sizeof(MixedPrecisionOpResult));
    op_result->used_precision = PRECISION_TYPE_FLOAT16;
    op_result->stability_status = STABILITY_STATUS_STABLE;
    
    size_t input_size = ffn_config->batch_size * ffn_config->input_dimension;
    size_t weight_size = ffn_config->input_dimension * ffn_config->hidden_dimension;
    size_t bias_size = ffn_config->hidden_dimension;
    size_t output_size = ffn_config->batch_size * ffn_config->hidden_dimension;
    
    // Allocate float16 conversion buffers
    uint16_t* input_float16 = (uint16_t*)malloc(input_size * sizeof(uint16_t));
    uint16_t* weight_float16 = (uint16_t*)malloc(weight_size * sizeof(uint16_t));
    uint16_t* bias_float16 = (uint16_t*)malloc(bias_size * sizeof(uint16_t));
    uint16_t* output_float16 = (uint16_t*)malloc(output_size * sizeof(uint16_t));
    
    if (!input_float16 || !weight_float16 || !bias_float16 || !output_float16) {
        free(input_float16);
        free(weight_float16);
        free(bias_float16);
        free(output_float16);
        return MIXED_PRECISION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Convert inputs to float16
    for (size_t i = 0; i < input_size; i++) {
        input_float16[i] = float32_to_float16(input_matrix[i]);
    }
    for (size_t i = 0; i < weight_size; i++) {
        weight_float16[i] = float32_to_float16(weight_matrix[i]);
    }
    for (size_t i = 0; i < bias_size; i++) {
        bias_float16[i] = float32_to_float16(bias_vector[i]);
    }
    
#ifdef __OBJC__
    @autoreleasepool {
        id<MTLBuffer> input_buffer = [engine->device newBufferWithBytes:input_float16
                                                                length:input_size * sizeof(uint16_t)
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> weight_buffer = [engine->device newBufferWithBytes:weight_float16
                                                                 length:weight_size * sizeof(uint16_t)
                                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bias_buffer = [engine->device newBufferWithBytes:bias_float16
                                                               length:bias_size * sizeof(uint16_t)
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [engine->device newBufferWithLength:output_size * sizeof(uint16_t)
                                                                   options:MTLResourceStorageModeShared];
        
        if (!input_buffer || !weight_buffer || !bias_buffer || !output_buffer) {
            free(input_float16);
            free(weight_float16);
            free(bias_float16);
            free(output_float16);
            return MIXED_PRECISION_ERROR_BUFFER_ALLOCATION_FAILED;
        }
        
        id<MTLCommandBuffer> command_buffer = [engine->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:engine->float16_ffn_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:bias_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        
        uint32_t input_dim = ffn_config->input_dimension;
        uint32_t hidden_dim = ffn_config->hidden_dimension;
        uint32_t batch_size = ffn_config->batch_size;
        
        [encoder setBytes:&input_dim length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&hidden_dim length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&batch_size length:sizeof(uint32_t) atIndex:6];
        
        MTLSize threadgroupSize = MTLSizeMake(32, 8, 1);
        MTLSize gridSize = MTLSizeMake((hidden_dim + 31) / 32, batch_size, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        memcpy(output_float16, output_buffer.contents, output_size * sizeof(uint16_t));
        
        if (command_buffer.error) {
            free(input_float16);
            free(weight_float16);
            free(bias_float16);
            free(output_float16);
            return MIXED_PRECISION_ERROR_EXECUTION_FAILED;
        }
    }
#else
    // CPU fallback FFN computation with float16
    for (uint32_t b = 0; b < ffn_config->batch_size; b++) {
        for (uint32_t h = 0; h < ffn_config->hidden_dimension; h++) {
            float sum = float16_to_float32(bias_float16[h]);
            for (uint32_t i = 0; i < ffn_config->input_dimension; i++) {
                float input_val = float16_to_float32(input_float16[b * ffn_config->input_dimension + i]);
                float weight_val = float16_to_float32(weight_float16[i * ffn_config->hidden_dimension + h]);
                sum += input_val * weight_val;
            }
            output_float16[b * ffn_config->hidden_dimension + h] = float32_to_float16(sum);
        }
    }
#endif
    
    // Convert output back to float32
    for (size_t i = 0; i < output_size; i++) {
        output_matrix[i] = float16_to_float32(output_float16[i]);
    }
    
    // Cleanup
    free(input_float16);
    free(weight_float16);
    free(bias_float16);
    free(output_float16);
    
    uint64_t end_time = get_current_time_microseconds();
    op_result->operation_time_us = end_time - start_time;
    op_result->precision_speedup = 1.5f; // Estimated speedup for float16
    op_result->memory_saved_bytes = (input_size + weight_size + output_size) * sizeof(float) / 2;
    
    // Update statistics
    engine->stats.total_operations++;
    engine->stats.float16_operations++;
    engine->stats.total_compute_time_us += op_result->operation_time_us;
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_compute_matrix_multiply_float16(MixedPrecisionEngine* engine,
                                                                   const float* matrix_a,
                                                                   const float* matrix_b,
                                                                   float* result_matrix,
                                                                   uint32_t rows_a,
                                                                   uint32_t cols_a,
                                                                   uint32_t cols_b,
                                                                   MixedPrecisionOpResult* op_result) {
    if (!engine || !matrix_a || !matrix_b || !result_matrix || !op_result || 
        !engine->initialized || rows_a == 0 || cols_a == 0 || cols_b == 0) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_current_time_microseconds();
    
    // Initialize operation result
    memset(op_result, 0, sizeof(MixedPrecisionOpResult));
    op_result->used_precision = PRECISION_TYPE_FLOAT16;
    op_result->stability_status = STABILITY_STATUS_STABLE;
    
    size_t size_a = rows_a * cols_a;
    size_t size_b = cols_a * cols_b;
    size_t size_result = rows_a * cols_b;
    
    // Allocate float16 buffers
    uint16_t* matrix_a_float16 = (uint16_t*)malloc(size_a * sizeof(uint16_t));
    uint16_t* matrix_b_float16 = (uint16_t*)malloc(size_b * sizeof(uint16_t));
    uint16_t* result_float16 = (uint16_t*)malloc(size_result * sizeof(uint16_t));
    
    if (!matrix_a_float16 || !matrix_b_float16 || !result_float16) {
        free(matrix_a_float16);
        free(matrix_b_float16);
        free(result_float16);
        return MIXED_PRECISION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Convert inputs to float16
    for (size_t i = 0; i < size_a; i++) {
        matrix_a_float16[i] = float32_to_float16(matrix_a[i]);
    }
    for (size_t i = 0; i < size_b; i++) {
        matrix_b_float16[i] = float32_to_float16(matrix_b[i]);
    }
    
#ifdef __OBJC__
    @autoreleasepool {
        id<MTLBuffer> buffer_a = [engine->device newBufferWithBytes:matrix_a_float16
                                                            length:size_a * sizeof(uint16_t)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_b = [engine->device newBufferWithBytes:matrix_b_float16
                                                            length:size_b * sizeof(uint16_t)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_result = [engine->device newBufferWithLength:size_result * sizeof(uint16_t)
                                                                   options:MTLResourceStorageModeShared];
        
        if (!buffer_a || !buffer_b || !buffer_result) {
            free(matrix_a_float16);
            free(matrix_b_float16);
            free(result_float16);
            return MIXED_PRECISION_ERROR_BUFFER_ALLOCATION_FAILED;
        }
        
        id<MTLCommandBuffer> command_buffer = [engine->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:engine->float16_matrix_multiply_pipeline];
        [encoder setBuffer:buffer_a offset:0 atIndex:0];
        [encoder setBuffer:buffer_b offset:0 atIndex:1];
        [encoder setBuffer:buffer_result offset:0 atIndex:2];
        [encoder setBytes:&rows_a length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&cols_a length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&cols_b length:sizeof(uint32_t) atIndex:5];
        
        MTLSize threadgroupSize = MTLSizeMake(32, 32, 1);
        MTLSize gridSize = MTLSizeMake((cols_b + 31) / 32, (rows_a + 31) / 32, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        memcpy(result_float16, buffer_result.contents, size_result * sizeof(uint16_t));
        
        if (command_buffer.error) {
            free(matrix_a_float16);
            free(matrix_b_float16);
            free(result_float16);
            return MIXED_PRECISION_ERROR_EXECUTION_FAILED;
        }
    }
#else
    // CPU fallback matrix multiplication
    for (uint32_t i = 0; i < rows_a; i++) {
        for (uint32_t j = 0; j < cols_b; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < cols_a; k++) {
                float a_val = float16_to_float32(matrix_a_float16[i * cols_a + k]);
                float b_val = float16_to_float32(matrix_b_float16[k * cols_b + j]);
                sum += a_val * b_val;
            }
            result_float16[i * cols_b + j] = float32_to_float16(sum);
        }
    }
#endif
    
    // Convert result back to float32
    for (size_t i = 0; i < size_result; i++) {
        result_matrix[i] = float16_to_float32(result_float16[i]);
    }
    
    // Cleanup
    free(matrix_a_float16);
    free(matrix_b_float16);
    free(result_float16);
    
    uint64_t end_time = get_current_time_microseconds();
    op_result->operation_time_us = end_time - start_time;
    op_result->precision_speedup = 1.8f; // Estimated speedup for float16 matrix multiply
    op_result->memory_saved_bytes = (size_a + size_b + size_result) * sizeof(float) / 2;
    
    // Update statistics
    engine->stats.total_operations++;
    engine->stats.float16_operations++;
    engine->stats.total_compute_time_us += op_result->operation_time_us;
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_compute_gelu_float16(MixedPrecisionEngine* engine,
                                                        const float* input_vectors,
                                                        float* output_vectors,
                                                        uint32_t total_elements,
                                                        MixedPrecisionOpResult* op_result) {
    if (!engine || !input_vectors || !output_vectors || !op_result || 
        !engine->initialized || total_elements == 0) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_current_time_microseconds();
    
    // Initialize operation result
    memset(op_result, 0, sizeof(MixedPrecisionOpResult));
    op_result->used_precision = PRECISION_TYPE_FLOAT16;
    op_result->stability_status = STABILITY_STATUS_STABLE;
    
    // Allocate float16 conversion buffers
    uint16_t* input_float16 = (uint16_t*)malloc(total_elements * sizeof(uint16_t));
    uint16_t* output_float16 = (uint16_t*)malloc(total_elements * sizeof(uint16_t));
    
    if (!input_float16 || !output_float16) {
        free(input_float16);
        free(output_float16);
        return MIXED_PRECISION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Convert input to float16
    for (uint32_t i = 0; i < total_elements; i++) {
        input_float16[i] = float32_to_float16(input_vectors[i]);
    }
    
#ifdef __OBJC__
    @autoreleasepool {
        id<MTLBuffer> input_buffer = [engine->device newBufferWithBytes:input_float16
                                                                length:total_elements * sizeof(uint16_t)
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [engine->device newBufferWithLength:total_elements * sizeof(uint16_t)
                                                                   options:MTLResourceStorageModeShared];
        
        if (!input_buffer || !output_buffer) {
            free(input_float16);
            free(output_float16);
            return MIXED_PRECISION_ERROR_BUFFER_ALLOCATION_FAILED;
        }
        
        id<MTLCommandBuffer> command_buffer = [engine->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:engine->float16_gelu_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBytes:&total_elements length:sizeof(uint32_t) atIndex:2];
        
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake((total_elements + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        memcpy(output_float16, output_buffer.contents, total_elements * sizeof(uint16_t));
        
        if (command_buffer.error) {
            free(input_float16);
            free(output_float16);
            return MIXED_PRECISION_ERROR_EXECUTION_FAILED;
        }
    }
#else
    // CPU fallback GELU computation
    for (uint32_t i = 0; i < total_elements; i++) {
        float x = float16_to_float32(input_float16[i]);
        // GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        float x_cubed = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x_cubed);
        float gelu_val = x * 0.5f * (1.0f + tanhf(inner));
        output_float16[i] = float32_to_float16(gelu_val);
    }
#endif
    
    // Convert output back to float32
    for (uint32_t i = 0; i < total_elements; i++) {
        output_vectors[i] = float16_to_float32(output_float16[i]);
    }
    
    // Cleanup
    free(input_float16);
    free(output_float16);
    
    uint64_t end_time = get_current_time_microseconds();
    op_result->operation_time_us = end_time - start_time;
    op_result->precision_speedup = 1.4f; // Estimated speedup for float16 GELU
    op_result->memory_saved_bytes = total_elements * sizeof(float) / 2;
    
    // Update statistics
    engine->stats.total_operations++;
    engine->stats.float16_operations++;
    engine->stats.total_compute_time_us += op_result->operation_time_us;
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_compute_swiglu_float16(MixedPrecisionEngine* engine,
                                                          const float* input_vectors,
                                                          const float* gate_vectors,
                                                          float* output_vectors,
                                                          uint32_t total_elements,
                                                          MixedPrecisionOpResult* op_result) {
    if (!engine || !input_vectors || !gate_vectors || !output_vectors || !op_result || 
        !engine->initialized || total_elements == 0) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_current_time_microseconds();
    
    // Initialize operation result
    memset(op_result, 0, sizeof(MixedPrecisionOpResult));
    op_result->used_precision = PRECISION_TYPE_FLOAT16;
    op_result->stability_status = STABILITY_STATUS_STABLE;
    
    // Allocate float16 conversion buffers
    uint16_t* input_float16 = (uint16_t*)malloc(total_elements * sizeof(uint16_t));
    uint16_t* gate_float16 = (uint16_t*)malloc(total_elements * sizeof(uint16_t));
    uint16_t* output_float16 = (uint16_t*)malloc(total_elements * sizeof(uint16_t));
    
    if (!input_float16 || !gate_float16 || !output_float16) {
        free(input_float16);
        free(gate_float16);
        free(output_float16);
        return MIXED_PRECISION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Convert inputs to float16
    for (uint32_t i = 0; i < total_elements; i++) {
        input_float16[i] = float32_to_float16(input_vectors[i]);
        gate_float16[i] = float32_to_float16(gate_vectors[i]);
    }
    
#ifdef __OBJC__
    @autoreleasepool {
        id<MTLBuffer> input_buffer = [engine->device newBufferWithBytes:input_float16
                                                                length:total_elements * sizeof(uint16_t)
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> gate_buffer = [engine->device newBufferWithBytes:gate_float16
                                                               length:total_elements * sizeof(uint16_t)
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [engine->device newBufferWithLength:total_elements * sizeof(uint16_t)
                                                                   options:MTLResourceStorageModeShared];
        
        if (!input_buffer || !gate_buffer || !output_buffer) {
            free(input_float16);
            free(gate_float16);
            free(output_float16);
            return MIXED_PRECISION_ERROR_BUFFER_ALLOCATION_FAILED;
        }
        
        id<MTLCommandBuffer> command_buffer = [engine->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:engine->float16_swiglu_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:gate_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBytes:&total_elements length:sizeof(uint32_t) atIndex:3];
        
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake((total_elements + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        memcpy(output_float16, output_buffer.contents, total_elements * sizeof(uint16_t));
        
        if (command_buffer.error) {
            free(input_float16);
            free(gate_float16);
            free(output_float16);
            return MIXED_PRECISION_ERROR_EXECUTION_FAILED;
        }
    }
#else
    // CPU fallback SwiGLU computation
    for (uint32_t i = 0; i < total_elements; i++) {
        float x = float16_to_float32(input_float16[i]);
        float gate = float16_to_float32(gate_float16[i]);
        // SwiGLU: x * silu(gate) where silu(x) = x * sigmoid(x)
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
        float silu_gate = gate * sigmoid_gate;
        float swiglu_val = x * silu_gate;
        output_float16[i] = float32_to_float16(swiglu_val);
    }
#endif
    
    // Convert output back to float32
    for (uint32_t i = 0; i < total_elements; i++) {
        output_vectors[i] = float16_to_float32(output_float16[i]);
    }
    
    // Cleanup
    free(input_float16);
    free(gate_float16);
    free(output_float16);
    
    uint64_t end_time = get_current_time_microseconds();
    op_result->operation_time_us = end_time - start_time;
    op_result->precision_speedup = 1.4f; // Estimated speedup for float16 SwiGLU
    op_result->memory_saved_bytes = total_elements * sizeof(float) / 2;
    
    // Update statistics
    engine->stats.total_operations++;
    engine->stats.float16_operations++;
    engine->stats.total_compute_time_us += op_result->operation_time_us;
    
    return MIXED_PRECISION_SUCCESS;
}

// Data Conversion and Utilities

MixedPrecisionError mixed_precision_convert_float32_to_float16(MixedPrecisionEngine* engine,
                                                              const float* float32_data,
                                                              uint16_t* float16_data,
                                                              size_t element_count) {
    if (!engine || !float32_data || !float16_data || element_count == 0) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    for (size_t i = 0; i < element_count; i++) {
        float16_data[i] = float32_to_float16(float32_data[i]);
    }
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_convert_float16_to_float32(MixedPrecisionEngine* engine,
                                                              const uint16_t* float16_data,
                                                              float* float32_data,
                                                              size_t element_count) {
    if (!engine || !float16_data || !float32_data || element_count == 0) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    for (size_t i = 0; i < element_count; i++) {
        float32_data[i] = float16_to_float32(float16_data[i]);
    }
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_check_numerical_issues(MixedPrecisionEngine* engine,
                                                          const float* data,
                                                          size_t element_count,
                                                          uint32_t* overflow_count,
                                                          uint32_t* underflow_count,
                                                          uint32_t* nan_count,
                                                          uint32_t* inf_count) {
    if (!engine || !data || element_count == 0 || 
        !overflow_count || !underflow_count || !nan_count || !inf_count) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    *overflow_count = 0;
    *underflow_count = 0;
    *nan_count = 0;
    *inf_count = 0;
    
    for (size_t i = 0; i < element_count; i++) {
        float val = data[i];
        
        if (isnan(val)) {
            (*nan_count)++;
        } else if (isinf(val)) {
            (*inf_count)++;
        } else if (fabs(val) > STABILITY_OVERFLOW_THRESHOLD) {
            (*overflow_count)++;
        } else if (fabs(val) < STABILITY_UNDERFLOW_THRESHOLD && val != 0.0f) {
            (*underflow_count)++;
        }
    }
    
    // Update engine statistics
    engine->stats.overflow_detections += *overflow_count;
    engine->stats.underflow_detections += *underflow_count;
    engine->stats.nan_detections += *nan_count;
    
    return MIXED_PRECISION_SUCCESS;
}

// Performance and Statistics

void mixed_precision_get_stats(MixedPrecisionEngine* engine,
                               MixedPrecisionStats* stats) {
    if (!engine || !stats) {
        return;
    }
    
    *stats = engine->stats;
    
    // Calculate derived statistics
    if (engine->stats.total_operations > 0) {
        stats->average_precision_speedup = 
            (engine->stats.float16_operations * 1.5f + engine->stats.float32_operations * 1.0f) / 
            engine->stats.total_operations;
    }
    
    stats->numerical_stability_score = engine->stability_state.stability_score;
}

void mixed_precision_reset_counters(MixedPrecisionEngine* engine) {
    if (!engine) {
        return;
    }
    
    memset(&engine->stats, 0, sizeof(MixedPrecisionStats));
    engine->stats.numerical_stability_score = 1.0f;
    engine->stability_check_counter = 0;
    engine->consecutive_instabilities = 0;
}

MixedPrecisionError mixed_precision_update_config(MixedPrecisionEngine* engine,
                                                  const MixedPrecisionConfig* config) {
    if (!engine || !config) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    engine->config = *config;
    return MIXED_PRECISION_SUCCESS;
}

void mixed_precision_destroy(MixedPrecisionEngine* engine) {
    if (!engine) {
        return;
    }
    
#ifdef __OBJC__
    // Release Metal objects
    if (engine->shader_library) {
        [engine->shader_library release];
    }
    if (engine->command_queue) {
        [engine->command_queue release];
    }
    if (engine->float32_attention_pipeline) {
        [engine->float32_attention_pipeline release];
    }
    if (engine->float32_softmax_pipeline) {
        [engine->float32_softmax_pipeline release];
    }
    if (engine->float32_layer_norm_pipeline) {
        [engine->float32_layer_norm_pipeline release];
    }
    if (engine->float16_matrix_multiply_pipeline) {
        [engine->float16_matrix_multiply_pipeline release];
    }
    if (engine->float16_ffn_pipeline) {
        [engine->float16_ffn_pipeline release];
    }
    if (engine->float16_gelu_pipeline) {
        [engine->float16_gelu_pipeline release];
    }
    if (engine->float16_swiglu_pipeline) {
        [engine->float16_swiglu_pipeline release];
    }
    if (engine->data_conversion_pipeline) {
        [engine->data_conversion_pipeline release];
    }
#endif
    
    // Free conversion buffers
    free(engine->float32_buffer);
    free(engine->float16_buffer);
    
    // Free engine structure
    free(engine);
}

// Configuration Functions

MixedPrecisionError mixed_precision_create_default_config(MixedPrecisionConfig* config) {
    if (!config) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(MixedPrecisionConfig));
    
    config->enable_automatic_precision_selection = true;
    config->enable_stability_monitoring = true;
    config->enable_gradient_scaling = true;
    config->enable_loss_scaling = true;
    config->enable_overflow_detection = true;
    config->enable_performance_optimization = true;
    config->stability_threshold = MIXED_PRECISION_DEFAULT_STABILITY_THRESHOLD;
    config->gradient_scale_factor = MIXED_PRECISION_DEFAULT_GRADIENT_SCALE;
    config->loss_scale_factor = MIXED_PRECISION_DEFAULT_LOSS_SCALE;
    config->stability_check_interval = MIXED_PRECISION_STABILITY_CHECK_INTERVAL;
    config->fallback_threshold = MIXED_PRECISION_FALLBACK_THRESHOLD;
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_create_apple_silicon_config(MixedPrecisionConfig* config) {
    if (!config) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    mixed_precision_create_default_config(config);
    
    // Apple Silicon optimizations
    config->enable_performance_optimization = true;
    config->stability_threshold = 1e-5f; // Slightly more aggressive
    config->gradient_scale_factor = 32768.0f; // Lower scaling for stability
    config->fallback_threshold = 3; // Faster fallback
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_create_conservative_config(MixedPrecisionConfig* config) {
    if (!config) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    mixed_precision_create_default_config(config);
    
    // Conservative settings prioritize stability
    config->enable_automatic_precision_selection = false; // Manual precision selection
    config->stability_threshold = 1e-7f; // Very strict threshold
    config->gradient_scale_factor = 128.0f; // Lower scaling
    config->fallback_threshold = 2; // Quick fallback
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_create_aggressive_config(MixedPrecisionConfig* config) {
    if (!config) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    mixed_precision_create_default_config(config);
    
    // Aggressive settings prioritize performance
    config->stability_threshold = 1e-4f; // Looser threshold
    config->gradient_scale_factor = 131072.0f; // Higher scaling
    config->fallback_threshold = 10; // Tolerate more instabilities
    config->stability_check_interval = 200; // Less frequent checking
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_validate_config(const MixedPrecisionConfig* config,
                                                    MetalComputeAccelerator* metal_accelerator,
                                                    bool* is_valid) {
    if (!config || !metal_accelerator || !is_valid) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    *is_valid = true;
    
    // Check Metal device capabilities
    MetalDeviceCapabilities capabilities;
    MetalComputeError error = metal_compute_get_device_capabilities(metal_accelerator, &capabilities);
    if (error != METAL_COMPUTE_SUCCESS) {
        *is_valid = false;
        return MIXED_PRECISION_SUCCESS;
    }
    
    // Check float16 support if mixed precision is enabled
    if (config->enable_automatic_precision_selection && !capabilities.supports_float16) {
        *is_valid = false;
        return MIXED_PRECISION_SUCCESS;
    }
    
    // Validate configuration parameters
    if (config->stability_threshold <= 0.0f || config->stability_threshold >= 1.0f) {
        *is_valid = false;
    }
    if (config->gradient_scale_factor <= 0.0f) {
        *is_valid = false;
    }
    if (config->loss_scale_factor <= 0.0f) {
        *is_valid = false;
    }
    if (config->stability_check_interval == 0) {
        *is_valid = false;
    }
    
    return MIXED_PRECISION_SUCCESS;
}

// Utility Functions

const char* mixed_precision_get_error_string(MixedPrecisionError error_code) {
    switch (error_code) {
        case MIXED_PRECISION_SUCCESS:
            return "Operation completed successfully";
        case MIXED_PRECISION_ERROR_INVALID_PARAM:
            return "Invalid parameter provided";
        case MIXED_PRECISION_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case MIXED_PRECISION_ERROR_UNSUPPORTED_PRECISION:
            return "Precision type not supported on this device";
        case MIXED_PRECISION_ERROR_NUMERICAL_INSTABILITY:
            return "Numerical instability detected";
        case MIXED_PRECISION_ERROR_OVERFLOW_DETECTED:
            return "Numerical overflow detected";
        case MIXED_PRECISION_ERROR_UNDERFLOW_DETECTED:
            return "Numerical underflow detected";
        case MIXED_PRECISION_ERROR_NAN_DETECTED:
            return "NaN (Not a Number) detected";
        case MIXED_PRECISION_ERROR_METAL_DEVICE_ERROR:
            return "Metal device error";
        case MIXED_PRECISION_ERROR_CONVERSION_FAILED:
            return "Precision conversion failed";
        case MIXED_PRECISION_ERROR_STABILITY_CHECK_FAILED:
            return "Numerical stability check failed";
        case MIXED_PRECISION_ERROR_FALLBACK_FAILED:
            return "Precision fallback mechanism failed";
        default:
            return "Unknown error";
    }
}

const char* mixed_precision_get_precision_type_string(PrecisionType precision_type) {
    switch (precision_type) {
        case PRECISION_TYPE_FLOAT32:
            return "Float32 (Full Precision)";
        case PRECISION_TYPE_FLOAT16:
            return "Float16 (Half Precision)";
        case PRECISION_TYPE_BFLOAT16:
            return "BFloat16 (Brain Float)";
        case PRECISION_TYPE_MIXED_ADAPTIVE:
            return "Mixed Adaptive Precision";
        case PRECISION_TYPE_MIXED_CONSERVATIVE:
            return "Mixed Conservative Precision";
        default:
            return "Unknown Precision Type";
    }
}

const char* mixed_precision_get_criticality_string(OperationCriticalLevel criticality) {
    switch (criticality) {
        case OPERATION_CRITICAL_LEVEL_CRITICAL:
            return "Critical (Attention, Gradients)";
        case OPERATION_CRITICAL_LEVEL_IMPORTANT:
            return "Important (Layer Norm, Softmax)";
        case OPERATION_CRITICAL_LEVEL_STANDARD:
            return "Standard (FFN Intermediate)";
        case OPERATION_CRITICAL_LEVEL_ACCELERATED:
            return "Accelerated (Speed Priority)";
        default:
            return "Unknown Criticality Level";
    }
}

const char* mixed_precision_get_stability_status_string(NumericalStabilityStatus status) {
    switch (status) {
        case STABILITY_STATUS_STABLE:
            return "Stable";
        case STABILITY_STATUS_WARNING:
            return "Warning";
        case STABILITY_STATUS_UNSTABLE:
            return "Unstable";
        case STABILITY_STATUS_CRITICAL:
            return "Critical";
        default:
            return "Unknown Status";
    }
}

MixedPrecisionError mixed_precision_estimate_memory_savings(size_t total_parameters,
                                                           float float16_ratio,
                                                           size_t* estimated_savings_mb) {
    if (!estimated_savings_mb || float16_ratio < 0.0f || float16_ratio > 1.0f) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    // Calculate memory savings
    // Float32: 4 bytes per parameter, Float16: 2 bytes per parameter
    size_t float32_memory = total_parameters * 4; // All parameters in float32
    size_t mixed_memory = (size_t)(total_parameters * (1.0f - float16_ratio) * 4) + 
                         (size_t)(total_parameters * float16_ratio * 2);
    
    size_t savings_bytes = float32_memory - mixed_memory;
    *estimated_savings_mb = savings_bytes / (1024 * 1024);
    
    return MIXED_PRECISION_SUCCESS;
}

MixedPrecisionError mixed_precision_estimate_performance_improvement(const char* operation_type,
                                                                    size_t data_size,
                                                                    PrecisionType precision_type,
                                                                    float* estimated_speedup) {
    if (!operation_type || !estimated_speedup || data_size == 0) {
        return MIXED_PRECISION_ERROR_INVALID_PARAM;
    }
    
    // Base speedup estimates for float16 vs float32 on Apple Silicon
    float base_speedup = 1.0f;
    
    if (precision_type == PRECISION_TYPE_FLOAT16) {
        if (strcmp(operation_type, "matrix_multiply") == 0) {
            base_speedup = 1.8f; // Matrix operations benefit significantly from float16
        } else if (strcmp(operation_type, "ffn") == 0) {
            base_speedup = 1.6f; // FFN operations see good speedup
        } else if (strcmp(operation_type, "activation") == 0) {
            base_speedup = 1.4f; // Activation functions get moderate speedup
        } else if (strcmp(operation_type, "attention") == 0) {
            base_speedup = 1.0f; // Attention stays in float32 for stability
        } else {
            base_speedup = 1.3f; // Default moderate speedup
        }
        
        // Adjust for data size (larger operations benefit more)
        float size_factor = 1.0f + fminf(0.5f, (float)data_size / (1024 * 1024)); // Up to 50% boost for large data
        base_speedup *= size_factor;
    }
    
    *estimated_speedup = base_speedup;
    return MIXED_PRECISION_SUCCESS;
}
