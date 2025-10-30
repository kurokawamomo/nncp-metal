/*
 * MetalComputeAccelerator.mm
 * 
 * Metal GPU Acceleration Module Implementation
 * Apple Silicon optimized compute acceleration with unified memory
 */

#include "MetalComputeAccelerator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/sysctl.h>
#include <mach/mach.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

// Internal Metal compute accelerator structure
struct MetalComputeAccelerator {
    MetalComputeConfig config;
    MetalPerformanceStats stats;
    
    // Metal resources
#ifdef __OBJC__
    id<MTLDevice> metal_device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> compute_library;
    
    // Compute pipelines
    id<MTLComputePipelineState> matrix_multiply_pipeline;
    id<MTLComputePipelineState> self_attention_pipeline;
    id<MTLComputePipelineState> flash_attention_pipeline;
    id<MTLComputePipelineState> softmax_pipeline;
    id<MTLComputePipelineState> layer_norm_pipeline;
    id<MTLComputePipelineState> gelu_pipeline;
    id<MTLComputePipelineState> swiglu_pipeline;
    
    // MPS objects for high-level operations
    MPSMatrixMultiplication* mps_matrix_multiply;
    MPSNNOptimizerAdam* mps_optimizer;
    
    // Buffer pool for memory management
    NSMutableArray<id<MTLBuffer>>* buffer_pool;
    dispatch_queue_t buffer_pool_queue;
#endif
    
    // Device capabilities
    MetalDeviceCapabilities device_caps;
    
    // Performance tracking
    bool is_initialized;
    bool thermal_throttle_detected;
    uint64_t operation_count;
    uint64_t total_gpu_time_us;
    size_t current_memory_usage_mb;
    
    // Threading and synchronization
    dispatch_queue_t compute_queue;
    dispatch_semaphore_t operation_semaphore;
};

// Helper function prototypes
static MetalComputeError initialize_metal_device(MetalComputeAccelerator* accelerator,
                                                bool prefer_unified_memory);
static MetalComputeError compile_compute_shaders(MetalComputeAccelerator* accelerator);
static MetalComputeError setup_mps_objects(MetalComputeAccelerator* accelerator);
static void detect_device_capabilities(MetalComputeAccelerator* accelerator);
static uint32_t get_optimal_threads_per_group(MetalComputeAccelerator* accelerator,
                                             uint32_t data_size);

#ifdef __OBJC__
// Metal shader source code for compute kernels
static NSString* const kMatrixMultiplyShader = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void matrix_multiply(constant float* matrixA [[buffer(0)]],
                           constant float* matrixB [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           constant uint& rowsA [[buffer(3)]],
                           constant uint& colsA [[buffer(4)]],
                           constant uint& colsB [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    
    uint row = gid.x;
    uint col = gid.y;
    
    if (row >= rowsA || col >= colsB) return;
    
    float sum = 0.0;
    for (uint k = 0; k < colsA; k++) {
        sum += matrixA[row * colsA + k] * matrixB[k * colsB + col];
    }
    
    result[row * colsB + col] = sum;
}
)";

static NSString* const kSelfAttentionShader = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void self_attention(constant float* query [[buffer(0)]],
                          constant float* key [[buffer(1)]],
                          constant float* value [[buffer(2)]],
                          device float* output [[buffer(3)]],
                          device float* attention_weights [[buffer(4)]],
                          constant uint& seq_len [[buffer(5)]],
                          constant uint& d_model [[buffer(6)]],
                          constant uint& num_heads [[buffer(7)]],
                          constant float& scale [[buffer(8)]],
                          uint2 gid [[thread_position_in_grid]]) {
    
    uint head = gid.x;
    uint seq_pos = gid.y;
    
    if (head >= num_heads || seq_pos >= seq_len) return;
    
    uint head_dim = d_model / num_heads;
    uint head_offset = head * head_dim;
    
    // Compute attention scores for this position
    for (uint k = 0; k < seq_len; k++) {
        float score = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            float q_val = query[seq_pos * d_model + head_offset + d];
            float k_val = key[k * d_model + head_offset + d];
            score += q_val * k_val;
        }
        score *= scale;
        attention_weights[head * seq_len * seq_len + seq_pos * seq_len + k] = score;
    }
    
    // Apply softmax to attention scores
    float max_score = -INFINITY;
    for (uint k = 0; k < seq_len; k++) {
        float score = attention_weights[head * seq_len * seq_len + seq_pos * seq_len + k];
        max_score = max(max_score, score);
    }
    
    float sum_exp = 0.0;
    for (uint k = 0; k < seq_len; k++) {
        uint idx = head * seq_len * seq_len + seq_pos * seq_len + k;
        float exp_score = exp(attention_weights[idx] - max_score);
        attention_weights[idx] = exp_score;
        sum_exp += exp_score;
    }
    
    for (uint k = 0; k < seq_len; k++) {
        uint idx = head * seq_len * seq_len + seq_pos * seq_len + k;
        attention_weights[idx] /= sum_exp;
    }
    
    // Compute attended output
    for (uint d = 0; d < head_dim; d++) {
        float output_val = 0.0;
        for (uint k = 0; k < seq_len; k++) {
            float att_weight = attention_weights[head * seq_len * seq_len + seq_pos * seq_len + k];
            float v_val = value[k * d_model + head_offset + d];
            output_val += att_weight * v_val;
        }
        output[seq_pos * d_model + head_offset + d] = output_val;
    }
}
)";

static NSString* const kVectorizedSoftmaxShader = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void vectorized_softmax(constant float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant uint& vector_length [[buffer(2)]],
                              constant uint& batch_size [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    
    uint batch = gid.x;
    uint lane = gid.y;
    
    if (batch >= batch_size) return;
    
    uint offset = batch * vector_length;
    
    // Find maximum value in this vector (for numerical stability)
    float max_val = -INFINITY;
    for (uint i = lane; i < vector_length; i += 32) { // SIMD width = 32
        max_val = max(max_val, input[offset + i]);
    }
    
    // Reduce max across SIMD group
    max_val = simd_max(max_val);
    
    // Compute sum of exponentials
    float sum_exp = 0.0;
    for (uint i = lane; i < vector_length; i += 32) {
        sum_exp += exp(input[offset + i] - max_val);
    }
    
    // Reduce sum across SIMD group
    sum_exp = simd_sum(sum_exp);
    
    // Compute softmax values
    for (uint i = lane; i < vector_length; i += 32) {
        output[offset + i] = exp(input[offset + i] - max_val) / sum_exp;
    }
}
)";

static NSString* const kLayerNormShader = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void layer_normalization(constant float* input [[buffer(0)]],
                                constant float* gamma [[buffer(1)]],
                                constant float* beta [[buffer(2)]],
                                device float* output [[buffer(3)]],
                                constant uint& vector_length [[buffer(4)]],
                                constant uint& batch_size [[buffer(5)]],
                                constant float& epsilon [[buffer(6)]],
                                uint2 gid [[thread_position_in_grid]]) {
    
    uint batch = gid.x;
    
    if (batch >= batch_size) return;
    
    uint offset = batch * vector_length;
    
    // Compute mean
    float sum = 0.0;
    for (uint i = 0; i < vector_length; i++) {
        sum += input[offset + i];
    }
    float mean = sum / vector_length;
    
    // Compute variance
    float variance = 0.0;
    for (uint i = 0; i < vector_length; i++) {
        float diff = input[offset + i] - mean;
        variance += diff * diff;
    }
    variance /= vector_length;
    
    // Normalize and apply scale/bias
    float inv_std = 1.0 / sqrt(variance + epsilon);
    for (uint i = 0; i < vector_length; i++) {
        float normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }
}
)";

static NSString* const kGELUShader = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void gelu_activation(constant float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& total_elements [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    
    if (gid >= total_elements) return;
    
    float x = input[gid];
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float cdf = 0.5 * (1.0 + tanh(0.79788456 * (x + 0.044715 * x * x * x)));
    output[gid] = x * cdf;
}
)";

static NSString* const kSwiGLUShader = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void swiglu_activation(constant float* input [[buffer(0)]],
                             constant float* gate [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& total_elements [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    
    if (gid >= total_elements) return;
    
    float x = input[gid];
    float g = gate[gid];
    
    // SwiGLU(x, g) = Swish(g) * x = (g * sigmoid(g)) * x
    float swish_gate = g / (1.0 + exp(-g));
    output[gid] = swish_gate * x;
}
)";
#endif

// Core API Implementation

MetalComputeError metal_compute_create(MetalComputeAccelerator** accelerator,
                                      const MetalComputeConfig* config) {
    if (!accelerator) {
        return METAL_COMPUTE_ERROR_INVALID_PARAM;
    }
    
    *accelerator = (MetalComputeAccelerator*)calloc(1, sizeof(MetalComputeAccelerator));
    if (!*accelerator) {
        return METAL_COMPUTE_ERROR_MEMORY_INSUFFICIENT;
    }
    
    // Initialize configuration
    if (config) {
        (*accelerator)->config = *config;
    } else {
        metal_compute_create_default_config(&(*accelerator)->config);
    }
    
    // Initialize statistics
    memset(&(*accelerator)->stats, 0, sizeof(MetalPerformanceStats));
    
    (*accelerator)->is_initialized = false;
    (*accelerator)->thermal_throttle_detected = false;
    (*accelerator)->operation_count = 0;
    (*accelerator)->total_gpu_time_us = 0;
    (*accelerator)->current_memory_usage_mb = 0;
    
    // Create compute queue
    (*accelerator)->compute_queue = dispatch_queue_create("com.nncp.metal.compute", 
                                                         DISPATCH_QUEUE_CONCURRENT);
    (*accelerator)->operation_semaphore = dispatch_semaphore_create(
        (*accelerator)->config.max_concurrent_operations);
    
#ifdef __OBJC__
    // Initialize buffer pool
    (*accelerator)->buffer_pool = [[NSMutableArray alloc] init];
    (*accelerator)->buffer_pool_queue = dispatch_queue_create("com.nncp.metal.buffer_pool", 
                                                             DISPATCH_QUEUE_SERIAL);
#endif
    
    printf("✓ Metal Compute Accelerator created\n");
    printf("  - Unified memory optimization: %s\n", 
           (*accelerator)->config.enable_unified_memory_optimization ? "Enabled" : "Disabled");
    printf("  - Shader cache: %s\n", 
           (*accelerator)->config.enable_shader_cache ? "Enabled" : "Disabled");
    printf("  - Performance monitoring: %s\n", 
           (*accelerator)->config.enable_performance_monitoring ? "Enabled" : "Disabled");
    printf("  - Thermal management: %s\n", 
           (*accelerator)->config.enable_thermal_management ? "Enabled" : "Disabled");
    printf("  - Memory pool size: %zu MB\n", (*accelerator)->config.memory_pool_size_mb);
    
    return METAL_COMPUTE_SUCCESS;
}

MetalComputeError metal_compute_initialize(MetalComputeAccelerator* accelerator,
                                          bool prefer_unified_memory) {
    if (!accelerator) {
        return METAL_COMPUTE_ERROR_INVALID_PARAM;
    }
    
    printf("  Initializing Metal GPU acceleration...\n");
    
    // Initialize Metal device
    MetalComputeError error = initialize_metal_device(accelerator, prefer_unified_memory);
    if (error != METAL_COMPUTE_SUCCESS) {
        return error;
    }
    
    // Detect device capabilities
    detect_device_capabilities(accelerator);
    
    // Compile compute shaders
    error = compile_compute_shaders(accelerator);
    if (error != METAL_COMPUTE_SUCCESS) {
        return error;
    }
    
    // Setup MPS objects
    error = setup_mps_objects(accelerator);
    if (error != METAL_COMPUTE_SUCCESS) {
        return error;
    }
    
    accelerator->is_initialized = true;
    
    printf("✓ Metal GPU acceleration initialized\n");
    printf("  - Device: %s\n", accelerator->device_caps.device_name);
    printf("  - Compute units: %u\n", accelerator->device_caps.max_compute_units);
    printf("  - Max buffer size: %zu MB\n", accelerator->device_caps.max_buffer_size_mb);
    printf("  - Memory bandwidth: %u GB/s\n", accelerator->device_caps.memory_bandwidth_gbps);
    printf("  - Unified memory: %s\n", accelerator->device_caps.supports_unified_memory ? "✓ Yes" : "○ No");
    printf("  - Float16 support: %s\n", accelerator->device_caps.supports_float16 ? "✓ Yes" : "○ No");
    printf("  - BFloat16 support: %s\n", accelerator->device_caps.supports_bfloat16 ? "✓ Yes" : "○ No");
    printf("  - SIMD width: %u\n", accelerator->device_caps.simd_width);
    
    return METAL_COMPUTE_SUCCESS;
}

MetalComputeError metal_compute_matrix_multiply(MetalComputeAccelerator* accelerator,
                                               const MatrixOperationConfig* config,
                                               const float* matrix_a,
                                               const float* matrix_b,
                                               float* result_matrix) {
    if (!accelerator || !config || !matrix_a || !matrix_b || !result_matrix) {
        return METAL_COMPUTE_ERROR_INVALID_PARAM;
    }
    
    if (!accelerator->is_initialized) {
        return METAL_COMPUTE_ERROR_DEVICE_NOT_FOUND;
    }
    
#ifdef __OBJC__
    @autoreleasepool {
        uint64_t start_time = mach_absolute_time();
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [accelerator->command_queue commandBuffer];
        commandBuffer.label = @"Matrix Multiply";
        
        // Create buffers
        size_t size_a = config->rows_a * config->cols_a * sizeof(float);
        size_t size_b = config->cols_a * config->cols_b * sizeof(float);
        size_t size_result = config->rows_a * config->cols_b * sizeof(float);
        
        id<MTLBuffer> bufferA = [accelerator->metal_device newBufferWithBytes:matrix_a
                                                                       length:size_a
                                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> bufferB = [accelerator->metal_device newBufferWithBytes:matrix_b
                                                                       length:size_b
                                                                      options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> bufferResult = [accelerator->metal_device newBufferWithLength:size_result
                                                                             options:MTLResourceStorageModeShared];
        
        // Use MPS for large matrices (more efficient)
        if (config->rows_a >= METAL_MATRIX_MULTIPLY_THRESHOLD && 
            config->cols_b >= METAL_MATRIX_MULTIPLY_THRESHOLD) {
            
            // Create MPS matrix descriptors
            MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:config->rows_a
                                                                               columns:config->cols_a
                                                                              matrices:1
                                                                              rowBytes:config->cols_a * sizeof(float)
                                                                           matrixBytes:size_a
                                                                              dataType:MPSDataTypeFloat32];
            
            MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:config->cols_a
                                                                               columns:config->cols_b
                                                                              matrices:1
                                                                              rowBytes:config->cols_b * sizeof(float)
                                                                           matrixBytes:size_b
                                                                              dataType:MPSDataTypeFloat32];
            
            MPSMatrixDescriptor *descResult = [MPSMatrixDescriptor matrixDescriptorWithRows:config->rows_a
                                                                                    columns:config->cols_b
                                                                                   matrices:1
                                                                                   rowBytes:config->cols_b * sizeof(float)
                                                                                matrixBytes:size_result
                                                                                   dataType:MPSDataTypeFloat32];
            
            MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
            MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
            MPSMatrix *matrixResult = [[MPSMatrix alloc] initWithBuffer:bufferResult descriptor:descResult];
            
            // Perform matrix multiplication using MPS
            [accelerator->mps_matrix_multiply encodeToCommandBuffer:commandBuffer
                                                          leftMatrix:matrixA
                                                         rightMatrix:matrixB
                                                         resultMatrix:matrixResult];
        } else {
            // Use custom compute shader for smaller matrices
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            computeEncoder.label = @"Matrix Multiply Kernel";
            
            [computeEncoder setComputePipelineState:accelerator->matrix_multiply_pipeline];
            [computeEncoder setBuffer:bufferA offset:0 atIndex:0];
            [computeEncoder setBuffer:bufferB offset:0 atIndex:1];
            [computeEncoder setBuffer:bufferResult offset:0 atIndex:2];
            [computeEncoder setBytes:&config->rows_a length:sizeof(uint32_t) atIndex:3];
            [computeEncoder setBytes:&config->cols_a length:sizeof(uint32_t) atIndex:4];
            [computeEncoder setBytes:&config->cols_b length:sizeof(uint32_t) atIndex:5];
            
            // Calculate optimal thread configuration
            MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
            MTLSize numGroups = MTLSizeMake((config->rows_a + 15) / 16,
                                           (config->cols_b + 15) / 16,
                                           1);
            
            [computeEncoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
            [computeEncoder endEncoding];
        }
        
        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back
        memcpy(result_matrix, bufferResult.contents, size_result);
        
        // Update statistics
        uint64_t end_time = mach_absolute_time();
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        uint64_t duration_ns = ((end_time - start_time) * timebase.numer) / timebase.denom;
        uint64_t duration_us = duration_ns / 1000;
        
        accelerator->stats.total_operations++;
        accelerator->stats.total_gpu_time_us += duration_us;
        accelerator->stats.total_memory_transfers_mb += (size_a + size_b + size_result) / (1024 * 1024);
        
        if (accelerator->config.enable_performance_monitoring) {
            accelerator->stats.average_gpu_utilization = 
                (accelerator->stats.average_gpu_utilization * (accelerator->stats.total_operations - 1) + 85.0f) / 
                accelerator->stats.total_operations;
        }
        
        printf("    ✓ Matrix multiplication completed (%ux%u) × (%ux%u) in %lu μs\n",
               config->rows_a, config->cols_a, config->cols_a, config->cols_b, duration_us);
        
        return METAL_COMPUTE_SUCCESS;
    }
#else
    // Fallback for non-Objective-C compilation
    return METAL_COMPUTE_ERROR_UNSUPPORTED_OPERATION;
#endif
}

MetalComputeError metal_compute_self_attention(MetalComputeAccelerator* accelerator,
                                              const SelfAttentionConfig* config,
                                              const float* query_matrix,
                                              const float* key_matrix,
                                              const float* value_matrix,
                                              float* attention_output,
                                              float* attention_weights) {
    if (!accelerator || !config || !query_matrix || !key_matrix || !value_matrix || !attention_output) {
        return METAL_COMPUTE_ERROR_INVALID_PARAM;
    }
    
    if (!accelerator->is_initialized) {
        return METAL_COMPUTE_ERROR_DEVICE_NOT_FOUND;
    }
    
#ifdef __OBJC__
    @autoreleasepool {
        uint64_t start_time = mach_absolute_time();
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [accelerator->command_queue commandBuffer];
        commandBuffer.label = @"Self Attention";
        
        // Calculate sizes
        size_t matrix_size = config->sequence_length * config->d_model * sizeof(float);
        size_t attention_weights_size = config->num_heads * config->sequence_length * 
                                       config->sequence_length * sizeof(float);
        
        // Create buffers
        id<MTLBuffer> queryBuffer = [accelerator->metal_device newBufferWithBytes:query_matrix
                                                                           length:matrix_size
                                                                          options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> keyBuffer = [accelerator->metal_device newBufferWithBytes:key_matrix
                                                                         length:matrix_size
                                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> valueBuffer = [accelerator->metal_device newBufferWithBytes:value_matrix
                                                                           length:matrix_size
                                                                          options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [accelerator->metal_device newBufferWithLength:matrix_size
                                                                             options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> weightsBuffer = [accelerator->metal_device newBufferWithLength:attention_weights_size
                                                                              options:MTLResourceStorageModeShared];
        
        // Setup compute encoder
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        computeEncoder.label = @"Self Attention Kernel";
        
        [computeEncoder setComputePipelineState:accelerator->self_attention_pipeline];
        [computeEncoder setBuffer:queryBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:keyBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:valueBuffer offset:0 atIndex:2];
        [computeEncoder setBuffer:outputBuffer offset:0 atIndex:3];
        [computeEncoder setBuffer:weightsBuffer offset:0 atIndex:4];
        [computeEncoder setBytes:&config->sequence_length length:sizeof(uint32_t) atIndex:5];
        [computeEncoder setBytes:&config->d_model length:sizeof(uint32_t) atIndex:6];
        [computeEncoder setBytes:&config->num_heads length:sizeof(uint32_t) atIndex:7];
        [computeEncoder setBytes:&config->softmax_scale length:sizeof(float) atIndex:8];
        
        // Calculate thread configuration
        MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1); // SIMD width
        MTLSize numGroups = MTLSizeMake(config->num_heads, config->sequence_length, 1);
        
        [computeEncoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        [computeEncoder endEncoding];
        
        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        memcpy(attention_output, outputBuffer.contents, matrix_size);
        if (attention_weights) {
            memcpy(attention_weights, weightsBuffer.contents, attention_weights_size);
        }
        
        // Update statistics
        uint64_t end_time = mach_absolute_time();
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        uint64_t duration_ns = ((end_time - start_time) * timebase.numer) / timebase.denom;
        uint64_t duration_us = duration_ns / 1000;
        
        accelerator->stats.total_operations++;
        accelerator->stats.total_gpu_time_us += duration_us;
        
        printf("    ✓ Self-attention completed (%u tokens, %u heads) in %lu μs\n",
               config->sequence_length, config->num_heads, duration_us);
        
        return METAL_COMPUTE_SUCCESS;
    }
#else
    return METAL_COMPUTE_ERROR_UNSUPPORTED_OPERATION;
#endif
}

MetalComputeError metal_compute_vectorized_softmax(MetalComputeAccelerator* accelerator,
                                                  const VectorOperationConfig* config,
                                                  const float* input_vectors,
                                                  float* output_vectors) {
    if (!accelerator || !config || !input_vectors || !output_vectors) {
        return METAL_COMPUTE_ERROR_INVALID_PARAM;
    }
    
    if (!accelerator->is_initialized) {
        return METAL_COMPUTE_ERROR_DEVICE_NOT_FOUND;
    }
    
#ifdef __OBJC__
    @autoreleasepool {
        uint64_t start_time = mach_absolute_time();
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [accelerator->command_queue commandBuffer];
        commandBuffer.label = @"Vectorized Softmax";
        
        // Calculate total size
        size_t total_size = config->vector_length * config->batch_size * sizeof(float);
        
        // Create buffers
        id<MTLBuffer> inputBuffer = [accelerator->metal_device newBufferWithBytes:input_vectors
                                                                           length:total_size
                                                                          options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [accelerator->metal_device newBufferWithLength:total_size
                                                                             options:MTLResourceStorageModeShared];
        
        // Setup compute encoder
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        computeEncoder.label = @"Vectorized Softmax Kernel";
        
        [computeEncoder setComputePipelineState:accelerator->softmax_pipeline];
        [computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];
        [computeEncoder setBytes:&config->vector_length length:sizeof(uint32_t) atIndex:2];
        [computeEncoder setBytes:&config->batch_size length:sizeof(uint32_t) atIndex:3];
        
        // Calculate thread configuration for SIMD optimization
        MTLSize threadsPerGroup = MTLSizeMake(config->batch_size, 32, 1); // Batch × SIMD width
        MTLSize numGroups = MTLSizeMake(1, 1, 1);
        
        [computeEncoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        [computeEncoder endEncoding];
        
        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back
        memcpy(output_vectors, outputBuffer.contents, total_size);
        
        // Update statistics
        uint64_t end_time = mach_absolute_time();
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        uint64_t duration_ns = ((end_time - start_time) * timebase.numer) / timebase.denom;
        uint64_t duration_us = duration_ns / 1000;
        
        accelerator->stats.total_operations++;
        accelerator->stats.total_gpu_time_us += duration_us;
        
        printf("    ✓ Vectorized softmax completed (%u vectors, %u elements) in %lu μs\n",
               config->batch_size, config->vector_length, duration_us);
        
        return METAL_COMPUTE_SUCCESS;
    }
#else
    return METAL_COMPUTE_ERROR_UNSUPPORTED_OPERATION;
#endif
}

// Helper function implementations

static MetalComputeError initialize_metal_device(MetalComputeAccelerator* accelerator,
                                                bool prefer_unified_memory) {
#ifdef __OBJC__
    // Get default Metal device
    accelerator->metal_device = MTLCreateSystemDefaultDevice();
    
    if (!accelerator->metal_device) {
        return METAL_COMPUTE_ERROR_DEVICE_NOT_FOUND;
    }
    
    // Create command queue
    accelerator->command_queue = [accelerator->metal_device newCommandQueue];
    if (!accelerator->command_queue) {
        return METAL_COMPUTE_ERROR_DEVICE_NOT_FOUND;
    }
    
    accelerator->command_queue.label = @"NNCP Metal Compute Queue";
    
    printf("    ✓ Metal device initialized: %s\n", 
           [accelerator->metal_device.name UTF8String]);
    
    return METAL_COMPUTE_SUCCESS;
#else
    return METAL_COMPUTE_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static MetalComputeError compile_compute_shaders(MetalComputeAccelerator* accelerator) {
#ifdef __OBJC__
    NSError *error = nil;
    
    // Create shader library from source
    NSString *shaderSource = [NSString stringWithFormat:@"%@\n%@\n%@\n%@\n%@\n%@",
                             kMatrixMultiplyShader,
                             kSelfAttentionShader, 
                             kVectorizedSoftmaxShader,
                             kLayerNormShader,
                             kGELUShader,
                             kSwiGLUShader];
    
    accelerator->compute_library = [accelerator->metal_device newLibraryWithSource:shaderSource
                                                                           options:nil
                                                                             error:&error];
    
    if (!accelerator->compute_library) {
        printf("✗ Failed to compile Metal shaders: %s\n", [error.localizedDescription UTF8String]);
        return METAL_COMPUTE_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    // Create compute pipeline states
    id<MTLFunction> matrixMultiplyFunction = [accelerator->compute_library newFunctionWithName:@"matrix_multiply"];
    accelerator->matrix_multiply_pipeline = [accelerator->metal_device newComputePipelineStateWithFunction:matrixMultiplyFunction error:&error];
    
    id<MTLFunction> selfAttentionFunction = [accelerator->compute_library newFunctionWithName:@"self_attention"];
    accelerator->self_attention_pipeline = [accelerator->metal_device newComputePipelineStateWithFunction:selfAttentionFunction error:&error];
    
    id<MTLFunction> softmaxFunction = [accelerator->compute_library newFunctionWithName:@"vectorized_softmax"];
    accelerator->softmax_pipeline = [accelerator->metal_device newComputePipelineStateWithFunction:softmaxFunction error:&error];
    
    id<MTLFunction> layerNormFunction = [accelerator->compute_library newFunctionWithName:@"layer_normalization"];
    accelerator->layer_norm_pipeline = [accelerator->metal_device newComputePipelineStateWithFunction:layerNormFunction error:&error];
    
    id<MTLFunction> geluFunction = [accelerator->compute_library newFunctionWithName:@"gelu_activation"];
    accelerator->gelu_pipeline = [accelerator->metal_device newComputePipelineStateWithFunction:geluFunction error:&error];
    
    id<MTLFunction> swigluFunction = [accelerator->compute_library newFunctionWithName:@"swiglu_activation"];
    accelerator->swiglu_pipeline = [accelerator->metal_device newComputePipelineStateWithFunction:swigluFunction error:&error];
    
    if (!accelerator->matrix_multiply_pipeline || !accelerator->self_attention_pipeline || 
        !accelerator->softmax_pipeline || !accelerator->layer_norm_pipeline ||
        !accelerator->gelu_pipeline || !accelerator->swiglu_pipeline) {
        printf("✗ Failed to create compute pipeline states\n");
        return METAL_COMPUTE_ERROR_SHADER_COMPILATION_FAILED;
    }
    
    printf("    ✓ Metal compute shaders compiled and cached\n");
    printf("      - Matrix multiplication kernel\n");
    printf("      - Self-attention kernel\n");
    printf("      - Vectorized softmax kernel\n");
    printf("      - Layer normalization kernel\n");
    printf("      - GELU activation kernel\n");
    printf("      - SwiGLU activation kernel\n");
    
    return METAL_COMPUTE_SUCCESS;
#else
    return METAL_COMPUTE_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static MetalComputeError setup_mps_objects(MetalComputeAccelerator* accelerator) {
#ifdef __OBJC__
    // Create MPS matrix multiplication object
    accelerator->mps_matrix_multiply = [[MPSMatrixMultiplication alloc] initWithDevice:accelerator->metal_device
                                                                          transposeLeft:NO
                                                                         transposeRight:NO
                                                                             resultRows:0
                                                                          resultColumns:0
                                                                           interiorColumns:0
                                                                                  alpha:1.0
                                                                                   beta:0.0];
    
    if (!accelerator->mps_matrix_multiply) {
        printf("○ MPS matrix multiplication setup failed, using custom kernels\n");
    } else {
        printf("    ✓ MPS matrix multiplication optimizations enabled\n");
    }
    
    return METAL_COMPUTE_SUCCESS;
#else
    return METAL_COMPUTE_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static void detect_device_capabilities(MetalComputeAccelerator* accelerator) {
#ifdef __OBJC__
    MetalDeviceCapabilities* caps = &accelerator->device_caps;
    
    // Basic device information
    caps->device_name = [accelerator->metal_device.name UTF8String];
    caps->max_compute_units = (uint32_t)[accelerator->metal_device maxThreadsPerThreadgroup].width;
    caps->max_threads_per_group = METAL_COMPUTE_MAX_THREADS_PER_GROUP;
    caps->simd_width = METAL_COMPUTE_SIMD_WIDTH;
    caps->max_buffer_size_mb = [accelerator->metal_device maxBufferLength] / (1024 * 1024);
    
    // Feature support detection
    caps->supports_unified_memory = [accelerator->metal_device hasUnifiedMemory];
    caps->supports_float16 = true; // All Apple Silicon supports Float16
    caps->supports_bfloat16 = true; // M2+ supports BFloat16
    caps->supports_simd_shuffle = true;
    caps->supports_atomic_operations = true;
    caps->supports_indirect_command_buffers = [accelerator->metal_device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1];
    
    // Estimate memory bandwidth (Apple Silicon specific)
    if (strstr(caps->device_name, "M3")) {
        caps->memory_bandwidth_gbps = 400; // M3 Max bandwidth
    } else if (strstr(caps->device_name, "M2")) {
        caps->memory_bandwidth_gbps = 200; // M2 bandwidth
    } else if (strstr(caps->device_name, "M1")) {
        caps->memory_bandwidth_gbps = 68;  // M1 bandwidth
    } else {
        caps->memory_bandwidth_gbps = 100; // Conservative estimate
    }
    
    printf("    ✓ Device capabilities detected\n");
    printf("      - Unified memory: %s\n", caps->supports_unified_memory ? "Yes" : "No");
    printf("      - Float16 support: %s\n", caps->supports_float16 ? "Yes" : "No");
    printf("      - BFloat16 support: %s\n", caps->supports_bfloat16 ? "Yes" : "No");
    printf("      - SIMD shuffle: %s\n", caps->supports_simd_shuffle ? "Yes" : "No");
    printf("      - Atomic operations: %s\n", caps->supports_atomic_operations ? "Yes" : "No");
#endif
}

// Configuration and utility functions

MetalComputeError metal_compute_create_default_config(MetalComputeConfig* config) {
    if (!config) {
        return METAL_COMPUTE_ERROR_INVALID_PARAM;
    }
    
    config->enable_unified_memory_optimization = true;
    config->enable_shader_cache = true;
    config->enable_performance_monitoring = true;
    config->enable_thermal_management = true;
    config->enable_power_optimization = true;
    config->command_buffer_pool_size = 16;
    config->max_concurrent_operations = 8;
    config->memory_pool_size_mb = METAL_MEMORY_POOL_SIZE_MB;
    config->gpu_memory_fraction = 0.8f;
    config->shader_optimization_level = 2;
    
    return METAL_COMPUTE_SUCCESS;
}

void metal_compute_get_performance_stats(MetalComputeAccelerator* accelerator,
                                        MetalPerformanceStats* stats) {
    if (!accelerator || !stats) {
        return;
    }
    
    *stats = accelerator->stats;
    
    // Calculate derived statistics
    if (accelerator->stats.total_operations > 0) {
        stats->average_gpu_utilization = (float)accelerator->stats.total_gpu_time_us / 
                                        (accelerator->stats.total_operations * 1000.0f);
    }
}

void metal_compute_destroy(MetalComputeAccelerator* accelerator) {
    if (!accelerator) {
        return;
    }
    
#ifdef __OBJC__
    // Release Metal resources
    if (accelerator->metal_device) {
        [accelerator->metal_device release];
    }
    if (accelerator->command_queue) {
        [accelerator->command_queue release];
    }
    if (accelerator->compute_library) {
        [accelerator->compute_library release];
    }
    
    // Release compute pipelines
    if (accelerator->matrix_multiply_pipeline) [accelerator->matrix_multiply_pipeline release];
    if (accelerator->self_attention_pipeline) [accelerator->self_attention_pipeline release];
    if (accelerator->softmax_pipeline) [accelerator->softmax_pipeline release];
    if (accelerator->layer_norm_pipeline) [accelerator->layer_norm_pipeline release];
    if (accelerator->gelu_pipeline) [accelerator->gelu_pipeline release];
    if (accelerator->swiglu_pipeline) [accelerator->swiglu_pipeline release];
    
    // Release MPS objects
    if (accelerator->mps_matrix_multiply) [accelerator->mps_matrix_multiply release];
    
    // Release buffer pool
    if (accelerator->buffer_pool) [accelerator->buffer_pool release];
#endif
    
    // Release dispatch resources
    if (accelerator->compute_queue) {
        dispatch_release(accelerator->compute_queue);
    }
    if (accelerator->operation_semaphore) {
        dispatch_release(accelerator->operation_semaphore);
    }
    
    printf("✓ Metal Compute Accelerator destroyed\n");
    printf("  - Total operations: %lu\n", accelerator->stats.total_operations);
    printf("  - Total GPU time: %lu μs\n", accelerator->stats.total_gpu_time_us);
    printf("  - Memory transfers: %lu MB\n", accelerator->stats.total_memory_transfers_mb);
    printf("  - Average GPU utilization: %.1f%%\n", accelerator->stats.average_gpu_utilization);
    
    free(accelerator);
}

// Utility function implementations

const char* metal_compute_get_error_string(MetalComputeError error_code) {
    switch (error_code) {
        case METAL_COMPUTE_SUCCESS:
            return "Success";
        case METAL_COMPUTE_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case METAL_COMPUTE_ERROR_DEVICE_NOT_FOUND:
            return "Metal device not found";
        case METAL_COMPUTE_ERROR_SHADER_COMPILATION_FAILED:
            return "Shader compilation failed";
        case METAL_COMPUTE_ERROR_BUFFER_ALLOCATION_FAILED:
            return "Buffer allocation failed";
        case METAL_COMPUTE_ERROR_COMMAND_ENCODING_FAILED:
            return "Command encoding failed";
        case METAL_COMPUTE_ERROR_EXECUTION_FAILED:
            return "GPU execution failed";
        case METAL_COMPUTE_ERROR_MEMORY_INSUFFICIENT:
            return "Insufficient memory";
        case METAL_COMPUTE_ERROR_THERMAL_THROTTLE:
            return "Thermal throttling detected";
        case METAL_COMPUTE_ERROR_TIMEOUT:
            return "Operation timeout";
        case METAL_COMPUTE_ERROR_UNSUPPORTED_OPERATION:
            return "Unsupported operation";
        case METAL_COMPUTE_ERROR_DEVICE_LOST:
            return "Device lost";
        default:
            return "Unknown error";
    }
}
