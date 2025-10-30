#ifdef USE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include "metal_context.h"

// Metal context implementation
struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> defaultLibrary;
};

// Check if Metal is available
bool metal_is_available(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
}

// Create Metal context
MetalError metal_context_create(MetalContext** context) {
    if (!context) return METAL_ERROR_INVALID_PARAMETER;
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        return METAL_ERROR_DEVICE_NOT_FOUND;
    }
    
    MetalContext* ctx = (MetalContext*)malloc(sizeof(MetalContext));
    if (!ctx) {
        return METAL_ERROR_OUT_OF_MEMORY;
    }
    
    ctx->device = device;
    ctx->commandQueue = [device newCommandQueue];
    ctx->defaultLibrary = nil; // Will be loaded later
    
    *context = ctx;
    return METAL_SUCCESS;
}

// Destroy Metal context
void metal_context_destroy(MetalContext* context) {
    if (context) {
        context->device = nil;
        context->commandQueue = nil;
        context->defaultLibrary = nil;
        free(context);
    }
}

// Buffer operations - stub implementations
MetalError metal_buffer_create(MetalContext* context, size_t size, MetalBuffer** buffer) {
    if (!context || !buffer) return METAL_ERROR_INVALID_PARAMETER;
    
    MetalBuffer* buf = (MetalBuffer*)malloc(sizeof(MetalBuffer));
    if (!buf) return METAL_ERROR_OUT_OF_MEMORY;
    
    id<MTLBuffer> metalBuffer = [context->device newBufferWithLength:size 
                                                             options:MTLResourceStorageModeShared];
    if (!metalBuffer) {
        free(buf);
        return METAL_ERROR_BUFFER_ALLOCATION;
    }
    
    buf->data = [metalBuffer contents];
    buf->size = size;
    buf->is_shared = true;
    buf->metal_buffer = (__bridge void*)metalBuffer;
    
    *buffer = buf;
    return METAL_SUCCESS;
}

void metal_buffer_destroy(MetalBuffer* buffer) {
    if (buffer) {
        // Metal buffer will be released automatically by ARC
        free(buffer);
    }
}

// Stub implementations for other functions
MetalError metal_buffer_create_with_data(MetalContext* context, const void* data, size_t size, MetalBuffer** buffer) {
    return METAL_ERROR_INVALID_PARAMETER; // Not implemented yet
}

MetalError metal_buffer_copy_to_gpu(MetalBuffer* buffer) {
    return METAL_SUCCESS; // No-op for shared memory
}

MetalError metal_buffer_copy_from_gpu(MetalBuffer* buffer) {
    return METAL_SUCCESS; // No-op for shared memory
}

MetalError metal_matrix_multiply(MetalContext* context,
                                const MetalBuffer* a, const MetalBuffer* b, MetalBuffer* c,
                                uint32_t m, uint32_t n, uint32_t k) {
    return METAL_ERROR_INVALID_PARAMETER; // Not implemented yet
}

MetalError metal_vector_add(MetalContext* context,
                           const MetalBuffer* a, const MetalBuffer* b, MetalBuffer* result,
                           uint32_t size) {
    return METAL_ERROR_INVALID_PARAMETER; // Not implemented yet
}

MetalError metal_activation_relu(MetalContext* context,
                                const MetalBuffer* input, MetalBuffer* output,
                                uint32_t size) {
    return METAL_ERROR_INVALID_PARAMETER; // Not implemented yet
}

MetalError metal_transformer_attention(MetalContext* context,
                                      const MetalBuffer* query, const MetalBuffer* key, const MetalBuffer* value,
                                      MetalBuffer* output,
                                      uint32_t seq_len, uint32_t head_dim) {
    return METAL_ERROR_INVALID_PARAMETER; // Not implemented yet
}

MetalError metal_layer_norm(MetalContext* context,
                           const MetalBuffer* input, MetalBuffer* output,
                           const MetalBuffer* gamma, const MetalBuffer* beta,
                           uint32_t size, float eps) {
    return METAL_ERROR_INVALID_PARAMETER; // Not implemented yet
}

const char* metal_error_string(MetalError error) {
    switch (error) {
        case METAL_SUCCESS: return "Success";
        case METAL_ERROR_DEVICE_NOT_FOUND: return "Metal device not found";
        case METAL_ERROR_LIBRARY_LOAD: return "Failed to load Metal library";
        case METAL_ERROR_KERNEL_COMPILE: return "Failed to compile Metal kernel";
        case METAL_ERROR_BUFFER_ALLOCATION: return "Failed to allocate Metal buffer";
        case METAL_ERROR_COMMAND_EXECUTION: return "Failed to execute Metal command";
        case METAL_ERROR_INVALID_PARAMETER: return "Invalid parameter";
        case METAL_ERROR_OUT_OF_MEMORY: return "Out of memory";
        default: return "Unknown error";
    }
}

void metal_print_device_info(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        NSLog(@"Metal Device: %@", device.name);
        NSLog(@"Registry ID: %llu", device.registryID);
        NSLog(@"Max working set size: %llu MB", device.recommendedMaxWorkingSetSize / (1024*1024));
        NSLog(@"Supports unified memory: %s", device.hasUnifiedMemory ? "YES" : "NO");
    } else {
        NSLog(@"No Metal device available");
    }
}

#endif /* USE_METAL */
