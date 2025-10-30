#ifndef METAL_CONTEXT_H
#define METAL_CONTEXT_H

#ifdef USE_METAL

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct MetalContext MetalContext;

// Error codes
typedef enum {
    METAL_SUCCESS = 0,
    METAL_ERROR_DEVICE_NOT_FOUND = -1,
    METAL_ERROR_LIBRARY_LOAD = -2,
    METAL_ERROR_KERNEL_COMPILE = -3,
    METAL_ERROR_BUFFER_ALLOCATION = -4,
    METAL_ERROR_COMMAND_EXECUTION = -5,
    METAL_ERROR_INVALID_PARAMETER = -6,
    METAL_ERROR_OUT_OF_MEMORY = -7
} MetalError;

// Buffer management
typedef struct MetalBuffer {
    void* data;                // CPU-accessible pointer
    size_t size;              // Buffer size in bytes
    bool is_shared;           // Uses shared memory
    void* metal_buffer;       // Opaque Metal buffer handle
} MetalBuffer;

// Context management
MetalError metal_context_create(MetalContext** context);
void metal_context_destroy(MetalContext* context);
bool metal_is_available(void);

// Buffer operations
MetalError metal_buffer_create(MetalContext* context, size_t size, MetalBuffer** buffer);
MetalError metal_buffer_create_with_data(MetalContext* context, const void* data, size_t size, MetalBuffer** buffer);
void metal_buffer_destroy(MetalBuffer* buffer);
MetalError metal_buffer_copy_to_gpu(MetalBuffer* buffer);
MetalError metal_buffer_copy_from_gpu(MetalBuffer* buffer);

// Compute operations
MetalError metal_matrix_multiply(MetalContext* context,
                                const MetalBuffer* a, const MetalBuffer* b, MetalBuffer* c,
                                uint32_t m, uint32_t n, uint32_t k);

MetalError metal_vector_add(MetalContext* context,
                           const MetalBuffer* a, const MetalBuffer* b, MetalBuffer* result,
                           uint32_t size);

MetalError metal_activation_relu(MetalContext* context,
                                const MetalBuffer* input, MetalBuffer* output,
                                uint32_t size);

// Neural network operations
MetalError metal_transformer_attention(MetalContext* context,
                                      const MetalBuffer* query, const MetalBuffer* key, const MetalBuffer* value,
                                      MetalBuffer* output,
                                      uint32_t seq_len, uint32_t head_dim);

MetalError metal_layer_norm(MetalContext* context,
                           const MetalBuffer* input, MetalBuffer* output,
                           const MetalBuffer* gamma, const MetalBuffer* beta,
                           uint32_t size, float eps);

// Utility functions
const char* metal_error_string(MetalError error);
void metal_print_device_info(void);

#ifdef __cplusplus
}
#endif

#endif /* USE_METAL */

#endif /* METAL_CONTEXT_H */
