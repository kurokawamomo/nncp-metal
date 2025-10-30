#ifndef COMPUTE_KERNELS_H
#define COMPUTE_KERNELS_H

#ifdef USE_METAL

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "metal_context.h"
#include "memory_manager.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct ComputeKernelContext ComputeKernelContext;

// Compute operation types
typedef enum {
    CK_OP_MATRIX_MULTIPLY = 0,
    CK_OP_CONVOLUTION_2D = 1,
    CK_OP_ATTENTION = 2,
    CK_OP_LAYER_NORM = 3,
    CK_OP_ACTIVATION = 4,
    CK_OP_POOLING = 5,
    CK_OP_EMBEDDING = 6,
    CK_OP_SOFTMAX = 7,
    CK_OP_COUNT = 8
} ComputeKernelOp;

// Activation function types
typedef enum {
    CK_ACTIVATION_RELU = 0,
    CK_ACTIVATION_GELU = 1,
    CK_ACTIVATION_SWISH = 2,
    CK_ACTIVATION_TANH = 3,
    CK_ACTIVATION_SIGMOID = 4,
    CK_ACTIVATION_SILU = 5
} ActivationType;

// Pooling types
typedef enum {
    CK_POOL_MAX = 0,
    CK_POOL_AVERAGE = 1,
    CK_POOL_ADAPTIVE_MAX = 2,
    CK_POOL_ADAPTIVE_AVERAGE = 3
} PoolingType;

// Tensor descriptor for multi-dimensional data
typedef struct {
    uint32_t dims[4];           // [batch, channels, height, width] or [batch, seq_len, hidden_dim, head_dim]
    uint32_t ndim;              // Number of dimensions (2, 3, or 4)
    uint32_t stride[4];         // Memory stride for each dimension
    size_t total_elements;      // Total number of elements
    size_t data_type_size;      // Size of each element (4 for float32)
} TensorDescriptor;

// Matrix multiplication parameters
typedef struct {
    TensorDescriptor a_desc;    // Input tensor A
    TensorDescriptor b_desc;    // Input tensor B  
    TensorDescriptor c_desc;    // Output tensor C
    bool transpose_a;           // Transpose A before multiplication
    bool transpose_b;           // Transpose B before multiplication
    float alpha;                // Scaling factor for A*B
    float beta;                 // Scaling factor for existing C
} MatMulParams;

// Convolution parameters
typedef struct {
    TensorDescriptor input_desc;    // Input tensor [N, C_in, H, W]
    TensorDescriptor weight_desc;   // Weight tensor [C_out, C_in, K_h, K_w]
    TensorDescriptor bias_desc;     // Bias tensor [C_out] (optional)
    TensorDescriptor output_desc;   // Output tensor [N, C_out, H_out, W_out]
    uint32_t stride_h, stride_w;    // Convolution strides
    uint32_t padding_h, padding_w;  // Padding
    uint32_t dilation_h, dilation_w; // Dilation factors
    uint32_t groups;                // Number of groups (1 for normal conv)
} ConvParams;

// Multi-head attention parameters  
typedef struct {
    TensorDescriptor query_desc;    // Query tensor [batch, seq_len, hidden_dim]
    TensorDescriptor key_desc;      // Key tensor [batch, seq_len, hidden_dim]
    TensorDescriptor value_desc;    // Value tensor [batch, seq_len, hidden_dim]
    TensorDescriptor output_desc;   // Output tensor [batch, seq_len, hidden_dim]
    TensorDescriptor mask_desc;     // Attention mask (optional)
    uint32_t num_heads;             // Number of attention heads
    uint32_t head_dim;              // Dimension per head
    float scale;                    // Attention scaling factor (1/sqrt(head_dim))
    bool causal_mask;               // Apply causal (lower triangular) mask
} AttentionParams;

// Layer normalization parameters
typedef struct {
    TensorDescriptor input_desc;    // Input tensor
    TensorDescriptor output_desc;   // Output tensor
    TensorDescriptor gamma_desc;    // Scale parameters
    TensorDescriptor beta_desc;     // Bias parameters  
    float epsilon;                  // Small value for numerical stability
    int32_t normalized_shape[4];    // Shape to normalize over
    uint32_t normalized_ndim;       // Number of dimensions to normalize
} LayerNormParams;

// Pooling parameters
typedef struct {
    TensorDescriptor input_desc;    // Input tensor [N, C, H, W]
    TensorDescriptor output_desc;   // Output tensor [N, C, H_out, W_out]
    PoolingType pool_type;          // Type of pooling
    uint32_t kernel_h, kernel_w;    // Pooling kernel size
    uint32_t stride_h, stride_w;    // Pooling strides
    uint32_t padding_h, padding_w;  // Padding
} PoolingParams;

// Performance metrics for kernel operations
typedef struct {
    double execution_time_ms;       // Kernel execution time
    double memory_bandwidth_gbps;   // Achieved memory bandwidth
    double compute_utilization;     // GPU compute utilization (0-1)
    size_t memory_used_bytes;       // Total memory used
    uint32_t thread_groups;         // Number of thread groups dispatched
    uint32_t threads_per_group;     // Threads per thread group
} KernelMetrics;

// Compute kernel context management
MetalError ck_context_create(ComputeKernelContext** context, MetalContext* metal_context, MMManager* memory_manager);
void ck_context_destroy(ComputeKernelContext* context);

// Basic linear algebra operations
MetalError ck_matrix_multiply(ComputeKernelContext* context,
                             const MMBuffer* a, const MMBuffer* b, MMBuffer* c,
                             const MatMulParams* params);

MetalError ck_batch_matrix_multiply(ComputeKernelContext* context,
                                   const MMBuffer* a, const MMBuffer* b, MMBuffer* c,
                                   const MatMulParams* params, uint32_t batch_size);

// Convolution operations
MetalError ck_convolution_2d(ComputeKernelContext* context,
                            const MMBuffer* input, const MMBuffer* weight, const MMBuffer* bias,
                            MMBuffer* output, const ConvParams* params);

MetalError ck_depthwise_convolution_2d(ComputeKernelContext* context,
                                      const MMBuffer* input, const MMBuffer* weight, const MMBuffer* bias,
                                      MMBuffer* output, const ConvParams* params);

// Attention mechanisms
MetalError ck_multi_head_attention(ComputeKernelContext* context,
                                  const MMBuffer* query, const MMBuffer* key, const MMBuffer* value,
                                  MMBuffer* output, const AttentionParams* params);

MetalError ck_scaled_dot_product_attention(ComputeKernelContext* context,
                                          const MMBuffer* query, const MMBuffer* key, const MMBuffer* value,
                                          MMBuffer* output, const AttentionParams* params);

// Normalization operations
MetalError ck_layer_normalization(ComputeKernelContext* context,
                                 const MMBuffer* input, const MMBuffer* gamma, const MMBuffer* beta,
                                 MMBuffer* output, const LayerNormParams* params);

MetalError ck_batch_normalization(ComputeKernelContext* context,
                                 const MMBuffer* input, const MMBuffer* gamma, const MMBuffer* beta,
                                 const MMBuffer* mean, const MMBuffer* variance,
                                 MMBuffer* output, float epsilon);

// Activation functions
MetalError ck_activation(ComputeKernelContext* context,
                        const MMBuffer* input, MMBuffer* output,
                        ActivationType activation_type, const TensorDescriptor* desc);

MetalError ck_glu_activation(ComputeKernelContext* context,
                            const MMBuffer* input, MMBuffer* output,
                            const TensorDescriptor* desc);

// Pooling operations
MetalError ck_pooling_2d(ComputeKernelContext* context,
                        const MMBuffer* input, MMBuffer* output,
                        const PoolingParams* params);

// Softmax and probability operations
MetalError ck_softmax(ComputeKernelContext* context,
                     const MMBuffer* input, MMBuffer* output,
                     const TensorDescriptor* desc, int32_t dim);

MetalError ck_log_softmax(ComputeKernelContext* context,
                         const MMBuffer* input, MMBuffer* output,
                         const TensorDescriptor* desc, int32_t dim);

// Embedding operations
MetalError ck_embedding_lookup(ComputeKernelContext* context,
                              const MMBuffer* weight, const MMBuffer* indices,
                              MMBuffer* output, uint32_t vocab_size, uint32_t embedding_dim);

// Utility operations
MetalError ck_tensor_copy(ComputeKernelContext* context,
                         const MMBuffer* src, MMBuffer* dst,
                         const TensorDescriptor* desc);

MetalError ck_tensor_fill(ComputeKernelContext* context,
                         MMBuffer* tensor, float value,
                         const TensorDescriptor* desc);

MetalError ck_tensor_scale(ComputeKernelContext* context,
                          const MMBuffer* input, MMBuffer* output,
                          float scale, const TensorDescriptor* desc);

MetalError ck_tensor_add(ComputeKernelContext* context,
                        const MMBuffer* a, const MMBuffer* b, MMBuffer* c,
                        const TensorDescriptor* desc);





// Performance and debugging
MetalError ck_get_kernel_metrics(ComputeKernelContext* context, ComputeKernelOp op, KernelMetrics* metrics);
void ck_print_kernel_metrics(const KernelMetrics* metrics, const char* operation_name);
MetalError ck_profile_operation(ComputeKernelContext* context, ComputeKernelOp op, bool enable);

// Kernel optimization and tuning
MetalError ck_optimize_kernel_params(ComputeKernelContext* context, ComputeKernelOp op, const TensorDescriptor* desc);
MetalError ck_set_thread_group_size(ComputeKernelContext* context, ComputeKernelOp op, uint32_t threads_per_group);

// Tensor descriptor utilities
MetalError ck_create_tensor_desc(TensorDescriptor* desc, const uint32_t* dims, uint32_t ndim);
MetalError ck_validate_tensor_desc(const TensorDescriptor* desc);
size_t ck_calculate_tensor_size(const TensorDescriptor* desc);
bool ck_tensor_desc_compatible(const TensorDescriptor* a, const TensorDescriptor* b);

// String utilities for debugging
const char* ck_op_string(ComputeKernelOp op);
const char* ck_activation_string(ActivationType activation);
const char* ck_pooling_string(PoolingType pooling);

#ifdef __cplusplus
}
#endif

#endif /* USE_METAL */

#endif /* COMPUTE_KERNELS_H */
