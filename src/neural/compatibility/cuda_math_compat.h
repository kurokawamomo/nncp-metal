/*
 * CUDA Mathematical Compatibility Layer
 * 
 * Ensures Metal operations produce identical results to CUDA by implementing
 * exact mathematical equivalence and deterministic computation patterns.
 * 
 * Based on original CUDA NNCP implementation for bit-perfect compatibility.
 */

#ifndef CUDA_MATH_COMPAT_H
#define CUDA_MATH_COMPAT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Math Configuration
typedef struct {
    float precision_tolerance;      // Numerical precision tolerance (e.g., 1e-6)
    bool enforce_deterministic;     // Force deterministic computation
    bool validate_intermediate;     // Validate intermediate results
    bool use_double_precision;      // Use double precision for critical ops
    int random_seed;               // Fixed seed for reproducible results
} CUDAMathConfig;

// CUDA Tensor Structure (compatible with original NCTensor)
typedef struct {
    float* data;                   // Tensor data (Metal buffer)
    size_t* shape;                 // Tensor dimensions
    size_t ndim;                   // Number of dimensions
    size_t total_size;             // Total number of elements
    bool requires_grad;            // Gradient computation flag
    bool is_metal_backed;          // Whether data is in Metal buffer
    void* metal_buffer;            // Metal buffer reference (id<MTLBuffer>)
} CUDACompatTensor;

// CUDA Math Operations (ensuring exact CUDA behavior)

// Basic arithmetic operations
CUDACompatTensor* cuda_compat_add(const CUDACompatTensor* a, const CUDACompatTensor* b, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_sub(const CUDACompatTensor* a, const CUDACompatTensor* b, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_mul(const CUDACompatTensor* a, const CUDACompatTensor* b, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_div(const CUDACompatTensor* a, const CUDACompatTensor* b, const CUDAMathConfig* config);

// Matrix operations (critical for CUDA equivalence)
CUDACompatTensor* cuda_compat_matmul(const CUDACompatTensor* a, const CUDACompatTensor* b, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_transpose(const CUDACompatTensor* input, const CUDAMathConfig* config);

// Neural network operations
CUDACompatTensor* cuda_compat_softmax(const CUDACompatTensor* input, int axis, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_layer_norm(const CUDACompatTensor* input, const CUDACompatTensor* gamma, 
                                         const CUDACompatTensor* beta, float eps, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_relu(const CUDACompatTensor* input, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_gelu(const CUDACompatTensor* input, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_tanh(const CUDACompatTensor* input, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_sigmoid(const CUDACompatTensor* input, const CUDAMathConfig* config);

// LSTM-specific operations (critical for LSTM profile compatibility)
typedef struct {
    CUDACompatTensor* hidden_state;  // h_t
    CUDACompatTensor* cell_state;    // c_t
} CUDACompatLSTMState;

CUDACompatLSTMState* cuda_compat_lstm_cell(const CUDACompatTensor* input, 
                                           const CUDACompatLSTMState* prev_state,
                                           const CUDACompatTensor* weight_ih,
                                           const CUDACompatTensor* weight_hh,
                                           const CUDACompatTensor* bias_ih,
                                           const CUDACompatTensor* bias_hh,
                                           const CUDAMathConfig* config);

// Attention mechanism (for Transformer compatibility)
CUDACompatTensor* cuda_compat_scaled_dot_product_attention(const CUDACompatTensor* query,
                                                          const CUDACompatTensor* key,
                                                          const CUDACompatTensor* value,
                                                          const CUDACompatTensor* mask,
                                                          float scale,
                                                          const CUDAMathConfig* config);

// Tensor management
CUDACompatTensor* cuda_compat_tensor_create(const size_t* shape, size_t ndim, const CUDAMathConfig* config);
CUDACompatTensor* cuda_compat_tensor_from_data(float* data, const size_t* shape, size_t ndim, 
                                               bool copy_data, const CUDAMathConfig* config);
void cuda_compat_tensor_free(CUDACompatTensor* tensor);
CUDACompatTensor* cuda_compat_tensor_clone(const CUDACompatTensor* tensor, const CUDAMathConfig* config);

// CUDA-equivalent initialization
void cuda_compat_tensor_fill(CUDACompatTensor* tensor, float value, const CUDAMathConfig* config);
void cuda_compat_tensor_normal(CUDACompatTensor* tensor, float mean, float std, const CUDAMathConfig* config);
void cuda_compat_tensor_uniform(CUDACompatTensor* tensor, float min, float max, const CUDAMathConfig* config);

// Precision and validation utilities
bool cuda_compat_tensor_allclose(const CUDACompatTensor* a, const CUDACompatTensor* b, 
                                 float rtol, float atol, const CUDAMathConfig* config);
float cuda_compat_tensor_max_diff(const CUDACompatTensor* a, const CUDACompatTensor* b, const CUDAMathConfig* config);
void cuda_compat_tensor_print_stats(const CUDACompatTensor* tensor, const char* name);

// Configuration management
CUDAMathConfig* cuda_math_config_create_default(void);
CUDAMathConfig* cuda_math_config_create_strict(void);  // Strictest CUDA compatibility
CUDAMathConfig* cuda_math_config_create_fast(void);   // Optimized for performance
void cuda_math_config_free(CUDAMathConfig* config);

// CUDA random state management (for reproducible results)
typedef struct {
    uint64_t state;
    uint64_t increment;
    bool is_initialized;
} CUDACompatRNG;

void cuda_compat_rng_init(CUDACompatRNG* rng, uint64_t seed);
float cuda_compat_rng_normal(CUDACompatRNG* rng, float mean, float std);
float cuda_compat_rng_uniform(CUDACompatRNG* rng, float min, float max);

// Metal integration utilities
bool cuda_compat_setup_metal_context(void);
void cuda_compat_cleanup_metal_context(void);
bool cuda_compat_tensor_to_metal(CUDACompatTensor* tensor);
bool cuda_compat_tensor_from_metal(CUDACompatTensor* tensor);

// Validation and debugging
typedef struct {
    size_t total_operations;
    size_t failed_validations;
    float max_observed_error;
    float avg_computation_time_ms;
} CUDACompatStats;

CUDACompatStats* cuda_compat_get_stats(void);
void cuda_compat_reset_stats(void);
void cuda_compat_enable_debugging(bool enable);

// Error handling
typedef enum {
    CUDA_COMPAT_SUCCESS = 0,
    CUDA_COMPAT_ERROR_INVALID_INPUT,
    CUDA_COMPAT_ERROR_DIMENSION_MISMATCH,
    CUDA_COMPAT_ERROR_MEMORY_ALLOCATION,
    CUDA_COMPAT_ERROR_METAL_FAILURE,
    CUDA_COMPAT_ERROR_PRECISION_EXCEEDED,
    CUDA_COMPAT_ERROR_VALIDATION_FAILED
} CUDACompatError;

const char* cuda_compat_error_string(CUDACompatError error);

#ifdef __cplusplus
}
#endif

#endif // CUDA_MATH_COMPAT_H
