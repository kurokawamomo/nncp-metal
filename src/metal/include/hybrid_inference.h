#ifndef HYBRID_INFERENCE_H
#define HYBRID_INFERENCE_H

#ifdef USE_METAL

#include "neural_engine.h"
#include "metal_context.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hybrid inference engine combining CoreML, MPS, and custom Metal kernels
typedef struct HybridInferenceEngine HybridInferenceEngine;

// Inference backend capabilities
typedef enum {
    BACKEND_CAPABILITY_ML_INFERENCE    = 1 << 0,  // Neural network models
    BACKEND_CAPABILITY_MATRIX_OPS      = 1 << 1,  // Optimized linear algebra
    BACKEND_CAPABILITY_CUSTOM_KERNELS  = 1 << 2,  // Custom compute shaders
    BACKEND_CAPABILITY_PREPROCESSING   = 1 << 3,  // Data preprocessing
    BACKEND_CAPABILITY_LOW_POWER       = 1 << 4,  // Power-efficient execution
    BACKEND_CAPABILITY_HIGH_THROUGHPUT = 1 << 5   // Maximum performance
} BackendCapabilities;

// Execution strategy for different operation types
typedef enum {
    EXECUTION_STRATEGY_AUTO = 0,       // Automatic selection
    EXECUTION_STRATEGY_NEURAL_ENGINE,  // Force Neural Engine (CoreML)
    EXECUTION_STRATEGY_MPS,            // Force Metal Performance Shaders
    EXECUTION_STRATEGY_CUSTOM_METAL,   // Force custom Metal kernels
    EXECUTION_STRATEGY_HYBRID          // Mixed execution across backends
} ExecutionStrategy;

// Performance profile for adaptive selection
typedef struct {
    float latency_requirement_ms;      // Maximum acceptable latency
    float power_budget_ratio;          // 0.0 = unlimited, 1.0 = maximum efficiency
    float accuracy_requirement;        // Minimum accuracy (for quantization decisions)
    bool real_time_priority;           // Prioritize consistent latency
    size_t memory_budget_mb;           // Maximum memory usage
} PerformanceProfile;

// Operation metadata for routing decisions
typedef struct {
    const char* operation_name;
    size_t input_size_bytes;
    size_t output_size_bytes;
    uint32_t batch_size;
    bool is_ml_operation;              // Neural network vs. general compute
    BackendCapabilities required_caps;
} OperationMetadata;

// Engine configuration
typedef struct {
    bool enable_coreml;                // Enable CoreML Neural Engine path
    bool enable_mps;                   // Enable Metal Performance Shaders
    bool enable_custom_kernels;        // Enable custom Metal compute shaders
    bool enable_dynamic_routing;       // Allow runtime backend switching
    PerformanceProfile default_profile;
    const char* model_cache_path;      // Directory for compiled models
} HybridEngineConfig;

// Engine lifecycle
int hybrid_engine_create(HybridInferenceEngine** engine, const HybridEngineConfig* config);
void hybrid_engine_destroy(HybridInferenceEngine* engine);
int hybrid_engine_get_capabilities(const HybridInferenceEngine* engine, BackendCapabilities* caps);

// Model management
typedef struct HybridModel HybridModel;

int hybrid_model_load_pytorch(HybridInferenceEngine* engine, 
                              const char* pytorch_path, 
                              HybridModel** model);

int hybrid_model_load_coreml(HybridInferenceEngine* engine,
                             const char* coreml_path,
                             HybridModel** model);

int hybrid_model_load_onnx(HybridInferenceEngine* engine,
                           const char* onnx_path,
                           HybridModel** model);

void hybrid_model_destroy(HybridModel* model);

// Model conversion and optimization
int hybrid_convert_pytorch_to_coreml(const char* pytorch_path, 
                                     const char* coreml_output_path,
                                     const char* optimization_profile);

int hybrid_optimize_model_for_device(HybridModel* model, 
                                     const PerformanceProfile* profile);

// High-level inference operations
int hybrid_infer_transformer(HybridInferenceEngine* engine,
                             HybridModel* model,
                             const float* input_tokens, uint32_t seq_len,
                             float* output_logits, uint32_t vocab_size);

int hybrid_infer_lstm(HybridInferenceEngine* engine,
                     HybridModel* model,
                     const float* input_sequence, uint32_t seq_len, uint32_t input_dim,
                     float* output_sequence, uint32_t output_dim);

// Mid-level MPS operations
int hybrid_matrix_multiply_mps(HybridInferenceEngine* engine,
                              const float* a, const float* b, float* c,
                              uint32_t m, uint32_t n, uint32_t k);

int hybrid_convolution_mps(HybridInferenceEngine* engine,
                          const float* input, const float* weights, float* output,
                          uint32_t batch, uint32_t channels, uint32_t height, uint32_t width,
                          uint32_t kernel_size, uint32_t stride, uint32_t padding);

int hybrid_attention_mps(HybridInferenceEngine* engine,
                        const float* query, const float* key, const float* value,
                        float* output,
                        uint32_t seq_len, uint32_t head_dim, uint32_t num_heads);

// Low-level custom Metal operations
int hybrid_custom_kernel_execute(HybridInferenceEngine* engine,
                                 const char* kernel_name,
                                 void** buffers, uint32_t num_buffers,
                                 uint32_t* buffer_sizes,
                                 uint32_t grid_x, uint32_t grid_y, uint32_t grid_z);

// Preprocessing operations (optimized for different backends)
int hybrid_preprocess_text(HybridInferenceEngine* engine,
                           const uint8_t* raw_text, size_t text_len,
                           uint32_t* tokens, size_t max_tokens,
                           const PerformanceProfile* profile);

int hybrid_preprocess_normalize(HybridInferenceEngine* engine,
                               const float* input, float* output, size_t size,
                               float mean, float std);

// Performance monitoring and optimization
typedef struct {
    double total_time_ms;
    double coreml_time_ms;
    double mps_time_ms;
    double custom_kernel_time_ms;
    double memory_transfer_time_ms;
    size_t peak_memory_usage_mb;
    uint32_t backend_switches;
    float power_efficiency_score;    // Higher is better
} HybridPerformanceMetrics;

int hybrid_get_performance_metrics(const HybridInferenceEngine* engine, 
                                  HybridPerformanceMetrics* metrics);

int hybrid_reset_performance_counters(HybridInferenceEngine* engine);

// Adaptive execution strategy
ExecutionStrategy hybrid_suggest_strategy(const HybridInferenceEngine* engine,
                                         const OperationMetadata* op_meta,
                                         const PerformanceProfile* profile);

int hybrid_set_execution_strategy(HybridInferenceEngine* engine,
                                 const char* operation_pattern,
                                 ExecutionStrategy strategy);

// Batch processing optimization
typedef struct {
    uint32_t optimal_batch_size;
    ExecutionStrategy recommended_strategy;
    float estimated_latency_ms;
    float estimated_power_cost;
} BatchOptimizationResult;

int hybrid_optimize_batch_processing(const HybridInferenceEngine* engine,
                                    const OperationMetadata* op_meta,
                                    uint32_t max_batch_size,
                                    const PerformanceProfile* profile,
                                    BatchOptimizationResult* result);

// Memory management across backends
int hybrid_allocate_shared_buffer(HybridInferenceEngine* engine,
                                 size_t size,
                                 void** cpu_ptr,
                                 void** metal_buffer,
                                 void** coreml_buffer);

void hybrid_free_shared_buffer(HybridInferenceEngine* engine,
                              void* cpu_ptr,
                              void* metal_buffer,
                              void* coreml_buffer);

// Model compilation and caching
int hybrid_compile_model_cache(HybridInferenceEngine* engine,
                              HybridModel* model,
                              const PerformanceProfile* profile);

int hybrid_load_compiled_cache(HybridInferenceEngine* engine,
                              const char* cache_path,
                              HybridModel** model);

// Debugging and profiling
typedef enum {
    PROFILING_LEVEL_NONE = 0,
    PROFILING_LEVEL_BASIC,      // Basic timing and memory
    PROFILING_LEVEL_DETAILED,   // Per-operation breakdown
    PROFILING_LEVEL_VERBOSE     // Full execution trace
} ProfilingLevel;

int hybrid_set_profiling_level(HybridInferenceEngine* engine, ProfilingLevel level);
int hybrid_export_profiling_data(const HybridInferenceEngine* engine, const char* output_path);

#ifdef __cplusplus
}
#endif

#endif /* USE_METAL */

#endif /* HYBRID_INFERENCE_H */
