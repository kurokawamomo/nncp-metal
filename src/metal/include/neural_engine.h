#ifndef NEURAL_ENGINE_H
#define NEURAL_ENGINE_H

#ifdef USE_METAL

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Neural Engine availability and configuration
typedef enum {
    NE_BACKEND_AUTO = 0,      // Automatic selection
    NE_BACKEND_NEURAL_ENGINE, // Force Neural Engine (CoreML)
    NE_BACKEND_METAL_GPU,     // Force Metal GPU
    NE_BACKEND_CPU           // Force CPU
} NEBackendType;

typedef struct {
    NEBackendType backend;
    bool neural_engine_available;
    bool metal_gpu_available;
    float neural_engine_performance;  // Relative performance score
    float metal_gpu_performance;
    size_t neural_engine_memory_mb;
    size_t metal_gpu_memory_mb;
} NESystemInfo;

// Neural Engine context
typedef struct NEContext NEContext;

// Context management
int ne_get_system_info(NESystemInfo* info);
int ne_context_create(NEContext** context, NEBackendType preferred_backend);
void ne_context_destroy(NEContext* context);
NEBackendType ne_context_get_backend(const NEContext* context);

// Model operations
typedef struct NEModel NEModel;

int ne_model_load_from_file(NEContext* context, const char* model_path, NEModel** model);
int ne_model_load_from_memory(NEContext* context, const void* model_data, size_t size, NEModel** model);
void ne_model_destroy(NEModel* model);

// Inference operations
int ne_model_predict(NEModel* model, 
                    const float* input, size_t input_size,
                    float* output, size_t output_size);

int ne_model_predict_batch(NEModel* model,
                          const float* inputs, size_t batch_size, size_t input_size,
                          float* outputs, size_t output_size);

// Performance monitoring
typedef struct {
    double inference_time_ms;
    double preprocessing_time_ms;
    double postprocessing_time_ms;
    size_t memory_used_bytes;
    NEBackendType backend_used;
} NEPerformanceMetrics;

int ne_get_performance_metrics(NEModel* model, NEPerformanceMetrics* metrics);

// Model conversion utilities (for CoreML integration)
int ne_convert_onnx_to_coreml(const char* onnx_path, const char* coreml_path);
int ne_convert_pytorch_to_coreml(const char* pytorch_path, const char* coreml_path);

// Adaptive backend selection
typedef struct {
    size_t input_size_threshold;     // Switch to Neural Engine for inputs larger than this
    float latency_preference;        // 0.0 = favor throughput, 1.0 = favor latency
    float power_preference;          // 0.0 = favor performance, 1.0 = favor battery
    bool enable_dynamic_switching;   // Allow runtime backend switching
} NEAdaptiveConfig;

int ne_configure_adaptive_backend(NEContext* context, const NEAdaptiveConfig* config);
NEBackendType ne_suggest_backend(const NEContext* context, size_t input_size, bool realtime_required);

#ifdef __cplusplus
}
#endif

#endif /* USE_METAL */

#endif /* NEURAL_ENGINE_H */
