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
    NE_BACKEND_METAL_GPU,     // Force Metal GPU (Native Transformer)
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

// Model operations
typedef struct NEModel NEModel;

void ne_model_destroy(NEModel* model);

// Inference operations
int ne_model_predict(NEModel* model,
                    const float* input, size_t input_size,
                    float* output, size_t output_size);

#ifdef __cplusplus
}
#endif

#endif /* USE_METAL */

#endif /* NEURAL_ENGINE_H */
