#ifdef USE_METAL

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Metal/Metal.h>

#include "neural_engine.h"
#include "metal_context.h"

// Internal Neural Engine context structure
@interface NEContextImpl : NSObject
@property (nonatomic, strong) MLModel* coremlModel;
@property (nonatomic, strong) id<MTLDevice> metalDevice;
@property (nonatomic) NEBackendType currentBackend;
@property (nonatomic) NEAdaptiveConfig adaptiveConfig;
@property (nonatomic) NEPerformanceMetrics lastMetrics;
@end

@implementation NEContextImpl
@end

// C wrapper implementation
struct NEContext {
    NEContextImpl* impl;
};

struct NEModel {
    NEContextImpl* context;
    MLModel* coremlModel;
    NSString* modelPath;
    NEPerformanceMetrics metrics;
};

// System information detection
int ne_get_system_info(NESystemInfo* info) {
    if (!info) return -1;
    
    memset(info, 0, sizeof(NESystemInfo));
    
    // Check Neural Engine availability (available on A11+ and M1+)
    if (@available(macOS 13.0, *)) {
        // Check if we're on Apple Silicon
        #ifdef __aarch64__
        info->neural_engine_available = true;
        info->neural_engine_performance = 1.0f;  // Baseline
        info->neural_engine_memory_mb = 1024;    // Approximate
        #else
        info->neural_engine_available = false;
        #endif
    }
    
    // Check Metal GPU availability
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        info->metal_gpu_available = true;
        info->metal_gpu_performance = 0.8f;  // Slightly lower than Neural Engine for ML
        
        if (@available(macOS 13.0, *)) {
            info->metal_gpu_memory_mb = (size_t)(device.recommendedMaxWorkingSetSize / (1024 * 1024));
        } else {
            info->metal_gpu_memory_mb = 4096;  // Conservative estimate
        }
    }
    
    // Default backend selection
    if (info->neural_engine_available) {
        info->backend = NE_BACKEND_NEURAL_ENGINE;
    } else if (info->metal_gpu_available) {
        info->backend = NE_BACKEND_METAL_GPU;
    } else {
        info->backend = NE_BACKEND_CPU;
    }
    
    return 0;
}

// Context creation
int ne_context_create(NEContext** context, NEBackendType preferred_backend) {
    if (!context) return -1;
    
    NEContext* ctx = (NEContext*)malloc(sizeof(NEContext));
    if (!ctx) return -1;
    
    ctx->impl = [[NEContextImpl alloc] init];
    ctx->impl.currentBackend = preferred_backend;
    
    // Initialize Metal device for fallback
    ctx->impl.metalDevice = MTLCreateSystemDefaultDevice();
    
    // Set default adaptive configuration
    NEAdaptiveConfig defaultConfig = {
        .input_size_threshold = 1024 * 1024,  // 1MB threshold
        .latency_preference = 0.5f,
        .power_preference = 0.3f,  // Slightly favor performance
        .enable_dynamic_switching = true
    };
    ctx->impl.adaptiveConfig = defaultConfig;
    
    *context = ctx;
    return 0;
}

void ne_context_destroy(NEContext* context) {
    if (context && context->impl) {
        context->impl = nil;  // ARC will handle cleanup
        free(context);
    }
}

NEBackendType ne_context_get_backend(const NEContext* context) {
    if (!context || !context->impl) return NE_BACKEND_CPU;
    return context->impl.currentBackend;
}

// Model loading
int ne_model_load_from_file(NEContext* context, const char* model_path, NEModel** model) {
    if (!context || !model_path || !model) return -1;
    
    NSString* path = [NSString stringWithUTF8String:model_path];
    NSURL* modelURL = [NSURL fileURLWithPath:path];
    
    NSError* error = nil;
    MLModel* coremlModel = [MLModel modelWithContentsOfURL:modelURL error:&error];
    
    if (!coremlModel || error) {
        NSLog(@"Failed to load CoreML model: %@", error.localizedDescription);
        return -1;
    }
    
    NEModel* modelWrapper = (NEModel*)malloc(sizeof(NEModel));
    if (!modelWrapper) return -1;
    
    modelWrapper->context = context->impl;
    modelWrapper->coremlModel = coremlModel;
    modelWrapper->modelPath = path;
    memset(&modelWrapper->metrics, 0, sizeof(NEPerformanceMetrics));
    
    *model = modelWrapper;
    return 0;
}

void ne_model_destroy(NEModel* model) {
    if (model) {
        model->coremlModel = nil;  // ARC cleanup
        free(model);
    }
}

// Inference operations
int ne_model_predict(NEModel* model, 
                    const float* input, size_t input_size,
                    float* output, size_t output_size) {
    if (!model || !input || !output) return -1;
    
    NSDate* startTime = [NSDate date];
    
    @try {
        // Create MLMultiArray from input data
        NSArray<NSNumber*>* shape = @[@(input_size)];
        MLMultiArray* inputArray = [[MLMultiArray alloc] initWithShape:shape 
                                                              dataType:MLMultiArrayDataTypeFloat32 
                                                                 error:nil];
        
        if (!inputArray) return -1;
        
        // Copy input data
        float* dataPointer = (float*)inputArray.dataPointer;
        memcpy(dataPointer, input, input_size * sizeof(float));
        
        // Create feature provider
        NSString* inputName = model->coremlModel.modelDescription.inputDescriptionsByName.allKeys.firstObject;
        if (!inputName) return -1;
        
        MLDictionaryFeatureProvider* inputProvider = 
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inputName: inputArray} error:nil];
        
        if (!inputProvider) return -1;
        
        // Perform prediction
        NSError* error = nil;
        id<MLFeatureProvider> outputProvider = [model->coremlModel predictionFromFeatures:inputProvider error:&error];
        
        if (!outputProvider || error) {
            NSLog(@"Prediction failed: %@", error.localizedDescription);
            return -1;
        }
        
        // Extract output
        NSString* outputName = model->coremlModel.modelDescription.outputDescriptionsByName.allKeys.firstObject;
        if (!outputName) return -1;
        
        MLFeatureValue* outputFeature = [outputProvider featureValueForName:outputName];
        if (!outputFeature || !outputFeature.multiArrayValue) return -1;
        
        MLMultiArray* outputArray = outputFeature.multiArrayValue;
        if (outputArray.count < output_size) return -1;
        
        // Copy output data
        float* outputDataPointer = (float*)outputArray.dataPointer;
        memcpy(output, outputDataPointer, output_size * sizeof(float));
        
        // Update metrics
        NSTimeInterval elapsedTime = [[NSDate date] timeIntervalSinceDate:startTime];
        model->metrics.inference_time_ms = elapsedTime * 1000.0;
        model->metrics.backend_used = NE_BACKEND_NEURAL_ENGINE;
        
        return 0;
        
    } @catch (NSException* exception) {
        NSLog(@"Neural Engine prediction exception: %@", exception.reason);
        return -1;
    }
}

// Batch inference
int ne_model_predict_batch(NEModel* model,
                          const float* inputs, size_t batch_size, size_t input_size,
                          float* outputs, size_t output_size) {
    if (!model || !inputs || !outputs || batch_size == 0) return -1;
    
    // Simple implementation: process each input sequentially
    // In practice, CoreML might support batch processing better
    for (size_t i = 0; i < batch_size; i++) {
        const float* current_input = inputs + (i * input_size);
        float* current_output = outputs + (i * output_size);
        
        int result = ne_model_predict(model, current_input, input_size, 
                                     current_output, output_size);
        if (result != 0) return result;
    }
    
    return 0;
}

// Performance metrics
int ne_get_performance_metrics(NEModel* model, NEPerformanceMetrics* metrics) {
    if (!model || !metrics) return -1;
    
    *metrics = model->metrics;
    return 0;
}

// Adaptive backend selection
int ne_configure_adaptive_backend(NEContext* context, const NEAdaptiveConfig* config) {
    if (!context || !config) return -1;
    
    context->impl.adaptiveConfig = *config;
    return 0;
}

NEBackendType ne_suggest_backend(const NEContext* context, size_t input_size, bool realtime_required) {
    if (!context) return NE_BACKEND_CPU;
    
    NESystemInfo info;
    ne_get_system_info(&info);
    
    NEAdaptiveConfig config = context->impl.adaptiveConfig;
    
    // For large inputs, prefer Neural Engine if available
    if (input_size > config.input_size_threshold && info.neural_engine_available) {
        return NE_BACKEND_NEURAL_ENGINE;
    }
    
    // For real-time requirements with power constraints
    if (realtime_required && config.power_preference > 0.5f && info.neural_engine_available) {
        return NE_BACKEND_NEURAL_ENGINE;
    }
    
    // For maximum performance, prefer Metal GPU
    if (config.latency_preference < 0.3f && info.metal_gpu_available) {
        return NE_BACKEND_METAL_GPU;
    }
    
    // Default selection
    if (info.neural_engine_available) {
        return NE_BACKEND_NEURAL_ENGINE;
    } else if (info.metal_gpu_available) {
        return NE_BACKEND_METAL_GPU;
    } else {
        return NE_BACKEND_CPU;
    }
}

// Model conversion utilities (simplified interface)
int ne_convert_onnx_to_coreml(const char* onnx_path, const char* coreml_path) {
    // This would require coremltools Python library
    // For now, return not implemented
    return -1;
}

int ne_convert_pytorch_to_coreml(const char* pytorch_path, const char* coreml_path) {
    // This would require coremltools Python library  
    // For now, return not implemented
    return -1;
}

#endif /* USE_METAL */
