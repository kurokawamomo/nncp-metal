#ifdef USE_METAL

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Metal/Metal.h>

#include "neural_engine.h"
#include "metal_context.h"
#include "../../neural/engines/gpu_native_transformer.h"
#include "../../neural/engines/gpu_native_lstm.h"
#include "../../neural/memory/unified_memory_manager.h"

// Internal Neural Engine context structure
@interface NEContextImpl : NSObject
@property (nonatomic, strong) MLModel* coremlModel;
@property (nonatomic, strong) id<MTLDevice> metalDevice;
@property (nonatomic) NEBackendType currentBackend;
@property (nonatomic) NEAdaptiveConfig adaptiveConfig;
@property (nonatomic) NEPerformanceMetrics lastMetrics;
@property (nonatomic) UnifiedMemoryManager* memoryManager;
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
    GPUTransformerContext* nativeTransformer;
    GPULSTMContext* nativeLSTM;
    NSString* modelPath;
    NEPerformanceMetrics metrics;
    bool isNative;
    bool isLSTM;
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
    
    // Initialize Metal device
    ctx->impl.metalDevice = MTLCreateSystemDefaultDevice();
    
    // Initialize Unified Memory Manager
    if (ctx->impl.metalDevice) {
        ctx->impl.memoryManager = unified_memory_manager_create(ctx->impl.metalDevice);
    }
    
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
        if (context->impl.memoryManager) {
            unified_memory_manager_destroy(context->impl.memoryManager);
        }
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
    modelWrapper->nativeTransformer = NULL;
    modelWrapper->nativeLSTM = NULL;
    modelWrapper->modelPath = path;
    modelWrapper->isNative = false;
    modelWrapper->isLSTM = false;
    memset(&modelWrapper->metrics, 0, sizeof(NEPerformanceMetrics));
    
    *model = modelWrapper;
    return 0;
}

int ne_model_load_from_memory(NEContext* context, const void* model_data, size_t size, NEModel** model) {
    // Not implemented for CoreML in-memory
    return -1;
}

// Helper to upload weights
static id<MTLBuffer> upload_weights(UnifiedMemoryManager* mgr, const float* data, size_t count) {
    if (!data || count == 0) return nil;
    size_t size = count * sizeof(float);
    void* ptr = unified_memory_alloc(mgr, size, UNIFIED_POOL_WEIGHTS, "weights");
    if (!ptr) return nil;
    memcpy(ptr, data, size);
    return unified_memory_get_buffer(mgr, ptr);
}

int ne_model_create_transformer(NEContext* context, const NETransformerConfig* config, const NETransformerWeights* weights, NEModel** model) {
    if (!context || !config || !weights || !model) return -1;
    
    if (!context->impl.metalDevice || !context->impl.memoryManager) {
        return -1; // Metal not available
    }
    
    NEModel* modelWrapper = (NEModel*)malloc(sizeof(NEModel));
    if (!modelWrapper) return -1;
    
    modelWrapper->context = context->impl;
    modelWrapper->coremlModel = nil;
    modelWrapper->modelPath = nil;
    modelWrapper->isNative = true;
    modelWrapper->isLSTM = false;
    memset(&modelWrapper->metrics, 0, sizeof(NEPerformanceMetrics));
    
    // Create Native Transformer
    GPUTransformerConfig gpuConfig;
    gpuConfig.num_layers = config->num_layers;
    gpuConfig.hidden_size = config->hidden_size;
    gpuConfig.num_heads = config->num_heads;
    gpuConfig.ffn_size = config->ffn_size;
    gpuConfig.context_length = config->context_length;
    gpuConfig.vocab_size = config->vocab_size;
    
    modelWrapper->nativeTransformer = gpu_transformer_create(context->impl.metalDevice, context->impl.memoryManager, gpuConfig);
    if (!modelWrapper->nativeTransformer) {
        free(modelWrapper);
        return -1;
    }
    
    // Upload weights
    UnifiedMemoryManager* mm = context->impl.memoryManager;
    
    id<MTLBuffer> w_embed = upload_weights(mm, weights->embed, config->vocab_size * config->hidden_size);
    id<MTLBuffer> w_pos = upload_weights(mm, weights->pos_embed, config->context_length * config->hidden_size);
    
    size_t attn_size = config->num_layers * config->hidden_size * config->hidden_size;
    id<MTLBuffer> w_q = upload_weights(mm, weights->attn_q, attn_size);
    id<MTLBuffer> w_k = upload_weights(mm, weights->attn_k, attn_size);
    id<MTLBuffer> w_v = upload_weights(mm, weights->attn_v, attn_size);
    id<MTLBuffer> w_out = upload_weights(mm, weights->attn_out, attn_size);
    
    size_t ffn_size = config->num_layers * config->hidden_size * config->ffn_size * 2; // *2 for GEGLU
    id<MTLBuffer> w_ffn1 = upload_weights(mm, weights->ffn_1, ffn_size);
    
    size_t ffn2_size = config->num_layers * config->ffn_size * config->hidden_size;
    id<MTLBuffer> w_ffn2 = upload_weights(mm, weights->ffn_2, ffn2_size);
    
    size_t ln_size = config->num_layers * 2 * config->hidden_size; // Gamma+Beta
    id<MTLBuffer> w_ln = upload_weights(mm, weights->ln_weights, ln_size);
    
    size_t final_ln_size = 2 * config->hidden_size;
    id<MTLBuffer> w_final_ln = upload_weights(mm, weights->final_ln_weights, final_ln_size);
    
    id<MTLBuffer> w_proj = upload_weights(mm, weights->out_proj, config->hidden_size * config->vocab_size);
    
    gpu_transformer_set_weights(modelWrapper->nativeTransformer,
                               w_embed, w_pos, w_q, w_k, w_v, w_out,
                               w_ffn1, w_ffn2, w_ln, w_final_ln, w_proj);
    
    *model = modelWrapper;
    return 0;
}

int ne_model_create_lstm(NEContext* context, const NELSTMConfig* config, const NELSTMWeights* weights, NEModel** model) {
    if (!context || !config || !weights || !model) return -1;
    
    if (!context->impl.metalDevice || !context->impl.memoryManager) {
        return -1; // Metal not available
    }
    
    NEModel* modelWrapper = (NEModel*)malloc(sizeof(NEModel));
    if (!modelWrapper) return -1;
    
    modelWrapper->context = context->impl;
    modelWrapper->coremlModel = nil;
    modelWrapper->modelPath = nil;
    modelWrapper->isNative = true;
    modelWrapper->isLSTM = true;
    memset(&modelWrapper->metrics, 0, sizeof(NEPerformanceMetrics));
    
    // Create Native LSTM
    GPULSTMConfig gpuConfig;
    gpuConfig.input_size = config->input_size;
    gpuConfig.hidden_size = config->hidden_size;
    gpuConfig.num_layers = config->num_layers;
    gpuConfig.seq_len = config->seq_len;
    gpuConfig.batch_size = config->batch_size;
    
    modelWrapper->nativeLSTM = gpu_lstm_create(context->impl.metalDevice, context->impl.memoryManager, gpuConfig);
    if (!modelWrapper->nativeLSTM) {
        free(modelWrapper);
        return -1;
    }
    
    // Upload weights
    UnifiedMemoryManager* mm = context->impl.memoryManager;
    
    // Weights sizes
    size_t ih_size = config->num_layers * 4 * config->hidden_size * config->input_size;
    size_t hh_size = config->num_layers * 4 * config->hidden_size * config->hidden_size;
    size_t bias_size = config->num_layers * 4 * config->hidden_size;
    
    id<MTLBuffer> w_ih = upload_weights(mm, weights->w_ih, ih_size);
    id<MTLBuffer> w_hh = upload_weights(mm, weights->w_hh, hh_size);
    id<MTLBuffer> bias = upload_weights(mm, weights->bias, bias_size);
    
    gpu_lstm_set_weights(modelWrapper->nativeLSTM, w_ih, w_hh, bias);
    
    *model = modelWrapper;
    return 0;
}

void ne_model_destroy(NEModel* model) {
    if (model) {
        if (model->isNative) {
            if (model->nativeTransformer) {
                gpu_transformer_destroy(model->nativeTransformer);
            }
            if (model->nativeLSTM) {
                gpu_lstm_destroy(model->nativeLSTM);
            }
        }
        model->coremlModel = nil;  // ARC cleanup
        free(model);
    }
}

// Inference operations
int ne_model_predict(NEModel* model, 
                    const float* input, size_t input_size,
                    float* output, size_t output_size) {
    // Single inference not supported for Native Transformer (batch optimized)
    // Or we can treat as batch=1
    if (model && model->isNative) {
        // Native transformer expects int32 tokens, not float input
        return -1; 
    }
    
    // Fallback to CoreML
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
                          const int32_t* inputs, size_t batch_size, size_t seq_len,
                          float* outputs, size_t output_size) {
    if (!model) return -1;
    
    if (model->isNative && !model->isLSTM) {
        NSDate* startTime = [NSDate date];
        
        bool success = gpu_transformer_predict_batch(model->nativeTransformer,
                                                    inputs,
                                                    batch_size,
                                                    seq_len,
                                                    outputs);
                                                    
        NSTimeInterval elapsedTime = [[NSDate date] timeIntervalSinceDate:startTime];
        model->metrics.inference_time_ms = elapsedTime * 1000.0;
        model->metrics.backend_used = NE_BACKEND_METAL_GPU;
        
        return success ? 0 : -1;
    }
    
    // CoreML fallback (not implemented for int32 input batch)
    return -1;
}

// LSTM Sequence Prediction
int ne_model_predict_lstm(NEModel* model,
                         const float* input, size_t batch_size, size_t seq_len,
                         float* output) {
    if (!model) return -1;
    
    if (model->isNative && model->isLSTM) {
        NSDate* startTime = [NSDate date];
        
        bool success = gpu_lstm_predict_sequence(model->nativeLSTM,
                                                input,
                                                batch_size,
                                                seq_len,
                                                output);
                                                
        NSTimeInterval elapsedTime = [[NSDate date] timeIntervalSinceDate:startTime];
        model->metrics.inference_time_ms = elapsedTime * 1000.0;
        model->metrics.backend_used = NE_BACKEND_METAL_GPU;
        
        return success ? 0 : -1;
    }
    
    return -1;
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
