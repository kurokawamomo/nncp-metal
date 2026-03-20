#ifdef USE_METAL

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Metal/Metal.h>

#include "neural_engine.h"
#include "metal_context.h"
#include "../../neural/unified_memory_manager.h"

// Internal Neural Engine context structure
@interface NEContextImpl : NSObject
@property (nonatomic, strong) MLModel* coremlModel;
@property (nonatomic, strong) id<MTLDevice> metalDevice;
@property (nonatomic) NEBackendType currentBackend;
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
};

// System information detection
int ne_get_system_info(NESystemInfo* info) {
    if (!info) return -1;

    memset(info, 0, sizeof(NESystemInfo));

    // Check Neural Engine availability (available on A11+ and M1+)
    if (@available(macOS 13.0, *)) {
        #ifdef __aarch64__
        info->neural_engine_available = true;
        info->neural_engine_performance = 1.0f;
        info->neural_engine_memory_mb = 1024;
        #else
        info->neural_engine_available = false;
        #endif
    }

    // Check Metal GPU availability
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        info->metal_gpu_available = true;
        info->metal_gpu_performance = 0.8f;
        if (@available(macOS 13.0, *)) {
            info->metal_gpu_memory_mb = (size_t)(device.recommendedMaxWorkingSetSize / (1024 * 1024));
        } else {
            info->metal_gpu_memory_mb = 4096;
        }
    }

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
    ctx->impl.metalDevice = MTLCreateSystemDefaultDevice();

    if (ctx->impl.metalDevice) {
        ctx->impl.memoryManager = unified_memory_manager_create(ctx->impl.metalDevice);
    }

    *context = ctx;
    return 0;
}

void ne_context_destroy(NEContext* context) {
    if (context && context->impl) {
        if (context->impl.memoryManager) {
            unified_memory_manager_destroy(context->impl.memoryManager);
        }
        context->impl = nil;
        free(context);
    }
}

void ne_model_destroy(NEModel* model) {
    if (model) {
        model->coremlModel = nil;
        free(model);
    }
}

// Inference via CoreML
int ne_model_predict(NEModel* model,
                    const float* input, size_t input_size,
                    float* output, size_t output_size) {
    if (!model || !input || !output) return -1;

    @try {
        NSArray<NSNumber*>* shape = @[@(input_size)];
        MLMultiArray* inputArray = [[MLMultiArray alloc] initWithShape:shape
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:nil];
        if (!inputArray) return -1;

        float* dataPointer = (float*)inputArray.dataPointer;
        memcpy(dataPointer, input, input_size * sizeof(float));

        NSString* inputName = model->coremlModel.modelDescription.inputDescriptionsByName.allKeys.firstObject;
        if (!inputName) return -1;

        MLDictionaryFeatureProvider* inputProvider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{inputName: inputArray} error:nil];
        if (!inputProvider) return -1;

        NSError* error = nil;
        id<MLFeatureProvider> outputProvider = [model->coremlModel predictionFromFeatures:inputProvider error:&error];
        if (!outputProvider || error) return -1;

        NSString* outputName = model->coremlModel.modelDescription.outputDescriptionsByName.allKeys.firstObject;
        if (!outputName) return -1;

        MLFeatureValue* outputFeature = [outputProvider featureValueForName:outputName];
        if (!outputFeature || !outputFeature.multiArrayValue) return -1;

        MLMultiArray* outputArray = outputFeature.multiArrayValue;
        if (outputArray.count < output_size) return -1;

        float* outputDataPointer = (float*)outputArray.dataPointer;
        memcpy(output, outputDataPointer, output_size * sizeof(float));

        return 0;

    } @catch (NSException* exception) {
        NSLog(@"Neural Engine prediction exception: %@", exception.reason);
        return -1;
    }
}

#endif /* USE_METAL */
