/*
 * apple_silicon_adapter.h
 *
 * Apple Silicon Compatibility Adapter
 * Detects chip model (M1/M2/M3) and configures optimization parameters.
 */

#ifndef APPLE_SILICON_ADAPTER_H
#define APPLE_SILICON_ADAPTER_H

#import <Metal/Metal.h>
#include <string>

enum class AppleSiliconChip {
    Unknown,
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M3Ultra,
    M4, // Future proofing
    Other
};

struct OptimizationParams {
    size_t preferredThreadgroupSize;
    size_t preferredBatchSize;
    bool supportsSimdShuffle;
    bool supportsBfloat16;
    bool supportsMatrixFloat32; // AMX support via Metal
    size_t l2CacheSize;
    size_t recommendedTileSizeM;
    size_t recommendedTileSizeN;
    size_t recommendedTileSizeK;
};

class AppleSiliconAdapter {
public:
    static AppleSiliconAdapter& getInstance();
    
    // Detect chip and capabilities
    void detectHardware(id<MTLDevice> device);
    
    // Get optimization parameters for current hardware
    OptimizationParams getOptimizationParams();
    
    // Get chip name string
    std::string getChipName();
    
    // Check specific feature support
    bool hasUnifiedMemory();
    bool hasNeuralEngine();
    
private:
    AppleSiliconAdapter();
    
    AppleSiliconChip chipType;
    OptimizationParams params;
    bool isDetected;
    std::string chipNameStr;
    
    void configureForM1Family();
    void configureForM2Family();
    void configureForM3Family();
    void configureGeneric();
};

#endif // APPLE_SILICON_ADAPTER_H
