/*
 * apple_silicon_adapter.mm
 *
 * Implementation of Apple Silicon Compatibility Adapter
 */

#include "apple_silicon_adapter.h"
#include <sys/sysctl.h>
#include <iostream>

AppleSiliconAdapter& AppleSiliconAdapter::getInstance() {
    static AppleSiliconAdapter instance;
    return instance;
}

AppleSiliconAdapter::AppleSiliconAdapter() : chipType(AppleSiliconChip::Unknown), isDetected(false) {
    // Default generic params
    params = {
        .preferredThreadgroupSize = 32, // Safe default
        .preferredBatchSize = 32,
        .supportsSimdShuffle = false,
        .supportsBfloat16 = false,
        .supportsMatrixFloat32 = false,
        .l2CacheSize = 0,
        .recommendedTileSizeM = 32,
        .recommendedTileSizeN = 32,
        .recommendedTileSizeK = 32
    };
}

void AppleSiliconAdapter::detectHardware(id<MTLDevice> device) {
    if (isDetected) return;
    
    // 1. Sysctl detection for precise chip model
    char buffer[128];
    size_t bufferLen = 128;
    if (sysctlbyname("machdep.cpu.brand_string", buffer, &bufferLen, NULL, 0) == 0) {
        chipNameStr = std::string(buffer);
    } else {
        chipNameStr = "Unknown Apple Silicon";
    }
    
    // Simple string matching (robust enough for this purpose)
    if (chipNameStr.find("M1 Ultra") != std::string::npos) chipType = AppleSiliconChip::M1Ultra;
    else if (chipNameStr.find("M1 Max") != std::string::npos) chipType = AppleSiliconChip::M1Max;
    else if (chipNameStr.find("M1 Pro") != std::string::npos) chipType = AppleSiliconChip::M1Pro;
    else if (chipNameStr.find("M1") != std::string::npos) chipType = AppleSiliconChip::M1;
    else if (chipNameStr.find("M2 Ultra") != std::string::npos) chipType = AppleSiliconChip::M2Ultra;
    else if (chipNameStr.find("M2 Max") != std::string::npos) chipType = AppleSiliconChip::M2Max;
    else if (chipNameStr.find("M2 Pro") != std::string::npos) chipType = AppleSiliconChip::M2Pro;
    else if (chipNameStr.find("M2") != std::string::npos) chipType = AppleSiliconChip::M2;
    else if (chipNameStr.find("M3 Ultra") != std::string::npos) chipType = AppleSiliconChip::M3Ultra;
    else if (chipNameStr.find("M3 Max") != std::string::npos) chipType = AppleSiliconChip::M3Max;
    else if (chipNameStr.find("M3 Pro") != std::string::npos) chipType = AppleSiliconChip::M3Pro;
    else if (chipNameStr.find("M3") != std::string::npos) chipType = AppleSiliconChip::M3;
    else chipType = AppleSiliconChip::Other;
    
    // 2. Feature detection via Metal Device
    if (device) {
        params.supportsSimdShuffle = [device supportsFamily:MTLGPUFamilyApple1]; // A11+ supports SIMD
        params.supportsBfloat16 = [device supportsFamily:MTLGPUFamilyApple6]; // A14/M1+ ? Actually M1 supports it via NE, GPU support varies.
        // M1 GPU doesn't natively support BFloat16 storage in buffers usually, but compute might.
        // Metal 3 adds more support.
        
        // Check for SIMD group size
        // Usually 32 on Apple GPUs
    }
    
    // 3. Configure Params
    switch (chipType) {
        case AppleSiliconChip::M1:
        case AppleSiliconChip::M1Pro:
        case AppleSiliconChip::M1Max:
        case AppleSiliconChip::M1Ultra:
            configureForM1Family();
            break;
        case AppleSiliconChip::M2:
        case AppleSiliconChip::M2Pro:
        case AppleSiliconChip::M2Max:
        case AppleSiliconChip::M2Ultra:
            configureForM2Family();
            break;
        case AppleSiliconChip::M3:
        case AppleSiliconChip::M3Pro:
        case AppleSiliconChip::M3Max:
        case AppleSiliconChip::M3Ultra:
            configureForM3Family();
            break;
        default:
            configureGeneric();
            break;
    }
    
    isDetected = true;
    printf("[AppleSilicon] Detected: %s\n", chipNameStr.c_str());
}

void AppleSiliconAdapter::configureForM1Family() {
    // M1 Family Characteristics
    // SIMD width: 32
    // Threadgroup max: 1024
    // Cache line: 128 bytes
    
    params.preferredThreadgroupSize = 32; // Align with SIMD
    params.preferredBatchSize = 32; // Good baseline
    
    // M1 specific tuning
    params.recommendedTileSizeM = 32;
    params.recommendedTileSizeN = 32;
    params.recommendedTileSizeK = 32;
    
    if (chipType == AppleSiliconChip::M1Max || chipType == AppleSiliconChip::M1Ultra) {
        // Larger GPU, can handle larger batches
        params.preferredBatchSize = 64; 
    }
}

void AppleSiliconAdapter::configureForM2Family() {
    // M2 Family - similar to M1 but higher bandwidth/flops
    params.preferredThreadgroupSize = 32;
    params.preferredBatchSize = 64; // M2 handles larger batches better
    
    params.recommendedTileSizeM = 32; // Still 32 is often optimal for SIMD-group matrix mul
    params.recommendedTileSizeN = 64; // Wider tiles might help
    params.recommendedTileSizeK = 32;
    
    if (chipType == AppleSiliconChip::M2Max || chipType == AppleSiliconChip::M2Ultra) {
        params.preferredBatchSize = 128;
    }
}

void AppleSiliconAdapter::configureForM3Family() {
    // M3 Family - Dynamic Caching, Hardware Ray Tracing (irrelevant here but indicates new arch)
    params.preferredThreadgroupSize = 32;
    params.preferredBatchSize = 64;
    
    // M3 might benefit from different tiling due to new register file architecture
    params.recommendedTileSizeM = 32;
    params.recommendedTileSizeN = 32;
    params.recommendedTileSizeK = 32; // Conservative defaults until profiled
    
    if (chipType == AppleSiliconChip::M3Max || chipType == AppleSiliconChip::M3Ultra) {
        params.preferredBatchSize = 256; // Massive throughput potential
    }
}

void AppleSiliconAdapter::configureGeneric() {
    // Fallback
    params.preferredThreadgroupSize = 32;
    params.preferredBatchSize = 16;
}

OptimizationParams AppleSiliconAdapter::getOptimizationParams() {
    return params;
}

std::string AppleSiliconAdapter::getChipName() {
    return chipNameStr;
}

bool AppleSiliconAdapter::hasUnifiedMemory() {
    // All Apple Silicon has UMA
    return chipType != AppleSiliconChip::Unknown && chipType != AppleSiliconChip::Other;
}

bool AppleSiliconAdapter::hasNeuralEngine() {
    // All M-series have NE
    return chipType != AppleSiliconChip::Unknown && chipType != AppleSiliconChip::Other;
}
