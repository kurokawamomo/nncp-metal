/*
 * Enhanced Neural Bridge with CUDA-Compatible Metal LSTM Integration
 * 
 * Replaces the Neural Bridge fallback with Enhanced Metal LSTM processing,
 * routing all compression to the strengthened Metal LSTM implementation.
 * 
 * This bridge provides CUDA mathematical equivalence and behavioral consistency.
 */

#ifndef NEURAL_BRIDGE_ENHANCED_H
#define NEURAL_BRIDGE_ENHANCED_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "../lstm/nncp_lstm_metal_enhanced.h"
#include "../handlers/small_file_handler.h"
#include "../validation/cuda_parameter_validator.h"
#include "../config/cuda_profiles.h"

#ifdef __cplusplus
extern "C" {
#endif

// Enhanced neural compression algorithm types (CUDA-compatible)
typedef enum {
    ENHANCED_NEURAL_ALGORITHM_METAL_LSTM = 0,    // Enhanced Metal LSTM (primary)
    ENHANCED_NEURAL_ALGORITHM_SMALL_FILE,        // Small file specialized handling
    ENHANCED_NEURAL_ALGORITHM_AUTO               // Automatic CUDA-profile selection
} EnhancedNeuralAlgorithm;

// Enhanced compression configuration
typedef struct {
    // Core settings
    EnhancedNeuralAlgorithm preferred_algorithm;
    const CUDAProfile* cuda_profile;             // CUDA profile for compatibility
    size_t memory_limit_bytes;
    uint8_t quality_level;                       // 1-10 quality level
    
    // CUDA compatibility settings
    bool enforce_cuda_compatibility;             // Strict CUDA compatibility mode
    bool validate_parameters;                    // Validate all parameters
    bool enable_small_file_optimization;        // Use small file handler when needed
    
    // Performance settings
    bool enable_gpu_acceleration;                // Metal GPU acceleration
    bool verbose_logging;                        // Debug logging
    float compression_target;                    // Target compression ratio (0.1-0.9)
    
    // Mathematical precision settings
    bool use_deterministic_processing;          // Deterministic results
    float precision_tolerance;                   // Numerical precision tolerance
    uint64_t random_seed;                       // Fixed seed for reproducibility
} EnhancedNeuralCompressionConfig;

// Enhanced compression result with CUDA compatibility metrics
typedef struct {
    bool success;                               // Whether compression succeeded
    size_t compressed_size;                     // Size of compressed data
    EnhancedNeuralAlgorithm algorithm_used;     // Algorithm that was used
    float compression_ratio;                    // Achieved compression ratio
    uint64_t processing_time_ns;                // Processing time in nanoseconds
    uint64_t memory_used_bytes;                 // Peak memory usage
    
    // CUDA compatibility metrics
    bool cuda_compatible;                       // Whether processing was CUDA-compatible
    bool parameter_validation_passed;          // Parameter validation result
    float max_precision_deviation;             // Maximum observed precision deviation
    const char* cuda_profile_used;             // Name of CUDA profile used
    
    // Quality metrics
    float quality_score;                       // Overall quality score (0-1)
    float entropy_preservation;                // Entropy preservation ratio
    bool meets_quality_threshold;              // Whether quality threshold was met
    
    char error_message[256];                   // Error message if failed
    char cuda_validation_report[512];         // CUDA validation details
} EnhancedNeuralCompressionResult;

// Enhanced decompression result
typedef struct {
    bool success;                              // Whether decompression succeeded
    size_t decompressed_size;                  // Size of decompressed data
    EnhancedNeuralAlgorithm algorithm_detected; // Algorithm detected from data
    uint64_t processing_time_ns;               // Processing time in nanoseconds
    
    // CUDA compatibility verification
    bool cuda_compatibility_verified;         // Whether CUDA compatibility was verified
    bool integrity_check_passed;              // Whether integrity check passed
    float precision_accuracy;                 // Precision accuracy (0-1)
    
    char error_message[256];                   // Error message if failed
} EnhancedNeuralDecompressionResult;

// Bridge context for managing multiple LSTM contexts
typedef struct {
    // Enhanced LSTM contexts
    EnhancedMetalLSTMContext* primary_lstm_context;
    EnhancedMetalLSTMContext* fallback_lstm_context;
    
    // Small file handler
    SmallFileHandler* small_file_handler;
    
    // Configuration and validation
    EnhancedNeuralCompressionConfig config;
    CUDAValidationReport* last_validation_report;
    
    // Performance tracking
    size_t total_compressions;
    size_t successful_compressions;
    double total_processing_time_ms;
    size_t total_bytes_processed;
    
    // CUDA compatibility tracking
    size_t cuda_compatible_operations;
    size_t parameter_validation_failures;
    float max_observed_deviation;
    
    bool initialized;
    char last_error[256];
} EnhancedNeuralBridgeContext;

// Lifecycle management
bool enhanced_neural_bridge_init(const EnhancedNeuralCompressionConfig* config);
void enhanced_neural_bridge_shutdown(void);
bool enhanced_neural_bridge_is_ready(void);
EnhancedNeuralBridgeContext* enhanced_neural_bridge_get_context(void);

// Configuration management
EnhancedNeuralCompressionConfig* enhanced_neural_config_create_default(void);
EnhancedNeuralCompressionConfig* enhanced_neural_config_create_for_profile(const CUDAProfile* profile);
EnhancedNeuralCompressionConfig* enhanced_neural_config_create_cuda_strict(void);
void enhanced_neural_config_free(EnhancedNeuralCompressionConfig* config);

// Core compression functions (CUDA-compatible)
bool enhanced_neural_bridge_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const EnhancedNeuralCompressionConfig* config,
    EnhancedNeuralCompressionResult* result
);

bool enhanced_neural_bridge_decompress(
    const uint8_t* compressed_data,
    size_t compressed_size,
    uint8_t* output_data,
    size_t output_capacity,
    EnhancedNeuralDecompressionResult* result
);

// Algorithm-specific compression (for compatibility with legacy code)
bool enhanced_neural_bridge_metal_lstm_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const EnhancedNeuralCompressionConfig* config,
    EnhancedNeuralCompressionResult* result
);

bool enhanced_neural_bridge_small_file_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const EnhancedNeuralCompressionConfig* config,
    EnhancedNeuralCompressionResult* result
);

// Utility and analysis functions
size_t enhanced_neural_bridge_estimate_compressed_size(size_t input_size, 
                                                       const EnhancedNeuralCompressionConfig* config);
bool enhanced_neural_bridge_algorithm_available(EnhancedNeuralAlgorithm algorithm);
const char* enhanced_neural_bridge_algorithm_name(EnhancedNeuralAlgorithm algorithm);
size_t enhanced_neural_bridge_memory_requirements(const EnhancedNeuralCompressionConfig* config);

// CUDA compatibility and validation
bool enhanced_neural_bridge_validate_cuda_compatibility(const EnhancedNeuralCompressionConfig* config);
bool enhanced_neural_bridge_validate_parameters(const EnhancedNeuralCompressionConfig* config,
                                               CUDAValidationReport** report);
bool enhanced_neural_bridge_verify_mathematical_equivalence(const uint8_t* test_data,
                                                           size_t test_size,
                                                           const EnhancedNeuralCompressionConfig* config);

// Quality assessment and reporting
float enhanced_neural_bridge_assess_compression_quality(const EnhancedNeuralCompressionResult* result);
bool enhanced_neural_bridge_meets_quality_requirements(const EnhancedNeuralCompressionResult* result,
                                                       const EnhancedNeuralCompressionConfig* config);
void enhanced_neural_bridge_generate_quality_report(const EnhancedNeuralCompressionResult* result,
                                                   char* report_buffer,
                                                   size_t buffer_size);

// Performance monitoring and statistics
void enhanced_neural_bridge_print_statistics(void);
void enhanced_neural_bridge_reset_statistics(void);
bool enhanced_neural_bridge_export_performance_data(const char* filename);

// Algorithm selection and optimization
EnhancedNeuralAlgorithm enhanced_neural_bridge_select_optimal_algorithm(size_t input_size,
                                                                        const uint8_t* input_data,
                                                                        const EnhancedNeuralCompressionConfig* config);
bool enhanced_neural_bridge_optimize_for_file_size(size_t file_size,
                                                  EnhancedNeuralCompressionConfig* config);

// Legacy compatibility functions (for seamless transition)
bool enhanced_neural_bridge_lstm_compress_legacy(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const void* legacy_config,  // NeuralCompressionConfig*
    void* legacy_result        // NeuralCompressionResult*
);

// Error handling and diagnostics
typedef enum {
    ENHANCED_NEURAL_SUCCESS = 0,
    ENHANCED_NEURAL_ERROR_NOT_INITIALIZED,
    ENHANCED_NEURAL_ERROR_INVALID_CONFIG,
    ENHANCED_NEURAL_ERROR_CUDA_INCOMPATIBLE,
    ENHANCED_NEURAL_ERROR_VALIDATION_FAILED,
    ENHANCED_NEURAL_ERROR_MEMORY_ALLOCATION,
    ENHANCED_NEURAL_ERROR_LSTM_PROCESSING,
    ENHANCED_NEURAL_ERROR_QUALITY_TOO_LOW,
    ENHANCED_NEURAL_ERROR_BUFFER_TOO_SMALL,
    ENHANCED_NEURAL_ERROR_UNKNOWN_ALGORITHM
} EnhancedNeuralError;

const char* enhanced_neural_bridge_error_string(EnhancedNeuralError error);
EnhancedNeuralError enhanced_neural_bridge_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_BRIDGE_ENHANCED_H