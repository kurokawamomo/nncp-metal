#ifndef COREML_LOADER_H
#define COREML_LOADER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for Apple CoreML framework integration
#ifdef __OBJC__
@class MLModel;
@class MLModelDescription;
@class MLFeatureDescription;
@class MLDictionaryFeatureProvider;
@class MLPredictionOptions;
#else
typedef void MLModel;
typedef void MLModelDescription;
typedef void MLFeatureDescription;
typedef void MLDictionaryFeatureProvider;
typedef void MLPredictionOptions;
#endif

// Error codes for CoreML model operations
typedef enum {
    COREML_SUCCESS = 0,
    COREML_ERROR_INVALID_PARAM,
    COREML_ERROR_MEMORY_ALLOCATION,
    COREML_ERROR_MODEL_LOAD_FAILED,
    COREML_ERROR_MODEL_NOT_FOUND,
    COREML_ERROR_INVALID_MODEL_FORMAT,
    COREML_ERROR_UNSUPPORTED_ARCHITECTURE,
    COREML_ERROR_CACHE_FULL,
    COREML_ERROR_VALIDATION_FAILED,
    COREML_ERROR_CONVERSION_FAILED,
    COREML_ERROR_PREDICTION_FAILED,
    COREML_ERROR_FEATURE_MISMATCH
} CoreMLError;

// Supported model architectures
typedef enum {
    COREML_ARCH_UNKNOWN = 0,
    COREML_ARCH_TRANSFORMER,
    COREML_ARCH_LSTM,
    COREML_ARCH_GRU,
    COREML_ARCH_BERT,
    COREML_ARCH_GPT,
    COREML_ARCH_T5,
    COREML_ARCH_CUSTOM
} CoreMLArchitecture;

// Model input/output feature types
typedef enum {
    COREML_FEATURE_UNKNOWN = 0,
    COREML_FEATURE_INT32,
    COREML_FEATURE_FLOAT32,
    COREML_FEATURE_DOUBLE,
    COREML_FEATURE_STRING,
    COREML_FEATURE_ARRAY,
    COREML_FEATURE_DICTIONARY,
    COREML_FEATURE_SEQUENCE
} CoreMLFeatureType;

// Model feature description
typedef struct {
    char name[256];                    // Feature name
    CoreMLFeatureType type;            // Feature data type
    uint32_t* shape;                   // Feature shape dimensions
    uint32_t shape_count;              // Number of shape dimensions
    bool is_optional;                  // Whether feature is optional
    char description[512];             // Feature description
} CoreMLFeatureInfo;

// Model metadata
typedef struct {
    char name[256];                    // Model name
    char version[64];                  // Model version
    char author[256];                  // Model author
    char description[1024];            // Model description
    CoreMLArchitecture architecture;   // Model architecture type
    uint32_t input_count;              // Number of input features
    uint32_t output_count;             // Number of output features
    CoreMLFeatureInfo* inputs;         // Input feature descriptions
    CoreMLFeatureInfo* outputs;        // Output feature descriptions
    uint64_t model_size_bytes;         // Model file size
    uint32_t parameters_count;         // Number of model parameters
    bool supports_batch_prediction;    // Whether model supports batching
} CoreMLModelInfo;

// Model cache entry
typedef struct {
    char file_path[1024];              // Path to model file
    char cache_key[256];               // Unique cache key
    void* model;                       // Loaded CoreML model
    CoreMLModelInfo info;              // Model metadata
    uint64_t load_timestamp;           // When model was loaded
    uint32_t access_count;             // Number of times accessed
    uint64_t last_access_time;         // Last access timestamp
    bool is_valid;                     // Whether model is valid
    uint32_t reference_count;          // Reference counting
} CoreMLModelEntry;

// Model loader configuration
typedef struct {
    uint32_t max_cached_models;        // Maximum models in cache
    uint32_t max_memory_mb;            // Maximum cache memory usage
    uint32_t cache_ttl_seconds;        // Cache time-to-live
    bool enable_model_validation;      // Enable model validation
    bool enable_performance_logging;   // Enable performance logging
    bool auto_cleanup;                 // Enable automatic cache cleanup
    char models_directory[1024];       // Default models directory
    char temp_directory[1024];         // Temporary files directory
} CoreMLLoaderConfig;

// Performance statistics
typedef struct {
    uint32_t models_loaded;            // Total models loaded
    uint32_t cache_hits;               // Cache hit count
    uint32_t cache_misses;             // Cache miss count
    uint32_t validation_failures;      // Validation failure count
    uint64_t total_load_time_ns;       // Total loading time
    uint64_t total_prediction_time_ns; // Total prediction time
    uint32_t memory_usage_mb;          // Current memory usage
    uint32_t max_memory_usage_mb;      // Peak memory usage
} CoreMLLoaderStats;

// Main CoreML model loader
typedef struct {
    CoreMLLoaderConfig config;         // Loader configuration
    CoreMLLoaderStats stats;           // Performance statistics
    CoreMLModelEntry* cache;           // Model cache array
    uint32_t cache_count;              // Current cache entries
    uint32_t cache_capacity;           // Cache capacity
    bool is_initialized;               // Initialization state
    void* platform_context;           // Platform-specific context
} CoreMLLoader;

// Prediction input/output data
typedef struct {
    char feature_name[256];            // Feature name
    CoreMLFeatureType type;            // Data type
    void* data;                        // Feature data pointer
    uint32_t* shape;                   // Data shape
    uint32_t shape_count;              // Number of dimensions
    size_t data_size;                  // Data size in bytes
} CoreMLFeatureData;

typedef struct {
    CoreMLFeatureData* inputs;         // Input features array
    uint32_t input_count;              // Number of input features
    CoreMLFeatureData* outputs;        // Output features array
    uint32_t output_count;             // Number of output features
    bool is_valid;                     // Whether data is valid
} CoreMLPredictionData;

// Core API Functions

/**
 * Create and initialize CoreML model loader
 * @param loader Pointer to store created loader
 * @param config Loader configuration
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_create(CoreMLLoader** loader, const CoreMLLoaderConfig* config);

/**
 * Load CoreML model from file
 * @param loader CoreML loader instance
 * @param file_path Path to CoreML model file
 * @param model_entry Pointer to store loaded model entry
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_load_model(CoreMLLoader* loader, 
                                     const char* file_path, 
                                     CoreMLModelEntry** model_entry);

/**
 * Get cached model by file path
 * @param loader CoreML loader instance
 * @param file_path Path to model file
 * @param model_entry Pointer to store model entry
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_get_cached_model(CoreMLLoader* loader,
                                           const char* file_path,
                                           CoreMLModelEntry** model_entry);

/**
 * Release model reference (decrements reference count)
 * @param loader CoreML loader instance
 * @param model_entry Model entry to release
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_release_model(CoreMLLoader* loader, CoreMLModelEntry* model_entry);

/**
 * Validate loaded CoreML model
 * @param loader CoreML loader instance
 * @param model_entry Model entry to validate
 * @return COREML_SUCCESS if valid, error code on failure
 */
CoreMLError coreml_loader_validate_model(CoreMLLoader* loader, CoreMLModelEntry* model_entry);

/**
 * Make prediction using loaded model
 * @param loader CoreML loader instance
 * @param model_entry Model to use for prediction
 * @param prediction_data Input/output data for prediction
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_predict(CoreMLLoader* loader,
                                  CoreMLModelEntry* model_entry,
                                  CoreMLPredictionData* prediction_data);

// Configuration Functions

/**
 * Create default loader configuration
 * @param config Pointer to store default configuration
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_config_create_default(CoreMLLoaderConfig* config);

/**
 * Update loader configuration
 * @param loader CoreML loader instance
 * @param config New configuration
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_update_config(CoreMLLoader* loader, const CoreMLLoaderConfig* config);

// Cache Management Functions

/**
 * Clear model cache
 * @param loader CoreML loader instance
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_clear_cache(CoreMLLoader* loader);

/**
 * Optimize cache memory usage
 * @param loader CoreML loader instance
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_optimize_cache(CoreMLLoader* loader);

/**
 * Get cache memory usage
 * @param loader CoreML loader instance
 * @param memory_usage_mb Pointer to store memory usage in MB
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_get_memory_usage(CoreMLLoader* loader, uint32_t* memory_usage_mb);

// Model Discovery and Management

/**
 * List available models in directory
 * @param loader CoreML loader instance
 * @param directory Directory to scan for models
 * @param model_paths Array to store model file paths
 * @param max_models Maximum number of models to list
 * @param actual_count Pointer to store actual number of models found
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_list_models(CoreMLLoader* loader,
                                      const char* directory,
                                      char** model_paths,
                                      uint32_t max_models,
                                      uint32_t* actual_count);

/**
 * Get model information without loading
 * @param loader CoreML loader instance
 * @param file_path Path to model file
 * @param model_info Pointer to store model information
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_get_model_info(CoreMLLoader* loader,
                                         const char* file_path,
                                         CoreMLModelInfo* model_info);

// Conversion Utilities

/**
 * Convert PyTorch model to CoreML format
 * @param pytorch_model_path Path to PyTorch model file
 * @param coreml_output_path Output path for CoreML model
 * @param conversion_options Optional conversion parameters
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_convert_from_pytorch(const char* pytorch_model_path,
                                        const char* coreml_output_path,
                                        const char* conversion_options);

/**
 * Convert ONNX model to CoreML format
 * @param onnx_model_path Path to ONNX model file
 * @param coreml_output_path Output path for CoreML model
 * @param conversion_options Optional conversion parameters
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_convert_from_onnx(const char* onnx_model_path,
                                     const char* coreml_output_path,
                                     const char* conversion_options);

// Statistics and Monitoring

/**
 * Get loader performance statistics
 * @param loader CoreML loader instance
 * @param stats Pointer to store statistics
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_get_stats(CoreMLLoader* loader, CoreMLLoaderStats* stats);

/**
 * Reset performance statistics
 * @param loader CoreML loader instance
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_loader_reset_stats(CoreMLLoader* loader);

// Utility Functions

/**
 * Get error message string
 * @param error_code CoreML error code
 * @return Human-readable error message
 */
const char* coreml_get_error_string(CoreMLError error_code);

/**
 * Check if CoreML framework is available on current platform
 * @return true if CoreML is available, false otherwise
 */
bool coreml_is_available(void);

/**
 * Get CoreML framework version
 * @param version_buffer Buffer to store version string
 * @param buffer_size Size of version buffer
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_get_framework_version(char* version_buffer, size_t buffer_size);

// Memory Management

/**
 * Create prediction data structure
 * @param input_count Number of input features
 * @param output_count Number of output features
 * @param prediction_data Pointer to store created structure
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_prediction_data_create(uint32_t input_count,
                                          uint32_t output_count,
                                          CoreMLPredictionData** prediction_data);

/**
 * Destroy prediction data structure
 * @param prediction_data Prediction data to destroy
 */
void coreml_prediction_data_destroy(CoreMLPredictionData* prediction_data);

/**
 * Create feature data
 * @param feature_name Name of the feature
 * @param type Feature data type
 * @param shape Feature shape dimensions
 * @param shape_count Number of shape dimensions
 * @param feature_data Pointer to store created feature data
 * @return COREML_SUCCESS on success, error code on failure
 */
CoreMLError coreml_feature_data_create(const char* feature_name,
                                       CoreMLFeatureType type,
                                       const uint32_t* shape,
                                       uint32_t shape_count,
                                       CoreMLFeatureData** feature_data);

/**
 * Destroy feature data
 * @param feature_data Feature data to destroy
 */
void coreml_feature_data_destroy(CoreMLFeatureData* feature_data);

/**
 * Destroy CoreML loader
 * @param loader CoreML loader to destroy
 */
void coreml_loader_destroy(CoreMLLoader* loader);

#ifdef __cplusplus
}
#endif

#endif // COREML_LOADER_H