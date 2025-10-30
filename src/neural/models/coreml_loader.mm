#include "coreml_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __APPLE__
#include <CoreML/CoreML.h>
#include <Foundation/Foundation.h>
#define COREML_AVAILABLE 1
#else
#define COREML_AVAILABLE 0
#endif

// Internal helper functions
static CoreMLError create_model_entry(const char* file_path, CoreMLModelEntry** entry);
static CoreMLError load_model_metadata(const char* file_path, CoreMLModelInfo* info);
static CoreMLError add_to_cache(CoreMLLoader* loader, CoreMLModelEntry* entry);
static CoreMLError remove_from_cache(CoreMLLoader* loader, const char* file_path);
static CoreMLModelEntry* find_in_cache(CoreMLLoader* loader, const char* file_path);
static uint64_t get_timestamp_ns(void);
static bool is_file_exists(const char* file_path);
static uint64_t get_file_size(const char* file_path);
static CoreMLError cleanup_expired_cache_entries(CoreMLLoader* loader);
static const char* architecture_to_string(CoreMLArchitecture arch);
static CoreMLArchitecture detect_model_architecture(const char* file_path);

// Error messages
static const char* error_messages[] = {
    "Success",
    "Invalid parameter",
    "Memory allocation failed", 
    "Model load failed",
    "Model not found",
    "Invalid model format",
    "Unsupported architecture",
    "Cache full",
    "Validation failed",
    "Conversion failed",
    "Prediction failed",
    "Feature mismatch"
};

// Core API Implementation

CoreMLError coreml_loader_create(CoreMLLoader** loader, const CoreMLLoaderConfig* config) {
    if (!loader) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    *loader = (CoreMLLoader*)calloc(1, sizeof(CoreMLLoader));
    if (!*loader) {
        return COREML_ERROR_MEMORY_ALLOCATION;
    }
    
    // Set configuration
    if (config) {
        (*loader)->config = *config;
    } else {
        coreml_loader_config_create_default(&(*loader)->config);
    }
    
    // Initialize cache
    (*loader)->cache_capacity = (*loader)->config.max_cached_models;
    if ((*loader)->cache_capacity > 0) {
        (*loader)->cache = (CoreMLModelEntry*)calloc((*loader)->cache_capacity, 
                                                    sizeof(CoreMLModelEntry));
        if (!(*loader)->cache) {
            free(*loader);
            *loader = NULL;
            return COREML_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    (*loader)->cache_count = 0;
    (*loader)->is_initialized = true;
    
    // Reset statistics
    memset(&(*loader)->stats, 0, sizeof(CoreMLLoaderStats));
    
    return COREML_SUCCESS;
}

CoreMLError coreml_loader_load_model(CoreMLLoader* loader, 
                                     const char* file_path, 
                                     CoreMLModelEntry** model_entry) {
    if (!loader || !file_path || !model_entry) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    if (!loader->is_initialized) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    // Check if file exists
    if (!is_file_exists(file_path)) {
        return COREML_ERROR_MODEL_NOT_FOUND;
    }
    
    // Check cache first
    CoreMLModelEntry* cached = find_in_cache(loader, file_path);
    if (cached) {
        cached->reference_count++;
        cached->access_count++;
        cached->last_access_time = get_timestamp_ns();
        *model_entry = cached;
        loader->stats.cache_hits++;
        return COREML_SUCCESS;
    }
    
    // Create new model entry
    CoreMLError error = create_model_entry(file_path, model_entry);
    if (error != COREML_SUCCESS) {
        loader->stats.validation_failures++;
        return error;
    }
    
#if COREML_AVAILABLE
    // Load CoreML model on Apple platforms
    @autoreleasepool {
        NSString* modelPath = [NSString stringWithUTF8String:file_path];
        NSURL* modelURL = [NSURL fileURLWithPath:modelPath];
        
        NSError* nsError = nil;
        MLModel* mlModel = [MLModel modelWithContentsOfURL:modelURL error:&nsError];
        
        if (!mlModel || nsError) {
            free(*model_entry);
            *model_entry = NULL;
            loader->stats.validation_failures++;
            return COREML_ERROR_MODEL_LOAD_FAILED;
        }
        
        (*model_entry)->model = (__bridge void*)mlModel;
        
        // Extract model metadata
        MLModelDescription* description = mlModel.modelDescription;
        if (description) {
            // Copy model metadata
            if (description.metadata[@"MLModelDescriptionKey"]) {
                NSString* desc = description.metadata[@"MLModelDescriptionKey"];
                strncpy((*model_entry)->info.description, [desc UTF8String], 
                       sizeof((*model_entry)->info.description) - 1);
            }
            
            if (description.metadata[@"MLModelVersionStringKey"]) {
                NSString* version = description.metadata[@"MLModelVersionStringKey"];
                strncpy((*model_entry)->info.version, [version UTF8String], 
                       sizeof((*model_entry)->info.version) - 1);
            }
            
            if (description.metadata[@"MLModelAuthorKey"]) {
                NSString* author = description.metadata[@"MLModelAuthorKey"];
                strncpy((*model_entry)->info.author, [author UTF8String], 
                       sizeof((*model_entry)->info.author) - 1);
            }
            
            // Extract input features
            (*model_entry)->info.input_count = (uint32_t)[description.inputDescriptionsByName count];
            if ((*model_entry)->info.input_count > 0) {
                (*model_entry)->info.inputs = (CoreMLFeatureInfo*)calloc((*model_entry)->info.input_count, 
                                                                        sizeof(CoreMLFeatureInfo));
                
                uint32_t idx = 0;
                for (NSString* inputName in description.inputDescriptionsByName) {
                    MLFeatureDescription* feature = description.inputDescriptionsByName[inputName];
                    strncpy((*model_entry)->info.inputs[idx].name, [inputName UTF8String], 
                           sizeof((*model_entry)->info.inputs[idx].name) - 1);
                    (*model_entry)->info.inputs[idx].is_optional = feature.optional;
                    idx++;
                }
            }
            
            // Extract output features
            (*model_entry)->info.output_count = (uint32_t)[description.outputDescriptionsByName count];
            if ((*model_entry)->info.output_count > 0) {
                (*model_entry)->info.outputs = (CoreMLFeatureInfo*)calloc((*model_entry)->info.output_count, 
                                                                         sizeof(CoreMLFeatureInfo));
                
                uint32_t idx = 0;
                for (NSString* outputName in description.outputDescriptionsByName) {
                    MLFeatureDescription* feature = description.outputDescriptionsByName[outputName];
                    strncpy((*model_entry)->info.outputs[idx].name, [outputName UTF8String], 
                           sizeof((*model_entry)->info.outputs[idx].name) - 1);
                    (*model_entry)->info.outputs[idx].is_optional = feature.optional;
                    idx++;
                }
            }
        }
    }
#else
    // Fallback for non-Apple platforms - load metadata only
    error = load_model_metadata(file_path, &(*model_entry)->info);
    if (error != COREML_SUCCESS) {
        free(*model_entry);
        *model_entry = NULL;
        return error;
    }
    (*model_entry)->model = NULL; // Cannot load actual model on non-Apple platforms
#endif
    
    // Validate model if enabled
    if (loader->config.enable_model_validation) {
        error = coreml_loader_validate_model(loader, *model_entry);
        if (error != COREML_SUCCESS) {
            if ((*model_entry)->model) {
#if COREML_AVAILABLE
                // Model cleanup handled by ARC
#endif
            }
            free((*model_entry)->info.inputs);
            free((*model_entry)->info.outputs);
            free(*model_entry);
            *model_entry = NULL;
            return error;
        }
    }
    
    // Add to cache
    add_to_cache(loader, *model_entry);
    
    // Update statistics
    uint64_t load_time = get_timestamp_ns() - start_time;
    loader->stats.models_loaded++;
    loader->stats.cache_misses++;
    loader->stats.total_load_time_ns += load_time;
    
    return COREML_SUCCESS;
}

CoreMLError coreml_loader_get_cached_model(CoreMLLoader* loader,
                                           const char* file_path,
                                           CoreMLModelEntry** model_entry) {
    if (!loader || !file_path || !model_entry) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    CoreMLModelEntry* cached = find_in_cache(loader, file_path);
    if (cached) {
        cached->reference_count++;
        cached->access_count++;
        cached->last_access_time = get_timestamp_ns();
        *model_entry = cached;
        loader->stats.cache_hits++;
        return COREML_SUCCESS;
    }
    
    *model_entry = NULL;
    loader->stats.cache_misses++;
    return COREML_ERROR_MODEL_NOT_FOUND;
}

CoreMLError coreml_loader_release_model(CoreMLLoader* loader, CoreMLModelEntry* model_entry) {
    if (!loader || !model_entry) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    if (model_entry->reference_count > 0) {
        model_entry->reference_count--;
    }
    
    // Cleanup if auto cleanup is enabled and no references
    if (loader->config.auto_cleanup && model_entry->reference_count == 0) {
        // Check if model is old enough to remove
        uint64_t current_time = get_timestamp_ns();
        uint64_t ttl_ns = (uint64_t)loader->config.cache_ttl_seconds * 1000000000ULL;
        
        if (current_time - model_entry->last_access_time > ttl_ns) {
            remove_from_cache(loader, model_entry->file_path);
        }
    }
    
    return COREML_SUCCESS;
}

CoreMLError coreml_loader_validate_model(CoreMLLoader* loader, CoreMLModelEntry* model_entry) {
    if (!loader || !model_entry) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    // Basic validation checks
    if (!model_entry->is_valid) {
        return COREML_ERROR_VALIDATION_FAILED;
    }
    
    // Check if file still exists
    if (!is_file_exists(model_entry->file_path)) {
        model_entry->is_valid = false;
        return COREML_ERROR_MODEL_NOT_FOUND;
    }
    
    // Validate architecture
    if (model_entry->info.architecture == COREML_ARCH_UNKNOWN) {
        return COREML_ERROR_UNSUPPORTED_ARCHITECTURE;
    }
    
    // Validate input/output features
    if (model_entry->info.input_count == 0 || model_entry->info.output_count == 0) {
        return COREML_ERROR_INVALID_MODEL_FORMAT;
    }
    
    return COREML_SUCCESS;
}

CoreMLError coreml_loader_predict(CoreMLLoader* loader,
                                  CoreMLModelEntry* model_entry,
                                  CoreMLPredictionData* prediction_data) {
    if (!loader || !model_entry || !prediction_data) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
#if COREML_AVAILABLE
    if (!model_entry->model) {
        return COREML_ERROR_MODEL_LOAD_FAILED;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    @autoreleasepool {
        MLModel* mlModel = (__bridge MLModel*)model_entry->model;
        
        // Create input feature provider
        NSMutableDictionary* inputDict = [[NSMutableDictionary alloc] init];
        
        for (uint32_t i = 0; i < prediction_data->input_count; i++) {
            CoreMLFeatureData* input = &prediction_data->inputs[i];
            NSString* featureName = [NSString stringWithUTF8String:input->feature_name];
            
            // Convert input data based on type
            id featureValue = nil;
            switch (input->type) {
                case COREML_FEATURE_FLOAT32: {
                    float* floatData = (float*)input->data;
                    NSMutableArray* array = [[NSMutableArray alloc] init];
                    size_t count = input->data_size / sizeof(float);
                    for (size_t j = 0; j < count; j++) {
                        [array addObject:@(floatData[j])];
                    }
                    featureValue = array;
                    break;
                }
                case COREML_FEATURE_INT32: {
                    int32_t* intData = (int32_t*)input->data;
                    NSMutableArray* array = [[NSMutableArray alloc] init];
                    size_t count = input->data_size / sizeof(int32_t);
                    for (size_t j = 0; j < count; j++) {
                        [array addObject:@(intData[j])];
                    }
                    featureValue = array;
                    break;
                }
                case COREML_FEATURE_STRING:
                    featureValue = [NSString stringWithUTF8String:(char*)input->data];
                    break;
                default:
                    return COREML_ERROR_FEATURE_MISMATCH;
            }
            
            if (featureValue) {
                inputDict[featureName] = featureValue;
            }
        }
        
        MLDictionaryFeatureProvider* inputProvider = 
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict error:nil];
        
        if (!inputProvider) {
            return COREML_ERROR_FEATURE_MISMATCH;
        }
        
        // Make prediction
        NSError* error = nil;
        id<MLFeatureProvider> output = [mlModel predictionFromFeatures:inputProvider error:&error];
        
        if (!output || error) {
            return COREML_ERROR_PREDICTION_FAILED;
        }
        
        // Extract output features
        NSSet* outputNames = output.featureNames;
        uint32_t outputIndex = 0;
        
        for (NSString* outputName in outputNames) {
            if (outputIndex >= prediction_data->output_count) {
                break;
            }
            
            MLFeatureValue* featureValue = [output featureValueForName:outputName];
            CoreMLFeatureData* outputData = &prediction_data->outputs[outputIndex];
            
            strncpy(outputData->feature_name, [outputName UTF8String], 
                   sizeof(outputData->feature_name) - 1);
            
            // Convert output data based on type
            if (featureValue.type == MLFeatureTypeMultiArray) {
                MLMultiArray* multiArray = featureValue.multiArrayValue;
                outputData->type = COREML_FEATURE_FLOAT32;
                
                // Allocate output data
                NSUInteger count = multiArray.count;
                outputData->data_size = count * sizeof(float);
                outputData->data = malloc(outputData->data_size);
                
                if (outputData->data) {
                    float* floatOutput = (float*)outputData->data;
                    for (NSUInteger i = 0; i < count; i++) {
                        floatOutput[i] = [[multiArray objectAtIndexedSubscript:i] floatValue];
                    }
                }
            }
            
            outputIndex++;
        }
    }
    
    // Update statistics
    uint64_t prediction_time = get_timestamp_ns() - start_time;
    loader->stats.total_prediction_time_ns += prediction_time;
    
    return COREML_SUCCESS;
#else
    // Cannot make predictions on non-Apple platforms
    return COREML_ERROR_PREDICTION_FAILED;
#endif
}

// Configuration Functions

CoreMLError coreml_loader_config_create_default(CoreMLLoaderConfig* config) {
    if (!config) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    config->max_cached_models = 10;
    config->max_memory_mb = 512;
    config->cache_ttl_seconds = 3600; // 1 hour
    config->enable_model_validation = true;
    config->enable_performance_logging = true;
    config->auto_cleanup = true;
    
    strncpy(config->models_directory, "models", sizeof(config->models_directory) - 1);
    strncpy(config->temp_directory, "/tmp", sizeof(config->temp_directory) - 1);
    
    return COREML_SUCCESS;
}

CoreMLError coreml_loader_update_config(CoreMLLoader* loader, const CoreMLLoaderConfig* config) {
    if (!loader || !config) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    // Update configuration
    loader->config = *config;
    
    // Resize cache if needed
    if (config->max_cached_models != loader->cache_capacity) {
        // Clear existing cache
        coreml_loader_clear_cache(loader);
        
        // Reallocate cache
        free(loader->cache);
        loader->cache_capacity = config->max_cached_models;
        
        if (loader->cache_capacity > 0) {
            loader->cache = (CoreMLModelEntry*)calloc(loader->cache_capacity, 
                                                     sizeof(CoreMLModelEntry));
            if (!loader->cache) {
                loader->cache_capacity = 0;
                return COREML_ERROR_MEMORY_ALLOCATION;
            }
        } else {
            loader->cache = NULL;
        }
        
        loader->cache_count = 0;
    }
    
    return COREML_SUCCESS;
}

// Cache Management Functions

CoreMLError coreml_loader_clear_cache(CoreMLLoader* loader) {
    if (!loader) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    for (uint32_t i = 0; i < loader->cache_count; i++) {
        CoreMLModelEntry* entry = &loader->cache[i];
        
        if (entry->model) {
#if COREML_AVAILABLE
            // Model cleanup handled by ARC
#endif
            entry->model = NULL;
        }
        
        free(entry->info.inputs);
        free(entry->info.outputs);
        
        memset(entry, 0, sizeof(CoreMLModelEntry));
    }
    
    loader->cache_count = 0;
    return COREML_SUCCESS;
}

CoreMLError coreml_loader_optimize_cache(CoreMLLoader* loader) {
    if (!loader) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    // Remove expired entries
    cleanup_expired_cache_entries(loader);
    
    // Compact cache array
    uint32_t write_index = 0;
    for (uint32_t read_index = 0; read_index < loader->cache_count; read_index++) {
        if (loader->cache[read_index].is_valid && loader->cache[read_index].model) {
            if (write_index != read_index) {
                loader->cache[write_index] = loader->cache[read_index];
                memset(&loader->cache[read_index], 0, sizeof(CoreMLModelEntry));
            }
            write_index++;
        }
    }
    
    loader->cache_count = write_index;
    return COREML_SUCCESS;
}

CoreMLError coreml_loader_get_memory_usage(CoreMLLoader* loader, uint32_t* memory_usage_mb) {
    if (!loader || !memory_usage_mb) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    uint64_t total_bytes = sizeof(CoreMLLoader);
    
    // Add cache memory
    total_bytes += loader->cache_capacity * sizeof(CoreMLModelEntry);
    
    for (uint32_t i = 0; i < loader->cache_count; i++) {
        total_bytes += loader->cache[i].info.model_size_bytes;
        total_bytes += loader->cache[i].info.input_count * sizeof(CoreMLFeatureInfo);
        total_bytes += loader->cache[i].info.output_count * sizeof(CoreMLFeatureInfo);
    }
    
    *memory_usage_mb = (uint32_t)((total_bytes + 1024 * 1024 - 1) / (1024 * 1024));
    
    loader->stats.memory_usage_mb = *memory_usage_mb;
    if (*memory_usage_mb > loader->stats.max_memory_usage_mb) {
        loader->stats.max_memory_usage_mb = *memory_usage_mb;
    }
    
    return COREML_SUCCESS;
}

// Model Discovery and Management

CoreMLError coreml_loader_list_models(CoreMLLoader* loader,
                                      const char* directory,
                                      char** model_paths,
                                      uint32_t max_models,
                                      uint32_t* actual_count) {
    if (!loader || !directory || !model_paths || !actual_count) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    *actual_count = 0;
    
    // Simple implementation - would need platform-specific directory enumeration
    // This is a placeholder implementation
    const char* extensions[] = {".mlmodel", ".mlmodelc", NULL};
    
    // For demonstration, return success with 0 models
    // A full implementation would scan the directory for CoreML files
    
    return COREML_SUCCESS;
}

CoreMLError coreml_loader_get_model_info(CoreMLLoader* loader,
                                         const char* file_path,
                                         CoreMLModelInfo* model_info) {
    if (!loader || !file_path || !model_info) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    if (!is_file_exists(file_path)) {
        return COREML_ERROR_MODEL_NOT_FOUND;
    }
    
    return load_model_metadata(file_path, model_info);
}

// Conversion Utilities

CoreMLError coreml_convert_from_pytorch(const char* pytorch_model_path,
                                        const char* coreml_output_path,
                                        const char* conversion_options) {
    // Placeholder implementation
    // Real implementation would need Python integration or conversion library
    (void)pytorch_model_path;
    (void)coreml_output_path;
    (void)conversion_options;
    
    return COREML_ERROR_CONVERSION_FAILED;
}

CoreMLError coreml_convert_from_onnx(const char* onnx_model_path,
                                     const char* coreml_output_path,
                                     const char* conversion_options) {
    // Placeholder implementation
    (void)onnx_model_path;
    (void)coreml_output_path;
    (void)conversion_options;
    
    return COREML_ERROR_CONVERSION_FAILED;
}

// Statistics and Monitoring

CoreMLError coreml_loader_get_stats(CoreMLLoader* loader, CoreMLLoaderStats* stats) {
    if (!loader || !stats) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    *stats = loader->stats;
    return COREML_SUCCESS;
}

CoreMLError coreml_loader_reset_stats(CoreMLLoader* loader) {
    if (!loader) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    memset(&loader->stats, 0, sizeof(CoreMLLoaderStats));
    return COREML_SUCCESS;
}

// Utility Functions

const char* coreml_get_error_string(CoreMLError error_code) {
    if (error_code < 0 || error_code >= sizeof(error_messages) / sizeof(error_messages[0])) {
        return "Unknown error";
    }
    return error_messages[error_code];
}

bool coreml_is_available(void) {
#if COREML_AVAILABLE
    return true;
#else
    return false;
#endif
}

CoreMLError coreml_get_framework_version(char* version_buffer, size_t buffer_size) {
    if (!version_buffer || buffer_size == 0) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
#if COREML_AVAILABLE
    strncpy(version_buffer, "CoreML 1.0+", buffer_size - 1);
    version_buffer[buffer_size - 1] = '\0';
#else
    strncpy(version_buffer, "CoreML Not Available", buffer_size - 1);
    version_buffer[buffer_size - 1] = '\0';
#endif
    
    return COREML_SUCCESS;
}

// Memory Management

CoreMLError coreml_prediction_data_create(uint32_t input_count,
                                          uint32_t output_count,
                                          CoreMLPredictionData** prediction_data) {
    if (!prediction_data || (input_count == 0 && output_count == 0)) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    *prediction_data = (CoreMLPredictionData*)calloc(1, sizeof(CoreMLPredictionData));
    if (!*prediction_data) {
        return COREML_ERROR_MEMORY_ALLOCATION;
    }
    
    if (input_count > 0) {
        (*prediction_data)->inputs = (CoreMLFeatureData*)calloc(input_count, sizeof(CoreMLFeatureData));
        if (!(*prediction_data)->inputs) {
            free(*prediction_data);
            *prediction_data = NULL;
            return COREML_ERROR_MEMORY_ALLOCATION;
        }
        (*prediction_data)->input_count = input_count;
    }
    
    if (output_count > 0) {
        (*prediction_data)->outputs = (CoreMLFeatureData*)calloc(output_count, sizeof(CoreMLFeatureData));
        if (!(*prediction_data)->outputs) {
            free((*prediction_data)->inputs);
            free(*prediction_data);
            *prediction_data = NULL;
            return COREML_ERROR_MEMORY_ALLOCATION;
        }
        (*prediction_data)->output_count = output_count;
    }
    
    (*prediction_data)->is_valid = true;
    return COREML_SUCCESS;
}

void coreml_prediction_data_destroy(CoreMLPredictionData* prediction_data) {
    if (!prediction_data) {
        return;
    }
    
    // Free input feature data
    for (uint32_t i = 0; i < prediction_data->input_count; i++) {
        free(prediction_data->inputs[i].data);
        free(prediction_data->inputs[i].shape);
    }
    free(prediction_data->inputs);
    
    // Free output feature data
    for (uint32_t i = 0; i < prediction_data->output_count; i++) {
        free(prediction_data->outputs[i].data);
        free(prediction_data->outputs[i].shape);
    }
    free(prediction_data->outputs);
    
    free(prediction_data);
}

CoreMLError coreml_feature_data_create(const char* feature_name,
                                       CoreMLFeatureType type,
                                       const uint32_t* shape,
                                       uint32_t shape_count,
                                       CoreMLFeatureData** feature_data) {
    if (!feature_name || !feature_data) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    *feature_data = (CoreMLFeatureData*)calloc(1, sizeof(CoreMLFeatureData));
    if (!*feature_data) {
        return COREML_ERROR_MEMORY_ALLOCATION;
    }
    
    strncpy((*feature_data)->feature_name, feature_name, sizeof((*feature_data)->feature_name) - 1);
    (*feature_data)->type = type;
    
    if (shape_count > 0 && shape) {
        (*feature_data)->shape = (uint32_t*)malloc(shape_count * sizeof(uint32_t));
        if (!(*feature_data)->shape) {
            free(*feature_data);
            *feature_data = NULL;
            return COREML_ERROR_MEMORY_ALLOCATION;
        }
        memcpy((*feature_data)->shape, shape, shape_count * sizeof(uint32_t));
        (*feature_data)->shape_count = shape_count;
    }
    
    return COREML_SUCCESS;
}

void coreml_feature_data_destroy(CoreMLFeatureData* feature_data) {
    if (!feature_data) {
        return;
    }
    
    free(feature_data->data);
    free(feature_data->shape);
    free(feature_data);
}

void coreml_loader_destroy(CoreMLLoader* loader) {
    if (!loader) {
        return;
    }
    
    coreml_loader_clear_cache(loader);
    free(loader->cache);
    free(loader);
}

// Internal helper function implementations

static CoreMLError create_model_entry(const char* file_path, CoreMLModelEntry** entry) {
    *entry = (CoreMLModelEntry*)calloc(1, sizeof(CoreMLModelEntry));
    if (!*entry) {
        return COREML_ERROR_MEMORY_ALLOCATION;
    }
    
    strncpy((*entry)->file_path, file_path, sizeof((*entry)->file_path) - 1);
    (*entry)->load_timestamp = get_timestamp_ns();
    (*entry)->last_access_time = (*entry)->load_timestamp;
    (*entry)->reference_count = 1;
    (*entry)->access_count = 1;
    (*entry)->is_valid = true;
    
    // Initialize model info
    memset(&(*entry)->info, 0, sizeof(CoreMLModelInfo));
    strncpy((*entry)->info.name, "Unknown Model", sizeof((*entry)->info.name) - 1);
    strncpy((*entry)->info.version, "1.0", sizeof((*entry)->info.version) - 1);
    (*entry)->info.architecture = detect_model_architecture(file_path);
    (*entry)->info.model_size_bytes = get_file_size(file_path);
    
    return COREML_SUCCESS;
}

static CoreMLError load_model_metadata(const char* file_path, CoreMLModelInfo* info) {
    if (!file_path || !info) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    // Initialize with defaults
    memset(info, 0, sizeof(CoreMLModelInfo));
    strncpy(info->name, "Model", sizeof(info->name) - 1);
    strncpy(info->version, "1.0", sizeof(info->version) - 1);
    info->architecture = detect_model_architecture(file_path);
    info->model_size_bytes = get_file_size(file_path);
    
    return COREML_SUCCESS;
}

static CoreMLError add_to_cache(CoreMLLoader* loader, CoreMLModelEntry* entry) {
    if (!loader || !entry || !loader->cache) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    // Check if cache is full
    if (loader->cache_count >= loader->cache_capacity) {
        // Find least recently used entry
        uint32_t lru_index = 0;
        uint64_t oldest_time = loader->cache[0].last_access_time;
        
        for (uint32_t i = 1; i < loader->cache_count; i++) {
            if (loader->cache[i].last_access_time < oldest_time &&
                loader->cache[i].reference_count == 0) {
                oldest_time = loader->cache[i].last_access_time;
                lru_index = i;
            }
        }
        
        // Remove LRU entry
        CoreMLModelEntry* lru_entry = &loader->cache[lru_index];
        if (lru_entry->model) {
#if COREML_AVAILABLE
            CFBridgingRelease(lru_entry->model);
#endif
        }
        free(lru_entry->info.inputs);
        free(lru_entry->info.outputs);
        
        // Replace with new entry
        loader->cache[lru_index] = *entry;
    } else {
        // Add to end of cache
        loader->cache[loader->cache_count] = *entry;
        loader->cache_count++;
    }
    
    return COREML_SUCCESS;
}

static CoreMLError remove_from_cache(CoreMLLoader* loader, const char* file_path) {
    if (!loader || !file_path) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    for (uint32_t i = 0; i < loader->cache_count; i++) {
        if (strcmp(loader->cache[i].file_path, file_path) == 0) {
            CoreMLModelEntry* entry = &loader->cache[i];
            
            if (entry->model) {
#if COREML_AVAILABLE
                // Model cleanup handled by ARC
#endif
            }
            free(entry->info.inputs);
            free(entry->info.outputs);
            
            // Shift remaining entries
            for (uint32_t j = i; j < loader->cache_count - 1; j++) {
                loader->cache[j] = loader->cache[j + 1];
            }
            
            loader->cache_count--;
            memset(&loader->cache[loader->cache_count], 0, sizeof(CoreMLModelEntry));
            
            return COREML_SUCCESS;
        }
    }
    
    return COREML_ERROR_MODEL_NOT_FOUND;
}

static CoreMLModelEntry* find_in_cache(CoreMLLoader* loader, const char* file_path) {
    if (!loader || !file_path) {
        return NULL;
    }
    
    for (uint32_t i = 0; i < loader->cache_count; i++) {
        if (strcmp(loader->cache[i].file_path, file_path) == 0 &&
            loader->cache[i].is_valid) {
            return &loader->cache[i];
        }
    }
    
    return NULL;
}

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

static bool is_file_exists(const char* file_path) {
    return access(file_path, F_OK) == 0;
}

static uint64_t get_file_size(const char* file_path) {
    struct stat st;
    if (stat(file_path, &st) == 0) {
        return (uint64_t)st.st_size;
    }
    return 0;
}

static CoreMLError cleanup_expired_cache_entries(CoreMLLoader* loader) {
    if (!loader) {
        return COREML_ERROR_INVALID_PARAM;
    }
    
    uint64_t current_time = get_timestamp_ns();
    uint64_t ttl_ns = (uint64_t)loader->config.cache_ttl_seconds * 1000000000ULL;
    
    uint32_t write_index = 0;
    for (uint32_t read_index = 0; read_index < loader->cache_count; read_index++) {
        CoreMLModelEntry* entry = &loader->cache[read_index];
        
        bool should_keep = entry->is_valid && 
                          entry->reference_count > 0 ||
                          (current_time - entry->last_access_time) < ttl_ns;
        
        if (should_keep) {
            if (write_index != read_index) {
                loader->cache[write_index] = loader->cache[read_index];
            }
            write_index++;
        } else {
            // Clean up expired entry
            if (entry->model) {
#if COREML_AVAILABLE
                // Model cleanup handled by ARC
#endif
            }
            free(entry->info.inputs);
            free(entry->info.outputs);
        }
    }
    
    // Clear remaining slots
    for (uint32_t i = write_index; i < loader->cache_count; i++) {
        memset(&loader->cache[i], 0, sizeof(CoreMLModelEntry));
    }
    
    loader->cache_count = write_index;
    return COREML_SUCCESS;
}

static const char* architecture_to_string(CoreMLArchitecture arch) {
    switch (arch) {
        case COREML_ARCH_TRANSFORMER: return "Transformer";
        case COREML_ARCH_LSTM: return "LSTM";
        case COREML_ARCH_GRU: return "GRU";
        case COREML_ARCH_BERT: return "BERT";
        case COREML_ARCH_GPT: return "GPT";
        case COREML_ARCH_T5: return "T5";
        case COREML_ARCH_CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

static CoreMLArchitecture detect_model_architecture(const char* file_path) {
    if (!file_path) {
        return COREML_ARCH_UNKNOWN;
    }
    
    // Simple heuristic based on filename
    if (strstr(file_path, "transformer") || strstr(file_path, "Transformer")) {
        return COREML_ARCH_TRANSFORMER;
    }
    if (strstr(file_path, "lstm") || strstr(file_path, "LSTM")) {
        return COREML_ARCH_LSTM;
    }
    if (strstr(file_path, "bert") || strstr(file_path, "BERT")) {
        return COREML_ARCH_BERT;
    }
    if (strstr(file_path, "gpt") || strstr(file_path, "GPT")) {
        return COREML_ARCH_GPT;
    }
    if (strstr(file_path, "t5") || strstr(file_path, "T5")) {
        return COREML_ARCH_T5;
    }
    if (strstr(file_path, "gru") || strstr(file_path, "GRU")) {
        return COREML_ARCH_GRU;
    }
    
    return COREML_ARCH_UNKNOWN;
}