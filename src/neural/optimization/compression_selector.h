#ifndef COMPRESSION_SELECTOR_H
#define COMPRESSION_SELECTOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __OBJC__
@class MPSGraph;
@class MPSGraphTensor;
@class MPSGraphExecutable;
@class MPSCommandBuffer;
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLBuffer;
#else
typedef void MPSGraph;
typedef void MPSGraphTensor;
typedef void MPSGraphExecutable;
typedef void MPSCommandBuffer;
typedef void* MTLDevice_t;
typedef void* MTLCommandQueue_t;
typedef void* MTLBuffer_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for compression selector operations
typedef enum {
    COMPRESSION_SELECTOR_SUCCESS = 0,
    COMPRESSION_SELECTOR_ERROR_INVALID_PARAM,
    COMPRESSION_SELECTOR_ERROR_MEMORY_ALLOCATION,
    COMPRESSION_SELECTOR_ERROR_DEVICE_NOT_FOUND,
    COMPRESSION_SELECTOR_ERROR_COMPUTE_FAILED,
    COMPRESSION_SELECTOR_ERROR_ANALYSIS_FAILED,
    COMPRESSION_SELECTOR_ERROR_PREDICTION_FAILED,
    COMPRESSION_SELECTOR_ERROR_MODEL_NOT_LOADED,
    COMPRESSION_SELECTOR_ERROR_INSUFFICIENT_DATA,
    COMPRESSION_SELECTOR_ERROR_FEATURE_EXTRACTION_FAILED,
    COMPRESSION_SELECTOR_ERROR_ALGORITHM_NOT_SUPPORTED,
    COMPRESSION_SELECTOR_ERROR_OPTIMIZATION_FAILED
} CompressionSelectorError;

// Available compression algorithms
typedef enum {
    COMPRESSION_ALGORITHM_NEURAL_QUANTIZATION = 0,
    COMPRESSION_ALGORITHM_RLE_LOSSLESS,
    COMPRESSION_ALGORITHM_NEURAL_LOSSLESS,
    COMPRESSION_ALGORITHM_TRANSFORMER_COMPRESSION,
    COMPRESSION_ALGORITHM_LSTM_COMPRESSION,
    COMPRESSION_ALGORITHM_HYBRID_NEURAL,
    COMPRESSION_ALGORITHM_ATTENTION_BASED,
    COMPRESSION_ALGORITHM_ADAPTIVE_QUANTIZATION,
    COMPRESSION_ALGORITHM_ENTROPY_CODING,
    COMPRESSION_ALGORITHM_RESIDUAL_COMPRESSION,
    COMPRESSION_ALGORITHM_HIERARCHICAL_COMPRESSION,
    COMPRESSION_ALGORITHM_AUTO_ENSEMBLE
} CompressionAlgorithm;

// Data type classifications
typedef enum {
    DATA_TYPE_TEXT = 0,
    DATA_TYPE_IMAGE,
    DATA_TYPE_AUDIO,
    DATA_TYPE_VIDEO,
    DATA_TYPE_BINARY,
    DATA_TYPE_TIME_SERIES,
    DATA_TYPE_STRUCTURED,
    DATA_TYPE_SPARSE,
    DATA_TYPE_DENSE,
    DATA_TYPE_SEQUENTIAL,
    DATA_TYPE_MIXED,
    DATA_TYPE_UNKNOWN
} DataTypeClassification;

// Compression objectives
typedef enum {
    COMPRESSION_OBJECTIVE_SIZE = 0,            // Minimize size
    COMPRESSION_OBJECTIVE_SPEED,               // Maximize speed
    COMPRESSION_OBJECTIVE_QUALITY,             // Maximize quality
    COMPRESSION_OBJECTIVE_BALANCED,            // Balance all factors
    COMPRESSION_OBJECTIVE_ENERGY_EFFICIENT,   // Minimize energy consumption
    COMPRESSION_OBJECTIVE_MEMORY_EFFICIENT,   // Minimize memory usage
    COMPRESSION_OBJECTIVE_CUSTOM              // Custom weighted objective
} CompressionObjective;

// Data characteristics for analysis
typedef struct {
    // Basic statistics
    uint64_t data_size;                       // Size of data in bytes
    uint32_t sample_count;                    // Number of samples
    float entropy;                            // Shannon entropy
    float compression_ratio_estimate;         // Estimated compression ratio
    
    // Statistical properties
    float mean;                              // Data mean
    float variance;                          // Data variance
    float skewness;                          // Distribution skewness
    float kurtosis;                          // Distribution kurtosis
    float sparsity;                          // Sparsity ratio (0-1)
    
    // Frequency domain properties
    float spectral_entropy;                  // Spectral entropy
    float dominant_frequency;                // Dominant frequency component
    float frequency_spread;                  // Frequency domain spread
    
    // Temporal properties (for sequences)
    float autocorrelation;                   // Autocorrelation strength
    float trend_strength;                    // Linear trend strength
    float seasonality_strength;              // Seasonal pattern strength
    float pattern_repetition;                // Pattern repetition measure
    
    // Spatial properties (for images/2D data)
    float spatial_correlation;              // Spatial correlation
    float edge_density;                      // Edge density measure
    float texture_complexity;               // Texture complexity
    
    // Information theoretical measures
    float mutual_information;                // Mutual information
    float conditional_entropy;               // Conditional entropy
    float information_content;               // Information content
    
    // Redundancy measures
    float redundancy_ratio;                  // Data redundancy ratio
    float predictability;                    // Predictability measure
    float compressibility_score;             // Overall compressibility
    
    // Data type classification confidence
    DataTypeClassification primary_type;     // Primary data type
    DataTypeClassification secondary_type;   // Secondary data type
    float type_confidence;                   // Classification confidence
} DataCharacteristics;

// Performance prediction for each algorithm
typedef struct {
    CompressionAlgorithm algorithm;          // Algorithm identifier
    float predicted_ratio;                   // Predicted compression ratio
    float predicted_speed;                   // Predicted compression speed (MB/s)
    float predicted_quality;                 // Predicted quality score (0-1)
    float energy_consumption;                // Predicted energy consumption
    float memory_usage;                      // Predicted memory usage (MB)
    float gpu_utilization;                   // Predicted GPU utilization (%)
    float confidence_score;                  // Prediction confidence (0-1)
    uint32_t estimated_time_ms;              // Estimated compression time
    bool is_supported;                       // Whether algorithm is supported
    float suitability_score;                 // Overall suitability score
} AlgorithmPerformancePrediction;

// Compression selection configuration
typedef struct {
    // Selection criteria
    CompressionObjective primary_objective;  // Primary optimization objective
    CompressionObjective secondary_objective; // Secondary objective
    float objective_weight_primary;          // Weight for primary objective (0-1)
    float objective_weight_secondary;        // Weight for secondary objective
    
    // Constraints
    float max_compression_time_ms;           // Maximum allowed compression time
    float min_compression_ratio;             // Minimum required compression ratio
    float min_quality_score;                 // Minimum quality requirement
    uint32_t max_memory_usage_mb;            // Maximum memory usage limit
    float max_energy_consumption;            // Maximum energy consumption
    
    // Analysis parameters
    uint32_t sample_size_for_analysis;       // Sample size for analysis
    float analysis_confidence_threshold;     // Minimum confidence for analysis
    uint32_t num_algorithms_to_evaluate;     // Number of top algorithms to evaluate
    bool enable_ensemble_methods;            // Enable ensemble compression
    
    // Machine learning parameters
    bool enable_ml_prediction;               // Enable ML-based prediction
    float ml_model_confidence_threshold;     // ML model confidence threshold
    uint32_t ml_feature_vector_size;         // Feature vector size for ML
    
    // Performance tuning
    bool enable_gpu_acceleration;            // Enable GPU acceleration
    uint32_t num_threads;                    // Number of processing threads
    bool enable_adaptive_sampling;          // Enable adaptive sampling
    uint32_t max_analysis_time_ms;           // Maximum analysis time
} CompressionSelectorConfig;

// Machine learning model for algorithm selection
typedef struct {
    void* model_weights;                     // Neural network weights
    void* feature_normalizer;                // Feature normalization parameters
    void* decision_tree;                     // Decision tree model
    void* ensemble_weights;                  // Ensemble model weights
    
    uint32_t input_feature_count;            // Number of input features
    uint32_t hidden_layer_count;             // Number of hidden layers
    uint32_t output_class_count;             // Number of output classes
    
    float model_accuracy;                    // Model accuracy on validation set
    uint32_t training_sample_count;          // Number of training samples
    uint32_t model_version;                  // Model version number
    bool is_trained;                         // Whether model is trained
} CompressionSelectorModel;

// Performance history for adaptive learning
typedef struct {
    CompressionAlgorithm algorithm;          // Algorithm used
    DataCharacteristics data_chars;          // Data characteristics
    float actual_ratio;                      // Actual compression ratio achieved
    float actual_speed;                      // Actual compression speed
    float actual_quality;                    // Actual quality achieved
    uint64_t timestamp;                      // Timestamp of compression
    float prediction_error;                  // Prediction error
    bool was_successful;                     // Whether compression succeeded
} PerformanceHistoryEntry;

// Metal buffers for GPU acceleration
typedef struct {
    void* feature_buffer;                    // Feature data buffer
    void* prediction_buffer;                 // Prediction output buffer
    void* model_weights_buffer;              // Model weights buffer
    void* intermediate_buffer;               // Intermediate computation buffer
    void* analysis_buffer;                   // Data analysis buffer
    
    size_t buffer_sizes[8];                  // Sizes of all buffers
    uint32_t num_allocated_buffers;          // Number of allocated buffers
} CompressionSelectorBuffers;

// Main compression selector context
typedef struct {
    CompressionSelectorConfig config;        // Selector configuration
    CompressionSelectorModel model;          // ML model for selection
    CompressionSelectorBuffers buffers;      // Metal buffers
    
    // Metal/MPS objects
    void* device;                           // Metal device
    void* command_queue;                    // Metal command queue
    void* analysis_graph;                   // Data analysis computation graph
    void* prediction_graph;                 // Prediction computation graph
    
    // Performance history
    PerformanceHistoryEntry* history;       // Performance history array
    uint32_t history_count;                 // Number of history entries
    uint32_t history_capacity;              // History array capacity
    
    // Algorithm availability
    bool available_algorithms[16];          // Available algorithm flags
    float algorithm_weights[16];            // Algorithm preference weights
    
    // Statistics
    uint64_t total_selections;              // Total number of selections made
    uint64_t successful_selections;         // Number of successful selections
    float average_prediction_accuracy;      // Average prediction accuracy
    
    // Internal state
    bool is_initialized;                    // Whether context is initialized
    bool model_loaded;                      // Whether ML model is loaded
    uint32_t current_sample_size;           // Current analysis sample size
    void* platform_context;                // Platform-specific context
} CompressionSelectorContext;

// Selection result
typedef struct {
    CompressionAlgorithm selected_algorithm; // Selected algorithm
    AlgorithmPerformancePrediction prediction; // Performance prediction
    DataCharacteristics data_analysis;       // Data analysis results
    
    float selection_confidence;             // Selection confidence (0-1)
    uint32_t num_algorithms_evaluated;      // Number of algorithms evaluated
    uint32_t analysis_time_ms;              // Time spent on analysis
    
    // Alternative options
    CompressionAlgorithm* alternative_algorithms; // Alternative algorithms
    float* alternative_scores;              // Scores for alternatives
    uint32_t num_alternatives;              // Number of alternatives
    
    // Detailed scoring
    float size_score;                       // Size optimization score
    float speed_score;                      // Speed optimization score
    float quality_score;                    // Quality optimization score
    float overall_score;                    // Overall optimization score
    
    bool requires_gpu;                      // Whether GPU is required
    uint32_t estimated_memory_mb;           // Estimated memory requirement
} CompressionSelectionResult;

// Core API Functions

/**
 * Create and initialize compression selector context
 * @param context Pointer to store created context
 * @param config Selector configuration
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_create(CompressionSelectorContext** context,
                                                    const CompressionSelectorConfig* config);

/**
 * Load machine learning model for algorithm selection
 * @param context Compression selector context
 * @param model_path Path to trained model file
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_load_model(CompressionSelectorContext* context,
                                                        const char* model_path);

/**
 * Analyze data characteristics
 * @param context Compression selector context
 * @param data Input data to analyze
 * @param data_size Size of input data
 * @param characteristics Output data characteristics
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_analyze_data(CompressionSelectorContext* context,
                                                          const void* data,
                                                          size_t data_size,
                                                          DataCharacteristics* characteristics);

/**
 * Select optimal compression algorithm
 * @param context Compression selector context
 * @param data Input data
 * @param data_size Size of input data
 * @param result Selection result
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_select_algorithm(CompressionSelectorContext* context,
                                                              const void* data,
                                                              size_t data_size,
                                                              CompressionSelectionResult* result);

/**
 * Update performance history with actual results
 * @param context Compression selector context
 * @param algorithm Algorithm that was used
 * @param data_chars Data characteristics
 * @param actual_ratio Actual compression ratio achieved
 * @param actual_speed Actual compression speed
 * @param actual_quality Actual quality achieved
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_update_history(CompressionSelectorContext* context,
                                                            CompressionAlgorithm algorithm,
                                                            const DataCharacteristics* data_chars,
                                                            float actual_ratio,
                                                            float actual_speed,
                                                            float actual_quality);

// Configuration Functions

/**
 * Create default compression selector configuration
 * @param config Pointer to store default configuration
 * @param objective Primary optimization objective
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_config_create_default(CompressionSelectorConfig* config,
                                                                   CompressionObjective objective);

/**
 * Set algorithm availability
 * @param context Compression selector context
 * @param algorithm Algorithm to configure
 * @param is_available Whether algorithm is available
 * @param preference_weight Preference weight (0-1)
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_set_algorithm_availability(CompressionSelectorContext* context,
                                                                        CompressionAlgorithm algorithm,
                                                                        bool is_available,
                                                                        float preference_weight);

// Analysis Functions

/**
 * Extract features for machine learning
 * @param characteristics Data characteristics
 * @param features Output feature vector
 * @param feature_count Number of features
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_extract_features(const DataCharacteristics* characteristics,
                                                              float* features,
                                                              uint32_t feature_count);

/**
 * Predict algorithm performance
 * @param context Compression selector context
 * @param characteristics Data characteristics
 * @param algorithm Algorithm to predict for
 * @param prediction Output performance prediction
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_predict_performance(CompressionSelectorContext* context,
                                                                 const DataCharacteristics* characteristics,
                                                                 CompressionAlgorithm algorithm,
                                                                 AlgorithmPerformancePrediction* prediction);

// Utility Functions

/**
 * Get algorithm name string
 * @param algorithm Algorithm identifier
 * @return Human-readable algorithm name
 */
const char* compression_selector_get_algorithm_name(CompressionAlgorithm algorithm);

/**
 * Get error message string
 * @param error_code Error code
 * @return Human-readable error message
 */
const char* compression_selector_get_error_string(CompressionSelectorError error_code);

/**
 * Calculate feature importance scores
 * @param context Compression selector context
 * @param importance_scores Output importance scores
 * @param feature_count Number of features
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_get_feature_importance(CompressionSelectorContext* context,
                                                                    float* importance_scores,
                                                                    uint32_t feature_count);

/**
 * Get performance statistics
 * @param context Compression selector context
 * @param total_selections Output total selections
 * @param accuracy Output average accuracy
 * @param avg_analysis_time Output average analysis time
 * @return COMPRESSION_SELECTOR_SUCCESS on success, error code on failure
 */
CompressionSelectorError compression_selector_get_statistics(CompressionSelectorContext* context,
                                                           uint64_t* total_selections,
                                                           float* accuracy,
                                                           float* avg_analysis_time);

/**
 * Destroy compression selector context
 * @param context Context to destroy
 */
void compression_selector_destroy(CompressionSelectorContext* context);

#ifdef __cplusplus
}
#endif

#endif // COMPRESSION_SELECTOR_H
