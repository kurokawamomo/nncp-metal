#ifndef MPS_LSTM_COMPRESSOR_H
#define MPS_LSTM_COMPRESSOR_H

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

// Include MPS LSTM for integration
#include "../ops/mps_lstm.h"

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for MPS LSTM Compressor operations
typedef enum {
    MPS_LSTM_COMPRESSOR_SUCCESS = 0,
    MPS_LSTM_COMPRESSOR_ERROR_INVALID_PARAM,
    MPS_LSTM_COMPRESSOR_ERROR_MEMORY_ALLOCATION,
    MPS_LSTM_COMPRESSOR_ERROR_DEVICE_NOT_FOUND,
    MPS_LSTM_COMPRESSOR_ERROR_COMPUTE_FAILED,
    MPS_LSTM_COMPRESSOR_ERROR_INVALID_DIMENSIONS,
    MPS_LSTM_COMPRESSOR_ERROR_BUFFER_ALLOCATION,
    MPS_LSTM_COMPRESSOR_ERROR_GRAPH_COMPILATION,
    MPS_LSTM_COMPRESSOR_ERROR_EXECUTION_FAILED,
    MPS_LSTM_COMPRESSOR_ERROR_UNSUPPORTED_OPERATION,
    MPS_LSTM_COMPRESSOR_ERROR_COMPRESSION_FAILED,
    MPS_LSTM_COMPRESSOR_ERROR_DECOMPRESSION_FAILED,
    MPS_LSTM_COMPRESSOR_ERROR_INVALID_COMPRESSION_RATIO
} MPSLSTMCompressorError;

// Compression modes for LSTM-based compression
typedef enum {
    MPS_LSTM_COMPRESSION_PREDICTIVE = 0,    // Predictive compression using LSTM
    MPS_LSTM_COMPRESSION_TEMPORAL,          // Temporal pattern compression
    MPS_LSTM_COMPRESSION_RESIDUAL,          // Residual-based compression
    MPS_LSTM_COMPRESSION_HIERARCHICAL,      // Multi-level hierarchical compression
    MPS_LSTM_COMPRESSION_ADAPTIVE           // Adaptive compression based on data
} MPSLSTMCompressionMode;

// Data types supported for compression
typedef enum {
    MPS_LSTM_DATA_TYPE_FLOAT32 = 0,
    MPS_LSTM_DATA_TYPE_FLOAT16,
    MPS_LSTM_DATA_TYPE_INT32,
    MPS_LSTM_DATA_TYPE_INT16,
    MPS_LSTM_DATA_TYPE_INT8,
    MPS_LSTM_DATA_TYPE_UINT8
} MPSLSTMDataType;

// Sequence processing strategies
typedef enum {
    MPS_LSTM_SEQUENCE_FIXED = 0,            // Fixed-length sequences
    MPS_LSTM_SEQUENCE_VARIABLE,             // Variable-length sequences
    MPS_LSTM_SEQUENCE_SLIDING_WINDOW,       // Sliding window processing
    MPS_LSTM_SEQUENCE_OVERLAPPING           // Overlapping window processing
} MPSLSTMSequenceStrategy;

// Quality vs compression trade-off levels
typedef enum {
    MPS_LSTM_QUALITY_LOSSLESS = 0,          // Lossless compression
    MPS_LSTM_QUALITY_HIGH,                  // High quality (minimal loss)
    MPS_LSTM_QUALITY_MEDIUM,                // Medium quality (balanced)
    MPS_LSTM_QUALITY_LOW,                   // Low quality (high compression)
    MPS_LSTM_QUALITY_CUSTOM                 // Custom quality settings
} MPSLSTMQualityLevel;

// LSTM compressor configuration
typedef struct {
    // Basic configuration
    uint32_t input_size;                    // Input feature dimension
    uint32_t hidden_size;                   // LSTM hidden dimension
    uint32_t num_layers;                    // Number of LSTM layers
    uint32_t sequence_length;               // Processing sequence length
    uint32_t batch_size;                    // Batch size for processing
    uint32_t prediction_horizon;            // Prediction horizon for compression
    
    // Compression settings
    MPSLSTMCompressionMode compression_mode; // Compression algorithm mode
    MPSLSTMDataType input_data_type;        // Input data type
    MPSLSTMDataType compressed_data_type;   // Compressed data type
    MPSLSTMSequenceStrategy sequence_strategy; // Sequence processing strategy
    MPSLSTMQualityLevel quality_level;      // Quality vs compression trade-off
    
    // Advanced settings
    float compression_ratio_target;         // Target compression ratio
    float prediction_threshold;             // Prediction accuracy threshold
    float quantization_scale;               // Quantization scale factor
    uint32_t residual_bits;                 // Bits for residual encoding
    bool enable_temporal_coherence;         // Enable temporal coherence
    bool enable_adaptive_prediction;        // Enable adaptive prediction
    bool enable_entropy_coding;             // Enable entropy coding
    
    // Performance settings
    bool use_bidirectional;                 // Use bidirectional LSTM
    bool enable_attention;                  // Enable attention mechanism
    float dropout_rate;                     // Dropout rate for training
    uint32_t num_threads;                   // Number of processing threads
    bool enable_gpu_acceleration;           // Enable GPU acceleration
} MPSLSTMCompressorConfig;

// Compression metadata
typedef struct {
    uint32_t original_size;                 // Original data size in bytes
    uint32_t compressed_size;               // Compressed data size in bytes
    float compression_ratio;                // Achieved compression ratio
    float prediction_accuracy;              // Prediction accuracy
    uint32_t num_sequences;                 // Number of sequences processed
    uint32_t sequence_length;               // Average sequence length
    
    // Quality metrics
    float mse;                              // Mean squared error
    float psnr;                             // Peak signal-to-noise ratio
    float ssim;                             // Structural similarity index
    
    // Timing information
    uint64_t compression_time_ns;           // Compression time in nanoseconds
    uint64_t decompression_time_ns;         // Decompression time in nanoseconds
    
    // Additional metadata
    MPSLSTMCompressionMode compression_mode; // Compression mode used
    MPSLSTMDataType data_type;              // Data type
    uint32_t version;                       // Compression format version
    uint32_t checksum;                      // Data integrity checksum
} MPSLSTMCompressionMetadata;

// LSTM model weights for compression
typedef struct {
    // Encoder LSTM weights
    void* encoder_input_weights;           // Input transformation weights
    void* encoder_hidden_weights;          // Hidden state weights
    void* encoder_cell_weights;            // Cell state weights
    void* encoder_output_weights;          // Output projection weights
    void* encoder_biases;                  // Encoder biases
    
    // Decoder LSTM weights (for reconstruction)
    void* decoder_input_weights;           // Decoder input weights
    void* decoder_hidden_weights;          // Decoder hidden weights
    void* decoder_cell_weights;            // Decoder cell weights
    void* decoder_output_weights;          // Decoder output weights
    void* decoder_biases;                  // Decoder biases
    
    // Predictor weights
    void* predictor_weights;               // Prediction layer weights
    void* predictor_biases;                // Prediction layer biases
    
    // Quantizer weights (if applicable)
    void* quantizer_codebook;              // Vector quantization codebook
    void* quantizer_mapping;               // Quantizer mapping weights
    
    size_t total_weights_size;             // Total size of all weights
    uint32_t weights_version;              // Weights version number
} MPSLSTMCompressorWeights;

// Metal buffers for LSTM compression
typedef struct {
    // Input/output buffers
    void* input_buffer;                    // Input data buffer
    void* compressed_buffer;               // Compressed data buffer
    void* decompressed_buffer;             // Decompressed data buffer
    void* metadata_buffer;                 // Metadata buffer
    
    // Intermediate computation buffers
    void* hidden_states_buffer;            // LSTM hidden states
    void* cell_states_buffer;              // LSTM cell states
    void* predictions_buffer;              // Prediction buffer
    void* residuals_buffer;                // Residual errors buffer
    void* quantized_buffer;                // Quantized data buffer
    
    // Temporary buffers
    void* temp_buffers[8];                 // Temporary computation buffers
    size_t buffer_sizes[16];               // Sizes of all buffers
    uint32_t num_allocated_buffers;        // Number of allocated buffers
} MPSLSTMCompressorBuffers;

// LSTM compression computation graphs
typedef struct {
    void* encoder_graph;                   // Encoder computation graph
    void* decoder_graph;                   // Decoder computation graph
    void* predictor_graph;                 // Predictor computation graph
    void* quantizer_graph;                 // Quantizer computation graph
    
    void* encoder_executable;              // Compiled encoder graph
    void* decoder_executable;              // Compiled decoder graph
    void* predictor_executable;            // Compiled predictor graph
    void* quantizer_executable;            // Compiled quantizer graph
    
    bool graphs_compiled;                  // Whether graphs are compiled
} MPSLSTMCompressorGraphs;

// Performance statistics for LSTM compression
typedef struct {
    uint64_t total_compressions;           // Total compression operations
    uint64_t total_decompressions;         // Total decompression operations
    uint64_t total_bytes_compressed;       // Total bytes compressed
    uint64_t total_bytes_decompressed;     // Total bytes decompressed
    
    uint64_t total_compression_time_ns;    // Total compression time
    uint64_t total_decompression_time_ns;  // Total decompression time
    
    float average_compression_ratio;       // Average compression ratio
    float average_compression_time_ms;     // Average compression time per MB
    float average_decompression_time_ms;   // Average decompression time per MB
    float average_prediction_accuracy;     // Average prediction accuracy
    
    float throughput_mbps_compression;     // Compression throughput (MB/s)
    float throughput_mbps_decompression;   // Decompression throughput (MB/s)
    
    uint32_t memory_usage_mb;              // Current memory usage
    uint32_t peak_memory_usage_mb;         // Peak memory usage
    
    // Quality metrics
    float best_compression_ratio;          // Best compression ratio achieved
    float worst_compression_ratio;         // Worst compression ratio
    float average_mse;                     // Average mean squared error
    float average_psnr;                    // Average PSNR
} MPSLSTMCompressorStats;

// Main LSTM compressor context
typedef struct {
    MPSLSTMCompressorConfig config;        // Compressor configuration
    MPSLSTMCompressorWeights weights;      // Model weights
    MPSLSTMCompressorBuffers buffers;      // Metal buffers
    MPSLSTMCompressorGraphs graphs;        // Computation graphs
    MPSLSTMCompressorStats stats;          // Performance statistics
    
    // Metal/MPS objects
    void* device;                          // Metal device
    void* command_queue;                   // Metal command queue
    
    // LSTM contexts for encoder/decoder
    MPSLSTMContext* encoder_context;       // Encoder LSTM context
    MPSLSTMContext* decoder_context;       // Decoder LSTM context
    MPSLSTMContext* predictor_context;     // Predictor LSTM context
    
    // Internal state
    bool is_initialized;                   // Whether context is initialized
    bool is_trained;                       // Whether model is trained
    uint32_t current_sequence_length;      // Current sequence length
    uint32_t current_batch_size;           // Current batch size
    void* platform_context;               // Platform-specific context
} MPSLSTMCompressorContext;

// Input/output data for compression
typedef struct {
    // Input data
    void* input_data;                      // Raw input data
    uint32_t input_size;                   // Size of input data
    MPSLSTMDataType input_type;            // Input data type
    
    // Output data
    void* compressed_data;                 // Compressed output data
    uint32_t compressed_size;              // Size of compressed data
    void* decompressed_data;               // Decompressed output data
    uint32_t decompressed_size;            // Size of decompressed data
    
    // Metadata
    MPSLSTMCompressionMetadata metadata;   // Compression metadata
    
    // Sequence information
    uint32_t num_sequences;                // Number of sequences
    uint32_t* sequence_lengths;            // Length of each sequence
    uint32_t batch_size;                   // Batch size
    
    bool owns_memory;                      // Whether this structure owns the memory
} MPSLSTMCompressorData;

// Core API Functions

/**
 * Create and initialize MPS LSTM compressor context
 * @param context Pointer to store created context
 * @param config LSTM compressor configuration
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_create(MPSLSTMCompressorContext** context,
                                                 const MPSLSTMCompressorConfig* config);

/**
 * Load pre-trained weights for LSTM compressor
 * @param context MPS LSTM compressor context
 * @param weights Pre-trained model weights
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_load_weights(MPSLSTMCompressorContext* context,
                                                       const MPSLSTMCompressorWeights* weights);

/**
 * Compress data using LSTM-based compression
 * @param context MPS LSTM compressor context
 * @param data Input/output data structure
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_compress(MPSLSTMCompressorContext* context,
                                                   MPSLSTMCompressorData* data);

/**
 * Decompress data using LSTM-based decompression
 * @param context MPS LSTM compressor context
 * @param data Input/output data structure
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_decompress(MPSLSTMCompressorContext* context,
                                                     MPSLSTMCompressorData* data);

/**
 * Train LSTM compressor on training data
 * @param context MPS LSTM compressor context
 * @param training_data Array of training data samples
 * @param num_samples Number of training samples
 * @param num_epochs Number of training epochs
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_train(MPSLSTMCompressorContext* context,
                                                MPSLSTMCompressorData* training_data,
                                                uint32_t num_samples,
                                                uint32_t num_epochs);

// Configuration Functions

/**
 * Create default LSTM compressor configuration
 * @param config Pointer to store default configuration
 * @param compression_mode Compression mode to use
 * @param input_size Input feature dimension
 * @param sequence_length Processing sequence length
 * @param quality_level Quality vs compression trade-off
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_config_create_default(MPSLSTMCompressorConfig* config,
                                                                MPSLSTMCompressionMode compression_mode,
                                                                uint32_t input_size,
                                                                uint32_t sequence_length,
                                                                MPSLSTMQualityLevel quality_level);

/**
 * Validate LSTM compressor configuration
 * @param config Configuration to validate
 * @return MPS_LSTM_COMPRESSOR_SUCCESS if valid, error code otherwise
 */
MPSLSTMCompressorError mps_lstm_compressor_config_validate(const MPSLSTMCompressorConfig* config);

/**
 * Calculate memory requirements for LSTM compression
 * @param config LSTM compressor configuration
 * @param memory_mb Pointer to store memory requirement in MB
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_calculate_memory_requirements(const MPSLSTMCompressorConfig* config,
                                                                        uint32_t* memory_mb);

// Specialized Compression Functions

/**
 * Compress time series data
 * @param context MPS LSTM compressor context
 * @param time_series Input time series data [batch, time, features]
 * @param compressed_output Output compressed data
 * @param metadata Compression metadata
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_compress_time_series(MPSLSTMCompressorContext* context,
                                                               const float* time_series,
                                                               void* compressed_output,
                                                               MPSLSTMCompressionMetadata* metadata);

/**
 * Compress sequence data with variable lengths
 * @param context MPS LSTM compressor context
 * @param sequences Array of sequence data
 * @param sequence_lengths Length of each sequence
 * @param num_sequences Number of sequences
 * @param compressed_output Output compressed data
 * @param metadata Compression metadata
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_compress_sequences(MPSLSTMCompressorContext* context,
                                                             const float** sequences,
                                                             const uint32_t* sequence_lengths,
                                                             uint32_t num_sequences,
                                                             void* compressed_output,
                                                             MPSLSTMCompressionMetadata* metadata);

/**
 * Compress streaming data in real-time
 * @param context MPS LSTM compressor context
 * @param stream_data Streaming input data
 * @param stream_length Length of stream data
 * @param compressed_chunks Output compressed chunks
 * @param num_chunks Number of output chunks
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_compress_stream(MPSLSTMCompressorContext* context,
                                                          const float* stream_data,
                                                          uint32_t stream_length,
                                                          void** compressed_chunks,
                                                          uint32_t* num_chunks);

// Weight Management Functions

/**
 * Create LSTM compressor weights structure
 * @param weights Pointer to store created weights
 * @param config LSTM compressor configuration
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_weights_create(MPSLSTMCompressorWeights** weights,
                                                         const MPSLSTMCompressorConfig* config);

/**
 * Initialize weights with random values for training
 * @param weights Weights structure to initialize
 * @param config LSTM compressor configuration
 * @param seed Random seed
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_weights_init_random(MPSLSTMCompressorWeights* weights,
                                                              const MPSLSTMCompressorConfig* config,
                                                              uint32_t seed);

/**
 * Save weights to file
 * @param weights Weights structure to save
 * @param filename Output filename
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_weights_save(const MPSLSTMCompressorWeights* weights,
                                                       const char* filename);

/**
 * Load weights from file
 * @param weights Weights structure to load into
 * @param filename Input filename
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_weights_load(MPSLSTMCompressorWeights* weights,
                                                       const char* filename);

/**
 * Destroy LSTM compressor weights structure
 * @param weights Weights to destroy
 */
void mps_lstm_compressor_weights_destroy(MPSLSTMCompressorWeights* weights);

// Data Management Functions

/**
 * Create LSTM compressor data structure
 * @param data Pointer to store created data structure
 * @param input_size Size of input data
 * @param data_type Data type
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_data_create(MPSLSTMCompressorData** data,
                                                      uint32_t input_size,
                                                      MPSLSTMDataType data_type);

/**
 * Destroy LSTM compressor data structure
 * @param data Data structure to destroy
 */
void mps_lstm_compressor_data_destroy(MPSLSTMCompressorData* data);

/**
 * Validate data structure
 * @param data Data structure to validate
 * @param config LSTM compressor configuration
 * @return MPS_LSTM_COMPRESSOR_SUCCESS if valid, error code otherwise
 */
MPSLSTMCompressorError mps_lstm_compressor_data_validate(const MPSLSTMCompressorData* data,
                                                        const MPSLSTMCompressorConfig* config);

// Statistics and Monitoring Functions

/**
 * Get LSTM compressor performance statistics
 * @param context MPS LSTM compressor context
 * @param stats Pointer to store statistics
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_get_stats(MPSLSTMCompressorContext* context,
                                                    MPSLSTMCompressorStats* stats);

/**
 * Reset performance statistics
 * @param context MPS LSTM compressor context
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_reset_stats(MPSLSTMCompressorContext* context);

/**
 * Get current memory usage
 * @param context MPS LSTM compressor context
 * @param memory_usage_mb Pointer to store memory usage in MB
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_get_memory_usage(MPSLSTMCompressorContext* context,
                                                           uint32_t* memory_usage_mb);

// Utility Functions

/**
 * Get error message string
 * @param error_code MPS LSTM compressor error code
 * @return Human-readable error message
 */
const char* mps_lstm_compressor_get_error_string(MPSLSTMCompressorError error_code);

/**
 * Check if MPS LSTM compressor is available on current device
 * @return true if available, false otherwise
 */
bool mps_lstm_compressor_is_available(void);

/**
 * Get compressor device information
 * @param device_name Buffer to store device name
 * @param buffer_size Size of device name buffer
 * @param compute_units Pointer to store number of compute units
 * @param max_memory_mb Pointer to store maximum memory in MB
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_get_device_info(char* device_name,
                                                          size_t buffer_size,
                                                          uint32_t* compute_units,
                                                          uint32_t* max_memory_mb);

/**
 * Calculate optimal compression parameters for given data
 * @param input_data Sample input data
 * @param input_size Size of input data
 * @param target_ratio Target compression ratio
 * @param config Output optimal configuration
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_optimize_config(const void* input_data,
                                                          uint32_t input_size,
                                                          float target_ratio,
                                                          MPSLSTMCompressorConfig* config);

/**
 * Destroy MPS LSTM compressor context
 * @param context Context to destroy
 */
void mps_lstm_compressor_destroy(MPSLSTMCompressorContext* context);

// Prediction Scoring Integration Functions

/**
 * Initialize prediction scoring integration for LSTM compressor
 * @param context MPS LSTM compressor context
 * @param prediction_scorer Advanced prediction scorer instance
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_init_prediction_scoring(MPSLSTMCompressorContext* context,
                                                                  void* prediction_scorer);

/**
 * Execute LSTM compression with prediction scoring integration
 * @param context MPS LSTM compressor context
 * @param input_data Input sequence data [sequence_length]
 * @param input_length Length of input sequence
 * @param prediction_candidates Output prediction candidates array
 * @param max_candidates Maximum number of candidates to generate
 * @param num_candidates Output number of candidates generated
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_predict_with_scoring(MPSLSTMCompressorContext* context,
                                                               const uint8_t* input_data,
                                                               uint32_t input_length,
                                                               void* prediction_candidates,
                                                               uint32_t max_candidates,
                                                               uint32_t* num_candidates);

/**
 * Update LSTM context with prediction scoring feedback
 * @param context MPS LSTM compressor context
 * @param actual_byte Actual byte that was compressed
 * @param predicted_byte Predicted byte from scoring system
 * @param compression_effectiveness Effectiveness score for this prediction
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_update_prediction_feedback(MPSLSTMCompressorContext* context,
                                                                     uint8_t actual_byte,
                                                                     uint8_t predicted_byte,
                                                                     float compression_effectiveness);

/**
 * Get LSTM state management statistics with prediction scoring integration
 * @param context MPS LSTM compressor context
 * @param state_efficiency Output pointer for state efficiency score
 * @param prediction_accuracy Output pointer for prediction accuracy
 * @param pattern_utilization Output pointer for pattern utilization rate
 * @return MPS_LSTM_COMPRESSOR_SUCCESS on success, error code on failure
 */
MPSLSTMCompressorError mps_lstm_compressor_get_prediction_stats(MPSLSTMCompressorContext* context,
                                                               float* state_efficiency,
                                                               float* prediction_accuracy,
                                                               float* pattern_utilization);

#ifdef __cplusplus
}
#endif

#endif // MPS_LSTM_COMPRESSOR_H
