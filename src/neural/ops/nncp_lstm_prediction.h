#ifndef NNCP_LSTM_PREDICTION_H
#define NNCP_LSTM_PREDICTION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations (include state manager header for complete type)
#include "nncp_lstm_state_manager.h"

// LSTM Prediction Error Codes
typedef enum {
    NNCP_PREDICTION_SUCCESS = 0,
    NNCP_PREDICTION_ERROR_INVALID_PARAM,
    NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION,
    NNCP_PREDICTION_ERROR_COMPUTATION_FAILED,
    NNCP_PREDICTION_ERROR_INVALID_DIMENSIONS,
    NNCP_PREDICTION_ERROR_OUTPUT_BUFFER_TOO_SMALL,
    NNCP_PREDICTION_ERROR_METAL_OPERATION_FAILED
} NNCPLSTMPredictionError;

// LSTM Prediction Configuration (matching CUDA implementation)
typedef struct {
    // Output configuration
    uint32_t n_symbols;             // Vocabulary size (default: 256)
    uint32_t n_embed_out;           // Number of layers to use for output (default: 4)
    uint32_t n_layers;              // Total number of LSTM layers
    uint32_t n_cells;               // Hidden state dimension per layer
    
    // Sequence processing
    uint32_t batch_size;            // Number of parallel sequences
    uint32_t seg_len;               // Sequence segment length
    bool full_connect;              // Use all previous layers (default: true)
    
    // Output processing options
    bool apply_softmax;             // Apply softmax to final logits
    float temperature;              // Temperature for sampling (1.0 = no scaling)
    bool use_log_probabilities;     // Output log probabilities instead of raw logits
    
    // Performance and debugging
    bool verbose_logging;           // Enable detailed logging
    bool validate_outputs;          // Validate output integrity
} NNCPLSTMPredictionConfig;

// LSTM Output Layer Weights (Final Classification Layer)
typedef struct {
    float* fc_weights;              // Final layer weights [n_symbols, n_total_inputs]
    float* fc_bias;                 // Final layer bias [n_symbols] (optional)
    uint32_t n_symbols;             // Output vocabulary size
    uint32_t n_total_inputs;        // Total input dimension (sum of all layer outputs)
    
    // Metal buffers
    void* fc_weights_buffer;        // Metal buffer for weights
    void* fc_bias_buffer;           // Metal buffer for bias (optional)
    
    // Metadata
    bool weights_loaded;            // Whether weights are loaded
    bool use_bias;                  // Whether to use bias terms
    float weight_scale;             // Weight initialization scale
} NNCPLSTMOutputLayer;

// LSTM Prediction Context
typedef struct {
    NNCPLSTMPredictionConfig config; // Prediction configuration
    NNCPLSTMOutputLayer output_layer; // Final classification layer
    
    // Temporary computation buffers
    float* layer_outputs_buffer;    // Concatenated layer outputs [n_total_inputs, batch*seq]
    float* logits_buffer;           // Output logits [n_symbols, batch*seq]
    float* probabilities_buffer;    // Output probabilities [n_symbols, batch*seq]
    
    // Metal buffers for GPU computation
    void* layer_outputs_metal;      // Metal buffer for layer outputs
    void* logits_metal;             // Metal buffer for logits
    void* probabilities_metal;      // Metal buffer for probabilities
    void* temp_buffers[4];          // Additional temporary buffers
    
    // Metal context
    void* device;                   // MTLDevice
    void* command_queue;            // MTLCommandQueue
    void* softmax_pipeline;         // Metal compute pipeline for softmax
    void* matmul_pipeline;          // Metal compute pipeline for matrix multiplication
    
    // State and performance tracking
    bool is_initialized;            // Initialization status
    uint64_t total_predictions;     // Total number of predictions made
    uint64_t total_compute_time_ns; // Total computation time
    float average_entropy;          // Average prediction entropy
    bool verbose_logging;           // Enable detailed logging
} NNCPLSTMPredictionContext;

// Prediction Result Information
typedef struct {
    float* logits;                  // Raw logits [n_symbols]
    float* probabilities;           // Probability distribution [n_symbols]
    uint32_t predicted_symbol;      // Most likely symbol
    float prediction_confidence;    // Confidence of prediction (max probability)
    float entropy;                  // Entropy of probability distribution
    uint64_t computation_time_ns;   // Time taken for this prediction
} NNCPLSTMPredictionResult;

// Batch Prediction Result
typedef struct {
    float* batch_logits;            // Batch logits [batch_size * seq_len, n_symbols]
    float* batch_probabilities;     // Batch probabilities [batch_size * seq_len, n_symbols]
    uint32_t* predicted_symbols;    // Predicted symbols [batch_size * seq_len]
    float* prediction_confidences;  // Prediction confidences [batch_size * seq_len]
    float* entropies;               // Entropy values [batch_size * seq_len]
    uint32_t batch_size;            // Actual batch size
    uint32_t seq_len;               // Actual sequence length
    uint64_t total_computation_time_ns; // Total computation time for batch
} NNCPLSTMBatchPredictionResult;

// Confidence Scoring Results
typedef struct {
    float basic_confidence;         // Basic confidence (max probability)
    float relative_confidence;      // Relative confidence vs second-best
    float prediction_margin;        // Margin between top two predictions
    float gap_ratio;               // Gap ratio (margin / sum of top two)
    float entropy_confidence;       // Entropy-based confidence (1 - normalized entropy)
    float gini_coefficient;        // Gini coefficient for concentration
    float effective_vocab_size;     // Effective vocabulary size (exp(entropy))
    float calibrated_confidence;    // Temperature-calibrated confidence
    float ensemble_confidence;      // Ensemble of multiple confidence measures
} NNCPLSTMConfidenceScores;

// Sampling Strategy Types
typedef enum {
    NNCP_SAMPLING_GREEDY = 0,           // Greedy selection (best symbol)
    NNCP_SAMPLING_MULTINOMIAL,          // Multinomial sampling from full distribution
    NNCP_SAMPLING_TEMPERATURE,          // Temperature-scaled sampling
    NNCP_SAMPLING_TOP_K,                // Top-k filtering + sampling
    NNCP_SAMPLING_TOP_P,                // Top-p (nucleus) filtering + sampling
    NNCP_SAMPLING_TOP_K_TOP_P,          // Combined top-k and top-p filtering
    NNCP_SAMPLING_CONFIDENCE_THRESHOLD, // Confidence threshold selection
    NNCP_SAMPLING_BEAM_SEARCH,          // Beam search (single step)
    NNCP_SAMPLING_ADAPTIVE,             // Adaptive sampling based on entropy
    NNCP_SAMPLING_COMPREHENSIVE         // Comprehensive strategy with fallbacks
} NNCPLSTMSamplingStrategy;

// Sampling Configuration
typedef struct {
    NNCPLSTMSamplingStrategy strategy;  // Primary sampling strategy
    
    // Temperature parameters
    float temperature;                  // Temperature for scaling (1.0 = no scaling)
    float min_temperature;              // Minimum temperature for adaptive sampling
    float max_temperature;              // Maximum temperature for adaptive sampling
    
    // Top-k parameters
    uint32_t top_k;                     // Number of top symbols to keep (0 = disabled)
    uint32_t min_top_k;                 // Minimum top-k for adaptive sampling
    uint32_t max_top_k;                 // Maximum top-k for adaptive sampling
    
    // Top-p parameters
    float top_p;                        // Cumulative probability threshold (0.0-1.0)
    float min_top_p;                    // Minimum top-p for adaptive sampling
    float max_top_p;                    // Maximum top-p for adaptive sampling
    
    // Confidence threshold parameters
    float confidence_threshold;         // Minimum confidence for selection
    bool use_relative_confidence;       // Use relative vs absolute confidence
    
    // Beam search parameters
    uint32_t beam_width;                // Beam width for beam search
    float beam_alpha;                   // Length normalization parameter
    
    // Adaptive sampling parameters
    float entropy_threshold_high;       // High entropy threshold for adaptation
    float entropy_threshold_low;        // Low entropy threshold for adaptation
    bool adaptive_temperature;          // Enable adaptive temperature
    bool adaptive_top_k;                // Enable adaptive top-k
    bool adaptive_top_p;                // Enable adaptive top-p
    
    // Fallback and safety parameters
    NNCPLSTMSamplingStrategy fallback_strategy; // Fallback if primary fails
    bool enable_fallbacks;              // Enable fallback strategies
    uint32_t max_retries;               // Maximum sampling retries
    float safety_temperature;          // Safety temperature for failed sampling
    
    // Random seed and reproducibility
    uint32_t random_seed;               // Random seed for reproducibility
    bool deterministic_mode;            // Force deterministic sampling when possible
} NNCPLSTMSamplingConfig;

// Sampling Information (output from sampling functions)
typedef struct {
    NNCPLSTMSamplingStrategy strategy_used; // Actual strategy used
    uint32_t selected_symbol;           // Selected symbol index
    float selection_probability;        // Probability of selected symbol
    float random_value_used;            // Random value used in sampling
    
    // Distribution properties before sampling
    float original_entropy;             // Original distribution entropy
    float effective_vocabulary_size;    // Effective vocabulary size
    uint32_t total_candidates;          // Total candidate symbols before filtering
    
    // Filtering information
    uint32_t candidates_after_top_k;    // Candidates remaining after top-k
    uint32_t candidates_after_top_p;    // Candidates remaining after top-p
    float cumulative_prob_kept;         // Cumulative probability after filtering
    float temperature_used;             // Actual temperature applied
    
    // Confidence and quality metrics
    float selection_confidence;         // Confidence in the selection
    float relative_confidence;          // Relative confidence vs second-best
    float margin_to_second_best;        // Probability margin to second choice
    
    // Adaptive sampling information
    bool adaptation_triggered;          // Whether adaptation was triggered
    float adaptation_factor;            // Adaptation factor applied
    const char* adaptation_reason;      // Reason for adaptation (if any)
    
    // Performance and debugging
    uint32_t sampling_attempts;         // Number of sampling attempts
    bool fallback_used;                 // Whether fallback strategy was used
    uint64_t sampling_time_ns;          // Time taken for sampling
} NNCPLSTMSamplingInfo;

// NNCP LSTM Prediction Statistics
typedef struct {
    uint64_t total_predictions;         // Total number of predictions made
    uint64_t total_prediction_time_ns;  // Total prediction time in nanoseconds
    uint64_t average_prediction_time_ns; // Average time per prediction
    uint32_t memory_usage_mb;           // Current memory usage in MB
    float cache_hit_rate;               // Cache hit rate (0.0 - 1.0)
    float entropy_distribution_mean;    // Mean entropy across predictions
    float entropy_distribution_variance; // Variance of entropy distribution
    float confidence_distribution_mean; // Mean confidence across predictions
    float confidence_distribution_variance; // Variance of confidence distribution
} NNCPLSTMPredictionStats;

// Core Prediction API

/**
 * Create LSTM prediction context
 * @param context Pointer to store created context
 * @param config Prediction configuration
 * @param device Metal device (NULL for CPU-only)
 * @param command_queue Metal command queue (NULL for CPU-only)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_create(NNCPLSTMPredictionContext** context,
                                                   const NNCPLSTMPredictionConfig* config,
                                                   void* device,
                                                   void* command_queue);

/**
 * Initialize output layer weights (CUDA-compatible initialization)
 * @param context Prediction context
 * @param weight_scale Weight initialization scale (sqrt(12.0/(n_cells*n_layers)))
 * @param random_seed Random seed for reproducibility
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_init_output_weights(NNCPLSTMPredictionContext* context,
                                                               float weight_scale,
                                                               uint32_t random_seed);

/**
 * Load pre-trained output layer weights
 * @param context Prediction context
 * @param fc_weights Final layer weights [n_symbols, n_total_inputs]
 * @param fc_bias Final layer bias [n_symbols] (NULL if not used)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_load_output_weights(NNCPLSTMPredictionContext* context,
                                                               const float* fc_weights,
                                                               const float* fc_bias);

/**
 * Generate predictions from LSTM layer outputs (CUDA-compatible)
 * @param context Prediction context
 * @param layer_outputs Concatenated layer outputs [n_layers, n_cells, batch*seq]
 * @param batch_size Number of sequences in batch
 * @param seq_len Sequence length
 * @param result Output prediction result
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_generate(NNCPLSTMPredictionContext* context,
                                                     const float* layer_outputs,
                                                     uint32_t batch_size,
                                                     uint32_t seq_len,
                                                     NNCPLSTMBatchPredictionResult* result);

/**
 * Generate single prediction from layer outputs
 * @param context Prediction context
 * @param layer_outputs Layer outputs [n_total_inputs]
 * @param result Output prediction result
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_generate_single(NNCPLSTMPredictionContext* context,
                                                           const float* layer_outputs,
                                                           NNCPLSTMPredictionResult* result);

/**
 * Convert logits to probability distribution
 * @param context Prediction context
 * @param logits Input logits [n_symbols]
 * @param probabilities Output probabilities [n_symbols]
 * @param temperature Temperature for scaling (1.0 = no scaling)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_softmax(NNCPLSTMPredictionContext* context,
                                                    const float* logits,
                                                    float* probabilities,
                                                    float temperature);

/**
 * Sample from probability distribution
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param random_value Random value [0, 1] for sampling
 * @param sampled_symbol Output sampled symbol
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_sample(const float* probabilities,
                                                   uint32_t n_symbols,
                                                   float random_value,
                                                   uint32_t* sampled_symbol);

/**
 * Get most likely symbol from probability distribution
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param best_symbol Output most likely symbol
 * @param confidence Output confidence (probability of best symbol)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_get_best(const float* probabilities,
                                                     uint32_t n_symbols,
                                                     uint32_t* best_symbol,
                                                     float* confidence);

/**
 * Calculate entropy of probability distribution
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param entropy Output entropy value
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_calculate_entropy(const float* probabilities,
                                                             uint32_t n_symbols,
                                                             float* entropy);

// Layer Output Processing

/**
 * Concatenate LSTM layer outputs (full_connect mode)
 * @param context Prediction context
 * @param layer_outputs Array of layer outputs [n_layers][n_cells, batch*seq]
 * @param batch_size Number of sequences
 * @param seq_len Sequence length
 * @param concatenated_output Output concatenated features [n_total_inputs, batch*seq]
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_concatenate_layers(NNCPLSTMPredictionContext* context,
                                                              const float** layer_outputs,
                                                              uint32_t batch_size,
                                                              uint32_t seq_len,
                                                              float* concatenated_output);

/**
 * Apply final linear transformation (fc_weights * layer_outputs + fc_bias)
 * @param context Prediction context
 * @param layer_outputs Input features [n_total_inputs, batch*seq]
 * @param batch_size Number of sequences
 * @param seq_len Sequence length
 * @param logits Output logits [n_symbols, batch*seq]
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_apply_final_linear(NNCPLSTMPredictionContext* context,
                                                              const float* layer_outputs,
                                                              uint32_t batch_size,
                                                              uint32_t seq_len,
                                                              float* logits);

// Byte Prediction API (CUDA-compatible)

/**
 * Predict single byte from layer outputs
 * @param context Prediction context
 * @param layer_outputs Input layer outputs [n_total_inputs]
 * @param predicted_byte Output predicted byte (0-255)
 * @param prediction_confidence Output confidence score (optional)
 * @param prediction_entropy Output entropy value (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_predict_byte(NNCPLSTMPredictionContext* context,
                                                         const float* layer_outputs,
                                                         uint8_t* predicted_byte,
                                                         float* prediction_confidence,
                                                         float* prediction_entropy);

/**
 * Predict byte sequence from layer outputs sequence
 * @param context Prediction context
 * @param layer_outputs_sequence Input layer outputs [n_total_inputs, sequence_length]
 * @param sequence_length Length of sequence to predict
 * @param predicted_bytes Output predicted bytes [sequence_length]
 * @param prediction_confidences Output confidence scores [sequence_length] (optional)
 * @param prediction_entropies Output entropy values [sequence_length] (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_predict_byte_sequence(NNCPLSTMPredictionContext* context,
                                                                  const float* layer_outputs_sequence,
                                                                  uint32_t sequence_length,
                                                                  uint8_t* predicted_bytes,
                                                                  float* prediction_confidences,
                                                                  float* prediction_entropies);

/**
 * Get top-K predictions for improved compression
 * @param probabilities Input probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param k Number of top predictions to return
 * @param top_symbols Output top symbol indices [k]
 * @param top_probabilities Output top probabilities [k]
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_get_top_k(const float* probabilities,
                                                       uint32_t n_symbols,
                                                       uint32_t k,
                                                       uint32_t* top_symbols,
                                                       float* top_probabilities);

// Probability Distribution Processing

/**
 * Normalize probability distribution to sum to 1.0
 * @param probabilities Probability distribution to normalize [n_symbols]
 * @param n_symbols Number of symbols
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_normalize_distribution(float* probabilities,
                                                                   uint32_t n_symbols);

/**
 * Apply temperature scaling to logits
 * @param logits Input logits [n_symbols]
 * @param scaled_logits Output temperature-scaled logits [n_symbols]
 * @param n_symbols Number of symbols
 * @param temperature Temperature parameter (1.0 = no scaling)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_apply_temperature(const float* logits,
                                                              float* scaled_logits,
                                                              uint32_t n_symbols,
                                                              float temperature);

/**
 * Apply top-p (nucleus) filtering to probability distribution
 * @param probabilities Probability distribution to filter [n_symbols]
 * @param n_symbols Number of symbols
 * @param p_threshold Cumulative probability threshold (0.0-1.0)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_apply_top_p_filtering(float* probabilities,
                                                                  uint32_t n_symbols,
                                                                  float p_threshold);

/**
 * Apply top-k filtering to probability distribution
 * @param probabilities Probability distribution to filter [n_symbols]
 * @param n_symbols Number of symbols
 * @param k Number of top probabilities to keep
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_apply_top_k_filtering(float* probabilities,
                                                                  uint32_t n_symbols,
                                                                  uint32_t k);

/**
 * Compute perplexity of probability distributions against target symbols
 * @param probabilities Probability distributions [sequence_length, n_symbols]
 * @param target_symbols Target symbol indices [sequence_length]
 * @param sequence_length Length of sequence
 * @param n_symbols Number of symbols
 * @param perplexity Output perplexity value
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_compute_perplexity(const float* probabilities,
                                                               const uint32_t* target_symbols,
                                                               uint32_t sequence_length,
                                                               uint32_t n_symbols,
                                                               float* perplexity);

/**
 * Compute cross-entropy loss of probability distributions against target symbols
 * @param probabilities Probability distributions [sequence_length, n_symbols]
 * @param target_symbols Target symbol indices [sequence_length]
 * @param sequence_length Length of sequence
 * @param n_symbols Number of symbols
 * @param cross_entropy Output cross-entropy loss
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_compute_cross_entropy(const float* probabilities,
                                                                  const uint32_t* target_symbols,
                                                                  uint32_t sequence_length,
                                                                  uint32_t n_symbols,
                                                                  float* cross_entropy);

/**
 * Compute statistics for probability distribution
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param entropy Output entropy value (optional)
 * @param max_prob Output maximum probability (optional)
 * @param max_symbol Output symbol with maximum probability (optional)
 * @param variance Output variance of distribution (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_distribution_statistics(const float* probabilities,
                                                                    uint32_t n_symbols,
                                                                    float* entropy,
                                                                    float* max_prob,
                                                                    uint32_t* max_symbol,
                                                                    float* variance);

// Prediction Confidence Scoring (CUDA-compatible)

/**
 * Compute basic and relative confidence scores
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param predicted_symbol Index of predicted symbol
 * @param confidence_score Output basic confidence score
 * @param relative_confidence Output relative confidence vs second-best (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_compute_confidence_score(const float* probabilities,
                                                                     uint32_t n_symbols,
                                                                     uint32_t predicted_symbol,
                                                                     float* confidence_score,
                                                                     float* relative_confidence);

/**
 * Compute prediction margin and gap ratio
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param margin Output margin between top two predictions
 * @param gap_ratio Output gap ratio (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_compute_prediction_margin(const float* probabilities,
                                                                      uint32_t n_symbols,
                                                                      float* margin,
                                                                      float* gap_ratio);

/**
 * Compute entropy-based confidence measure
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param normalized_confidence Output normalized confidence (1 - normalized_entropy)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_compute_entropy_based_confidence(const float* probabilities,
                                                                             uint32_t n_symbols,
                                                                             float* normalized_confidence);

/**
 * Compute concentration measures (Gini coefficient, effective vocabulary size)
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param gini_coefficient Output Gini coefficient (optional)
 * @param effective_vocabulary_size Output effective vocabulary size (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_compute_concentration_measure(const float* probabilities,
                                                                          uint32_t n_symbols,
                                                                          float* gini_coefficient,
                                                                          float* effective_vocabulary_size);

/**
 * Compute temperature-calibrated confidence
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param predicted_symbol Index of predicted symbol
 * @param temperature Temperature parameter for calibration
 * @param calibrated_confidence Output calibrated confidence
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_compute_calibrated_confidence(const float* probabilities,
                                                                          uint32_t n_symbols,
                                                                          uint32_t predicted_symbol,
                                                                          float temperature,
                                                                          float* calibrated_confidence);

/**
 * Ensemble multiple confidence measures with weights
 * @param individual_confidences Array of individual confidence scores
 * @param weights Weight for each confidence measure (NULL for equal weights)
 * @param n_methods Number of confidence methods
 * @param ensemble_confidence Output weighted ensemble confidence
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_confidence_ensemble(const float* individual_confidences,
                                                                const float* weights,
                                                                uint32_t n_methods,
                                                                float* ensemble_confidence);

/**
 * Compute comprehensive confidence scores using multiple methods
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param predicted_symbol Index of predicted symbol
 * @param temperature Temperature parameter for calibration
 * @param scores Output comprehensive confidence scores
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_compute_comprehensive_confidence(const float* probabilities,
                                                                             uint32_t n_symbols,
                                                                             uint32_t predicted_symbol,
                                                                             float temperature,
                                                                             NNCPLSTMConfidenceScores* scores);

// Sampling and Selection Logic (CUDA-compatible)

/**
 * Create default sampling configuration
 * @param config Pointer to store default configuration
 * @param strategy Primary sampling strategy to use
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_sampling_config_create_default(NNCPLSTMSamplingConfig* config,
                                                                           NNCPLSTMSamplingStrategy strategy);

/**
 * Create CUDA-compatible sampling configuration
 * @param config Pointer to store configuration
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_sampling_config_create_cuda_compatible(NNCPLSTMSamplingConfig* config);

/**
 * Validate sampling configuration
 * @param config Sampling configuration to validate
 * @return NNCP_PREDICTION_SUCCESS if valid, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_sampling_config_validate(const NNCPLSTMSamplingConfig* config);

/**
 * Greedy selection (select symbol with highest probability)
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_greedy_selection(const float* probabilities,
                                                             uint32_t n_symbols,
                                                             uint32_t* selected_symbol,
                                                             NNCPLSTMSamplingInfo* info);

/**
 * Multinomial sampling from probability distribution
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param random_value Random value [0, 1] for sampling
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_multinomial_sampling(const float* probabilities,
                                                                 uint32_t n_symbols,
                                                                 float random_value,
                                                                 uint32_t* selected_symbol,
                                                                 NNCPLSTMSamplingInfo* info);

/**
 * Temperature-scaled sampling
 * @param logits Input logits [n_symbols]
 * @param probabilities Working probability buffer [n_symbols]
 * @param n_symbols Number of symbols
 * @param temperature Temperature for scaling
 * @param random_value Random value [0, 1] for sampling
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_temperature_sampling(const float* logits,
                                                                 float* probabilities,
                                                                 uint32_t n_symbols,
                                                                 float temperature,
                                                                 float random_value,
                                                                 uint32_t* selected_symbol,
                                                                 NNCPLSTMSamplingInfo* info);

/**
 * Top-k sampling (filter to top k symbols, then sample)
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param k Number of top symbols to keep
 * @param random_value Random value [0, 1] for sampling
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_top_k_sampling(float* probabilities,
                                                           uint32_t n_symbols,
                                                           uint32_t k,
                                                           float random_value,
                                                           uint32_t* selected_symbol,
                                                           NNCPLSTMSamplingInfo* info);

/**
 * Top-p (nucleus) sampling
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param p_threshold Cumulative probability threshold
 * @param random_value Random value [0, 1] for sampling
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_top_p_sampling(float* probabilities,
                                                           uint32_t n_symbols,
                                                           float p_threshold,
                                                           float random_value,
                                                           uint32_t* selected_symbol,
                                                           NNCPLSTMSamplingInfo* info);

/**
 * Combined top-k and top-p sampling
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param k Number of top symbols for top-k filtering
 * @param p_threshold Cumulative probability threshold for top-p
 * @param random_value Random value [0, 1] for sampling
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_top_k_top_p_sampling(float* probabilities,
                                                                 uint32_t n_symbols,
                                                                 uint32_t k,
                                                                 float p_threshold,
                                                                 float random_value,
                                                                 uint32_t* selected_symbol,
                                                                 NNCPLSTMSamplingInfo* info);

/**
 * Confidence threshold selection
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param confidence_threshold Minimum confidence required
 * @param use_relative_confidence Use relative vs absolute confidence
 * @param fallback_symbol Fallback symbol if no symbol meets threshold
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_confidence_threshold_selection(const float* probabilities,
                                                                           uint32_t n_symbols,
                                                                           float confidence_threshold,
                                                                           bool use_relative_confidence,
                                                                           uint32_t fallback_symbol,
                                                                           uint32_t* selected_symbol,
                                                                           NNCPLSTMSamplingInfo* info);

/**
 * Single-step beam search selection
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param beam_width Number of candidates to consider
 * @param beam_alpha Length normalization parameter
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_beam_search_step(const float* probabilities,
                                                             uint32_t n_symbols,
                                                             uint32_t beam_width,
                                                             float beam_alpha,
                                                             uint32_t* selected_symbol,
                                                             NNCPLSTMSamplingInfo* info);

/**
 * Adaptive sampling based on entropy and distribution characteristics
 * @param logits Input logits [n_symbols]
 * @param probabilities Working probability buffer [n_symbols]
 * @param n_symbols Number of symbols
 * @param base_config Base sampling configuration
 * @param random_value Random value [0, 1] for sampling
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_adaptive_sampling(const float* logits,
                                                              float* probabilities,
                                                              uint32_t n_symbols,
                                                              const NNCPLSTMSamplingConfig* base_config,
                                                              float random_value,
                                                              uint32_t* selected_symbol,
                                                              NNCPLSTMSamplingInfo* info);

/**
 * Comprehensive sampling with multiple strategies and fallbacks
 * @param logits Input logits [n_symbols]
 * @param probabilities Working probability buffer [n_symbols]
 * @param n_symbols Number of symbols
 * @param config Sampling configuration
 * @param random_value Random value [0, 1] for sampling
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_comprehensive_sampling(const float* logits,
                                                                   float* probabilities,
                                                                   uint32_t n_symbols,
                                                                   const NNCPLSTMSamplingConfig* config,
                                                                   float random_value,
                                                                   uint32_t* selected_symbol,
                                                                   NNCPLSTMSamplingInfo* info);

/**
 * Select symbol using specified sampling strategy
 * @param logits Input logits [n_symbols] (optional, used for temperature sampling)
 * @param probabilities Probability distribution [n_symbols]
 * @param n_symbols Number of symbols
 * @param strategy Sampling strategy to use
 * @param config Sampling configuration
 * @param random_value Random value [0, 1] for sampling
 * @param selected_symbol Output selected symbol index
 * @param info Output sampling information (optional)
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_select_symbol(const float* logits,
                                                          float* probabilities,
                                                          uint32_t n_symbols,
                                                          NNCPLSTMSamplingStrategy strategy,
                                                          const NNCPLSTMSamplingConfig* config,
                                                          float random_value,
                                                          uint32_t* selected_symbol,
                                                          NNCPLSTMSamplingInfo* info);

// Configuration and Utility Functions

/**
 * Create default prediction configuration
 * @param config Pointer to store default configuration
 * @param n_symbols Vocabulary size
 * @param n_layers Number of LSTM layers
 * @param n_cells Hidden state dimension
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_config_create_default(NNCPLSTMPredictionConfig* config,
                                                                 uint32_t n_symbols,
                                                                 uint32_t n_layers,
                                                                 uint32_t n_cells);

/**
 * Create CUDA-compatible prediction configuration
 * @param config Pointer to store configuration
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_config_create_cuda_compatible(NNCPLSTMPredictionConfig* config);

/**
 * Validate prediction configuration
 * @param config Configuration to validate
 * @return NNCP_PREDICTION_SUCCESS if valid, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_config_validate(const NNCPLSTMPredictionConfig* config);

/**
 * Calculate memory requirements for prediction
 * @param config Prediction configuration
 * @param memory_mb Pointer to store memory requirement in MB
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_calculate_memory_requirements(const NNCPLSTMPredictionConfig* config,
                                                                         uint32_t* memory_mb);

/**
 * Get error message string
 * @param error_code Prediction error code
 * @return Human-readable error message
 */
const char* nncp_lstm_prediction_get_error_string(NNCPLSTMPredictionError error_code);

/**
 * Get prediction performance statistics
 * @param context Prediction context
 * @param total_predictions Pointer to store total predictions
 * @param avg_computation_time_ms Pointer to store average computation time
 * @param avg_entropy Pointer to store average entropy
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_get_stats(NNCPLSTMPredictionContext* context,
                                                      uint64_t* total_predictions,
                                                      float* avg_computation_time_ms,
                                                      float* avg_entropy);

// Result Management Functions

/**
 * Create batch prediction result
 * @param result Pointer to store created result
 * @param batch_size Number of sequences
 * @param seq_len Sequence length
 * @param n_symbols Vocabulary size
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_result_create(NNCPLSTMBatchPredictionResult** result,
                                                          uint32_t batch_size,
                                                          uint32_t seq_len,
                                                          uint32_t n_symbols);

/**
 * Destroy batch prediction result
 * @param result Result to destroy
 */
void nncp_lstm_prediction_result_destroy(NNCPLSTMBatchPredictionResult* result);

/**
 * Copy prediction result data
 * @param src Source result
 * @param dst Destination result
 * @return NNCP_PREDICTION_SUCCESS on success, error code on failure
 */
NNCPLSTMPredictionError nncp_lstm_prediction_result_copy(const NNCPLSTMBatchPredictionResult* src,
                                                        NNCPLSTMBatchPredictionResult* dst);

/**
 * Destroy prediction context
 * @param context Prediction context to destroy
 */
void nncp_lstm_prediction_destroy(NNCPLSTMPredictionContext* context);

#ifdef __cplusplus
}
#endif

#endif // NNCP_LSTM_PREDICTION_H
