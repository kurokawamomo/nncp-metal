#include "nncp_lstm_prediction.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#import <Metal/Metal.h>
#define NNCP_METAL_AVAILABLE 1
#else
#define NNCP_METAL_AVAILABLE 0
#endif

// Internal helper functions
static NNCPLSTMPredictionError allocate_prediction_buffers(NNCPLSTMPredictionContext* context);
static NNCPLSTMPredictionError initialize_metal_pipelines(NNCPLSTMPredictionContext* context);
static uint64_t get_timestamp_ns(void);
static void validate_prediction_outputs(const float* data, size_t size, const char* output_name);

// Error messages
static const char* prediction_error_messages[] = {
    "Success",
    "Invalid parameter", 
    "Memory allocation failed",
    "Computation failed",
    "Invalid dimensions",
    "Output buffer too small",
    "Metal operation failed"
};

// Core Prediction API Implementation

NNCPLSTMPredictionError nncp_lstm_prediction_create(NNCPLSTMPredictionContext** context,
                                                   const NNCPLSTMPredictionConfig* config,
                                                   void* device,
                                                   void* command_queue) {
    if (!context || !config) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (config->n_symbols == 0 || config->n_layers == 0 || config->n_cells == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    *context = (NNCPLSTMPredictionContext*)calloc(1, sizeof(NNCPLSTMPredictionContext));
    if (!*context) {
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    NNCPLSTMPredictionContext* ctx = *context;
    
    // Copy configuration
    ctx->config = *config;
    
    // Calculate total input dimension (sum of all layer outputs)
    uint32_t n_total_inputs;
    if (config->full_connect) {
        // Use all layers: n_embed_out * n_cells
        n_total_inputs = config->n_embed_out * config->n_cells;
    } else {
        // Use only top layer: n_cells
        n_total_inputs = config->n_cells;
    }
    
    // Initialize output layer structure
    ctx->output_layer.n_symbols = config->n_symbols;
    ctx->output_layer.n_total_inputs = n_total_inputs;
    ctx->output_layer.weights_loaded = false;
    ctx->output_layer.use_bias = true; // Default: use bias terms
    ctx->output_layer.weight_scale = sqrtf(12.0f / (config->n_cells * config->n_layers));
    
    // Store Metal context
    ctx->device = device;
    ctx->command_queue = command_queue;
    
    // Allocate computation buffers
    NNCPLSTMPredictionError error = allocate_prediction_buffers(ctx);
    if (error != NNCP_PREDICTION_SUCCESS) {
        nncp_lstm_prediction_destroy(ctx);
        *context = NULL;
        return error;
    }
    
    // Initialize Metal pipelines if device available
    if (device) {
        error = initialize_metal_pipelines(ctx);
        if (error != NNCP_PREDICTION_SUCCESS) {
            printf("WARNING: Failed to initialize Metal pipelines, falling back to CPU\\n");
        }
    }
    
    // Initialize performance tracking
    ctx->total_predictions = 0;
    ctx->total_compute_time_ns = 0;
    ctx->average_entropy = 0.0f;
    ctx->verbose_logging = config->verbose_logging;
    ctx->is_initialized = true;
    
    if (config->verbose_logging) {
        printf("NNCP LSTM Prediction Context created:\\n");
        printf("  Vocabulary size: %u\\n", config->n_symbols);
        printf("  Total input dimension: %u\\n", n_total_inputs);
        printf("  Full connect mode: %s\\n", config->full_connect ? "enabled" : "disabled");
        printf("  Metal acceleration: %s\\n", device ? "enabled" : "disabled");
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_init_output_weights(NNCPLSTMPredictionContext* context,
                                                               float weight_scale,
                                                               uint32_t random_seed) {
    if (!context) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Set random seed for reproducibility
    srand(random_seed);
    
    uint32_t n_weights = context->output_layer.n_symbols * context->output_layer.n_total_inputs;
    
    // Allocate weight matrix [n_symbols, n_total_inputs]
    context->output_layer.fc_weights = (float*)malloc(n_weights * sizeof(float));
    if (!context->output_layer.fc_weights) {
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize weights with Xavier/Glorot initialization
    float variance = weight_scale;
    float range = sqrtf(3.0f * variance);
    
    for (uint32_t i = 0; i < n_weights; i++) {
        // Uniform distribution [-range, range]
        float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        context->output_layer.fc_weights[i] = r * range;
    }
    
    // Initialize bias terms if enabled
    if (context->output_layer.use_bias) {
        context->output_layer.fc_bias = (float*)calloc(context->output_layer.n_symbols, sizeof(float));
        if (!context->output_layer.fc_bias) {
            free(context->output_layer.fc_weights);
            context->output_layer.fc_weights = NULL;
            return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
        }
        
        // Initialize bias to small values
        for (uint32_t i = 0; i < context->output_layer.n_symbols; i++) {
            context->output_layer.fc_bias[i] = 0.01f * ((float)rand() / RAND_MAX - 0.5f);
        }
    }
    
    // Create Metal buffers if available
    if (context->device) {
#if NNCP_METAL_AVAILABLE
        @autoreleasepool {
            id<MTLDevice> metalDevice = (__bridge id<MTLDevice>)context->device;
            
            size_t weights_size = n_weights * sizeof(float);
            id<MTLBuffer> weightsBuffer = [metalDevice newBufferWithBytes:context->output_layer.fc_weights
                                                                   length:weights_size
                                                                  options:MTLResourceStorageModeShared];
            
            if (weightsBuffer) {
                context->output_layer.fc_weights_buffer = (__bridge void*)weightsBuffer;
            }
            
            if (context->output_layer.fc_bias) {
                size_t bias_size = context->output_layer.n_symbols * sizeof(float);
                id<MTLBuffer> biasBuffer = [metalDevice newBufferWithBytes:context->output_layer.fc_bias
                                                                    length:bias_size
                                                                   options:MTLResourceStorageModeShared];
                
                if (biasBuffer) {
                    context->output_layer.fc_bias_buffer = (__bridge void*)biasBuffer;
                }
            }
        }
#endif
    }
    
    context->output_layer.weights_loaded = true;
    context->output_layer.weight_scale = weight_scale;
    
    if (context->verbose_logging) {
        printf("Initialized output weights: %u x %u, scale=%.6f\\n",
               context->output_layer.n_symbols, context->output_layer.n_total_inputs, weight_scale);
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_load_output_weights(NNCPLSTMPredictionContext* context,
                                                               const float* fc_weights,
                                                               const float* fc_bias) {
    if (!context || !fc_weights) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint32_t n_weights = context->output_layer.n_symbols * context->output_layer.n_total_inputs;
    
    // Allocate and copy weights
    context->output_layer.fc_weights = (float*)malloc(n_weights * sizeof(float));
    if (!context->output_layer.fc_weights) {
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    memcpy(context->output_layer.fc_weights, fc_weights, n_weights * sizeof(float));
    
    // Copy bias if provided
    if (fc_bias) {
        context->output_layer.fc_bias = (float*)malloc(context->output_layer.n_symbols * sizeof(float));
        if (!context->output_layer.fc_bias) {
            free(context->output_layer.fc_weights);
            context->output_layer.fc_weights = NULL;
            return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
        }
        
        memcpy(context->output_layer.fc_bias, fc_bias, 
               context->output_layer.n_symbols * sizeof(float));
        context->output_layer.use_bias = true;
    } else {
        context->output_layer.use_bias = false;
    }
    
    // Create Metal buffers if available
    if (context->device) {
#if NNCP_METAL_AVAILABLE
        @autoreleasepool {
            id<MTLDevice> metalDevice = (__bridge id<MTLDevice>)context->device;
            
            size_t weights_size = n_weights * sizeof(float);
            id<MTLBuffer> weightsBuffer = [metalDevice newBufferWithBytes:context->output_layer.fc_weights
                                                                   length:weights_size
                                                                  options:MTLResourceStorageModeShared];
            
            if (weightsBuffer) {
                context->output_layer.fc_weights_buffer = (__bridge void*)weightsBuffer;
            }
            
            if (context->output_layer.fc_bias) {
                size_t bias_size = context->output_layer.n_symbols * sizeof(float);
                id<MTLBuffer> biasBuffer = [metalDevice newBufferWithBytes:context->output_layer.fc_bias
                                                                    length:bias_size
                                                                   options:MTLResourceStorageModeShared];
                
                if (biasBuffer) {
                    context->output_layer.fc_bias_buffer = (__bridge void*)biasBuffer;
                }
            }
        }
#endif
    }
    
    context->output_layer.weights_loaded = true;
    
    if (context->verbose_logging) {
        printf("Loaded pre-trained output weights: %u x %u\\n",
               context->output_layer.n_symbols, context->output_layer.n_total_inputs);
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

// Softmax computation (CUDA-compatible)
NNCPLSTMPredictionError nncp_lstm_prediction_softmax(NNCPLSTMPredictionContext* context,
                                                    const float* logits,
                                                    float* probabilities,
                                                    float temperature) {
    if (!context || !logits || !probabilities) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (temperature <= 0.0f) {
        temperature = 1.0f;
    }
    
    uint32_t n_symbols = context->config.n_symbols;
    
    // Find maximum logit for numerical stability
    float max_logit = logits[0];
    for (uint32_t i = 1; i < n_symbols; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // Compute exp(logits/temperature - max_logit) and sum
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        float scaled_logit = (logits[i] - max_logit) / temperature;
        probabilities[i] = expf(scaled_logit);
        sum_exp += probabilities[i];
    }
    
    // Normalize to get probabilities
    if (sum_exp > 0.0f) {
        for (uint32_t i = 0; i < n_symbols; i++) {
            probabilities[i] /= sum_exp;
        }
    } else {
        // Fallback: uniform distribution
        float uniform_prob = 1.0f / n_symbols;
        for (uint32_t i = 0; i < n_symbols; i++) {
            probabilities[i] = uniform_prob;
        }
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

// Apply final linear transformation: logits = fc_weights * layer_outputs + fc_bias
NNCPLSTMPredictionError nncp_lstm_prediction_apply_final_linear(NNCPLSTMPredictionContext* context,
                                                              const float* layer_outputs,
                                                              uint32_t batch_size,
                                                              uint32_t seq_len,
                                                              float* logits) {
    if (!context || !layer_outputs || !logits) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (!context->output_layer.weights_loaded) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint32_t n_symbols = context->output_layer.n_symbols;
    uint32_t n_inputs = context->output_layer.n_total_inputs;
    uint32_t batch_seq = batch_size * seq_len;
    
    // Matrix multiplication: logits = weights @ layer_outputs
    // weights: [n_symbols, n_inputs], layer_outputs: [n_inputs, batch_seq]
    // logits: [n_symbols, batch_seq]
    
    for (uint32_t symbol = 0; symbol < n_symbols; symbol++) {
        for (uint32_t pos = 0; pos < batch_seq; pos++) {
            float sum = 0.0f;
            
            // Dot product of weight row with input column
            for (uint32_t input = 0; input < n_inputs; input++) {
                float weight = context->output_layer.fc_weights[symbol * n_inputs + input];
                float input_val = layer_outputs[input * batch_seq + pos];
                sum += weight * input_val;
            }
            
            // Add bias if enabled
            if (context->output_layer.use_bias) {
                sum += context->output_layer.fc_bias[symbol];
            }
            
            logits[symbol * batch_seq + pos] = sum;
        }
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

// Generate single prediction from layer outputs
NNCPLSTMPredictionError nncp_lstm_prediction_generate_single(NNCPLSTMPredictionContext* context,
                                                           const float* layer_outputs,
                                                           NNCPLSTMPredictionResult* result) {
    if (!context || !layer_outputs || !result) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    uint32_t n_symbols = context->config.n_symbols;
    uint32_t n_inputs = context->output_layer.n_total_inputs;
    
    // Allocate temporary buffers
    float* logits = (float*)malloc(n_symbols * sizeof(float));
    float* probabilities = (float*)malloc(n_symbols * sizeof(float));
    
    if (!logits || !probabilities) {
        free(logits);
        free(probabilities);
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Compute logits: logits = fc_weights @ layer_outputs + fc_bias
    for (uint32_t symbol = 0; symbol < n_symbols; symbol++) {
        float sum = 0.0f;
        
        for (uint32_t input = 0; input < n_inputs; input++) {
            sum += context->output_layer.fc_weights[symbol * n_inputs + input] * layer_outputs[input];
        }
        
        if (context->output_layer.use_bias) {
            sum += context->output_layer.fc_bias[symbol];
        }
        
        logits[symbol] = sum;
    }
    
    // Apply softmax to get probabilities
    NNCPLSTMPredictionError error = nncp_lstm_prediction_softmax(context, logits, probabilities, 
                                                               context->config.temperature);
    if (error != NNCP_PREDICTION_SUCCESS) {
        free(logits);
        free(probabilities);
        return error;
    }
    
    // Find most likely symbol and confidence
    uint32_t best_symbol = 0;
    float max_prob = probabilities[0];
    for (uint32_t i = 1; i < n_symbols; i++) {
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            best_symbol = i;
        }
    }
    
    // Calculate entropy
    float entropy = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        if (probabilities[i] > 1e-10f) {
            entropy -= probabilities[i] * logf(probabilities[i]);
        }
    }
    
    // Fill result structure
    result->logits = logits;
    result->probabilities = probabilities;
    result->predicted_symbol = best_symbol;
    result->prediction_confidence = max_prob;
    result->entropy = entropy;
    result->computation_time_ns = get_timestamp_ns() - start_time;
    
    // Update performance tracking
    context->total_predictions++;
    context->total_compute_time_ns += result->computation_time_ns;
    context->average_entropy = (context->average_entropy * (context->total_predictions - 1) + entropy) / 
                              context->total_predictions;
    
    if (context->verbose_logging) {
        printf("Prediction: symbol=%u, confidence=%.4f, entropy=%.4f\\n",
               best_symbol, max_prob, entropy);
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

// Byte Prediction Logic (CUDA-compatible)

NNCPLSTMPredictionError nncp_lstm_prediction_sample(const float* probabilities,
                                                   uint32_t n_symbols,
                                                   float random_value,
                                                   uint32_t* sampled_symbol) {
    if (!probabilities || !sampled_symbol || random_value < 0.0f || random_value > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Cumulative distribution sampling (CUDA behavior)
    float cumulative = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        cumulative += probabilities[i];
        if (random_value <= cumulative) {
            *sampled_symbol = i;
            return NNCP_PREDICTION_SUCCESS;
        }
    }
    
    // Fallback: return last symbol
    *sampled_symbol = n_symbols - 1;
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_get_best(const float* probabilities,
                                                     uint32_t n_symbols,
                                                     uint32_t* best_symbol,
                                                     float* confidence) {
    if (!probabilities || !best_symbol || !confidence) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint32_t best_idx = 0;
    float max_prob = probabilities[0];
    
    for (uint32_t i = 1; i < n_symbols; i++) {
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            best_idx = i;
        }
    }
    
    *best_symbol = best_idx;
    *confidence = max_prob;
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_calculate_entropy(const float* probabilities,
                                                             uint32_t n_symbols,
                                                             float* entropy) {
    if (!probabilities || !entropy) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    float h = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        if (probabilities[i] > 1e-10f) {
            h -= probabilities[i] * logf(probabilities[i]);
        }
    }
    
    *entropy = h;
    return NNCP_PREDICTION_SUCCESS;
}

// Byte prediction for compression (CUDA-compatible)
NNCPLSTMPredictionError nncp_lstm_prediction_predict_byte(NNCPLSTMPredictionContext* context,
                                                         const float* layer_outputs,
                                                         uint8_t* predicted_byte,
                                                         float* prediction_confidence,
                                                         float* prediction_entropy) {
    if (!context || !layer_outputs || !predicted_byte) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Generate single prediction
    NNCPLSTMPredictionResult result;
    NNCPLSTMPredictionError error = nncp_lstm_prediction_generate_single(context, layer_outputs, &result);
    if (error != NNCP_PREDICTION_SUCCESS) {
        return error;
    }
    
    // Extract byte prediction (symbol should be 0-255 for byte vocabulary)
    if (result.predicted_symbol > 255) {
        free(result.logits);
        free(result.probabilities);
        return NNCP_PREDICTION_ERROR_INVALID_DIMENSIONS;
    }
    
    *predicted_byte = (uint8_t)result.predicted_symbol;
    
    if (prediction_confidence) {
        *prediction_confidence = result.prediction_confidence;
    }
    
    if (prediction_entropy) {
        *prediction_entropy = result.entropy;
    }
    
    // Clean up result buffers
    free(result.logits);
    free(result.probabilities);
    
    return NNCP_PREDICTION_SUCCESS;
}

// Batch byte prediction for sequences
NNCPLSTMPredictionError nncp_lstm_prediction_predict_byte_sequence(NNCPLSTMPredictionContext* context,
                                                                  const float* layer_outputs_sequence,
                                                                  uint32_t sequence_length,
                                                                  uint8_t* predicted_bytes,
                                                                  float* prediction_confidences,
                                                                  float* prediction_entropies) {
    if (!context || !layer_outputs_sequence || !predicted_bytes || sequence_length == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint32_t n_inputs = context->output_layer.n_total_inputs;
    
    for (uint32_t seq_pos = 0; seq_pos < sequence_length; seq_pos++) {
        const float* current_outputs = &layer_outputs_sequence[seq_pos * n_inputs];
        
        float confidence = 0.0f;
        float entropy = 0.0f;
        
        NNCPLSTMPredictionError error = nncp_lstm_prediction_predict_byte(
            context, current_outputs, &predicted_bytes[seq_pos], 
            prediction_confidences ? &confidence : NULL,
            prediction_entropies ? &entropy : NULL
        );
        
        if (error != NNCP_PREDICTION_SUCCESS) {
            return error;
        }
        
        if (prediction_confidences) {
            prediction_confidences[seq_pos] = confidence;
        }
        
        if (prediction_entropies) {
            prediction_entropies[seq_pos] = entropy;
        }
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

// Top-K prediction for improved compression
NNCPLSTMPredictionError nncp_lstm_prediction_get_top_k(const float* probabilities,
                                                       uint32_t n_symbols,
                                                       uint32_t k,
                                                       uint32_t* top_symbols,
                                                       float* top_probabilities) {
    if (!probabilities || !top_symbols || !top_probabilities || k == 0 || k > n_symbols) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Simple selection sort for top-k (efficient for small k)
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best_idx = 0;
        float best_prob = -1.0f;
        
        // Find best probability not already selected
        for (uint32_t j = 0; j < n_symbols; j++) {
            bool already_selected = false;
            for (uint32_t m = 0; m < i; m++) {
                if (top_symbols[m] == j) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && probabilities[j] > best_prob) {
                best_prob = probabilities[j];
                best_idx = j;
            }
        }
        
        top_symbols[i] = best_idx;
        top_probabilities[i] = best_prob;
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

// Probability Distribution Processing (CUDA-compatible)

NNCPLSTMPredictionError nncp_lstm_prediction_normalize_distribution(float* probabilities,
                                                                   uint32_t n_symbols) {
    if (!probabilities) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Calculate sum
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        sum += probabilities[i];
    }
    
    // Normalize if sum is valid
    if (sum > 1e-10f) {
        for (uint32_t i = 0; i < n_symbols; i++) {
            probabilities[i] /= sum;
        }
    } else {
        // Fallback: uniform distribution
        float uniform_prob = 1.0f / n_symbols;
        for (uint32_t i = 0; i < n_symbols; i++) {
            probabilities[i] = uniform_prob;
        }
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_apply_temperature(const float* logits,
                                                              float* scaled_logits,
                                                              uint32_t n_symbols,
                                                              float temperature) {
    if (!logits || !scaled_logits || temperature <= 0.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Apply temperature scaling: logits / temperature
    for (uint32_t i = 0; i < n_symbols; i++) {
        scaled_logits[i] = logits[i] / temperature;
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_apply_top_p_filtering(float* probabilities,
                                                                  uint32_t n_symbols,
                                                                  float p_threshold) {
    if (!probabilities || p_threshold <= 0.0f || p_threshold > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Create sorted indices array
    typedef struct {
        uint32_t index;
        float probability;
    } IndexProbPair;
    
    IndexProbPair* pairs = (IndexProbPair*)malloc(n_symbols * sizeof(IndexProbPair));
    if (!pairs) {
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize pairs
    for (uint32_t i = 0; i < n_symbols; i++) {
        pairs[i].index = i;
        pairs[i].probability = probabilities[i];
    }
    
    // Simple bubble sort by probability (descending)
    for (uint32_t i = 0; i < n_symbols - 1; i++) {
        for (uint32_t j = 0; j < n_symbols - 1 - i; j++) {
            if (pairs[j].probability < pairs[j + 1].probability) {
                IndexProbPair temp = pairs[j];
                pairs[j] = pairs[j + 1];
                pairs[j + 1] = temp;
            }
        }
    }
    
    // Find cutoff point for top-p
    float cumulative = 0.0f;
    uint32_t cutoff = n_symbols;
    
    for (uint32_t i = 0; i < n_symbols; i++) {
        cumulative += pairs[i].probability;
        if (cumulative >= p_threshold) {
            cutoff = i + 1;
            break;
        }
    }
    
    // Zero out probabilities below cutoff
    for (uint32_t i = 0; i < n_symbols; i++) {
        probabilities[i] = 0.0f;
    }
    
    // Restore top-p probabilities
    for (uint32_t i = 0; i < cutoff; i++) {
        probabilities[pairs[i].index] = pairs[i].probability;
    }
    
    free(pairs);
    
    // Renormalize
    return nncp_lstm_prediction_normalize_distribution(probabilities, n_symbols);
}

NNCPLSTMPredictionError nncp_lstm_prediction_apply_top_k_filtering(float* probabilities,
                                                                  uint32_t n_symbols,
                                                                  uint32_t k) {
    if (!probabilities || k == 0 || k > n_symbols) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Find k-th largest probability using partial sort
    float* sorted_probs = (float*)malloc(n_symbols * sizeof(float));
    uint32_t* indices = (uint32_t*)malloc(n_symbols * sizeof(uint32_t));
    
    if (!sorted_probs || !indices) {
        free(sorted_probs);
        free(indices);
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy and initialize indices
    for (uint32_t i = 0; i < n_symbols; i++) {
        sorted_probs[i] = probabilities[i];
        indices[i] = i;
    }
    
    // Partial sort to find top k elements
    for (uint32_t i = 0; i < k; i++) {
        uint32_t max_idx = i;
        for (uint32_t j = i + 1; j < n_symbols; j++) {
            if (sorted_probs[j] > sorted_probs[max_idx]) {
                max_idx = j;
            }
        }
        
        // Swap probabilities and indices
        if (max_idx != i) {
            float temp_prob = sorted_probs[i];
            sorted_probs[i] = sorted_probs[max_idx];
            sorted_probs[max_idx] = temp_prob;
            
            uint32_t temp_idx = indices[i];
            indices[i] = indices[max_idx];
            indices[max_idx] = temp_idx;
        }
    }
    
    // Get k-th largest probability as threshold
    float threshold = sorted_probs[k - 1];
    
    // Zero out probabilities below threshold
    for (uint32_t i = 0; i < n_symbols; i++) {
        if (probabilities[i] < threshold) {
            probabilities[i] = 0.0f;
        }
    }
    
    free(sorted_probs);
    free(indices);
    
    // Renormalize
    return nncp_lstm_prediction_normalize_distribution(probabilities, n_symbols);
}

NNCPLSTMPredictionError nncp_lstm_prediction_compute_perplexity(const float* probabilities,
                                                               const uint32_t* target_symbols,
                                                               uint32_t sequence_length,
                                                               uint32_t n_symbols,
                                                               float* perplexity) {
    if (!probabilities || !target_symbols || !perplexity || sequence_length == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    double log_likelihood = 0.0;
    
    for (uint32_t seq_pos = 0; seq_pos < sequence_length; seq_pos++) {
        uint32_t target = target_symbols[seq_pos];
        
        if (target >= n_symbols) {
            return NNCP_PREDICTION_ERROR_INVALID_DIMENSIONS;
        }
        
        float prob = probabilities[seq_pos * n_symbols + target];
        
        // Avoid log(0) by using small epsilon
        if (prob < 1e-10f) {
            prob = 1e-10f;
        }
        
        log_likelihood += log((double)prob);
    }
    
    // Perplexity = exp(-log_likelihood / sequence_length)
    double avg_log_likelihood = log_likelihood / sequence_length;
    *perplexity = (float)exp(-avg_log_likelihood);
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_compute_cross_entropy(const float* probabilities,
                                                                  const uint32_t* target_symbols,
                                                                  uint32_t sequence_length,
                                                                  uint32_t n_symbols,
                                                                  float* cross_entropy) {
    if (!probabilities || !target_symbols || !cross_entropy || sequence_length == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    double total_loss = 0.0;
    
    for (uint32_t seq_pos = 0; seq_pos < sequence_length; seq_pos++) {
        uint32_t target = target_symbols[seq_pos];
        
        if (target >= n_symbols) {
            return NNCP_PREDICTION_ERROR_INVALID_DIMENSIONS;
        }
        
        float prob = probabilities[seq_pos * n_symbols + target];
        
        // Avoid log(0) by using small epsilon
        if (prob < 1e-10f) {
            prob = 1e-10f;
        }
        
        total_loss -= log((double)prob);
    }
    
    *cross_entropy = (float)(total_loss / sequence_length);
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_distribution_statistics(const float* probabilities,
                                                                    uint32_t n_symbols,
                                                                    float* entropy,
                                                                    float* max_prob,
                                                                    uint32_t* max_symbol,
                                                                    float* variance) {
    if (!probabilities) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Calculate entropy
    if (entropy) {
        float h = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (probabilities[i] > 1e-10f) {
                h -= probabilities[i] * logf(probabilities[i]);
            }
        }
        *entropy = h;
    }
    
    // Find maximum probability and symbol
    if (max_prob || max_symbol) {
        float best_prob = probabilities[0];
        uint32_t best_idx = 0;
        
        for (uint32_t i = 1; i < n_symbols; i++) {
            if (probabilities[i] > best_prob) {
                best_prob = probabilities[i];
                best_idx = i;
            }
        }
        
        if (max_prob) *max_prob = best_prob;
        if (max_symbol) *max_symbol = best_idx;
    }
    
    // Calculate variance
    if (variance) {
        // Calculate mean (should be 1/n_symbols for uniform distribution)
        float mean = 1.0f / n_symbols;
        float var = 0.0f;
        
        for (uint32_t i = 0; i < n_symbols; i++) {
            float diff = probabilities[i] - mean;
            var += diff * diff;
        }
        
        *variance = var / n_symbols;
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

// Prediction Confidence Scoring (CUDA-compatible)

NNCPLSTMPredictionError nncp_lstm_prediction_compute_confidence_score(const float* probabilities,
                                                                     uint32_t n_symbols,
                                                                     uint32_t predicted_symbol,
                                                                     float* confidence_score,
                                                                     float* relative_confidence) {
    if (!probabilities || !confidence_score || predicted_symbol >= n_symbols) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Basic confidence is the probability of predicted symbol
    *confidence_score = probabilities[predicted_symbol];
    
    if (relative_confidence) {
        // Find second highest probability for relative confidence
        float max_prob = 0.0f;
        float second_max = 0.0f;
        
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (probabilities[i] > max_prob) {
                second_max = max_prob;
                max_prob = probabilities[i];
            } else if (probabilities[i] > second_max) {
                second_max = probabilities[i];
            }
        }
        
        // Relative confidence = (max - second_max) / max
        if (max_prob > 1e-10f) {
            *relative_confidence = (max_prob - second_max) / max_prob;
        } else {
            *relative_confidence = 0.0f;
        }
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_compute_prediction_margin(const float* probabilities,
                                                                      uint32_t n_symbols,
                                                                      float* margin,
                                                                      float* gap_ratio) {
    if (!probabilities || !margin) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Find top two probabilities
    float first_max = 0.0f;
    float second_max = 0.0f;
    uint32_t first_idx = 0;
    
    for (uint32_t i = 0; i < n_symbols; i++) {
        if (probabilities[i] > first_max) {
            second_max = first_max;
            first_max = probabilities[i];
            first_idx = i;
        } else if (probabilities[i] > second_max) {
            second_max = probabilities[i];
        }
    }
    
    // Margin is absolute difference
    *margin = first_max - second_max;
    
    if (gap_ratio) {
        // Gap ratio = margin / (first + second)
        float sum = first_max + second_max;
        if (sum > 1e-10f) {
            *gap_ratio = *margin / sum;
        } else {
            *gap_ratio = 0.0f;
        }
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_compute_entropy_based_confidence(const float* probabilities,
                                                                             uint32_t n_symbols,
                                                                             float* normalized_confidence) {
    if (!probabilities || !normalized_confidence) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Compute entropy
    float entropy = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        if (probabilities[i] > 1e-10f) {
            entropy -= probabilities[i] * logf(probabilities[i]);
        }
    }
    
    // Maximum entropy for uniform distribution
    float max_entropy = logf((float)n_symbols);
    
    // Normalized confidence = 1 - (entropy / max_entropy)
    if (max_entropy > 1e-10f) {
        *normalized_confidence = 1.0f - (entropy / max_entropy);
    } else {
        *normalized_confidence = 1.0f;
    }
    
    // Ensure valid range [0, 1]
    if (*normalized_confidence < 0.0f) *normalized_confidence = 0.0f;
    if (*normalized_confidence > 1.0f) *normalized_confidence = 1.0f;
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_compute_concentration_measure(const float* probabilities,
                                                                          uint32_t n_symbols,
                                                                          float* gini_coefficient,
                                                                          float* effective_vocabulary_size) {
    if (!probabilities) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (gini_coefficient) {
        // Compute Gini coefficient for concentration measure
        float* sorted_probs = (float*)malloc(n_symbols * sizeof(float));
        if (!sorted_probs) {
            return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
        }
        
        // Copy and sort probabilities
        for (uint32_t i = 0; i < n_symbols; i++) {
            sorted_probs[i] = probabilities[i];
        }
        
        // Simple bubble sort
        for (uint32_t i = 0; i < n_symbols - 1; i++) {
            for (uint32_t j = 0; j < n_symbols - 1 - i; j++) {
                if (sorted_probs[j] > sorted_probs[j + 1]) {
                    float temp = sorted_probs[j];
                    sorted_probs[j] = sorted_probs[j + 1];
                    sorted_probs[j + 1] = temp;
                }
            }
        }
        
        // Calculate Gini coefficient
        float gini = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            gini += (2.0f * (i + 1) - n_symbols - 1.0f) * sorted_probs[i];
        }
        gini /= (n_symbols - 1.0f);
        
        *gini_coefficient = gini;
        free(sorted_probs);
    }
    
    if (effective_vocabulary_size) {
        // Effective vocabulary size = exp(entropy)
        float entropy = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (probabilities[i] > 1e-10f) {
                entropy -= probabilities[i] * logf(probabilities[i]);
            }
        }
        *effective_vocabulary_size = expf(entropy);
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_compute_calibrated_confidence(const float* probabilities,
                                                                          uint32_t n_symbols,
                                                                          uint32_t predicted_symbol,
                                                                          float temperature,
                                                                          float* calibrated_confidence) {
    if (!probabilities || !calibrated_confidence || predicted_symbol >= n_symbols || temperature <= 0.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Apply temperature calibration to the predicted probability
    float raw_prob = probabilities[predicted_symbol];
    
    // Temperature calibration: p_calibrated = sigmoid(logit / temperature)
    // logit = log(p / (1 - p))
    if (raw_prob >= 1.0f - 1e-10f) {
        raw_prob = 1.0f - 1e-10f;  // Avoid log(0)
    }
    if (raw_prob <= 1e-10f) {
        raw_prob = 1e-10f;  // Avoid log(inf)
    }
    
    float logit = logf(raw_prob / (1.0f - raw_prob));
    float calibrated_logit = logit / temperature;
    
    // Convert back to probability using sigmoid
    *calibrated_confidence = 1.0f / (1.0f + expf(-calibrated_logit));
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_confidence_ensemble(const float* individual_confidences,
                                                                const float* weights,
                                                                uint32_t n_methods,
                                                                float* ensemble_confidence) {
    if (!individual_confidences || !ensemble_confidence || n_methods == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    for (uint32_t i = 0; i < n_methods; i++) {
        float weight = weights ? weights[i] : 1.0f;
        weighted_sum += individual_confidences[i] * weight;
        weight_sum += weight;
    }
    
    if (weight_sum > 1e-10f) {
        *ensemble_confidence = weighted_sum / weight_sum;
    } else {
        *ensemble_confidence = 0.0f;
    }
    
    // Ensure valid range [0, 1]
    if (*ensemble_confidence < 0.0f) *ensemble_confidence = 0.0f;
    if (*ensemble_confidence > 1.0f) *ensemble_confidence = 1.0f;
    
    return NNCP_PREDICTION_SUCCESS;
}

// CUDA-compatible comprehensive confidence scoring
NNCPLSTMPredictionError nncp_lstm_prediction_compute_comprehensive_confidence(const float* probabilities,
                                                                             uint32_t n_symbols,
                                                                             uint32_t predicted_symbol,
                                                                             float temperature,
                                                                             NNCPLSTMConfidenceScores* scores) {
    if (!probabilities || !scores || predicted_symbol >= n_symbols) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    NNCPLSTMPredictionError error;
    
    // Basic confidence score
    error = nncp_lstm_prediction_compute_confidence_score(probabilities, n_symbols, predicted_symbol,
                                                         &scores->basic_confidence, &scores->relative_confidence);
    if (error != NNCP_PREDICTION_SUCCESS) return error;
    
    // Prediction margin
    error = nncp_lstm_prediction_compute_prediction_margin(probabilities, n_symbols,
                                                          &scores->prediction_margin, &scores->gap_ratio);
    if (error != NNCP_PREDICTION_SUCCESS) return error;
    
    // Entropy-based confidence
    error = nncp_lstm_prediction_compute_entropy_based_confidence(probabilities, n_symbols,
                                                                 &scores->entropy_confidence);
    if (error != NNCP_PREDICTION_SUCCESS) return error;
    
    // Concentration measures
    error = nncp_lstm_prediction_compute_concentration_measure(probabilities, n_symbols,
                                                              &scores->gini_coefficient, &scores->effective_vocab_size);
    if (error != NNCP_PREDICTION_SUCCESS) return error;
    
    // Calibrated confidence
    if (temperature > 0.0f) {
        error = nncp_lstm_prediction_compute_calibrated_confidence(probabilities, n_symbols, predicted_symbol,
                                                                  temperature, &scores->calibrated_confidence);
        if (error != NNCP_PREDICTION_SUCCESS) return error;
    } else {
        scores->calibrated_confidence = scores->basic_confidence;
    }
    
    // Ensemble confidence (weighted average of multiple measures)
    float individual_scores[] = {
        scores->basic_confidence,
        scores->relative_confidence,
        scores->entropy_confidence,
        scores->calibrated_confidence
    };
    float weights[] = {0.3f, 0.2f, 0.3f, 0.2f};  // CUDA-compatible weights
    
    error = nncp_lstm_prediction_confidence_ensemble(individual_scores, weights, 4, &scores->ensemble_confidence);
    if (error != NNCP_PREDICTION_SUCCESS) return error;
    
    return NNCP_PREDICTION_SUCCESS;
}

// Sampling and Selection Logic (CUDA-compatible)

NNCPLSTMPredictionError nncp_lstm_prediction_greedy_selection(const float* probabilities,
                                                             uint32_t n_symbols,
                                                             uint32_t* selected_symbol) {
    if (!probabilities || !selected_symbol) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Find symbol with maximum probability
    uint32_t best_idx = 0;
    float max_prob = probabilities[0];
    
    for (uint32_t i = 1; i < n_symbols; i++) {
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            best_idx = i;
        }
    }
    
    *selected_symbol = best_idx;
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_multinomial_sampling(const float* probabilities,
                                                                 uint32_t n_symbols,
                                                                 float random_value,
                                                                 uint32_t* sampled_symbol) {
    if (!probabilities || !sampled_symbol || random_value < 0.0f || random_value > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Cumulative distribution sampling (same as existing nncp_lstm_prediction_sample)
    float cumulative = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        cumulative += probabilities[i];
        if (random_value <= cumulative) {
            *sampled_symbol = i;
            return NNCP_PREDICTION_SUCCESS;
        }
    }
    
    // Fallback: return last symbol
    *sampled_symbol = n_symbols - 1;
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_temperature_sampling(const float* logits,
                                                                 uint32_t n_symbols,
                                                                 float temperature,
                                                                 float random_value,
                                                                 uint32_t* sampled_symbol) {
    if (!logits || !sampled_symbol || temperature <= 0.0f || random_value < 0.0f || random_value > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Apply temperature scaling and convert to probabilities
    float* scaled_logits = (float*)malloc(n_symbols * sizeof(float));
    float* probabilities = (float*)malloc(n_symbols * sizeof(float));
    
    if (!scaled_logits || !probabilities) {
        free(scaled_logits);
        free(probabilities);
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Apply temperature scaling
    for (uint32_t i = 0; i < n_symbols; i++) {
        scaled_logits[i] = logits[i] / temperature;
    }
    
    // Convert to probabilities with numerical stability
    float max_logit = scaled_logits[0];
    for (uint32_t i = 1; i < n_symbols; i++) {
        if (scaled_logits[i] > max_logit) {
            max_logit = scaled_logits[i];
        }
    }
    
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        probabilities[i] = expf(scaled_logits[i] - max_logit);
        sum_exp += probabilities[i];
    }
    
    // Normalize
    for (uint32_t i = 0; i < n_symbols; i++) {
        probabilities[i] /= sum_exp;
    }
    
    // Sample from the temperature-scaled distribution
    NNCPLSTMPredictionError error = nncp_lstm_prediction_multinomial_sampling(probabilities, n_symbols, 
                                                                             random_value, sampled_symbol);
    
    free(scaled_logits);
    free(probabilities);
    
    return error;
}

NNCPLSTMPredictionError nncp_lstm_prediction_top_k_sampling(const float* probabilities,
                                                           uint32_t n_symbols,
                                                           uint32_t k,
                                                           float random_value,
                                                           uint32_t* sampled_symbol) {
    if (!probabilities || !sampled_symbol || k == 0 || k > n_symbols || 
        random_value < 0.0f || random_value > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Create filtered distribution
    float* filtered_probs = (float*)calloc(n_symbols, sizeof(float));
    if (!filtered_probs) {
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy original probabilities
    for (uint32_t i = 0; i < n_symbols; i++) {
        filtered_probs[i] = probabilities[i];
    }
    
    // Apply top-k filtering
    NNCPLSTMPredictionError error = nncp_lstm_prediction_apply_top_k_filtering(filtered_probs, n_symbols, k);
    if (error != NNCP_PREDICTION_SUCCESS) {
        free(filtered_probs);
        return error;
    }
    
    // Sample from filtered distribution
    error = nncp_lstm_prediction_multinomial_sampling(filtered_probs, n_symbols, random_value, sampled_symbol);
    
    free(filtered_probs);
    return error;
}

NNCPLSTMPredictionError nncp_lstm_prediction_top_p_sampling(const float* probabilities,
                                                           uint32_t n_symbols,
                                                           float p_threshold,
                                                           float random_value,
                                                           uint32_t* sampled_symbol) {
    if (!probabilities || !sampled_symbol || p_threshold <= 0.0f || p_threshold > 1.0f ||
        random_value < 0.0f || random_value > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Create filtered distribution
    float* filtered_probs = (float*)malloc(n_symbols * sizeof(float));
    if (!filtered_probs) {
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy original probabilities
    for (uint32_t i = 0; i < n_symbols; i++) {
        filtered_probs[i] = probabilities[i];
    }
    
    // Apply top-p filtering
    NNCPLSTMPredictionError error = nncp_lstm_prediction_apply_top_p_filtering(filtered_probs, n_symbols, p_threshold);
    if (error != NNCP_PREDICTION_SUCCESS) {
        free(filtered_probs);
        return error;
    }
    
    // Sample from filtered distribution
    error = nncp_lstm_prediction_multinomial_sampling(filtered_probs, n_symbols, random_value, sampled_symbol);
    
    free(filtered_probs);
    return error;
}

NNCPLSTMPredictionError nncp_lstm_prediction_confidence_threshold_selection(const float* probabilities,
                                                                           uint32_t n_symbols,
                                                                           float confidence_threshold,
                                                                           uint32_t* selected_symbol,
                                                                           bool* meets_threshold) {
    if (!probabilities || !selected_symbol || !meets_threshold || 
        confidence_threshold < 0.0f || confidence_threshold > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Find best symbol and its confidence
    uint32_t best_idx = 0;
    float max_prob = probabilities[0];
    
    for (uint32_t i = 1; i < n_symbols; i++) {
        if (probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            best_idx = i;
        }
    }
    
    *selected_symbol = best_idx;
    *meets_threshold = (max_prob >= confidence_threshold);
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_beam_search_step(const float* probabilities,
                                                             uint32_t n_symbols,
                                                             const float* current_beam_scores,
                                                             const uint32_t* current_beam_symbols,
                                                             uint32_t beam_size,
                                                             float* next_beam_scores,
                                                             uint32_t* next_beam_symbols,
                                                             uint32_t* next_beam_parents) {
    if (!probabilities || !current_beam_scores || !current_beam_symbols ||
        !next_beam_scores || !next_beam_symbols || !next_beam_parents || beam_size == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Generate all candidate expansions
    typedef struct {
        float score;
        uint32_t symbol;
        uint32_t parent;
    } BeamCandidate;
    
    uint32_t total_candidates = beam_size * n_symbols;
    BeamCandidate* candidates = (BeamCandidate*)malloc(total_candidates * sizeof(BeamCandidate));
    if (!candidates) {
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Generate all beam expansions
    uint32_t candidate_idx = 0;
    for (uint32_t beam = 0; beam < beam_size; beam++) {
        for (uint32_t symbol = 0; symbol < n_symbols; symbol++) {
            candidates[candidate_idx].score = current_beam_scores[beam] + logf(probabilities[symbol] + 1e-10f);
            candidates[candidate_idx].symbol = symbol;
            candidates[candidate_idx].parent = beam;
            candidate_idx++;
        }
    }
    
    // Sort candidates by score (descending)
    for (uint32_t i = 0; i < total_candidates - 1; i++) {
        for (uint32_t j = 0; j < total_candidates - 1 - i; j++) {
            if (candidates[j].score < candidates[j + 1].score) {
                BeamCandidate temp = candidates[j];
                candidates[j] = candidates[j + 1];
                candidates[j + 1] = temp;
            }
        }
    }
    
    // Select top beam_size candidates
    for (uint32_t i = 0; i < beam_size && i < total_candidates; i++) {
        next_beam_scores[i] = candidates[i].score;
        next_beam_symbols[i] = candidates[i].symbol;
        next_beam_parents[i] = candidates[i].parent;
    }
    
    free(candidates);
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_adaptive_sampling(const float* probabilities,
                                                              uint32_t n_symbols,
                                                              float entropy_threshold,
                                                              float random_value,
                                                              uint32_t* selected_symbol,
                                                              bool* used_sampling) {
    if (!probabilities || !selected_symbol || !used_sampling ||
        entropy_threshold < 0.0f || random_value < 0.0f || random_value > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Calculate entropy
    float entropy = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        if (probabilities[i] > 1e-10f) {
            entropy -= probabilities[i] * logf(probabilities[i]);
        }
    }
    
    if (entropy < entropy_threshold) {
        // Low entropy: use greedy selection
        *used_sampling = false;
        return nncp_lstm_prediction_greedy_selection(probabilities, n_symbols, selected_symbol);
    } else {
        // High entropy: use sampling
        *used_sampling = true;
        return nncp_lstm_prediction_multinomial_sampling(probabilities, n_symbols, random_value, selected_symbol);
    }
}

// CUDA-compatible comprehensive sampling strategy
NNCPLSTMPredictionError nncp_lstm_prediction_comprehensive_sampling(const float* logits,
                                                                   float* probabilities,
                                                                   uint32_t n_symbols,
                                                                   const NNCPLSTMSamplingConfig* config,
                                                                   float random_value,
                                                                   uint32_t* selected_symbol,
                                                                   NNCPLSTMSamplingInfo* info) {
    if (!probabilities || !selected_symbol || !config) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    NNCPLSTMPredictionError error;
    
    switch (config->strategy) {
        case NNCP_SAMPLING_GREEDY:
            error = nncp_lstm_prediction_greedy_selection(probabilities, n_symbols, selected_symbol);
            if (info) {
                info->strategy_used = NNCP_SAMPLING_GREEDY;
                info->temperature_used = 0.0f;
                info->fallback_used = false;
            }
            break;
            
        case NNCP_SAMPLING_MULTINOMIAL:
            error = nncp_lstm_prediction_multinomial_sampling(probabilities, n_symbols, random_value, selected_symbol);
            if (info) {
                info->strategy_used = NNCP_SAMPLING_MULTINOMIAL;
                info->temperature_used = 1.0f;
                info->fallback_used = false;
            }
            break;
            
        case NNCP_SAMPLING_TEMPERATURE:
            if (!logits) {
                return NNCP_PREDICTION_ERROR_INVALID_PARAM;
            }
            error = nncp_lstm_prediction_temperature_sampling(logits, n_symbols, config->temperature, 
                                                             random_value, selected_symbol);
            if (info) {
                info->strategy_used = NNCP_SAMPLING_TEMPERATURE;
                info->temperature_used = config->temperature;
                info->fallback_used = false;
            }
            break;
            
        case NNCP_SAMPLING_TOP_K:
            error = nncp_lstm_prediction_top_k_sampling(probabilities, n_symbols, config->top_k, 
                                                       random_value, selected_symbol);
            if (info) {
                info->strategy_used = NNCP_SAMPLING_TOP_K;
                info->temperature_used = 1.0f;
                info->candidates_after_top_k = (uint32_t)config->top_k;
                info->fallback_used = false;
            }
            break;
            
        case NNCP_SAMPLING_TOP_P:
            error = nncp_lstm_prediction_top_p_sampling(probabilities, n_symbols, config->top_p, 
                                                       random_value, selected_symbol);
            if (info) {
                info->strategy_used = NNCP_SAMPLING_TOP_P;
                info->temperature_used = 1.0f;
                info->cumulative_prob_kept = config->top_p;
                info->fallback_used = false;
            }
            break;
            
        case NNCP_SAMPLING_ADAPTIVE:
            error = nncp_lstm_prediction_adaptive_sampling(probabilities, n_symbols, config->entropy_threshold_low,
                                                          random_value, selected_symbol, &info->fallback_used);
            if (info) {
                info->temperature_used = info->fallback_used ? 1.0f : 0.0f;
                info->strategy_used = NNCP_SAMPLING_COMPREHENSIVE;
            }
            break;
            
        default:
            return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Fill additional sampling info if requested
    if (info && error == NNCP_PREDICTION_SUCCESS) {
        info->selection_probability = probabilities[*selected_symbol];
        
        // Calculate entropy
        info->original_entropy = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (probabilities[i] > 1e-10f) {
                info->original_entropy -= probabilities[i] * logf(probabilities[i]);
            }
        }
        
        // Calculate effective vocabulary size
        info->effective_vocabulary_size = expf(info->original_entropy);
    }
    
    return error;
}

// Configuration and utility functions

NNCPLSTMPredictionError nncp_lstm_prediction_config_create_cuda_compatible(NNCPLSTMPredictionConfig* config) {
    if (!config) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(NNCPLSTMPredictionConfig));
    
    // CUDA-compatible defaults
    config->n_symbols = 256;           // Byte vocabulary
    config->n_embed_out = 4;           // Use 4 layers for output
    config->n_layers = 4;              // Total layers
    config->n_cells = 352;             // Hidden dimension
    config->batch_size = 1;            // Single sequence
    config->seg_len = 32;              // Segment length
    config->full_connect = true;       // Use all layers
    config->apply_softmax = true;      // Apply softmax
    config->temperature = 1.0f;        // No temperature scaling
    config->use_log_probabilities = false; // Raw probabilities
    config->verbose_logging = false;   // Minimal logging
    config->validate_outputs = true;   // Validate outputs
    
    return NNCP_PREDICTION_SUCCESS;
}

const char* nncp_lstm_prediction_get_error_string(NNCPLSTMPredictionError error_code) {
    if (error_code < 0 || error_code >= sizeof(prediction_error_messages) / sizeof(prediction_error_messages[0])) {
        return "Unknown error";
    }
    return prediction_error_messages[error_code];
}

void nncp_lstm_prediction_destroy(NNCPLSTMPredictionContext* context) {
    if (!context) {
        return;
    }
    
    // Free output layer weights
    free(context->output_layer.fc_weights);
    free(context->output_layer.fc_bias);
    
    // Free computation buffers
    free(context->layer_outputs_buffer);
    free(context->logits_buffer);
    free(context->probabilities_buffer);
    
    // Metal buffers are automatically released
    
    if (context->verbose_logging) {
        printf("NNCP LSTM Prediction Context destroyed\\n");
        printf("  Total predictions: %llu\\n", (unsigned long long)context->total_predictions);
        printf("  Average computation time: %.3f ms\\n", 
               context->total_predictions > 0 ? 
               (double)context->total_compute_time_ns / (context->total_predictions * 1000000.0) : 0.0);
        printf("  Average entropy: %.4f\\n", context->average_entropy);
    }
    
    free(context);
}

// Internal helper function implementations

static NNCPLSTMPredictionError allocate_prediction_buffers(NNCPLSTMPredictionContext* context) {
    if (!context) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint32_t max_batch_seq = context->config.batch_size * context->config.seg_len;
    uint32_t n_total_inputs = context->output_layer.n_total_inputs;
    uint32_t n_symbols = context->config.n_symbols;
    
    // Allocate CPU buffers
    context->layer_outputs_buffer = (float*)malloc(n_total_inputs * max_batch_seq * sizeof(float));
    context->logits_buffer = (float*)malloc(n_symbols * max_batch_seq * sizeof(float));
    context->probabilities_buffer = (float*)malloc(n_symbols * max_batch_seq * sizeof(float));
    
    if (!context->layer_outputs_buffer || !context->logits_buffer || !context->probabilities_buffer) {
        return NNCP_PREDICTION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize to zero
    memset(context->layer_outputs_buffer, 0, n_total_inputs * max_batch_seq * sizeof(float));
    memset(context->logits_buffer, 0, n_symbols * max_batch_seq * sizeof(float));
    memset(context->probabilities_buffer, 0, n_symbols * max_batch_seq * sizeof(float));
    
    return NNCP_PREDICTION_SUCCESS;
}

static NNCPLSTMPredictionError initialize_metal_pipelines(NNCPLSTMPredictionContext* context) {
    if (!context || !context->device) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Metal pipeline initialization would go here
    // For now, just placeholder implementation
    
    return NNCP_PREDICTION_SUCCESS;
}

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

static void validate_prediction_outputs(const float* data, size_t size, const char* output_name) {
    if (!data || !output_name) {
        return;
    }
    
    bool has_nan = false;
    bool has_inf = false;
    float min_val = data[0];
    float max_val = data[0];
    float sum = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float val = data[i];
        
        if (isnan(val)) {
            has_nan = true;
        }
        if (isinf(val)) {
            has_inf = true;
        }
        
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    
    if (has_nan || has_inf) {
        printf("WARNING: %s contains invalid values (NaN: %s, Inf: %s)\\n",
               output_name, has_nan ? "yes" : "no", has_inf ? "yes" : "no");
    }
    
    printf("Output %s: min=%.6f, max=%.6f, sum=%.6f\n", output_name, min_val, max_val, sum);
}

// ========================================
// Sampling and Selection Logic Implementation
// ========================================

// CUDA-compatible sampling function implementations

NNCPLSTMPredictionError nncp_lstm_prediction_sampling_config_create_default(NNCPLSTMSamplingConfig* config,
                                                                           NNCPLSTMSamplingStrategy strategy) {
    if (!config) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(NNCPLSTMSamplingConfig));
    
    // Set default values
    config->strategy = strategy;
    config->temperature = 1.0f;
    config->min_temperature = 0.1f;
    config->max_temperature = 2.0f;
    config->top_k = 50;
    config->min_top_k = 5;
    config->max_top_k = 100;
    config->top_p = 0.9f;
    config->min_top_p = 0.1f;
    config->max_top_p = 1.0f;
    config->confidence_threshold = 0.5f;
    config->use_relative_confidence = false;
    config->beam_width = 5;
    config->beam_alpha = 0.6f;
    config->entropy_threshold_high = 5.0f;
    config->entropy_threshold_low = 2.0f;
    config->adaptive_temperature = true;
    config->adaptive_top_k = true;
    config->adaptive_top_p = true;
    config->fallback_strategy = NNCP_SAMPLING_GREEDY;
    config->enable_fallbacks = true;
    config->max_retries = 3;
    config->safety_temperature = 1.0f;
    config->random_seed = 12345;
    config->deterministic_mode = false;
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_sampling_config_create_cuda_compatible(NNCPLSTMSamplingConfig* config) {
    if (!config) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Create configuration matching original CUDA implementation
    NNCPLSTMPredictionError error = nncp_lstm_prediction_sampling_config_create_default(config, NNCP_SAMPLING_TEMPERATURE);
    if (error != NNCP_PREDICTION_SUCCESS) {
        return error;
    }
    
    // CUDA-specific settings
    config->temperature = 1.0f;
    config->top_k = 0; // Disabled by default
    config->top_p = 0.0f; // Disabled by default
    config->enable_fallbacks = false; // CUDA implementation was deterministic
    config->deterministic_mode = true;
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_sampling_config_validate(const NNCPLSTMSamplingConfig* config) {
    if (!config) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Validate temperature parameters
    if (config->temperature <= 0.0f || config->temperature > 10.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    if (config->min_temperature <= 0.0f || config->min_temperature > config->max_temperature) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Validate top-k parameters
    if (config->top_k > 0 && (config->min_top_k == 0 || config->min_top_k > config->max_top_k)) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Validate top-p parameters
    if (config->top_p < 0.0f || config->top_p > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    if (config->min_top_p <= 0.0f || config->min_top_p > config->max_top_p || config->max_top_p > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Validate confidence threshold
    if (config->confidence_threshold < 0.0f || config->confidence_threshold > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Validate beam search parameters
    if (config->beam_width == 0 || config->beam_width > 1000) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Validate entropy thresholds
    if (config->entropy_threshold_low < 0.0f || config->entropy_threshold_high < config->entropy_threshold_low) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_greedy_selection(const float* probabilities,
                                                             uint32_t n_symbols,
                                                             uint32_t* selected_symbol,
                                                             NNCPLSTMSamplingInfo* info) {
    if (!probabilities || !selected_symbol || n_symbols == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    // Find symbol with maximum probability
    uint32_t best_symbol = 0;
    float best_prob = probabilities[0];
    
    for (uint32_t i = 1; i < n_symbols; i++) {
        if (probabilities[i] > best_prob) {
            best_prob = probabilities[i];
            best_symbol = i;
        }
    }
    
    *selected_symbol = best_symbol;
    
    // Fill sampling info if requested
    if (info) {
        memset(info, 0, sizeof(NNCPLSTMSamplingInfo));
        info->strategy_used = NNCP_SAMPLING_GREEDY;
        info->selected_symbol = best_symbol;
        info->selection_probability = best_prob;
        info->random_value_used = 0.0f; // No randomness in greedy selection
        info->total_candidates = n_symbols;
        info->candidates_after_top_k = n_symbols;
        info->candidates_after_top_p = n_symbols;
        info->cumulative_prob_kept = 1.0f;
        info->temperature_used = 0.0f; // No temperature in greedy
        info->selection_confidence = best_prob;
        info->sampling_attempts = 1;
        info->fallback_used = false;
        info->sampling_time_ns = get_timestamp_ns() - start_time;
        
        // Calculate original entropy
        info->original_entropy = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (probabilities[i] > 1e-10f) {
                info->original_entropy -= probabilities[i] * logf(probabilities[i]);
            }
        }
        info->effective_vocabulary_size = expf(info->original_entropy);
        
        // Calculate relative confidence (margin to second best)
        float second_best = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (i != best_symbol && probabilities[i] > second_best) {
                second_best = probabilities[i];
            }
        }
        info->relative_confidence = best_prob - second_best;
        info->margin_to_second_best = info->relative_confidence;
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_multinomial_sampling(const float* probabilities,
                                                                 uint32_t n_symbols,
                                                                 float random_value,
                                                                 uint32_t* selected_symbol,
                                                                 NNCPLSTMSamplingInfo* info) {
    if (!probabilities || !selected_symbol || n_symbols == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (random_value < 0.0f || random_value >= 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    // Cumulative sampling
    float cumulative = 0.0f;
    uint32_t selected = n_symbols - 1; // Fallback to last symbol
    
    for (uint32_t i = 0; i < n_symbols; i++) {
        cumulative += probabilities[i];
        if (random_value < cumulative) {
            selected = i;
            break;
        }
    }
    
    *selected_symbol = selected;
    
    // Fill sampling info if requested
    if (info) {
        memset(info, 0, sizeof(NNCPLSTMSamplingInfo));
        info->strategy_used = NNCP_SAMPLING_MULTINOMIAL;
        info->selected_symbol = selected;
        info->selection_probability = probabilities[selected];
        info->random_value_used = random_value;
        info->total_candidates = n_symbols;
        info->candidates_after_top_k = n_symbols;
        info->candidates_after_top_p = n_symbols;
        info->cumulative_prob_kept = 1.0f;
        info->temperature_used = 1.0f;
        info->selection_confidence = probabilities[selected];
        info->sampling_attempts = 1;
        info->fallback_used = false;
        info->sampling_time_ns = get_timestamp_ns() - start_time;
        
        // Calculate original entropy
        info->original_entropy = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (probabilities[i] > 1e-10f) {
                info->original_entropy -= probabilities[i] * logf(probabilities[i]);
            }
        }
        info->effective_vocabulary_size = expf(info->original_entropy);
        
        // Calculate relative confidence (margin to second best)
        float best_prob = 0.0f;
        float second_best = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (probabilities[i] > best_prob) {
                second_best = best_prob;
                best_prob = probabilities[i];
            } else if (probabilities[i] > second_best) {
                second_best = probabilities[i];
            }
        }
        info->relative_confidence = best_prob - second_best;
        info->margin_to_second_best = info->relative_confidence;
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_temperature_sampling(const float* logits,
                                                                 float* probabilities,
                                                                 uint32_t n_symbols,
                                                                 float temperature,
                                                                 float random_value,
                                                                 uint32_t* selected_symbol,
                                                                 NNCPLSTMSamplingInfo* info) {
    if (!logits || !probabilities || !selected_symbol || n_symbols == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (temperature <= 0.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (random_value < 0.0f || random_value >= 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    // Apply temperature scaling and compute softmax
    NNCPLSTMPredictionError error = nncp_lstm_prediction_apply_temperature(logits, probabilities, n_symbols, temperature);
    if (error != NNCP_PREDICTION_SUCCESS) {
        return error;
    }
    
    // Convert to probabilities with softmax
    float max_logit = probabilities[0];
    for (uint32_t i = 1; i < n_symbols; i++) {
        if (probabilities[i] > max_logit) {
            max_logit = probabilities[i];
        }
    }
    
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < n_symbols; i++) {
        probabilities[i] = expf(probabilities[i] - max_logit);
        sum_exp += probabilities[i];
    }
    
    for (uint32_t i = 0; i < n_symbols; i++) {
        probabilities[i] /= sum_exp;
    }
    
    // Sample from the temperature-scaled distribution
    error = nncp_lstm_prediction_multinomial_sampling(probabilities, n_symbols, random_value, selected_symbol, info);
    if (error != NNCP_PREDICTION_SUCCESS) {
        return error;
    }
    
    // Update sampling info to reflect temperature sampling
    if (info) {
        info->strategy_used = NNCP_SAMPLING_TEMPERATURE;
        info->temperature_used = temperature;
        info->sampling_time_ns = get_timestamp_ns() - start_time;
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_top_k_sampling(float* probabilities,
                                                           uint32_t n_symbols,
                                                           uint32_t k,
                                                           float random_value,
                                                           uint32_t* selected_symbol,
                                                           NNCPLSTMSamplingInfo* info) {
    if (!probabilities || !selected_symbol || n_symbols == 0 || k == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (random_value < 0.0f || random_value >= 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_timestamp_ns();
    uint32_t effective_k = (k > n_symbols) ? n_symbols : k;
    
    // Apply top-k filtering
    NNCPLSTMPredictionError error = nncp_lstm_prediction_apply_top_k_filtering(probabilities, n_symbols, effective_k);
    if (error != NNCP_PREDICTION_SUCCESS) {
        return error;
    }
    
    // Sample from filtered distribution
    error = nncp_lstm_prediction_multinomial_sampling(probabilities, n_symbols, random_value, selected_symbol, info);
    if (error != NNCP_PREDICTION_SUCCESS) {
        return error;
    }
    
    // Update sampling info to reflect top-k sampling
    if (info) {
        info->strategy_used = NNCP_SAMPLING_TOP_K;
        info->candidates_after_top_k = effective_k;
        info->sampling_time_ns = get_timestamp_ns() - start_time;
        
        // Calculate cumulative probability kept
        info->cumulative_prob_kept = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (probabilities[i] > 0.0f) {
                info->cumulative_prob_kept += probabilities[i];
            }
        }
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_top_p_sampling(float* probabilities,
                                                           uint32_t n_symbols,
                                                           float p_threshold,
                                                           float random_value,
                                                           uint32_t* selected_symbol,
                                                           NNCPLSTMSamplingInfo* info) {
    if (!probabilities || !selected_symbol || n_symbols == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (p_threshold <= 0.0f || p_threshold > 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    if (random_value < 0.0f || random_value >= 1.0f) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    uint64_t start_time = get_timestamp_ns();
    
    // Apply top-p filtering
    NNCPLSTMPredictionError error = nncp_lstm_prediction_apply_top_p_filtering(probabilities, n_symbols, p_threshold);
    if (error != NNCP_PREDICTION_SUCCESS) {
        return error;
    }
    
    // Sample from filtered distribution
    error = nncp_lstm_prediction_multinomial_sampling(probabilities, n_symbols, random_value, selected_symbol, info);
    if (error != NNCP_PREDICTION_SUCCESS) {
        return error;
    }
    
    // Update sampling info to reflect top-p sampling
    if (info) {
        info->strategy_used = NNCP_SAMPLING_TOP_P;
        info->sampling_time_ns = get_timestamp_ns() - start_time;
        
        // Count candidates and calculate cumulative probability kept
        uint32_t candidates = 0;
        info->cumulative_prob_kept = 0.0f;
        for (uint32_t i = 0; i < n_symbols; i++) {
            if (probabilities[i] > 0.0f) {
                candidates++;
                info->cumulative_prob_kept += probabilities[i];
            }
        }
        info->candidates_after_top_p = candidates;
    }
    
    return NNCP_PREDICTION_SUCCESS;
}

NNCPLSTMPredictionError nncp_lstm_prediction_select_symbol(const float* logits,
                                                          float* probabilities,
                                                          uint32_t n_symbols,
                                                          NNCPLSTMSamplingStrategy strategy,
                                                          const NNCPLSTMSamplingConfig* config,
                                                          float random_value,
                                                          uint32_t* selected_symbol,
                                                          NNCPLSTMSamplingInfo* info) {
    if (!probabilities || !selected_symbol || !config || n_symbols == 0) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Validate configuration
    NNCPLSTMPredictionError error = nncp_lstm_prediction_sampling_config_validate(config);
    if (error != NNCP_PREDICTION_SUCCESS) {
        return error;
    }
    
    // Route to appropriate sampling function
    switch (strategy) {
        case NNCP_SAMPLING_GREEDY:
            return nncp_lstm_prediction_greedy_selection(probabilities, n_symbols, selected_symbol, info);
            
        case NNCP_SAMPLING_MULTINOMIAL:
            return nncp_lstm_prediction_multinomial_sampling(probabilities, n_symbols, random_value, selected_symbol, info);
            
        case NNCP_SAMPLING_TEMPERATURE:
            if (!logits) {
                return NNCP_PREDICTION_ERROR_INVALID_PARAM;
            }
            return nncp_lstm_prediction_temperature_sampling(logits, probabilities, n_symbols, 
                                                           config->temperature, random_value, selected_symbol, info);
            
        case NNCP_SAMPLING_TOP_K:
            return nncp_lstm_prediction_top_k_sampling(probabilities, n_symbols, config->top_k, 
                                                      random_value, selected_symbol, info);
            
        case NNCP_SAMPLING_TOP_P:
            return nncp_lstm_prediction_top_p_sampling(probabilities, n_symbols, config->top_p, 
                                                      random_value, selected_symbol, info);
            
        case NNCP_SAMPLING_COMPREHENSIVE:
            if (!logits) {
                return NNCP_PREDICTION_ERROR_INVALID_PARAM;
            }
            return nncp_lstm_prediction_comprehensive_sampling(logits, probabilities, n_symbols, config, 
                                                              random_value, selected_symbol, info);
            
        default:
            return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
}

NNCPLSTMPredictionError nncp_lstm_prediction_get_stats(NNCPLSTMPredictionContext* context,
                                                      NNCPLSTMPredictionStats* stats) {
    if (!context || !stats) {
        return NNCP_PREDICTION_ERROR_INVALID_PARAM;
    }
    
    // Fill prediction stats from context
    // For now, provide placeholder values
    stats->total_predictions = 0;
    stats->total_prediction_time_ns = 0;
    stats->average_prediction_time_ns = 0;
    stats->memory_usage_mb = 0;
    stats->cache_hit_rate = 0.0f;
    stats->entropy_distribution_mean = 0.0f;
    stats->entropy_distribution_variance = 0.0f;
    stats->confidence_distribution_mean = 0.0f;
    stats->confidence_distribution_variance = 0.0f;
    
    return NNCP_PREDICTION_SUCCESS;
}