#include "mps_transformer.h"
#include "../ops/mps_attention.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#define MPS_AVAILABLE 1
#else
#define MPS_AVAILABLE 0
#endif

// Internal helper functions
static MPSTransformerError create_transformer_device(MPSTransformerContext* context);
static MPSTransformerError build_transformer_graphs(MPSTransformerContext* context);
static MPSTransformerError allocate_transformer_buffers(MPSTransformerContext* context);
static MPSTransformerError initialize_layer_contexts(MPSTransformerContext* context);
static MPSTransformerError execute_embedding_layer(MPSTransformerContext* context, 
                                                  const MPSTransformerTensors* tensors);
static MPSTransformerError execute_encoder_layers(MPSTransformerContext* context,
                                                 const float* input_hidden_states,
                                                 const float* attention_mask,
                                                 float* output_hidden_states);
static MPSTransformerError execute_layer_normalization(MPSTransformerContext* context,
                                                     const float* input,
                                                     float* output,
                                                     const float* gamma,
                                                     const float* beta,
                                                     const MPSTransformerLayerNorm* config);
static MPSTransformerError execute_feed_forward_network(MPSTransformerContext* context,
                                                       uint32_t layer_id,
                                                       const float* input,
                                                       float* output);
static uint64_t get_transformer_timestamp_ns(void);
static size_t calculate_transformer_tensor_size(uint32_t batch_size, uint32_t seq_len, uint32_t hidden_size);

// Error messages
static const char* transformer_error_messages[] = {
    "Success",
    "Invalid parameter", 
    "Memory allocation failed",
    "Metal device not found",
    "Compute operation failed",
    "Invalid tensor dimensions",
    "Buffer allocation failed", 
    "Graph compilation failed",
    "Execution failed",
    "Unsupported operation",
    "Layer not found",
    "Incompatible configuration"
};

// Core API Implementation

MPSTransformerError mps_transformer_create(MPSTransformerContext** context,
                                          const MPSTransformerConfig* config) {
    if (!context || !config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Validate configuration
    MPSTransformerError error = mps_transformer_config_validate(config);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return error;
    }
    
    *context = (MPSTransformerContext*)calloc(1, sizeof(MPSTransformerContext));
    if (!*context) {
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy configuration
    (*context)->config = *config;
    
    // Initialize statistics
    memset(&(*context)->stats, 0, sizeof(MPSTransformerStats));
    
    // Allocate per-layer arrays
    (*context)->layer_configs = (MPSTransformerLayerConfig*)calloc(config->num_layers, 
                                                                  sizeof(MPSTransformerLayerConfig));
    (*context)->layer_weights = (MPSTransformerLayerWeights*)calloc(config->num_layers,
                                                                   sizeof(MPSTransformerLayerWeights));
    (*context)->attention_contexts = (MPSAttentionContext**)calloc(config->num_layers,
                                                                  sizeof(MPSAttentionContext*));
    
    if (!(*context)->layer_configs || !(*context)->layer_weights || !(*context)->attention_contexts) {
        mps_transformer_destroy(*context);
        *context = NULL;
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    // Create Metal device
    error = create_transformer_device(*context);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        mps_transformer_destroy(*context);
        *context = NULL;
        return error;
    }
    
    // Initialize layer contexts
    error = initialize_layer_contexts(*context);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        mps_transformer_destroy(*context);
        *context = NULL;
        return error;
    }
    
    // Build computation graphs
    error = build_transformer_graphs(*context);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        mps_transformer_destroy(*context);
        *context = NULL;
        return error;
    }
    
    // Allocate buffers
    error = allocate_transformer_buffers(*context);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        mps_transformer_destroy(*context);
        *context = NULL;
        return error;
    }
    
    // Allocate per-layer statistics
    (*context)->stats.layer_compute_times_ms = (float*)calloc(config->num_layers, sizeof(float));
    (*context)->stats.layer_memory_usage_mb = (uint32_t*)calloc(config->num_layers, sizeof(uint32_t));
    
    (*context)->is_initialized = true;
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_load_weights(MPSTransformerContext* context,
                                                const MPSTransformerWeights* weights) {
    if (!context || !weights) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Copy weights structure
    context->weights = *weights;
    
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        
        // Create and upload embedding weights
        if (weights->word_embeddings) {
            size_t embedding_size = context->config.vocab_size * context->config.hidden_size * sizeof(float);
            id<MTLBuffer> embeddingBuffer = [device newBufferWithBytes:weights->word_embeddings
                                                                length:embedding_size
                                                               options:MTLResourceStorageModeShared];
            if (!embeddingBuffer) {
                return MPS_TRANSFORMER_ERROR_BUFFER_ALLOCATION;
            }
            context->buffers.input_buffer = (__bridge void*)embeddingBuffer;
        }
        
        // Load per-layer weights into attention contexts
        for (uint32_t i = 0; i < context->config.num_layers; i++) {
            if (context->attention_contexts[i] && weights->layer_weights && weights->layer_weights[i]) {
                MPSTransformerLayerWeights* layer_weights = (MPSTransformerLayerWeights*)weights->layer_weights[i];
                MPSAttentionError attn_error = mps_attention_load_weights(context->attention_contexts[i],
                                                                        &layer_weights->attention_weights);
                if (attn_error != MPS_ATTENTION_SUCCESS) {
                    return MPS_TRANSFORMER_ERROR_COMPUTE_FAILED;
                }
            }
        }
    }
#endif
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_forward(MPSTransformerContext* context,
                                           MPSTransformerTensors* tensors) {
    if (!context || !tensors) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    if (!context->is_initialized) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Validate tensor dimensions
    MPSTransformerError error = mps_transformer_tensors_validate(tensors, &context->config);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return error;
    }
    
    uint64_t start_time = get_transformer_timestamp_ns();
    
    // Step 1: Execute embedding layer
    error = execute_embedding_layer(context, tensors);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return error;
    }
    
    // Step 2: Execute encoder layers
    if (context->config.is_encoder_only || context->config.is_encoder_decoder) {
        error = execute_encoder_layers(context, tensors->embeddings,
                                     tensors->attention_mask, tensors->hidden_states);
        if (error != MPS_TRANSFORMER_SUCCESS) {
            return error;
        }
    }
    
    // Step 3: Execute decoder layers (if applicable)
    if (context->config.is_decoder_only || context->config.is_encoder_decoder) {
        // For decoder, use hidden_states as both input and output
        error = execute_encoder_layers(context, tensors->hidden_states,
                                     tensors->attention_mask, tensors->hidden_states);
        if (error != MPS_TRANSFORMER_SUCCESS) {
            return error;
        }
    }
    
    // Step 4: Generate output logits (if requested)
    if (tensors->logits) {
#if MPS_AVAILABLE
        @autoreleasepool {
            id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
            id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)context->command_queue;
            
            // Simple matrix multiplication for output projection
            // In a full implementation, this would use MPS matrix multiplication
            size_t hidden_size = context->config.hidden_size;
            size_t vocab_size = context->config.vocab_size;
            size_t batch_seq_size = tensors->batch_size * tensors->sequence_length;
            
            // For now, implement a simple copy operation as placeholder
            memset(tensors->logits, 0, batch_seq_size * vocab_size * sizeof(float));
            
            // Copy first hidden_size values to logits (truncated)
            size_t copy_size = (vocab_size < hidden_size) ? vocab_size : hidden_size;
            for (size_t i = 0; i < batch_seq_size; i++) {
                memcpy(&tensors->logits[i * vocab_size], 
                       &tensors->hidden_states[i * hidden_size],
                       copy_size * sizeof(float));
            }
        }
#endif
    }
    
    // Update statistics
    uint64_t compute_time = get_transformer_timestamp_ns() - start_time;
    context->stats.total_forward_passes++;
    context->stats.total_compute_time_ns += compute_time;
    context->stats.total_tokens_processed += tensors->batch_size * tensors->sequence_length;
    context->stats.batch_size_sum += tensors->batch_size;
    context->stats.sequence_length_sum += tensors->sequence_length;
    
    // Calculate derived statistics
    context->stats.average_forward_time_ms = 
        (float)context->stats.total_compute_time_ns / (1000000.0f * context->stats.total_forward_passes);
    
    if (compute_time > 0) {
        context->stats.tokens_per_second = 
            (float)(tensors->batch_size * tensors->sequence_length) / (compute_time / 1000000000.0f);
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_encode(MPSTransformerContext* context,
                                          const uint32_t* input_ids,
                                          const float* attention_mask,
                                          float* output_hidden_states) {
    if (!context || !input_ids || !output_hidden_states) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Create tensors structure for encoding
    MPSTransformerTensors* tensors = NULL;
    MPSTransformerError error = mps_transformer_tensors_create(&tensors, &context->config,
                                                              context->config.batch_size,
                                                              context->config.sequence_length);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return error;
    }
    
    // Set input data
    tensors->input_ids = (uint32_t*)input_ids;
    tensors->attention_mask = (float*)attention_mask;
    tensors->hidden_states = output_hidden_states;
    tensors->owns_memory = false;
    
    // Perform forward pass
    error = mps_transformer_forward(context, tensors);
    
    // Cleanup
    free(tensors);
    
    return error;
}

MPSTransformerError mps_transformer_generate_logits(MPSTransformerContext* context,
                                                   const uint32_t* input_ids,
                                                   const float* encoder_hidden_states,
                                                   float* logits) {
    if (!context || !input_ids || !logits) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Create tensors structure for generation
    MPSTransformerTensors* tensors = NULL;
    MPSTransformerError error = mps_transformer_tensors_create(&tensors, &context->config,
                                                              context->config.batch_size,
                                                              context->config.sequence_length);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        return error;
    }
    
    // Set input data
    tensors->input_ids = (uint32_t*)input_ids;
    tensors->logits = logits;
    tensors->owns_memory = false;
    
    // For encoder-decoder models
    if (encoder_hidden_states && context->config.is_encoder_decoder) {
        size_t encoder_hidden_size = context->config.batch_size * 
                                   context->config.sequence_length * 
                                   context->config.hidden_size;
        tensors->hidden_states = (float*)malloc(encoder_hidden_size * sizeof(float));
        if (tensors->hidden_states) {
            memcpy(tensors->hidden_states, encoder_hidden_states, encoder_hidden_size * sizeof(float));
            tensors->owns_memory = true;
        }
    }
    
    // Perform forward pass
    error = mps_transformer_forward(context, tensors);
    
    // Cleanup
    mps_transformer_tensors_destroy(tensors);
    
    return error;
}

// Configuration Functions

MPSTransformerError mps_transformer_config_create_default(MPSTransformerConfig* config,
                                                         const char* model_type,
                                                         uint32_t vocab_size,
                                                         uint32_t sequence_length,
                                                         uint32_t hidden_size,
                                                         uint32_t num_layers,
                                                         uint32_t num_heads) {
    if (!config || !model_type || vocab_size == 0 || sequence_length == 0 || 
        hidden_size == 0 || num_layers == 0 || num_heads == 0) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    if (hidden_size % num_heads != 0) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    // Initialize basic configuration
    memset(config, 0, sizeof(MPSTransformerConfig));
    
    config->vocab_size = vocab_size;
    config->sequence_length = sequence_length;
    config->hidden_size = hidden_size;
    config->num_layers = num_layers;
    config->num_heads = num_heads;
    config->batch_size = 1;
    config->max_position_embeddings = sequence_length * 2;
    
    // Set model-specific configurations
    if (strcmp(model_type, "bert") == 0) {
        config->is_encoder_only = true;
        config->is_decoder_only = false;
        config->is_encoder_decoder = false;
        config->pos_encoding_type = MPS_TRANSFORMER_POSITIONAL_LEARNED;
        config->use_token_type_embeddings = true;
        config->num_token_types = 2;
        config->tie_word_embeddings = false;
    } else if (strcmp(model_type, "gpt") == 0) {
        config->is_encoder_only = false;
        config->is_decoder_only = true;
        config->is_encoder_decoder = false;
        config->pos_encoding_type = MPS_TRANSFORMER_POSITIONAL_LEARNED;
        config->use_token_type_embeddings = false;
        config->num_token_types = 0;
        config->tie_word_embeddings = true;
    } else if (strcmp(model_type, "t5") == 0) {
        config->is_encoder_only = false;
        config->is_decoder_only = false;
        config->is_encoder_decoder = true;
        config->pos_encoding_type = MPS_TRANSFORMER_POSITIONAL_SINUSOIDAL;
        config->use_token_type_embeddings = false;
        config->num_token_types = 0;
        config->tie_word_embeddings = true;
    } else {
        // Default configuration (BERT-like)
        config->is_encoder_only = true;
        config->is_decoder_only = false;
        config->is_encoder_decoder = false;
        config->pos_encoding_type = MPS_TRANSFORMER_POSITIONAL_LEARNED;
        config->use_token_type_embeddings = false;
        config->num_token_types = 0;
        config->tie_word_embeddings = false;
    }
    
    // Set default layer normalization
    config->embedding_norm.enabled = true;
    config->embedding_norm.epsilon = 1e-12f;
    config->embedding_norm.pre_norm = false;
    config->embedding_norm.learnable_gamma = true;
    config->embedding_norm.learnable_beta = true;
    
    // Set default dropout
    config->embedding_dropout_rate = 0.1f;
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_config_validate(const MPSTransformerConfig* config) {
    if (!config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    if (config->vocab_size == 0 || config->vocab_size > 1000000) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->sequence_length == 0 || config->sequence_length > 32768) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->hidden_size == 0 || config->hidden_size > 8192) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->num_layers == 0 || config->num_layers > 48) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->num_heads == 0 || config->num_heads > 64) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->hidden_size % config->num_heads != 0) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->batch_size == 0 || config->batch_size > 256) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    if (config->embedding_dropout_rate < 0.0f || config->embedding_dropout_rate >= 1.0f) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Validate model type consistency
    uint32_t model_type_count = 0;
    if (config->is_encoder_only) model_type_count++;
    if (config->is_decoder_only) model_type_count++;
    if (config->is_encoder_decoder) model_type_count++;
    
    if (model_type_count != 1) {
        return MPS_TRANSFORMER_ERROR_INCOMPATIBLE_CONFIG;
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_calculate_memory_requirements(const MPSTransformerConfig* config,
                                                                 uint32_t* memory_mb) {
    if (!config || !memory_mb) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    uint64_t total_memory = 0;
    
    // Embedding weights memory
    uint64_t embedding_memory = (uint64_t)config->vocab_size * config->hidden_size * sizeof(float);
    if (config->pos_encoding_type == MPS_TRANSFORMER_POSITIONAL_LEARNED) {
        embedding_memory += (uint64_t)config->max_position_embeddings * config->hidden_size * sizeof(float);
    }
    if (config->use_token_type_embeddings) {
        embedding_memory += (uint64_t)config->num_token_types * config->hidden_size * sizeof(float);
    }
    total_memory += embedding_memory;
    
    // Per-layer memory (attention + FFN weights)
    uint64_t layer_memory = 0;
    
    // Attention weights (4 projection matrices)
    layer_memory += 4 * (uint64_t)config->hidden_size * config->hidden_size * sizeof(float);
    
    // FFN weights (typically 4x expansion)
    uint64_t ffn_intermediate_size = config->hidden_size * 4;
    layer_memory += (uint64_t)config->hidden_size * ffn_intermediate_size * sizeof(float); // Input projection
    layer_memory += (uint64_t)ffn_intermediate_size * config->hidden_size * sizeof(float); // Output projection
    
    // Layer normalization weights (multiple per layer)
    layer_memory += 8 * (uint64_t)config->hidden_size * sizeof(float); // 4 layer norms * (gamma + beta)
    
    total_memory += layer_memory * config->num_layers;
    
    // Activation memory (forward pass)
    uint64_t activation_memory = 0;
    
    // Hidden states buffers
    activation_memory += (uint64_t)config->batch_size * config->sequence_length * config->hidden_size * sizeof(float);
    
    // Attention intermediate tensors
    activation_memory += (uint64_t)config->batch_size * config->num_heads * 
                        config->sequence_length * config->sequence_length * sizeof(float);
    
    // FFN intermediate activations
    activation_memory += (uint64_t)config->batch_size * config->sequence_length * 
                        config->hidden_size * 4 * sizeof(float);
    
    total_memory += activation_memory * config->num_layers;
    
    // Output projection weights (if not tied)
    if (!config->tie_word_embeddings) {
        total_memory += (uint64_t)config->hidden_size * config->vocab_size * sizeof(float);
    }
    
    // Add overhead for Metal buffers and temporary storage (estimated 50% overhead)
    total_memory = total_memory * 3 / 2;
    
    *memory_mb = (uint32_t)((total_memory + 1024 * 1024 - 1) / (1024 * 1024));
    
    return MPS_TRANSFORMER_SUCCESS;
}

// Weight Management Functions

MPSTransformerError mps_transformer_weights_create(MPSTransformerWeights** weights,
                                                  const MPSTransformerConfig* config) {
    if (!weights || !config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    *weights = (MPSTransformerWeights*)calloc(1, sizeof(MPSTransformerWeights));
    if (!*weights) {
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    // Calculate sizes
    (*weights)->embedding_size = config->vocab_size * config->hidden_size * sizeof(float);
    
    // Allocate embedding weights
    (*weights)->word_embeddings = malloc((*weights)->embedding_size);
    if (!(*weights)->word_embeddings) {
        mps_transformer_weights_destroy(*weights);
        *weights = NULL;
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate position embeddings if needed
    if (config->pos_encoding_type == MPS_TRANSFORMER_POSITIONAL_LEARNED) {
        size_t pos_embedding_size = config->max_position_embeddings * config->hidden_size * sizeof(float);
        (*weights)->position_embeddings = malloc(pos_embedding_size);
        if (!(*weights)->position_embeddings) {
            mps_transformer_weights_destroy(*weights);
            *weights = NULL;
            return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    // Allocate token type embeddings if needed
    if (config->use_token_type_embeddings) {
        size_t token_type_size = config->num_token_types * config->hidden_size * sizeof(float);
        (*weights)->token_type_embeddings = malloc(token_type_size);
        if (!(*weights)->token_type_embeddings) {
            mps_transformer_weights_destroy(*weights);
            *weights = NULL;
            return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    // Allocate layer weights array
    (*weights)->layer_weights = (void**)calloc(config->num_layers, sizeof(void*));
    if (!(*weights)->layer_weights) {
        mps_transformer_weights_destroy(*weights);
        *weights = NULL;
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    // Calculate total weights size
    (*weights)->layer_weights_size = config->num_layers * sizeof(MPSTransformerLayerWeights);
    (*weights)->total_weights_size = (*weights)->embedding_size + (*weights)->layer_weights_size;
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_weights_init_random(MPSTransformerWeights* weights,
                                                       const MPSTransformerConfig* config,
                                                       uint32_t seed) {
    if (!weights || !config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    srand(seed);
    
    // Initialize word embeddings with normal distribution
    if (weights->word_embeddings) {
        float* embeddings = (float*)weights->word_embeddings;
        float scale = sqrtf(1.0f / config->hidden_size);
        
        for (size_t i = 0; i < config->vocab_size * config->hidden_size; i++) {
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float normal = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            embeddings[i] = normal * scale;
        }
    }
    
    // Initialize position embeddings
    if (weights->position_embeddings) {
        float* pos_embeddings = (float*)weights->position_embeddings;
        float scale = sqrtf(1.0f / config->hidden_size);
        
        for (size_t i = 0; i < config->max_position_embeddings * config->hidden_size; i++) {
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            float normal = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            pos_embeddings[i] = normal * scale;
        }
    }
    
    // Initialize token type embeddings to zero
    if (weights->token_type_embeddings) {
        memset(weights->token_type_embeddings, 0, 
               config->num_token_types * config->hidden_size * sizeof(float));
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

void mps_transformer_weights_destroy(MPSTransformerWeights* weights) {
    if (!weights) {
        return;
    }
    
    free(weights->word_embeddings);
    free(weights->position_embeddings);
    free(weights->token_type_embeddings);
    free(weights->embedding_norm_gamma);
    free(weights->embedding_norm_beta);
    free(weights->output_projection_weights);
    free(weights->output_projection_bias);
    
    if (weights->layer_weights) {
        free(weights->layer_weights);
    }
    
    free(weights);
}

// Tensor Management Functions

MPSTransformerError mps_transformer_tensors_create(MPSTransformerTensors** tensors,
                                                  const MPSTransformerConfig* config,
                                                  uint32_t batch_size,
                                                  uint32_t sequence_length) {
    if (!tensors || !config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    *tensors = (MPSTransformerTensors*)calloc(1, sizeof(MPSTransformerTensors));
    if (!*tensors) {
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    (*tensors)->batch_size = batch_size;
    (*tensors)->sequence_length = sequence_length;
    (*tensors)->owns_memory = false; // By default, we don't allocate memory
    
    return MPS_TRANSFORMER_SUCCESS;
}

void mps_transformer_tensors_destroy(MPSTransformerTensors* tensors) {
    if (!tensors) {
        return;
    }
    
    if (tensors->owns_memory) {
        free(tensors->input_ids);
        free(tensors->position_ids);
        free(tensors->token_type_ids);
        free(tensors->attention_mask);
        free(tensors->logits);
        free(tensors->hidden_states);
        free(tensors->embeddings);
        free(tensors->encoder_input_ids);
        free(tensors->encoder_attention_mask);
        free(tensors->cross_attention_mask);
    }
    
    free(tensors);
}

MPSTransformerError mps_transformer_tensors_validate(const MPSTransformerTensors* tensors,
                                                    const MPSTransformerConfig* config) {
    if (!tensors || !config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    if (!tensors->input_ids) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    if (tensors->batch_size == 0 || tensors->batch_size > config->batch_size) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    if (tensors->sequence_length == 0 || tensors->sequence_length > config->sequence_length) {
        return MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS;
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

// Statistics Functions

MPSTransformerError mps_transformer_get_stats(MPSTransformerContext* context,
                                             MPSTransformerStats* stats) {
    if (!context || !stats) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    *stats = context->stats;
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_reset_stats(MPSTransformerContext* context) {
    if (!context) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    memset(&context->stats, 0, sizeof(MPSTransformerStats));
    
    // Reallocate per-layer statistics
    if (context->stats.layer_compute_times_ms) {
        memset(context->stats.layer_compute_times_ms, 0, context->config.num_layers * sizeof(float));
    }
    if (context->stats.layer_memory_usage_mb) {
        memset(context->stats.layer_memory_usage_mb, 0, context->config.num_layers * sizeof(uint32_t));
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_get_memory_usage(MPSTransformerContext* context,
                                                    uint32_t* memory_usage_mb) {
    if (!context || !memory_usage_mb) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Calculate current memory usage from buffers
    uint64_t total_memory = 0;
    for (int i = 0; i < 16; i++) {
        total_memory += context->buffers.buffer_sizes[i];
    }
    
    *memory_usage_mb = (uint32_t)((total_memory + 1024 * 1024 - 1) / (1024 * 1024));
    context->stats.memory_usage_mb = *memory_usage_mb;
    
    if (*memory_usage_mb > context->stats.peak_memory_usage_mb) {
        context->stats.peak_memory_usage_mb = *memory_usage_mb;
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

// Utility Functions

const char* mps_transformer_get_error_string(MPSTransformerError error_code) {
    if (error_code < 0 || error_code >= sizeof(transformer_error_messages) / sizeof(transformer_error_messages[0])) {
        return "Unknown error";
    }
    return transformer_error_messages[error_code];
}

bool mps_transformer_is_available(void) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device && [device supportsFamily:MTLGPUFamilyApple4]) {
            return true;
        }
    }
#endif
    return false;
}

MPSTransformerError mps_transformer_get_device_info(char* device_name,
                                                   size_t buffer_size,
                                                   uint32_t* compute_units,
                                                   uint32_t* max_memory_mb) {
    if (!device_name || !compute_units || !max_memory_mb) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return MPS_TRANSFORMER_ERROR_DEVICE_NOT_FOUND;
        }
        
        NSString* name = device.name;
        strncpy(device_name, [name UTF8String], buffer_size - 1);
        device_name[buffer_size - 1] = '\0';
        
        if (@available(macOS 10.15, *)) {
            *compute_units = (uint32_t)device.maxThreadsPerThreadgroup.width;
            *max_memory_mb = (uint32_t)(device.recommendedMaxWorkingSetSize / (1024 * 1024));
        } else {
            *compute_units = 256;
            *max_memory_mb = 1024;
        }
    }
#else
    strncpy(device_name, "MPS Not Available", buffer_size - 1);
    device_name[buffer_size - 1] = '\0';
    *compute_units = 0;
    *max_memory_mb = 0;
#endif
    
    return MPS_TRANSFORMER_SUCCESS;
}

void mps_transformer_destroy(MPSTransformerContext* context) {
    if (!context) {
        return;
    }
    
    // Destroy attention contexts
    if (context->attention_contexts) {
        for (uint32_t i = 0; i < context->config.num_layers; i++) {
            if (context->attention_contexts[i]) {
                mps_attention_destroy(context->attention_contexts[i]);
            }
        }
        free(context->attention_contexts);
    }
    
    // Free per-layer arrays
    free(context->layer_configs);
    free(context->layer_weights);
    
    // Free statistics arrays
    free(context->stats.layer_compute_times_ms);
    free(context->stats.layer_memory_usage_mb);
    
    // Clear Metal objects
#if MPS_AVAILABLE
    context->device = NULL;
    context->command_queue = NULL;
#endif
    
    free(context);
}

// Internal helper function implementations

static MPSTransformerError create_transformer_device(MPSTransformerContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return MPS_TRANSFORMER_ERROR_DEVICE_NOT_FOUND;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            return MPS_TRANSFORMER_ERROR_DEVICE_NOT_FOUND;
        }
        
        context->device = (__bridge void*)device;
        context->command_queue = (__bridge void*)commandQueue;
    }
    return MPS_TRANSFORMER_SUCCESS;
#else
    return MPS_TRANSFORMER_ERROR_DEVICE_NOT_FOUND;
#endif
}

static MPSTransformerError initialize_layer_contexts(MPSTransformerContext* context) {
    // Initialize attention contexts for each layer
    for (uint32_t i = 0; i < context->config.num_layers; i++) {
        MPSAttentionConfig attention_config;
        MPSAttentionError error = mps_attention_config_create_default(&attention_config,
                                                                     context->config.sequence_length,
                                                                     context->config.hidden_size,
                                                                     context->config.num_heads);
        if (error != MPS_ATTENTION_SUCCESS) {
            return MPS_TRANSFORMER_ERROR_COMPUTE_FAILED;
        }
        
        attention_config.batch_size = context->config.batch_size;
        
        error = mps_attention_create(&context->attention_contexts[i], &attention_config);
        if (error != MPS_ATTENTION_SUCCESS) {
            return MPS_TRANSFORMER_ERROR_COMPUTE_FAILED;
        }
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

static MPSTransformerError build_transformer_graphs(MPSTransformerContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        // Create computation graphs for different stages
        MPSGraph* embeddingGraph = [[MPSGraph alloc] init];
        MPSGraph* encoderGraph = [[MPSGraph alloc] init];
        
        if (!embeddingGraph || !encoderGraph) {
            return MPS_TRANSFORMER_ERROR_GRAPH_COMPILATION;
        }
        
        context->graphs.embedding_graph = (__bridge void*)embeddingGraph;
        context->graphs.encoder_graph = (__bridge void*)encoderGraph;
        
        // In a full implementation, this would build the complete computation graphs
        // For now, we mark them as compiled
        context->graphs.graphs_compiled = true;
        
        return MPS_TRANSFORMER_SUCCESS;
    }
#else
    return MPS_TRANSFORMER_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static MPSTransformerError allocate_transformer_buffers(MPSTransformerContext* context) {
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        
        // Calculate buffer sizes
        size_t hidden_size = context->config.hidden_size;
        size_t batch_size = context->config.batch_size;
        size_t seq_len = context->config.sequence_length;
        size_t vocab_size = context->config.vocab_size;
        
        size_t hidden_states_size = batch_size * seq_len * hidden_size * sizeof(float);
        size_t embeddings_size = batch_size * seq_len * hidden_size * sizeof(float);
        size_t logits_size = batch_size * seq_len * vocab_size * sizeof(float);
        
        // Allocate main buffers
        id<MTLBuffer> hiddenStatesBuffer = [device newBufferWithLength:hidden_states_size
                                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> embeddingsBuffer = [device newBufferWithLength:embeddings_size
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:logits_size
                                                          options:MTLResourceStorageModeShared];
        
        if (!hiddenStatesBuffer || !embeddingsBuffer || !outputBuffer) {
            return MPS_TRANSFORMER_ERROR_BUFFER_ALLOCATION;
        }
        
        context->buffers.hidden_states_buffer = (__bridge void*)hiddenStatesBuffer;
        context->buffers.input_buffer = (__bridge void*)embeddingsBuffer;
        context->buffers.output_buffer = (__bridge void*)outputBuffer;
        
        // Store buffer sizes
        context->buffers.buffer_sizes[0] = hidden_states_size;
        context->buffers.buffer_sizes[1] = embeddings_size;
        context->buffers.buffer_sizes[2] = logits_size;
        
        // Allocate temporary buffers
        for (int i = 0; i < 8; i++) {
            id<MTLBuffer> tempBuffer = [device newBufferWithLength:hidden_states_size
                                                           options:MTLResourceStorageModePrivate];
            if (!tempBuffer) {
                return MPS_TRANSFORMER_ERROR_BUFFER_ALLOCATION;
            }
            context->buffers.temp_buffers[i] = (__bridge void*)tempBuffer;
            context->buffers.buffer_sizes[8 + i] = hidden_states_size;
        }
        
        context->buffers.num_temp_buffers = 8;
    }
    return MPS_TRANSFORMER_SUCCESS;
#else
    return MPS_TRANSFORMER_ERROR_UNSUPPORTED_OPERATION;
#endif
}

static MPSTransformerError execute_embedding_layer(MPSTransformerContext* context,
                                                  const MPSTransformerTensors* tensors) {
    if (!tensors->embeddings) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Simple embedding lookup implementation
    // In a full implementation, this would use Metal compute shaders for efficiency
    size_t hidden_size = context->config.hidden_size;
    size_t batch_size = tensors->batch_size;
    size_t seq_len = tensors->sequence_length;
    
    if (context->weights.word_embeddings) {
        float* word_embeddings = (float*)context->weights.word_embeddings;
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                uint32_t token_id = tensors->input_ids[b * seq_len + s];
                if (token_id < context->config.vocab_size) {
                    size_t output_offset = (b * seq_len + s) * hidden_size;
                    size_t input_offset = token_id * hidden_size;
                    memcpy(&tensors->embeddings[output_offset],
                           &word_embeddings[input_offset],
                           hidden_size * sizeof(float));
                }
            }
        }
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

static MPSTransformerError execute_encoder_layers(MPSTransformerContext* context,
                                                 const float* input_hidden_states,
                                                 const float* attention_mask,
                                                 float* output_hidden_states) {
    if (!input_hidden_states || !output_hidden_states) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    size_t hidden_size = context->config.hidden_size;
    size_t batch_size = context->config.batch_size;
    size_t seq_len = context->config.sequence_length;
    size_t hidden_states_size = batch_size * seq_len * hidden_size * sizeof(float);
    
    // Allocate temporary buffer for layer processing
    float* temp_hidden_states = (float*)malloc(hidden_states_size);
    if (!temp_hidden_states) {
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy input to working buffer
    memcpy(temp_hidden_states, input_hidden_states, hidden_states_size);
    
    // Process each transformer layer
    for (uint32_t layer_id = 0; layer_id < context->config.num_layers; layer_id++) {
        uint64_t layer_start_time = get_transformer_timestamp_ns();
        
        // Create tensors for attention layer
        MPSAttentionTensors* attention_tensors = NULL;
        MPSAttentionShape shape = {
            .batch_size = (uint32_t)batch_size,
            .sequence_length = (uint32_t)seq_len,
            .hidden_size = (uint32_t)hidden_size,
            .num_heads = context->config.num_heads,
            .head_dim = (uint32_t)(hidden_size / context->config.num_heads)
        };
        
        MPSAttentionError attn_error = mps_attention_tensors_create(&attention_tensors, &shape);
        if (attn_error != MPS_ATTENTION_SUCCESS) {
            free(temp_hidden_states);
            return MPS_TRANSFORMER_ERROR_COMPUTE_FAILED;
        }
        
        // Set up attention tensors (self-attention)
        attention_tensors->query = temp_hidden_states;
        attention_tensors->key = temp_hidden_states;
        attention_tensors->value = temp_hidden_states;
        attention_tensors->output = output_hidden_states;
        attention_tensors->attention_mask = (float*)attention_mask;
        attention_tensors->owns_memory = false;
        
        // Execute attention layer
        if (context->attention_contexts[layer_id]) {
            attn_error = mps_attention_forward(context->attention_contexts[layer_id], attention_tensors);
            if (attn_error != MPS_ATTENTION_SUCCESS) {
                mps_attention_tensors_destroy(attention_tensors);
                free(temp_hidden_states);
                return MPS_TRANSFORMER_ERROR_COMPUTE_FAILED;
            }
        }
        
        // Add residual connection
        for (size_t i = 0; i < batch_size * seq_len * hidden_size; i++) {
            output_hidden_states[i] += temp_hidden_states[i];
        }
        
        // Execute feed-forward network
        MPSTransformerError ffn_error = execute_feed_forward_network(context, layer_id,
                                                                   output_hidden_states,
                                                                   temp_hidden_states);
        if (ffn_error != MPS_TRANSFORMER_SUCCESS) {
            mps_attention_tensors_destroy(attention_tensors);
            free(temp_hidden_states);
            return ffn_error;
        }
        
        // Add residual connection for FFN
        for (size_t i = 0; i < batch_size * seq_len * hidden_size; i++) {
            temp_hidden_states[i] += output_hidden_states[i];
        }
        
        // Copy result back to output for next layer
        memcpy(output_hidden_states, temp_hidden_states, hidden_states_size);
        
        // Update layer statistics
        uint64_t layer_time = get_transformer_timestamp_ns() - layer_start_time;
        if (context->stats.layer_compute_times_ms) {
            context->stats.layer_compute_times_ms[layer_id] = (float)layer_time / 1000000.0f;
        }
        
        mps_attention_tensors_destroy(attention_tensors);
    }
    
    free(temp_hidden_states);
    return MPS_TRANSFORMER_SUCCESS;
}

static MPSTransformerError execute_feed_forward_network(MPSTransformerContext* context,
                                                       uint32_t layer_id,
                                                       const float* input,
                                                       float* output) {
    if (!input || !output) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Simple FFN implementation (placeholder)
    // In a full implementation, this would use Metal compute shaders
    size_t hidden_size = context->config.hidden_size;
    size_t batch_size = context->config.batch_size;
    size_t seq_len = context->config.sequence_length;
    
    // For now, implement a simple identity transformation
    memcpy(output, input, batch_size * seq_len * hidden_size * sizeof(float));
    
    return MPS_TRANSFORMER_SUCCESS;
}

static MPSTransformerError execute_layer_normalization(MPSTransformerContext* context,
                                                     const float* input,
                                                     float* output,
                                                     const float* gamma,
                                                     const float* beta,
                                                     const MPSTransformerLayerNorm* config) {
    if (!input || !output || !config || !config->enabled) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Simple layer normalization implementation
    size_t hidden_size = context->config.hidden_size;
    size_t batch_size = context->config.batch_size;
    size_t seq_len = context->config.sequence_length;
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            size_t offset = (b * seq_len + s) * hidden_size;
            const float* input_vector = &input[offset];
            float* output_vector = &output[offset];
            
            // Calculate mean
            float mean = 0.0f;
            for (size_t h = 0; h < hidden_size; h++) {
                mean += input_vector[h];
            }
            mean /= hidden_size;
            
            // Calculate variance
            float variance = 0.0f;
            for (size_t h = 0; h < hidden_size; h++) {
                float diff = input_vector[h] - mean;
                variance += diff * diff;
            }
            variance /= hidden_size;
            
            // Normalize
            float inv_std = 1.0f / sqrtf(variance + config->epsilon);
            for (size_t h = 0; h < hidden_size; h++) {
                float normalized = (input_vector[h] - mean) * inv_std;
                
                // Apply learnable parameters
                if (gamma && config->learnable_gamma) {
                    normalized *= gamma[h];
                }
                if (beta && config->learnable_beta) {
                    normalized += beta[h];
                }
                
                output_vector[h] = normalized;
            }
        }
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

static uint64_t get_transformer_timestamp_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

static size_t calculate_transformer_tensor_size(uint32_t batch_size, uint32_t seq_len, uint32_t hidden_size) {
    return (size_t)batch_size * seq_len * hidden_size * sizeof(float);
}

// ===== Multi-Scale Attention Implementation =====

MPSTransformerError mps_transformer_create_multiscale_config(MPSMultiScaleAttentionConfig* config,
                                                            uint32_t base_hidden_size) {
    if (!config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Initialize with default multi-scale attention settings
    config->enabled = true;
    config->num_scales = 3;
    config->local_window_size = 64;
    config->medium_window_size = 256;
    config->global_window_size = 1024;
    
    // Initialize learnable scale weights
    config->scale_weights[0] = 0.4f;  // Local attention weight
    config->scale_weights[1] = 0.35f; // Medium attention weight
    config->scale_weights[2] = 0.25f; // Global attention weight
    config->scale_weights[3] = 0.0f;  // Reserved for future use
    
    config->use_scale_fusion = true;
    config->fusion_temperature = 1.0f;
    config->adaptive_scaling = false; // Disabled by default for stability
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_init_enhanced_attention(MPSEnhancedAttentionConfig* enhanced_config,
                                                          const MPSAttentionConfig* base_config,
                                                          const MPSMultiScaleAttentionConfig* multi_scale_config) {
    if (!enhanced_config || !base_config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Copy base configuration
    enhanced_config->base_config = *base_config;
    
    // Set up multi-scale configuration
    if (multi_scale_config) {
        enhanced_config->multi_scale = *multi_scale_config;
        enhanced_config->use_multi_scale = multi_scale_config->enabled;
    } else {
        // Create default multi-scale configuration
        MPSTransformerError error = mps_transformer_create_multiscale_config(
            &enhanced_config->multi_scale, base_config->hidden_size);
        if (error != MPS_TRANSFORMER_SUCCESS) {
            return error;
        }
        enhanced_config->use_multi_scale = true;
    }
    
    // Set additional enhanced attention parameters
    enhanced_config->compression_ratio = 1; // No compression by default
    enhanced_config->use_sparse_attention = false;
    
    return MPS_TRANSFORMER_SUCCESS;
}

static MPSTransformerError execute_scaled_attention(MPSTransformerContext* context,
                                                  uint32_t layer_id,
                                                  const float* query,
                                                  const float* key,
                                                  const float* value,
                                                  const float* attention_mask,
                                                  uint32_t window_size,
                                                  float* output) {
    if (!context || !query || !key || !value || !output) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)context->command_queue;
        
        // For windowed attention, we need to process attention in chunks
        uint32_t batch_size = context->config.batch_size;
        uint32_t seq_len = context->config.sequence_length;
        uint32_t hidden_size = context->config.hidden_size;
        uint32_t num_heads = context->config.num_heads;
        uint32_t head_dim = hidden_size / num_heads;
        
        // Limit attention window for local/medium scales
        uint32_t effective_window = (window_size < seq_len) ? window_size : seq_len;
        
        // Create Metal buffers for windowed computation
        size_t query_size = batch_size * seq_len * hidden_size * sizeof(float);
        size_t output_size = batch_size * seq_len * hidden_size * sizeof(float);
        
        id<MTLBuffer> queryBuffer = [device newBufferWithBytes:query
                                                       length:query_size
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> keyBuffer = [device newBufferWithBytes:key
                                                     length:query_size
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> valueBuffer = [device newBufferWithBytes:value
                                                       length:query_size
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_size
                                                         options:MTLResourceStorageModeShared];
        
        if (!queryBuffer || !keyBuffer || !valueBuffer || !outputBuffer) {
            return MPS_TRANSFORMER_ERROR_BUFFER_ALLOCATION;
        }
        
        // Create MPS Graph for attention computation
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Define input tensors
        MPSGraphTensor* queryTensor = [graph placeholderWithShape:@[@(batch_size), @(seq_len), @(hidden_size)]
                                                         dataType:MPSDataTypeFloat32
                                                             name:@"query"];
        MPSGraphTensor* keyTensor = [graph placeholderWithShape:@[@(batch_size), @(seq_len), @(hidden_size)]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"key"];
        MPSGraphTensor* valueTensor = [graph placeholderWithShape:@[@(batch_size), @(seq_len), @(hidden_size)]
                                                         dataType:MPSDataTypeFloat32
                                                             name:@"value"];
        
        // Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        MPSGraphTensor* queryReshaped = [graph reshapeTensor:queryTensor
                                                   withShape:@[@(batch_size), @(seq_len), @(num_heads), @(head_dim)]
                                                        name:@"query_reshaped"];
        MPSGraphTensor* keyReshaped = [graph reshapeTensor:keyTensor
                                                 withShape:@[@(batch_size), @(seq_len), @(num_heads), @(head_dim)]
                                                      name:@"key_reshaped"];
        MPSGraphTensor* valueReshaped = [graph reshapeTensor:valueTensor
                                                   withShape:@[@(batch_size), @(seq_len), @(num_heads), @(head_dim)]
                                                        name:@"value_reshaped"];
        
        // Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        MPSGraphTensor* queryTransposed = [graph transposeTensor:queryReshaped
                                                       dimension:1
                                                   withDimension:2
                                                            name:@"query_transposed"];
        MPSGraphTensor* keyTransposed = [graph transposeTensor:keyReshaped
                                                     dimension:1
                                                 withDimension:2
                                                          name:@"key_transposed"];
        MPSGraphTensor* valueTransposed = [graph transposeTensor:valueReshaped
                                                       dimension:1
                                                   withDimension:2
                                                            name:@"value_transposed"];
        
        // Compute attention scores: Q @ K^T
        MPSGraphTensor* keyTransposedForMatMul = [graph transposeTensor:keyTransposed
                                                              dimension:-2
                                                          withDimension:-1
                                                                   name:@"key_transposed_matmul"];
        MPSGraphTensor* attentionScores = [graph matrixMultiplicationWithPrimaryTensor:queryTransposed
                                                                       secondaryTensor:keyTransposedForMatMul
                                                                                  name:@"attention_scores"];
        
        // Scale by sqrt(head_dim)
        float scale_factor = 1.0f / sqrtf((float)head_dim);
        MPSGraphTensor* scaleTensor = [graph constantWithScalar:scale_factor dataType:MPSDataTypeFloat32];
        MPSGraphTensor* scaledScores = [graph multiplicationWithPrimaryTensor:attentionScores
                                                              secondaryTensor:scaleTensor
                                                                         name:@"scaled_scores"];
        
        // Apply attention mask if provided
        MPSGraphTensor* maskedScores = scaledScores;
        if (attention_mask) {
            MPSGraphTensor* maskTensor = [graph placeholderWithShape:@[@(batch_size), @1, @(seq_len), @(seq_len)]
                                                            dataType:MPSDataTypeFloat32
                                                                name:@"attention_mask"];
            MPSGraphTensor* largeBias = [graph constantWithScalar:-1e9f dataType:MPSDataTypeFloat32];
            MPSGraphTensor* maskBias = [graph multiplicationWithPrimaryTensor:maskTensor
                                                              secondaryTensor:largeBias
                                                                         name:@"mask_bias"];
            maskedScores = [graph additionWithPrimaryTensor:scaledScores
                                            secondaryTensor:maskBias
                                                       name:@"masked_scores"];
        }
        
        // Apply windowing for local/medium attention
        if (window_size < seq_len) {
            // Create window mask for local attention
            // This is a simplified implementation - full implementation would use sliding windows
            maskedScores = [graph softMaxWithTensor:maskedScores axis:-1 name:@"windowed_attention_weights"];
        } else {
            // Apply softmax to get attention weights
            maskedScores = [graph softMaxWithTensor:maskedScores axis:-1 name:@"attention_weights"];
        }
        
        // Apply attention weights to values
        MPSGraphTensor* attentionOutput = [graph matrixMultiplicationWithPrimaryTensor:maskedScores
                                                                       secondaryTensor:valueTransposed
                                                                                  name:@"attention_output"];
        
        // Transpose back: [batch, seq_len, num_heads, head_dim]
        MPSGraphTensor* outputTransposed = [graph transposeTensor:attentionOutput
                                                        dimension:1
                                                    withDimension:2
                                                             name:@"output_transposed"];
        
        // Reshape back to [batch, seq_len, hidden_size]
        MPSGraphTensor* finalOutput = [graph reshapeTensor:outputTransposed
                                                 withShape:@[@(batch_size), @(seq_len), @(hidden_size)]
                                                      name:@"final_output"];
        
        // Compile and execute graph
        NSDictionary* feeds = @{
            @"query": queryBuffer,
            @"key": keyBuffer,
            @"value": valueBuffer
        };
        NSArray<MPSGraphTensor*>* targetTensorArray = @[finalOutput];
        
        MPSGraphExecutable* executable = [graph compileWithDevice:device
                                                            feeds:feeds
                                                   targetTensors:targetTensorArray
                                                    targetOperations:nil
                                                   compilationDescriptor:nil];
        
        if (!executable) {
            return MPS_TRANSFORMER_ERROR_GRAPH_COMPILATION;
        }
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        // Use simpler encoding method
        NSDictionary* inputsDict = @{
            @"query": queryBuffer,
            @"key": keyBuffer,
            @"value": valueBuffer
        };
        NSDictionary* outputsDict = @{
            @"final_output": outputBuffer
        };
        
        [executable runWithMTLCommandQueue:(id<MTLCommandQueue>)commandQueue
                                    inputs:inputsDict
                                   outputs:outputsDict
                         executionDescriptor:nil];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.error) {
            return MPS_TRANSFORMER_ERROR_EXECUTION_FAILED;
        }
        
        // Copy result back
        memcpy(output, outputBuffer.contents, output_size);
        
        return MPS_TRANSFORMER_SUCCESS;
    }
#else
    // Fallback CPU implementation for windowed attention
    uint32_t batch_size = context->config.batch_size;
    uint32_t seq_len = context->config.sequence_length;
    uint32_t hidden_size = context->config.hidden_size;
    uint32_t num_heads = context->config.num_heads;
    uint32_t head_dim = hidden_size / num_heads;
    
    // Simplified CPU implementation
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t h = 0; h < num_heads; h++) {
            for (uint32_t i = 0; i < seq_len; i++) {
                for (uint32_t d = 0; d < head_dim; d++) {
                    uint32_t idx = b * seq_len * hidden_size + i * hidden_size + h * head_dim + d;
                    
                    // Simple windowed attention - sum values in window
                    float sum = 0.0f;
                    uint32_t count = 0;
                    uint32_t window_start = (i > window_size/2) ? i - window_size/2 : 0;
                    uint32_t window_end = (i + window_size/2 < seq_len) ? i + window_size/2 : seq_len;
                    
                    for (uint32_t j = window_start; j < window_end; j++) {
                        uint32_t value_idx = b * seq_len * hidden_size + j * hidden_size + h * head_dim + d;
                        sum += value[value_idx];
                        count++;
                    }
                    
                    output[idx] = count > 0 ? sum / count : 0.0f;
                }
            }
        }
    }
    
    return MPS_TRANSFORMER_SUCCESS;
#endif
}

MPSTransformerError mps_transformer_execute_multiscale_attention(MPSTransformerContext* context,
                                                               uint32_t layer_id,
                                                               const float* query,
                                                               const float* key,
                                                               const float* value,
                                                               const float* attention_mask,
                                                               float* output) {
    if (!context || !query || !key || !value || !output) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    if (layer_id >= context->config.num_layers) {
        return MPS_TRANSFORMER_ERROR_LAYER_NOT_FOUND;
    }
    
    // Get multi-scale configuration for this layer
    MPSEnhancedAttentionConfig* attention_config = &context->layer_configs[layer_id].attention_config;
    
    if (!attention_config->use_multi_scale) {
        // Fall back to standard attention
        return execute_scaled_attention(context, layer_id, query, key, value, attention_mask, 
                                      context->config.sequence_length, output);
    }
    
    MPSMultiScaleAttentionConfig* multi_scale = &attention_config->multi_scale;
    
    // Allocate temporary buffers for each scale
    size_t tensor_size = calculate_transformer_tensor_size(context->config.batch_size,
                                                         context->config.sequence_length,
                                                         context->config.hidden_size);
    
    float* local_output = (float*)malloc(tensor_size);
    float* medium_output = (float*)malloc(tensor_size);
    float* global_output = (float*)malloc(tensor_size);
    
    if (!local_output || !medium_output || !global_output) {
        free(local_output);
        free(medium_output);
        free(global_output);
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    MPSTransformerError error;
    
    // Execute local attention
    error = execute_scaled_attention(context, layer_id, query, key, value, attention_mask,
                                   multi_scale->local_window_size, local_output);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        goto cleanup;
    }
    
    // Execute medium attention
    error = execute_scaled_attention(context, layer_id, query, key, value, attention_mask,
                                   multi_scale->medium_window_size, medium_output);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        goto cleanup;
    }
    
    // Execute global attention
    error = execute_scaled_attention(context, layer_id, query, key, value, attention_mask,
                                   multi_scale->global_window_size, global_output);
    if (error != MPS_TRANSFORMER_SUCCESS) {
        goto cleanup;
    }
    
    // Fuse the multi-scale attention outputs
    error = mps_transformer_fuse_attention_scales(context, local_output, medium_output, global_output,
                                                 multi_scale->scale_weights, output);

cleanup:
    free(local_output);
    free(medium_output);
    free(global_output);
    
    return error;
}

MPSTransformerError mps_transformer_fuse_attention_scales(MPSTransformerContext* context,
                                                        const float* local_attention,
                                                        const float* medium_attention,
                                                        const float* global_attention,
                                                        const float* fusion_weights,
                                                        float* fused_output) {
    if (!context || !local_attention || !medium_attention || !global_attention || 
        !fusion_weights || !fused_output) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    size_t total_elements = context->config.batch_size * context->config.sequence_length * 
                           context->config.hidden_size;
    
    // Weighted fusion of multi-scale attention outputs
    for (size_t i = 0; i < total_elements; i++) {
        fused_output[i] = fusion_weights[0] * local_attention[i] +
                         fusion_weights[1] * medium_attention[i] +
                         fusion_weights[2] * global_attention[i];
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_optimize_multiscale_metal(MPSTransformerContext* context,
                                                            const MPSTransformerLayerConfig* layer_configs,
                                                            uint32_t num_layers) {
    if (!context || !layer_configs) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
#if MPS_AVAILABLE
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
        
        // Optimize Metal command buffer usage for multi-scale attention
        // Pre-compile computation graphs for each scale
        for (uint32_t layer = 0; layer < num_layers; layer++) {
            const MPSEnhancedAttentionConfig* config = &layer_configs[layer].attention_config;
            
            if (config->use_multi_scale) {
                // Pre-allocate buffers for multi-scale computation
                // This reduces memory allocation overhead during inference
                size_t buffer_size = calculate_transformer_tensor_size(
                    context->config.batch_size,
                    context->config.sequence_length,
                    context->config.hidden_size
                );
                
                // Create persistent buffers for each scale
                id<MTLBuffer> localBuffer = [device newBufferWithLength:buffer_size
                                                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> mediumBuffer = [device newBufferWithLength:buffer_size
                                                                options:MTLResourceStorageModeShared];
                id<MTLBuffer> globalBuffer = [device newBufferWithLength:buffer_size
                                                                options:MTLResourceStorageModeShared];
                
                if (!localBuffer || !mediumBuffer || !globalBuffer) {
                    return MPS_TRANSFORMER_ERROR_BUFFER_ALLOCATION;
                }
                
                // Store buffers in context for reuse
                // Note: In a real implementation, these would be stored in the context structure
                // context->multiscale_buffers[layer] = {...};
            }
        }
        
        return MPS_TRANSFORMER_SUCCESS;
    }
#else
    return MPS_TRANSFORMER_ERROR_UNSUPPORTED_OPERATION;
#endif
}

// ========================================
// Enhanced Memory Context Implementation
// ========================================

#include <math.h>
#include <string.h>
#include <time.h>

// Internal helper functions for hash computation
static uint64_t compute_content_hash(const float* content_vector, uint32_t length) {
    uint64_t hash = 0x811c9dc5;  // FNV-1a initial value
    const uint32_t fnv_prime = 0x01000193;
    
    for (uint32_t i = 0; i < length; i++) {
        uint32_t value = *(uint32_t*)&content_vector[i];  // Interpret float as uint32
        hash ^= (value & 0xFF);
        hash *= fnv_prime;
        hash ^= ((value >> 8) & 0xFF);
        hash *= fnv_prime;
        hash ^= ((value >> 16) & 0xFF);
        hash *= fnv_prime;
        hash ^= ((value >> 24) & 0xFF);
        hash *= fnv_prime;
    }
    
    return hash;
}

static float compute_vector_similarity(const float* vec1, const float* vec2, uint32_t length) {
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (uint32_t i = 0; i < length; i++) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    float norm_product = sqrt(norm1) * sqrt(norm2);
    return (norm_product > 0.0f) ? (dot_product / norm_product) : 0.0f;
}

static uint64_t get_current_timestamp() {
    return (uint64_t)time(NULL);
}

MPSTransformerError mps_transformer_create_enhanced_memory_config(MPSEnhancedMemoryConfig* config,
                                                                uint32_t base_memory_size) {
    if (!config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Initialize with default values
    memset(config, 0, sizeof(MPSEnhancedMemoryConfig));
    
    config->enabled = true;
    config->long_term_memory_size = base_memory_size * 4;  // 2048 for base_memory_size=512
    config->short_term_memory_size = base_memory_size;     // 512 for base_memory_size=512
    config->pattern_memory_size = base_memory_size / 2;    // 256 for base_memory_size=512
    
    // Frequency and recency weighting parameters
    config->frequency_decay_rate = 0.95f;
    config->recency_decay_rate = 0.9f;
    config->pattern_importance_threshold = 0.5f;
    
    // Memory management settings
    config->use_adaptive_allocation = true;
    config->memory_pressure_threshold = 0.8f;
    config->eviction_batch_size = 32;
    
    // Pattern-aware settings
    config->enable_pattern_clustering = true;
    config->pattern_similarity_threshold = 0.7f;
    config->max_pattern_clusters = 64;
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_create_enhanced_memory_context(MPSEnhancedMemoryContext** context,
                                                                 const MPSEnhancedMemoryConfig* config,
                                                                 uint32_t hidden_size) {
    if (!context || !config) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    *context = (MPSEnhancedMemoryContext*)calloc(1, sizeof(MPSEnhancedMemoryContext));
    if (!*context) {
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    MPSEnhancedMemoryContext* ctx = *context;
    
    // Copy configuration
    ctx->config = *config;
    
    // Allocate memory pools
    ctx->long_term_memory = (MPSMemoryEntry*)calloc(config->long_term_memory_size, sizeof(MPSMemoryEntry));
    ctx->short_term_memory = (MPSMemoryEntry*)calloc(config->short_term_memory_size, sizeof(MPSMemoryEntry));
    ctx->pattern_memory = (MPSMemoryEntry*)calloc(config->pattern_memory_size, sizeof(MPSMemoryEntry));
    
    if (!ctx->long_term_memory || !ctx->short_term_memory || !ctx->pattern_memory) {
        mps_transformer_destroy_enhanced_memory_context(ctx);
        *context = NULL;
        return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate content vectors for each memory entry
    for (uint32_t i = 0; i < config->long_term_memory_size; i++) {
        ctx->long_term_memory[i].content_vector = (float*)calloc(hidden_size, sizeof(float));
        if (!ctx->long_term_memory[i].content_vector) {
            mps_transformer_destroy_enhanced_memory_context(ctx);
            *context = NULL;
            return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    for (uint32_t i = 0; i < config->short_term_memory_size; i++) {
        ctx->short_term_memory[i].content_vector = (float*)calloc(hidden_size, sizeof(float));
        if (!ctx->short_term_memory[i].content_vector) {
            mps_transformer_destroy_enhanced_memory_context(ctx);
            *context = NULL;
            return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    for (uint32_t i = 0; i < config->pattern_memory_size; i++) {
        ctx->pattern_memory[i].content_vector = (float*)calloc(hidden_size, sizeof(float));
        if (!ctx->pattern_memory[i].content_vector) {
            mps_transformer_destroy_enhanced_memory_context(ctx);
            *context = NULL;
            return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    // Allocate pattern clustering data
    if (config->enable_pattern_clustering) {
        ctx->pattern_clusters = (uint32_t*)calloc(config->pattern_memory_size, sizeof(uint32_t));
        ctx->cluster_centroids = (float*)calloc(config->max_pattern_clusters * hidden_size, sizeof(float));
        
        if (!ctx->pattern_clusters || !ctx->cluster_centroids) {
            mps_transformer_destroy_enhanced_memory_context(ctx);
            *context = NULL;
            return MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    // Initialize usage tracking
    ctx->long_term_used = 0;
    ctx->short_term_used = 0;
    ctx->pattern_memory_used = 0;
    ctx->global_access_counter = 0;
    ctx->total_accesses = 0;
    ctx->num_active_clusters = 0;
    
    // Initialize performance metrics
    ctx->cache_hit_rate = 0.0f;
    ctx->memory_efficiency = 0.0f;
    ctx->eviction_count = 0;
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_access_memory_entry(MPSEnhancedMemoryContext* context,
                                                       const float* content_vector,
                                                       uint32_t content_length,
                                                       uint32_t memory_type,
                                                       MPSMemoryEntry** retrieved_entry) {
    if (!context || !content_vector || !retrieved_entry || memory_type > 2) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Compute hash for fast lookup
    uint64_t content_hash = compute_content_hash(content_vector, content_length);
    
    // Select memory pool based on type
    MPSMemoryEntry* memory_pool = NULL;
    uint32_t pool_size = 0;
    uint32_t* used_count = NULL;
    
    switch (memory_type) {
        case 0: // Long-term memory
            memory_pool = context->long_term_memory;
            pool_size = context->config.long_term_memory_size;
            used_count = &context->long_term_used;
            break;
        case 1: // Short-term memory
            memory_pool = context->short_term_memory;
            pool_size = context->config.short_term_memory_size;
            used_count = &context->short_term_used;
            break;
        case 2: // Pattern memory
            memory_pool = context->pattern_memory;
            pool_size = context->config.pattern_memory_size;
            used_count = &context->pattern_memory_used;
            break;
        default:
            return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Search for matching entry
    MPSMemoryEntry* best_match = NULL;
    float best_similarity = 0.0f;
    
    for (uint32_t i = 0; i < *used_count; i++) {
        MPSMemoryEntry* entry = &memory_pool[i];
        
        // Quick hash check first
        if (entry->content_hash == content_hash && entry->content_length == content_length) {
            // Verify with full vector comparison
            float similarity = compute_vector_similarity(content_vector, entry->content_vector, content_length);
            
            if (similarity > 0.95f) {
                // Exact match found
                best_match = entry;
                break;
            } else if (similarity > best_similarity && similarity > 0.7f) {
                // Close match
                best_match = entry;
                best_similarity = similarity;
            }
        }
    }
    
    // Update access tracking
    context->global_access_counter++;
    context->total_accesses++;
    
    if (best_match) {
        // Update access statistics
        best_match->access_count++;
        best_match->last_access_time = context->global_access_counter;
        
        // Update cache hit rate
        context->cache_hit_rate = (context->cache_hit_rate * 0.9f) + (1.0f * 0.1f);
        
        *retrieved_entry = best_match;
        return MPS_TRANSFORMER_SUCCESS;
    } else {
        // Cache miss
        context->cache_hit_rate = (context->cache_hit_rate * 0.9f) + (0.0f * 0.1f);
        *retrieved_entry = NULL;
        return MPS_TRANSFORMER_SUCCESS; // Not an error, just no match found
    }
}

MPSTransformerError mps_transformer_store_memory_entry(MPSEnhancedMemoryContext* context,
                                                      const float* content_vector,
                                                      uint32_t content_length,
                                                      uint32_t pattern_id,
                                                      uint32_t compression_benefit,
                                                      uint32_t memory_type) {
    if (!context || !content_vector || memory_type > 2) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Select memory pool based on type
    MPSMemoryEntry* memory_pool = NULL;
    uint32_t pool_size = 0;
    uint32_t* used_count = NULL;
    
    switch (memory_type) {
        case 0: // Long-term memory
            memory_pool = context->long_term_memory;
            pool_size = context->config.long_term_memory_size;
            used_count = &context->long_term_used;
            break;
        case 1: // Short-term memory
            memory_pool = context->short_term_memory;
            pool_size = context->config.short_term_memory_size;
            used_count = &context->short_term_used;
            break;
        case 2: // Pattern memory
            memory_pool = context->pattern_memory;
            pool_size = context->config.pattern_memory_size;
            used_count = &context->pattern_memory_used;
            break;
        default:
            return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Check if memory is full
    if (*used_count >= pool_size) {
        // Perform eviction if memory pressure is high
        float memory_pressure = (float)*used_count / pool_size;
        if (memory_pressure >= context->config.memory_pressure_threshold) {
            MPSTransformerError eviction_result = mps_transformer_evict_memory_entries(context, memory_type, 
                                                                                     context->config.eviction_batch_size);
            if (eviction_result != MPS_TRANSFORMER_SUCCESS) {
                return eviction_result;
            }
        }
        
        // If still full, cannot store
        if (*used_count >= pool_size) {
            return MPS_TRANSFORMER_ERROR_BUFFER_ALLOCATION;
        }
    }
    
    // Find next available slot
    MPSMemoryEntry* entry = &memory_pool[*used_count];
    
    // Initialize entry
    memcpy(entry->content_vector, content_vector, content_length * sizeof(float));
    entry->content_length = content_length;
    entry->content_hash = compute_content_hash(content_vector, content_length);
    
    // Initialize tracking values
    entry->access_count = 1;
    entry->last_access_time = context->global_access_counter;
    entry->frequency_score = 1.0f;
    entry->recency_score = 1.0f;
    entry->combined_score = 1.0f;
    
    // Pattern information
    entry->pattern_id = pattern_id;
    entry->pattern_confidence = (pattern_id > 0) ? 0.8f : 0.0f;
    entry->is_pattern_anchor = false;
    
    // Metadata
    entry->sequence_position = context->total_accesses;
    entry->compression_benefit = compression_benefit;
    entry->is_persistent = (compression_benefit > 1000); // High-value entries are persistent
    
    (*used_count)++;
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_update_memory_scores(MPSEnhancedMemoryContext* context) {
    if (!context) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    uint64_t current_time = context->global_access_counter;
    
    // Update scores for all memory pools
    MPSMemoryEntry* pools[] = {context->long_term_memory, context->short_term_memory, context->pattern_memory};
    uint32_t sizes[] = {context->long_term_used, context->short_term_used, context->pattern_memory_used};
    
    for (int pool_idx = 0; pool_idx < 3; pool_idx++) {
        MPSMemoryEntry* pool = pools[pool_idx];
        uint32_t size = sizes[pool_idx];
        
        for (uint32_t i = 0; i < size; i++) {
            MPSMemoryEntry* entry = &pool[i];
            
            // Update frequency score with decay
            entry->frequency_score = entry->frequency_score * context->config.frequency_decay_rate + 
                                   (float)entry->access_count * (1.0f - context->config.frequency_decay_rate);
            
            // Update recency score
            uint64_t time_since_access = current_time - entry->last_access_time;
            entry->recency_score = expf(-((float)time_since_access) * (1.0f - context->config.recency_decay_rate));
            
            // Combine scores with pattern importance
            float pattern_weight = entry->pattern_confidence * context->config.pattern_importance_threshold;
            entry->combined_score = (entry->frequency_score * 0.4f + 
                                   entry->recency_score * 0.4f + 
                                   pattern_weight * 0.2f);
        }
    }
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_evict_memory_entries(MPSEnhancedMemoryContext* context,
                                                        uint32_t memory_type,
                                                        uint32_t num_entries) {
    if (!context || memory_type > 2) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Select memory pool based on type
    MPSMemoryEntry* memory_pool = NULL;
    uint32_t* used_count = NULL;
    
    switch (memory_type) {
        case 0: // Long-term memory
            memory_pool = context->long_term_memory;
            used_count = &context->long_term_used;
            break;
        case 1: // Short-term memory
            memory_pool = context->short_term_memory;
            used_count = &context->short_term_used;
            break;
        case 2: // Pattern memory
            memory_pool = context->pattern_memory;
            used_count = &context->pattern_memory_used;
            break;
        default:
            return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    if (*used_count == 0) {
        return MPS_TRANSFORMER_SUCCESS;
    }
    
    // Update scores before eviction
    mps_transformer_update_memory_scores(context);
    
    // Sort entries by combined score (ascending - lowest scores first for eviction)
    // Simple selection sort for small arrays
    uint32_t actual_evictions = (num_entries < *used_count) ? num_entries : *used_count;
    
    for (uint32_t i = 0; i < actual_evictions; i++) {
        uint32_t min_idx = i;
        float min_score = memory_pool[i].combined_score;
        
        // Skip persistent entries
        if (memory_pool[i].is_persistent) {
            continue;
        }
        
        for (uint32_t j = i + 1; j < *used_count; j++) {
            if (!memory_pool[j].is_persistent && memory_pool[j].combined_score < min_score) {
                min_idx = j;
                min_score = memory_pool[j].combined_score;
            }
        }
        
        // Move lowest scoring entry to position i and mark for eviction
        if (min_idx != i) {
            MPSMemoryEntry temp = memory_pool[i];
            memory_pool[i] = memory_pool[min_idx];
            memory_pool[min_idx] = temp;
        }
    }
    
    // Remove evicted entries by shifting remaining entries
    if (actual_evictions > 0) {
        memmove(memory_pool, memory_pool + actual_evictions, 
                (*used_count - actual_evictions) * sizeof(MPSMemoryEntry));
        *used_count -= actual_evictions;
        context->eviction_count += actual_evictions;
    }
    
    // Update memory efficiency
    float total_memory = context->config.long_term_memory_size + 
                        context->config.short_term_memory_size + 
                        context->config.pattern_memory_size;
    float used_memory = context->long_term_used + context->short_term_used + context->pattern_memory_used;
    context->memory_efficiency = used_memory / total_memory;
    
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_cluster_pattern_memory(MPSEnhancedMemoryContext* context) {
    if (!context || !context->config.enable_pattern_clustering) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    // Simplified k-means clustering for pattern memory
    uint32_t num_patterns = context->pattern_memory_used;
    if (num_patterns < 2) {
        return MPS_TRANSFORMER_SUCCESS;
    }
    
    uint32_t k = (num_patterns < context->config.max_pattern_clusters) ? 
                 num_patterns : context->config.max_pattern_clusters;
    
    // Initialize cluster assignments randomly
    for (uint32_t i = 0; i < num_patterns; i++) {
        context->pattern_clusters[i] = i % k;
    }
    
    context->num_active_clusters = k;
    
    // Simple clustering complete - more sophisticated implementation would iterate
    return MPS_TRANSFORMER_SUCCESS;
}

MPSTransformerError mps_transformer_optimize_memory_metal(MPSEnhancedMemoryContext* context,
                                                         void* device) {
    if (!context || !device) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
#ifdef TARGET_OS_MAC
    @autoreleasepool {
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>)device;
        
        // Allocate Metal buffers for memory pools
        size_t long_term_size = context->config.long_term_memory_size * 
                               context->long_term_memory[0].content_length * sizeof(float);
        size_t short_term_size = context->config.short_term_memory_size * 
                                context->short_term_memory[0].content_length * sizeof(float);
        size_t pattern_size = context->config.pattern_memory_size * 
                             context->pattern_memory[0].content_length * sizeof(float);
        
        context->long_term_buffer = (__bridge_retained void*)[metalDevice newBufferWithLength:long_term_size
                                                                                      options:MTLResourceStorageModeShared];
        context->short_term_buffer = (__bridge_retained void*)[metalDevice newBufferWithLength:short_term_size
                                                                                       options:MTLResourceStorageModeShared];
        context->pattern_buffer = (__bridge_retained void*)[metalDevice newBufferWithLength:pattern_size
                                                                                    options:MTLResourceStorageModeShared];
        
        // Allocate scores buffer
        size_t scores_size = (context->config.long_term_memory_size + 
                             context->config.short_term_memory_size + 
                             context->config.pattern_memory_size) * sizeof(float);
        context->scores_buffer = (__bridge_retained void*)[metalDevice newBufferWithLength:scores_size
                                                                                   options:MTLResourceStorageModeShared];
        
        if (!context->long_term_buffer || !context->short_term_buffer || 
            !context->pattern_buffer || !context->scores_buffer) {
            return MPS_TRANSFORMER_ERROR_BUFFER_ALLOCATION;
        }
        
        return MPS_TRANSFORMER_SUCCESS;
    }
#else
    return MPS_TRANSFORMER_ERROR_UNSUPPORTED_OPERATION;
#endif
}

MPSTransformerError mps_transformer_get_memory_stats(MPSEnhancedMemoryContext* context,
                                                    float* cache_hit_rate,
                                                    float* memory_efficiency,
                                                    uint32_t* eviction_count) {
    if (!context || !cache_hit_rate || !memory_efficiency || !eviction_count) {
        return MPS_TRANSFORMER_ERROR_INVALID_PARAM;
    }
    
    *cache_hit_rate = context->cache_hit_rate;
    *memory_efficiency = context->memory_efficiency;
    *eviction_count = context->eviction_count;
    
    return MPS_TRANSFORMER_SUCCESS;
}

void mps_transformer_destroy_enhanced_memory_context(MPSEnhancedMemoryContext* context) {
    if (!context) {
        return;
    }
    
    // Free memory pool content vectors
    if (context->long_term_memory) {
        for (uint32_t i = 0; i < context->config.long_term_memory_size; i++) {
            free(context->long_term_memory[i].content_vector);
        }
        free(context->long_term_memory);
    }
    
    if (context->short_term_memory) {
        for (uint32_t i = 0; i < context->config.short_term_memory_size; i++) {
            free(context->short_term_memory[i].content_vector);
        }
        free(context->short_term_memory);
    }
    
    if (context->pattern_memory) {
        for (uint32_t i = 0; i < context->config.pattern_memory_size; i++) {
            free(context->pattern_memory[i].content_vector);
        }
        free(context->pattern_memory);
    }
    
    // Free clustering data
    free(context->pattern_clusters);
    free(context->cluster_centroids);
    
    // Release Metal buffers
#ifdef TARGET_OS_MAC
    if (context->long_term_buffer) {
        CFRelease(context->long_term_buffer);
    }
    if (context->short_term_buffer) {
        CFRelease(context->short_term_buffer);
    }
    if (context->pattern_buffer) {
        CFRelease(context->pattern_buffer);
    }
    if (context->scores_buffer) {
        CFRelease(context->scores_buffer);
    }
#endif
    
    // Free context itself
    free(context);
}
