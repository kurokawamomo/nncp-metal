#ifndef MPS_TRANSFORMER_H
#define MPS_TRANSFORMER_H

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

// Include MPS Attention for integration
#include "../ops/mps_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for MPS Transformer operations
typedef enum {
    MPS_TRANSFORMER_SUCCESS = 0,
    MPS_TRANSFORMER_ERROR_INVALID_PARAM,
    MPS_TRANSFORMER_ERROR_MEMORY_ALLOCATION,
    MPS_TRANSFORMER_ERROR_DEVICE_NOT_FOUND,
    MPS_TRANSFORMER_ERROR_COMPUTE_FAILED,
    MPS_TRANSFORMER_ERROR_INVALID_DIMENSIONS,
    MPS_TRANSFORMER_ERROR_BUFFER_ALLOCATION,
    MPS_TRANSFORMER_ERROR_GRAPH_COMPILATION,
    MPS_TRANSFORMER_ERROR_EXECUTION_FAILED,
    MPS_TRANSFORMER_ERROR_UNSUPPORTED_OPERATION,
    MPS_TRANSFORMER_ERROR_LAYER_NOT_FOUND,
    MPS_TRANSFORMER_ERROR_INCOMPATIBLE_CONFIG
} MPSTransformerError;

// Transformer layer types
typedef enum {
    MPS_TRANSFORMER_ENCODER_LAYER = 0,
    MPS_TRANSFORMER_DECODER_LAYER,
    MPS_TRANSFORMER_ENCODER_DECODER_LAYER
} MPSTransformerLayerType;

// Activation functions for feed-forward networks
typedef enum {
    MPS_TRANSFORMER_ACTIVATION_RELU = 0,
    MPS_TRANSFORMER_ACTIVATION_GELU,
    MPS_TRANSFORMER_ACTIVATION_SWISH,
    MPS_TRANSFORMER_ACTIVATION_SIGMOID,
    MPS_TRANSFORMER_ACTIVATION_TANH
} MPSTransformerActivation;

// Positional encoding types
typedef enum {
    MPS_TRANSFORMER_POSITIONAL_LEARNED = 0,
    MPS_TRANSFORMER_POSITIONAL_SINUSOIDAL,
    MPS_TRANSFORMER_POSITIONAL_ROTARY,
    MPS_TRANSFORMER_POSITIONAL_ALIBI
} MPSTransformerPositionalEncoding;

// Multi-scale attention configuration
typedef struct {
    bool enabled;                          // Whether multi-scale attention is enabled
    uint32_t num_scales;                   // Number of attention scales (default: 3)
    uint32_t local_window_size;            // Local attention window size (default: 64)
    uint32_t medium_window_size;           // Medium attention window size (default: 256)
    uint32_t global_window_size;           // Global attention window size (default: 1024)
    float scale_weights[4];                // Learnable weights for each scale
    bool use_scale_fusion;                 // Whether to use attention fusion
    float fusion_temperature;             // Temperature for attention fusion
    bool adaptive_scaling;                 // Whether to use adaptive scale selection
} MPSMultiScaleAttentionConfig;

// Enhanced memory context configuration
typedef struct {
    bool enabled;                          // Whether enhanced memory context is enabled
    uint32_t long_term_memory_size;        // Long-term memory capacity (default: 2048)
    uint32_t short_term_memory_size;       // Short-term memory capacity (default: 512)
    uint32_t pattern_memory_size;          // Pattern-specific memory capacity (default: 256)
    
    // Frequency and recency weighting parameters
    float frequency_decay_rate;            // Decay rate for frequency scores (default: 0.95)
    float recency_decay_rate;              // Decay rate for recency scores (default: 0.9)
    float pattern_importance_threshold;    // Threshold for pattern importance (default: 0.5)
    
    // Memory management settings
    bool use_adaptive_allocation;          // Whether to use adaptive memory allocation
    float memory_pressure_threshold;       // Memory pressure threshold (default: 0.8)
    uint32_t eviction_batch_size;          // Number of entries to evict at once (default: 32)
    
    // Pattern-aware settings
    bool enable_pattern_clustering;        // Whether to cluster similar patterns
    float pattern_similarity_threshold;    // Similarity threshold for clustering (default: 0.7)
    uint32_t max_pattern_clusters;         // Maximum number of pattern clusters (default: 64)
} MPSEnhancedMemoryConfig;

// Memory entry for enhanced context management
typedef struct {
    // Content data
    float* content_vector;                 // Content representation vector
    uint32_t content_length;               // Length of content in tokens
    uint64_t content_hash;                 // Hash of content for fast comparison
    
    // Frequency and recency tracking
    uint32_t access_count;                 // Number of times accessed
    uint64_t last_access_time;             // Last access timestamp
    float frequency_score;                 // Computed frequency score
    float recency_score;                   // Computed recency score
    float combined_score;                  // Combined frequency + recency score
    
    // Pattern information
    uint32_t pattern_id;                   // Associated pattern identifier
    float pattern_confidence;              // Confidence in pattern association
    bool is_pattern_anchor;                // Whether this is a pattern anchor point
    
    // Metadata
    uint32_t sequence_position;            // Position in original sequence
    uint32_t compression_benefit;          // Estimated compression benefit
    bool is_persistent;                    // Whether entry should persist across resets
} MPSMemoryEntry;

// Enhanced memory context structure
typedef struct {
    MPSEnhancedMemoryConfig config;        // Configuration parameters
    
    // Memory pools
    MPSMemoryEntry* long_term_memory;      // Long-term memory pool
    MPSMemoryEntry* short_term_memory;     // Short-term memory pool
    MPSMemoryEntry* pattern_memory;        // Pattern-specific memory pool
    
    // Current usage tracking
    uint32_t long_term_used;               // Current long-term memory usage
    uint32_t short_term_used;              // Current short-term memory usage
    uint32_t pattern_memory_used;          // Current pattern memory usage
    
    // Access tracking
    uint64_t global_access_counter;        // Global access counter for recency
    uint64_t total_accesses;               // Total number of memory accesses
    
    // Pattern clustering data
    uint32_t* pattern_clusters;            // Cluster assignments for patterns
    float* cluster_centroids;              // Centroids for pattern clusters
    uint32_t num_active_clusters;          // Number of currently active clusters
    
    // Performance metrics
    float cache_hit_rate;                  // Recent cache hit rate
    float memory_efficiency;               // Memory utilization efficiency
    uint32_t eviction_count;               // Number of evictions performed
    
    // Metal buffers for GPU operations
    void* long_term_buffer;                // Metal buffer for long-term memory
    void* short_term_buffer;               // Metal buffer for short-term memory
    void* pattern_buffer;                  // Metal buffer for pattern memory
    void* scores_buffer;                   // Metal buffer for computed scores
} MPSEnhancedMemoryContext;

// Enhanced attention configuration with multi-scale support
typedef struct {
    MPSAttentionConfig base_config;        // Base attention configuration
    MPSMultiScaleAttentionConfig multi_scale; // Multi-scale specific configuration
    MPSEnhancedMemoryConfig memory_config; // Enhanced memory configuration
    bool use_multi_scale;                  // Whether to use multi-scale attention
    bool use_enhanced_memory;              // Whether to use enhanced memory context
    uint32_t compression_ratio;            // Compression ratio for efficiency
    bool use_sparse_attention;             // Whether to use sparse attention patterns
} MPSEnhancedAttentionConfig;

// Layer normalization configurations
typedef struct {
    bool enabled;                          // Whether layer norm is enabled
    float epsilon;                         // Epsilon for numerical stability
    bool pre_norm;                         // Pre-norm vs post-norm
    bool learnable_gamma;                  // Whether gamma is learnable
    bool learnable_beta;                   // Whether beta is learnable
} MPSTransformerLayerNorm;

// Feed-forward network configuration
typedef struct {
    uint32_t hidden_size;                  // Input/output hidden size
    uint32_t intermediate_size;            // Intermediate layer size (usually 4x hidden)
    MPSTransformerActivation activation;   // Activation function
    float dropout_rate;                    // Dropout rate
    bool use_bias;                         // Whether to use bias in linear layers
    bool use_glu;                          // Whether to use Gated Linear Unit
} MPSTransformerFFNConfig;

// Transformer layer configuration
typedef struct {
    uint32_t layer_id;                     // Layer identifier
    MPSTransformerLayerType layer_type;    // Type of transformer layer
    MPSEnhancedAttentionConfig attention_config; // Enhanced attention mechanism configuration
    MPSTransformerFFNConfig ffn_config;    // Feed-forward network configuration
    MPSTransformerLayerNorm pre_attention_norm;  // Layer norm before attention
    MPSTransformerLayerNorm post_attention_norm; // Layer norm after attention
    MPSTransformerLayerNorm pre_ffn_norm;        // Layer norm before FFN
    MPSTransformerLayerNorm post_ffn_norm;       // Layer norm after FFN
    float residual_dropout_rate;           // Dropout rate for residual connections
    bool use_cross_attention;              // Whether to use cross-attention (decoder)
    bool use_causal_mask;                  // Whether to apply causal masking
} MPSTransformerLayerConfig;

// Complete transformer model configuration
typedef struct {
    uint32_t num_layers;                   // Number of transformer layers
    uint32_t vocab_size;                   // Vocabulary size
    uint32_t sequence_length;              // Maximum sequence length
    uint32_t hidden_size;                  // Hidden dimension size
    uint32_t num_heads;                    // Number of attention heads
    uint32_t batch_size;                   // Batch size
    uint32_t max_position_embeddings;      // Maximum position embeddings
    
    MPSTransformerPositionalEncoding pos_encoding_type; // Positional encoding type
    MPSTransformerLayerNorm embedding_norm;             // Layer norm for embeddings
    float embedding_dropout_rate;          // Dropout rate for embeddings
    
    bool tie_word_embeddings;              // Whether to tie input/output embeddings
    bool use_token_type_embeddings;        // Whether to use token type embeddings
    uint32_t num_token_types;              // Number of token types (for BERT-style)
    
    // Model-specific configurations
    bool is_encoder_only;                  // Encoder-only model (like BERT)
    bool is_decoder_only;                  // Decoder-only model (like GPT)
    bool is_encoder_decoder;               // Encoder-decoder model (like T5)
} MPSTransformerConfig;

// Transformer weights and embeddings
typedef struct {
    // Embedding weights
    void* word_embeddings;                 // Word embedding matrix [vocab_size, hidden_size]
    void* position_embeddings;             // Position embedding matrix [max_pos, hidden_size]
    void* token_type_embeddings;           // Token type embeddings [num_types, hidden_size]
    
    // Layer normalization weights (if used globally)
    void* embedding_norm_gamma;            // Embedding layer norm gamma [hidden_size]
    void* embedding_norm_beta;             // Embedding layer norm beta [hidden_size]
    
    // Output projection weights (for language modeling)
    void* output_projection_weights;       // Output projection [hidden_size, vocab_size]
    void* output_projection_bias;          // Output projection bias [vocab_size]
    
    // Per-layer weights stored separately
    void** layer_weights;                  // Array of per-layer weight pointers
    
    size_t embedding_size;                 // Size of embedding matrices in bytes
    size_t layer_weights_size;             // Size of layer weights in bytes
    size_t total_weights_size;             // Total size of all weights
} MPSTransformerWeights;

// Transformer layer weights
typedef struct {
    // Attention weights (using MPS Attention structure)
    MPSAttentionWeights attention_weights;
    MPSAttentionWeights cross_attention_weights; // For decoder layers
    
    // Feed-forward network weights
    void* ffn_input_weights;               // FFN input projection [hidden_size, intermediate_size]
    void* ffn_input_bias;                  // FFN input bias [intermediate_size]
    void* ffn_output_weights;              // FFN output projection [intermediate_size, hidden_size]
    void* ffn_output_bias;                 // FFN output bias [hidden_size]
    void* ffn_gate_weights;                // FFN gate weights (for GLU) [hidden_size, intermediate_size]
    void* ffn_gate_bias;                   // FFN gate bias [intermediate_size]
    
    // Layer normalization weights
    void* pre_attention_norm_gamma;        // Pre-attention layer norm gamma [hidden_size]
    void* pre_attention_norm_beta;         // Pre-attention layer norm beta [hidden_size]
    void* post_attention_norm_gamma;       // Post-attention layer norm gamma [hidden_size]
    void* post_attention_norm_beta;        // Post-attention layer norm beta [hidden_size]
    void* pre_ffn_norm_gamma;              // Pre-FFN layer norm gamma [hidden_size]
    void* pre_ffn_norm_beta;               // Pre-FFN layer norm beta [hidden_size]
    void* post_ffn_norm_gamma;             // Post-FFN layer norm gamma [hidden_size]
    void* post_ffn_norm_beta;              // Post-FFN layer norm beta [hidden_size]
    
    size_t weights_size;                   // Total size of layer weights
} MPSTransformerLayerWeights;

// Metal buffers for transformer computation
typedef struct {
    // Input/output buffers
    void* input_buffer;                    // Input embeddings buffer
    void* output_buffer;                   // Output logits buffer
    void* position_ids_buffer;             // Position indices buffer
    void* token_type_ids_buffer;           // Token type indices buffer
    void* attention_mask_buffer;           // Attention mask buffer
    
    // Intermediate computation buffers
    void* hidden_states_buffer;            // Hidden states buffer
    void* residual_buffer;                 // Residual connection buffer
    void* normalized_buffer;               // Layer norm output buffer
    void* ffn_intermediate_buffer;         // FFN intermediate buffer
    void* attention_output_buffer;         // Attention output buffer
    
    // Per-layer buffers
    void** layer_buffers;                  // Array of per-layer buffer arrays
    
    // Temporary computation buffers
    void* temp_buffers[8];                 // Temporary buffers for various operations
    size_t buffer_sizes[16];               // Sizes of all buffers
    uint32_t num_temp_buffers;             // Number of temporary buffers allocated
} MPSTransformerBuffers;

// Transformer computation graph
typedef struct {
    void* embedding_graph;                 // Embedding computation graph
    void* encoder_graph;                   // Encoder computation graph
    void* decoder_graph;                   // Decoder computation graph
    void* output_graph;                    // Output projection graph
    
    void* embedding_executable;            // Compiled embedding graph
    void* encoder_executable;              // Compiled encoder graph
    void* decoder_executable;              // Compiled decoder graph
    void* output_executable;               // Compiled output graph
    
    bool graphs_compiled;                  // Whether graphs are compiled
} MPSTransformerGraphs;

// Performance statistics for transformer
typedef struct {
    uint64_t total_forward_passes;         // Total forward passes
    uint64_t total_compute_time_ns;        // Total compute time
    uint64_t total_tokens_processed;       // Total tokens processed
    uint32_t batch_size_sum;               // Sum of batch sizes
    uint32_t sequence_length_sum;          // Sum of sequence lengths
    
    float average_forward_time_ms;         // Average forward pass time
    float tokens_per_second;               // Tokens processed per second
    float gflops_achieved;                 // GFLOPS performance
    
    uint32_t memory_usage_mb;              // Current memory usage
    uint32_t peak_memory_usage_mb;         // Peak memory usage
    uint32_t cache_hit_rate_percent;       // Cache hit rate percentage
    
    // Per-layer statistics
    float* layer_compute_times_ms;         // Compute time per layer
    uint32_t* layer_memory_usage_mb;       // Memory usage per layer
} MPSTransformerStats;

// Main transformer context
typedef struct {
    MPSTransformerConfig config;           // Transformer configuration
    MPSTransformerWeights weights;         // Model weights
    MPSTransformerBuffers buffers;         // Metal buffers
    MPSTransformerGraphs graphs;           // Computation graphs
    MPSTransformerStats stats;             // Performance statistics
    
    // Metal/MPS objects
    void* device;                          // Metal device
    void* command_queue;                   // Metal command queue
    
    // Layer contexts (for attention and FFN operations)
    MPSAttentionContext** attention_contexts;    // Per-layer attention contexts
    MPSTransformerLayerConfig* layer_configs;    // Per-layer configurations
    MPSTransformerLayerWeights* layer_weights;   // Per-layer weights
    
    // Enhanced memory context
    MPSEnhancedMemoryContext* memory_context;    // Enhanced memory context for optimization
    
    // Internal state
    bool is_initialized;                   // Whether context is initialized
    uint32_t current_sequence_length;      // Current processing sequence length
    uint32_t current_batch_size;           // Current processing batch size
    bool in_training_mode;                 // Whether in training mode
    void* platform_context;               // Platform-specific context
} MPSTransformerContext;

// Input/output tensors for transformer
typedef struct {
    // Input tensors
    uint32_t* input_ids;                   // Input token IDs [batch, seq_len]
    uint32_t* position_ids;                // Position IDs [batch, seq_len] (optional)
    uint32_t* token_type_ids;              // Token type IDs [batch, seq_len] (optional)
    float* attention_mask;                 // Attention mask [batch, seq_len] or [batch, seq_len, seq_len]
    
    // Output tensors
    float* logits;                         // Output logits [batch, seq_len, vocab_size]
    float* hidden_states;                  // Final hidden states [batch, seq_len, hidden_size]
    float* embeddings;                     // Input embeddings [batch, seq_len, hidden_size]
    
    // Encoder-decoder specific
    uint32_t* encoder_input_ids;           // Encoder input IDs [batch, enc_seq_len]
    float* encoder_attention_mask;         // Encoder attention mask [batch, enc_seq_len]
    float* cross_attention_mask;           // Cross-attention mask [batch, dec_seq_len, enc_seq_len]
    
    // Tensor dimensions
    uint32_t batch_size;                   // Current batch size
    uint32_t sequence_length;              // Current sequence length
    uint32_t encoder_sequence_length;      // Encoder sequence length (for enc-dec)
    bool owns_memory;                      // Whether this structure owns the memory
} MPSTransformerTensors;

// Multi-Scale Attention API Functions

/**
 * Create multi-scale attention configuration with default settings
 * @param config Multi-scale attention configuration to initialize
 * @param base_hidden_size Base hidden size for scaling
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_create_multiscale_config(MPSMultiScaleAttentionConfig* config,
                                                            uint32_t base_hidden_size);

/**
 * Initialize enhanced attention configuration with multi-scale support
 * @param enhanced_config Enhanced attention configuration to initialize
 * @param base_config Base attention configuration
 * @param multi_scale_config Multi-scale specific configuration
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_init_enhanced_attention(MPSEnhancedAttentionConfig* enhanced_config,
                                                          const MPSAttentionConfig* base_config,
                                                          const MPSMultiScaleAttentionConfig* multi_scale_config);

/**
 * Execute multi-scale attention computation
 * @param context MPS transformer context
 * @param layer_id Layer identifier for attention
 * @param query Query tensor [batch, seq_len, hidden_size]
 * @param key Key tensor [batch, seq_len, hidden_size]
 * @param value Value tensor [batch, seq_len, hidden_size]
 * @param attention_mask Attention mask [batch, seq_len, seq_len]
 * @param output Output tensor [batch, seq_len, hidden_size]
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_execute_multiscale_attention(MPSTransformerContext* context,
                                                               uint32_t layer_id,
                                                               const float* query,
                                                               const float* key,
                                                               const float* value,
                                                               const float* attention_mask,
                                                               float* output);

/**
 * Fuse multi-scale attention outputs with learnable weights
 * @param context MPS transformer context
 * @param local_attention Local attention output [batch, seq_len, hidden_size]
 * @param medium_attention Medium attention output [batch, seq_len, hidden_size]
 * @param global_attention Global attention output [batch, seq_len, hidden_size]
 * @param fusion_weights Learnable fusion weights [3]
 * @param fused_output Fused attention output [batch, seq_len, hidden_size]
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_fuse_attention_scales(MPSTransformerContext* context,
                                                        const float* local_attention,
                                                        const float* medium_attention,
                                                        const float* global_attention,
                                                        const float* fusion_weights,
                                                        float* fused_output);

/**
 * Optimize multi-scale attention for Metal performance
 * @param context MPS transformer context
 * @param layer_configs Array of layer configurations
 * @param num_layers Number of layers
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_optimize_multiscale_metal(MPSTransformerContext* context,
                                                            const MPSTransformerLayerConfig* layer_configs,
                                                            uint32_t num_layers);

// Enhanced Memory Context API Functions

/**
 * Create enhanced memory context configuration with default settings
 * @param config Enhanced memory configuration to initialize
 * @param base_memory_size Base memory size for scaling
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_create_enhanced_memory_config(MPSEnhancedMemoryConfig* config,
                                                                uint32_t base_memory_size);

/**
 * Create enhanced memory context with specified configuration
 * @param context Pointer to store created memory context
 * @param config Enhanced memory configuration
 * @param hidden_size Hidden dimension size for memory vectors
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_create_enhanced_memory_context(MPSEnhancedMemoryContext** context,
                                                                 const MPSEnhancedMemoryConfig* config,
                                                                 uint32_t hidden_size);

/**
 * Access memory entry with frequency and recency updating
 * @param context Enhanced memory context
 * @param content_vector Content representation vector
 * @param content_length Length of content in tokens
 * @param memory_type Type of memory (0=long-term, 1=short-term, 2=pattern)
 * @param retrieved_entry Output pointer for retrieved memory entry
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_access_memory_entry(MPSEnhancedMemoryContext* context,
                                                       const float* content_vector,
                                                       uint32_t content_length,
                                                       uint32_t memory_type,
                                                       MPSMemoryEntry** retrieved_entry);

/**
 * Store new memory entry with intelligent allocation
 * @param context Enhanced memory context
 * @param content_vector Content representation vector
 * @param content_length Length of content in tokens
 * @param pattern_id Associated pattern identifier
 * @param compression_benefit Estimated compression benefit
 * @param memory_type Preferred memory type (0=long-term, 1=short-term, 2=pattern)
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_store_memory_entry(MPSEnhancedMemoryContext* context,
                                                      const float* content_vector,
                                                      uint32_t content_length,
                                                      uint32_t pattern_id,
                                                      uint32_t compression_benefit,
                                                      uint32_t memory_type);

/**
 * Update frequency and recency scores for all memory entries
 * @param context Enhanced memory context
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_update_memory_scores(MPSEnhancedMemoryContext* context);

/**
 * Perform memory eviction based on combined scores
 * @param context Enhanced memory context
 * @param memory_type Type of memory to evict from (0=long-term, 1=short-term, 2=pattern)
 * @param num_entries Number of entries to evict
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_evict_memory_entries(MPSEnhancedMemoryContext* context,
                                                        uint32_t memory_type,
                                                        uint32_t num_entries);

/**
 * Cluster pattern memory entries for improved access patterns
 * @param context Enhanced memory context
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_cluster_pattern_memory(MPSEnhancedMemoryContext* context);

/**
 * Optimize memory context for Metal performance
 * @param context Enhanced memory context
 * @param device Metal device for buffer allocation
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_optimize_memory_metal(MPSEnhancedMemoryContext* context,
                                                         void* device);

/**
 * Get memory context performance statistics
 * @param context Enhanced memory context
 * @param cache_hit_rate Output pointer for cache hit rate
 * @param memory_efficiency Output pointer for memory efficiency
 * @param eviction_count Output pointer for eviction count
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_get_memory_stats(MPSEnhancedMemoryContext* context,
                                                    float* cache_hit_rate,
                                                    float* memory_efficiency,
                                                    uint32_t* eviction_count);

/**
 * Destroy enhanced memory context and free resources
 * @param context Enhanced memory context to destroy
 */
void mps_transformer_destroy_enhanced_memory_context(MPSEnhancedMemoryContext* context);

// Prediction Scoring Integration Functions

/**
 * Initialize prediction scoring integration for Transformer
 * @param context MPS transformer context
 * @param prediction_scorer Advanced prediction scorer instance
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_init_prediction_scoring(MPSTransformerContext* context,
                                                           void* prediction_scorer);

/**
 * Execute Transformer forward pass with prediction scoring integration
 * @param context MPS transformer context
 * @param input_data Input sequence data [batch_size, sequence_length]
 * @param input_length Length of input sequence
 * @param prediction_candidates Output prediction candidates array
 * @param max_candidates Maximum number of candidates to generate
 * @param num_candidates Output number of candidates generated
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_predict_with_scoring(MPSTransformerContext* context,
                                                        const uint8_t* input_data,
                                                        uint32_t input_length,
                                                        void* prediction_candidates,
                                                        uint32_t max_candidates,
                                                        uint32_t* num_candidates);

/**
 * Update Transformer context with prediction scoring feedback
 * @param context MPS transformer context
 * @param actual_byte Actual byte that was compressed
 * @param predicted_byte Predicted byte from scoring system
 * @param compression_effectiveness Effectiveness score for this prediction
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_update_prediction_feedback(MPSTransformerContext* context,
                                                             uint8_t actual_byte,
                                                             uint8_t predicted_byte,
                                                             float compression_effectiveness);

// Core API Functions

/**
 * Create and initialize MPS transformer context
 * @param context Pointer to store created context
 * @param config Transformer configuration
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_create(MPSTransformerContext** context,
                                          const MPSTransformerConfig* config);

/**
 * Load transformer weights from memory
 * @param context MPS transformer context
 * @param weights Transformer weights
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_load_weights(MPSTransformerContext* context,
                                                const MPSTransformerWeights* weights);

/**
 * Perform forward pass through transformer
 * @param context MPS transformer context
 * @param tensors Input/output tensors
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_forward(MPSTransformerContext* context,
                                           MPSTransformerTensors* tensors);

/**
 * Encode input sequence (encoder-only or encoder-decoder)
 * @param context MPS transformer context
 * @param input_ids Input token IDs [batch, seq_len]
 * @param attention_mask Attention mask [batch, seq_len] (optional)
 * @param output_hidden_states Output hidden states [batch, seq_len, hidden_size]
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_encode(MPSTransformerContext* context,
                                          const uint32_t* input_ids,
                                          const float* attention_mask,
                                          float* output_hidden_states);

/**
 * Generate next token logits (decoder-only or encoder-decoder)
 * @param context MPS transformer context
 * @param input_ids Input token IDs [batch, seq_len]
 * @param encoder_hidden_states Encoder hidden states [batch, enc_seq_len, hidden_size] (optional)
 * @param logits Output logits [batch, seq_len, vocab_size]
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_generate_logits(MPSTransformerContext* context,
                                                   const uint32_t* input_ids,
                                                   const float* encoder_hidden_states,
                                                   float* logits);

// Configuration Functions

/**
 * Create default transformer configuration
 * @param config Pointer to store default configuration
 * @param model_type Model type ("bert", "gpt", "t5", etc.)
 * @param vocab_size Vocabulary size
 * @param sequence_length Maximum sequence length
 * @param hidden_size Hidden dimension size
 * @param num_layers Number of transformer layers
 * @param num_heads Number of attention heads
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_config_create_default(MPSTransformerConfig* config,
                                                         const char* model_type,
                                                         uint32_t vocab_size,
                                                         uint32_t sequence_length,
                                                         uint32_t hidden_size,
                                                         uint32_t num_layers,
                                                         uint32_t num_heads);

/**
 * Validate transformer configuration
 * @param config Configuration to validate
 * @return MPS_TRANSFORMER_SUCCESS if valid, error code otherwise
 */
MPSTransformerError mps_transformer_config_validate(const MPSTransformerConfig* config);

/**
 * Calculate memory requirements for transformer
 * @param config Transformer configuration
 * @param memory_mb Pointer to store memory requirement in MB
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_calculate_memory_requirements(const MPSTransformerConfig* config,
                                                                 uint32_t* memory_mb);

// Layer-specific Functions

/**
 * Add transformer layer to model
 * @param context MPS transformer context
 * @param layer_config Layer configuration
 * @param layer_weights Layer weights
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_add_layer(MPSTransformerContext* context,
                                             const MPSTransformerLayerConfig* layer_config,
                                             const MPSTransformerLayerWeights* layer_weights);

/**
 * Process single transformer layer
 * @param context MPS transformer context
 * @param layer_id Layer identifier
 * @param input_hidden_states Input hidden states [batch, seq_len, hidden_size]
 * @param encoder_hidden_states Encoder hidden states [batch, enc_seq_len, hidden_size] (optional)
 * @param attention_mask Attention mask
 * @param output_hidden_states Output hidden states [batch, seq_len, hidden_size]
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_process_layer(MPSTransformerContext* context,
                                                 uint32_t layer_id,
                                                 const float* input_hidden_states,
                                                 const float* encoder_hidden_states,
                                                 const float* attention_mask,
                                                 float* output_hidden_states);

// Weight Management Functions

/**
 * Create transformer weights structure
 * @param weights Pointer to store created weights
 * @param config Transformer configuration
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_weights_create(MPSTransformerWeights** weights,
                                                  const MPSTransformerConfig* config);

/**
 * Load weights from file or buffer
 * @param weights Weights structure to load into
 * @param data_buffer Buffer containing weights data
 * @param buffer_size Size of data buffer
 * @param format Weight format ("pytorch", "tensorflow", "onnx", etc.)
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_weights_load_from_buffer(MPSTransformerWeights* weights,
                                                           const void* data_buffer,
                                                           size_t buffer_size,
                                                           const char* format);

/**
 * Initialize weights with random values
 * @param weights Weights structure to initialize
 * @param config Transformer configuration
 * @param seed Random seed
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_weights_init_random(MPSTransformerWeights* weights,
                                                       const MPSTransformerConfig* config,
                                                       uint32_t seed);

/**
 * Destroy transformer weights structure
 * @param weights Weights to destroy
 */
void mps_transformer_weights_destroy(MPSTransformerWeights* weights);

// Tensor Management Functions

/**
 * Create transformer tensors structure
 * @param tensors Pointer to store created tensors
 * @param config Transformer configuration
 * @param batch_size Batch size
 * @param sequence_length Sequence length
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_tensors_create(MPSTransformerTensors** tensors,
                                                  const MPSTransformerConfig* config,
                                                  uint32_t batch_size,
                                                  uint32_t sequence_length);

/**
 * Destroy transformer tensors structure
 * @param tensors Tensors to destroy
 */
void mps_transformer_tensors_destroy(MPSTransformerTensors* tensors);

/**
 * Validate tensor dimensions
 * @param tensors Tensors to validate
 * @param config Transformer configuration
 * @return MPS_TRANSFORMER_SUCCESS if valid, error code otherwise
 */
MPSTransformerError mps_transformer_tensors_validate(const MPSTransformerTensors* tensors,
                                                    const MPSTransformerConfig* config);

// Statistics and Monitoring Functions

/**
 * Get transformer performance statistics
 * @param context MPS transformer context
 * @param stats Pointer to store statistics
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_get_stats(MPSTransformerContext* context,
                                             MPSTransformerStats* stats);

/**
 * Reset performance statistics
 * @param context MPS transformer context
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_reset_stats(MPSTransformerContext* context);

/**
 * Get current memory usage
 * @param context MPS transformer context
 * @param memory_usage_mb Pointer to store memory usage in MB
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_get_memory_usage(MPSTransformerContext* context,
                                                    uint32_t* memory_usage_mb);

// Utility Functions

/**
 * Get error message string
 * @param error_code MPS transformer error code
 * @return Human-readable error message
 */
const char* mps_transformer_get_error_string(MPSTransformerError error_code);

/**
 * Check if MPS transformer is available on current device
 * @return true if available, false otherwise
 */
bool mps_transformer_is_available(void);

/**
 * Get transformer device information
 * @param device_name Buffer to store device name
 * @param buffer_size Size of device name buffer
 * @param compute_units Pointer to store number of compute units
 * @param max_memory_mb Pointer to store maximum memory in MB
 * @return MPS_TRANSFORMER_SUCCESS on success, error code on failure
 */
MPSTransformerError mps_transformer_get_device_info(char* device_name,
                                                   size_t buffer_size,
                                                   uint32_t* compute_units,
                                                   uint32_t* max_memory_mb);

/**
 * Destroy MPS transformer context
 * @param context Context to destroy
 */
void mps_transformer_destroy(MPSTransformerContext* context);

#ifdef __cplusplus
}
#endif

#endif // MPS_TRANSFORMER_H
