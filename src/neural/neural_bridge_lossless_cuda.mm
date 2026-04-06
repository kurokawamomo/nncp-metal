/*
 * NNCP Lossless Compression with Authentic CUDA Algorithm Port
 * Based on original NNCP implementation by Fabrice Bellard
 * 
 * This is a complete port of the original CUDA Transformer compression algorithm
 * that performs true lossless compression using neural network predictions.
 * Uses authentic arithmetic coding with write_sym/read_sym from original nncp.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mach/mach_time.h>
#include <unistd.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "neural_bridge.h"
#include "layer_flow_optimizer.h"
#include "neural_weights.h"
#include "mps_transformer_graph.h"
#include "online_trainer.h"

// Fix BOOL definition for compatibility
#ifndef BOOL
#define BOOL bool
#endif

float g_lr_override = 0.0f;  // 0 = use default 1e-4; set via --lr CLI flag
int   g_vocab_size_override = 0;  // 0 = use default 256; set to 256+n_words for preprocessing
NNCPProfileConfig g_nncp_profile = {256, 4, 512, 8, 32, 16, 32}; // default profile

// D_POS = max(MEM_LEN+SEG_LEN, MEM_LEN*2); equals 64 for default profile (16/32/32)
#define NNCP_D_POS() ((size_t)((g_nncp_profile.mem_len >= g_nncp_profile.seg_len) \
                               ? g_nncp_profile.mem_len * 2 \
                               : g_nncp_profile.mem_len + g_nncp_profile.seg_len))

#ifdef __cplusplus
extern "C" {
#endif

// Original NNCP constants (CUDA compatible)
#define NNCP_FILE_MAGIC   0xb727ac58
#define NNCP_FILE_VERSION 1

// Original NNCP arithmetic coding (exact CUDA implementation)
#define PROB_UNIT_BITS 15
#define PROB_UNIT (1 << PROB_UNIT_BITS)
#define RANGE_MIN_BITS 16
#define RANGE_MIN ((0xff << (RANGE_MIN_BITS - 8)) + 1)
#define RANGE_MAX (0xff << RANGE_MIN_BITS)

// Arithmetic encoder/decoder states (from original arith.h)
typedef struct {
    uint32_t range;
    uint32_t low;
    uint8_t current_byte;
    uint32_t n_bytes;
    uint8_t *buf;
    size_t buf_size;
    size_t idx;
    uint64_t byte_count;
} PutBitState;

typedef struct {
    uint8_t *buf;
    int buf_len;
    int buf_size;
    int idx;
    uint32_t low;
    uint32_t range;
    BOOL eof_reached;
    uint64_t byte_count;
} GetBitState;

// Original NNCP arithmetic coding functions (from arith.c)
static void put_bit_init(PutBitState *s, uint8_t *buf, int buf_size);
static void put_bit(PutBitState *s, int prob0, int bit);
static int64_t put_bit_flush(PutBitState *s);
static void get_bit_init(GetBitState *s, uint8_t *buf, size_t buf_size);
static int get_bit(GetBitState *s, int prob0);

// Original NNCP symbol encoding (from cp_utils.c)
static void write_sym(PutBitState *pb, const float *prob_table, int n_symb, int sym);
static int read_sym(GetBitState *gb, const float *prob_table, int n_symb);
static float vec_sum_f32(const float* data, int len);
static inline int clamp_int(int val, int min_val, int max_val);

// Metal Transformer integration
static bool get_metal_transformer_prediction(const int32_t* context, int context_len, 
                                           float* probabilities, int vocab_size);

// Metal Transformer Model Structure (CUDA-compatible design)
typedef struct MetalTransformerModel {
    // Metal device and command management
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    
    // Model architecture parameters (based on original CUDA specs)
    uint32_t context_length;        // 64 (original CUDA context size)
    uint32_t vocab_size;           // 256 (byte vocabulary)
    uint32_t hidden_size;          // 512 (balanced performance/memory)
    uint32_t num_attention_heads;   // 8 (efficient parallel processing)
    uint32_t num_layers;           // 4 (sufficient depth for byte prediction)
    uint32_t feed_forward_size;    // 512 (2x hidden size, original default)
    
    // Model weight buffers
    id<MTLBuffer> embedding_weights;     // [256 x hidden_size] input embeddings
    id<MTLBuffer> position_embeddings;   // [context_length x hidden_size] positional
    id<MTLBuffer> attention_weights_q;   // [num_layers x hidden_size x hidden_size] query
    id<MTLBuffer> attention_weights_k;   // [num_layers x hidden_size x hidden_size] key  
    id<MTLBuffer> attention_weights_v;   // [num_layers x hidden_size x hidden_size] value
    id<MTLBuffer> attention_output_weights; // [num_layers x hidden_size x hidden_size] output projection
    id<MTLBuffer> ffn_weights_1;         // [num_layers x hidden_size x feed_forward_size] 
    id<MTLBuffer> ffn_weights_2;         // [num_layers x feed_forward_size x hidden_size]
    id<MTLBuffer> layer_norm_weights;    // [num_layers x 2 x hidden_size] pre/post norm
    id<MTLBuffer> output_projection;     // [hidden_size x vocab_size] final projection

    // Bias buffers (use_bias=1: learned by gradient descent, zero-initialized)
    id<MTLBuffer> bias_k;     /* [num_layers x hidden_size] */
    id<MTLBuffer> bias_v;     /* [num_layers x hidden_size] */
    id<MTLBuffer> bias_o;     /* [num_layers x hidden_size] */
    id<MTLBuffer> bias_ffn1;  /* [num_layers x feed_forward_size] */
    id<MTLBuffer> bias_ffn2;  /* [num_layers x hidden_size] */
    id<MTLBuffer> bias_out;   /* [vocab_size] */
    id<MTLBuffer> rel_r;      /* [NH * HD * D_POS] = [8*32*64] rel PE proj  */
    id<MTLBuffer> b_rel_r;    /* [NH * total_len]  = [8*64]    rel PE bias  */
    id<MTLBuffer> ln_final;   /* [2 * hidden_size]: gamma_f, beta_f for LN_FINAL */

    // Computation buffers
    id<MTLBuffer> context_buffer;        // Input context [context_length]
    id<MTLBuffer> embedded_buffer;       // [context_length x hidden_size]
    id<MTLBuffer> attention_buffer;      // [context_length x hidden_size] 
    id<MTLBuffer> ffn_buffer;           // [context_length x hidden_size]
    id<MTLBuffer> logits_buffer;        // [vocab_size] output predictions
    
    // Metal compute pipelines
    id<MTLComputePipelineState> embedding_pipeline;
    id<MTLComputePipelineState> attention_pipeline;
    id<MTLComputePipelineState> ffn_pipeline;
    id<MTLComputePipelineState> output_pipeline;
    
    // Model state
    bool is_initialized;
    bool weights_loaded;
    uint32_t max_sequence_length;
} MetalTransformerModel;

// Metal Compute Shader sources for Transformer operations
static const char* transformer_embedding_shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void transformer_embedding(
    const device int32_t* input_tokens [[buffer(0)]],
    const device float* embedding_weights [[buffer(1)]],
    const device float* position_embeddings [[buffer(2)]],
    device float* output_embeddings [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint seq_len = 65536; // context_length from model
    uint hidden_size = 256;
    
    if (gid >= seq_len) return;
    
    int32_t token = input_tokens[gid];
    if (token < 0 || token >= 256) return; // vocab_size check
    
    // Combine token embedding + positional embedding
    // Use modulo 64 for position to support batched streams
    uint pos_idx = gid % 64;
    
    for (uint h = 0; h < hidden_size; h++) {
        float token_emb = embedding_weights[token * hidden_size + h];
        float pos_emb = position_embeddings[pos_idx * hidden_size + h];
        output_embeddings[gid * hidden_size + h] = token_emb + pos_emb;
    }
}
)";

static const char* transformer_attention_shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void transformer_self_attention(
    const device float* input_embeddings [[buffer(0)]],
    const device float* query_weights [[buffer(1)]],
    const device float* key_weights [[buffer(2)]],
    const device float* value_weights [[buffer(3)]],
    const device float* output_weights [[buffer(4)]],
    device float* output_embeddings [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint seq_len = 65536; // Max context (1024 streams * 64 seq)
    uint hidden_size = 256;
    uint head_dim = hidden_size / 8; // 8 attention heads
    uint real_seq_len = 64; // Per-stream sequence length
    
    uint seq_idx = gid.x;
    uint head_idx = gid.y;
    
    if (seq_idx >= seq_len || head_idx >= 8) return;
    
    // Simplified self-attention computation
    // Query, Key, Value projections for current head
    uint head_offset = head_idx * head_dim;
    
    // Calculate batch boundaries for independent streams
    uint batch_base = (seq_idx / real_seq_len) * real_seq_len;
    uint local_idx = seq_idx % real_seq_len;
    
    for (uint out_dim = 0; out_dim < head_dim; out_dim++) {
        float attention_sum = 0.0f;
        
        // Compute attention weights and apply to values
        // Causal masking within the stream: attend to 0..local_idx
        for (uint k = 0; k <= local_idx; k++) {
            uint src_idx = batch_base + k;
            
            float attention_weight = 1.0f / (local_idx + 1); // Simplified attention
            
            // Value projection
            float value = 0.0f;
            for (uint in_dim = 0; in_dim < hidden_size; in_dim++) {
                value += input_embeddings[src_idx * hidden_size + in_dim] * 
                        value_weights[head_offset * hidden_size + in_dim * head_dim + out_dim];
            }
            attention_sum += attention_weight * value;
        }
        
        output_embeddings[seq_idx * hidden_size + head_offset + out_dim] = attention_sum;
    }
}
)";

static const char* transformer_ffn_shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void transformer_feed_forward(
    const device float* input_embeddings [[buffer(0)]],
    const device float* ffn_weights_1 [[buffer(1)]],
    const device float* ffn_weights_2 [[buffer(2)]],
    device float* output_embeddings [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint seq_len = 4096;
    uint hidden_size = 256;
    uint ff_size = 1024;
    
    if (gid >= seq_len) return;
    
    // Feed-forward: hidden -> ff_size -> hidden with ReLU
    for (uint out_dim = 0; out_dim < hidden_size; out_dim++) {
        float output = 0.0f;
        
        // First layer: hidden -> ff_size with ReLU
        for (uint ff_dim = 0; ff_dim < ff_size; ff_dim++) {
            float ff_value = 0.0f;
            for (uint in_dim = 0; in_dim < hidden_size; in_dim++) {
                ff_value += input_embeddings[gid * hidden_size + in_dim] * 
                           ffn_weights_1[in_dim * ff_size + ff_dim];
            }
            ff_value = max(0.0f, ff_value); // ReLU activation
            
            // Second layer: ff_size -> hidden
            output += ff_value * ffn_weights_2[ff_dim * hidden_size + out_dim];
        }
        
        output_embeddings[gid * hidden_size + out_dim] = output;
    }
}
)";

static const char* transformer_output_shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void transformer_output_projection(
    const device float* hidden_states [[buffer(0)]],
    const device float* output_weights [[buffer(1)]],
    device float* logits [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint seq_len = 4096;
    uint hidden_size = 256;
    uint vocab_size = 256;
    
    uint seq_idx = gid.x;
    uint vocab_idx = gid.y;
    
    if (seq_idx >= seq_len || vocab_idx >= vocab_size) return;
    
    float logit = 0.0f;
    for (uint h = 0; h < hidden_size; h++) {
        logit += hidden_states[seq_idx * hidden_size + h] * 
                 output_weights[h * vocab_size + vocab_idx];
    }
    logits[seq_idx * vocab_size + vocab_idx] = logit;
}
)";

// External function declarations
extern MetalTransformerModel* get_shared_transformer_model(void);
extern bool metal_transformer_prediction(MetalTransformerModel* model,
                                        id<MTLBuffer> hidden_buffer,
                                        id<MTLBuffer> output_logits,
                                        size_t seq_len,
                                        id<MTLCommandBuffer> commandBuffer);

// Implementation of arithmetic coding (exact CUDA port)
static void put_bit_init(PutBitState *s, uint8_t *buf, int buf_size) {
    s->low = 0;
    s->range = RANGE_MAX;
    s->current_byte = 0xff;
    s->n_bytes = 0;
    s->buf = buf;
    s->buf_size = buf_size;
    s->idx = 0;
    s->byte_count = 0;
    assert(PROB_UNIT <= RANGE_MIN);
}

static void put_byte(PutBitState *s, int v) {
    if (s->idx < s->buf_size) {
        s->buf[s->idx++] = v;
    }
}

static void put_val(PutBitState *s, int v) {
    uint32_t carry, b;
    
    if (v == 0xff) {
        s->n_bytes++;
    } else {
        if (s->n_bytes > 0) {
            carry = v >> 8;
            put_byte(s, s->current_byte + carry);
            b = (0xff + carry) & 0xff;
            while (s->n_bytes > 1) {
                put_byte(s, b);
                s->n_bytes--;
            }
        }
        s->n_bytes = 1;
        s->current_byte = v;
    }
}

static void put_bit_renorm(PutBitState *s) {
    uint32_t v;
    while (s->range < RANGE_MIN) {
        v = s->low >> RANGE_MIN_BITS;
        put_val(s, v);
        s->low = (s->low & ((1 << RANGE_MIN_BITS) - 1)) << 8;
        s->range <<= 8;
    }
}

static void put_bit(PutBitState *s, int prob0, int bit) {
    int range0 = ((uint64_t)s->range * prob0) >> PROB_UNIT_BITS;
    assert(range0 > 0);
    assert(range0 < s->range);
    
    if (!bit) {
        s->range = range0;
    } else {
        s->low += range0;
        s->range -= range0;
    }
    
    put_bit_renorm(s);
}

static int64_t put_bit_flush(PutBitState *s) {
    int n, val, mask;
    
    if (s->range < (1 << RANGE_MIN_BITS)) {
        put_val(s, s->low >> RANGE_MIN_BITS);
        s->low = (s->low & ((1 << RANGE_MIN_BITS) - 1)) << 8;
        s->range <<= 8;
    }
    
    n = 0;
    while ((1 << (n + 1)) <= s->range)
        n++;
    
    val = s->low;
    mask = (1 << n) - 1;
    if ((val & mask) != 0)
        val = (val + (1 << n)) & ~mask;
    
    put_val(s, val >> RANGE_MIN_BITS);
    if (s->n_bytes > 0) {
        put_val(s, 0);
    }
    
    return s->idx;
}

// Helper functions
static float vec_sum_f32(const float* data, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum;
}

static inline int clamp_int(int val, int min_val, int max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

// Original NNCP write_sym (from cp_utils.c)
static void write_sym(PutBitState *pb, const float *prob_table, int n_symb, int sym) {
    int start, range, prob0, bit, range0;
    float p, p0;
    
    start = 0;
    range = n_symb;
    p = 1.0;
    int iterations = 0;
    
    while (range > 1 && iterations < 20) {  // Add iteration limit to prevent infinite loop
        iterations++;
        
        range0 = range >> 1;
        p0 = vec_sum_f32(prob_table + start, range0);
        prob0 = lrintf(p0 * PROB_UNIT / p);
        prob0 = clamp_int(prob0, 1, PROB_UNIT - 1);
        bit = sym >= (start + range0);
        
        put_bit(pb, prob0, bit);
        
        if (bit) {
            start += range0;
            range = range - range0;
            p = p - p0;
        } else {
            p = p0;
            range = range0;
        }
    }
}

// Decoder implementation
static void refill(GetBitState *s) {
    s->range <<= 8;
    s->low <<= 8;
    if (s->idx < s->buf_len && !s->eof_reached) {
        s->low += s->buf[s->idx++];
    }
}

static void get_bit_init(GetBitState *s, uint8_t *buf, size_t buf_size) {
    s->buf = buf;
    s->buf_len = buf_size;
    s->buf_size = buf_size;
    s->idx = 0;
    s->low = 0;
    s->range = 0;
    s->eof_reached = false;
    s->byte_count = buf_size;
    
    for (int i = 0; i <= RANGE_MIN_BITS; i += 8) {
        refill(s);
    }
    s->range = RANGE_MAX;
}

static int get_bit(GetBitState *s, int prob0) {
    int b;
    uint32_t range0;
    
    range0 = ((uint64_t)s->range * prob0) >> PROB_UNIT_BITS;
    b = s->low >= range0;
    
    if (b) {
        s->low -= range0;
        s->range -= range0;
    } else {
        s->range = range0;
    }
    
    while (s->range < RANGE_MIN) {
        refill(s);
    }
    return b;
}

static int read_sym(GetBitState *gb, const float *prob_table, int n_symb) {
    int start, range, prob0, bit, range0;
    float p, p0;
    
    start = 0;
    range = n_symb;
    p = 1.0;
    
    while (range > 1) {
        range0 = range >> 1;
        p0 = vec_sum_f32(prob_table + start, range0);
        prob0 = lrintf(p0 * PROB_UNIT / p);
        prob0 = clamp_int(prob0, 1, PROB_UNIT - 1);
        bit = get_bit(gb, prob0);
        
        if (bit) {
            start += range0;
            range = range - range0;
            p = p - p0;
        } else {
            p = p0;
            range = range0;
        }
    }
    return start;
}



// High-performance CPU Transformer prediction (CUDA-compatible architecture)
static bool cpu_transformer_prediction_cuda_optimized(MetalTransformerModel* model, 
                                                       const int32_t* context, int context_len,
                                                       float* probabilities, int vocab_size) {
    if (!model || context_len > 64) return false;
    
    // Advanced adaptive frequency analysis with sequence pattern recognition
    float byte_frequencies[256] = {0};
    float bigram_scores[256] = {0}; // 2-gram pattern scores
    float trigram_scores[256] = {0}; // 3-gram pattern scores
    float context_entropy = 0.0f;
    float sequence_predictability = 0.0f;
    
    if (context_len > 0) {
        // Calculate byte frequency distribution in current context
        for (int i = 0; i < context_len; i++) {
            int byte_val = context[i] % 256;
            byte_frequencies[byte_val] += 1.0f;
        }
        
        // Advanced sequence pattern analysis for programming languages
        float prog_lang_indicators[10] = {0}; // Track various programming constructs
        
        for (int i = 0; i < context_len; i++) {
            int byte_val = context[i] % 256;
            
            // Programming language pattern detection
            if (byte_val == '{' || byte_val == '}') prog_lang_indicators[0] += 1.0f; // Braces
            if (byte_val == '(' || byte_val == ')') prog_lang_indicators[1] += 1.0f; // Parentheses
            if (byte_val == ';') prog_lang_indicators[2] += 1.0f; // Semicolons
            if (byte_val == '=' || byte_val == '+' || byte_val == '-') prog_lang_indicators[3] += 1.0f; // Operators
            if (byte_val == '"' || byte_val == '\'') prog_lang_indicators[4] += 1.0f; // Quotes
            if (byte_val == 10 || byte_val == 13) prog_lang_indicators[5] += 1.0f; // Newlines
            if (byte_val == 32 || byte_val == 9) prog_lang_indicators[6] += 1.0f; // Whitespace
            if ((byte_val >= 'a' && byte_val <= 'z') || (byte_val >= 'A' && byte_val <= 'Z')) prog_lang_indicators[7] += 1.0f; // Letters
            if (byte_val >= '0' && byte_val <= '9') prog_lang_indicators[8] += 1.0f; // Numbers
            if (byte_val == '.' || byte_val == ',' || byte_val == ':') prog_lang_indicators[9] += 1.0f; // Punctuation
            
            // Bigram pattern analysis (last 2 bytes)
            if (i > 0) {
                int prev_byte = context[i-1] % 256;
                int pattern_score = 0;
                
                // Common programming patterns get higher scores
                if ((prev_byte == '=' && byte_val == '=') ||  // ==
                    (prev_byte == '!' && byte_val == '=') ||  // !=
                    (prev_byte == '<' && byte_val == '=') ||  // <=
                    (prev_byte == '>' && byte_val == '=') ||  // >=
                    (prev_byte == '&' && byte_val == '&') ||  // &&
                    (prev_byte == '|' && byte_val == '|')) {  // ||
                    pattern_score = 3;
                } else if ((prev_byte == '/' && byte_val == '/') ||  // //
                           (prev_byte == '/' && byte_val == '*') ||  // /*
                           (prev_byte == '*' && byte_val == '/')) {  // */
                    pattern_score = 2;
                } else if ((prev_byte >= 'a' && prev_byte <= 'z') && 
                           (byte_val >= 'a' && byte_val <= 'z')) {    // letter sequences
                    pattern_score = 1;
                }
                
                bigram_scores[byte_val] += pattern_score * 0.5f;
            }
            
            // Trigram pattern analysis (last 3 bytes)
            if (i > 1) {
                int prev2_byte = context[i-2] % 256;
                int prev1_byte = context[i-1] % 256;
                
                // Function call patterns: "();"
                if (prev2_byte == ')' && prev1_byte == ';') {
                    trigram_scores[byte_val] += 2.0f;
                }
                // Array access: "[i]"
                if (prev2_byte == '[' && prev1_byte >= '0' && prev1_byte <= '9' && byte_val == ']') {
                    trigram_scores[byte_val] += 2.0f;
                }
                // Conditional logic: "if "
                if (prev2_byte == 'i' && prev1_byte == 'f' && byte_val == ' ') {
                    trigram_scores[byte_val] += 3.0f;
                }
            }
        }
        
        // Calculate sequence predictability based on programming patterns
        float total_pattern_score = 0.0f;
        for (int i = 0; i < 10; i++) {
            total_pattern_score += prog_lang_indicators[i];
        }
        sequence_predictability = fminf(1.0f, total_pattern_score / (context_len * 0.7f));
        
        // Normalize frequencies and calculate entropy
        float total_bytes = (float)context_len;
        for (int i = 0; i < 256; i++) {
            if (byte_frequencies[i] > 0) {
                byte_frequencies[i] /= total_bytes;
                context_entropy -= byte_frequencies[i] * logf(byte_frequencies[i] + 1e-8f);
            }
            
            // Normalize pattern scores
            bigram_scores[i] = fminf(2.0f, bigram_scores[i] / fmaxf(1.0f, total_bytes * 0.1f));
            trigram_scores[i] = fminf(3.0f, trigram_scores[i] / fmaxf(1.0f, total_bytes * 0.1f));
        }
        context_entropy /= logf(2.0f); // Convert to bits
    }
    
    // Advanced adaptive factors for intelligent prediction
    float complexity_factor = fminf(1.0f, context_entropy / 6.0f); // Normalize 0-1
    float predictability_boost = 0.5f + 1.5f * sequence_predictability; // Scale 0.5-2.0
    float frequency_bias_strength = (0.2f + 0.8f * complexity_factor) * predictability_boost; // Enhanced scaling
    
    // CUDA-compatible optimized parameters  
    const int hidden_size = 256;        // Reduced from 512 for speed
    const int num_heads = 4;             // Reduced from 8 for speed  
    const int head_dim = hidden_size / num_heads;  // 64
    const int ffn_size = 512;            // Reduced from 2048 for speed
    const int seq_len = context_len > 0 ? context_len : 1;
    
    // Stack-allocated buffers for maximum performance
    float embeddings[64 * 256];         // seq_len * hidden_size
    float attention_out[64 * 256];      // attention output
    float ffn_out[64 * 256];            // feed-forward output
    float logits[256];                  // vocab logits
    
    // Step 1: Enhanced Token Embeddings with Contextual Intelligence
    for (int pos = 0; pos < seq_len; pos++) {
        int token = (pos < context_len) ? context[pos] : 256;  // BOS padding
        token = token % 256;  // Clamp to vocabulary
        
        // Advanced embedding with multi-pattern intelligence
        for (int h = 0; h < hidden_size; h++) {
            // Multi-factor adaptive embedding
            float base_embed = 0.15f * (token / 255.0f);
            
            // Apply frequency adaptation
            if (byte_frequencies[token] > 0) {
                float freq_factor = 1.0f + byte_frequencies[token] * complexity_factor * 1.2f;
                base_embed *= freq_factor;
            }
            
            // Apply sequence pattern boosting
            float pattern_boost = 1.0f;
            if (bigram_scores[token] > 0) pattern_boost += bigram_scores[token] * 0.3f;
            if (trigram_scores[token] > 0) pattern_boost += trigram_scores[token] * 0.2f;
            base_embed *= pattern_boost;
            
            float embed = base_embed;  // Multi-pattern adapted embedding
            float pos_enc = sinf((pos * 1000.0f + h) / 10000.0f) * 0.8f;  // Reduced positional influence
            
            // Advanced contextual bias with programming language awareness
            float contextual_bias = 0.0f;
            if (pos > 0) {
                int prev_token = (pos-1 < context_len) ? context[pos-1] : 256;
                prev_token = prev_token % 256;
                
                // Enhanced programming language pattern recognition
                if (token >= 32 && token <= 126) contextual_bias += 0.06f * predictability_boost;  // Printable ASCII
                if (token >= 65 && token <= 90) contextual_bias += 0.04f * predictability_boost;   // Uppercase
                if (token >= 97 && token <= 122) contextual_bias += 0.10f * predictability_boost;  // Lowercase
                
                // Specific programming patterns
                if ((prev_token == '=' && token == '=') ||
                    (prev_token == '!' && token == '=') ||
                    (prev_token == '<' && token == '=') ||
                    (prev_token == '>' && token == '=')) {
                    contextual_bias += 0.15f * predictability_boost;  // Comparison operators
                }
                
                if ((prev_token == '/' && token == '/') ||
                    (prev_token == '/' && token == '*')) {
                    contextual_bias += 0.12f * predictability_boost;  // Comment start patterns
                }
                
                if ((prev_token >= 'a' && prev_token <= 'z') && 
                    (token >= 'a' && token <= 'z')) {
                    contextual_bias += 0.08f * predictability_boost;  // Identifier continuation
                }
                if (token >= 48 && token <= 57) contextual_bias += 0.05f;   // Digits
                
                // Sequential character patterns (balanced)
                if (abs(token - prev_token) == 1) contextual_bias += 0.08f; // Sequential
                if (token == prev_token) contextual_bias += 0.15f;          // Repeated characters
                if (token == ' ' && prev_token != ' ') contextual_bias += 0.1f; // Word boundaries
                
                // Programming language patterns (balanced) 
                if (token == '{' || token == '}') contextual_bias += 0.08f; // Braces
                if (token == '(' || token == ')') contextual_bias += 0.08f; // Parentheses
                if (token == ';' || token == ',') contextual_bias += 0.08f; // Punctuation
            }
            
            embeddings[pos * hidden_size + h] = embed + 0.1f * pos_enc + 0.05f * contextual_bias;
        }
    }
    
    // Step 2: Self-Attention (CUDA-compatible streamlined)
    for (int pos = 0; pos < seq_len; pos++) {
        float* att_out = &attention_out[pos * hidden_size];
        
        // Multi-head attention simplified: average pooling with learned weights
        for (int h = 0; h < hidden_size; h++) {
            float attention_sum = 0.0f;
            float weight_sum = 0.0f;
            
            // Attend to all previous positions including current
            for (int src = 0; src <= pos; src++) {
                float query = embeddings[pos * hidden_size + h];
                float key = embeddings[src * hidden_size + h];
                
                // Simplified attention weight: dot product + positional bias
                float att_weight = query * key + 0.1f / (1.0f + abs(pos - src));
                att_weight = expf(att_weight * 0.1f);  // Softmax component
                
                attention_sum += att_weight * embeddings[src * hidden_size + h];
                weight_sum += att_weight;
            }
            
            // Normalized attention output
            att_out[h] = (weight_sum > 0.0f) ? attention_sum / weight_sum : 0.0f;
        }
    }
    
    // Step 3: Feed-Forward Network (CUDA-compatible two-layer MLP)
    for (int pos = 0; pos < seq_len; pos++) {
        float* ffn_output = &ffn_out[pos * hidden_size];
        float* att_input = &attention_out[pos * hidden_size];
        
        // First layer: hidden -> ffn_size with ReLU
        float ffn_hidden[512];  // Stack allocation for speed
        for (int f = 0; f < ffn_size; f++) {
            float sum = 0.0f;
            for (int h = 0; h < hidden_size; h++) {
                // Simplified weight: hash-based deterministic initialization
                float weight = sinf((h * 7 + f * 13) * 0.01f) * 0.1f;
                sum += att_input[h] * weight;
            }
            ffn_hidden[f] = fmaxf(0.0f, sum);  // ReLU activation
        }
        
        // Second layer: ffn_size -> hidden with residual connection
        for (int h = 0; h < hidden_size; h++) {
            float sum = 0.0f;
            for (int f = 0; f < ffn_size; f++) {
                // Simplified weight: hash-based deterministic initialization  
                float weight = cosf((f * 11 + h * 17) * 0.01f) * 0.1f;
                sum += ffn_hidden[f] * weight;
            }
            // Residual connection + layer normalization
            ffn_output[h] = att_input[h] + 0.1f * sum;
        }
    }
    
    // Step 4: Intelligent Output Projection with Contextual Predictions
    int last_pos = seq_len - 1;
    float* final_hidden = &ffn_out[last_pos * hidden_size];
    
    // Get recent context for intelligent biasing
    int last_byte = (context_len > 0) ? context[context_len - 1] : 0;
    int prev_byte = (context_len > 1) ? context[context_len - 2] : 0;
    
    for (int vocab = 0; vocab < vocab_size; vocab++) {
        // Base logit from hidden state transformation
        float logit = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            // Multi-dimensional adaptive weight computation
            float base_weight = sinf((h * 23 + vocab * 29) * 0.01f) * 0.04f;
            
            // Multi-factor adaptive scaling
            float adaptive_factor = 1.0f;
            if (vocab < 256) {
                // Frequency-based adaptation
                if (byte_frequencies[vocab] > 0) {
                    adaptive_factor *= 1.0f + byte_frequencies[vocab] * complexity_factor * 0.8f;
                }
                
                // Pattern-based adaptation
                if (bigram_scores[vocab] > 0) {
                    adaptive_factor *= 1.0f + bigram_scores[vocab] * 0.4f;
                }
                
                if (trigram_scores[vocab] > 0) {
                    adaptive_factor *= 1.0f + trigram_scores[vocab] * 0.3f;
                }
                
                // Sequence predictability adaptation
                adaptive_factor *= predictability_boost;
            }
            
            float out_weight = base_weight * adaptive_factor;
            logit += final_hidden[h] * out_weight;
        }
        
        // Apply advanced multi-pattern contextual biasing
        float contextual_boost = 0.0f;
        
        // Enhanced frequency-based bias for observed bytes
        if (vocab < 256 && byte_frequencies[vocab] > 0) {
            float frequency_boost = byte_frequencies[vocab] * frequency_bias_strength * 1.8f;
            contextual_boost += frequency_boost;
        }
        
        // Pattern-based bias enhancement
        if (vocab < 256) {
            if (bigram_scores[vocab] > 0) {
                contextual_boost += bigram_scores[vocab] * 0.6f;
            }
            if (trigram_scores[vocab] > 0) {
                contextual_boost += trigram_scores[vocab] * 0.4f;
            }
        }
        
        if (vocab < 256) {  // Enhanced character-specific biasing
            // Programming language character preferences
            if (vocab >= 32 && vocab <= 126) contextual_boost += 0.12f * predictability_boost;  // Printable ASCII
            if (vocab >= 97 && vocab <= 122) contextual_boost += 0.20f * predictability_boost;  // Lowercase letters
            if (vocab >= 65 && vocab <= 90) contextual_boost += 0.10f * predictability_boost;   // Uppercase letters
            if (vocab >= 48 && vocab <= 57) contextual_boost += 0.08f * predictability_boost;   // Digits
            if (vocab == ' ') contextual_boost += 0.18f * predictability_boost;                 // Space
            if (vocab == 10 || vocab == 9) contextual_boost += 0.12f * predictability_boost;   // Newline and tab
            
            // Enhanced repetition and sequence patterns
            if (vocab == last_byte) contextual_boost += 0.4f * predictability_boost;           // Character repetition
            if (abs(vocab - last_byte) <= 1) contextual_boost += 0.20f * predictability_boost; // Adjacent characters
            
            // Extended programming language patterns with predictability scaling
            if (last_byte == 'c' && vocab == 'o') contextual_boost += 0.25f * predictability_boost; // "co"
            if (last_byte == 't' && vocab == 'h') contextual_boost += 0.25f * predictability_boost; // "th"
            if (last_byte == 'e' && vocab == 'r') contextual_boost += 0.20f * predictability_boost; // "er"
            if (last_byte == 'i' && vocab == 'n') contextual_boost += 0.22f * predictability_boost; // "in"
            if (last_byte == 'o' && vocab == 'n') contextual_boost += 0.20f * predictability_boost; // "on"
            if (last_byte == 'a' && vocab == 'n') contextual_boost += 0.18f * predictability_boost; // "an"
            if (last_byte == 'r' && vocab == 'e') contextual_boost += 0.18f * predictability_boost; // "re"
            
            // Programming-specific operator sequences
            if ((last_byte == '=' && vocab == '=') ||
                (last_byte == '!' && vocab == '=') ||
                (last_byte == '<' && vocab == '=') ||
                (last_byte == '>' && vocab == '=')) {
                contextual_boost += 0.35f * predictability_boost;
            }
            
            // Comment and string patterns
            if ((last_byte == '/' && vocab == '/') ||
                (last_byte == '/' && vocab == '*')) {
                contextual_boost += 0.30f * predictability_boost;
            }
            if (last_byte == 'i' && vocab == 'n') contextual_boost += 0.25f; // "in" (in, function)
            
            // TypeScript/JavaScript specific sequences  
            if (last_byte == 'f' && vocab == 'u') contextual_boost += 0.3f; // "fu" (function)
            if (last_byte == 'v' && vocab == 'a') contextual_boost += 0.25f; // "va" (var, value)
            if (last_byte == 'l' && vocab == 'e') contextual_boost += 0.25f; // "le" (let, file)
            if (last_byte == 'r' && vocab == 'e') contextual_boost += 0.25f; // "re" (return, require)
            
            // Bracket matching boost
            if ((last_byte == '(' && vocab == ')') || 
                (last_byte == '{' && vocab == '}') || 
                (last_byte == '[' && vocab == ']')) contextual_boost += 0.6f;
            
            // Common word endings (more selective)
            if ((prev_byte == 'i' && last_byte == 'o' && vocab == 'n') || // "ion"
                (prev_byte == 'e' && last_byte == 'r' && vocab == 's')) { // "ers"
                contextual_boost += 0.4f;
            }
            
            // Whitespace context predictions (reduced)
            if (last_byte == ' ') {
                if (vocab >= 97 && vocab <= 122) contextual_boost += 0.25f; // Letter after space
                if (vocab == '{' || vocab == '}') contextual_boost += 0.2f; // Braces after space
            }
        }
        
        logits[vocab] = logit + contextual_boost;
    }
    
    // Step 5: Softmax Normalization (CUDA-compatible numerical stability)
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probabilities[i] = expf(logits[i] - max_logit);
        sum_exp += probabilities[i];
    }
    
    // Final normalization
    if (sum_exp > 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            probabilities[i] /= sum_exp;
        }
    }
    
    return true;
}

// Metal Transformer prediction integration
static bool get_metal_transformer_prediction(const int32_t* context, int context_len, 
                                           float* probabilities, int vocab_size) {
    // Performance measurement for debugging
    static int call_count = 0;
    static double total_time = 0.0;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    call_count++;
    // printf("[PERF] Transformer call #%d starting...", call_count);
    fflush(stdout);
    
    // Get Metal Transformer model instance
    MetalTransformerModel* model = get_shared_transformer_model();
    if (!model) {
        // Fallback to uniform distribution
        for (int j = 0; j < vocab_size; j++) {
            probabilities[j] = 1.0f / vocab_size;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_time += elapsed;
        return false;
    }
    
    @autoreleasepool {
        // Use model's device instead of creating new one
        id<MTLDevice> device = model->device;
    
        size_t seq_len = MIN(context_len > 0 ? context_len : 1, model->context_length);
        
        // Reuse model's existing buffers to avoid allocation overhead
        id<MTLBuffer> hidden_buffer = model->embedded_buffer;
        id<MTLBuffer> logits_buffer = model->logits_buffer;
    
        // Copy context data to model's context buffer
        int32_t* context_data = (int32_t*)[model->context_buffer contents];
        for (size_t i = 0; i < model->context_length; i++) {
            if (i < context_len) {
                context_data[i] = context[i];
            } else {
                context_data[i] = 256; // Padding token (out of vocab range)
            }
        }
    
        // Use model's command queue for better performance
        id<MTLCommandBuffer> commandBuffer = [model->command_queue commandBuffer];
    
    // Use optimized CPU Transformer prediction (CUDA-compatible)
        bool success = cpu_transformer_prediction_cuda_optimized(model, context, context_len, probabilities, vocab_size);
    
    if (success) {
        // CPU implementation already wrote probabilities directly
        
        // Performance measurement completion
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_time += elapsed;
        
        // printf(" completed in %.2fms\n", elapsed * 1000.0);
        fflush(stdout);
        
        return true;
    } else {
        // Fallback to context-aware distribution
        for (int j = 0; j < vocab_size; j++) {
            if (j < 256) {
                float bias = 1.0f;
                if (context_len > 0) {
                    int last_byte = context[context_len - 1];
                    if (last_byte < 256 && abs(j - last_byte) < 32) {
                        bias = 1.3f; // Favor nearby values
                    }
                }
                probabilities[j] = bias / vocab_size;
            } else {
                probabilities[j] = 0.1f / vocab_size; // BOS/EOS
            }
        }
        
        // Normalize
        float total = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            total += probabilities[j];
        }
        if (total > 0.0f) {
            for (int j = 0; j < vocab_size; j++) {
                probabilities[j] /= total;
            }
        }
        
        // Performance measurement completion
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_time += elapsed;
        
        return false;
    } // end autoreleasepool
    } // end @autoreleasepool
}

// Shared Flow Context
static FlowOptimizerContext* g_flow_ctx = NULL;

static FlowOptimizerContext* get_shared_flow_context() {
    if (!g_flow_ctx) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        g_flow_ctx = flow_optimizer_create(device);
        
        // Get the shared model to sync config and weights
        MetalTransformerModel* model = get_shared_transformer_model();
        
        GPUTransformerConfig trf_config = {
            .num_layers = model ? model->num_layers : 4,
            .hidden_size = model ? model->hidden_size : 512,
            .num_heads = model ? model->num_attention_heads : 8,
            .ffn_size = model ? model->feed_forward_size : 2048,
            .context_length = model ? model->context_length : 64,
            .vocab_size = model ? model->vocab_size : 256
        };
        
        if (!flow_optimizer_setup_transformer(g_flow_ctx, trf_config)) {
            return NULL;
        }
        
        if (model) {
            flow_optimizer_set_transformer_weights(g_flow_ctx,
                model->embedding_weights,
                model->position_embeddings,
                model->attention_weights_q,
                model->attention_weights_k,
                model->attention_weights_v,
                model->attention_output_weights,
                model->ffn_weights_1,
                model->ffn_weights_2,
                model->layer_norm_weights,
                model->output_projection
            );
        }
    }
    return g_flow_ctx;
}

// Shared MPS Transformer Context (batched KV-cache decode path)
/* -------------------------------------------------------------------------
 * Session weight snapshots
 *
 * On the first call to ensure_session_weights() the current MTLBuffer
 * contents are copied to CPU-side float* buffers.  Subsequent calls to
 * reset_model_to_session_weights() restore those buffers, guaranteeing that
 * every compress / decompress session starts from the same initial weights.
 *
 * Position embeddings are excluded from the snapshot: they are computed by
 * the sinusoidal formula (deterministic, never mutated), so they are
 * always identical without explicit snapshotting.
 * ---------------------------------------------------------------------- */

static float* g_session_init_embed     = NULL;  /* [V * H]          */
static float* g_session_init_attn_q    = NULL;  /* [L * H * H]      */
static float* g_session_init_attn_k    = NULL;  /* [L * H * H]      */
static float* g_session_init_attn_v    = NULL;  /* [L * H * H]      */
static float* g_session_init_attn_out  = NULL;  /* [L * H * H]      */
static float* g_session_init_ffn1      = NULL;  /* [L * H * FFS]    */
static float* g_session_init_ffn2      = NULL;  /* [L * FFS * H]    */
static float* g_session_init_ln        = NULL;  /* [L * 4 * H]      */
static float* g_session_init_out_proj  = NULL;  /* [H * V]          */
static float* g_session_init_bias_k    = NULL;  /* [L * H]          */
static float* g_session_init_bias_v    = NULL;  /* [L * H]          */
static float* g_session_init_bias_o    = NULL;  /* [L * H]          */
static float* g_session_init_bias_ffn1 = NULL;  /* [L * FFS]        */
static float* g_session_init_bias_ffn2 = NULL;  /* [L * H]          */
static float* g_session_init_bias_out  = NULL;  /* [V]              */
static float* g_session_init_rel_r     = NULL;  /* [NH * HD * D_POS] */
static float* g_session_init_b_rel_r   = NULL;  /* [NH * total_len]  */
static float* g_session_init_ln_final  = NULL;  /* [2 * H]           */
static bool   g_session_weights_ready  = false;

/* Snapshot the model's current weights into CPU buffers (called once). */
static void ensure_session_weights(MetalTransformerModel* model) {
    if (g_session_weights_ready || !model) return;

    const size_t L   = model->num_layers;
    const size_t H   = model->hidden_size;
    const size_t V   = model->vocab_size;
    const size_t FFS = model->feed_forward_size;

#define SNAPSHOT(dst, buf, n) do { \
    size_t _sz = (n) * sizeof(float); \
    (dst) = (float*)malloc(_sz); \
    if (!(dst)) { fprintf(stderr, "[session_weights] malloc failed\n"); return; } \
    memcpy((dst), [(buf) contents], _sz); \
} while (0)

    SNAPSHOT(g_session_init_embed,    model->embedding_weights,       V * H);
    SNAPSHOT(g_session_init_attn_q,   model->attention_weights_q,     L * H * H);
    SNAPSHOT(g_session_init_attn_k,   model->attention_weights_k,     L * H * H);
    SNAPSHOT(g_session_init_attn_v,   model->attention_weights_v,     L * H * H);
    SNAPSHOT(g_session_init_attn_out, model->attention_output_weights, L * H * H);
    SNAPSHOT(g_session_init_ffn1,     model->ffn_weights_1,           L * H * FFS * 2);
    SNAPSHOT(g_session_init_ffn2,     model->ffn_weights_2,           L * FFS * H);
    SNAPSHOT(g_session_init_ln,       model->layer_norm_weights,      L * 4 * H);
    SNAPSHOT(g_session_init_out_proj, model->output_projection,       H * V);
    SNAPSHOT(g_session_init_bias_k,    model->bias_k,    L * H);
    SNAPSHOT(g_session_init_bias_v,    model->bias_v,    L * H);
    SNAPSHOT(g_session_init_bias_o,    model->bias_o,    L * H);
    SNAPSHOT(g_session_init_bias_ffn1, model->bias_ffn1, L * FFS * 2);
    SNAPSHOT(g_session_init_bias_ffn2, model->bias_ffn2, L * H);
    SNAPSHOT(g_session_init_bias_out,  model->bias_out,  V);
    {
        const size_t NH_ = model->num_attention_heads;
        const size_t HD_ = model->hidden_size / NH_;
        const size_t DP_ = NNCP_D_POS();
        SNAPSHOT(g_session_init_rel_r,   model->rel_r,   NH_ * HD_ * DP_); /* [NH,HD,D_POS] */
        SNAPSHOT(g_session_init_b_rel_r, model->b_rel_r, NH_ * DP_);       /* [NH,D_POS] */
    }
    if (model->ln_final)
        SNAPSHOT(g_session_init_ln_final, model->ln_final, 2 * H);

#undef SNAPSHOT

    g_session_weights_ready = true;
}

/* Restore MTLBuffer contents from the snapshot (call at compress/decompress start). */
static void reset_model_to_session_weights(MetalTransformerModel* model) {
    ensure_session_weights(model);
    if (!g_session_weights_ready || !model) return;

    const size_t L   = model->num_layers;
    const size_t H   = model->hidden_size;
    const size_t V   = model->vocab_size;
    const size_t FFS = model->feed_forward_size;

#define RESTORE(src, buf, n) \
    memcpy([(buf) contents], (src), (n) * sizeof(float))

    RESTORE(g_session_init_embed,    model->embedding_weights,        V * H);
    RESTORE(g_session_init_attn_q,   model->attention_weights_q,      L * H * H);
    RESTORE(g_session_init_attn_k,   model->attention_weights_k,      L * H * H);
    RESTORE(g_session_init_attn_v,   model->attention_weights_v,      L * H * H);
    RESTORE(g_session_init_attn_out, model->attention_output_weights,  L * H * H);
    RESTORE(g_session_init_ffn1,     model->ffn_weights_1,            L * H * FFS * 2);
    RESTORE(g_session_init_ffn2,     model->ffn_weights_2,            L * FFS * H);
    RESTORE(g_session_init_ln,       model->layer_norm_weights,       L * 4 * H);
    RESTORE(g_session_init_out_proj, model->output_projection,        H * V);
    RESTORE(g_session_init_bias_k,    model->bias_k,    L * H);
    RESTORE(g_session_init_bias_v,    model->bias_v,    L * H);
    RESTORE(g_session_init_bias_o,    model->bias_o,    L * H);
    RESTORE(g_session_init_bias_ffn1, model->bias_ffn1, L * FFS * 2);
    RESTORE(g_session_init_bias_ffn2, model->bias_ffn2, L * H);
    RESTORE(g_session_init_bias_out,  model->bias_out,  V);
    {
        const size_t NH_ = model->num_attention_heads;
        const size_t HD_ = model->hidden_size / NH_;
        const size_t DP_ = NNCP_D_POS();
        if (g_session_init_rel_r)   RESTORE(g_session_init_rel_r,   model->rel_r,   NH_ * HD_ * DP_);
        if (g_session_init_b_rel_r) RESTORE(g_session_init_b_rel_r, model->b_rel_r, NH_ * DP_);
    }
    if (g_session_init_ln_final && model->ln_final)
        RESTORE(g_session_init_ln_final, model->ln_final, 2 * H);

#undef RESTORE

}

static OnlineTrainer*         g_online_trainer = NULL;
static MPSTransformerContext* g_mps_ctx = NULL;

static MPSTransformerContext* get_shared_mps_ctx() {
    if (!g_mps_ctx) {
        MetalTransformerModel* model = get_shared_transformer_model();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        MPSTransformerConfig cfg = {
            .num_layers  = (uint32_t)(model ? model->num_layers            : 4),
            .hidden_size = (uint32_t)(model ? model->hidden_size           : 512),
            .num_heads   = (uint32_t)(model ? model->num_attention_heads   : 8),
            .head_dim    = (uint32_t)((model ? model->hidden_size : 512) /
                                      (model ? model->num_attention_heads : 8)),
            .ffn_size    = (uint32_t)(model ? model->feed_forward_size     : 2048),
            .vocab_size  = (uint32_t)(model ? model->vocab_size            : 256),
            // Use max_sequence_length (64) not context_length (65536) to keep KV cache small
            .max_seq_len = (uint32_t)(model ? model->max_sequence_length   : 64),
        };
        g_mps_ctx = mps_transformer_create(device, cfg);
        if (g_mps_ctx && model) {
            mps_transformer_set_weights(g_mps_ctx,
                model->embedding_weights,
                model->position_embeddings,
                model->attention_weights_q,
                model->attention_weights_k,
                model->attention_weights_v,
                model->attention_output_weights,
                model->ffn_weights_1,
                model->ffn_weights_2,
                model->layer_norm_weights,
                model->output_projection,
                model->bias_k,
                model->bias_v,
                model->bias_o,
                model->bias_ffn1,
                model->bias_ffn2,
                model->bias_out,
                model->rel_r,
                model->b_rel_r,
                model->ln_final);
        }
    }
    return g_mps_ctx;
}

// Profile-driven: set g_nncp_profile before compress/decompress
#define NUM_STREAMS  (g_nncp_profile.num_streams)
#define SEG_LEN      (g_nncp_profile.seg_len)
#define MEM_LEN      (g_nncp_profile.mem_len)
#define BLOCK_LEN    500000  // lookahead block size (original default)

static mach_timebase_info_data_t g_tb = {};

// Main CUDA-compatible lossless compression function — segment-based (Case 3 architecture)
size_t neural_bridge_cuda_lossless_compress(const uint8_t* input_data, size_t input_size,
                                           uint8_t* output_data, size_t output_capacity,
                                           const NeuralCompressionConfig* config) {
    if (!input_data || !output_data || input_size == 0) {
        printf("CUDA Lossless compression: Invalid input parameters\n");
        return 0;
    }

    /* Reset model to deterministic session weights so compress and decompress
     * always start from the identical initial state. */
    reset_model_to_session_weights(get_shared_transformer_model());


    // Each stream processes ceil(input_size / NUM_STREAMS) bytes.
    const size_t stride = (input_size + NUM_STREAMS - 1) / NUM_STREAMS;

    // ---- Arithmetic encoders ----
    const int ns = NUM_STREAMS;  // capture once (runtime value)
    PutBitState* encoders = (PutBitState*)malloc((size_t)ns * sizeof(PutBitState));
    uint8_t** stream_buffers = (uint8_t**)malloc((size_t)ns * sizeof(uint8_t*));
    if (!encoders || !stream_buffers) { free(encoders); free(stream_buffers); return 0; }
    const size_t est_capacity = stride * 2 + 4096;
    for (int i = 0; i < ns; i++) {
        stream_buffers[i] = (uint8_t*)malloc(est_capacity);
        if (!stream_buffers[i]) {
            for (int j = 0; j < i; j++) free(stream_buffers[j]);
            free(stream_buffers); free(encoders);
            return 0;
        }
        put_bit_init(&encoders[i], stream_buffers[i], est_capacity);
    }

    const int vocab_size = 256;

    // ---- MPS context ----
    MPSTransformerContext* mps_ctx = get_shared_mps_ctx();
    if (!mps_ctx) {
        for (int i = 0; i < ns; i++) free(stream_buffers[i]);
        free(stream_buffers); free(encoders);
        return 0;
    }
    mps_transformer_reset_kv_cache(mps_ctx);

    // ---- Online trainer ----
    if (!g_online_trainer) {
        g_online_trainer = online_trainer_create(MTLCreateSystemDefaultDevice(),
                                                  mps_ctx, (g_lr_override > 0.0f) ? g_lr_override : 3e-4f,
                                                  input_size);
    }
    if (g_online_trainer) {
        online_trainer_reset_session(g_online_trainer, false);
    }

    if (g_tb.denom == 0) mach_timebase_info(&g_tb);
    int    perf_count        = 0;
    double perf_decode_total = 0.0;
    double perf_train_total  = 0.0;

    // ---- Segment-level working buffers (heap: 16×32×256×4 ≈ 512 KB) ----
    float*   seg_logits  = (float*)  malloc((size_t)NUM_STREAMS * SEG_LEN * vocab_size * sizeof(float));
    float*   probs       = (float*)  malloc(vocab_size * sizeof(float));
    int32_t* seg_tokens  = (int32_t*)malloc((size_t)NUM_STREAMS * SEG_LEN * sizeof(int32_t));
    int32_t* seg_targets = (int32_t*)malloc((size_t)NUM_STREAMS * SEG_LEN * sizeof(int32_t));
    if (!seg_logits || !probs || !seg_tokens || !seg_targets) {
        free(seg_logits); free(probs); free(seg_tokens); free(seg_targets);
        for (int i = 0; i < ns; i++) free(stream_buffers[i]);
        free(stream_buffers); free(encoders);
        return 0;
    }

    // ---- Block-level loop (Transformer-XL: BLOCK_LEN bytes per stream before memory shift) ----
    const size_t total_blocks = (stride + BLOCK_LEN - 1) / BLOCK_LEN;
    size_t file_pos  = 0;
    size_t block_num = 0;

    while (file_pos < stride) {
        const size_t block_bytes = ((stride - file_pos) < (size_t)BLOCK_LEN)
                                   ? (stride - file_pos) : (size_t)BLOCK_LEN;

        // Original NNCP trf_reset(): zero KV memory at each block boundary (fresh context per block).
        mps_transformer_reset_kv_cache(mps_ctx);

        // ---- Segment loop within this block ----
        size_t block_idx = 0;
        while (block_idx < block_bytes) {

            // 1. Build input tokens:
            //    seg_tokens[s * SEG_LEN + t] = the byte PRECEDING position
            //    (file_pos + block_idx + t) in stream s, or BOS(0) for the very first byte.
            memset(seg_tokens, 0, (size_t)NUM_STREAMS * SEG_LEN * sizeof(int32_t));
            for (int s = 0; s < NUM_STREAMS; s++) {
                for (int t = 0; t < SEG_LEN; t++) {
                    const size_t abs_pos  = file_pos + block_idx + (size_t)t;
                    const size_t data_off = (size_t)s * stride + abs_pos;
                    // preceding byte: data_off-1 must be >= s*stride (within this stream)
                    if (abs_pos > 0 && data_off > 0 && (data_off - 1) < input_size) {
                        seg_tokens[s * SEG_LEN + t] = (int32_t)input_data[data_off - 1];
                    }
                    // else keep BOS = 0
                }
            }

            // Phase M: latch pre-segment KV memory BEFORE forward pass
            if (g_online_trainer) online_trainer_latch_kv_memory(g_online_trainer);

            // 2. Segment forward pass → seg_logits[NUM_STREAMS × SEG_LEN × vocab_size].
            uint64_t perf_t0 = mach_absolute_time();
            mps_transformer_execute_segment(mps_ctx, seg_tokens, NUM_STREAMS, SEG_LEN, seg_logits);
            uint64_t perf_t1 = mach_absolute_time();

            // 4. Arithmetic encode + buffer training pairs (t outer, s inner for training symmetry).
            for (int t = 0; t < SEG_LEN; t++) {
                const size_t abs_pos = file_pos + block_idx + (size_t)t;
                if (abs_pos >= file_pos + block_bytes) break;  // past this block

                for (int s = 0; s < NUM_STREAMS; s++) {
                    const size_t data_off = (size_t)s * stride + abs_pos;
                    if (data_off >= input_size) continue;  // stream s exhausted

                    float* raw = seg_logits + ((size_t)(s * SEG_LEN + t)) * vocab_size;

                    // Softmax with NaN/Inf guard: if any logit is non-finite, use uniform
                    bool has_nan = false;
                    for (int k = 0; k < vocab_size && !has_nan; k++)
                        if (!isfinite(raw[k])) has_nan = true;
                    if (has_nan) {
                        const float unif = 1.0f / (float)vocab_size;
                        for (int k = 0; k < vocab_size; k++) probs[k] = unif;
                    } else {
                        float max_l = raw[0];
                        for (int k = 1; k < vocab_size; k++)
                            if (raw[k] > max_l) max_l = raw[k];
                        float sum = 0.0f;
                        for (int k = 0; k < vocab_size; k++) {
                            probs[k] = expf(raw[k] - max_l);
                            sum += probs[k];
                        }
                        for (int k = 0; k < vocab_size; k++) probs[k] /= sum;
                    }

                    const uint8_t byte_val = input_data[data_off];
                    write_sym(&encoders[s], probs, vocab_size, (int)byte_val);

                    // Accumulate training target for segment-level backward pass.
                    if (g_online_trainer)
                        seg_targets[s * SEG_LEN + t] = (int32_t)byte_val;
                }
            }

            // 5. One backward pass over the full [NUM_STREAMS × SEG_LEN] segment.
            uint64_t perf_t2 = perf_t1;
            if (g_online_trainer && !getenv("NNCP_NO_TRAIN")) {
                online_trainer_train_segment_batch(g_online_trainer,
                    seg_tokens, seg_targets, NUM_STREAMS, SEG_LEN);
                perf_t2 = mach_absolute_time();
            }

            block_idx += SEG_LEN;

            ++perf_count;
            double decode_ms = (double)(perf_t1 - perf_t0) * g_tb.numer / g_tb.denom * 1e-6;
            double train_ms  = (double)(perf_t2 - perf_t1) * g_tb.numer / g_tb.denom * 1e-6;
            perf_decode_total += decode_ms;
            perf_train_total  += train_ms;
            if (perf_count % 5 == 0 && !isatty(STDERR_FILENO)) {
                fprintf(stderr, "[PERF] seg=%d decode=%.1fms train=%.1fms\n",
                        perf_count, decode_ms, train_ms);
            }

            double pct = (double)(file_pos + block_idx) / (double)stride * 100.0;
            if (pct > 100.0) pct = 100.0;
            printf("\rcompress %.1f%%", pct);
            fflush(stdout);
        }

        block_num++;
        file_pos += block_bytes;
    }

    if (perf_count > 0 && !isatty(STDERR_FILENO)) {
        fprintf(stderr, "[PERF] compress total: segs=%d decode_avg=%.1fms train_avg=%.1fms\n",
                perf_count,
                perf_decode_total / perf_count,
                perf_train_total  / perf_count);
    }

    free(seg_logits);
    free(probs);
    free(seg_tokens);
    free(seg_targets);

    // ---- Output format (self-describing) ----
    // [uint32: num_streams][uint32: original_size][uint32 × num_streams: per-stream bytes][stream 0]...
    *(uint32_t*)(output_data + 0)              = (uint32_t)ns;
    *(uint32_t*)(output_data + sizeof(uint32_t)) = (uint32_t)input_size;
    uint32_t* size_table = (uint32_t*)(output_data + 2 * sizeof(uint32_t));
    size_t current_output_offset = 2 * sizeof(uint32_t) + sizeof(uint32_t) * (size_t)ns;

    for (int s = 0; s < ns; s++) {
        int64_t s_size = put_bit_flush(&encoders[s]);
        if (current_output_offset + (size_t)s_size > output_capacity) {
            printf("Error: Output buffer too small\n");
            for (int i = s; i < ns; i++) free(stream_buffers[i]);
            free(stream_buffers); free(encoders);
            return 0;
        }
        size_table[s] = (uint32_t)s_size;
        memcpy(output_data + current_output_offset, stream_buffers[s], (size_t)s_size);
        current_output_offset += (size_t)s_size;
        free(stream_buffers[s]);
    }
    free(stream_buffers);
    free(encoders);

    printf("\rcompress %zu -> %zu bytes (%.1f%%)\n", input_size, current_output_offset,
           (double)current_output_offset * 100.0 / (double)input_size);
    return current_output_offset;
}

// Main CUDA-compatible lossless decompression function — segment-based (symmetric with compress)
size_t neural_bridge_cuda_lossless_decompress(const uint8_t* input_data, size_t input_size,
                                             uint8_t* output_data, size_t output_capacity) {
    if (!input_data || !output_data || input_size == 0) {
        printf("CUDA Lossless decompression: Invalid input parameters\n");
        return 0;
    }

    /* Reset model to the same deterministic session weights used by compress. */
    reset_model_to_session_weights(get_shared_transformer_model());


    // 1. Parse self-describing header
    //    [uint32: num_streams][uint32: original_size][uint32 × num_streams: per-stream sizes][stream data ...]
    if (input_size < 2 * sizeof(uint32_t)) {
        printf("Error: Compressed data too small to hold header\n");
        return 0;
    }
    uint32_t file_num_streams = *(const uint32_t*)(input_data + 0);
    uint32_t embedded_original_size = *(const uint32_t*)(input_data + sizeof(uint32_t));
    if (file_num_streams == 0 || file_num_streams > 1024) {
        printf("Error: Invalid num_streams in header: %u\n", file_num_streams);
        return 0;
    }
    if (embedded_original_size == 0) {
        printf("Error: Invalid original_size in header: %u\n", embedded_original_size);
        return 0;
    }
    // Use embedded original_size as the authoritative output size
    output_capacity = (size_t)embedded_original_size;
    size_t header_bytes = 2 * sizeof(uint32_t) + sizeof(uint32_t) * file_num_streams;
    if (input_size < header_bytes) {
        printf("Error: Compressed data too small for stream-sizes table\n");
        return 0;
    }
    const uint32_t* size_table = (const uint32_t*)(input_data + 2 * sizeof(uint32_t));
    size_t current_input_offset = header_bytes;

    // Validate total
    size_t total_compressed_size = current_input_offset;
    for (uint32_t i = 0; i < file_num_streams; i++) total_compressed_size += size_table[i];
    if (total_compressed_size > input_size) {
        printf("Error: Compressed data size mismatch (expected %zu, got %zu)\n",
               total_compressed_size, input_size);
        return 0;
    }

    // 2. Initialize arithmetic decoders
    GetBitState* decoders       = (GetBitState*)malloc(file_num_streams * sizeof(GetBitState));
    size_t*      stream_limits  = (size_t*)     malloc(file_num_streams * sizeof(size_t));
    size_t*      decoded_counts = (size_t*)     calloc(file_num_streams, sizeof(size_t));
    if (!decoders || !stream_limits || !decoded_counts) {
        free(decoders); free(stream_limits); free(decoded_counts);
        printf("Error: OOM allocating decoder arrays\n");
        return 0;
    }

    // stride: bytes per stream (mirrors compress side)
    const size_t stride = (output_capacity + file_num_streams - 1) / file_num_streams;

    for (uint32_t i = 0; i < file_num_streams; i++) {
        get_bit_init(&decoders[i], (uint8_t*)(input_data + current_input_offset), size_table[i]);
        current_input_offset += size_table[i];

        const size_t off = (size_t)i * stride;
        if (off >= output_capacity) {
            stream_limits[i] = 0;
        } else {
            stream_limits[i] = stride;
            if (off + stride > output_capacity)
                stream_limits[i] = output_capacity - off;
        }
    }

    // 3. MPS context + online trainer (same init as compress)
    const int vocab_size = 256;

    MPSTransformerContext* mps_ctx = get_shared_mps_ctx();
    if (!mps_ctx) {
        free(decoders); free(stream_limits); free(decoded_counts);
        return 0;
    }

    // ---- Online trainer ----
    if (!g_online_trainer) {
        g_online_trainer = online_trainer_create(MTLCreateSystemDefaultDevice(),
                                                  mps_ctx, (g_lr_override > 0.0f) ? g_lr_override : 3e-4f,
                                                  output_capacity);
    }
    if (g_online_trainer) {
        online_trainer_reset_session(g_online_trainer, false);
    }

    // ---- Segment-level working buffers ----
    float*   seg_logits   = (float*)  malloc((size_t)file_num_streams * vocab_size * sizeof(float));
    float*   probs        = (float*)  malloc(vocab_size * sizeof(float));
    int32_t* one_tok      = (int32_t*)malloc((size_t)file_num_streams * sizeof(int32_t));
    int32_t* last_decoded = (int32_t*)calloc(file_num_streams, sizeof(int32_t));
    // Accumulators for segment-level training (mirrors compress layout: [s*SEG_LEN + t])
    int32_t* dec_seg_inputs  = (int32_t*)calloc((size_t)file_num_streams * SEG_LEN, sizeof(int32_t));
    int32_t* dec_seg_targets = (int32_t*)calloc((size_t)file_num_streams * SEG_LEN, sizeof(int32_t));

    if (!seg_logits || !probs || !one_tok || !last_decoded ||
        !dec_seg_inputs || !dec_seg_targets) {
        free(seg_logits); free(probs); free(one_tok); free(last_decoded);
        free(dec_seg_inputs); free(dec_seg_targets);
        free(decoders); free(stream_limits); free(decoded_counts);
        printf("Error: OOM allocating decode working buffers\n");
        return 0;
    }

    size_t total_decoded = 0;

    if (g_tb.denom == 0) mach_timebase_info(&g_tb);
    int    perf_count        = 0;
    double perf_decode_total = 0.0;
    double perf_train_total  = 0.0;

    // Transformer-XL: reset KV cache at session start.
    mps_transformer_reset_kv_cache(mps_ctx);

    // ---- Block-level loop (mirrors compress) ----
    const size_t total_blocks = (stride + BLOCK_LEN - 1) / BLOCK_LEN;
    size_t file_pos  = 0;
    size_t block_num = 0;

    while (file_pos < stride) {
        const size_t block_bytes = ((stride - file_pos) < (size_t)BLOCK_LEN)
                                   ? (stride - file_pos) : (size_t)BLOCK_LEN;

        // Original NNCP trf_reset(): zero KV memory at each block boundary (fresh context per block).
        mps_transformer_reset_kv_cache(mps_ctx);

        // ---- Segment loop within block ----
        size_t block_idx = 0;
        while (block_idx < block_bytes) {

            // Phase M: latch pre-segment KV memory BEFORE the per-byte forward passes
            if (g_online_trainer) online_trainer_latch_kv_memory(g_online_trainer);

            uint64_t perf_t0 = mach_absolute_time();

            // ---- Position loop within segment (autoregressive: token known after each decode) ----
            for (int t = 0; t < SEG_LEN; t++) {
                const size_t abs_pos = file_pos + block_idx + (size_t)t;
                if (abs_pos >= file_pos + block_bytes) break;  // past this block

                // Check whether any stream still has bytes to decode in this segment
                bool seg_active = false;
                for (uint32_t s = 0; s < file_num_streams; s++) {
                    if (decoded_counts[s] < stream_limits[s]) { seg_active = true; break; }
                }
                if (!seg_active) goto done;

                // 1. Build input tokens: one preceding decoded byte per stream
                //    (last_decoded[s] is updated after each decoded position)
                for (uint32_t s = 0; s < file_num_streams; s++) {
                    const size_t out_off = (size_t)s * stride + abs_pos;
                    // BOS when stream s hasn't produced any byte yet
                    one_tok[s] = (abs_pos > 0 && out_off > (size_t)s * stride)
                                 ? last_decoded[s] : 0;
                }

                // 2. Single-position segment forward pass
                //    → seg_logits[file_num_streams × 1 × vocab_size]
                mps_transformer_execute_segment(mps_ctx, one_tok,
                                                (int)file_num_streams, 1, seg_logits);

                // 3. Arithmetic decode + buffer training pairs (s outer, same order as compress)
                for (uint32_t s = 0; s < file_num_streams; s++) {
                    // Always record one_tok[s] so the training input matches compress's
                    // seg_tokens, which is filled for the preceding-byte even when the
                    // target is out-of-bounds (data_off-1 valid, data_off exhausted).
                    if (g_online_trainer && t < SEG_LEN)
                        dec_seg_inputs[s * SEG_LEN + t] = one_tok[s];

                    if (decoded_counts[s] >= stream_limits[s]) continue;

                    const size_t out_off = (size_t)s * stride + abs_pos;
                    if (out_off >= output_capacity) continue;

                    float* raw = seg_logits + (size_t)s * vocab_size;

                    // Softmax with NaN/Inf guard: if any logit is non-finite, use uniform
                    bool has_nan = false;
                    for (int k = 0; k < vocab_size && !has_nan; k++)
                        if (!isfinite(raw[k])) has_nan = true;
                    if (has_nan) {
                        const float unif = 1.0f / (float)vocab_size;
                        for (int k = 0; k < vocab_size; k++) probs[k] = unif;
                    } else {
                        float max_l = raw[0];
                        for (int k = 1; k < vocab_size; k++)
                            if (raw[k] > max_l) max_l = raw[k];
                        float sum = 0.0f;
                        for (int k = 0; k < vocab_size; k++) {
                            probs[k] = expf(raw[k] - max_l);
                            sum += probs[k];
                        }
                        for (int k = 0; k < vocab_size; k++) probs[k] /= sum;
                    }

                    // Arithmetic decode
                    int sym = read_sym(&decoders[s], probs, vocab_size);
                    if (sym < 0) {
                        decoded_counts[s] = stream_limits[s];
                        continue;
                    }

                    output_data[out_off] = (uint8_t)sym;
                    decoded_counts[s]++;
                    total_decoded++;

                    // Accumulate target for segment-level training.
                    if (g_online_trainer && t < SEG_LEN)
                        dec_seg_targets[s * SEG_LEN + t] = (int32_t)sym;

                    // Advance preceding-byte for next position of this stream.
                    last_decoded[s] = (int32_t)sym;
                }
            }

            uint64_t perf_t1 = mach_absolute_time();

            // 4. One backward pass over the full segment (mirrors compress).
            if (g_online_trainer && !getenv("NNCP_NO_TRAIN"))
                online_trainer_train_segment_batch(g_online_trainer,
                    dec_seg_inputs, dec_seg_targets, (int)file_num_streams, SEG_LEN);

            uint64_t perf_t2 = mach_absolute_time();

            block_idx += SEG_LEN;

            ++perf_count;
            double decode_ms = (double)(perf_t1 - perf_t0) * g_tb.numer / g_tb.denom * 1e-6;
            double train_ms  = (double)(perf_t2 - perf_t1) * g_tb.numer / g_tb.denom * 1e-6;
            perf_decode_total += decode_ms;
            perf_train_total  += train_ms;
            if (perf_count % 5 == 0 && !isatty(STDERR_FILENO)) {
                fprintf(stderr, "[PERF] seg=%d decode=%.1fms train=%.1fms\n",
                        perf_count, decode_ms, train_ms);
            }

            double pct = (double)(file_pos + block_idx) / (double)stride * 100.0;
            if (pct > 100.0) pct = 100.0;
            fprintf(stderr, "\rdecompress %.1f%%", pct);
            fflush(stderr);
        }

        block_num++;
        file_pos += block_bytes;
    }

done:
    free(seg_logits);
    free(probs);
    free(one_tok);
    free(last_decoded);
    free(dec_seg_inputs);
    free(dec_seg_targets);
    free(decoders);
    free(stream_limits);
    free(decoded_counts);

    if (perf_count > 0 && !isatty(STDERR_FILENO)) {
        fprintf(stderr, "[PERF] decompress total: segs=%d decode_avg=%.1fms train_avg=%.1fms\n",
                perf_count,
                perf_decode_total / perf_count,
                perf_train_total  / perf_count);
    }

    fprintf(stderr, "\rdecompress %u -> %zu bytes\n", embedded_original_size, total_decoded);
    return total_decoded;
}

/* -------------------------------------------------------------------------
 * Symbol-level compress/decompress (preprocessing / vocab > 256 mode)
 * These mirror neural_bridge_cuda_lossless_compress/decompress but operate
 * on uint16_t token arrays instead of raw uint8_t bytes.
 * ---------------------------------------------------------------------- */

size_t neural_bridge_compress_symbols(
    const uint16_t *tokens, size_t n_tokens,
    uint8_t *output_data, size_t output_cap,
    int vocab_size, size_t total_input_bytes)
{
    if (!tokens || !output_data || n_tokens == 0) return 0;

    reset_model_to_session_weights(get_shared_transformer_model());

    const size_t stride = (n_tokens + NUM_STREAMS - 1) / NUM_STREAMS;

    PutBitState encoders[NUM_STREAMS];
    uint8_t *stream_bufs[NUM_STREAMS];
    const size_t est_cap = stride * 2 * sizeof(uint16_t) + 4096;
    for (int i = 0; i < NUM_STREAMS; i++) {
        stream_bufs[i] = (uint8_t *)malloc(est_cap);
        if (!stream_bufs[i]) {
            for (int j = 0; j < i; j++) free(stream_bufs[j]);
            return 0;
        }
        put_bit_init(&encoders[i], stream_bufs[i], est_cap);
    }

    MPSTransformerContext *mps_ctx = get_shared_mps_ctx();
    if (!mps_ctx) {
        for (int i = 0; i < NUM_STREAMS; i++) free(stream_bufs[i]);
        return 0;
    }
    mps_transformer_reset_kv_cache(mps_ctx);

    if (!g_online_trainer) {
        g_online_trainer = online_trainer_create(MTLCreateSystemDefaultDevice(),
                                                  mps_ctx,
                                                  (g_lr_override > 0.0f) ? g_lr_override : 3e-4f,
                                                  total_input_bytes);
    }
    if (g_online_trainer) online_trainer_reset_session(g_online_trainer, false);

    if (g_tb.denom == 0) mach_timebase_info(&g_tb);

    float   *seg_logits  = (float *)   malloc((size_t)NUM_STREAMS * SEG_LEN * vocab_size * sizeof(float));
    float   *probs       = (float *)   malloc((size_t)vocab_size * sizeof(float));
    int32_t *seg_tokens  = (int32_t *) malloc((size_t)NUM_STREAMS * SEG_LEN * sizeof(int32_t));
    int32_t *seg_targets = (int32_t *) malloc((size_t)NUM_STREAMS * SEG_LEN * sizeof(int32_t));
    if (!seg_logits || !probs || !seg_tokens || !seg_targets) {
        free(seg_logits); free(probs); free(seg_tokens); free(seg_targets);
        for (int i = 0; i < NUM_STREAMS; i++) free(stream_bufs[i]);
        return 0;
    }

    const size_t total_blocks = (stride + BLOCK_LEN - 1) / BLOCK_LEN;
    size_t file_pos = 0;

    while (file_pos < stride) {
        const size_t block_tokens = ((stride - file_pos) < (size_t)BLOCK_LEN)
                                    ? (stride - file_pos) : (size_t)BLOCK_LEN;
        mps_transformer_reset_kv_cache(mps_ctx);

        size_t block_idx = 0;
        while (block_idx < block_tokens) {
            memset(seg_tokens, 0, (size_t)NUM_STREAMS * SEG_LEN * sizeof(int32_t));
            for (int s = 0; s < NUM_STREAMS; s++) {
                for (int t = 0; t < SEG_LEN; t++) {
                    const size_t abs_pos  = file_pos + block_idx + (size_t)t;
                    const size_t data_off = (size_t)s * stride + abs_pos;
                    if (abs_pos > 0 && data_off > 0 && (data_off - 1) < n_tokens)
                        seg_tokens[s * SEG_LEN + t] = (int32_t)tokens[data_off - 1];
                }
            }
            if (g_online_trainer) online_trainer_latch_kv_memory(g_online_trainer);

            mps_transformer_execute_segment(mps_ctx, seg_tokens, NUM_STREAMS, SEG_LEN, seg_logits);

            for (int t = 0; t < SEG_LEN; t++) {
                const size_t abs_pos = file_pos + block_idx + (size_t)t;
                if (abs_pos >= file_pos + block_tokens) break;
                for (int s = 0; s < NUM_STREAMS; s++) {
                    const size_t data_off = (size_t)s * stride + abs_pos;
                    if (data_off >= n_tokens) continue;

                    float *raw = seg_logits + ((size_t)(s * SEG_LEN + t)) * vocab_size;
                    bool has_nan = false;
                    for (int k = 0; k < vocab_size && !has_nan; k++)
                        if (!isfinite(raw[k])) has_nan = true;
                    if (has_nan) {
                        const float unif = 1.0f / (float)vocab_size;
                        for (int k = 0; k < vocab_size; k++) probs[k] = unif;
                    } else {
                        float mx = raw[0];
                        for (int k = 1; k < vocab_size; k++) if (raw[k] > mx) mx = raw[k];
                        float sum = 0.0f;
                        for (int k = 0; k < vocab_size; k++) { probs[k] = expf(raw[k] - mx); sum += probs[k]; }
                        for (int k = 0; k < vocab_size; k++) probs[k] /= sum;
                    }

                    const uint16_t tok = tokens[data_off];
                    write_sym(&encoders[s], probs, vocab_size, (int)tok);
                    if (g_online_trainer)
                        seg_targets[s * SEG_LEN + t] = (int32_t)tok;
                }
            }

            if (g_online_trainer && !getenv("NNCP_NO_TRAIN")) {
                online_trainer_train_segment_batch(g_online_trainer,
                    seg_tokens, seg_targets, NUM_STREAMS, SEG_LEN);
            }

            block_idx += SEG_LEN;

            double pct = (double)(file_pos + block_idx) / (double)stride * 100.0;
            if (pct > 100.0) pct = 100.0;
            printf("\rcompress %.1f%%", pct);
            fflush(stdout);
        }
        file_pos += block_tokens;
    }
    (void)total_blocks;

    free(seg_logits); free(probs); free(seg_tokens); free(seg_targets);

    *(uint32_t *)(output_data + 0)              = (uint32_t)NUM_STREAMS;
    *(uint32_t *)(output_data + sizeof(uint32_t)) = (uint32_t)n_tokens;
    uint32_t *size_table = (uint32_t *)(output_data + 2 * sizeof(uint32_t));
    size_t cur_out = 2 * sizeof(uint32_t) + sizeof(uint32_t) * NUM_STREAMS;

    for (int s = 0; s < NUM_STREAMS; s++) {
        int64_t s_size = put_bit_flush(&encoders[s]);
        if (cur_out + (size_t)s_size > output_cap) {
            printf("Error: output buffer too small\n");
            for (int i = s; i < NUM_STREAMS; i++) free(stream_bufs[i]);
            return 0;
        }
        size_table[s] = (uint32_t)s_size;
        memcpy(output_data + cur_out, stream_bufs[s], (size_t)s_size);
        cur_out += (size_t)s_size;
        free(stream_bufs[s]);
    }

    printf("\rcompress %zu tokens -> %zu bytes (%.1f%%)\n",
           n_tokens, cur_out, (double)cur_out / (double)n_tokens * 50.0);
    return cur_out;
}

size_t neural_bridge_decompress_symbols(
    const uint8_t *input_data, size_t input_size,
    uint16_t *tokens_out, size_t max_tokens,
    int vocab_size)
{
    if (!input_data || !tokens_out || input_size == 0) return 0;

    reset_model_to_session_weights(get_shared_transformer_model());

    if (input_size < 2 * sizeof(uint32_t)) return 0;
    uint32_t file_num_streams  = *(const uint32_t *)(input_data + 0);
    uint32_t embedded_n_tokens = *(const uint32_t *)(input_data + sizeof(uint32_t));
    if (file_num_streams == 0 || file_num_streams > 1024) return 0;
    if (embedded_n_tokens == 0 || embedded_n_tokens > (uint32_t)max_tokens) return 0;

    size_t n_tokens = embedded_n_tokens;
    size_t header_bytes = 2 * sizeof(uint32_t) + sizeof(uint32_t) * file_num_streams;
    if (input_size < header_bytes) return 0;
    const uint32_t *size_table = (const uint32_t *)(input_data + 2 * sizeof(uint32_t));
    size_t cur_in = header_bytes;

    GetBitState *decoders      = (GetBitState *)malloc(file_num_streams * sizeof(GetBitState));
    size_t *stream_limits      = (size_t *)malloc(file_num_streams * sizeof(size_t));
    size_t *decoded_counts     = (size_t *)calloc(file_num_streams, sizeof(size_t));
    if (!decoders || !stream_limits || !decoded_counts) {
        free(decoders); free(stream_limits); free(decoded_counts); return 0;
    }

    const size_t stride = (n_tokens + file_num_streams - 1) / file_num_streams;
    for (uint32_t i = 0; i < file_num_streams; i++) {
        get_bit_init(&decoders[i], (uint8_t *)(input_data + cur_in), size_table[i]);
        cur_in += size_table[i];
        const size_t off = (size_t)i * stride;
        stream_limits[i] = (off >= n_tokens) ? 0
                         : ((off + stride > n_tokens) ? n_tokens - off : stride);
    }

    MPSTransformerContext *mps_ctx = get_shared_mps_ctx();
    if (!mps_ctx) { free(decoders); free(stream_limits); free(decoded_counts); return 0; }

    if (!g_online_trainer) {
        g_online_trainer = online_trainer_create(MTLCreateSystemDefaultDevice(),
                                                  mps_ctx,
                                                  (g_lr_override > 0.0f) ? g_lr_override : 3e-4f,
                                                  n_tokens);
    }
    if (g_online_trainer) online_trainer_reset_session(g_online_trainer, false);
    mps_transformer_reset_kv_cache(mps_ctx);

    /* Autoregressive decoding: one forward pass per TOKEN POSITION (seq_len=1).
     * Mirrors neural_bridge_cuda_lossless_decompress exactly, using uint16_t output. */
    float   *seg_logits      = (float *)  malloc((size_t)file_num_streams * vocab_size * sizeof(float));
    float   *probs           = (float *)  malloc((size_t)vocab_size * sizeof(float));
    int32_t *one_tok         = (int32_t *)calloc((size_t)file_num_streams, sizeof(int32_t));
    int32_t *last_decoded    = (int32_t *)calloc((size_t)file_num_streams, sizeof(int32_t));
    int32_t *dec_seg_inputs  = (int32_t *)calloc((size_t)file_num_streams * SEG_LEN, sizeof(int32_t));
    int32_t *dec_seg_targets = (int32_t *)calloc((size_t)file_num_streams * SEG_LEN, sizeof(int32_t));
    if (!seg_logits || !probs || !one_tok || !last_decoded ||
        !dec_seg_inputs || !dec_seg_targets) {
        free(seg_logits); free(probs); free(one_tok); free(last_decoded);
        free(dec_seg_inputs); free(dec_seg_targets);
        free(decoders); free(stream_limits); free(decoded_counts); return 0;
    }

    size_t total_decoded = 0;
    size_t file_pos      = 0;

    while (file_pos < stride) {
        const size_t block_tokens = ((stride - file_pos) < (size_t)BLOCK_LEN)
                                    ? (stride - file_pos) : (size_t)BLOCK_LEN;
        mps_transformer_reset_kv_cache(mps_ctx);

        size_t block_idx = 0;
        while (block_idx < block_tokens) {
            if (g_online_trainer) online_trainer_latch_kv_memory(g_online_trainer);

            /* Token-level loop within segment (autoregressive) */
            for (int t = 0; t < SEG_LEN; t++) {
                const size_t abs_pos = file_pos + block_idx + (size_t)t;
                if (abs_pos >= file_pos + block_tokens) break;

                bool seg_active = false;
                for (uint32_t s = 0; s < file_num_streams; s++)
                    if (decoded_counts[s] < stream_limits[s]) { seg_active = true; break; }
                if (!seg_active) goto sym_done;

                /* Input: last decoded token per stream (or BOS=0 at stream start) */
                for (uint32_t s = 0; s < file_num_streams; s++) {
                    const size_t out_off = (size_t)s * stride + abs_pos;
                    one_tok[s] = (abs_pos > 0 && out_off > (size_t)s * stride)
                                 ? last_decoded[s] : 0;
                }

                /* seq_len=1 forward pass */
                mps_transformer_execute_segment(mps_ctx, one_tok,
                                                (int)file_num_streams, 1, seg_logits);

                for (uint32_t s = 0; s < file_num_streams; s++) {
                    if (g_online_trainer && t < SEG_LEN)
                        dec_seg_inputs[s * SEG_LEN + t] = one_tok[s];

                    if (decoded_counts[s] >= stream_limits[s]) continue;

                    const size_t out_off = (size_t)s * stride + abs_pos;
                    if (out_off >= n_tokens) continue;

                    float *raw = seg_logits + (size_t)s * vocab_size;
                    bool has_nan = false;
                    for (int k = 0; k < vocab_size && !has_nan; k++)
                        if (!isfinite(raw[k])) has_nan = true;
                    if (has_nan) {
                        const float unif = 1.0f / (float)vocab_size;
                        for (int k = 0; k < vocab_size; k++) probs[k] = unif;
                    } else {
                        float mx = raw[0];
                        for (int k = 1; k < vocab_size; k++) if (raw[k] > mx) mx = raw[k];
                        float sum = 0.0f;
                        for (int k = 0; k < vocab_size; k++) { probs[k] = expf(raw[k] - mx); sum += probs[k]; }
                        for (int k = 0; k < vocab_size; k++) probs[k] /= sum;
                    }

                    int sym = read_sym(&decoders[s], probs, vocab_size);
                    if (sym < 0) { decoded_counts[s] = stream_limits[s]; continue; }
                    if (sym >= vocab_size) sym = 0;

                    tokens_out[out_off] = (uint16_t)sym;
                    decoded_counts[s]++;
                    total_decoded++;

                    if (g_online_trainer && t < SEG_LEN)
                        dec_seg_targets[s * SEG_LEN + t] = (int32_t)sym;

                    last_decoded[s] = (int32_t)sym;
                }
            }

            /* Segment-level training (mirrors compress) */
            if (g_online_trainer && !getenv("NNCP_NO_TRAIN"))
                online_trainer_train_segment_batch(g_online_trainer,
                    dec_seg_inputs, dec_seg_targets, (int)file_num_streams, SEG_LEN);

            block_idx += SEG_LEN;

            double pct = (double)(file_pos + block_idx) / (double)stride * 100.0;
            if (pct > 100.0) pct = 100.0;
            fprintf(stderr, "\rdecompress %.1f%%", pct);
            fflush(stderr);
        }
        file_pos += block_tokens;
    }

sym_done:
    free(seg_logits); free(probs); free(one_tok); free(last_decoded);
    free(dec_seg_inputs); free(dec_seg_targets);
    free(decoders); free(stream_limits); free(decoded_counts);

    fprintf(stderr, "\rdecompress %zu tokens\n", total_decoded);
    return total_decoded;
}

// Metal Transformer Model Implementation

// Helper function to create Metal compute pipeline state
static id<MTLComputePipelineState> create_compute_pipeline(id<MTLDevice> device, const char* shader_source, const char* function_name) {
    NSString* source = [NSString stringWithUTF8String:shader_source];
    NSError* error = nil;
    
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        printf("Failed to create Metal library: %s\n", error.localizedDescription.UTF8String);
        return nil;
    }
    
    NSString* funcName = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> function = [library newFunctionWithName:funcName];
    if (!function) {
        printf("Failed to find Metal function: %s\n", function_name);
        return nil;
    }
    
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        printf("Failed to create compute pipeline: %s\n", error.localizedDescription.UTF8String);
        return nil;
    }
    
    return pipeline;
}

// Initialize Metal Transformer model weights.
// Matches original NNCP default profile: U(-0.0625, 0.0625) for all weight matrices
// (init_range=1.0 / sqrt(d_model=256) = 0.0625). Biases stay zero. LayerNorm: gamma=1, beta=0.
static void initialize_transformer_weights(MetalTransformerModel* model) {
    if (!model || !model->device) return;

    const uint32_t L   = model->num_layers;
    const uint32_t H   = model->hidden_size;
    const uint32_t V   = model->vocab_size;
    const uint32_t FFS = model->feed_forward_size;
    const float    INIT_SCALE = 0.0625f; /* 1.0 / sqrt(256) */

    // Embedding
    nn_weights_init_uniform(
        (float*)model->embedding_weights.contents,
        (size_t)V * H, INIT_SCALE, 42u);

    // Positional embeddings: sinusoidal (deterministic by formula, no randomness)
    float* pos_data = (float*)model->position_embeddings.contents;
    for (uint32_t pos = 0; pos < model->context_length; pos++) {
        for (uint32_t dim = 0; dim < H; dim++) {
            float angle = pos / powf(10000.0f, 2.0f * (dim / 2) / H);
            pos_data[pos * H + dim] = (dim % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }

    // Attention Q / K / V / Out  (seeds 43-46)
    nn_weights_init_uniform(
        (float*)model->attention_weights_q.contents,
        (size_t)L * H * H, INIT_SCALE, 43u);
    nn_weights_init_uniform(
        (float*)model->attention_weights_k.contents,
        (size_t)L * H * H, INIT_SCALE, 44u);
    nn_weights_init_uniform(
        (float*)model->attention_weights_v.contents,
        (size_t)L * H * H, INIT_SCALE, 45u);
    nn_weights_init_uniform(
        (float*)model->attention_output_weights.contents,
        (size_t)L * H * H, INIT_SCALE, 46u);

    // FFN weights (seeds 47-48); ffn_weights_1 is 2x for GeGLU (value + gate)
    nn_weights_init_uniform(
        (float*)model->ffn_weights_1.contents,
        (size_t)L * H * FFS * 2, INIT_SCALE, 47u);
    const float ff2_scale = INIT_SCALE * sqrtf((float)H / (float)FFS);  // ≈ 0.0442
    nn_weights_init_uniform(
        (float*)model->ffn_weights_2.contents,
        (size_t)L * FFS * H, ff2_scale, 48u);

    // LayerNorm: gamma=1, beta=0
    nn_weights_init_layer_norm(
        (float*)model->layer_norm_weights.contents, H, L);

    // LN_FINAL: gamma=1, beta=0
    if (model->ln_final) {
        float* p = (float*)[model->ln_final contents];
        for (size_t i = 0; i < H; i++) { p[i] = 1.0f; p[H + i] = 0.0f; }
    }

    // Output projection (seed 49)
    nn_weights_init_uniform(
        (float*)model->output_projection.contents,
        (size_t)H * V, INIT_SCALE, 49u);

    // Relative PE: w_rel_r ~ U(±0.0625) seed 50; b_rel_r = 0
    {
        const uint32_t NH_ = model->num_attention_heads;
        const uint32_t HD_ = H / NH_;
        const size_t   DP_ = NNCP_D_POS();
        nn_weights_init_uniform(
            (float*)model->rel_r.contents,
            (size_t)NH_ * HD_ * DP_, INIT_SCALE, 50u);  /* [NH, HD, D_POS] */
        memset([model->b_rel_r contents], 0, (size_t)NH_ * DP_ * sizeof(float)); /* [NH, D_POS] */
    }

}

// Create and initialize Metal Transformer model
static MetalTransformerModel* create_transformer_model(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("Error: Metal device not available\n");
        return NULL;
    }
    
    MetalTransformerModel* model = (MetalTransformerModel*)calloc(1, sizeof(MetalTransformerModel));
    if (!model) {
        printf("Error: Failed to allocate Transformer model memory\n");
        return NULL;
    }
    
    // Initialize model parameters from runtime profile
    model->device = device;
    model->command_queue = [device newCommandQueue];
    model->context_length = 1024 * 64;
    model->vocab_size           = (g_vocab_size_override > 0) ? (uint32_t)g_vocab_size_override : 256;
    model->hidden_size          = (uint32_t)g_nncp_profile.h;
    model->num_attention_heads  = (uint32_t)g_nncp_profile.nh;
    model->num_layers           = (uint32_t)g_nncp_profile.l;
    model->feed_forward_size    = (uint32_t)g_nncp_profile.f;
    model->max_sequence_length  = (uint32_t)g_nncp_profile.mem_len;
    
    // Allocate weight buffers
    model->embedding_weights = [device newBufferWithLength:model->vocab_size * model->hidden_size * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    model->position_embeddings = [device newBufferWithLength:model->context_length * model->hidden_size * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    model->attention_weights_q = [device newBufferWithLength:model->num_layers * model->hidden_size * model->hidden_size * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    model->attention_weights_k = [device newBufferWithLength:model->num_layers * model->hidden_size * model->hidden_size * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    model->attention_weights_v = [device newBufferWithLength:model->num_layers * model->hidden_size * model->hidden_size * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    model->attention_output_weights = [device newBufferWithLength:model->num_layers * model->hidden_size * model->hidden_size * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
    // GeGLU FFN: FFN1 projects hidden → 2*feed_forward_size (first half=value, second half=gate)
    model->ffn_weights_1 = [device newBufferWithLength:model->num_layers * model->hidden_size * model->feed_forward_size * 2 * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    model->ffn_weights_2 = [device newBufferWithLength:model->num_layers * model->feed_forward_size * model->hidden_size * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    model->layer_norm_weights = [device newBufferWithLength:model->num_layers * 4 * model->hidden_size * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    model->output_projection = [device newBufferWithLength:model->hidden_size * model->vocab_size * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

    // Allocate bias buffers (zero-initialized: use_bias=1, learned from scratch)
    {
        MTLResourceOptions opts = MTLResourceStorageModeShared;
        size_t L = model->num_layers, H = model->hidden_size, F = model->feed_forward_size, V = model->vocab_size;
        model->bias_k    = [device newBufferWithLength:L * H * sizeof(float) options:opts];
        model->bias_v    = [device newBufferWithLength:L * H * sizeof(float) options:opts];
        model->bias_o    = [device newBufferWithLength:L * H * sizeof(float) options:opts];
        model->bias_ffn1 = [device newBufferWithLength:L * 2 * F * sizeof(float) options:opts];
        model->bias_ffn2 = [device newBufferWithLength:L * H * sizeof(float) options:opts];
        model->bias_out  = [device newBufferWithLength:V     * sizeof(float) options:opts];
        const size_t NH_ = model->num_attention_heads;
        const size_t HD_ = H / NH_;
        const size_t DP_ = NNCP_D_POS();  /* MEM_LEN*2 or MEM_LEN+SEG_LEN, =64 for default */
        model->rel_r   = [device newBufferWithLength:NH_ * HD_ * DP_ * sizeof(float) options:opts]; /* [NH,HD,D_POS] */
        model->b_rel_r = [device newBufferWithLength:NH_ * DP_       * sizeof(float) options:opts]; /* [NH,D_POS] */
        if (model->rel_r)   memset([model->rel_r   contents], 0, NH_ * HD_ * DP_ * sizeof(float));
        if (model->b_rel_r) memset([model->b_rel_r contents], 0, NH_ * DP_       * sizeof(float));
        model->ln_final = [device newBufferWithLength:2 * H * sizeof(float) options:opts]; /* [2,H] */
        if (model->ln_final) {
            float* p = (float*)[model->ln_final contents];
            for (size_t i = 0; i < H; i++) { p[i] = 1.0f; p[H + i] = 0.0f; } /* gamma=1, beta=0 */
        }
        if (model->bias_k)    memset([model->bias_k    contents], 0, L * H * sizeof(float));
        if (model->bias_v)    memset([model->bias_v    contents], 0, L * H * sizeof(float));
        if (model->bias_o)    memset([model->bias_o    contents], 0, L * H * sizeof(float));
        if (model->bias_ffn1) memset([model->bias_ffn1 contents], 0, L * 2 * F * sizeof(float));
        if (model->bias_ffn2) memset([model->bias_ffn2 contents], 0, L * H * sizeof(float));
        if (model->bias_out)  memset([model->bias_out  contents], 0, V     * sizeof(float));
    }

    // Allocate computation buffers
    model->context_buffer = [device newBufferWithLength:model->context_length * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
    model->embedded_buffer = [device newBufferWithLength:model->context_length * model->hidden_size * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    model->attention_buffer = [device newBufferWithLength:model->context_length * model->hidden_size * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    model->ffn_buffer = [device newBufferWithLength:model->context_length * model->hidden_size * sizeof(float)
                                             options:MTLResourceStorageModeShared];
    model->logits_buffer = [device newBufferWithLength:model->vocab_size * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    
    // Create compute pipelines
    model->embedding_pipeline = create_compute_pipeline(device, transformer_embedding_shader, "transformer_embedding");
    model->attention_pipeline = create_compute_pipeline(device, transformer_attention_shader, "transformer_self_attention");
    model->ffn_pipeline = create_compute_pipeline(device, transformer_ffn_shader, "transformer_feed_forward");
    model->output_pipeline = create_compute_pipeline(device, transformer_output_shader, "transformer_output_projection");
    
    // Verify all allocations succeeded
    if (!model->embedding_weights || !model->position_embeddings ||
        !model->attention_weights_q || !model->attention_weights_k || !model->attention_weights_v ||
        !model->attention_output_weights || !model->ffn_weights_1 || !model->ffn_weights_2 ||
        !model->layer_norm_weights || !model->output_projection ||
        !model->bias_k || !model->bias_v || !model->bias_o ||
        !model->bias_ffn1 || !model->bias_ffn2 || !model->bias_out ||
        !model->rel_r || !model->b_rel_r || !model->ln_final ||
        !model->context_buffer || !model->embedded_buffer || !model->attention_buffer ||
        !model->ffn_buffer || !model->logits_buffer ||
        !model->embedding_pipeline || !model->attention_pipeline || !model->ffn_pipeline || !model->output_pipeline) {
        printf("Error: Failed to allocate Metal Transformer resources\n");
        free(model);
        return NULL;
    }
    
    // Always use deterministic Xavier init (LCG seed) — no pre-trained weights
    initialize_transformer_weights(model);

    model->is_initialized = true;
    model->weights_loaded = true;
    
    
    return model;
}

// Real implementations for Metal Transformer functions
MetalTransformerModel* get_shared_transformer_model(void) {
    static MetalTransformerModel* shared_model = NULL;
    static dispatch_once_t once_token;
    
    dispatch_once(&once_token, ^{
        shared_model = create_transformer_model();
        if (!shared_model) {
            fprintf(stderr, "Error: Failed to initialize Transformer model\n");
        }
    });
    
    return shared_model;
}

bool metal_transformer_prediction(MetalTransformerModel* model,
                                  id<MTLBuffer> hidden_buffer,
                                  id<MTLBuffer> output_logits,
                                  size_t seq_len,
                                  id<MTLCommandBuffer> commandBuffer) {
    if (!model || !model->is_initialized) {
        printf("[Metal Transformer] Error: Model not initialized\n");
        return false;
    }
    
    // Allow larger batches if we are doing multi-stream
    // But we need to be careful about positional embeddings
    // For now, assume seq_len <= 64, or if > 64, it's a flat batch?
    // If flow_optimizer loops, seq_len will be <= 64.
    if (seq_len > model->context_length) {
        // printf("[Metal Transformer] Error: Sequence length %zu exceeds context length %u\n", seq_len, model->context_length);
        // return false;
    }
    
    @autoreleasepool {
        // Create compute encoder
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Step 1: Embedding layer
        [encoder setComputePipelineState:model->embedding_pipeline];
        [encoder setBuffer:model->context_buffer offset:0 atIndex:0];  // input tokens
        [encoder setBuffer:model->embedding_weights offset:0 atIndex:1];
        [encoder setBuffer:model->position_embeddings offset:0 atIndex:2];
        [encoder setBuffer:model->embedded_buffer offset:0 atIndex:3];
        
        MTLSize embeddingThreads = MTLSizeMake(seq_len, 1, 1);
        MTLSize embeddingThreadgroupSize = MTLSizeMake(MIN(seq_len, 64), 1, 1);
        [encoder dispatchThreads:embeddingThreads threadsPerThreadgroup:embeddingThreadgroupSize];
        
        // Memory barrier
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        
        // Step 2: Transformer layers (simplified - single layer for now)
        // Self-attention
        [encoder setComputePipelineState:model->attention_pipeline];
        [encoder setBuffer:model->embedded_buffer offset:0 atIndex:0];  // input
        [encoder setBuffer:model->attention_weights_q offset:0 atIndex:1];
        [encoder setBuffer:model->attention_weights_k offset:0 atIndex:2];
        [encoder setBuffer:model->attention_weights_v offset:0 atIndex:3];
        [encoder setBuffer:model->attention_output_weights offset:0 atIndex:4];
        [encoder setBuffer:model->attention_buffer offset:0 atIndex:5];  // output
        
        MTLSize attentionThreads = MTLSizeMake(seq_len, model->num_attention_heads, 1);
        MTLSize attentionThreadgroupSize = MTLSizeMake(MIN(seq_len, 8), MIN(model->num_attention_heads, 8), 1);
        [encoder dispatchThreads:attentionThreads threadsPerThreadgroup:attentionThreadgroupSize];
        
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        
        // Feed-forward network
        [encoder setComputePipelineState:model->ffn_pipeline];
        [encoder setBuffer:model->attention_buffer offset:0 atIndex:0];  // input
        [encoder setBuffer:model->ffn_weights_1 offset:0 atIndex:1];
        [encoder setBuffer:model->ffn_weights_2 offset:0 atIndex:2];
        [encoder setBuffer:model->ffn_buffer offset:0 atIndex:3];  // output
        
        MTLSize ffnThreads = MTLSizeMake(seq_len, 1, 1);
        MTLSize ffnThreadgroupSize = MTLSizeMake(MIN(seq_len, 64), 1, 1);
        [encoder dispatchThreads:ffnThreads threadsPerThreadgroup:ffnThreadgroupSize];
        
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        
        // Step 3: Output projection (Metal)
        [encoder setComputePipelineState:model->output_pipeline];
        [encoder setBuffer:model->ffn_buffer offset:0 atIndex:0];  // hidden states
        [encoder setBuffer:model->output_projection offset:0 atIndex:1]; // weights
        [encoder setBuffer:output_logits offset:0 atIndex:2]; // logits
        
        MTLSize outputThreads = MTLSizeMake(seq_len, model->vocab_size, 1);
        MTLSize outputThreadgroupSize = MTLSizeMake(MIN(seq_len, 8), MIN(model->vocab_size, 32), 1);
        [encoder dispatchThreads:outputThreads threadsPerThreadgroup:outputThreadgroupSize];
        
        [encoder endEncoding];
        
        // Commit and wait for GPU completion (CUDA-compatible synchronous processing)
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // CUDA-compatible output projection and softmax (complete implementation)
        // float* ffn_output = (float*)model->ffn_buffer.contents; // Not needed on CPU anymore
        // float* output_weights = (float*)model->output_projection.contents; // Not needed on CPU anymore
        float* logits = (float*)output_logits.contents;
        
        // Use last position output for next token prediction
        // If seq_len is small (e.g. 1-64), we usually want the last prediction?
        // But for batch processing, we might want ALL predictions?
        // In compression/decompression loop, we use `logits[pos]` for all pos.
        // So we need to compute logits for ALL positions, not just last.
        
        for (size_t pos = 0; pos < seq_len; pos++) {
            // const float* hidden_state = &ffn_output[pos * model->hidden_size]; // Moved to GPU
            float* logits_pos = &logits[pos * model->vocab_size];
            
            // Full matrix multiplication - Moved to GPU
            /*
            for (uint32_t vocab = 0; vocab < model->vocab_size; vocab++) {
                float logit = 0.0f;
                for (uint32_t h = 0; h < model->hidden_size; h++) {
                    logit += hidden_state[h] * output_weights[h * model->vocab_size + vocab];
                }
                logits_pos[vocab] = logit;
            }
            */
            
            // Softmax
            float max_logit = logits_pos[0];
            for (uint32_t vocab = 1; vocab < model->vocab_size; vocab++) {
                if (logits_pos[vocab] > max_logit) max_logit = logits_pos[vocab];
            }
            
            float sum_exp = 0.0f;
            for (uint32_t vocab = 0; vocab < model->vocab_size; vocab++) {
                logits_pos[vocab] = expf(logits_pos[vocab] - max_logit);
                sum_exp += logits_pos[vocab];
            }
            
            if (sum_exp > 0.0f) {
                for (uint32_t vocab = 0; vocab < model->vocab_size; vocab++) {
                    logits_pos[vocab] /= sum_exp;
                }
            }
        }
        
        return true;
    }
}


#ifdef __cplusplus
}
#endif