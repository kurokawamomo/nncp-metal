/*
 * Original NNCP Algorithm Port to Metal
 * 
 * Based on original nncp.c by Fabrice Bellard
 * Ported for Apple Silicon Metal acceleration
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef USE_METAL
#include "metal_context.h"
#include "neural_engine.h"
#include "memory_manager.h"
#endif

// Original NNCP model parameters (from original implementation)
typedef struct {
    int n_symbols;      // 256 for byte-level
    int batch_size;     // n_streams (original: variable)
    int seg_len;        // n_states (original: variable)
    int n_layers;       // transformer layers
    int d_model;        // model dimension
    int n_heads;        // attention heads
    bool use_metal;     // Metal acceleration flag
} NNCPOriginalParams;

// Original NNCP model state
typedef struct {
    NNCPOriginalParams params;
    void* model_data;   // Metal buffers/tensors
    int train_step;
    float lr;
    bool initialized;
#ifdef USE_METAL
    // Neural Engine components
    MetalContext* metal_context;
    NEContext* ne_context;
    NEModel* ne_model;
#endif
} NNCPOriginalState;

// Original NNCP Arithmetic Coding from arith.h
typedef void PutBitWriteFunc(void *opaque, const uint8_t *buf, size_t buf_size);
typedef size_t GetBitReadFunc(void *opaque, uint8_t *buf, size_t buf_size);

typedef struct {
    uint32_t range;
    uint32_t low;
    uint8_t current_byte;
    uint32_t n_bytes;
    uint8_t *buf;
    size_t buf_size;
    size_t idx; /* current position in bytes */
    PutBitWriteFunc *write_func;
    void *opaque;
    uint64_t byte_count;
} PutBitState;

typedef struct {
    uint8_t *buf;
    int buf_len;
    int buf_size;
    int idx;
    uint32_t low;
    uint32_t range;
    bool eof_reached;
    GetBitReadFunc *read_func;
    void *opaque;
    uint64_t byte_count;
} GetBitState;

// Forward declarations
static bool nncp_original_init(NNCPOriginalState* state, const NNCPOriginalParams* params);
static void nncp_original_cleanup(NNCPOriginalState* state);
static bool nncp_original_model_eval(NNCPOriginalState* state, 
                                     const int32_t* input, int seq_len,
                                     float* output_probs);
static void nncp_original_model_update(NNCPOriginalState* state,
                                       const int32_t* expected_output, int seq_len);

// Original NNCP Arithmetic Coding Functions
static void put_bit_init(PutBitState *s, uint8_t *buf, int buf_size, PutBitWriteFunc *write_func, void *opaque);
static void put_bit(PutBitState *s, int prob0, int bit);
static int64_t put_bit_flush(PutBitState *s);
static void get_bit_init(GetBitState *s, uint8_t *buf, size_t buf_size, GetBitReadFunc *read_func, void *opaque);
static int get_bit(GetBitState *s, int prob0);
static void write_sym(PutBitState *pb, const float *prob_table, int n_symb, int sym);
static int read_sym(GetBitState *gb, const float *prob_table, int n_symb);

/*
 * Optimize parameters based on input size to minimize overhead for small files
 * Smaller seg_len reduces overhead for very small files (< 100 bytes)
 */
static void nncp_original_optimize_params_for_size(NNCPOriginalParams* params, size_t input_size)
{
    if (!params) return;
    
    // Default values
    params->batch_size = 1;  // Always 1 for simplicity
    
    // Adjust seg_len based on input size to minimize overhead
    if (input_size < 100) {
        // Very small files: minimize block overhead
        params->seg_len = 1;
    } else if (input_size < 500) {
        // Small files: small block size
        params->seg_len = 2;
    } else if (input_size < 2048) {
        // Medium files: medium block size
        params->seg_len = 4;
    } else {
        // Larger files: standard block size
        params->seg_len = 8;
    }
}

/*
 * Main compression function implementing original NNCP algorithm
 * Based on process_block() from original nncp.c
 */
bool nncp_original_compress(const uint8_t* input_data, size_t input_size,
                           uint8_t* output_data, size_t output_capacity,
                           size_t* compressed_size, bool verbose)
{
    if (!input_data || input_size == 0 || !output_data || output_capacity == 0) {
        return false;
    }

    // Initialize original NNCP parameters
    NNCPOriginalParams params = {
        .n_symbols = 256,           // byte-level compression
        .batch_size = 1,            // n_streams (default, will be optimized)
        .seg_len = 8,               // n_states (default, will be optimized)
        .n_layers = 2,              // reduced for testing
        .d_model = 128,             // reduced for testing
        .n_heads = 4,               // reduced for testing
        .use_metal = false          // Disable Metal for CPU fallback
    };
    
    // Optimize parameters based on input size
    nncp_original_optimize_params_for_size(&params, input_size);

    NNCPOriginalState state = {0};
    if (!nncp_original_init(&state, &params)) {
        if (verbose) printf("Failed to initialize NNCP original state\n");
        return false;
    }

    PutBitState encoder;
    put_bit_init(&encoder, output_data, output_capacity, NULL, NULL);

    // Process data in blocks (similar to original process_block)
    size_t block_size = params.batch_size * params.seg_len;
    size_t num_blocks = (input_size + block_size - 1) / block_size;
    
    if (verbose) {
        printf("[NNCP Original] Processing %zu bytes in %zu blocks\n", input_size, num_blocks);
        printf("[NNCP Original] Block size: %zu (batch_size=%d, seg_len=%d)\n", 
               block_size, params.batch_size, params.seg_len);
    }

    for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        size_t block_start = block_idx * block_size;
        size_t current_block_size = (block_start + block_size > input_size) ? 
                                   (input_size - block_start) : block_size;

        // Process each sequence position
        for (int pos = 0; pos < params.seg_len; pos++) {
            // Prepare input for model (context from previous tokens)
            int32_t model_input[params.batch_size];
            for (int stream = 0; stream < params.batch_size; stream++) {
                size_t input_pos = block_start + stream * params.seg_len + pos;
                if (pos == 0 || input_pos == 0) {
                    model_input[stream] = 0; // start token
                } else {
                    model_input[stream] = (input_pos - 1 < input_size) ? 
                                         input_data[input_pos - 1] : 0;
                }
            }

            // Model evaluation (this would be Metal-accelerated)
            float output_probs[params.batch_size * params.n_symbols];
            if (!nncp_original_model_eval(&state, model_input, params.batch_size, output_probs)) {
                if (verbose) printf("Model eval failed at block %zu, pos %d\n", block_idx, pos);
                nncp_original_cleanup(&state);
                return false;
            }

            // Encode symbols using arithmetic coding
            int32_t expected_symbols[params.batch_size];
            for (int stream = 0; stream < params.batch_size; stream++) {
                size_t input_pos = block_start + stream * params.seg_len + pos;
                int symbol = (input_pos < input_size) ? input_data[input_pos] : 0;
                
                if (input_pos < input_size) {
                    write_sym(&encoder, &output_probs[stream * params.n_symbols], 
                             params.n_symbols, symbol);
                }
                expected_symbols[stream] = symbol;
            }

            // Model update (training step)
            nncp_original_model_update(&state, expected_symbols, params.batch_size);
        }

        if (verbose && (block_idx % 10 == 0)) {
            printf("\r[NNCP Original] Processed block %zu/%zu", block_idx + 1, num_blocks);
            fflush(stdout);
        }
    }
    
    // Ensure new line after progress display
    if (verbose && num_blocks > 0) {
        printf("\n");
    }

    int64_t flushed_size = put_bit_flush(&encoder);
    *compressed_size = encoder.idx;
    
    nncp_original_cleanup(&state);

    if (verbose) {
        printf("\n"); // New line after progress updates
        float ratio = (float)*compressed_size / input_size * 100.0f;
        printf("[NNCP Original] Compression: %zu -> %zu bytes (%.1f%%)\n", 
               input_size, *compressed_size, ratio);
    }

    return true;
}

/*
 * Initialize NNCP original state
 */
static bool nncp_original_init(NNCPOriginalState* state, const NNCPOriginalParams* params)
{
    if (!state || !params) return false;

    state->params = *params;
    state->train_step = 0;
    state->lr = 0.001f; // default learning rate
    
#ifdef USE_METAL
    if (params->use_metal) {
        // Initialize Metal context
        if (metal_context_create(&state->metal_context) != 0) {
            printf("[NNCP Original] Failed to create Metal context\n");
            return false;
        }
        
        // Initialize Neural Engine context
        if (ne_context_create(&state->ne_context, NE_BACKEND_AUTO) != 0) {
            printf("[NNCP Original] Failed to create Neural Engine context\n");
            metal_context_destroy(state->metal_context);
            return false;
        }
        
        // Note: Model loading would happen here when we have actual model files
        // For now, we'll use CPU fallback
        state->ne_model = NULL;
        
        printf("[NNCP Original] Neural Engine initialized successfully\n");
    }
#endif
    
    state->initialized = true;
    return true;
}

/*
 * Cleanup NNCP original state
 */
static void nncp_original_cleanup(NNCPOriginalState* state)
{
    if (!state || !state->initialized) return;
    
#ifdef USE_METAL
    if (state->params.use_metal) {
        // Clean up Neural Engine components in reverse order
        if (state->ne_model) {
            ne_model_destroy(state->ne_model);
            state->ne_model = NULL;
        }
        if (state->ne_context) {
            ne_context_destroy(state->ne_context);
            state->ne_context = NULL;
        }
        if (state->metal_context) {
            metal_context_destroy(state->metal_context);
            state->metal_context = NULL;
        }
        
        printf("[NNCP Original] Neural Engine cleaned up\n");
    }
#endif
    
    state->initialized = false;
}

/*
 * Model evaluation - Metal Transformer/LSTM implementation
 */
static bool nncp_original_model_eval(NNCPOriginalState* state, 
                                     const int32_t* input, int seq_len,
                                     float* output_probs)
{
    if (!state || !state->initialized || !input || !output_probs) return false;

#ifdef USE_METAL
    if (state->params.use_metal && state->ne_context && state->ne_model) {
        // Prepare input for Neural Engine (convert int32_t to float)
        float* ne_input = malloc(seq_len * sizeof(float));
        if (!ne_input) return false;
        
        for (int i = 0; i < seq_len; i++) {
            ne_input[i] = (float)input[i];
        }
        
        // Use Neural Engine for model evaluation
        int result = ne_model_predict(
            state->ne_model,
            ne_input, seq_len,
            output_probs, state->params.n_symbols * seq_len
        );
        
        free(ne_input);
        
        if (result == 0) {
            return true;
        } else {
            printf("[NNCP Original] Neural Engine evaluation failed, falling back to CPU\n");
        }
    }
#endif

    // Fallback: Simple uniform distribution (CPU fallback)
    float uniform_prob = 1.0f / state->params.n_symbols;
    for (int i = 0; i < seq_len * state->params.n_symbols; i++) {
        output_probs[i] = uniform_prob;
    }

    return true;
}

/*
 * Model update - placeholder for Metal implementation
 * TODO: Implement proper Metal gradient computation and model update
 */
static void nncp_original_model_update(NNCPOriginalState* state,
                                       const int32_t* expected_output, int seq_len)
{
    if (!state || !state->initialized) return;

    // Placeholder: increment training step
    state->train_step++;
}



/*
 * Encode symbol using arithmetic coding (simplified)
 * TODO: Implement proper arithmetic coding from original nncp.c
 */
// Helper function - simple implementation of vec_sum_f32
static float vec_sum_f32(const float* data, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum;
}

// Helper function - clamp integer to range
static inline int clamp_int(int val, int min_val, int max_val) {
    if (val < min_val)
        return min_val;
    else if (val > max_val)
        return max_val;
    else
        return val;
}

// Original NNCP Arithmetic Coding Implementation
#define PROB_UNIT_BITS 15
#define PROB_UNIT (1 << PROB_UNIT_BITS)

#define RANGE_MIN_BITS 16
#define RANGE_MIN ((0xff << (RANGE_MIN_BITS - 8)) + 1)
#define RANGE_MAX (0xff << RANGE_MIN_BITS)

// Simplified write function for compression
static void put_bit_write_func(void *opaque, const uint8_t *buf, size_t buf_size)
{
    // Data is already written to the buffer in put_bit_init
}

// Simplified read function for decompression
static size_t get_bit_read_func(void *opaque, uint8_t *buf, size_t buf_size)
{
    // Data is already in the buffer from get_bit_init
    return 0;
}

static void put_bit_init(PutBitState *s, uint8_t *buf, int buf_size,
                         PutBitWriteFunc *write_func, void *opaque)
{
    s->low = 0;
    s->range = RANGE_MAX;
    s->current_byte = 0xff;
    s->n_bytes = 0;
    s->buf = buf;
    s->buf_size = buf_size;
    s->idx = 0;
    s->write_func = write_func;  // Use provided write_func, or NULL for buffer mode
    s->opaque = opaque;
    s->byte_count = 0;
}

static void put_byte(PutBitState *s, int v)
{
    if (s->idx >= s->buf_size) {
        // Buffer is full, flush it
        if (s->write_func) {
            s->byte_count += s->idx;
            s->write_func(s->opaque, s->buf, s->idx);
            s->idx = 0;
        } else {
            // No write function, just limit writes to buffer size
            return;
        }
    }
    s->buf[s->idx++] = v;
}

static void put_val(PutBitState *s, int v)
{
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

static void put_val_flush(PutBitState *s)
{
    if (s->n_bytes > 0) {
        put_val(s, 0);
    }
}

static void put_bit_renorm(PutBitState *s)
{
    uint32_t v;
    /* after renormalisation:
       0 <= low <= RANGE_MAX
       RANGE_MIN <= range <= RANGE_MAX
       In the worst case before normalisation:
       low_max = 2 * RANGE_MAX hence v <= 0x1fe
    */
    while (s->range < RANGE_MIN) {
        v = s->low >> RANGE_MIN_BITS;
        put_val(s, v);
        s->low = (s->low & ((1 << RANGE_MIN_BITS) - 1)) << 8;
        s->range <<= 8;
    }
}

/* 0 < prob0 < PROB_UNIT */
static void put_bit(PutBitState *s, int prob0, int bit)
{
    uint32_t range0;
    
    range0 = ((uint64_t)s->range * prob0) >> PROB_UNIT_BITS;
    if (!bit) {
        s->range = range0;
    } else {
        s->low += range0;
        s->range -= range0;
    }
    
    put_bit_renorm(s);
}

static int64_t put_bit_flush(PutBitState *s)
{
    int n, val, mask;

    /* force larger range */
    if (s->range < (1 << RANGE_MIN_BITS)) {
        put_val(s, s->low >> RANGE_MIN_BITS);
        s->low = (s->low & ((1 << RANGE_MIN_BITS) - 1)) << 8;
        s->range <<= 8;
    }

    /* largest n such as 2^n <= range */
    n = 0;
    while ((1 << (n + 1)) <= s->range)
        n++;

    val = s->low;
    mask = (1 << n) - 1;
    if ((val & mask) != 0)
        val = (val + (1 << n)) & ~mask;

    put_val(s, val >> RANGE_MIN_BITS);
    put_val_flush(s);
    if (s->idx > 0 && s->write_func) {
        s->byte_count += s->idx;
        s->write_func(s->opaque, s->buf, s->idx);
        s->idx = 0;
    }
    return s->byte_count + s->idx;  // Return total bytes written
}

static void refill(GetBitState *s)
{
    s->range <<= 8;
    s->low <<= 8;
    if (s->idx >= s->buf_len) {
        if (!s->read_func || s->eof_reached)
            return; /* pad with zeros */
        s->buf_len = s->read_func(s->opaque, s->buf, s->buf_size);
        s->idx = 0;
        if (s->buf_len == 0) {
            s->eof_reached = true;
            s->byte_count++;
            return; /* pad with zeros */
        } else {
            s->byte_count += s->buf_len;
        }
    }
    s->low += s->buf[s->idx++];
}

static void get_bit_init(GetBitState *s, uint8_t *buf, size_t buf_size,
                         GetBitReadFunc *read_func, void *opaque)
{
    int i;
    s->buf_size = buf_size;
    s->buf = buf;
    s->read_func = read_func ? read_func : get_bit_read_func;
    s->opaque = opaque;
    if (read_func) {
        s->buf_len = 0;
    } else {
        /* prefilled buffer */
        s->buf_len = buf_size;
    }
    s->byte_count = s->buf_len;
    s->eof_reached = false;
    s->range = 0;
    s->low = 0;
    s->idx = 0;
    for (i = 0; i <= RANGE_MIN_BITS; i += 8) {
        refill(s);
    }
    s->range = RANGE_MAX;
}

static int get_bit(GetBitState *s, int prob0)
{
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
    while (s->range < RANGE_MIN)
        refill(s);
    return b;
}

// Original NNCP write_sym function
static void write_sym(PutBitState *pb, const float *prob_table, int n_symb, int sym)
{
    int start, range, prob0, bit, range0;
    float p, p0;
    
    start = 0;
    range = n_symb;
    p = 1.0; /* invariant: p=sum(prob_table[start...start + range]) */
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

// Original NNCP read_sym function
static int read_sym(GetBitState *gb, const float *prob_table, int n_symb)
{
    int start, range, prob0, bit, range0;
    float p, p0;
    
    start = 0;
    range = n_symb;
    p = 1.0; /* invariant: p=sum(prob_table[start...start + range]) */
    int iterations = 0;
    while (range > 1 && iterations < 20) {  // Add iteration limit to prevent infinite loop
        iterations++;
        
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

/*
 * Main decompression function implementing original NNCP algorithm
 * Based on process_block() from original nncp.c (decode mode)
 */
bool nncp_original_decompress(const uint8_t* compressed_data, size_t compressed_size,
                             uint8_t* output_data, size_t output_capacity,
                             size_t* decompressed_size, bool verbose)
{
    // Always log for debugging UTF-8 issues
    printf("[NNCP Original] LEGACY DECOMPRESSION CALLED: %zu bytes\n", compressed_size);
    
    if (!compressed_data || compressed_size == 0 || !output_data || output_capacity == 0) {
        return false;
    }

    // Initialize original NNCP parameters
    NNCPOriginalParams params = {
        .n_symbols = 256,           // byte-level compression
        .batch_size = 1,            // n_streams (default, will be optimized)
        .seg_len = 8,               // n_states (default, will be optimized)
        .n_layers = 2,              // reduced for testing
        .d_model = 128,             // reduced for testing
        .n_heads = 4,               // reduced for testing
        .use_metal = false          // Disable Metal for CPU fallback
    };
    
    // For decompression, estimate original input size from compressed size
    // (assuming compression ratio is at least 50%, which is conservative)
    size_t estimated_input_size = compressed_size * 2;
    
    // Optimize parameters based on estimated input size (same logic as compression)
    nncp_original_optimize_params_for_size(&params, estimated_input_size);
    
    NNCPOriginalState state = {0};
    if (!nncp_original_init(&state, &params)) {
        if (verbose) printf("Failed to initialize NNCP original state for decompression\n");
        return false;
    }

    GetBitState decoder;
    get_bit_init(&decoder, (uint8_t*)compressed_data, compressed_size, NULL, NULL);

    // Use actual output capacity for decompression target
    size_t target_output_size = output_capacity;  // This should be the expected original size

    size_t output_pos = 0;
    size_t block_size = params.batch_size * params.seg_len;
    
    if (verbose) {
        printf("[NNCP Original] Starting decompression of %zu bytes\n", compressed_size);
        printf("[NNCP Original] Block size: %zu (batch_size=%d, seg_len=%d)\n", 
               block_size, params.batch_size, params.seg_len);
    }

    // Process blocks until we've filled the output or exhausted input
    size_t block_idx = 0;
    while (output_pos < target_output_size && !decoder.eof_reached) {
        
        // Process each sequence position
        for (int pos = 0; pos < params.seg_len && output_pos < target_output_size; pos++) {
            // Prepare input for model (context from previous tokens)
            int32_t model_input[params.batch_size];
            for (int stream = 0; stream < params.batch_size; stream++) {
                if (pos == 0 || output_pos == 0) {
                    model_input[stream] = 0; // start token
                } else {
                    size_t prev_pos = output_pos - 1;
                    model_input[stream] = (prev_pos < output_pos) ? output_data[prev_pos] : 0;
                }
            }

            // Model evaluation
            float output_probs[params.batch_size * params.n_symbols];
            if (!nncp_original_model_eval(&state, model_input, params.batch_size, output_probs)) {
                if (verbose) printf("Model eval failed during decompression at block %zu, pos %d\n", block_idx, pos);
                nncp_original_cleanup(&state);
                return false;
            }

            // Decode symbols using arithmetic coding
            int32_t decoded_symbols[params.batch_size];
            for (int stream = 0; stream < params.batch_size; stream++) {
                if (output_pos < target_output_size) {
                    int symbol = read_sym(&decoder, &output_probs[stream * params.n_symbols], params.n_symbols);
                    
                    if (verbose && output_pos >= target_output_size - 5) {
                        printf("[DEBUG] Symbol %d at pos %zu, target %zu\n", symbol, output_pos, target_output_size);
                    }
                    
                    // Check for end marker
                    if (symbol == 0xFF) {
                        // End of data reached
                        if (verbose) printf("[DEBUG] End marker found at pos %zu\n", output_pos);
                        *decompressed_size = output_pos;
                        nncp_original_cleanup(&state);
                        return true;
                    }
                    
                    output_data[output_pos++] = (uint8_t)symbol;
                    decoded_symbols[stream] = symbol;
                } else {
                    if (verbose && output_pos < target_output_size) {
                        printf("[DEBUG] Stopping decode: pos=%zu, target=%zu, eof=%d\n", 
                               output_pos, target_output_size, decoder.eof_reached);
                    }
                    decoded_symbols[stream] = 0;
                }
            }

            // Model update (training step) - same as compression for consistency
            nncp_original_model_update(&state, decoded_symbols, params.batch_size);
        }

        block_idx++;
        if (verbose && (block_idx % 100 == 0)) {
            printf("\r[NNCP Original] Decompressed block %zu, output: %zu bytes", block_idx, output_pos);
            fflush(stdout);
        }
    }
    
    // Ensure new line after progress display
    if (verbose && block_idx > 0) {
        printf("\n");
    }

    *decompressed_size = output_pos;
    nncp_original_cleanup(&state);

    if (verbose) {
        printf("[NNCP Original] Decompression completed: %zu -> %zu bytes\n", 
               compressed_size, *decompressed_size);
    }

    return true;
}
