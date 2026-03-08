/*
 * neural_weights.h
 *
 * Binary weight file format (.nncpw) for the NNCP Metal Transformer model.
 *
 * File layout
 * -----------
 *   [0..4]   Magic      : "NNCPW"  (5 bytes, no null terminator in file)
 *   [5..8]   Version    : uint32   (= 1)
 *   [9..32]  Config     : 6 × uint32  (num_layers, hidden_size, num_heads,
 *                                       ffn_size, vocab_size, max_seq_len)
 *   [33..]   Weights    : float32 arrays (see order below)
 *
 * Weight arrays (in order):
 *   1. embedding_weights       vocab_size  × hidden_size
 *   2. position_embeddings     max_seq_len × hidden_size
 *   3. attention_weights_q     num_layers  × hidden_size × hidden_size
 *   4. attention_weights_k     num_layers  × hidden_size × hidden_size
 *   5. attention_weights_v     num_layers  × hidden_size × hidden_size
 *   6. attention_output_weights num_layers × hidden_size × hidden_size
 *   7. ffn_weights_1           num_layers  × hidden_size × ffn_size × 2
 *   8. ffn_weights_2           num_layers  × ffn_size    × hidden_size
 *   9. layer_norm_weights      num_layers  × 2           × hidden_size
 *  10. final_layer_norm_weights 2          × hidden_size
 *  11. output_projection       hidden_size × vocab_size
 *
 * Pure C — no Objective-C imports required.
 */

#ifndef NEURAL_WEIGHTS_H
#define NEURAL_WEIGHTS_H

#include <stdint.h>
#include <stdbool.h>

#define NN_WEIGHTS_MAGIC   "NNCPW"
#define NN_WEIGHTS_VERSION ((uint32_t)1)

typedef struct NNWeightsConfig {
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t num_heads;
    uint32_t ffn_size;
    uint32_t vocab_size;
    uint32_t max_seq_len;
} NNWeightsConfig;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Save model weights to a binary .nncpw file.
 * Returns true on success, false on any I/O error.
 */
bool nn_weights_save(const char* path,
                     const NNWeightsConfig* cfg,
                     const float* embed,
                     const float* pos_embed,
                     const float* attn_q,
                     const float* attn_k,
                     const float* attn_v,
                     const float* attn_out,
                     const float* ffn1,
                     const float* ffn2,
                     const float* ln,
                     const float* final_ln,
                     const float* out_proj);

/**
 * Load model weights from a .nncpw file.
 * cfg_out  : receives the config stored in the file.
 * All float* buffers must already be allocated with the correct sizes
 * (matching the config returned in cfg_out).  Pass NULL to skip a tensor.
 * Returns true on success, false if the file is missing, corrupted, or
 * has a version mismatch.
 */
bool nn_weights_load(const char* path,
                     NNWeightsConfig* cfg_out,
                     float* embed,
                     float* pos_embed,
                     float* attn_q,
                     float* attn_k,
                     float* attn_v,
                     float* attn_out,
                     float* ffn1,
                     float* ffn2,
                     float* ln,
                     float* final_ln,
                     float* out_proj);

/**
 * Returns the default weight file path: "$HOME/.config/nncp/model.nncpw".
 * The returned pointer is valid for the lifetime of the process.
 */
const char* nn_weights_default_path(void);

/**
 * Ensure the directory component of `path` exists (creates it if necessary).
 * Returns true on success.
 */
bool nn_weights_ensure_dir(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_WEIGHTS_H */
