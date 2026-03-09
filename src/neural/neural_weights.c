/*
 * neural_weights.c
 *
 * Deterministic weight initialisation helpers for the NNCP Metal Transformer.
 * Pre-trained weight files (.nncpw) are removed — weights are always
 * initialised from a fixed LCG seed and trained online during compression.
 */

#include "neural_weights.h"

#include <string.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * Deterministic initialisation helpers
 * ---------------------------------------------------------------------- */

void nn_weights_init_deterministic(float* buf, size_t n_elements,
                                   uint32_t fan_in, uint32_t seed) {
    if (!buf || n_elements == 0 || fan_in == 0) return;
    float scale = sqrtf(2.0f / (float)fan_in);
    uint32_t state = seed;
    for (size_t i = 0; i < n_elements; i++) {
        state = state * 1664525u + 1013904223u;
        /* Map upper 24 bits to [-1, 1), then scale */
        float v = ((float)(state >> 8) / (float)(1u << 23)) - 1.0f;
        buf[i] = v * scale;
    }
}

void nn_weights_init_uniform(float* buf, size_t n_elements, float scale, uint32_t seed) {
    if (!buf || n_elements == 0) return;
    uint32_t state = seed;
    for (size_t i = 0; i < n_elements; i++) {
        state = state * 1664525u + 1013904223u;
        float v = ((float)(state >> 8) / (float)(1u << 23)) - 1.0f;
        buf[i] = v * scale;
    }
}

void nn_weights_init_zeros(float* buf, size_t n_elements) {
    if (!buf || n_elements == 0) return;
    memset(buf, 0, n_elements * sizeof(float));
}

void nn_weights_init_layer_norm(float* gamma_beta, uint32_t hidden_size,
                                uint32_t n_layers) {
    if (!gamma_beta || hidden_size == 0 || n_layers == 0) return;
    /* Layout: [n_layers, 2, hidden_size]
     * gamma = [layer, 0, :] → 1.0
     * beta  = [layer, 1, :] → 0.0  */
    for (uint32_t l = 0; l < n_layers; l++) {
        float* gamma = gamma_beta + (size_t)l * 2 * hidden_size;
        float* beta  = gamma + hidden_size;
        for (uint32_t h = 0; h < hidden_size; h++) gamma[h] = 1.0f;
        for (uint32_t h = 0; h < hidden_size; h++) beta[h]  = 0.0f;
    }
}
