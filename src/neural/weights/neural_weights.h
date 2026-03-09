/*
 * neural_weights.h
 *
 * Deterministic weight initialisation helpers for the NNCP Metal Transformer.
 * Pre-trained weight files (.nncpw) are not used — the model is always
 * initialised from a fixed LCG seed and trained online during compression.
 *
 * Pure C — no Objective-C imports required.
 */

#ifndef NEURAL_WEIGHTS_H
#define NEURAL_WEIGHTS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Deterministic initialisation helpers
 * ---------------------------------------------------------------------- */

/**
 * Fill `buf` with deterministic Xavier-like values using a linear
 * congruential generator keyed by `seed`.
 *
 *   scale = sqrt(2 / fan_in)
 *   LCG:  state = state * 1664525 + 1013904223  (Numerical Recipes)
 *   value = ((state >> 8) / 2^23 - 1.0) * scale   → uniform in (-scale, scale)
 *
 * Using the same (seed, fan_in) pair on compress and decompress sides
 * guarantees bit-identical initial weights.
 */
void nn_weights_init_deterministic(float* buf, size_t n_elements,
                                   uint32_t fan_in, uint32_t seed);

/**
 * Fill `buf` with zeros.
 */
void nn_weights_init_zeros(float* buf, size_t n_elements);

/**
 * Initialise LayerNorm weight blocks: gamma=1.0, beta=0.0.
 *
 * Layout expected: [n_layers, 2, hidden_size]
 *   gamma = slice [:, 0, :]
 *   beta  = slice [:, 1, :]
 *
 * Pass n_layers=1 and the gamma_beta pointer for the final LayerNorm
 * (layout [2, hidden_size]).
 */
void nn_weights_init_layer_norm(float* gamma_beta, uint32_t hidden_size,
                                uint32_t n_layers);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_WEIGHTS_H */
