/*
 * neural_weights.c
 *
 * Pure-C implementation of the .nncpw weight file format.
 * See neural_weights.h for the binary layout specification.
 */

#include "neural_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <errno.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

static bool write_floats(FILE* f, const float* data, size_t count) {
    if (!data || count == 0) return true;
    return fwrite(data, sizeof(float), count, f) == count;
}

static bool read_floats(FILE* f, float* data, size_t count) {
    if (!data || count == 0) {
        /* Skip this section by seeking ahead */
        return fseek(f, (long)(count * sizeof(float)), SEEK_CUR) == 0;
    }
    return fread(data, sizeof(float), count, f) == count;
}

/* Recursively create directories for the path (like `mkdir -p dirname(path)`). */
bool nn_weights_ensure_dir(const char* path) {
    /* Work on a mutable copy of the directory component */
    char buf[4096];
    strncpy(buf, path, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    /* Strip filename — find last slash */
    char* last_slash = strrchr(buf, '/');
    if (!last_slash) return true;  /* relative path, no dir component */
    *last_slash = '\0';

    /* Walk through each component and mkdir as needed */
    for (char* p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(buf, 0755) != 0 && errno != EEXIST) return false;
            *p = '/';
        }
    }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) return false;
    return true;
}

/* -------------------------------------------------------------------------
 * Default path
 * ---------------------------------------------------------------------- */

const char* nn_weights_default_path(void) {
    static char path[4096] = {0};
    if (path[0] != '\0') return path;

    const char* home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(path, sizeof(path), "%s/.config/nncp/model.nncpw", home);
    return path;
}

/* -------------------------------------------------------------------------
 * Save
 * ---------------------------------------------------------------------- */

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
                     const float* out_proj) {
    if (!path || !cfg) return false;

    if (!nn_weights_ensure_dir(path)) {
        fprintf(stderr, "[nn_weights] Cannot create directory for: %s\n", path);
        return false;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[nn_weights] Cannot open for writing: %s (%s)\n",
                path, strerror(errno));
        return false;
    }

    bool ok = true;

    /* Magic (5 bytes, no null) */
    ok = ok && (fwrite(NN_WEIGHTS_MAGIC, 1, 5, f) == 5);

    /* Version */
    uint32_t ver = NN_WEIGHTS_VERSION;
    ok = ok && (fwrite(&ver, sizeof(uint32_t), 1, f) == 1);

    /* Config: 6 × uint32 */
    ok = ok && (fwrite(&cfg->num_layers,  sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fwrite(&cfg->hidden_size, sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fwrite(&cfg->num_heads,   sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fwrite(&cfg->ffn_size,    sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fwrite(&cfg->vocab_size,  sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fwrite(&cfg->max_seq_len, sizeof(uint32_t), 1, f) == 1);

    const uint32_t L  = cfg->num_layers;
    const uint32_t H  = cfg->hidden_size;
    const uint32_t F  = cfg->ffn_size;
    const uint32_t V  = cfg->vocab_size;
    const uint32_t S  = cfg->max_seq_len;

    /* Weight arrays */
    ok = ok && write_floats(f, embed,     (size_t)V * H);
    ok = ok && write_floats(f, pos_embed, (size_t)S * H);
    ok = ok && write_floats(f, attn_q,    (size_t)L * H * H);
    ok = ok && write_floats(f, attn_k,    (size_t)L * H * H);
    ok = ok && write_floats(f, attn_v,    (size_t)L * H * H);
    ok = ok && write_floats(f, attn_out,  (size_t)L * H * H);
    ok = ok && write_floats(f, ffn1,      (size_t)L * H * F * 2);
    ok = ok && write_floats(f, ffn2,      (size_t)L * F * H);
    ok = ok && write_floats(f, ln,        (size_t)L * 2 * H);
    ok = ok && write_floats(f, final_ln,  (size_t)2 * H);
    ok = ok && write_floats(f, out_proj,  (size_t)H * V);

    fclose(f);
    if (!ok) {
        fprintf(stderr, "[nn_weights] Write error for: %s\n", path);
        remove(path);  /* Remove partial file */
    }
    return ok;
}

/* -------------------------------------------------------------------------
 * Load
 * ---------------------------------------------------------------------- */

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
                     float* out_proj) {
    if (!path || !cfg_out) return false;

    FILE* f = fopen(path, "rb");
    if (!f) return false;  /* File simply doesn't exist yet — not an error */

    bool ok = true;

    /* Magic */
    char magic[6] = {0};
    ok = ok && (fread(magic, 1, 5, f) == 5);
    if (ok && memcmp(magic, NN_WEIGHTS_MAGIC, 5) != 0) {
        fprintf(stderr, "[nn_weights] Bad magic in: %s\n", path);
        fclose(f);
        return false;
    }

    /* Version */
    uint32_t ver = 0;
    ok = ok && (fread(&ver, sizeof(uint32_t), 1, f) == 1);
    if (ok && ver != NN_WEIGHTS_VERSION) {
        fprintf(stderr, "[nn_weights] Unsupported version %u in: %s\n", ver, path);
        fclose(f);
        return false;
    }

    /* Config */
    ok = ok && (fread(&cfg_out->num_layers,  sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fread(&cfg_out->hidden_size, sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fread(&cfg_out->num_heads,   sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fread(&cfg_out->ffn_size,    sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fread(&cfg_out->vocab_size,  sizeof(uint32_t), 1, f) == 1);
    ok = ok && (fread(&cfg_out->max_seq_len, sizeof(uint32_t), 1, f) == 1);

    if (!ok) {
        fprintf(stderr, "[nn_weights] Failed to read header from: %s\n", path);
        fclose(f);
        return false;
    }

    const uint32_t L = cfg_out->num_layers;
    const uint32_t H = cfg_out->hidden_size;
    const uint32_t F = cfg_out->ffn_size;
    const uint32_t V = cfg_out->vocab_size;
    const uint32_t S = cfg_out->max_seq_len;

    ok = ok && read_floats(f, embed,     (size_t)V * H);
    ok = ok && read_floats(f, pos_embed, (size_t)S * H);
    ok = ok && read_floats(f, attn_q,    (size_t)L * H * H);
    ok = ok && read_floats(f, attn_k,    (size_t)L * H * H);
    ok = ok && read_floats(f, attn_v,    (size_t)L * H * H);
    ok = ok && read_floats(f, attn_out,  (size_t)L * H * H);
    ok = ok && read_floats(f, ffn1,      (size_t)L * H * F * 2);
    ok = ok && read_floats(f, ffn2,      (size_t)L * F * H);
    ok = ok && read_floats(f, ln,        (size_t)L * 2 * H);
    ok = ok && read_floats(f, final_ln,  (size_t)2 * H);
    ok = ok && read_floats(f, out_proj,  (size_t)H * V);

    fclose(f);
    if (!ok) {
        fprintf(stderr, "[nn_weights] Read error for weight data in: %s\n", path);
    }
    return ok;
}

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
