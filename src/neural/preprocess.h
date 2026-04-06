/*
 * NNCP Preprocessor — Memory-buffer API
 *
 * Ports the dictionary-based preprocessor from the original NNCP (Fabrice Bellard, MIT).
 * Case/space encoding + iterative bigram merging, vocab 256 → up to 256+n_words.
 *
 * DataSymbol (uint16_t):
 *   0–255  : byte values (incl. case/space special codes 1–4)
 *   256+   : dictionary words
 */

#ifndef NNCP_PREPROCESS_H
#define NNCP_PREPROCESS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Encode raw bytes to preprocessed symbols.
 *
 * input      : raw input bytes
 * input_len  : number of input bytes
 * out_symbols: *out_symbols is set to a malloc'd uint16_t array (caller must free)
 * out_n_sym  : number of uint16_t symbols written
 * out_dict   : *out_dict is set to a malloc'd byte buffer containing the
 *              serialized dictionary text (caller must free).  Format:
 *              one word per line, raw bytes, '\n' escaped as "\\n", '\\' as "\\\\"
 * out_dict_len: number of bytes in *out_dict
 * n_words    : approximate maximum dictionary size (e.g. 4096)
 * min_freq   : minimum bigram frequency to keep (e.g. 512)
 *
 * Returns the actual number of words added to the dictionary (>= 0), or -1 on error.
 */
int nncp_preprocess_encode(
    const uint8_t *input, size_t input_len,
    uint16_t **out_symbols, size_t *out_n_sym,
    uint8_t **out_dict, size_t *out_dict_len,
    int n_words, int min_freq);

/*
 * Decode preprocessed symbols back to raw bytes.
 *
 * symbols    : uint16_t symbol stream produced by nncp_preprocess_encode
 * n_symbols  : length of symbol stream
 * out_bytes  : *out_bytes is set to a malloc'd byte array (caller must free)
 * out_len    : number of bytes in *out_bytes
 * dict_data  : dictionary bytes produced by nncp_preprocess_encode
 * dict_len   : length of dict_data
 *
 * Returns 0 on success, -1 on error.
 */
int nncp_preprocess_decode(
    const uint16_t *symbols, size_t n_symbols,
    uint8_t **out_bytes, size_t *out_len,
    const uint8_t *dict_data, size_t dict_len);

#ifdef __cplusplus
}
#endif

#endif /* NNCP_PREPROCESS_H */
