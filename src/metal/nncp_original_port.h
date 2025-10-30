/*
 * Original NNCP Algorithm Port to Metal - Header
 * 
 * Based on original nncp.c by Fabrice Bellard
 * Ported for Apple Silicon Metal acceleration
 */

#ifndef NNCP_ORIGINAL_PORT_H
#define NNCP_ORIGINAL_PORT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for arithmetic coding
typedef void (*PutBitWriteFunc)(void *opaque, const uint8_t *buf, size_t buf_size);
typedef size_t (*GetBitReadFunc)(void *opaque, uint8_t *buf, size_t buf_size);

// PutBitState for arithmetic encoding
typedef struct {
    uint32_t low;
    uint32_t range;
    uint8_t current_byte;
    int n_bytes;
    uint8_t *buf;
    int buf_size;
    int idx;
    PutBitWriteFunc *write_func;
    void *opaque;
    uint64_t byte_count;
} PutBitState;

// GetBitState for arithmetic decoding
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

// Arithmetic coding function declarations
void put_bit_init(PutBitState *s, uint8_t *buf, int buf_size, PutBitWriteFunc *write_func, void *opaque);
void put_bit(PutBitState *s, int prob0, int bit);
int64_t put_bit_flush(PutBitState *s);
void get_bit_init(GetBitState *s, uint8_t *buf, size_t buf_size, GetBitReadFunc *read_func, void *opaque);
int get_bit(GetBitState *s, int prob0);
void write_sym(PutBitState *pb, const float *prob_table, int n_symb, int sym);
int read_sym(GetBitState *gb, const float *prob_table, int n_symb);

/*
 * Main compression function implementing original NNCP algorithm
 * Based on process_block() from original nncp.c
 * 
 * @param input_data: Input data to compress
 * @param input_size: Size of input data in bytes
 * @param output_data: Output buffer for compressed data
 * @param output_capacity: Capacity of output buffer
 * @param compressed_size: [out] Actual compressed size
 * @param verbose: Enable verbose logging
 * @return true on success, false on failure
 */
bool nncp_original_compress(const uint8_t* input_data, size_t input_size,
                           uint8_t* output_data, size_t output_capacity,
                           size_t* compressed_size, bool verbose);

/*
 * Decompression function (TODO: implement)
 */
bool nncp_original_decompress(const uint8_t* compressed_data, size_t compressed_size,
                             uint8_t* output_data, size_t output_capacity,
                             size_t* decompressed_size, bool verbose);

#ifdef __cplusplus
}
#endif

#endif // NNCP_ORIGINAL_PORT_H
