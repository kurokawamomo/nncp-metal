/*
 * NNCP Preprocessor — Memory-buffer implementation
 *
 * Ported from the original NNCP (Fabrice Bellard, MIT License).
 * Original: https://bellard.org/nncp/
 *
 * This file implements case/space encoding and iterative bigram merging
 * (BPE-like dictionary compression) to expand the effective vocabulary from
 * 256 byte values to up to 256+n_words symbols before neural compression.
 */

#include "preprocess.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Compat helpers (replaces cutils.h / cp_utils.h from original NNCP)
 * ---------------------------------------------------------------------- */

typedef uint16_t DataSymbol;
typedef int      BOOL;
#define TRUE  1
#define FALSE 0

static inline void *pp_mallocz(size_t n) { return calloc(1, n); }
static inline int   pp_min_int(int a, int b) { return (a < b) ? a : b; }
static inline size_t pp_max_sz(size_t a, size_t b) { return (a > b) ? a : b; }

/* count leading zeros (portable: clang/gcc __builtin_clz) */
static inline int pp_clz32(uint32_t n)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(n);
#else
    int i;
    for (i = 31; i >= 0; i--)
        if (n & (1u << i)) return 31 - i;
    return 32;
#endif
}

/* -------------------------------------------------------------------------
 * WordList — hash-table of 2-symbol bigrams
 * ---------------------------------------------------------------------- */

/* number of reserved single-byte symbols */
#define NS 256

typedef struct Word {
    uint32_t next;  /* hash chain, -1 = end */
    uint32_t freq;
    int64_t  score;
    uint32_t len;   /* always 2 for bigrams */
    DataSymbol buf[2];
} Word;

typedef struct {
    Word   *words;
    size_t  word_count;
    size_t  word_size;
    uint32_t *hash_table;
    int hash_size;
    int hash_bits;
} WordList;

static uint32_t hash_calc(const DataSymbol *buf, int len, int n_bits)
{
    uint32_t h = 1;
    for (int i = 0; i < len; i++)
        h = h * 314159 + buf[i];
    return h & ((1 << n_bits) - 1);
}

static void hash_resize(WordList *s, int hash_bits)
{
    s->hash_bits = hash_bits;
    s->hash_size = 1 << hash_bits;
    free(s->hash_table);
    s->hash_table = (uint32_t *)malloc(sizeof(s->hash_table[0]) * s->hash_size);
    for (int i = 0; i < s->hash_size; i++)
        s->hash_table[i] = (uint32_t)-1;
    for (int i = 0; i < (int)s->word_count; i++) {
        Word *p = &s->words[i];
        int h = (int)hash_calc(p->buf, (int)p->len, s->hash_bits);
        p->next = s->hash_table[h];
        s->hash_table[h] = (uint32_t)i;
    }
}

static WordList *word_list_init(void)
{
    WordList *s = (WordList *)pp_mallocz(sizeof(WordList));
    hash_resize(s, 12);
    return s;
}

static void word_list_end(WordList *s)
{
    free(s->words);
    free(s->hash_table);
    free(s);
}

#define HASH_SIZE_FACTOR 2

static Word *word_find_add(WordList *s, const DataSymbol *buf, int len, int add)
{
    uint32_t h = hash_calc(buf, len, s->hash_bits);
    uint32_t idx = s->hash_table[h];
    while (idx != (uint32_t)-1) {
        Word *p = &s->words[idx];
        if (p->len == (uint32_t)len && !memcmp(p->buf, buf, len * sizeof(buf[0])))
            return p;
        idx = p->next;
    }
    if (!add) return NULL;

    if (s->word_count >= s->word_size) {
        size_t new_size = s->word_size + s->word_size / 2;
        if (new_size < 32) new_size = 32;
        if (s->word_count + 1 > new_size) new_size = s->word_count + 1;
        s->words = (Word *)realloc(s->words, new_size * sizeof(s->words[0]));
        s->word_size = new_size;
    }
    if ((s->word_count * HASH_SIZE_FACTOR) > (size_t)s->hash_size) {
        int hb = s->hash_bits;
        while ((s->word_count * HASH_SIZE_FACTOR) > (size_t)(1 << hb)) hb++;
        hash_resize(s, hb);
        h = hash_calc(buf, len, s->hash_bits);
    }
    idx = (uint32_t)s->word_count++;
    Word *p = &s->words[idx];
    p->freq = 0;
    p->len  = (uint32_t)len;
    for (int i = 0; i < len; i++) p->buf[i] = buf[i];
    p->next = s->hash_table[h];
    s->hash_table[h] = idx;
    return p;
}

/* -------------------------------------------------------------------------
 * Buffer helper
 * ---------------------------------------------------------------------- */

static void buf_realloc(DataSymbol **pbuf, size_t *pbuf_size, size_t new_size)
{
    if (new_size <= *pbuf_size) return;
    new_size = pp_max_sz(new_size, *pbuf_size + (*pbuf_size) / 8);
    *pbuf      = (DataSymbol *)realloc(*pbuf, new_size * sizeof(**pbuf));
    *pbuf_size = new_size;
}

/* Expand a symbol recursively; only leaves with freq>0 (or raw bytes) are output */
static void out_word(DataSymbol **pbuf, size_t *pbuf_size,
                     size_t *pbuf_pos, WordList *s, uint32_t c)
{
    if (c < NS) {
        size_t pos = *pbuf_pos;
        if (pos >= *pbuf_size)
            buf_realloc(pbuf, pbuf_size, pos + 1);
        (*pbuf)[pos++] = (DataSymbol)c;
        *pbuf_pos = pos;
    } else {
        Word *p = &s->words[c - NS];
        if (p->freq == 0) {
            out_word(pbuf, pbuf_size, pbuf_pos, s, p->buf[0]);
            out_word(pbuf, pbuf_size, pbuf_pos, s, p->buf[1]);
        } else {
            size_t pos = *pbuf_pos;
            if (pos >= *pbuf_size)
                buf_realloc(pbuf, pbuf_size, pos + 1);
            (*pbuf)[pos++] = (DataSymbol)c;
            *pbuf_pos = pos;
        }
    }
}

/* -------------------------------------------------------------------------
 * Case/Space encoding
 * ---------------------------------------------------------------------- */

#define CH_NO_SPACE    1
#define CH_TO_UPPER    2
#define CH_FIRST_UPPER 3
#define CH_ESCAPE      4

static int is_word_char(int c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= 128);
}
static int is_upper(int c) { return (c >= 'A' && c <= 'Z'); }
static int is_lower(int c) { return (c >= 'a' && c <= 'z') || (c >= 128); }

/*
 * Apply case/space encoding: words get CH_NO_SPACE? CH_TO_UPPER? SPACE lowercase...
 * Returns malloc'd DataSymbol buffer; caller must free.
 */
static DataSymbol *case_space_encoding(size_t *pobuf_len,
                                        DataSymbol *buf, size_t buf_len)
{
    DataSymbol *obuf;
    size_t obuf_size = buf_len;
    size_t k = 0;

    obuf = (DataSymbol *)malloc(sizeof(buf[0]) * obuf_size);
    if (!obuf) return NULL;

    for (size_t i = 0; i < buf_len; ) {
        if (is_word_char(buf[i])) {
            size_t j = i + 1;
            int ch_type;
            if (is_lower(buf[i])) {
                while (j < buf_len && is_lower(buf[j])) j++;
                ch_type = 0;
            } else if (j < buf_len && is_upper(buf[j])) {
                while (j < buf_len && is_upper(buf[j])) j++;
                ch_type = CH_TO_UPPER;
            } else {
                while (j < buf_len && is_lower(buf[j])) j++;
                ch_type = CH_FIRST_UPPER;
            }
            size_t len = j - i;

            BOOL has_space = FALSE;
            if (k > 0 && obuf[k - 1] == ' ') {
                has_space = TRUE;
                k--;
            }
            buf_realloc(&obuf, &obuf_size, k + len + 3);
            if (!has_space)
                obuf[k++] = CH_NO_SPACE;
            if (ch_type != 0)
                obuf[k++] = (DataSymbol)ch_type;
            obuf[k++] = ' ';
            for (size_t l = 0; l < len; l++) {
                int c = buf[i + l];
                if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
                obuf[k++] = (DataSymbol)c;
            }
            i += len;
        } else if (buf[i] == CH_NO_SPACE ||
                   buf[i] == CH_TO_UPPER ||
                   buf[i] == CH_FIRST_UPPER ||
                   buf[i] == CH_ESCAPE) {
            buf_realloc(&obuf, &obuf_size, k + 2);
            obuf[k++] = CH_ESCAPE;
            obuf[k++] = buf[i++];
        } else {
            buf_realloc(&obuf, &obuf_size, k + 1);
            obuf[k++] = buf[i++];
        }
    }
    obuf         = (DataSymbol *)realloc(obuf, sizeof(obuf[0]) * (k ? k : 1));
    *pobuf_len   = k;
    return obuf;
}

/* -------------------------------------------------------------------------
 * Case/Space decoding state machine
 * ---------------------------------------------------------------------- */

typedef struct {
    BOOL has_space;
    int  ch_type;
    int  ch_type1;
    BOOL has_escape;
} CaseSpaceDecodeState;

static void case_space_decode_init(CaseSpaceDecodeState *s)
{
    s->ch_type   = 0;
    s->has_space = TRUE;
    s->ch_type1  = 0;
    s->has_escape = FALSE;
}

/* return byte to output, or -1 if consumed/modifier */
static int case_space_decode(CaseSpaceDecodeState *s, int c)
{
    if (s->has_escape) {
        s->has_escape = FALSE;
    } else if (c == CH_TO_UPPER || c == CH_FIRST_UPPER) {
        s->ch_type = c;
        return -1;
    } else if (c == CH_NO_SPACE) {
        s->has_space = FALSE;
        return -1;
    } else if (c == CH_ESCAPE) {
        s->has_escape = TRUE;
        return -1;
    } else if (c == ' ') {
        s->ch_type1  = s->ch_type;
        s->ch_type   = 0;
        if (!s->has_space) { s->has_space = TRUE; return -1; }
        s->has_space = TRUE;
    } else {
        if (s->ch_type1 == CH_TO_UPPER || s->ch_type1 == CH_FIRST_UPPER) {
            if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';
            if (s->ch_type1 == CH_FIRST_UPPER) s->ch_type1 = 0;
        }
        s->has_space = TRUE;
    }
    return c;
}

/* -------------------------------------------------------------------------
 * Scoring / frequency helpers
 * ---------------------------------------------------------------------- */

#define FRAC_BITS 10
#define FRAC_ONE  (1 << FRAC_BITS)

static const uint16_t log2_table[FRAC_ONE] = {
 0x000, 0x001, 0x003, 0x004, 0x006, 0x007, 0x009, 0x00a,
 0x00b, 0x00d, 0x00e, 0x010, 0x011, 0x013, 0x014, 0x015,
 0x017, 0x018, 0x01a, 0x01b, 0x01d, 0x01e, 0x01f, 0x021,
 0x022, 0x024, 0x025, 0x026, 0x028, 0x029, 0x02b, 0x02c,
 0x02d, 0x02f, 0x030, 0x032, 0x033, 0x034, 0x036, 0x037,
 0x039, 0x03a, 0x03b, 0x03d, 0x03e, 0x040, 0x041, 0x042,
 0x044, 0x045, 0x046, 0x048, 0x049, 0x04b, 0x04c, 0x04d,
 0x04f, 0x050, 0x051, 0x053, 0x054, 0x055, 0x057, 0x058,
 0x05a, 0x05b, 0x05c, 0x05e, 0x05f, 0x060, 0x062, 0x063,
 0x064, 0x066, 0x067, 0x068, 0x06a, 0x06b, 0x06c, 0x06e,
 0x06f, 0x070, 0x072, 0x073, 0x074, 0x076, 0x077, 0x078,
 0x07a, 0x07b, 0x07c, 0x07e, 0x07f, 0x080, 0x082, 0x083,
 0x084, 0x086, 0x087, 0x088, 0x08a, 0x08b, 0x08c, 0x08e,
 0x08f, 0x090, 0x092, 0x093, 0x094, 0x095, 0x097, 0x098,
 0x099, 0x09b, 0x09c, 0x09d, 0x09f, 0x0a0, 0x0a1, 0x0a2,
 0x0a4, 0x0a5, 0x0a6, 0x0a8, 0x0a9, 0x0aa, 0x0ab, 0x0ad,
 0x0ae, 0x0af, 0x0b1, 0x0b2, 0x0b3, 0x0b4, 0x0b6, 0x0b7,
 0x0b8, 0x0b9, 0x0bb, 0x0bc, 0x0bd, 0x0bf, 0x0c0, 0x0c1,
 0x0c2, 0x0c4, 0x0c5, 0x0c6, 0x0c7, 0x0c9, 0x0ca, 0x0cb,
 0x0cc, 0x0ce, 0x0cf, 0x0d0, 0x0d1, 0x0d3, 0x0d4, 0x0d5,
 0x0d6, 0x0d8, 0x0d9, 0x0da, 0x0db, 0x0dd, 0x0de, 0x0df,
 0x0e0, 0x0e2, 0x0e3, 0x0e4, 0x0e5, 0x0e7, 0x0e8, 0x0e9,
 0x0ea, 0x0ec, 0x0ed, 0x0ee, 0x0ef, 0x0f0, 0x0f2, 0x0f3,
 0x0f4, 0x0f5, 0x0f7, 0x0f8, 0x0f9, 0x0fa, 0x0fb, 0x0fd,
 0x0fe, 0x0ff, 0x100, 0x102, 0x103, 0x104, 0x105, 0x106,
 0x108, 0x109, 0x10a, 0x10b, 0x10c, 0x10e, 0x10f, 0x110,
 0x111, 0x112, 0x114, 0x115, 0x116, 0x117, 0x118, 0x11a,
 0x11b, 0x11c, 0x11d, 0x11e, 0x120, 0x121, 0x122, 0x123,
 0x124, 0x125, 0x127, 0x128, 0x129, 0x12a, 0x12b, 0x12d,
 0x12e, 0x12f, 0x130, 0x131, 0x132, 0x134, 0x135, 0x136,
 0x137, 0x138, 0x139, 0x13b, 0x13c, 0x13d, 0x13e, 0x13f,
 0x140, 0x142, 0x143, 0x144, 0x145, 0x146, 0x147, 0x148,
 0x14a, 0x14b, 0x14c, 0x14d, 0x14e, 0x14f, 0x151, 0x152,
 0x153, 0x154, 0x155, 0x156, 0x157, 0x159, 0x15a, 0x15b,
 0x15c, 0x15d, 0x15e, 0x15f, 0x161, 0x162, 0x163, 0x164,
 0x165, 0x166, 0x167, 0x168, 0x16a, 0x16b, 0x16c, 0x16d,
 0x16e, 0x16f, 0x170, 0x172, 0x173, 0x174, 0x175, 0x176,
 0x177, 0x178, 0x179, 0x17a, 0x17c, 0x17d, 0x17e, 0x17f,
 0x180, 0x181, 0x182, 0x183, 0x184, 0x186, 0x187, 0x188,
 0x189, 0x18a, 0x18b, 0x18c, 0x18d, 0x18e, 0x190, 0x191,
 0x192, 0x193, 0x194, 0x195, 0x196, 0x197, 0x198, 0x199,
 0x19b, 0x19c, 0x19d, 0x19e, 0x19f, 0x1a0, 0x1a1, 0x1a2,
 0x1a3, 0x1a4, 0x1a5, 0x1a6, 0x1a8, 0x1a9, 0x1aa, 0x1ab,
 0x1ac, 0x1ad, 0x1ae, 0x1af, 0x1b0, 0x1b1, 0x1b2, 0x1b3,
 0x1b4, 0x1b6, 0x1b7, 0x1b8, 0x1b9, 0x1ba, 0x1bb, 0x1bc,
 0x1bd, 0x1be, 0x1bf, 0x1c0, 0x1c1, 0x1c2, 0x1c3, 0x1c5,
 0x1c6, 0x1c7, 0x1c8, 0x1c9, 0x1ca, 0x1cb, 0x1cc, 0x1cd,
 0x1ce, 0x1cf, 0x1d0, 0x1d1, 0x1d2, 0x1d3, 0x1d4, 0x1d5,
 0x1d6, 0x1d8, 0x1d9, 0x1da, 0x1db, 0x1dc, 0x1dd, 0x1de,
 0x1df, 0x1e0, 0x1e1, 0x1e2, 0x1e3, 0x1e4, 0x1e5, 0x1e6,
 0x1e7, 0x1e8, 0x1e9, 0x1ea, 0x1eb, 0x1ec, 0x1ed, 0x1ee,
 0x1ef, 0x1f0, 0x1f1, 0x1f3, 0x1f4, 0x1f5, 0x1f6, 0x1f7,
 0x1f8, 0x1f9, 0x1fa, 0x1fb, 0x1fc, 0x1fd, 0x1fe, 0x1ff,
 0x200, 0x201, 0x202, 0x203, 0x204, 0x205, 0x206, 0x207,
 0x208, 0x209, 0x20a, 0x20b, 0x20c, 0x20d, 0x20e, 0x20f,
 0x210, 0x211, 0x212, 0x213, 0x214, 0x215, 0x216, 0x217,
 0x218, 0x219, 0x21a, 0x21b, 0x21c, 0x21d, 0x21e, 0x21f,
 0x220, 0x221, 0x222, 0x223, 0x224, 0x225, 0x226, 0x227,
 0x228, 0x229, 0x22a, 0x22b, 0x22c, 0x22d, 0x22e, 0x22f,
 0x230, 0x231, 0x232, 0x233, 0x234, 0x235, 0x236, 0x237,
 0x238, 0x239, 0x23a, 0x23b, 0x23c, 0x23d, 0x23e, 0x23f,
 0x240, 0x241, 0x242, 0x243, 0x244, 0x245, 0x246, 0x247,
 0x248, 0x249, 0x249, 0x24a, 0x24b, 0x24c, 0x24d, 0x24e,
 0x24f, 0x250, 0x251, 0x252, 0x253, 0x254, 0x255, 0x256,
 0x257, 0x258, 0x259, 0x25a, 0x25b, 0x25c, 0x25d, 0x25e,
 0x25f, 0x260, 0x261, 0x262, 0x262, 0x263, 0x264, 0x265,
 0x266, 0x267, 0x268, 0x269, 0x26a, 0x26b, 0x26c, 0x26d,
 0x26e, 0x26f, 0x270, 0x271, 0x272, 0x273, 0x274, 0x275,
 0x275, 0x276, 0x277, 0x278, 0x279, 0x27a, 0x27b, 0x27c,
 0x27d, 0x27e, 0x27f, 0x280, 0x281, 0x282, 0x283, 0x284,
 0x284, 0x285, 0x286, 0x287, 0x288, 0x289, 0x28a, 0x28b,
 0x28c, 0x28d, 0x28e, 0x28f, 0x290, 0x291, 0x291, 0x292,
 0x293, 0x294, 0x295, 0x296, 0x297, 0x298, 0x299, 0x29a,
 0x29b, 0x29c, 0x29d, 0x29d, 0x29e, 0x29f, 0x2a0, 0x2a1,
 0x2a2, 0x2a3, 0x2a4, 0x2a5, 0x2a6, 0x2a7, 0x2a7, 0x2a8,
 0x2a9, 0x2aa, 0x2ab, 0x2ac, 0x2ad, 0x2ae, 0x2af, 0x2b0,
 0x2b1, 0x2b1, 0x2b2, 0x2b3, 0x2b4, 0x2b5, 0x2b6, 0x2b7,
 0x2b8, 0x2b9, 0x2ba, 0x2ba, 0x2bb, 0x2bc, 0x2bd, 0x2be,
 0x2bf, 0x2c0, 0x2c1, 0x2c2, 0x2c3, 0x2c3, 0x2c4, 0x2c5,
 0x2c6, 0x2c7, 0x2c8, 0x2c9, 0x2ca, 0x2cb, 0x2cb, 0x2cc,
 0x2cd, 0x2ce, 0x2cf, 0x2d0, 0x2d1, 0x2d2, 0x2d3, 0x2d3,
 0x2d4, 0x2d5, 0x2d6, 0x2d7, 0x2d8, 0x2d9, 0x2da, 0x2db,
 0x2db, 0x2dc, 0x2dd, 0x2de, 0x2df, 0x2e0, 0x2e1, 0x2e2,
 0x2e2, 0x2e3, 0x2e4, 0x2e5, 0x2e6, 0x2e7, 0x2e8, 0x2e9,
 0x2e9, 0x2ea, 0x2eb, 0x2ec, 0x2ed, 0x2ee, 0x2ef, 0x2ef,
 0x2f0, 0x2f1, 0x2f2, 0x2f3, 0x2f4, 0x2f5, 0x2f6, 0x2f6,
 0x2f7, 0x2f8, 0x2f9, 0x2fa, 0x2fb, 0x2fc, 0x2fc, 0x2fd,
 0x2fe, 0x2ff, 0x300, 0x301, 0x302, 0x302, 0x303, 0x304,
 0x305, 0x306, 0x307, 0x308, 0x308, 0x309, 0x30a, 0x30b,
 0x30c, 0x30d, 0x30e, 0x30e, 0x30f, 0x310, 0x311, 0x312,
 0x313, 0x313, 0x314, 0x315, 0x316, 0x317, 0x318, 0x319,
 0x319, 0x31a, 0x31b, 0x31c, 0x31d, 0x31e, 0x31e, 0x31f,
 0x320, 0x321, 0x322, 0x323, 0x323, 0x324, 0x325, 0x326,
 0x327, 0x328, 0x328, 0x329, 0x32a, 0x32b, 0x32c, 0x32d,
 0x32d, 0x32e, 0x32f, 0x330, 0x331, 0x332, 0x332, 0x333,
 0x334, 0x335, 0x336, 0x337, 0x337, 0x338, 0x339, 0x33a,
 0x33b, 0x33c, 0x33c, 0x33d, 0x33e, 0x33f, 0x340, 0x340,
 0x341, 0x342, 0x343, 0x344, 0x345, 0x345, 0x346, 0x347,
 0x348, 0x349, 0x349, 0x34a, 0x34b, 0x34c, 0x34d, 0x34e,
 0x34e, 0x34f, 0x350, 0x351, 0x352, 0x352, 0x353, 0x354,
 0x355, 0x356, 0x356, 0x357, 0x358, 0x359, 0x35a, 0x35b,
 0x35b, 0x35c, 0x35d, 0x35e, 0x35f, 0x35f, 0x360, 0x361,
 0x362, 0x363, 0x363, 0x364, 0x365, 0x366, 0x367, 0x367,
 0x368, 0x369, 0x36a, 0x36b, 0x36b, 0x36c, 0x36d, 0x36e,
 0x36f, 0x36f, 0x370, 0x371, 0x372, 0x373, 0x373, 0x374,
 0x375, 0x376, 0x377, 0x377, 0x378, 0x379, 0x37a, 0x37a,
 0x37b, 0x37c, 0x37d, 0x37e, 0x37e, 0x37f, 0x380, 0x381,
 0x382, 0x382, 0x383, 0x384, 0x385, 0x385, 0x386, 0x387,
 0x388, 0x389, 0x389, 0x38a, 0x38b, 0x38c, 0x38d, 0x38d,
 0x38e, 0x38f, 0x390, 0x390, 0x391, 0x392, 0x393, 0x394,
 0x394, 0x395, 0x396, 0x397, 0x397, 0x398, 0x399, 0x39a,
 0x39a, 0x39b, 0x39c, 0x39d, 0x39e, 0x39e, 0x39f, 0x3a0,
 0x3a1, 0x3a1, 0x3a2, 0x3a3, 0x3a4, 0x3a4, 0x3a5, 0x3a6,
 0x3a7, 0x3a8, 0x3a8, 0x3a9, 0x3aa, 0x3ab, 0x3ab, 0x3ac,
 0x3ad, 0x3ae, 0x3ae, 0x3af, 0x3b0, 0x3b1, 0x3b1, 0x3b2,
 0x3b3, 0x3b4, 0x3b5, 0x3b5, 0x3b6, 0x3b7, 0x3b8, 0x3b8,
 0x3b9, 0x3ba, 0x3bb, 0x3bb, 0x3bc, 0x3bd, 0x3be, 0x3be,
 0x3bf, 0x3c0, 0x3c1, 0x3c1, 0x3c2, 0x3c3, 0x3c4, 0x3c4,
 0x3c5, 0x3c6, 0x3c7, 0x3c7, 0x3c8, 0x3c9, 0x3ca, 0x3ca,
 0x3cb, 0x3cc, 0x3cd, 0x3cd, 0x3ce, 0x3cf, 0x3d0, 0x3d0,
 0x3d1, 0x3d2, 0x3d3, 0x3d3, 0x3d4, 0x3d5, 0x3d6, 0x3d6,
 0x3d7, 0x3d8, 0x3d9, 0x3d9, 0x3da, 0x3db, 0x3db, 0x3dc,
 0x3dd, 0x3de, 0x3de, 0x3df, 0x3e0, 0x3e1, 0x3e1, 0x3e2,
 0x3e3, 0x3e4, 0x3e4, 0x3e5, 0x3e6, 0x3e7, 0x3e7, 0x3e8,
 0x3e9, 0x3e9, 0x3ea, 0x3eb, 0x3ec, 0x3ec, 0x3ed, 0x3ee,
 0x3ef, 0x3ef, 0x3f0, 0x3f1, 0x3f2, 0x3f2, 0x3f3, 0x3f4,
 0x3f4, 0x3f5, 0x3f6, 0x3f7, 0x3f7, 0x3f8, 0x3f9, 0x3f9,
 0x3fa, 0x3fb, 0x3fc, 0x3fc, 0x3fd, 0x3fe, 0x3ff, 0x3ff,
};

static int pp_int_log2(uint32_t n)
{
    int l, r;
    if (n == 0) return 0;
    l = pp_clz32(n);
    n = (n << l) >> (31 - FRAC_BITS);
    r = log2_table[n - FRAC_ONE] + ((31 - l) << FRAC_BITS);
    return r;
}

static int64_t pp_int_log2_frac(uint32_t n, uint32_t d)
{
    return pp_int_log2(n) - pp_int_log2(d);
}

/* -------------------------------------------------------------------------
 * Frequency counting
 * ---------------------------------------------------------------------- */

static void compute_word_freq(WordList *s, uint32_t *char_freq,
                               const DataSymbol *buf, size_t buf_size)
{
    for (int i = 0; i < (int)s->word_count; i++)
        s->words[i].freq = 0;
    for (int i = 0; i < NS; i++)
        char_freq[i] = 0;
    for (size_t i = 0; i < buf_size; i++) {
        uint32_t c = buf[i];
        if (c >= NS)
            s->words[c - NS].freq++;
        else
            char_freq[c]++;
    }
}

static int64_t get_n_bits(int c, WordList *s,
                           const uint32_t *char_freq, uint32_t tot_freq)
{
    if (c < NS)
        return -pp_int_log2_frac(char_freq[c], tot_freq);
    Word *p = &s->words[c - NS];
    if (p->freq == 0)
        return get_n_bits((int)p->buf[0], s, char_freq, tot_freq) +
               get_n_bits((int)p->buf[1], s, char_freq, tot_freq);
    return -pp_int_log2_frac(p->freq, tot_freq);
}

#define SUBST_COST         (int)(7.0 * FRAC_ONE + 0.5)
#define TOT_FREQ_RED_BITS  (int)(1.3 * FRAC_ONE + 0.5)

static int64_t compute_score(const Word *p, WordList *cw,
                              const uint32_t *char_freq, uint32_t tot_freq)
{
    if (p->freq <= 1) return -1;
    int64_t old_bits = (get_n_bits((int)p->buf[0], cw, char_freq, tot_freq) +
                        get_n_bits((int)p->buf[1], cw, char_freq, tot_freq)) * p->freq;
    int64_t new_bits = (-pp_int_log2_frac(p->freq, tot_freq) + TOT_FREQ_RED_BITS) *
                       (int64_t)p->freq + SUBST_COST;
    return old_bits - new_bits;
}

/* -------------------------------------------------------------------------
 * Word selection (choose best non-overlapping bigrams)
 * ---------------------------------------------------------------------- */

static int word_score_cmp(const void *a1, const void *a2)
{
    const Word *p1 = (const Word *)a1;
    const Word *p2 = (const Word *)a2;
    if (p1->score > p2->score) return -1;
    if (p1->score == p2->score) return 0;
    return 1;
}

#define MAX_WORDS_PER_ITER 100

static int select_best_words(WordList *s, int n, WordList *cw,
                              const uint32_t *char_freq, uint32_t tot_freq,
                              int min_word_freq)
{
    for (int i = 0; i < (int)s->word_count; i++) {
        Word *p = &s->words[i];
        p->score = compute_score(p, cw, char_freq, tot_freq);
    }
    qsort(s->words, s->word_count, sizeof(s->words[0]), word_score_cmp);

    uint8_t *bm_start = (uint8_t *)pp_mallocz(NS + cw->word_count);
    uint8_t *bm_end   = (uint8_t *)pp_mallocz(NS + cw->word_count);
    int j = 0;
    for (int i = 0; i < (int)s->word_count; i++) {
        Word *p = &s->words[i];
        if (p->score <= 0 || p->freq < (uint32_t)min_word_freq) break;
        if (bm_end[p->buf[0]] || bm_start[p->buf[1]]) continue;
        bm_start[p->buf[0]] = 1;
        bm_end[p->buf[1]]   = 1;
        word_find_add(cw, p->buf, 2, TRUE);
        if (++j >= n) break;
    }
    free(bm_start);
    free(bm_end);
    return j;
}

/* Remove low-freq words from the word list and compact the buffer */
static int update_word_freq(WordList *s, uint32_t *char_freq,
                             DataSymbol **pbuf, size_t *pbuf_size,
                             int min_word_freq)
{
    compute_word_freq(s, char_freq, *pbuf, *pbuf_size);

    int word_count = 0;
    for (int i = 0; i < (int)s->word_count; i++) {
        Word *p = &s->words[i];
        if (p->freq >= (uint32_t)min_word_freq) {
            word_count++;
        } else {
            p->freq = 0;  /* mark deleted */
        }
    }
    if (word_count == (int)s->word_count) return word_count;

    /* compact the buffer, replacing deleted words with their component symbols */
    DataSymbol *obuf   = (DataSymbol *)malloc(sizeof(obuf[0]) * (*pbuf_size));
    size_t       obuf_sz = *pbuf_size;
    size_t       buf_pos = 0;
    for (size_t i = 0; i < *pbuf_size; i++)
        out_word(&obuf, &obuf_sz, &buf_pos, s, (*pbuf)[i]);
    free(*pbuf);

    compute_word_freq(s, char_freq, obuf, buf_pos);
    *pbuf      = obuf;
    *pbuf_size = buf_pos;
    return word_count;
}

/* -------------------------------------------------------------------------
 * Dictionary serialization (text format: one word per line)
 * ---------------------------------------------------------------------- */

/* Expand word code c to its raw bytes in dst[0..], return count of bytes written.
   dst_size is capacity; returns 0 if buffer full. */
static size_t expand_word_bytes(uint8_t *dst, size_t dst_cap, size_t dst_pos,
                                 WordList *s, uint32_t c)
{
    if (c < NS) {
        if (dst_pos < dst_cap) {
            dst[dst_pos++] = (uint8_t)c;
        }
        return dst_pos;
    }
    Word *p = &s->words[c - NS];
    dst_pos = expand_word_bytes(dst, dst_cap, dst_pos, s, p->buf[0]);
    dst_pos = expand_word_bytes(dst, dst_cap, dst_pos, s, p->buf[1]);
    return dst_pos;
}

/* Serialise dictionary words into a malloc'd byte buffer (text format).
 * word_tab[i] gives the global code (< NS → byte, >= NS → word index).
 * Only codes >= NS (actual words) are serialised; each word is on its own line.
 * '\n' in word → "\\n", '\\' → "\\\\"
 * Returns malloc'd buffer and sets *out_len. */
static uint8_t *dict_serialize(WordList *s, const uint32_t *word_tab,
                                int word_count, size_t *out_len)
{
    /* worst case: each byte → 2 chars, newline, * word_count words * max_word_bytes */
    const int MAX_WORD_BYTES = 512;
    size_t cap = (size_t)word_count * (MAX_WORD_BYTES * 2 + 2);
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return NULL;

    size_t pos = 0;
    uint8_t tmp[MAX_WORD_BYTES];
    for (int i = 0; i < word_count; i++) {
        uint32_t code = word_tab[i];
        if (code < NS) {
            /* single-byte symbol: just emit as literal line */
            if (code == '\n') {
                if (pos + 3 > cap) { free(buf); return NULL; }
                buf[pos++] = '\\'; buf[pos++] = 'n'; buf[pos++] = '\n';
            } else if (code == '\\') {
                if (pos + 3 > cap) { free(buf); return NULL; }
                buf[pos++] = '\\'; buf[pos++] = '\\'; buf[pos++] = '\n';
            } else {
                if (pos + 2 > cap) { free(buf); return NULL; }
                buf[pos++] = (uint8_t)code; buf[pos++] = '\n';
            }
        } else {
            /* dictionary word: expand and escape */
            size_t n = expand_word_bytes(tmp, MAX_WORD_BYTES, 0, s, code);
            for (size_t j = 0; j < n; j++) {
                if (tmp[j] == '\n') {
                    if (pos + 2 > cap) { free(buf); return NULL; }
                    buf[pos++] = '\\'; buf[pos++] = 'n';
                } else if (tmp[j] == '\\') {
                    if (pos + 2 > cap) { free(buf); return NULL; }
                    buf[pos++] = '\\'; buf[pos++] = '\\';
                } else {
                    if (pos + 1 > cap) { free(buf); return NULL; }
                    buf[pos++] = tmp[j];
                }
            }
            if (pos + 1 > cap) { free(buf); return NULL; }
            buf[pos++] = '\n';
        }
    }
    *out_len = pos;
    return buf;
}

/* -------------------------------------------------------------------------
 * StringTable (for decoder word list)
 * ---------------------------------------------------------------------- */

typedef struct {
    uint32_t len;
    uint8_t  data[1];  /* flexible: allocated with extra space */
} StrEntry;

typedef struct {
    StrEntry **tab;
    size_t     count;
    size_t     cap;
} StrTable;

static StrTable *strtable_init(void)
{
    return (StrTable *)pp_mallocz(sizeof(StrTable));
}

static int strtable_add(StrTable *st, const uint8_t *data, uint32_t len)
{
    if (st->count >= st->cap) {
        size_t new_cap = st->cap ? st->cap * 2 : 64;
        StrEntry **new_tab = (StrEntry **)realloc(st->tab, new_cap * sizeof(StrEntry *));
        if (!new_tab) return -1;
        st->tab = new_tab;
        st->cap = new_cap;
    }
    StrEntry *e = (StrEntry *)malloc(sizeof(StrEntry) + len);
    if (!e) return -1;
    e->len = len;
    memcpy(e->data, data, len);
    st->tab[st->count++] = e;
    return 0;
}

static void strtable_free(StrTable *st)
{
    for (size_t i = 0; i < st->count; i++) free(st->tab[i]);
    free(st->tab);
    free(st);
}

/* Parse the serialised dictionary text and build a StrTable.
 * One entry per line; "\\n" → '\n', "\\\\" → '\\'. */
static StrTable *dict_deserialise(const uint8_t *data, size_t len)
{
    StrTable *st = strtable_init();
    if (!st) return NULL;

    uint8_t  buf[4096];
    uint32_t blen = 0;
    size_t   i = 0;
    while (i < len) {
        int c = data[i++];
        if (c == '\n') {
            if (blen > 0) {
                if (strtable_add(st, buf, blen) < 0) {
                    strtable_free(st);
                    return NULL;
                }
            }
            blen = 0;
        } else if (c == '\\' && i < len) {
            int nc = data[i++];
            if (nc == 'n')  { buf[blen++] = '\n'; }
            else            { buf[blen++] = (uint8_t)nc; }
        } else {
            buf[blen++] = (uint8_t)c;
        }
        if (blen >= sizeof(buf) - 4) {
            /* word too long — return error */
            strtable_free(st);
            return NULL;
        }
    }
    if (blen > 0) {
        if (strtable_add(st, buf, blen) < 0) {
            strtable_free(st); return NULL;
        }
    }
    return st;
}

/* -------------------------------------------------------------------------
 * Sort words (lexicographic) and build conversion table
 * ---------------------------------------------------------------------- */

typedef struct {
    WordList *s;
    uint32_t *char_freq;
} SortState;

static int word_lex_cmp(const void *a1, const void *a2, void *arg)
{
    SortState *ss = (SortState *)arg;
    uint32_t c1 = *(DataSymbol *)a1;
    uint32_t c2 = *(DataSymbol *)a2;
    uint8_t buf1[512], buf2[512];
    size_t n1 = expand_word_bytes(buf1, sizeof(buf1), 0, ss->s, c1);
    size_t n2 = expand_word_bytes(buf2, sizeof(buf2), 0, ss->s, c2);
    int res = memcmp(buf1, buf2, n1 < n2 ? n1 : n2);
    if (res) return res;
    return (n1 < n2) ? -1 : (n1 == n2) ? 0 : 1;
}

/* qsort_r wrapper for portability */
static SortState *g_sort_state;
static int word_lex_cmp_compat(const void *a, const void *b)
{
    return word_lex_cmp(a, b, g_sort_state);
}

/* Build word_tab in lexicographic order.
 * Returns total entries (NS + word_count words that have freq > 0). */
static int build_word_tab(WordList *s, uint32_t **ptab)
{
    uint32_t n_total = (uint32_t)(NS + s->word_count);
    uint32_t *tab = (uint32_t *)malloc(sizeof(tab[0]) * n_total);
    if (!tab) return -1;

    int j = 0;
    for (int i = 0; i < NS; i++)
        tab[j++] = (uint32_t)i;
    for (int i = 0; i < (int)s->word_count; i++) {
        if (s->words[i].freq > 0)
            tab[j++] = (uint32_t)(NS + i);
    }
    /* sort only the word entries (i >= NS) lexicographically */
    SortState ss = { s, NULL };
    g_sort_state = &ss;
    qsort(tab + NS, (size_t)(j - NS), sizeof(tab[0]), word_lex_cmp_compat);
    *ptab = tab;
    return j;
}

/* -------------------------------------------------------------------------
 * Public API: nncp_preprocess_encode
 * ---------------------------------------------------------------------- */

int nncp_preprocess_encode(
    const uint8_t *input, size_t input_len,
    uint16_t **out_symbols, size_t *out_n_sym,
    uint8_t **out_dict, size_t *out_dict_len,
    int n_words, int min_freq)
{
    if (!input || !out_symbols || !out_n_sym || !out_dict || !out_dict_len)
        return -1;

    /* 1. Convert input bytes to DataSymbol (trivial uint16_t cast) */
    DataSymbol *buf = (DataSymbol *)malloc(input_len * sizeof(DataSymbol));
    if (!buf) return -1;
    for (size_t i = 0; i < input_len; i++) buf[i] = (DataSymbol)input[i];

    /* 2. Case/space encoding */
    size_t buf_len = 0;
    DataSymbol *buf2 = case_space_encoding(&buf_len, buf, input_len);
    free(buf);
    if (!buf2) return -1;
    buf = buf2;

    /* 3. Bigram merging */
    WordList *s = word_list_init();
    uint32_t *char_freq = (uint32_t *)pp_mallocz(sizeof(uint32_t) * NS);
    if (!char_freq) { free(buf); word_list_end(s); return -1; }

    compute_word_freq(s, char_freq, buf, buf_len);

    int max_new_words = n_words - NS;
    if (max_new_words < 0) max_new_words = 0;

    int word_count = 0, word_count_prev = 0;
    for (word_count = 0; word_count < max_new_words; ) {
        if (buf_len < 2) break;

        WordList *ws = word_list_init();
        for (size_t i = 0; i < buf_len - 1; i++) {
            Word *p = word_find_add(ws, buf + i, 2, TRUE);
            p->freq++;
        }
        int n = select_best_words(ws,
                                  pp_min_int(MAX_WORDS_PER_ITER, max_new_words - word_count),
                                  s, char_freq, (uint32_t)buf_len, min_freq);
        word_list_end(ws);
        if (n == 0) break;

        /* substitute bigrams in buffer */
        size_t j = 0;
        for (size_t i = 0; i < buf_len; ) {
            if (i + 1 < buf_len) {
                Word *p = word_find_add(s, buf + i, 2, FALSE);
                if (p) {
                    buf[j++] = (DataSymbol)(NS + (p - s->words));
                    i += 2;
                    continue;
                }
            }
            buf[j++] = buf[i++];
        }
        buf_len = j;

        word_count_prev = word_count;
        word_count = update_word_freq(s, char_freq, &buf, &buf_len, min_freq);
        if (word_count >= max_new_words || word_count == word_count_prev) break;
    }

    /* 4. Build sorted word table */
    uint32_t *word_tab = NULL;
    int total_entries = build_word_tab(s, &word_tab);
    if (total_entries < 0) {
        free(buf); free(char_freq); word_list_end(s); return -1;
    }
    int actual_word_count = total_entries - NS; /* words added */

    /* 5. Build convert_table: old_code → new_index */
    int n_old = NS + (int)s->word_count;
    uint32_t *conv = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)n_old);
    if (!conv) { free(buf); free(char_freq); free(word_tab); word_list_end(s); return -1; }
    for (int i = 0; i < n_old; i++) conv[i] = (uint32_t)-1;
    for (int i = 0; i < total_entries; i++)
        conv[word_tab[i]] = (uint32_t)i;

    /* 6. Convert buf (uses old codes) to final symbols via conv */
    uint16_t *out = (uint16_t *)malloc(buf_len * sizeof(uint16_t));
    if (!out) {
        free(buf); free(char_freq); free(word_tab); free(conv); word_list_end(s); return -1;
    }
    /* buf may still contain multi-symbol words; expand them via conv */
    /* We need to expand each buf[i] to its final symbol(s) */
    size_t out_pos = 0;
    for (size_t i = 0; i < buf_len; i++) {
        uint32_t c = buf[i];
        uint32_t idx = conv[c];
        if (idx != (uint32_t)-1) {
            /* resize if needed */
            if (out_pos >= buf_len) {
                uint16_t *tmp = (uint16_t *)realloc(out, (out_pos + 1) * sizeof(uint16_t));
                if (!tmp) { free(out); free(buf); free(char_freq); free(word_tab); free(conv); word_list_end(s); return -1; }
                out = tmp;
            }
            out[out_pos++] = (uint16_t)idx;
        } else {
            /* symbol was deleted (freq < min_freq); expand to component symbols */
            /* simplified: expand recursively */
            /* We'll just write directly using a small stack */
            uint32_t stack[64];
            int sp = 0;
            stack[sp++] = c;
            while (sp > 0) {
                uint32_t s2 = stack[--sp];
                if (s2 < NS) {
                    if (conv[s2] != (uint32_t)-1) {
                        if (out_pos >= buf_len) {
                            uint16_t *tmp = (uint16_t *)realloc(out, (out_pos + 1) * sizeof(uint16_t));
                            if (!tmp) { free(out); free(buf); free(char_freq); free(word_tab); free(conv); word_list_end(s); return -1; }
                            out = tmp;
                        }
                        out[out_pos++] = (uint16_t)conv[s2];
                    }
                } else {
                    Word *wp = &s->words[s2 - NS];
                    /* push children in reverse order */
                    if (sp + 2 <= 64) {
                        stack[sp++] = wp->buf[1];
                        stack[sp++] = wp->buf[0];
                    }
                }
            }
        }
    }

    /* 7. Serialise dictionary (only the word entries, NS onward) */
    size_t dict_len = 0;
    uint8_t *dict_buf = dict_serialize(s, word_tab + NS, actual_word_count, &dict_len);
    if (!dict_buf) {
        free(out); free(buf); free(char_freq); free(word_tab); free(conv);
        word_list_end(s); return -1;
    }

    free(buf);
    free(char_freq);
    free(word_tab);
    free(conv);
    word_list_end(s);

    *out_symbols  = out;
    *out_n_sym    = out_pos;
    *out_dict     = dict_buf;
    *out_dict_len = dict_len;
    return actual_word_count;
}

/* -------------------------------------------------------------------------
 * Public API: nncp_preprocess_decode
 * ---------------------------------------------------------------------- */

int nncp_preprocess_decode(
    const uint16_t *symbols, size_t n_symbols,
    uint8_t **out_bytes, size_t *out_len,
    const uint8_t *dict_data, size_t dict_len)
{
    if (!symbols || !out_bytes || !out_len || !dict_data) return -1;

    /* Parse dictionary */
    StrTable *st = dict_deserialise(dict_data, dict_len);
    if (!st) return -1;

    /* Total vocab = NS (bytes 0-255) + st->count (dictionary words) */
    size_t vocab = NS + st->count;

    /* Build a flat "string table" indexed by symbol:
     * symbol 0..255 → just that byte
     * symbol 256..vocab-1 → dictionary entry from st */

    /* Decode symbol stream → raw bytes through case_space_decode */
    /* Allocate output: upper bound = n_symbols * max_word_len */
    size_t out_cap = n_symbols * 4 + 4096;
    uint8_t *out = (uint8_t *)malloc(out_cap);
    if (!out) { strtable_free(st); return -1; }

    CaseSpaceDecodeState cs;
    case_space_decode_init(&cs);
    size_t out_pos = 0;

    for (size_t i = 0; i < n_symbols; i++) {
        uint16_t sym = symbols[i];
        if (sym >= vocab) {
            /* invalid symbol */
            free(out); strtable_free(st); return -1;
        }

        /* Expand this symbol to its raw bytes, then case_space_decode each */
        const uint8_t *bytes;
        uint32_t blen;
        uint8_t single_byte[1];

        if (sym < NS) {
            single_byte[0] = (uint8_t)sym;
            bytes = single_byte;
            blen  = 1;
        } else {
            StrEntry *e = st->tab[sym - NS];
            bytes = e->data;
            blen  = e->len;
        }

        for (uint32_t j = 0; j < blen; j++) {
            int b = case_space_decode(&cs, bytes[j]);
            if (b >= 0) {
                if (out_pos >= out_cap) {
                    out_cap *= 2;
                    uint8_t *tmp = (uint8_t *)realloc(out, out_cap);
                    if (!tmp) { free(out); strtable_free(st); return -1; }
                    out = tmp;
                }
                out[out_pos++] = (uint8_t)b;
            }
        }
    }

    strtable_free(st);
    *out_bytes = out;
    *out_len   = out_pos;
    return 0;
}
