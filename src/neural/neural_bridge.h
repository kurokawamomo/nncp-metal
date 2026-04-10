/**
 * @file neural_bridge.h
 * @brief Neural compression bridge interface
 * 
 * This bridge connects the Phase 2B neural compression engines
 * (Transformer and LSTM) to the main compression pipeline through
 * the integration layer.
 */

#ifndef NEURAL_BRIDGE_H
#define NEURAL_BRIDGE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Neural compression algorithm types
 */
typedef enum {
    NEURAL_ALGORITHM_TRANSFORMER = 0,    ///< Transformer-based compression
    NEURAL_ALGORITHM_LSTM,              ///< LSTM-based compression
    NEURAL_ALGORITHM_AUTO               ///< Automatic selection
} NeuralAlgorithm;

/**
 * @brief Neural compression result
 */
typedef struct {
    bool success;                       ///< Whether compression succeeded
    size_t compressed_size;             ///< Size of compressed data
    NeuralAlgorithm algorithm_used;     ///< Algorithm that was used
    float compression_ratio;            ///< Achieved compression ratio
    uint64_t processing_time_ns;        ///< Processing time in nanoseconds
    uint64_t memory_used_bytes;         ///< Peak memory usage
    char error_message[256];            ///< Error message if failed
} NeuralCompressionResult;

/**
 * @brief Neural decompression result
 */
typedef struct {
    bool success;                       ///< Whether decompression succeeded
    size_t decompressed_size;           ///< Size of decompressed data
    NeuralAlgorithm algorithm_detected; ///< Algorithm detected from data
    uint64_t processing_time_ns;        ///< Processing time in nanoseconds
    char error_message[256];            ///< Error message if failed
} NeuralDecompressionResult;

/**
 * @brief Neural compression configuration
 */
typedef struct {
    NeuralAlgorithm preferred_algorithm; ///< Preferred algorithm
    size_t memory_limit_bytes;          ///< Memory usage limit
    uint32_t quality_level;             ///< Quality level (1-10, higher = better quality)
    bool enable_gpu_acceleration;       ///< Enable Metal GPU acceleration
    bool verbose_logging;               ///< Enable verbose logging
    float compression_target;           ///< Target compression ratio (0.0-1.0)
} NeuralCompressionConfig;

/**
 * @brief Initialize the neural bridge system
 * @param config Configuration for neural compression
 * @return true if initialization succeeded, false otherwise
 */
bool neural_bridge_init(const NeuralCompressionConfig* config);

/**
 * @brief Shutdown the neural bridge and free resources
 */
void neural_bridge_shutdown(void);

/**
 * @brief Check if neural bridge is initialized and ready
 * @return true if ready, false otherwise
 */
bool neural_bridge_is_ready(void);

/**
 * @brief Compress data using Transformer neural compression
 * @param input_data Input data buffer
 * @param input_size Size of input data
 * @param output_data Output buffer (must be pre-allocated)
 * @param output_capacity Size of output buffer
 * @param config Compression configuration
 * @param result Compression result structure
 * @return true if compression succeeded, false otherwise
 */
bool neural_bridge_transformer_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const NeuralCompressionConfig* config,
    NeuralCompressionResult* result
);

/**
 * @brief Decompress data using Transformer neural decompression
 * @param input_data Compressed input data
 * @param input_size Size of compressed data
 * @param output_data Output buffer (must be pre-allocated)
 * @param output_capacity Size of output buffer
 * @param result Decompression result structure
 * @return true if decompression succeeded, false otherwise
 */
bool neural_bridge_transformer_decompress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    NeuralDecompressionResult* result
);

/**
 * @brief Compress data using LSTM neural compression
 * @param input_data Input data buffer
 * @param input_size Size of input data
 * @param output_data Output buffer (must be pre-allocated)
 * @param output_capacity Size of output buffer
 * @param config Compression configuration
 * @param result Compression result structure
 * @return true if compression succeeded, false otherwise
 */
bool neural_bridge_lstm_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const NeuralCompressionConfig* config,
    NeuralCompressionResult* result
);

/**
 * @brief Decompress data using LSTM neural decompression
 * @param input_data Compressed input data
 * @param input_size Size of compressed data
 * @param output_data Output buffer (must be pre-allocated)
 * @param output_capacity Size of output buffer
 * @param result Decompression result structure
 * @return true if decompression succeeded, false otherwise
 */
bool neural_bridge_lstm_decompress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    NeuralDecompressionResult* result
);

/**
 * @brief Estimate compressed size for a given algorithm
 * @param input_size Size of input data
 * @param algorithm Target neural algorithm
 * @return Estimated compressed size in bytes
 */
size_t neural_bridge_estimate_compressed_size(
    size_t input_size,
    NeuralAlgorithm algorithm
);

/**
 * @brief Check if a specific neural algorithm is available
 * @param algorithm Algorithm to check
 * @return true if available, false otherwise
 */
bool neural_bridge_algorithm_available(NeuralAlgorithm algorithm);

/**
 * @brief Get human-readable name for neural algorithm
 * @param algorithm Neural algorithm enum
 * @return String name of the algorithm
 */
const char* neural_bridge_algorithm_name(NeuralAlgorithm algorithm);

/**
 * @brief Get memory requirements for neural compression
 * @param input_size Size of input data
 * @param algorithm Target algorithm
 * @return Estimated memory requirements in bytes
 */
size_t neural_bridge_memory_requirements(
    size_t input_size,
    NeuralAlgorithm algorithm
);

// Cross-Algorithm Validation Functions

/**
 * @brief Validate compression consistency between Transformer and LSTM
 * @param test_data Input test data
 * @param test_size Size of test data
 * @param config Neural compression configuration
 * @param transformer_result Output Transformer result
 * @param lstm_result Output LSTM result
 * @return true if both algorithms maintain consistency, false otherwise
 */
bool neural_bridge_validate_cross_algorithm_consistency(
    const uint8_t* test_data,
    size_t test_size,
    const NeuralCompressionConfig* config,
    NeuralCompressionResult* transformer_result,
    NeuralCompressionResult* lstm_result
);

/**
 * @brief Perform regression testing on both algorithms
 * @param test_datasets Array of test datasets
 * @param num_datasets Number of test datasets
 * @param config Neural compression configuration
 * @param regression_results Output regression test results
 * @return true if regression tests pass, false otherwise
 */
bool neural_bridge_perform_regression_testing(
    const void* test_datasets,
    size_t num_datasets,
    const NeuralCompressionConfig* config,
    void* regression_results
);

/**
 * @brief Validate lossless integrity across both algorithms
 * @param test_data Input test data
 * @param test_size Size of test data
 * @param config Neural compression configuration
 * @return true if both algorithms maintain lossless integrity, false otherwise
 */
bool neural_bridge_validate_lossless_integrity(
    const uint8_t* test_data,
    size_t test_size,
    const NeuralCompressionConfig* config
);

/**
 * @brief Generate comprehensive validation report
 * @param test_results Array of test results
 * @param num_results Number of results
 * @param report_buffer Output buffer for validation report
 * @param buffer_size Size of report buffer
 * @return true if report generated successfully, false otherwise
 */
bool neural_bridge_generate_validation_report(
    const void* test_results,
    size_t num_results,
    char* report_buffer,
    size_t buffer_size
);

/**
 * @brief Lossless NNCP Transformer compression (authentic CUDA port)
 * @param input_data Input data buffer
 * @param input_size Size of input data
 * @param output_data Output buffer (must be pre-allocated)
 * @param output_capacity Size of output buffer
 * @param config Compression configuration
 * @return Size of compressed data, or 0 if failed
 */
size_t neural_bridge_lossless_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const NeuralCompressionConfig* config
);

/**
 * @brief Lossless NNCP Transformer decompression (authentic CUDA port)
 * @param input_data Compressed input data
 * @param input_size Size of compressed data
 * @param output_data Output buffer (must be pre-allocated)
 * @param output_capacity Size of output buffer
 * @return Size of decompressed data, or 0 if failed
 */
size_t neural_bridge_lossless_decompress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity
);

/** Global LR override set by --lr CLI flag. 0.0 = use default (1e-4). */
extern float g_lr_override;

/**
 * @brief Global vocab-size override for preprocessing mode.
 * Set to 256+n_words BEFORE the first call to compress/decompress.
 * 0 = use default (256).
 */
extern int g_vocab_size_override;

/**
 * @brief Runtime model profile: all parameters must be set BEFORE the first
 *        call to compress/decompress (i.e., before model/graph creation).
 *        Default values match the original NNCP "default" profile.
 */
typedef struct {
    int h;           /* hidden_size           (default: 256)  */
    int l;           /* num_layers            (default:   4)  */
    int f;           /* feed_forward_size     (default: 512)  */
    int nh;          /* num_attention_heads   (default:   8)  */
    int mem_len;     /* MEM_LEN / kv_memory_len (default: 32)  */
    int num_streams; /* NUM_STREAMS           (default:  16)  */
    int seg_len;     /* SEG_LEN               (default:  32)  */
    int d_pos;       /* rel-PE dim for W_rel_r (default: 32, enwik8: 320) */
} NNCPProfileConfig;

extern NNCPProfileConfig g_nncp_profile;

/**
 * @brief Compress uint16_t token stream (preprocessing mode).
 *
 * tokens      : symbol stream (values 0..vocab_size-1)
 * n_tokens    : number of tokens
 * output_data : caller-allocated output buffer
 * output_cap  : capacity of output_data
 * vocab_size  : effective vocabulary size (must match model creation)
 * total_input_bytes : original raw file size (for LR schedule)
 *
 * Returns compressed byte count, or 0 on error.
 */
size_t neural_bridge_compress_symbols(
    const uint16_t *tokens,
    size_t          n_tokens,
    uint8_t        *output_data,
    size_t          output_cap,
    int             vocab_size,
    size_t          total_input_bytes);

/**
 * @brief Decompress to uint16_t token stream (preprocessing mode).
 *
 * input_data  : compressed bytes
 * input_size  : size of input_data
 * tokens_out  : caller-allocated uint16_t buffer (capacity = max_tokens)
 * max_tokens  : capacity of tokens_out in symbols
 * vocab_size  : effective vocabulary size (must match compression)
 *
 * Returns number of tokens written, or 0 on error.
 */
size_t neural_bridge_decompress_symbols(
    const uint8_t  *input_data,
    size_t          input_size,
    uint16_t       *tokens_out,
    size_t          max_tokens,
    int             vocab_size);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_BRIDGE_H