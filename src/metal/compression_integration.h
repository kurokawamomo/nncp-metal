#ifndef COMPRESSION_INTEGRATION_H
#define COMPRESSION_INTEGRATION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compression algorithm types supported by the integration layer
 */
typedef enum {
    COMPRESSION_ALGORITHM_TRANSFORMER = 1, ///< Transformer-based neural compression
    COMPRESSION_ALGORITHM_LSTM,           ///< LSTM-based neural compression
    COMPRESSION_ALGORITHM_AUTO            ///< Automatic selection based on data characteristics
} CompressionAlgorithm;

/**
 * @brief Compression result structure
 */
typedef struct {
    bool success;                        ///< Whether compression succeeded
    size_t compressed_size;              ///< Size of compressed data in bytes
    CompressionAlgorithm algorithm_used; ///< Algorithm that was actually used
    char error_message[256];             ///< Error message if compression failed
    float compression_ratio;             ///< Achieved compression ratio (compressed/original)
    uint64_t processing_time_ns;         ///< Processing time in nanoseconds
    size_t memory_used_bytes;            ///< Peak memory usage during compression
    size_t buffer_allocations;           ///< Number of buffer allocations made
} CompressionResult;;

/**
 * @brief Decompression result structure
 */
typedef struct {
    bool success;                        ///< Whether decompression succeeded
    size_t decompressed_size;            ///< Size of decompressed data in bytes
    CompressionAlgorithm algorithm_detected; ///< Algorithm detected from data
    char error_message[256];             ///< Error message if decompression failed
    uint64_t processing_time_ns;         ///< Processing time in nanoseconds
    size_t memory_used_bytes;            ///< Peak memory usage during decompression
    size_t buffer_allocations;           ///< Number of buffer allocations made
} DecompressionResult;;

/**
 * @brief Compression integration configuration
 */
typedef struct {
    CompressionAlgorithm preferred_algorithm; ///< Preferred algorithm (or AUTO for selection)
    bool enable_fallback;                     ///< Enable fallback to RLE on failure
    bool verbose_logging;                     ///< Enable verbose logging
    size_t memory_limit_bytes;                ///< Memory usage limit (0 = no limit)
    uint32_t quantization_bits;               ///< Quantization bits for lossy compression
} CompressionConfig;

/// Memory pool for optimized buffer allocation
typedef struct {
    uint8_t* buffer;                          ///< Pre-allocated buffer
    size_t size;                              ///< Total buffer size
    size_t used;                              ///< Currently used bytes
    bool in_use;                              ///< Whether buffer is currently allocated
    uint64_t last_used_time;                  ///< Last usage timestamp for eviction
} MemoryPoolBlock;

/// Memory management configuration and statistics
typedef struct {
    MemoryPoolBlock* blocks;                  ///< Array of memory blocks
    size_t num_blocks;                        ///< Number of blocks in pool
    size_t total_allocated;                   ///< Total allocated memory
    size_t peak_usage;                        ///< Peak memory usage
    size_t allocation_count;                  ///< Number of allocations
    size_t deallocation_count;                ///< Number of deallocations
    bool pool_enabled;                        ///< Whether memory pooling is enabled
    size_t block_size;                        ///< Default block size
} MemoryManager;

/**
 * @brief Initialize the compression integration system
 * @param config Configuration for the compression system
 * @return true if initialization succeeded, false otherwise
 */
// Memory management functions
/**
 * Initialize memory management system
 * @param max_blocks Maximum number of memory blocks to allocate
 * @param block_size Default size for each memory block
 * @return true on success, false on failure
 */
bool memory_manager_init(size_t max_blocks, size_t block_size);

/**
 * Shutdown memory management system and free all resources
 */
void memory_manager_shutdown(void);

/**
 * Allocate memory from pool or system
 * @param size Size in bytes to allocate
 * @return Pointer to allocated memory or NULL on failure
 */
void* memory_manager_alloc(size_t size);

/**
 * Free memory back to pool or system
 * @param ptr Pointer to memory to free
 * @param size Size of the allocated block
 */
void memory_manager_free(void* ptr, size_t size);

/**
 * Get current memory usage statistics
 * @return Pointer to memory manager statistics
 */
const MemoryManager* memory_manager_get_stats(void);

/**
 * Force cleanup of unused memory blocks
 * @param max_age_ms Maximum age in milliseconds before block is freed
 */
void memory_manager_cleanup(uint64_t max_age_ms);

bool compression_integration_init(const CompressionConfig* config);

/**
 * @brief Shutdown the compression integration system and free resources
 */
void compression_integration_shutdown(void);

/**
 * @brief Compress data using the integration layer
 * @param input_data Input data buffer
 * @param input_size Size of input data in bytes
 * @param output_data Output buffer (must be pre-allocated)
 * @param output_capacity Size of output buffer
 * @param config Compression configuration
 * @param result Compression result structure
 * @return true if compression succeeded, false otherwise
 */
bool compression_integration_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const CompressionConfig* config,
    CompressionResult* result
);

/**
 * @brief Decompress data using the integration layer
 * @param input_data Compressed input data
 * @param input_size Size of compressed data in bytes
 * @param output_data Output buffer (must be pre-allocated)
 * @param output_capacity Size of output buffer
 * @param result Decompression result structure
 * @return true if decompression succeeded, false otherwise
 */
bool compression_integration_decompress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    DecompressionResult* result
);

/**
 * @brief Estimate the optimal output buffer size for compression
 * @param input_size Size of input data
 * @param algorithm Target compression algorithm
 * @return Recommended output buffer size in bytes
 */
size_t compression_integration_estimate_output_size(
    size_t input_size,
    CompressionAlgorithm algorithm
);

/**
 * @brief Get algorithm selection based on data characteristics
 * @param input_data Input data for analysis
 * @param input_size Size of input data
 * @return Recommended compression algorithm
 */
CompressionAlgorithm compression_integration_select_algorithm(
    const uint8_t* input_data,
    size_t input_size
);

/**
 * @brief Get human-readable name for compression algorithm
 * @param algorithm Compression algorithm enum
 * @return String name of the algorithm
 */
const char* compression_integration_algorithm_name(CompressionAlgorithm algorithm);

/**
 * @brief Check if a specific algorithm is available
 * @param algorithm Algorithm to check
 * @return true if algorithm is available, false otherwise
 */
bool compression_integration_algorithm_available(CompressionAlgorithm algorithm);

#ifdef __cplusplus
}
#endif

#endif // COMPRESSION_INTEGRATION_H
