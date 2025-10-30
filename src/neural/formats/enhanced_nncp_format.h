#ifndef ENHANCED_NNCP_FORMAT_H
#define ENHANCED_NNCP_FORMAT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Magic numbers and version information
#define ENNCP_MAGIC_NUMBER          0x4E4E4350  // "NNCP" in big endian
#define ENNCP_VERSION_MAJOR         2
#define ENNCP_VERSION_MINOR         1
#define ENNCP_VERSION_PATCH         0
#define ENNCP_MIN_SUPPORTED_VERSION 0x02000000  // Major.Minor.Patch format

// Maximum sizes and limits
#define ENNCP_MAX_METADATA_SIZE     (64 * 1024)    // 64KB metadata
#define ENNCP_MAX_CUSTOM_FIELDS     256             // Maximum custom metadata fields
#define ENNCP_MAX_FIELD_NAME_LEN    64              // Maximum field name length
#define ENNCP_MAX_FIELD_VALUE_LEN   1024            // Maximum field value length
#define ENNCP_MAX_COMPRESSION_STAGES 8              // Maximum compression stages
#define ENNCP_MAX_QUALITY_LEVELS    16              // Maximum quality levels

// Error codes for enhanced NNCP format operations
typedef enum {
    ENNCP_SUCCESS = 0,
    ENNCP_ERROR_INVALID_PARAM,
    ENNCP_ERROR_MEMORY_ALLOCATION,
    ENNCP_ERROR_FILE_IO,
    ENNCP_ERROR_INVALID_FORMAT,
    ENNCP_ERROR_UNSUPPORTED_VERSION,
    ENNCP_ERROR_CORRUPTED_DATA,
    ENNCP_ERROR_CHECKSUM_MISMATCH,
    ENNCP_ERROR_METADATA_TOO_LARGE,
    ENNCP_ERROR_COMPRESSION_FAILED,
    ENNCP_ERROR_DECOMPRESSION_FAILED,
    ENNCP_ERROR_INCOMPATIBLE_FORMAT,
    ENNCP_ERROR_FIELD_NOT_FOUND,
    ENNCP_ERROR_FIELD_LIMIT_EXCEEDED,
    ENNCP_ERROR_INVALID_ENCRYPTION,
    ENNCP_ERROR_VERIFICATION_FAILED
} ENNCPError;

// Data types supported by the format
typedef enum {
    ENNCP_DATA_TYPE_UNKNOWN = 0,
    ENNCP_DATA_TYPE_TEXT,
    ENNCP_DATA_TYPE_BINARY,
    ENNCP_DATA_TYPE_IMAGE,
    ENNCP_DATA_TYPE_AUDIO,
    ENNCP_DATA_TYPE_VIDEO,
    ENNCP_DATA_TYPE_DOCUMENT,
    ENNCP_DATA_TYPE_ARCHIVE,
    ENNCP_DATA_TYPE_DATABASE,
    ENNCP_DATA_TYPE_LOG,
    ENNCP_DATA_TYPE_SCIENTIFIC,
    ENNCP_DATA_TYPE_TIME_SERIES,
    ENNCP_DATA_TYPE_NEURAL_NETWORK,
    ENNCP_DATA_TYPE_STRUCTURED,
    ENNCP_DATA_TYPE_SPARSE,
    ENNCP_DATA_TYPE_MIXED
} ENNCPDataType;

// Compression algorithms supported
typedef enum {
    ENNCP_COMPRESSION_NONE = 0,
    ENNCP_COMPRESSION_RLE,
    ENNCP_COMPRESSION_HUFFMAN,
    ENNCP_COMPRESSION_LZ4,
    ENNCP_COMPRESSION_ZSTD,
    ENNCP_COMPRESSION_NEURAL_QUANTIZATION,
    ENNCP_COMPRESSION_NEURAL_LOSSLESS,
    ENNCP_COMPRESSION_TRANSFORMER,
    ENNCP_COMPRESSION_LSTM,
    ENNCP_COMPRESSION_ATTENTION,
    ENNCP_COMPRESSION_HYBRID,
    ENNCP_COMPRESSION_ADAPTIVE,
    ENNCP_COMPRESSION_ENSEMBLE,
    ENNCP_COMPRESSION_CUSTOM
} ENNCPCompressionAlgorithm;

// Encryption algorithms
typedef enum {
    ENNCP_ENCRYPTION_NONE = 0,
    ENNCP_ENCRYPTION_AES128,
    ENNCP_ENCRYPTION_AES256,
    ENNCP_ENCRYPTION_CHACHA20,
    ENNCP_ENCRYPTION_CUSTOM
} ENNCPEncryptionAlgorithm;

// Checksum algorithms
typedef enum {
    ENNCP_CHECKSUM_NONE = 0,
    ENNCP_CHECKSUM_CRC32,
    ENNCP_CHECKSUM_MD5,
    ENNCP_CHECKSUM_SHA1,
    ENNCP_CHECKSUM_SHA256,
    ENNCP_CHECKSUM_SHA512,
    ENNCP_CHECKSUM_BLAKE3,
    ENNCP_CHECKSUM_XXHASH
} ENNCPChecksumAlgorithm;

// Custom metadata field
typedef struct {
    char name[ENNCP_MAX_FIELD_NAME_LEN];        // Field name
    char value[ENNCP_MAX_FIELD_VALUE_LEN];      // Field value
    uint16_t name_length;                       // Actual name length
    uint16_t value_length;                      // Actual value length
    uint32_t flags;                             // Field flags (reserved)
} ENNCPCustomField;

// Compression stage information
typedef struct {
    ENNCPCompressionAlgorithm algorithm;        // Compression algorithm used
    uint32_t original_size;                     // Original size before this stage
    uint32_t compressed_size;                   // Size after this stage
    float compression_ratio;                    // Compression ratio for this stage
    uint32_t compression_time_ms;               // Time taken for compression
    uint32_t quality_level;                     // Quality level used (0-15)
    uint32_t parameters[8];                     // Algorithm-specific parameters
    float metrics[4];                           // Quality metrics (PSNR, SSIM, etc.)
} ENNCPCompressionStage;

// Quality and performance metrics
typedef struct {
    float compression_ratio;                    // Overall compression ratio
    float quality_score;                        // Overall quality score (0-1)
    float psnr;                                // Peak Signal-to-Noise Ratio
    float ssim;                                // Structural Similarity Index
    float mse;                                 // Mean Squared Error
    float mae;                                 // Mean Absolute Error
    float perceptual_quality;                  // Perceptual quality score
    float compression_efficiency;              // Compression efficiency score
    uint32_t compression_time_ms;              // Total compression time
    uint32_t decompression_time_ms;            // Estimated decompression time
    uint32_t memory_usage_kb;                  // Memory usage during compression
    uint32_t cpu_usage_percent;                // CPU usage percentage
    float energy_consumption;                  // Energy consumption estimate
    float throughput_mbps;                     // Compression throughput
} ENNCPQualityMetrics;

// Compatibility information
typedef struct {
    uint32_t format_version;                   // Format version
    uint32_t min_decoder_version;              // Minimum decoder version required
    uint32_t feature_flags;                    // Feature compatibility flags
    uint32_t extension_flags;                  // Extension compatibility flags
    char decoder_requirements[128];            // Decoder requirements description
    char compatibility_notes[256];             // Compatibility notes
} ENNCPCompatibilityInfo;

// Security and verification information
typedef struct {
    ENNCPEncryptionAlgorithm encryption_algorithm; // Encryption algorithm
    ENNCPChecksumAlgorithm checksum_algorithm;  // Checksum algorithm
    uint8_t encryption_key_hash[32];           // Hash of encryption key (for verification)
    uint8_t data_checksum[64];                 // Data integrity checksum
    uint8_t metadata_checksum[32];             // Metadata integrity checksum
    uint8_t digital_signature[256];            // Digital signature (optional)
    uint32_t signature_length;                 // Actual signature length
    uint64_t timestamp;                        // Creation timestamp
    char creator_id[64];                       // Creator identifier
    bool is_encrypted;                         // Whether data is encrypted
    bool is_signed;                            // Whether file is digitally signed
} ENNCPSecurityInfo;

// Enhanced NNCP metadata structure
typedef struct {
    // Header information
    uint32_t magic_number;                     // Magic number for format identification
    uint32_t format_version;                   // Format version
    uint32_t metadata_size;                    // Size of this metadata structure
    uint32_t total_file_size;                  // Total file size including data
    
    // Data information
    ENNCPDataType data_type;                   // Type of data stored
    uint64_t original_size;                    // Original uncompressed size
    uint64_t compressed_size;                  // Compressed data size
    uint32_t data_offset;                      // Offset to compressed data
    uint32_t num_compression_stages;           // Number of compression stages
    
    // Timestamps and creation info
    uint64_t creation_timestamp;               // Creation time (Unix timestamp)
    uint64_t modification_timestamp;           // Last modification time
    char creator_software[64];                 // Software that created this file
    char creator_version[32];                  // Version of creator software
    char creator_platform[32];                // Platform (macOS, Windows, Linux)
    char source_filename[256];                 // Original filename
    
    // Compression information
    ENNCPCompressionStage compression_stages[ENNCP_MAX_COMPRESSION_STAGES];
    ENNCPQualityMetrics quality_metrics;       // Quality and performance metrics
    
    // Compatibility and features
    ENNCPCompatibilityInfo compatibility;      // Compatibility information
    ENNCPSecurityInfo security;                // Security and verification info
    
    // Custom metadata fields
    uint32_t num_custom_fields;                // Number of custom fields
    ENNCPCustomField custom_fields[ENNCP_MAX_CUSTOM_FIELDS];
    
    // Additional features
    bool supports_streaming;                   // Supports streaming decompression
    bool supports_random_access;               // Supports random access
    bool supports_partial_decode;              // Supports partial decoding
    bool is_lossy_compression;                 // Whether compression is lossy
    uint32_t block_size;                       // Block size for block-based compression
    uint32_t chunk_count;                      // Number of data chunks
    
    // Reserved for future extensions
    uint8_t reserved[512];                     // Reserved space for future features
} ENNCPMetadata;

// File handle for enhanced NNCP files
typedef struct {
    FILE* file_handle;                         // File handle
    ENNCPMetadata metadata;                    // File metadata
    bool is_open;                              // Whether file is open
    bool is_writable;                          // Whether file is writable
    bool metadata_loaded;                      // Whether metadata is loaded
    uint64_t current_position;                 // Current read/write position
    void* compression_context;                 // Compression context
    void* decompression_context;               // Decompression context
    uint8_t* buffer;                          // Internal buffer
    size_t buffer_size;                       // Buffer size
    char filename[512];                       // Filename
} ENNCPFile;

// Data chunk for streaming operations
typedef struct {
    uint32_t chunk_id;                        // Chunk identifier
    uint32_t chunk_size;                      // Size of this chunk
    uint32_t original_chunk_size;             // Original size before compression
    uint32_t chunk_offset;                    // Offset within original data
    ENNCPCompressionAlgorithm compression;    // Compression used for this chunk
    uint8_t chunk_checksum[32];               // Chunk integrity checksum
    void* chunk_data;                         // Actual chunk data
    bool is_compressed;                       // Whether chunk is compressed
} ENNCPChunk;

// Core API Functions

/**
 * Create a new enhanced NNCP file
 * @param filename Path to file to create
 * @param file_handle Output file handle
 * @param data_type Type of data to be stored
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_create_file(const char* filename, ENNCPFile** file_handle, ENNCPDataType data_type);

/**
 * Open an existing enhanced NNCP file
 * @param filename Path to file to open
 * @param file_handle Output file handle
 * @param read_only Whether to open in read-only mode
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_open_file(const char* filename, ENNCPFile** file_handle, bool read_only);

/**
 * Write compressed data to enhanced NNCP file
 * @param file_handle File handle
 * @param data Data to write
 * @param data_size Size of data
 * @param compression_algorithm Compression algorithm to use
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_write_data(ENNCPFile* file_handle, const void* data, size_t data_size,
                           ENNCPCompressionAlgorithm compression_algorithm);

/**
 * Read and decompress data from enhanced NNCP file
 * @param file_handle File handle
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 * @param bytes_read Number of bytes actually read
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_read_data(ENNCPFile* file_handle, void* buffer, size_t buffer_size, size_t* bytes_read);

/**
 * Close enhanced NNCP file
 * @param file_handle File handle to close
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_close_file(ENNCPFile* file_handle);

// Metadata Management Functions

/**
 * Set custom metadata field
 * @param file_handle File handle
 * @param field_name Field name
 * @param field_value Field value
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_set_custom_field(ENNCPFile* file_handle, const char* field_name, const char* field_value);

/**
 * Get custom metadata field value
 * @param file_handle File handle
 * @param field_name Field name
 * @param field_value Output buffer for field value
 * @param value_size Size of output buffer
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_get_custom_field(ENNCPFile* file_handle, const char* field_name, 
                                 char* field_value, size_t value_size);

/**
 * Remove custom metadata field
 * @param file_handle File handle
 * @param field_name Field name to remove
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_remove_custom_field(ENNCPFile* file_handle, const char* field_name);

/**
 * Get all custom field names
 * @param file_handle File handle
 * @param field_names Output array of field names
 * @param max_fields Maximum number of fields to return
 * @param num_fields Number of fields actually returned
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_list_custom_fields(ENNCPFile* file_handle, char field_names[][ENNCP_MAX_FIELD_NAME_LEN],
                                   uint32_t max_fields, uint32_t* num_fields);

// Compression and Quality Functions

/**
 * Set compression parameters
 * @param file_handle File handle
 * @param algorithm Compression algorithm
 * @param quality_level Quality level (0-15, higher = better quality)
 * @param custom_params Custom algorithm parameters
 * @param num_params Number of custom parameters
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_set_compression_params(ENNCPFile* file_handle, ENNCPCompressionAlgorithm algorithm,
                                       uint32_t quality_level, const uint32_t* custom_params, 
                                       uint32_t num_params);

/**
 * Get quality metrics for the file
 * @param file_handle File handle
 * @param metrics Output quality metrics
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_get_quality_metrics(ENNCPFile* file_handle, ENNCPQualityMetrics* metrics);

/**
 * Validate file integrity
 * @param file_handle File handle
 * @param validation_level Level of validation (0=basic, 1=full, 2=deep)
 * @return ENNCP_SUCCESS if valid, error code if invalid
 */
ENNCPError enncp_validate_integrity(ENNCPFile* file_handle, uint32_t validation_level);

// Security Functions

/**
 * Set encryption for the file
 * @param file_handle File handle
 * @param algorithm Encryption algorithm
 * @param key Encryption key
 * @param key_size Size of encryption key
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_set_encryption(ENNCPFile* file_handle, ENNCPEncryptionAlgorithm algorithm,
                               const uint8_t* key, size_t key_size);

/**
 * Add digital signature to the file
 * @param file_handle File handle
 * @param signature Digital signature
 * @param signature_size Size of signature
 * @param creator_id Creator identifier
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_add_signature(ENNCPFile* file_handle, const uint8_t* signature, 
                              size_t signature_size, const char* creator_id);

/**
 * Verify digital signature
 * @param file_handle File handle
 * @param public_key Public key for verification
 * @param key_size Size of public key
 * @return ENNCP_SUCCESS if valid, error code if invalid
 */
ENNCPError enncp_verify_signature(ENNCPFile* file_handle, const uint8_t* public_key, size_t key_size);

// Compatibility and Version Functions

/**
 * Check format compatibility
 * @param file_handle File handle
 * @param required_features Required feature flags
 * @param is_compatible Output compatibility status
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_check_compatibility(ENNCPFile* file_handle, uint32_t required_features, bool* is_compatible);

/**
 * Upgrade file format to newer version
 * @param filename File to upgrade
 * @param target_version Target format version
 * @param backup_original Whether to create backup of original
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_upgrade_format(const char* filename, uint32_t target_version, bool backup_original);

// Streaming and Random Access Functions

/**
 * Enable streaming mode for the file
 * @param file_handle File handle
 * @param chunk_size Chunk size for streaming
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_enable_streaming(ENNCPFile* file_handle, uint32_t chunk_size);

/**
 * Seek to specific position in uncompressed data
 * @param file_handle File handle
 * @param offset Offset in uncompressed data
 * @param whence Seek origin (SEEK_SET, SEEK_CUR, SEEK_END)
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_seek(ENNCPFile* file_handle, int64_t offset, int whence);

/**
 * Get current position in uncompressed data
 * @param file_handle File handle
 * @param position Output current position
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_tell(ENNCPFile* file_handle, uint64_t* position);

// Utility Functions

/**
 * Get format version string
 * @param version Version number
 * @param version_string Output version string
 * @param string_size Size of output buffer
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_get_version_string(uint32_t version, char* version_string, size_t string_size);

/**
 * Get error message string
 * @param error_code Error code
 * @return Human-readable error message
 */
const char* enncp_get_error_string(ENNCPError error_code);

/**
 * Get compression algorithm name
 * @param algorithm Compression algorithm
 * @return Algorithm name string
 */
const char* enncp_get_compression_name(ENNCPCompressionAlgorithm algorithm);

/**
 * Get data type name
 * @param data_type Data type
 * @return Data type name string
 */
const char* enncp_get_data_type_name(ENNCPDataType data_type);

/**
 * Calculate optimal block size for data
 * @param data_size Size of data
 * @param data_type Type of data
 * @param recommended_block_size Output recommended block size
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_calculate_optimal_block_size(uint64_t data_size, ENNCPDataType data_type, 
                                             uint32_t* recommended_block_size);

/**
 * Estimate compression ratio for data
 * @param data Sample data
 * @param data_size Size of sample data
 * @param algorithm Compression algorithm
 * @param estimated_ratio Output estimated compression ratio
 * @return ENNCP_SUCCESS on success, error code on failure
 */
ENNCPError enncp_estimate_compression_ratio(const void* data, size_t data_size,
                                           ENNCPCompressionAlgorithm algorithm, float* estimated_ratio);

#ifdef __cplusplus
}
#endif

#endif // ENHANCED_NNCP_FORMAT_H
