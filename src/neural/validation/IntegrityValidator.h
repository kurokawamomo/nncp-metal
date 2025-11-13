/*
 * IntegrityValidator.h
 * 
 * Data Integrity Validation Engine for Neural Compression
 * Comprehensive checksum verification, decompression integrity checks,
 * numerical stability validation, and data consistency assurance
 */

#ifndef INTEGRITY_VALIDATOR_H
#define INTEGRITY_VALIDATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <CryptoKit/CryptoKit.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct IntegrityValidator IntegrityValidator;

// Checksum algorithm types
typedef enum {
    CHECKSUM_ALGORITHM_CRC32 = 0,         // CRC32 checksum
    CHECKSUM_ALGORITHM_CRC64,             // CRC64 checksum
    CHECKSUM_ALGORITHM_ADLER32,           // Adler-32 checksum
    CHECKSUM_ALGORITHM_SHA1,              // SHA-1 hash
    CHECKSUM_ALGORITHM_SHA256,            // SHA-256 hash
    CHECKSUM_ALGORITHM_SHA512,            // SHA-512 hash
    CHECKSUM_ALGORITHM_BLAKE3,            // BLAKE3 hash (fastest)
    CHECKSUM_ALGORITHM_XXHASH64,          // xxHash 64-bit (very fast)
    CHECKSUM_ALGORITHM_MURMUR3,           // MurmurHash3 (fast)
    CHECKSUM_ALGORITHM_CITYHASH64         // CityHash 64-bit
} ChecksumAlgorithm;

// Data integrity validation levels
typedef enum {
    VALIDATION_LEVEL_BASIC = 0,           // Basic checksum validation
    VALIDATION_LEVEL_STANDARD,            // Standard validation with multiple checksums
    VALIDATION_LEVEL_COMPREHENSIVE,       // Comprehensive with deep validation
    VALIDATION_LEVEL_PARANOID,            // Paranoid level with extensive checks
    VALIDATION_LEVEL_CUSTOM              // Custom validation configuration
} ValidationLevel;

// Numerical stability check types
typedef enum {
    NUMERICAL_CHECK_NONE = 0,             // No numerical stability checking
    NUMERICAL_CHECK_BASIC,                // Basic range and NaN/Inf checks
    NUMERICAL_CHECK_STATISTICAL,          // Statistical distribution checks
    NUMERICAL_CHECK_PRECISION,            // Floating-point precision validation
    NUMERICAL_CHECK_COMPREHENSIVE         // Comprehensive numerical validation
} NumericalStabilityCheck;

// Data type classifications for validation
typedef enum {
    DATA_TYPE_BINARY = 0,                 // Raw binary data
    DATA_TYPE_TEXT,                       // Text/string data
    DATA_TYPE_FLOAT32,                    // 32-bit floating-point arrays
    DATA_TYPE_FLOAT64,                    // 64-bit floating-point arrays
    DATA_TYPE_INT8,                       // 8-bit integer arrays
    DATA_TYPE_INT16,                      // 16-bit integer arrays
    DATA_TYPE_INT32,                      // 32-bit integer arrays
    DATA_TYPE_INT64,                      // 64-bit integer arrays
    DATA_TYPE_MIXED,                      // Mixed data types
    DATA_TYPE_NEURAL_WEIGHTS              // Neural network weight matrices
} DataType;

// Validation configuration
typedef struct {
    ValidationLevel validation_level;      // Overall validation level
    ChecksumAlgorithm primary_checksum;    // Primary checksum algorithm
    ChecksumAlgorithm secondary_checksum;  // Secondary checksum for verification
    NumericalStabilityCheck numerical_validation; // Numerical stability checking
    bool enable_deep_content_validation;   // Enable deep content validation
    bool enable_compression_ratio_validation; // Enable compression ratio validation
    bool enable_timing_attack_protection;  // Enable timing attack protection
    bool enable_memory_pattern_validation; // Enable memory pattern validation
    bool enable_entropy_validation;        // Enable data entropy validation
    bool enable_statistical_validation;    // Enable statistical property validation
    uint32_t max_validation_time_ms;       // Maximum validation time (milliseconds)
    float numerical_tolerance_epsilon;     // Numerical comparison tolerance
    uint32_t sample_validation_percentage;  // Percentage of data to sample validate (1-100)
    uint64_t max_data_size_for_full_validation; // Max size for full validation (bytes)
} IntegrityValidationConfig;

// Checksum result structure
typedef struct {
    ChecksumAlgorithm algorithm;           // Checksum algorithm used
    uint64_t checksum_value_64;            // 64-bit checksum value
    uint8_t checksum_hash[64];             // Hash value (for hash algorithms)
    uint32_t checksum_size_bytes;          // Size of checksum in bytes
    uint64_t computation_time_ns;          // Time taken to compute checksum
    bool is_hash_algorithm;                // True if algorithm produces hash
    char algorithm_name[32];               // Human-readable algorithm name
} ChecksumResult;

// Data integrity metrics
typedef struct {
    uint64_t total_bytes_validated;        // Total bytes validated
    uint64_t validation_time_ns;           // Total validation time
    uint32_t checksum_matches;             // Number of matching checksums
    uint32_t checksum_mismatches;          // Number of mismatched checksums
    uint32_t numerical_errors_detected;    // Numerical errors detected
    uint32_t content_errors_detected;      // Content errors detected
    uint32_t statistical_anomalies;        // Statistical anomalies found
    float data_entropy_original;           // Original data entropy
    float data_entropy_recovered;          // Recovered data entropy
    float compression_ratio_achieved;      // Compression ratio achieved
    float data_similarity_score;           // Data similarity score (0.0-1.0)
    bool passes_all_checks;                // All validation checks passed
    char validation_summary[512];          // Human-readable validation summary
} IntegrityValidationResult;

// Numerical stability analysis
typedef struct {
    uint32_t nan_values_detected;          // NaN values detected
    uint32_t inf_values_detected;          // Infinite values detected
    uint32_t denormal_values_detected;     // Denormal values detected
    uint32_t out_of_range_values;          // Values outside expected range
    double mean_absolute_error;            // Mean absolute error
    double max_absolute_error;             // Maximum absolute error
    double relative_error_mean;            // Mean relative error
    double relative_error_max;             // Maximum relative error
    float precision_loss_percentage;       // Precision loss percentage
    float dynamic_range_compression;       // Dynamic range compression factor
    bool numerical_stability_acceptable;   // Overall numerical stability assessment
    char stability_assessment[256];        // Detailed stability assessment
} NumericalStabilityResult;

// Deep content validation result
typedef struct {
    bool content_structure_valid;          // Content structure is valid
    bool header_integrity_valid;           // Header integrity is valid
    bool metadata_consistency_valid;       // Metadata consistency is valid
    bool cross_reference_validity;         // Cross-references are valid
    uint32_t corrupted_blocks_detected;    // Number of corrupted data blocks
    uint32_t missing_data_segments;        // Missing data segments
    uint32_t duplicate_data_segments;      // Duplicate data segments
    float content_completeness_score;      // Content completeness (0.0-1.0)
    float structural_integrity_score;      // Structural integrity (0.0-1.0)
    char content_validation_details[1024]; // Detailed content validation results
} DeepContentValidationResult;

// Comprehensive validation report
typedef struct {
    IntegrityValidationConfig config;      // Validation configuration used
    ChecksumResult primary_checksum_result; // Primary checksum result
    ChecksumResult secondary_checksum_result; // Secondary checksum result
    NumericalStabilityResult numerical_result; // Numerical stability analysis
    DeepContentValidationResult content_result; // Deep content validation
    IntegrityValidationResult overall_result; // Overall validation result
    uint64_t original_data_size;           // Original data size
    uint64_t compressed_data_size;         // Compressed data size
    uint64_t decompressed_data_size;       // Decompressed data size
    bool size_consistency_valid;           // Size consistency is valid
    bool checksum_consistency_valid;       // Checksum consistency is valid
    bool numerical_consistency_valid;      // Numerical consistency is valid
    bool content_consistency_valid;        // Content consistency is valid
    float overall_integrity_score;         // Overall integrity score (0.0-1.0)
    char validation_timestamp[64];         // Validation timestamp
    char integrity_certificate[1024];      // Integrity certificate/summary
} ComprehensiveValidationReport;

// Error codes for integrity validation
typedef enum {
    INTEGRITY_VALIDATOR_SUCCESS = 0,
    INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM,
    INTEGRITY_VALIDATOR_ERROR_MEMORY_ALLOCATION,
    INTEGRITY_VALIDATOR_ERROR_CHECKSUM_MISMATCH,
    INTEGRITY_VALIDATOR_ERROR_NUMERICAL_INSTABILITY,
    INTEGRITY_VALIDATOR_ERROR_CONTENT_CORRUPTION,
    INTEGRITY_VALIDATOR_ERROR_SIZE_MISMATCH,
    INTEGRITY_VALIDATOR_ERROR_ENTROPY_VIOLATION,
    INTEGRITY_VALIDATOR_ERROR_STATISTICAL_ANOMALY,
    INTEGRITY_VALIDATOR_ERROR_VALIDATION_TIMEOUT,
    INTEGRITY_VALIDATOR_ERROR_UNSUPPORTED_ALGORITHM,
    INTEGRITY_VALIDATOR_ERROR_INSUFFICIENT_DATA,
    INTEGRITY_VALIDATOR_ERROR_CONFIGURATION_INVALID
} IntegrityValidatorError;

// Core API Functions

/**
 * Create integrity validator instance
 * @param validator Pointer to store created validator
 * @param config Integrity validation configuration
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_create(IntegrityValidator** validator,
                                                   const IntegrityValidationConfig* config);

/**
 * Initialize integrity validator with system capabilities
 * @param validator Integrity validator instance
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_initialize(IntegrityValidator* validator);

// Checksum Computation and Verification

/**
 * Compute checksum for given data
 * @param validator Integrity validator instance
 * @param data Data to compute checksum for
 * @param data_size Size of data in bytes
 * @param algorithm Checksum algorithm to use
 * @param checksum_result Output checksum result
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_compute_checksum(IntegrityValidator* validator,
                                                            const void* data,
                                                            size_t data_size,
                                                            ChecksumAlgorithm algorithm,
                                                            ChecksumResult* checksum_result);

/**
 * Verify checksum against expected value
 * @param validator Integrity validator instance
 * @param data Data to verify
 * @param data_size Size of data in bytes
 * @param expected_checksum Expected checksum result
 * @param verification_passed Output boolean for verification result
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_verify_checksum(IntegrityValidator* validator,
                                                           const void* data,
                                                           size_t data_size,
                                                           const ChecksumResult* expected_checksum,
                                                           bool* verification_passed);

/**
 * Compute multiple checksums for redundant verification
 * @param validator Integrity validator instance
 * @param data Data to compute checksums for
 * @param data_size Size of data in bytes
 * @param algorithms Array of checksum algorithms
 * @param algorithm_count Number of algorithms
 * @param checksum_results Array of output checksum results
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_compute_multiple_checksums(IntegrityValidator* validator,
                                                                       const void* data,
                                                                       size_t data_size,
                                                                       const ChecksumAlgorithm* algorithms,
                                                                       uint32_t algorithm_count,
                                                                       ChecksumResult* checksum_results);

// Compression-Decompression Integrity Validation

/**
 * Validate compression integrity (pre-compression validation)
 * @param validator Integrity validator instance
 * @param original_data Original data before compression
 * @param data_size Size of original data
 * @param data_type Type of data being compressed
 * @param pre_compression_result Output pre-compression validation result
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_pre_compression(IntegrityValidator* validator,
                                                                     const void* original_data,
                                                                     size_t data_size,
                                                                     DataType data_type,
                                                                     IntegrityValidationResult* pre_compression_result);

/**
 * Validate decompression integrity (post-decompression validation)
 * @param validator Integrity validator instance
 * @param original_data Original data before compression
 * @param original_size Size of original data
 * @param decompressed_data Decompressed data
 * @param decompressed_size Size of decompressed data
 * @param data_type Type of data
 * @param post_decompression_result Output post-decompression validation result
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_post_decompression(IntegrityValidator* validator,
                                                                        const void* original_data,
                                                                        size_t original_size,
                                                                        const void* decompressed_data,
                                                                        size_t decompressed_size,
                                                                        DataType data_type,
                                                                        IntegrityValidationResult* post_decompression_result);

/**
 * Perform comprehensive round-trip validation
 * @param validator Integrity validator instance
 * @param original_data Original input data
 * @param original_size Size of original data
 * @param compressed_data Compressed data
 * @param compressed_size Size of compressed data
 * @param decompressed_data Decompressed data
 * @param decompressed_size Size of decompressed data
 * @param data_type Type of data being validated
 * @param comprehensive_report Output comprehensive validation report
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_round_trip(IntegrityValidator* validator,
                                                               const void* original_data,
                                                               size_t original_size,
                                                               const void* compressed_data,
                                                               size_t compressed_size,
                                                               const void* decompressed_data,
                                                               size_t decompressed_size,
                                                               DataType data_type,
                                                               ComprehensiveValidationReport* comprehensive_report);

// Numerical Stability Validation

/**
 * Validate numerical stability of floating-point data
 * @param validator Integrity validator instance
 * @param original_float_data Original floating-point data
 * @param recovered_float_data Recovered floating-point data after compression/decompression
 * @param element_count Number of floating-point elements
 * @param is_double_precision True for double precision, false for single precision
 * @param stability_result Output numerical stability analysis
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_numerical_stability(IntegrityValidator* validator,
                                                                         const void* original_float_data,
                                                                         const void* recovered_float_data,
                                                                         size_t element_count,
                                                                         bool is_double_precision,
                                                                         NumericalStabilityResult* stability_result);

/**
 * Validate neural network weights stability
 * @param validator Integrity validator instance
 * @param original_weights Original neural network weights
 * @param recovered_weights Recovered weights after compression/decompression
 * @param weight_matrices Number of weight matrices
 * @param matrix_dimensions Array of matrix dimensions (rows, cols for each matrix)
 * @param stability_result Output stability analysis
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_neural_weights_stability(IntegrityValidator* validator,
                                                                              const float* const* original_weights,
                                                                              const float* const* recovered_weights,
                                                                              uint32_t weight_matrices,
                                                                              const uint32_t* matrix_dimensions,
                                                                              NumericalStabilityResult* stability_result);

/**
 * Check for numerical anomalies in floating-point data
 * @param validator Integrity validator instance
 * @param float_data Floating-point data to check
 * @param element_count Number of elements
 * @param is_double_precision True for double precision
 * @param nan_count Output count of NaN values
 * @param inf_count Output count of infinite values
 * @param denormal_count Output count of denormal values
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_check_numerical_anomalies(IntegrityValidator* validator,
                                                                      const void* float_data,
                                                                      size_t element_count,
                                                                      bool is_double_precision,
                                                                      uint32_t* nan_count,
                                                                      uint32_t* inf_count,
                                                                      uint32_t* denormal_count);

// Deep Content Validation

/**
 * Perform deep content structure validation
 * @param validator Integrity validator instance
 * @param data Data to validate
 * @param data_size Size of data
 * @param data_type Type of data
 * @param expected_structure_info Optional expected structure information
 * @param content_result Output deep content validation result
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_content_structure(IntegrityValidator* validator,
                                                                       const void* data,
                                                                       size_t data_size,
                                                                       DataType data_type,
                                                                       const void* expected_structure_info,
                                                                       DeepContentValidationResult* content_result);

/**
 * Validate data entropy and randomness properties
 * @param validator Integrity validator instance
 * @param original_data Original data
 * @param recovered_data Recovered data
 * @param data_size Size of data
 * @param original_entropy Output entropy of original data
 * @param recovered_entropy Output entropy of recovered data
 * @param entropy_similarity Output entropy similarity score
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_entropy(IntegrityValidator* validator,
                                                            const void* original_data,
                                                            const void* recovered_data,
                                                            size_t data_size,
                                                            float* original_entropy,
                                                            float* recovered_entropy,
                                                            float* entropy_similarity);

/**
 * Validate statistical properties of data
 * @param validator Integrity validator instance
 * @param original_data Original numerical data
 * @param recovered_data Recovered numerical data
 * @param element_count Number of numerical elements
 * @param data_type Numerical data type
 * @param statistical_similarity Output statistical similarity score
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_statistical_properties(IntegrityValidator* validator,
                                                                            const void* original_data,
                                                                            const void* recovered_data,
                                                                            size_t element_count,
                                                                            DataType data_type,
                                                                            float* statistical_similarity);

// Performance and Security

/**
 * Perform constant-time validation to prevent timing attacks
 * @param validator Integrity validator instance
 * @param data1 First data buffer
 * @param data2 Second data buffer
 * @param data_size Size of both data buffers
 * @param buffers_equal Output boolean for equality
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_constant_time_compare(IntegrityValidator* validator,
                                                                 const void* data1,
                                                                 const void* data2,
                                                                 size_t data_size,
                                                                 bool* buffers_equal);

/**
 * Validate memory access patterns for security
 * @param validator Integrity validator instance
 * @param data Data buffer to validate
 * @param data_size Size of data buffer
 * @param expected_pattern_info Expected access pattern information
 * @param pattern_valid Output boolean for pattern validity
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_memory_patterns(IntegrityValidator* validator,
                                                                     const void* data,
                                                                     size_t data_size,
                                                                     const void* expected_pattern_info,
                                                                     bool* pattern_valid);

// Configuration and Utility Functions

/**
 * Create default integrity validation configuration
 * @param config Output default configuration
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_create_default_config(IntegrityValidationConfig* config);

/**
 * Create paranoid (maximum security) validation configuration
 * @param config Output paranoid configuration
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_create_paranoid_config(IntegrityValidationConfig* config);

/**
 * Create performance-optimized validation configuration
 * @param config Output performance-optimized configuration
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_create_performance_config(IntegrityValidationConfig* config);

/**
 * Validate integrity validation configuration
 * @param config Configuration to validate
 * @param is_valid Output boolean for configuration validity
 * @param validation_message Output validation message buffer
 * @param message_size Size of validation message buffer
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_validate_config(const IntegrityValidationConfig* config,
                                                           bool* is_valid,
                                                           char* validation_message,
                                                           size_t message_size);

/**
 * Get recommended validation level for data type and size
 * @param data_type Type of data to validate
 * @param data_size Size of data
 * @param performance_priority Performance vs security priority (0.0-1.0)
 * @param recommended_level Output recommended validation level
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_get_recommended_level(DataType data_type,
                                                                 size_t data_size,
                                                                 float performance_priority,
                                                                 ValidationLevel* recommended_level);

/**
 * Benchmark validation algorithm performance
 * @param validator Integrity validator instance
 * @param test_data Test data for benchmarking
 * @param test_data_size Size of test data
 * @param algorithm Checksum algorithm to benchmark
 * @param iterations Number of benchmark iterations
 * @param average_time_ns Output average computation time per iteration
 * @param throughput_mbps Output throughput in MB/s
 * @return INTEGRITY_VALIDATOR_SUCCESS on success, error code on failure
 */
IntegrityValidatorError integrity_validator_benchmark_algorithm(IntegrityValidator* validator,
                                                               const void* test_data,
                                                               size_t test_data_size,
                                                               ChecksumAlgorithm algorithm,
                                                               uint32_t iterations,
                                                               uint64_t* average_time_ns,
                                                               float* throughput_mbps);

/**
 * Destroy integrity validator and free resources
 * @param validator Integrity validator instance to destroy
 */
void integrity_validator_destroy(IntegrityValidator* validator);

// Utility Functions

/**
 * Get error string for integrity validator error code
 * @param error_code IntegrityValidatorError code
 * @return Human-readable error message
 */
const char* integrity_validator_get_error_string(IntegrityValidatorError error_code);

/**
 * Get checksum algorithm string
 * @param algorithm Checksum algorithm enum
 * @return Human-readable algorithm name
 */
const char* integrity_validator_get_algorithm_string(ChecksumAlgorithm algorithm);

/**
 * Get validation level string
 * @param level Validation level enum
 * @return Human-readable validation level name
 */
const char* integrity_validator_get_validation_level_string(ValidationLevel level);

/**
 * Get data type string
 * @param data_type Data type enum
 * @return Human-readable data type name
 */
const char* integrity_validator_get_data_type_string(DataType data_type);

/**
 * Calculate expected checksum computation time
 * @param algorithm Checksum algorithm
 * @param data_size Size of data in bytes
 * @return Expected computation time in nanoseconds
 */
uint64_t integrity_validator_estimate_checksum_time(ChecksumAlgorithm algorithm, size_t data_size);

/**
 * Check if algorithm is cryptographic hash
 * @param algorithm Checksum algorithm
 * @return True if algorithm is cryptographic hash
 */
bool integrity_validator_is_cryptographic_hash(ChecksumAlgorithm algorithm);

/**
 * Get checksum size for algorithm
 * @param algorithm Checksum algorithm
 * @return Size of checksum in bytes
 */
uint32_t integrity_validator_get_checksum_size(ChecksumAlgorithm algorithm);

// Constants for integrity validation

// Algorithm performance characteristics (operations per second on Apple Silicon)
#define XXHASH64_OPS_PER_SECOND_M3 15000000000ULL    // ~15 GB/s on M3
#define CRC32_OPS_PER_SECOND_M3 8000000000ULL        // ~8 GB/s on M3
#define BLAKE3_OPS_PER_SECOND_M3 2000000000ULL       // ~2 GB/s on M3
#define SHA256_OPS_PER_SECOND_M3 500000000ULL        // ~500 MB/s on M3

// Validation thresholds
#define NUMERICAL_TOLERANCE_DEFAULT 1e-6f             // Default numerical tolerance
#define ENTROPY_SIMILARITY_THRESHOLD 0.95f           // Entropy similarity threshold
#define STATISTICAL_SIMILARITY_THRESHOLD 0.98f       // Statistical similarity threshold
#define INTEGRITY_SCORE_EXCELLENT 0.99f              // Excellent integrity score
#define INTEGRITY_SCORE_GOOD 0.95f                   // Good integrity score
#define INTEGRITY_SCORE_ACCEPTABLE 0.90f             // Acceptable integrity score

// Performance limits
#define MAX_VALIDATION_TIME_DEFAULT_MS 5000          // Default max validation time
#define MAX_FULL_VALIDATION_SIZE_MB 100              // Max size for full validation
#define SAMPLE_VALIDATION_DEFAULT_PERCENTAGE 10     // Default sample percentage

// Security constants
#define TIMING_ATTACK_PROTECTION_THRESHOLD 1000000   // Timing protection threshold (ns)
#define MEMORY_PATTERN_CHECK_BLOCK_SIZE 4096         // Memory pattern check block size

#ifdef __cplusplus
}
#endif

#endif // INTEGRITY_VALIDATOR_H
