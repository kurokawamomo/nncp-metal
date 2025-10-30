/*
 * IntegrityValidator.mm
 * 
 * Data Integrity Validation Engine for Neural Compression
 * Comprehensive checksum verification, decompression integrity checks,
 * numerical stability validation, and data consistency assurance
 */

#import "IntegrityValidator.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <zlib.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#import <CommonCrypto/CommonCrypto.h>
#endif

// Internal integrity validator structure
struct IntegrityValidator {
    IntegrityValidationConfig config;
    
    // Timing infrastructure
    mach_timebase_info_data_t timebase_info;
    
    // Statistical analysis buffers
    double* numerical_analysis_buffer;
    size_t analysis_buffer_size;
    
    // Checksum computation contexts
    CC_SHA1_CTX sha1_context;
    CC_SHA256_CTX sha256_context;
    CC_SHA512_CTX sha512_context;
    
    // Performance tracking
    uint64_t total_validations_performed;
    uint64_t total_bytes_validated;
    uint64_t total_validation_time_ns;
    
    // Threading and synchronization
    pthread_mutex_t validation_mutex;
    pthread_cond_t validation_condition;
    
    // Memory management
    uint8_t* temp_validation_buffer;
    size_t temp_buffer_size;
    
    // Algorithm performance cache
    uint64_t algorithm_performance_cache[10]; // Per algorithm timing cache
};

// Helper macros
#define NANOSECONDS_PER_SECOND 1000000000ULL
#define MICROSECONDS_PER_SECOND 1000000ULL

// Convert mach absolute time to nanoseconds
static inline uint64_t mach_time_to_nanoseconds(uint64_t mach_time, mach_timebase_info_data_t* timebase) {
    return (mach_time * timebase->numer) / timebase->denom;
}

// Get current nanosecond timestamp
static inline uint64_t get_nanosecond_timestamp() {
    return mach_absolute_time();
}

// Constant-time memory comparison
static bool constant_time_memcmp(const void* ptr1, const void* ptr2, size_t size) {
    const unsigned char* p1 = (const unsigned char*)ptr1;
    const unsigned char* p2 = (const unsigned char*)ptr2;
    unsigned char result = 0;
    
    for (size_t i = 0; i < size; i++) {
        result |= p1[i] ^ p2[i];
    }
    
    return result == 0;
}

// Fast CRC32 implementation using Apple's optimized zlib
static uint32_t compute_crc32(const void* data, size_t size) {
    return crc32(crc32(0L, Z_NULL, 0), (const Bytef*)data, (uInt)size);
}

// CRC64 implementation (simplified)
static uint64_t compute_crc64(const void* data, size_t size) {
    static const uint64_t crc64_table[256] = {
        0x0000000000000000ULL, 0x42F0E1EBA9EA3693ULL, 0x85E1C3D753D46D26ULL, 0xC711223CFA3E5BB5ULL,
        // ... (full table would be here in production code)
        0x9B64C2B0EC63F226ULL, 0xD9945837D49C2A95ULL, 0x1E85DEF0A2C37900ULL, 0x5C753F1FB5DAC893ULL
    };
    
    uint64_t crc = 0xFFFFFFFFFFFFFFFFULL;
    const uint8_t* bytes = (const uint8_t*)data;
    
    for (size_t i = 0; i < size; i++) {
        crc = crc64_table[(crc ^ bytes[i]) & 0xFF] ^ (crc >> 8);
    }
    
    return crc ^ 0xFFFFFFFFFFFFFFFFULL;
}

// Adler-32 checksum
static uint32_t compute_adler32(const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t a = 1, b = 0;
    const uint32_t MOD_ADLER = 65521;
    
    for (size_t i = 0; i < size; i++) {
        a = (a + bytes[i]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    
    return (b << 16) | a;
}

// xxHash64 implementation (simplified)
static uint64_t compute_xxhash64(const void* data, size_t size) {
    // Simplified xxHash64 - production code would use full implementation
    const uint8_t* bytes = (const uint8_t*)data;
    uint64_t hash = 0x9E3779B185EBCA87ULL;
    
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 0xC2B2AE3D27D4EB4FULL;
        hash ^= hash >> 33;
    }
    
    return hash;
}

// MurmurHash3 64-bit
static uint64_t compute_murmur3(const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint64_t hash = 0x9747B28C;
    const uint64_t c1 = 0x87C37B91114253D5ULL;
    const uint64_t c2 = 0x4CF5AD432745937FULL;
    
    for (size_t i = 0; i < size; i++) {
        uint64_t k = bytes[i];
        k *= c1;
        k = (k << 31) | (k >> 33);
        k *= c2;
        hash ^= k;
        hash = ((hash << 27) | (hash >> 37)) * 5 + 0x52DCE729;
    }
    
    hash ^= size;
    hash ^= hash >> 33;
    hash *= 0xFF51AFD7ED558CCDULL;
    hash ^= hash >> 33;
    
    return hash;
}

// Statistical analysis functions
static double calculate_mean(const double* values, size_t count) {
    if (count == 0) return 0.0;
    
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    return sum / count;
}

static double calculate_variance(const double* values, size_t count, double mean) {
    if (count <= 1) return 0.0;
    
    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < count; i++) {
        double diff = values[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / (count - 1);
}

static double calculate_entropy(const void* data, size_t size) {
    if (size == 0) return 0.0;
    
    // Calculate byte frequency
    uint32_t frequency[256] = {0};
    const uint8_t* bytes = (const uint8_t*)data;
    
    for (size_t i = 0; i < size; i++) {
        frequency[bytes[i]]++;
    }
    
    // Calculate Shannon entropy
    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            double prob = (double)frequency[i] / size;
            entropy -= prob * log2(prob);
        }
    }
    
    return entropy;
}

// Core API Implementation

IntegrityValidatorError integrity_validator_create(IntegrityValidator** validator,
                                                   const IntegrityValidationConfig* config) {
    if (!validator || !config) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    IntegrityValidator* new_validator = calloc(1, sizeof(IntegrityValidator));
    if (!new_validator) {
        return INTEGRITY_VALIDATOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy configuration
    memcpy(&new_validator->config, config, sizeof(IntegrityValidationConfig));
    
    // Initialize timing infrastructure
    mach_timebase_info(&new_validator->timebase_info);
    
    // Initialize analysis buffer
    new_validator->analysis_buffer_size = 1024 * 1024; // 1MB buffer
    new_validator->numerical_analysis_buffer = calloc(new_validator->analysis_buffer_size, sizeof(double));
    if (!new_validator->numerical_analysis_buffer) {
        free(new_validator);
        return INTEGRITY_VALIDATOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize temporary validation buffer
    new_validator->temp_buffer_size = 64 * 1024; // 64KB temp buffer
    new_validator->temp_validation_buffer = calloc(new_validator->temp_buffer_size, sizeof(uint8_t));
    if (!new_validator->temp_validation_buffer) {
        free(new_validator->numerical_analysis_buffer);
        free(new_validator);
        return INTEGRITY_VALIDATOR_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize synchronization primitives
    pthread_mutex_init(&new_validator->validation_mutex, NULL);
    pthread_cond_init(&new_validator->validation_condition, NULL);
    
    // Initialize hash contexts
    CC_SHA1_Init(&new_validator->sha1_context);
    CC_SHA256_Init(&new_validator->sha256_context);
    CC_SHA512_Init(&new_validator->sha512_context);
    
    *validator = new_validator;
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_initialize(IntegrityValidator* validator) {
    if (!validator) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    // Reset performance tracking
    validator->total_validations_performed = 0;
    validator->total_bytes_validated = 0;
    validator->total_validation_time_ns = 0;
    
    // Initialize algorithm performance cache with estimated values
    validator->algorithm_performance_cache[CHECKSUM_ALGORITHM_CRC32] = CRC32_OPS_PER_SECOND_M3;
    validator->algorithm_performance_cache[CHECKSUM_ALGORITHM_CRC64] = CRC32_OPS_PER_SECOND_M3 / 2;
    validator->algorithm_performance_cache[CHECKSUM_ALGORITHM_XXHASH64] = XXHASH64_OPS_PER_SECOND_M3;
    validator->algorithm_performance_cache[CHECKSUM_ALGORITHM_SHA256] = SHA256_OPS_PER_SECOND_M3;
    validator->algorithm_performance_cache[CHECKSUM_ALGORITHM_BLAKE3] = BLAKE3_OPS_PER_SECOND_M3;
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_compute_checksum(IntegrityValidator* validator,
                                                            const void* data,
                                                            size_t data_size,
                                                            ChecksumAlgorithm algorithm,
                                                            ChecksumResult* checksum_result) {
    if (!validator || !data || !checksum_result || data_size == 0) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    memset(checksum_result, 0, sizeof(ChecksumResult));
    checksum_result->algorithm = algorithm;
    
    uint64_t start_time = get_nanosecond_timestamp();
    
    switch (algorithm) {
        case CHECKSUM_ALGORITHM_CRC32: {
            uint32_t crc = compute_crc32(data, data_size);
            checksum_result->checksum_value_64 = crc;
            checksum_result->checksum_size_bytes = 4;
            checksum_result->is_hash_algorithm = false;
            strncpy(checksum_result->algorithm_name, "CRC32", sizeof(checksum_result->algorithm_name));
            break;
        }
        
        case CHECKSUM_ALGORITHM_CRC64: {
            uint64_t crc = compute_crc64(data, data_size);
            checksum_result->checksum_value_64 = crc;
            checksum_result->checksum_size_bytes = 8;
            checksum_result->is_hash_algorithm = false;
            strncpy(checksum_result->algorithm_name, "CRC64", sizeof(checksum_result->algorithm_name));
            break;
        }
        
        case CHECKSUM_ALGORITHM_ADLER32: {
            uint32_t adler = compute_adler32(data, data_size);
            checksum_result->checksum_value_64 = adler;
            checksum_result->checksum_size_bytes = 4;
            checksum_result->is_hash_algorithm = false;
            strncpy(checksum_result->algorithm_name, "Adler32", sizeof(checksum_result->algorithm_name));
            break;
        }
        
        case CHECKSUM_ALGORITHM_SHA1: {
            CC_SHA1_CTX context;
            CC_SHA1_Init(&context);
            CC_SHA1_Update(&context, data, (CC_LONG)data_size);
            CC_SHA1_Final(checksum_result->checksum_hash, &context);
            checksum_result->checksum_size_bytes = CC_SHA1_DIGEST_LENGTH;
            checksum_result->is_hash_algorithm = true;
            strncpy(checksum_result->algorithm_name, "SHA1", sizeof(checksum_result->algorithm_name));
            break;
        }
        
        case CHECKSUM_ALGORITHM_SHA256: {
            CC_SHA256_CTX context;
            CC_SHA256_Init(&context);
            CC_SHA256_Update(&context, data, (CC_LONG)data_size);
            CC_SHA256_Final(checksum_result->checksum_hash, &context);
            checksum_result->checksum_size_bytes = CC_SHA256_DIGEST_LENGTH;
            checksum_result->is_hash_algorithm = true;
            strncpy(checksum_result->algorithm_name, "SHA256", sizeof(checksum_result->algorithm_name));
            break;
        }
        
        case CHECKSUM_ALGORITHM_SHA512: {
            CC_SHA512_CTX context;
            CC_SHA512_Init(&context);
            CC_SHA512_Update(&context, data, (CC_LONG)data_size);
            CC_SHA512_Final(checksum_result->checksum_hash, &context);
            checksum_result->checksum_size_bytes = CC_SHA512_DIGEST_LENGTH;
            checksum_result->is_hash_algorithm = true;
            strncpy(checksum_result->algorithm_name, "SHA512", sizeof(checksum_result->algorithm_name));
            break;
        }
        
        case CHECKSUM_ALGORITHM_XXHASH64: {
            uint64_t hash = compute_xxhash64(data, data_size);
            checksum_result->checksum_value_64 = hash;
            checksum_result->checksum_size_bytes = 8;
            checksum_result->is_hash_algorithm = false;
            strncpy(checksum_result->algorithm_name, "xxHash64", sizeof(checksum_result->algorithm_name));
            break;
        }
        
        case CHECKSUM_ALGORITHM_MURMUR3: {
            uint64_t hash = compute_murmur3(data, data_size);
            checksum_result->checksum_value_64 = hash;
            checksum_result->checksum_size_bytes = 8;
            checksum_result->is_hash_algorithm = false;
            strncpy(checksum_result->algorithm_name, "MurmurHash3", sizeof(checksum_result->algorithm_name));
            break;
        }
        
        default:
            return INTEGRITY_VALIDATOR_ERROR_UNSUPPORTED_ALGORITHM;
    }
    
    uint64_t end_time = get_nanosecond_timestamp();
    checksum_result->computation_time_ns = mach_time_to_nanoseconds(end_time - start_time, 
                                                                   &validator->timebase_info);
    
    // Update performance tracking
    pthread_mutex_lock(&validator->validation_mutex);
    validator->total_validations_performed++;
    validator->total_bytes_validated += data_size;
    validator->total_validation_time_ns += checksum_result->computation_time_ns;
    pthread_mutex_unlock(&validator->validation_mutex);
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_verify_checksum(IntegrityValidator* validator,
                                                           const void* data,
                                                           size_t data_size,
                                                           const ChecksumResult* expected_checksum,
                                                           bool* verification_passed) {
    if (!validator || !data || !expected_checksum || !verification_passed || data_size == 0) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    ChecksumResult computed_checksum;
    IntegrityValidatorError result = integrity_validator_compute_checksum(validator, data, data_size,
                                                                         expected_checksum->algorithm,
                                                                         &computed_checksum);
    
    if (result != INTEGRITY_VALIDATOR_SUCCESS) {
        return result;
    }
    
    // Verify checksum based on algorithm type
    if (computed_checksum.is_hash_algorithm) {
        // Compare hash values using constant-time comparison
        *verification_passed = constant_time_memcmp(computed_checksum.checksum_hash,
                                                   expected_checksum->checksum_hash,
                                                   computed_checksum.checksum_size_bytes);
    } else {
        // Compare numerical checksum values
        *verification_passed = (computed_checksum.checksum_value_64 == expected_checksum->checksum_value_64);
    }
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_compute_multiple_checksums(IntegrityValidator* validator,
                                                                       const void* data,
                                                                       size_t data_size,
                                                                       const ChecksumAlgorithm* algorithms,
                                                                       uint32_t algorithm_count,
                                                                       ChecksumResult* checksum_results) {
    if (!validator || !data || !algorithms || !checksum_results || 
        data_size == 0 || algorithm_count == 0) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    for (uint32_t i = 0; i < algorithm_count; i++) {
        IntegrityValidatorError result = integrity_validator_compute_checksum(validator, data, data_size,
                                                                             algorithms[i],
                                                                             &checksum_results[i]);
        if (result != INTEGRITY_VALIDATOR_SUCCESS) {
            return result;
        }
    }
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_validate_pre_compression(IntegrityValidator* validator,
                                                                     const void* original_data,
                                                                     size_t data_size,
                                                                     DataType data_type,
                                                                     IntegrityValidationResult* pre_compression_result) {
    if (!validator || !original_data || !pre_compression_result || data_size == 0) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    memset(pre_compression_result, 0, sizeof(IntegrityValidationResult));
    
    uint64_t validation_start_time = get_nanosecond_timestamp();
    
    // Compute primary checksum
    ChecksumResult primary_checksum;
    IntegrityValidatorError result = integrity_validator_compute_checksum(validator, original_data, data_size,
                                                                         validator->config.primary_checksum,
                                                                         &primary_checksum);
    if (result != INTEGRITY_VALIDATOR_SUCCESS) {
        return result;
    }
    
    pre_compression_result->checksum_matches = 1;
    
    // Compute secondary checksum if configured
    if (validator->config.secondary_checksum != validator->config.primary_checksum) {
        ChecksumResult secondary_checksum;
        result = integrity_validator_compute_checksum(validator, original_data, data_size,
                                                     validator->config.secondary_checksum,
                                                     &secondary_checksum);
        if (result != INTEGRITY_VALIDATOR_SUCCESS) {
            return result;
        }
        pre_compression_result->checksum_matches++;
    }
    
    // Calculate data entropy
    pre_compression_result->data_entropy_original = calculate_entropy(original_data, data_size);
    
    // Perform numerical validation if applicable
    if (data_type == DATA_TYPE_FLOAT32 || data_type == DATA_TYPE_FLOAT64 || 
        data_type == DATA_TYPE_NEURAL_WEIGHTS) {
        
        size_t element_count;
        bool is_double_precision = (data_type == DATA_TYPE_FLOAT64);
        
        if (is_double_precision) {
            element_count = data_size / sizeof(double);
        } else {
            element_count = data_size / sizeof(float);
        }
        
        // Check for numerical anomalies
        uint32_t nan_count, inf_count, denormal_count;
        result = integrity_validator_check_numerical_anomalies(validator, original_data, element_count,
                                                              is_double_precision, &nan_count, &inf_count,
                                                              &denormal_count);
        
        if (result != INTEGRITY_VALIDATOR_SUCCESS) {
            return result;
        }
        
        pre_compression_result->numerical_errors_detected = nan_count + inf_count + denormal_count;
    }
    
    // Set validation metadata
    pre_compression_result->total_bytes_validated = data_size;
    
    uint64_t validation_end_time = get_nanosecond_timestamp();
    pre_compression_result->validation_time_ns = mach_time_to_nanoseconds(
        validation_end_time - validation_start_time, &validator->timebase_info);
    
    pre_compression_result->passes_all_checks = (pre_compression_result->numerical_errors_detected == 0);
    
    snprintf(pre_compression_result->validation_summary, sizeof(pre_compression_result->validation_summary),
             "Pre-compression validation: %s, %.1f%% data entropy, %u numerical errors",
             pre_compression_result->passes_all_checks ? "PASSED" : "FAILED",
             pre_compression_result->data_entropy_original,
             pre_compression_result->numerical_errors_detected);
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_validate_post_decompression(IntegrityValidator* validator,
                                                                        const void* original_data,
                                                                        size_t original_size,
                                                                        const void* decompressed_data,
                                                                        size_t decompressed_size,
                                                                        DataType data_type,
                                                                        IntegrityValidationResult* post_decompression_result) {
    if (!validator || !original_data || !decompressed_data || !post_decompression_result ||
        original_size == 0 || decompressed_size == 0) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    memset(post_decompression_result, 0, sizeof(IntegrityValidationResult));
    
    uint64_t validation_start_time = get_nanosecond_timestamp();
    
    // Check size consistency
    if (original_size != decompressed_size) {
        post_decompression_result->passes_all_checks = false;
        snprintf(post_decompression_result->validation_summary, 
                sizeof(post_decompression_result->validation_summary),
                "Size mismatch: original %zu bytes, decompressed %zu bytes", 
                original_size, decompressed_size);
        return INTEGRITY_VALIDATOR_ERROR_SIZE_MISMATCH;
    }
    
    // Compute checksums for both datasets
    ChecksumResult original_checksum, decompressed_checksum;
    IntegrityValidatorError result;
    
    result = integrity_validator_compute_checksum(validator, original_data, original_size,
                                                 validator->config.primary_checksum,
                                                 &original_checksum);
    if (result != INTEGRITY_VALIDATOR_SUCCESS) {
        return result;
    }
    
    result = integrity_validator_compute_checksum(validator, decompressed_data, decompressed_size,
                                                 validator->config.primary_checksum,
                                                 &decompressed_checksum);
    if (result != INTEGRITY_VALIDATOR_SUCCESS) {
        return result;
    }
    
    // Verify checksum match
    bool checksum_match;
    result = integrity_validator_verify_checksum(validator, decompressed_data, decompressed_size,
                                                &original_checksum, &checksum_match);
    if (result != INTEGRITY_VALIDATOR_SUCCESS) {
        return result;
    }
    
    if (checksum_match) {
        post_decompression_result->checksum_matches = 1;
    } else {
        post_decompression_result->checksum_mismatches = 1;
        post_decompression_result->passes_all_checks = false;
    }
    
    // Calculate data entropies
    post_decompression_result->data_entropy_original = calculate_entropy(original_data, original_size);
    post_decompression_result->data_entropy_recovered = calculate_entropy(decompressed_data, decompressed_size);
    
    // Calculate data similarity score using byte-wise comparison
    size_t matching_bytes = 0;
    const uint8_t* orig_bytes = (const uint8_t*)original_data;
    const uint8_t* decomp_bytes = (const uint8_t*)decompressed_data;
    
    for (size_t i = 0; i < original_size; i++) {
        if (orig_bytes[i] == decomp_bytes[i]) {
            matching_bytes++;
        }
    }
    
    post_decompression_result->data_similarity_score = (float)matching_bytes / original_size;
    
    // Perform numerical validation if applicable
    if (data_type == DATA_TYPE_FLOAT32 || data_type == DATA_TYPE_FLOAT64 || 
        data_type == DATA_TYPE_NEURAL_WEIGHTS) {
        
        size_t element_count;
        bool is_double_precision = (data_type == DATA_TYPE_FLOAT64);
        
        if (is_double_precision) {
            element_count = original_size / sizeof(double);
        } else {
            element_count = original_size / sizeof(float);
        }
        
        // Validate numerical stability
        NumericalStabilityResult stability_result;
        result = integrity_validator_validate_numerical_stability(validator, original_data, 
                                                                 decompressed_data, element_count,
                                                                 is_double_precision, &stability_result);
        
        if (result != INTEGRITY_VALIDATOR_SUCCESS) {
            return result;
        }
        
        post_decompression_result->numerical_errors_detected = 
            stability_result.nan_values_detected + 
            stability_result.inf_values_detected + 
            stability_result.denormal_values_detected;
        
        if (!stability_result.numerical_stability_acceptable) {
            post_decompression_result->passes_all_checks = false;
        }
    }
    
    // Set validation metadata
    post_decompression_result->total_bytes_validated = original_size + decompressed_size;
    
    uint64_t validation_end_time = get_nanosecond_timestamp();
    post_decompression_result->validation_time_ns = mach_time_to_nanoseconds(
        validation_end_time - validation_start_time, &validator->timebase_info);
    
    // Final validation assessment
    post_decompression_result->passes_all_checks = 
        checksum_match && 
        (post_decompression_result->numerical_errors_detected == 0) &&
        (post_decompression_result->data_similarity_score > 0.9999f);
    
    snprintf(post_decompression_result->validation_summary, 
            sizeof(post_decompression_result->validation_summary),
             "Post-decompression validation: %s, %.6f similarity, %u/%u checksums match",
             post_decompression_result->passes_all_checks ? "PASSED" : "FAILED",
             post_decompression_result->data_similarity_score,
             post_decompression_result->checksum_matches,
             post_decompression_result->checksum_matches + post_decompression_result->checksum_mismatches);
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_validate_numerical_stability(IntegrityValidator* validator,
                                                                         const void* original_float_data,
                                                                         const void* recovered_float_data,
                                                                         size_t element_count,
                                                                         bool is_double_precision,
                                                                         NumericalStabilityResult* stability_result) {
    if (!validator || !original_float_data || !recovered_float_data || 
        !stability_result || element_count == 0) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    memset(stability_result, 0, sizeof(NumericalStabilityResult));
    
    double sum_abs_error = 0.0;
    double sum_rel_error = 0.0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    uint32_t valid_comparisons = 0;
    
    if (is_double_precision) {
        const double* orig_data = (const double*)original_float_data;
        const double* recv_data = (const double*)recovered_float_data;
        
        for (size_t i = 0; i < element_count; i++) {
            double orig = orig_data[i];
            double recv = recv_data[i];
            
            // Check for numerical anomalies
            if (isnan(recv)) {
                stability_result->nan_values_detected++;
                continue;
            }
            if (isinf(recv)) {
                stability_result->inf_values_detected++;
                continue;
            }
            if (fpclassify(recv) == FP_SUBNORMAL) {
                stability_result->denormal_values_detected++;
            }
            
            // Calculate errors
            double abs_error = fabs(orig - recv);
            sum_abs_error += abs_error;
            if (abs_error > max_abs_error) {
                max_abs_error = abs_error;
            }
            
            if (orig != 0.0) {
                double rel_error = abs_error / fabs(orig);
                sum_rel_error += rel_error;
                if (rel_error > max_rel_error) {
                    max_rel_error = rel_error;
                }
                valid_comparisons++;
            }
        }
    } else {
        const float* orig_data = (const float*)original_float_data;
        const float* recv_data = (const float*)recovered_float_data;
        
        for (size_t i = 0; i < element_count; i++) {
            float orig = orig_data[i];
            float recv = recv_data[i];
            
            // Check for numerical anomalies
            if (isnan(recv)) {
                stability_result->nan_values_detected++;
                continue;
            }
            if (isinf(recv)) {
                stability_result->inf_values_detected++;
                continue;
            }
            if (fpclassify(recv) == FP_SUBNORMAL) {
                stability_result->denormal_values_detected++;
            }
            
            // Calculate errors
            double abs_error = fabs(orig - recv);
            sum_abs_error += abs_error;
            if (abs_error > max_abs_error) {
                max_abs_error = abs_error;
            }
            
            if (orig != 0.0f) {
                double rel_error = abs_error / fabs(orig);
                sum_rel_error += rel_error;
                if (rel_error > max_rel_error) {
                    max_rel_error = rel_error;
                }
                valid_comparisons++;
            }
        }
    }
    
    // Calculate error statistics
    stability_result->mean_absolute_error = sum_abs_error / element_count;
    stability_result->max_absolute_error = max_abs_error;
    
    if (valid_comparisons > 0) {
        stability_result->relative_error_mean = sum_rel_error / valid_comparisons;
        stability_result->relative_error_max = max_rel_error;
    }
    
    // Calculate precision loss percentage
    uint32_t total_anomalies = stability_result->nan_values_detected + 
                              stability_result->inf_values_detected + 
                              stability_result->denormal_values_detected;
    stability_result->precision_loss_percentage = (float)total_anomalies / element_count * 100.0f;
    
    // Assess numerical stability
    bool acceptable_max_error = stability_result->max_absolute_error < validator->config.numerical_tolerance_epsilon * 10.0f;
    bool acceptable_mean_error = stability_result->mean_absolute_error < validator->config.numerical_tolerance_epsilon;
    bool acceptable_anomalies = total_anomalies == 0;
    bool acceptable_relative_error = stability_result->relative_error_max < 1e-6;
    
    stability_result->numerical_stability_acceptable = 
        acceptable_max_error && acceptable_mean_error && acceptable_anomalies && acceptable_relative_error;
    
    // Generate stability assessment
    if (stability_result->numerical_stability_acceptable) {
        snprintf(stability_result->stability_assessment, sizeof(stability_result->stability_assessment),
                "Excellent numerical stability: max_abs_err=%.2e, mean_abs_err=%.2e, anomalies=%u",
                stability_result->max_absolute_error,
                stability_result->mean_absolute_error,
                total_anomalies);
    } else {
        snprintf(stability_result->stability_assessment, sizeof(stability_result->stability_assessment),
                "Poor numerical stability: max_abs_err=%.2e, mean_abs_err=%.2e, anomalies=%u",
                stability_result->max_absolute_error,
                stability_result->mean_absolute_error,
                total_anomalies);
    }
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_check_numerical_anomalies(IntegrityValidator* validator,
                                                                      const void* float_data,
                                                                      size_t element_count,
                                                                      bool is_double_precision,
                                                                      uint32_t* nan_count,
                                                                      uint32_t* inf_count,
                                                                      uint32_t* denormal_count) {
    if (!validator || !float_data || !nan_count || !inf_count || !denormal_count || element_count == 0) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    *nan_count = 0;
    *inf_count = 0;
    *denormal_count = 0;
    
    if (is_double_precision) {
        const double* data = (const double*)float_data;
        for (size_t i = 0; i < element_count; i++) {
            if (isnan(data[i])) {
                (*nan_count)++;
            } else if (isinf(data[i])) {
                (*inf_count)++;
            } else if (fpclassify(data[i]) == FP_SUBNORMAL) {
                (*denormal_count)++;
            }
        }
    } else {
        const float* data = (const float*)float_data;
        for (size_t i = 0; i < element_count; i++) {
            if (isnan(data[i])) {
                (*nan_count)++;
            } else if (isinf(data[i])) {
                (*inf_count)++;
            } else if (fpclassify(data[i]) == FP_SUBNORMAL) {
                (*denormal_count)++;
            }
        }
    }
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_constant_time_compare(IntegrityValidator* validator,
                                                                 const void* data1,
                                                                 const void* data2,
                                                                 size_t data_size,
                                                                 bool* buffers_equal) {
    if (!validator || !data1 || !data2 || !buffers_equal || data_size == 0) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    *buffers_equal = constant_time_memcmp(data1, data2, data_size);
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

// Configuration functions

IntegrityValidatorError integrity_validator_create_default_config(IntegrityValidationConfig* config) {
    if (!config) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(IntegrityValidationConfig));
    
    config->validation_level = VALIDATION_LEVEL_STANDARD;
    config->primary_checksum = CHECKSUM_ALGORITHM_XXHASH64;
    config->secondary_checksum = CHECKSUM_ALGORITHM_CRC32;
    config->numerical_validation = NUMERICAL_CHECK_BASIC;
    config->enable_deep_content_validation = false;
    config->enable_compression_ratio_validation = true;
    config->enable_timing_attack_protection = false;
    config->enable_memory_pattern_validation = false;
    config->enable_entropy_validation = true;
    config->enable_statistical_validation = false;
    config->max_validation_time_ms = MAX_VALIDATION_TIME_DEFAULT_MS;
    config->numerical_tolerance_epsilon = NUMERICAL_TOLERANCE_DEFAULT;
    config->sample_validation_percentage = SAMPLE_VALIDATION_DEFAULT_PERCENTAGE;
    config->max_data_size_for_full_validation = MAX_FULL_VALIDATION_SIZE_MB * 1024 * 1024;
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

IntegrityValidatorError integrity_validator_create_paranoid_config(IntegrityValidationConfig* config) {
    if (!config) {
        return INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(IntegrityValidationConfig));
    
    config->validation_level = VALIDATION_LEVEL_PARANOID;
    config->primary_checksum = CHECKSUM_ALGORITHM_SHA256;
    config->secondary_checksum = CHECKSUM_ALGORITHM_BLAKE3;
    config->numerical_validation = NUMERICAL_CHECK_COMPREHENSIVE;
    config->enable_deep_content_validation = true;
    config->enable_compression_ratio_validation = true;
    config->enable_timing_attack_protection = true;
    config->enable_memory_pattern_validation = true;
    config->enable_entropy_validation = true;
    config->enable_statistical_validation = true;
    config->max_validation_time_ms = MAX_VALIDATION_TIME_DEFAULT_MS * 5;
    config->numerical_tolerance_epsilon = NUMERICAL_TOLERANCE_DEFAULT / 10.0f;
    config->sample_validation_percentage = 100; // Full validation
    config->max_data_size_for_full_validation = UINT64_MAX;
    
    return INTEGRITY_VALIDATOR_SUCCESS;
}

// Utility functions

const char* integrity_validator_get_error_string(IntegrityValidatorError error_code) {
    switch (error_code) {
        case INTEGRITY_VALIDATOR_SUCCESS:
            return "Success";
        case INTEGRITY_VALIDATOR_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case INTEGRITY_VALIDATOR_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case INTEGRITY_VALIDATOR_ERROR_CHECKSUM_MISMATCH:
            return "Checksum mismatch";
        case INTEGRITY_VALIDATOR_ERROR_NUMERICAL_INSTABILITY:
            return "Numerical instability detected";
        case INTEGRITY_VALIDATOR_ERROR_CONTENT_CORRUPTION:
            return "Content corruption detected";
        case INTEGRITY_VALIDATOR_ERROR_SIZE_MISMATCH:
            return "Data size mismatch";
        case INTEGRITY_VALIDATOR_ERROR_ENTROPY_VIOLATION:
            return "Entropy validation failed";
        case INTEGRITY_VALIDATOR_ERROR_STATISTICAL_ANOMALY:
            return "Statistical anomaly detected";
        case INTEGRITY_VALIDATOR_ERROR_VALIDATION_TIMEOUT:
            return "Validation timeout";
        case INTEGRITY_VALIDATOR_ERROR_UNSUPPORTED_ALGORITHM:
            return "Unsupported algorithm";
        case INTEGRITY_VALIDATOR_ERROR_INSUFFICIENT_DATA:
            return "Insufficient data for validation";
        case INTEGRITY_VALIDATOR_ERROR_CONFIGURATION_INVALID:
            return "Invalid configuration";
        default:
            return "Unknown error";
    }
}

const char* integrity_validator_get_algorithm_string(ChecksumAlgorithm algorithm) {
    switch (algorithm) {
        case CHECKSUM_ALGORITHM_CRC32: return "CRC32";
        case CHECKSUM_ALGORITHM_CRC64: return "CRC64";
        case CHECKSUM_ALGORITHM_ADLER32: return "Adler32";
        case CHECKSUM_ALGORITHM_SHA1: return "SHA1";
        case CHECKSUM_ALGORITHM_SHA256: return "SHA256";
        case CHECKSUM_ALGORITHM_SHA512: return "SHA512";
        case CHECKSUM_ALGORITHM_BLAKE3: return "BLAKE3";
        case CHECKSUM_ALGORITHM_XXHASH64: return "xxHash64";
        case CHECKSUM_ALGORITHM_MURMUR3: return "MurmurHash3";
        case CHECKSUM_ALGORITHM_CITYHASH64: return "CityHash64";
        default: return "Unknown";
    }
}

bool integrity_validator_is_cryptographic_hash(ChecksumAlgorithm algorithm) {
    switch (algorithm) {
        case CHECKSUM_ALGORITHM_SHA1:
        case CHECKSUM_ALGORITHM_SHA256:
        case CHECKSUM_ALGORITHM_SHA512:
        case CHECKSUM_ALGORITHM_BLAKE3:
            return true;
        default:
            return false;
    }
}

uint32_t integrity_validator_get_checksum_size(ChecksumAlgorithm algorithm) {
    switch (algorithm) {
        case CHECKSUM_ALGORITHM_CRC32:
        case CHECKSUM_ALGORITHM_ADLER32:
            return 4;
        case CHECKSUM_ALGORITHM_CRC64:
        case CHECKSUM_ALGORITHM_XXHASH64:
        case CHECKSUM_ALGORITHM_MURMUR3:
        case CHECKSUM_ALGORITHM_CITYHASH64:
            return 8;
        case CHECKSUM_ALGORITHM_SHA1:
            return 20;
        case CHECKSUM_ALGORITHM_SHA256:
        case CHECKSUM_ALGORITHM_BLAKE3:
            return 32;
        case CHECKSUM_ALGORITHM_SHA512:
            return 64;
        default:
            return 0;
    }
}

void integrity_validator_destroy(IntegrityValidator* validator) {
    if (!validator) return;
    
    // Clean up memory
    if (validator->numerical_analysis_buffer) {
        free(validator->numerical_analysis_buffer);
    }
    
    if (validator->temp_validation_buffer) {
        free(validator->temp_validation_buffer);
    }
    
    // Clean up synchronization primitives
    pthread_mutex_destroy(&validator->validation_mutex);
    pthread_cond_destroy(&validator->validation_condition);
    
    free(validator);
}
