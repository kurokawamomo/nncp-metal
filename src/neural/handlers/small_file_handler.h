/*
 * Small File Handler for CUDA-Compatible LSTM Processing
 * 
 * Dedicated module ensuring identical LSTM processing for all file sizes,
 * particularly critical for small files where padding and segmentation
 * strategies can significantly impact compression quality.
 * 
 * Based on original CUDA NNCP implementation requirements.
 */

#ifndef SMALL_FILE_HANDLER_H
#define SMALL_FILE_HANDLER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../compatibility/cuda_math_compat.h"
#include "../config/cuda_profiles.h"
#include "../lstm/nncp_lstm_metal_enhanced.h"
#include "../validation/cuda_parameter_validator.h"

#ifdef __cplusplus
extern "C" {
#endif

// File size categories for specialized handling
typedef enum {
    SMALL_FILE_TINY = 0,        // 0-256 bytes
    SMALL_FILE_SMALL,           // 257-1024 bytes  
    SMALL_FILE_MEDIUM,          // 1025-4096 bytes
    SMALL_FILE_LARGE,           // 4097-16384 bytes
    SMALL_FILE_VERY_LARGE,      // 16385+ bytes (handed off to standard processing)
    SMALL_FILE_CATEGORY_COUNT
} SmallFileCategory;

// Padding strategies for small files
typedef enum {
    PADDING_STRATEGY_ZERO = 0,       // Zero padding (default)
    PADDING_STRATEGY_REPEAT,         // Repeat file content
    PADDING_STRATEGY_REFLECT,        // Reflect file content
    PADDING_STRATEGY_STATISTICAL,    // Statistical padding based on content
    PADDING_STRATEGY_CUDA_COMPATIBLE, // CUDA-compatible padding (recommended)
    PADDING_STRATEGY_COUNT
} PaddingStrategy;

// Small file processing configuration
typedef struct {
    // Core processing parameters
    SmallFileCategory category;
    PaddingStrategy padding_strategy;
    int32_t target_segment_length;      // Target length after padding (must match seg_len)
    bool preserve_original_length;      // Track original length for decompression
    
    // CUDA compatibility settings
    bool enforce_cuda_alignment;        // Ensure CUDA-compatible memory alignment
    bool use_deterministic_padding;     // Use deterministic padding for reproducibility
    uint64_t padding_seed;             // Seed for statistical padding
    
    // Quality optimization
    bool enable_content_analysis;       // Analyze content for optimal padding
    bool enable_entropy_preservation;   // Preserve entropy characteristics
    float min_compression_ratio;        // Minimum acceptable compression ratio
    
    // Performance settings
    bool cache_padding_patterns;        // Cache commonly used padding patterns
    bool optimize_for_speed;           // Optimize for speed vs quality
    size_t max_memory_usage_bytes;     // Memory budget for small file processing
} SmallFileConfig;

// Small file processing context
typedef struct {
    // Configuration
    SmallFileConfig config;
    
    // CUDA-compatible LSTM context
    EnhancedMetalLSTMContext* lstm_context;
    
    // File processing state
    uint8_t* original_data;            // Original file data
    size_t original_size;              // Original file size
    uint8_t* padded_data;             // Padded data for LSTM processing
    size_t padded_size;               // Padded data size
    
    // Padding metadata
    uint8_t* padding_pattern;         // Generated padding pattern
    size_t padding_pattern_size;      // Padding pattern size
    uint32_t padding_checksum;        // Checksum for padding verification
    
    // Processing statistics
    double processing_time_ms;         // Total processing time
    float compression_ratio;           // Achieved compression ratio
    size_t total_operations;          // Total LSTM operations
    
    // Quality metrics
    float entropy_original;           // Original data entropy
    float entropy_padded;            // Padded data entropy
    float quality_score;             // Overall quality score (0-1)
    
    // Error tracking
    bool processing_successful;       // Whether processing succeeded
    char last_error_message[256];    // Last error message
} SmallFileHandler;

// Handler lifecycle management
SmallFileHandler* small_file_handler_create(const SmallFileConfig* config);
void small_file_handler_free(SmallFileHandler* handler);
bool small_file_handler_reset(SmallFileHandler* handler);

// Configuration management
SmallFileConfig* small_file_config_create_for_category(SmallFileCategory category);
SmallFileConfig* small_file_config_create_cuda_compatible(const CUDAProfile* profile);
SmallFileConfig* small_file_config_create_optimized(size_t file_size, bool prefer_quality);
void small_file_config_free(SmallFileConfig* config);

// File categorization and analysis
SmallFileCategory small_file_categorize(size_t file_size);
bool small_file_requires_special_handling(size_t file_size);
size_t small_file_estimate_processing_memory(size_t file_size, const SmallFileConfig* config);

// Core processing functions
bool small_file_handler_process(SmallFileHandler* handler, 
                               const uint8_t* input_data, 
                               size_t input_size,
                               uint8_t** output_data,
                               size_t* output_size);

bool small_file_handler_decompress(SmallFileHandler* handler,
                                  const uint8_t* compressed_data,
                                  size_t compressed_size,
                                  uint8_t** decompressed_data,
                                  size_t* decompressed_size);

// Padding strategy implementation
bool small_file_apply_padding(SmallFileHandler* handler,
                             const uint8_t* input_data,
                             size_t input_size,
                             uint8_t** padded_data,
                             size_t* padded_size);

bool small_file_remove_padding(SmallFileHandler* handler,
                              const uint8_t* padded_data,
                              size_t padded_size,
                              uint8_t** original_data,
                              size_t* original_size);

// Specific padding strategy implementations
bool small_file_pad_zero(const uint8_t* input, size_t input_size, 
                        uint8_t** output, size_t target_size);
bool small_file_pad_repeat(const uint8_t* input, size_t input_size,
                          uint8_t** output, size_t target_size);
bool small_file_pad_reflect(const uint8_t* input, size_t input_size,
                           uint8_t** output, size_t target_size);
bool small_file_pad_statistical(const uint8_t* input, size_t input_size,
                               uint8_t** output, size_t target_size,
                               uint64_t seed);
bool small_file_pad_cuda_compatible(const uint8_t* input, size_t input_size,
                                   uint8_t** output, size_t target_size,
                                   const CUDAProfile* profile);

// Content analysis and optimization
float small_file_analyze_entropy(const uint8_t* data, size_t size);
bool small_file_analyze_patterns(const uint8_t* data, size_t size,
                                uint8_t** pattern, size_t* pattern_size);
PaddingStrategy small_file_recommend_padding_strategy(const uint8_t* data, size_t size);

// CUDA compatibility validation
bool small_file_validate_cuda_compatibility(const SmallFileHandler* handler);
bool small_file_check_segment_alignment(const SmallFileHandler* handler);
bool small_file_verify_lstm_processing(SmallFileHandler* handler);

// Quality assessment
float small_file_calculate_quality_score(const SmallFileHandler* handler);
bool small_file_meets_quality_threshold(const SmallFileHandler* handler, float threshold);
void small_file_generate_quality_report(const SmallFileHandler* handler, 
                                       char* report_buffer, 
                                       size_t buffer_size);

// Metadata management for decompression
typedef struct {
    size_t original_size;              // Original file size
    SmallFileCategory category;        // File category
    PaddingStrategy padding_strategy;  // Used padding strategy
    uint32_t padding_checksum;        // Padding verification checksum
    uint8_t compression_version;      // Version for compatibility
    uint8_t reserved[7];              // Reserved for future use
} SmallFileMetadata;

bool small_file_encode_metadata(const SmallFileHandler* handler,
                               uint8_t* metadata_buffer,
                               size_t buffer_size);
bool small_file_decode_metadata(const uint8_t* metadata_buffer,
                               size_t buffer_size,
                               SmallFileMetadata* metadata);

// Performance optimization
bool small_file_optimize_memory_usage(SmallFileHandler* handler);
bool small_file_enable_caching(SmallFileHandler* handler, size_t cache_size_bytes);
void small_file_clear_cache(SmallFileHandler* handler);

// Batch processing for multiple small files
typedef struct {
    SmallFileHandler** handlers;      // Array of handlers
    size_t handler_count;            // Number of handlers
    size_t max_concurrent_files;     // Maximum concurrent processing
    bool use_shared_lstm_context;    // Share LSTM context across files
} SmallFileBatchProcessor;

SmallFileBatchProcessor* small_file_batch_processor_create(size_t max_files);
void small_file_batch_processor_free(SmallFileBatchProcessor* processor);
bool small_file_batch_process(SmallFileBatchProcessor* processor,
                             const uint8_t** input_files,
                             const size_t* input_sizes,
                             size_t file_count,
                             uint8_t*** output_files,
                             size_t** output_sizes);

// Debugging and diagnostics
void small_file_handler_print_stats(const SmallFileHandler* handler);
void small_file_handler_print_config(const SmallFileConfig* config);
bool small_file_handler_export_debug_info(const SmallFileHandler* handler,
                                         const char* filename);

// Error handling
typedef enum {
    SMALL_FILE_SUCCESS = 0,
    SMALL_FILE_ERROR_INVALID_INPUT,
    SMALL_FILE_ERROR_MEMORY_ALLOCATION,
    SMALL_FILE_ERROR_PADDING_FAILED,
    SMALL_FILE_ERROR_LSTM_PROCESSING_FAILED,
    SMALL_FILE_ERROR_CUDA_INCOMPATIBLE,
    SMALL_FILE_ERROR_QUALITY_TOO_LOW,
    SMALL_FILE_ERROR_METADATA_INVALID,
    SMALL_FILE_ERROR_DECOMPRESSION_FAILED
} SmallFileError;

const char* small_file_error_string(SmallFileError error);

// Integration with main compression pipeline
bool small_file_should_use_handler(size_t file_size, const CUDAProfile* profile);
SmallFileHandler* small_file_create_handler_for_profile(const CUDAProfile* profile);
bool small_file_integrate_with_neural_bridge(SmallFileHandler* handler,
                                            void* neural_bridge_context);

#ifdef __cplusplus
}
#endif

#endif // SMALL_FILE_HANDLER_H