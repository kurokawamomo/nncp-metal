/*
 * DynamicContextSelector.h
 * 
 * Dynamic Context Selection System
 * Implements intelligent file characteristic analysis, content type detection,
 * and optimal context length determination for adaptive compression
 */

#ifndef DYNAMIC_CONTEXT_SELECTOR_H
#define DYNAMIC_CONTEXT_SELECTOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "AdaptiveContextManager.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct DynamicContextSelector DynamicContextSelector;

// Content type classification
typedef enum {
    CONTENT_TYPE_UNKNOWN = 0,
    CONTENT_TYPE_TEXT,                 // Plain text documents
    CONTENT_TYPE_CODE,                 // Source code files
    CONTENT_TYPE_BINARY,               // Binary data files
    CONTENT_TYPE_STRUCTURED_DATA,      // JSON, XML, CSV, etc.
    CONTENT_TYPE_DOCUMENTATION,        // Documentation/markup files
    CONTENT_TYPE_LOG,                  // Log files with patterns
    CONTENT_TYPE_CONFIG,               // Configuration files
    CONTENT_TYPE_MIXED                 // Multiple content types
} ContentType;

// File characteristic patterns
typedef enum {
    FILE_PATTERN_RANDOM = 0,           // Random/unstructured content
    FILE_PATTERN_REPETITIVE,           // Highly repetitive patterns
    FILE_PATTERN_HIERARCHICAL,         // Hierarchical structure
    FILE_PATTERN_SEQUENTIAL,           // Sequential/temporal patterns
    FILE_PATTERN_TABULAR,              // Table-like structured data
    FILE_PATTERN_LINGUISTIC,           // Natural language patterns
    FILE_PATTERN_ALGORITHMIC,          // Algorithmic/computational patterns
    FILE_PATTERN_HYBRID               // Mixed pattern types
} FilePattern;

// Domain adaptation strategies
typedef enum {
    DOMAIN_ADAPTATION_GENERAL = 0,     // General purpose adaptation
    DOMAIN_ADAPTATION_CODE,            // Source code specialized
    DOMAIN_ADAPTATION_TEXT,            // Natural language specialized
    DOMAIN_ADAPTATION_DATA,            // Structured data specialized
    DOMAIN_ADAPTATION_SCIENTIFIC,      // Scientific document specialized
    DOMAIN_ADAPTATION_MULTIMEDIA,      // Multimedia content specialized
    DOMAIN_ADAPTATION_CONFIGURATION    // Configuration/settings specialized
} DomainAdaptationType;

// Context length optimization strategies
typedef enum {
    CONTEXT_OPT_CONSERVATIVE = 0,      // Conservative context selection
    CONTEXT_OPT_BALANCED,              // Balanced context selection
    CONTEXT_OPT_AGGRESSIVE,            // Aggressive larger context
    CONTEXT_OPT_ADAPTIVE,              // Dynamic adaptive selection
    CONTEXT_OPT_MEMORY_CONSTRAINED     // Memory-constrained optimization
} ContextOptimizationStrategy;

// Dynamic context selector configuration
typedef struct {
    bool enable_file_analysis;         // Enable file characteristic analysis
    bool enable_content_detection;     // Enable automatic content type detection
    bool enable_pattern_recognition;   // Enable file pattern recognition
    bool enable_domain_adaptation;     // Enable domain-specific adaptation
    bool enable_real_time_optimization; // Enable real-time optimization
    float analysis_threshold;          // Analysis confidence threshold
    uint32_t sample_size;              // File sample size for analysis (bytes)
    uint32_t min_context_length;       // Minimum context length
    uint32_t max_context_length;       // Maximum context length
    ContextOptimizationStrategy optimization_strategy; // Context optimization strategy
    DomainAdaptationType default_domain; // Default domain adaptation
} DynamicContextConfig;

// File characteristic analysis results
typedef struct {
    ContentType detected_content_type;  // Detected content type
    FilePattern primary_pattern;        // Primary file pattern
    FilePattern secondary_pattern;      // Secondary pattern (if mixed)
    float content_entropy;              // Content entropy measure
    float repetition_ratio;             // Pattern repetition ratio
    float structural_complexity;        // Structural complexity score
    float linguistic_density;           // Linguistic content density
    float code_density;                 // Code-like pattern density
    float data_regularity;              // Data structure regularity
    uint32_t average_line_length;       // Average line length
    uint32_t vocabulary_size;           // Unique token vocabulary size
    uint32_t pattern_frequency[16];     // Top pattern frequencies
    bool has_hierarchical_structure;    // Has hierarchical organization
    bool has_temporal_patterns;         // Has time-based patterns
    bool has_cross_references;          // Has internal references
} FileCharacteristics;

// Content type detection features
typedef struct {
    float text_score;                   // Text content likelihood
    float code_score;                   // Source code likelihood
    float binary_score;                 // Binary content likelihood
    float structured_data_score;        // Structured data likelihood
    float documentation_score;          // Documentation likelihood
    float log_score;                    // Log file likelihood
    float config_score;                 // Configuration likelihood
    ContentType primary_type;           // Primary detected type
    ContentType secondary_type;         // Secondary type (if mixed)
    float confidence_score;             // Detection confidence (0.0-1.0)
    uint32_t evidence_count;            // Number of supporting evidence
    bool is_mixed_content;              // Contains multiple content types
} ContentTypeAnalysis;

// Optimal context determination results
typedef struct {
    uint32_t recommended_context_length; // Recommended context length
    uint32_t minimum_effective_length;   // Minimum effective length
    uint32_t maximum_beneficial_length;  // Maximum beneficial length
    float compression_ratio_prediction;  // Predicted compression ratio
    float processing_time_estimate;      // Estimated processing time (ms)
    size_t memory_requirement_mb;        // Memory requirement estimate
    ContextOptimizationStrategy selected_strategy; // Selected optimization strategy
    DomainAdaptationType selected_domain; // Selected domain adaptation
    bool use_hierarchical_processing;    // Use hierarchical attention
    bool use_sparse_attention;          // Use sparse attention patterns
    float confidence_score;             // Recommendation confidence
} OptimalContextDetermination;

// Domain adaptation prediction
typedef struct {
    DomainAdaptationType predicted_domain; // Predicted optimal domain
    float domain_confidence;            // Domain prediction confidence
    float general_effectiveness;        // General adaptation effectiveness
    float code_effectiveness;           // Code domain effectiveness
    float text_effectiveness;           // Text domain effectiveness
    float data_effectiveness;           // Data domain effectiveness
    float scientific_effectiveness;     // Scientific domain effectiveness
    float multimedia_effectiveness;     // Multimedia domain effectiveness
    float config_effectiveness;         // Configuration domain effectiveness
    DomainAdaptationType fallback_domain; // Fallback domain if primary fails
    bool requires_specialized_processing; // Needs specialized handling
} DomainAdaptationPrediction;

// Dynamic context selector statistics
typedef struct {
    uint32_t files_analyzed;            // Total files analyzed
    uint32_t content_types_detected;    // Content types successfully detected
    uint64_t total_analysis_time_us;    // Total analysis time
    uint64_t average_analysis_time_us;  // Average analysis time per file
    float overall_accuracy;             // Overall prediction accuracy
    uint32_t successful_optimizations;  // Successful context optimizations
    uint32_t optimization_improvements; // Context optimization improvements
    float average_compression_gain;     // Average compression ratio improvement
    float average_speed_improvement;    // Average processing speed improvement
    size_t memory_savings_mb;           // Total memory savings achieved
    uint32_t domain_adaptations;        // Successful domain adaptations
} DynamicContextStats;

// Error codes for dynamic context selection
typedef enum {
    DYNAMIC_CONTEXT_SUCCESS = 0,
    DYNAMIC_CONTEXT_ERROR_INVALID_PARAM,
    DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION,
    DYNAMIC_CONTEXT_ERROR_FILE_ACCESS,
    DYNAMIC_CONTEXT_ERROR_ANALYSIS_FAILED,
    DYNAMIC_CONTEXT_ERROR_CONTENT_DETECTION_FAILED,
    DYNAMIC_CONTEXT_ERROR_PATTERN_RECOGNITION_FAILED,
    DYNAMIC_CONTEXT_ERROR_OPTIMIZATION_FAILED,
    DYNAMIC_CONTEXT_ERROR_DOMAIN_ADAPTATION_FAILED,
    DYNAMIC_CONTEXT_ERROR_INSUFFICIENT_DATA,
    DYNAMIC_CONTEXT_ERROR_INVALID_CONTENT_TYPE,
    DYNAMIC_CONTEXT_ERROR_CONFIGURATION_INVALID
} DynamicContextError;

// Core API Functions

/**
 * Create dynamic context selector
 * @param selector Pointer to store created selector
 * @param config Dynamic context configuration
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_create(DynamicContextSelector** selector,
                                          const DynamicContextConfig* config);

/**
 * Initialize dynamic context selector with adaptive context manager
 * @param selector Dynamic context selector instance
 * @param context_manager Parent adaptive context manager
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_initialize(DynamicContextSelector* selector,
                                              AdaptiveContextManager* context_manager);

/**
 * Analyze file characteristics for optimal context selection
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param characteristics Output file characteristics analysis
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_analyze_file_characteristics(DynamicContextSelector* selector,
                                                                const uint8_t* file_data,
                                                                size_t file_size,
                                                                FileCharacteristics* characteristics);

/**
 * Automatically detect content type from file data
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param content_analysis Output content type analysis
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_detect_content_type(DynamicContextSelector* selector,
                                                       const uint8_t* file_data,
                                                       size_t file_size,
                                                       ContentTypeAnalysis* content_analysis);

/**
 * Determine optimal context length for given file characteristics
 * @param selector Dynamic context selector instance
 * @param characteristics File characteristics
 * @param content_analysis Content type analysis
 * @param optimal_context Output optimal context determination
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_determine_optimal_length(DynamicContextSelector* selector,
                                                           const FileCharacteristics* characteristics,
                                                           const ContentTypeAnalysis* content_analysis,
                                                           OptimalContextDetermination* optimal_context);

/**
 * Predict optimal domain adaptation strategy
 * @param selector Dynamic context selector instance
 * @param characteristics File characteristics
 * @param content_analysis Content type analysis
 * @param domain_prediction Output domain adaptation prediction
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_predict_domain_adaptation(DynamicContextSelector* selector,
                                                             const FileCharacteristics* characteristics,
                                                             const ContentTypeAnalysis* content_analysis,
                                                             DomainAdaptationPrediction* domain_prediction);

/**
 * Perform comprehensive context selection for file
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param recommended_context_length Output recommended context length
 * @param recommended_domain Output recommended domain adaptation
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_select_optimal_context(DynamicContextSelector* selector,
                                                          const uint8_t* file_data,
                                                          size_t file_size,
                                                          uint32_t* recommended_context_length,
                                                          DomainAdaptationType* recommended_domain);

/**
 * Update context selection based on processing results
 * @param selector Dynamic context selector instance
 * @param actual_compression_ratio Achieved compression ratio
 * @param actual_processing_time Actual processing time (ms)
 * @param actual_memory_usage Actual memory usage (MB)
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_update_from_results(DynamicContextSelector* selector,
                                                       float actual_compression_ratio,
                                                       float actual_processing_time,
                                                       size_t actual_memory_usage);

/**
 * Get dynamic context selection statistics
 * @param selector Dynamic context selector instance
 * @param stats Output selection statistics
 */
void dynamic_context_get_stats(DynamicContextSelector* selector,
                               DynamicContextStats* stats);

/**
 * Update dynamic context configuration
 * @param selector Dynamic context selector instance
 * @param config New configuration settings
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_update_config(DynamicContextSelector* selector,
                                                  const DynamicContextConfig* config);

/**
 * Destroy dynamic context selector and free resources
 * @param selector Dynamic context selector instance to destroy
 */
void dynamic_context_destroy(DynamicContextSelector* selector);

// File Analysis Functions

/**
 * Analyze file entropy and complexity patterns
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param entropy_score Output entropy score
 * @param complexity_score Output complexity score
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_analyze_entropy_complexity(DynamicContextSelector* selector,
                                                              const uint8_t* file_data,
                                                              size_t file_size,
                                                              float* entropy_score,
                                                              float* complexity_score);

/**
 * Recognize structural patterns in file content
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param primary_pattern Output primary pattern type
 * @param pattern_confidence Output pattern confidence
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_recognize_file_patterns(DynamicContextSelector* selector,
                                                           const uint8_t* file_data,
                                                           size_t file_size,
                                                           FilePattern* primary_pattern,
                                                           float* pattern_confidence);

/**
 * Extract vocabulary and linguistic features
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param vocabulary_size Output vocabulary size
 * @param linguistic_score Output linguistic content score
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_extract_vocabulary_features(DynamicContextSelector* selector,
                                                               const uint8_t* file_data,
                                                               size_t file_size,
                                                               uint32_t* vocabulary_size,
                                                               float* linguistic_score);

// Content Detection Functions

/**
 * Detect text content characteristics
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param text_score Output text content score
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_detect_text_content(DynamicContextSelector* selector,
                                                       const uint8_t* file_data,
                                                       size_t file_size,
                                                       float* text_score);

/**
 * Detect source code patterns
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param code_score Output code content score
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_detect_code_patterns(DynamicContextSelector* selector,
                                                        const uint8_t* file_data,
                                                        size_t file_size,
                                                        float* code_score);

/**
 * Detect binary data characteristics
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param binary_score Output binary content score
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_detect_binary_content(DynamicContextSelector* selector,
                                                         const uint8_t* file_data,
                                                         size_t file_size,
                                                         float* binary_score);

/**
 * Detect structured data formats
 * @param selector Dynamic context selector instance
 * @param file_data File content data
 * @param file_size File size in bytes
 * @param structured_score Output structured data score
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_detect_structured_data(DynamicContextSelector* selector,
                                                          const uint8_t* file_data,
                                                          size_t file_size,
                                                          float* structured_score);

// Context Optimization Functions

/**
 * Calculate context length for repetitive content
 * @param repetition_ratio Content repetition ratio
 * @param base_context_length Base context length
 * @param optimized_length Output optimized context length
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_optimize_for_repetitive_content(float repetition_ratio,
                                                                   uint32_t base_context_length,
                                                                   uint32_t* optimized_length);

/**
 * Calculate context length for hierarchical content
 * @param structural_complexity Structural complexity score
 * @param base_context_length Base context length
 * @param optimized_length Output optimized context length
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_optimize_for_hierarchical_content(float structural_complexity,
                                                                     uint32_t base_context_length,
                                                                     uint32_t* optimized_length);

/**
 * Calculate context length for linguistic content
 * @param linguistic_density Linguistic content density
 * @param vocabulary_size Vocabulary size
 * @param optimized_length Output optimized context length
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_optimize_for_linguistic_content(float linguistic_density,
                                                                   uint32_t vocabulary_size,
                                                                   uint32_t* optimized_length);

// Configuration Functions

/**
 * Create default dynamic context configuration
 * @param config Output default configuration
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_create_default_config(DynamicContextConfig* config);

/**
 * Create memory-optimized dynamic context configuration
 * @param available_memory_mb Available memory in MB
 * @param config Output memory-optimized configuration
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_create_memory_optimized_config(size_t available_memory_mb,
                                                                  DynamicContextConfig* config);

/**
 * Create performance-optimized dynamic context configuration
 * @param target_speed_factor Target speed optimization factor
 * @param config Output performance-optimized configuration
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_create_performance_optimized_config(float target_speed_factor,
                                                                       DynamicContextConfig* config);

/**
 * Validate dynamic context configuration
 * @param config Configuration to validate
 * @param is_valid Output boolean for validity
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_validate_config(const DynamicContextConfig* config,
                                                    bool* is_valid);

// Utility Functions

/**
 * Get error string for dynamic context error code
 * @param error_code DynamicContextError code
 * @return Human-readable error message
 */
const char* dynamic_context_get_error_string(DynamicContextError error_code);

/**
 * Convert content type to string
 * @param content_type Content type
 * @return Human-readable content type name
 */
const char* dynamic_context_content_type_to_string(ContentType content_type);

/**
 * Convert file pattern to string
 * @param pattern File pattern type
 * @return Human-readable pattern name
 */
const char* dynamic_context_file_pattern_to_string(FilePattern pattern);

/**
 * Convert domain adaptation type to string
 * @param domain_type Domain adaptation type
 * @return Human-readable domain type name
 */
const char* dynamic_context_domain_type_to_string(DomainAdaptationType domain_type);

/**
 * Convert optimization strategy to string
 * @param strategy Context optimization strategy
 * @return Human-readable strategy name
 */
const char* dynamic_context_optimization_strategy_to_string(ContextOptimizationStrategy strategy);

/**
 * Estimate computational cost for dynamic context selection
 * @param file_size File size in bytes
 * @param enable_full_analysis Enable comprehensive analysis
 * @param cost_estimate Output relative cost estimate
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_estimate_analysis_cost(size_t file_size,
                                                          bool enable_full_analysis,
                                                          float* cost_estimate);

/**
 * Calculate optimal sample size for file analysis
 * @param file_size Total file size in bytes
 * @param target_accuracy Target accuracy level (0.0-1.0)
 * @param optimal_sample_size Output optimal sample size
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_calculate_optimal_sample_size(size_t file_size,
                                                                 float target_accuracy,
                                                                 uint32_t* optimal_sample_size);

// Constants for dynamic context selection
#define DYNAMIC_CONTEXT_DEFAULT_SAMPLE_SIZE 4096     // Default sample size (4KB)
#define DYNAMIC_CONTEXT_MIN_SAMPLE_SIZE 1024         // Minimum sample size (1KB)
#define DYNAMIC_CONTEXT_MAX_SAMPLE_SIZE 32768        // Maximum sample size (32KB)
#define DYNAMIC_CONTEXT_DEFAULT_THRESHOLD 0.8f       // Default analysis threshold
#define DYNAMIC_CONTEXT_MIN_CONTEXT_LENGTH 64        // Minimum context length
#define DYNAMIC_CONTEXT_MAX_CONTEXT_LENGTH 2048      // Maximum context length

// Content type detection thresholds
#define CONTENT_DETECTION_TEXT_THRESHOLD 0.7f        // Text content threshold
#define CONTENT_DETECTION_CODE_THRESHOLD 0.6f        // Code content threshold
#define CONTENT_DETECTION_BINARY_THRESHOLD 0.5f      // Binary content threshold
#define CONTENT_DETECTION_STRUCTURED_THRESHOLD 0.75f // Structured data threshold

// Pattern recognition constants
#define PATTERN_RECOGNITION_MIN_CONFIDENCE 0.6f      // Minimum pattern confidence
#define PATTERN_REPETITION_THRESHOLD 0.4f            // Pattern repetition threshold
#define PATTERN_COMPLEXITY_THRESHOLD 0.5f            // Pattern complexity threshold
#define PATTERN_HIERARCHY_THRESHOLD 0.7f             // Hierarchical pattern threshold

// Optimization constants
#define OPTIMIZATION_MEMORY_FACTOR 0.8f              // Memory optimization factor
#define OPTIMIZATION_SPEED_FACTOR 1.2f               // Speed optimization factor
#define OPTIMIZATION_QUALITY_FACTOR 1.1f             // Quality optimization factor
#define OPTIMIZATION_BALANCE_THRESHOLD 0.5f          // Balance optimization threshold

#ifdef __cplusplus
}
#endif

#endif // DYNAMIC_CONTEXT_SELECTOR_H
