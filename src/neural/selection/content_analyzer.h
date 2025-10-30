#ifndef CONTENT_ANALYZER_H
#define CONTENT_ANALYZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct ContentAnalyzer ContentAnalyzer;
typedef struct ContentCharacteristics ContentCharacteristics;
typedef struct ContentPattern ContentPattern;

// Content type enumeration
typedef enum {
    CONTENT_TYPE_UNKNOWN = 0,
    CONTENT_TYPE_NATURAL_LANGUAGE,
    CONTENT_TYPE_SOURCE_CODE,
    CONTENT_TYPE_DOCUMENTATION,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_XML,
    CONTENT_TYPE_CSV,
    CONTENT_TYPE_BINARY,
    CONTENT_TYPE_COMPRESSED,
    CONTENT_TYPE_RANDOM,
    CONTENT_TYPE_MIXED
} ContentType;

// Language detection for text content
typedef enum {
    LANGUAGE_UNKNOWN = 0,
    LANGUAGE_ENGLISH,
    LANGUAGE_PROGRAMMING_C,
    LANGUAGE_PROGRAMMING_PYTHON,
    LANGUAGE_PROGRAMMING_JAVASCRIPT,
    LANGUAGE_PROGRAMMING_JAVA,
    LANGUAGE_MARKUP_HTML,
    LANGUAGE_MARKUP_XML,
    LANGUAGE_DATA_JSON,
    LANGUAGE_DATA_CSV,
    LANGUAGE_OTHER
} LanguageType;

// Pattern structure for detected patterns
struct ContentPattern {
    uint8_t *pattern_bytes;
    size_t pattern_length;
    size_t frequency;
    size_t first_occurrence;
    size_t last_occurrence;
    float confidence;
};

// Comprehensive content characteristics
struct ContentCharacteristics {
    // Basic metrics
    size_t total_bytes;
    size_t unique_bytes;
    float entropy;
    float compression_ratio_estimate;
    
    // Content type analysis
    ContentType primary_type;
    ContentType secondary_type;
    float type_confidence;
    LanguageType detected_language;
    float language_confidence;
    
    // Text analysis
    size_t printable_chars;
    size_t whitespace_chars;
    size_t control_chars;
    size_t extended_chars;
    float text_probability;
    
    // Structural analysis
    size_t line_count;
    size_t word_count;
    float average_line_length;
    float line_length_variance;
    bool has_consistent_structure;
    
    // Pattern analysis
    size_t repeated_sequences;
    size_t unique_patterns;
    float pattern_regularity;
    float repetition_factor;
    ContentPattern *top_patterns;
    size_t pattern_count;
    
    // Byte distribution
    uint32_t byte_frequencies[256];
    float byte_distribution_entropy;
    float distribution_skewness;
    float distribution_kurtosis;
    
    // Structural markers
    bool has_json_markers;
    bool has_xml_markers;
    bool has_csv_markers;
    bool has_binary_markers;
    bool has_compressed_markers;
    
    // Complexity metrics
    float syntactic_complexity;
    float semantic_complexity;
    float predictability_score;
    float compressibility_score;
    
    // Performance indicators
    float transformer_suitability;
    float lstm_suitability;
    float hybrid_suitability;
    float analysis_confidence;
};

// Configuration for content analyzer
typedef struct {
    size_t max_analysis_size;       // Maximum bytes to analyze (default: 4096)
    size_t min_pattern_length;      // Minimum pattern length to detect (default: 2)
    size_t max_pattern_length;      // Maximum pattern length to detect (default: 16)
    size_t max_patterns_to_track;   // Maximum patterns to remember (default: 100)
    bool enable_language_detection; // Enable programming language detection
    bool enable_deep_analysis;      // Enable computationally expensive analysis
    bool enable_structural_analysis; // Enable structure-specific analysis
    float confidence_threshold;     // Minimum confidence for type detection
} ContentAnalyzerConfig;

// Content analyzer main structure
struct ContentAnalyzer {
    ContentAnalyzerConfig config;
    
    // Analysis state
    uint64_t total_analyses;
    uint64_t successful_analyses;
    float average_analysis_time;
    
    // Pattern detection cache
    ContentPattern *pattern_cache;
    size_t cache_size;
    size_t cache_capacity;
    
    // Language detection models (simplified)
    float language_models[16][256];    // Basic byte frequency models
    bool models_initialized;
    
    // Performance tracking
    uint64_t text_detections;
    uint64_t binary_detections;
    uint64_t structured_detections;
    uint64_t compressed_detections;
};

// Return codes
typedef enum {
    CONTENT_ANALYZER_SUCCESS = 0,
    CONTENT_ANALYZER_ERROR_INVALID_PARAM = -1,
    CONTENT_ANALYZER_ERROR_MEMORY_ALLOCATION = -2,
    CONTENT_ANALYZER_ERROR_INSUFFICIENT_DATA = -3,
    CONTENT_ANALYZER_ERROR_ANALYSIS_FAILED = -4,
    CONTENT_ANALYZER_ERROR_MODEL_NOT_READY = -5
} ContentAnalyzerResult;

// Core API functions
ContentAnalyzer* content_analyzer_create(const ContentAnalyzerConfig *config);
void content_analyzer_destroy(ContentAnalyzer *analyzer);

// Primary analysis function
ContentAnalyzerResult content_analyzer_analyze_content(
    ContentAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    ContentCharacteristics *characteristics
);

// Specialized analysis functions
ContentAnalyzerResult content_analyzer_detect_content_type(
    ContentAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    ContentType *primary_type,
    ContentType *secondary_type,
    float *confidence
);

ContentAnalyzerResult content_analyzer_detect_language(
    ContentAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    LanguageType *language,
    float *confidence
);

ContentAnalyzerResult content_analyzer_analyze_patterns(
    ContentAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    ContentPattern **patterns,
    size_t *pattern_count
);

ContentAnalyzerResult content_analyzer_calculate_entropy_distribution(
    ContentAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    float *entropy_distribution,
    size_t distribution_size
);

// Content type utilities
bool content_analyzer_is_text_data(const ContentCharacteristics *characteristics);
bool content_analyzer_is_structured_data(const ContentCharacteristics *characteristics);
bool content_analyzer_is_binary_data(const ContentCharacteristics *characteristics);
bool content_analyzer_is_compressed_data(const ContentCharacteristics *characteristics);
bool content_analyzer_is_random_data(const ContentCharacteristics *characteristics);

// Algorithm suitability assessment
float content_analyzer_transformer_suitability(const ContentCharacteristics *characteristics);
float content_analyzer_lstm_suitability(const ContentCharacteristics *characteristics);
float content_analyzer_hybrid_suitability(const ContentCharacteristics *characteristics);

// Performance prediction
ContentAnalyzerResult content_analyzer_predict_compression_performance(
    ContentAnalyzer *analyzer,
    const ContentCharacteristics *characteristics,
    float *transformer_ratio_estimate,
    float *lstm_ratio_estimate,
    float *confidence
);

// Configuration helpers
ContentAnalyzerConfig content_analyzer_config_default(void);
ContentAnalyzerConfig content_analyzer_config_fast(void);
ContentAnalyzerConfig content_analyzer_config_thorough(void);
bool content_analyzer_config_validate(const ContentAnalyzerConfig *config);

// Statistics and monitoring
typedef struct {
    uint64_t total_analyses;
    uint64_t successful_analyses;
    uint64_t text_detections;
    uint64_t binary_detections;
    uint64_t structured_detections;
    uint64_t compressed_detections;
    uint64_t random_detections;
    float average_analysis_time;
    float average_confidence;
    size_t patterns_detected;
    size_t languages_detected;
} ContentAnalyzerStats;

ContentAnalyzerStats content_analyzer_get_stats(const ContentAnalyzer *analyzer);
void content_analyzer_reset_stats(ContentAnalyzer *analyzer);

// Cache management
void content_analyzer_clear_pattern_cache(ContentAnalyzer *analyzer);
size_t content_analyzer_get_cache_size(const ContentAnalyzer *analyzer);

// Batch analysis for performance testing
ContentAnalyzerResult content_analyzer_batch_analyze(
    ContentAnalyzer *analyzer,
    const uint8_t **data_samples,
    const size_t *sample_lengths,
    size_t sample_count,
    ContentCharacteristics *results,
    float *average_processing_time
);

// Debug and introspection
typedef struct {
    float byte_frequency_histogram[256];
    float pattern_length_distribution[64];
    float entropy_progression[32];
    float complexity_metrics[8];
    char debug_info[1024];
} ContentAnalysisDebugInfo;

ContentAnalyzerResult content_analyzer_get_debug_info(
    ContentAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    ContentAnalysisDebugInfo *debug_info
);

#ifdef __cplusplus
}
#endif

#endif // CONTENT_ANALYZER_H
