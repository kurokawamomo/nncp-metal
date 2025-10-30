#ifndef ENHANCED_SELECTOR_H
#define ENHANCED_SELECTOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct EnhancedSelector EnhancedSelector;
typedef struct ContentAnalysis ContentAnalysis;

// Content analysis structure - enhanced version
struct ContentAnalysis {
    // Entropy metrics
    float average_entropy;
    float entropy_variance;
    float local_entropy_variance;
    float conditional_entropy;
    
    // Pattern characteristics
    float repetition_ratio;
    float pattern_density;
    float pattern_complexity;
    float sequence_regularity;
    
    // Text characteristics
    bool has_text_patterns;
    bool has_structured_data;
    bool has_random_data;
    bool has_binary_data;
    
    // Data distribution
    float symbol_distribution_uniformity;
    float byte_frequency_skewness;
    float compression_potential_estimate;
    
    // Algorithm performance prediction
    float transformer_predicted_ratio;
    float transformer_confidence;
    float lstm_predicted_ratio;
    float lstm_confidence;
    float confidence_level;
    
    // Content type classification
    float text_probability;
    float binary_probability;
    float structured_probability;
    float compressed_probability;
};

// Algorithm recommendation enumeration
typedef enum {
    SELECTION_TRANSFORMER_FAVORED,
    SELECTION_LSTM_FAVORED,
    SELECTION_HYBRID_APPROACH,
    SELECTION_ANALYSIS_NEEDED,
    SELECTION_ERROR
} AlgorithmRecommendation;

// Enhanced selector configuration
typedef struct {
    size_t analysis_window_size;    // Size of data to analyze (default: 1024)
    size_t min_analysis_size;       // Minimum data size for analysis (default: 64)
    bool enable_deep_analysis;      // Enable computationally expensive analysis
    bool enable_ml_prediction;      // Enable ML-based algorithm prediction
    float entropy_weight;           // Weight for entropy-based selection
    float pattern_weight;           // Weight for pattern-based selection
    float performance_weight;       // Weight for predicted performance
    float confidence_threshold;     // Minimum confidence for recommendation
} EnhancedSelectorConfig;

// Enhanced selector structure
struct EnhancedSelector {
    EnhancedSelectorConfig config;
    
    // Analysis components
    void *entropy_analyzer;         // EntropyAnalyzer instance
    void *pattern_matcher;          // PatternMatcher instance
    
    // Content classification models
    float *text_classification_weights;
    float *binary_classification_weights;
    size_t classification_model_size;
    
    // Performance prediction
    float transformer_performance_model[16];
    float lstm_performance_model[16];
    bool models_trained;
    
    // Analysis cache
    ContentAnalysis cached_analysis;
    bool cache_valid;
    uint32_t cache_data_hash;
    
    // Statistics
    uint64_t total_analyses;
    uint64_t transformer_selections;
    uint64_t lstm_selections;
    uint64_t hybrid_selections;
    float average_confidence;
};

// Return codes
typedef enum {
    ENHANCED_SELECTOR_SUCCESS = 0,
    ENHANCED_SELECTOR_ERROR_INVALID_PARAM = -1,
    ENHANCED_SELECTOR_ERROR_MEMORY_ALLOCATION = -2,
    ENHANCED_SELECTOR_ERROR_INSUFFICIENT_DATA = -3,
    ENHANCED_SELECTOR_ERROR_ANALYSIS_FAILED = -4,
    ENHANCED_SELECTOR_ERROR_MODEL_NOT_TRAINED = -5
} EnhancedSelectorResult;

// Core API functions
EnhancedSelector* enhanced_selector_create(const EnhancedSelectorConfig *config);
void enhanced_selector_destroy(EnhancedSelector *selector);

// Content analysis functions
EnhancedSelectorResult enhanced_selector_analyze_content(
    EnhancedSelector *selector,
    const uint8_t *data,
    size_t data_length,
    ContentAnalysis *analysis
);

EnhancedSelectorResult enhanced_selector_quick_analysis(
    EnhancedSelector *selector,
    const uint8_t *data,
    size_t data_length,
    ContentAnalysis *analysis
);

// Algorithm selection functions
AlgorithmRecommendation enhanced_selector_recommend_algorithm(
    EnhancedSelector *selector,
    const ContentAnalysis *analysis
);

EnhancedSelectorResult enhanced_selector_select_best_algorithm(
    EnhancedSelector *selector,
    const uint8_t *data,
    size_t data_length,
    AlgorithmRecommendation *recommendation,
    float *confidence
);

// Performance prediction functions
EnhancedSelectorResult enhanced_selector_predict_compression_ratio(
    EnhancedSelector *selector,
    const ContentAnalysis *analysis,
    float *transformer_ratio,
    float *lstm_ratio
);

float enhanced_selector_estimate_processing_time(
    EnhancedSelector *selector,
    const ContentAnalysis *analysis,
    AlgorithmRecommendation algorithm,
    size_t data_size
);

// Model training and updates
EnhancedSelectorResult enhanced_selector_update_performance_model(
    EnhancedSelector *selector,
    const ContentAnalysis *analysis,
    AlgorithmRecommendation algorithm,
    float actual_compression_ratio,
    float actual_processing_time
);

// Configuration helpers
EnhancedSelectorConfig enhanced_selector_config_default(void);
EnhancedSelectorConfig enhanced_selector_config_fast(void);
EnhancedSelectorConfig enhanced_selector_config_accurate(void);
bool enhanced_selector_config_validate(const EnhancedSelectorConfig *config);

// Content characterization utilities
bool enhanced_selector_is_text_content(const ContentAnalysis *analysis);
bool enhanced_selector_is_binary_content(const ContentAnalysis *analysis);
bool enhanced_selector_is_structured_content(const ContentAnalysis *analysis);
bool enhanced_selector_is_random_content(const ContentAnalysis *analysis);
bool enhanced_selector_is_compressed_content(const ContentAnalysis *analysis);

// Integration with existing compression selector
typedef struct {
    bool use_entropy_analysis;
    bool use_pattern_analysis;
    bool use_performance_prediction;
    float entropy_threshold_transformer;
    float entropy_threshold_lstm;
    float pattern_density_threshold;
} IntegrationConfig;

EnhancedSelectorResult enhanced_selector_integrate_with_existing(
    EnhancedSelector *selector,
    const uint8_t *data,
    size_t data_length,
    const IntegrationConfig *integration_config,
    int *existing_algorithm_choice,  // 0=LSTM, 1=Transformer
    float *enhancement_confidence
);

// Statistics and debugging
typedef struct {
    uint64_t total_analyses;
    uint64_t transformer_recommendations;
    uint64_t lstm_recommendations;
    uint64_t hybrid_recommendations;
    float average_confidence;
    float transformer_accuracy;
    float lstm_accuracy;
    size_t cache_hits;
    size_t cache_misses;
} EnhancedSelectorStats;

EnhancedSelectorStats enhanced_selector_get_stats(const EnhancedSelector *selector);
void enhanced_selector_reset_stats(EnhancedSelector *selector);

// Cache management
void enhanced_selector_clear_cache(EnhancedSelector *selector);
bool enhanced_selector_is_cache_valid(const EnhancedSelector *selector, const uint8_t *data, size_t length);

// Advanced analysis features
typedef struct {
    float entropy_distribution[256];   // Entropy histogram
    float pattern_lengths[64];         // Pattern length distribution
    float byte_transitions[256][256];  // Byte transition matrix (sparse)
    float compression_score_components[8]; // Detailed scoring breakdown
} AdvancedAnalysisResults;

EnhancedSelectorResult enhanced_selector_advanced_analysis(
    EnhancedSelector *selector,
    const uint8_t *data,
    size_t data_length,
    AdvancedAnalysisResults *results
);

#ifdef __cplusplus
}
#endif

#endif // ENHANCED_SELECTOR_H
