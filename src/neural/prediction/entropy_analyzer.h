#ifndef ENTROPY_ANALYZER_H
#define ENTROPY_ANALYZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct EntropyAnalyzer EntropyAnalyzer;
typedef struct EntropyMetrics EntropyMetrics;

// Entropy metrics structure
struct EntropyMetrics {
    float shannon_entropy;        // Shannon entropy in bits
    float conditional_entropy;    // Conditional entropy for prediction
    float mutual_information;     // Mutual information between contexts
    float entropy_variance;       // Variance in local entropy
    float compression_potential;  // Estimated compression potential
    size_t unique_symbols;        // Number of unique symbols
    float symbol_distribution;    // Symbol distribution uniformity
};

// Entropy analyzer configuration
typedef struct {
    size_t window_size;          // Analysis window size (default: 256)
    size_t context_length;       // Context length for conditional entropy
    size_t block_size;           // Block size for variance calculation
    bool enable_conditional;     // Enable conditional entropy calculation
    bool enable_variance;        // Enable entropy variance calculation
    bool enable_mutual_info;     // Enable mutual information calculation
} EntropyAnalyzerConfig;

// Entropy analyzer structure
struct EntropyAnalyzer {
    EntropyAnalyzerConfig config;
    
    // Symbol frequency tracking
    uint32_t *symbol_frequencies;
    uint32_t *context_frequencies;
    uint32_t **conditional_frequencies;  // 2D array for conditional probabilities
    
    // Analysis buffers
    uint8_t *analysis_buffer;
    size_t buffer_size;
    size_t buffer_position;
    
    // Metrics tracking
    EntropyMetrics current_metrics;
    EntropyMetrics running_average;
    uint64_t analysis_count;
    
    // Performance optimization
    float *precomputed_log2_table;
    bool cache_valid;
};

// Return codes
typedef enum {
    ENTROPY_ANALYZER_SUCCESS = 0,
    ENTROPY_ANALYZER_ERROR_INVALID_PARAM = -1,
    ENTROPY_ANALYZER_ERROR_MEMORY_ALLOCATION = -2,
    ENTROPY_ANALYZER_ERROR_INSUFFICIENT_DATA = -3,
    ENTROPY_ANALYZER_ERROR_INVALID_WINDOW = -4
} EntropyAnalyzerResult;

// Core API functions
EntropyAnalyzer* entropy_analyzer_create(const EntropyAnalyzerConfig *config);
void entropy_analyzer_destroy(EntropyAnalyzer *analyzer);

// Analysis functions
EntropyAnalyzerResult entropy_analyzer_process_data(
    EntropyAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length
);

EntropyAnalyzerResult entropy_analyzer_calculate_metrics(
    EntropyAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    EntropyMetrics *metrics
);

// Window-based analysis
EntropyAnalyzerResult entropy_analyzer_sliding_window(
    EntropyAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    size_t window_size,
    EntropyMetrics **window_metrics,
    size_t *num_windows
);

// Real-time analysis for compression
float entropy_analyzer_local_entropy(
    const uint8_t *data,
    size_t data_length,
    size_t position,
    size_t context_size
);

float entropy_analyzer_conditional_entropy(
    const uint8_t *data,
    size_t data_length,
    size_t context_length
);

float entropy_analyzer_compression_potential(
    const EntropyMetrics *metrics
);

// Utility functions
EntropyMetrics entropy_analyzer_get_current_metrics(const EntropyAnalyzer *analyzer);
EntropyMetrics entropy_analyzer_get_average_metrics(const EntropyAnalyzer *analyzer);

void entropy_analyzer_reset(EntropyAnalyzer *analyzer);
bool entropy_analyzer_is_ready(const EntropyAnalyzer *analyzer);

// Configuration helpers
EntropyAnalyzerConfig entropy_analyzer_config_default(void);
EntropyAnalyzerConfig entropy_analyzer_config_fast(void);      // Fast analysis
EntropyAnalyzerConfig entropy_analyzer_config_accurate(void);  // Accurate analysis
bool entropy_analyzer_config_validate(const EntropyAnalyzerConfig *config);

// Performance optimization functions
EntropyAnalyzerResult entropy_analyzer_precompute_tables(EntropyAnalyzer *analyzer);
void entropy_analyzer_invalidate_cache(EntropyAnalyzer *analyzer);

// Statistical functions for advanced analysis
float calculate_shannon_entropy(const uint32_t *frequencies, size_t num_symbols, uint32_t total_count);
float calculate_conditional_entropy_from_frequencies(
    const uint32_t **conditional_freq,
    const uint32_t *context_freq,
    size_t num_contexts,
    size_t num_symbols
);
float calculate_mutual_information(
    const uint32_t *joint_freq,
    const uint32_t *marginal_x,
    const uint32_t *marginal_y,
    size_t size_x,
    size_t size_y,
    uint32_t total_count
);

// Symbol distribution analysis
typedef struct {
    uint8_t symbol;
    uint32_t frequency;
    float probability;
    float information_content;
} SymbolInfo;

EntropyAnalyzerResult entropy_analyzer_get_symbol_distribution(
    EntropyAnalyzer *analyzer,
    SymbolInfo **symbols,
    size_t *num_symbols
);

// Content characterization helpers
bool entropy_analyzer_is_random_data(const EntropyMetrics *metrics);
bool entropy_analyzer_is_structured_data(const EntropyMetrics *metrics);
bool entropy_analyzer_is_text_data(const EntropyMetrics *metrics);
float entropy_analyzer_predict_compression_ratio(const EntropyMetrics *metrics);

// Integration with prediction scoring
typedef struct {
    float entropy_score;
    float context_relevance;
    float prediction_confidence;
    bool is_predictable;
} PredictionEntropyInfo;

PredictionEntropyInfo entropy_analyzer_analyze_for_prediction(
    EntropyAnalyzer *analyzer,
    const uint8_t *context,
    size_t context_length,
    uint8_t candidate_byte
);

#ifdef __cplusplus
}
#endif

#endif // ENTROPY_ANALYZER_H
