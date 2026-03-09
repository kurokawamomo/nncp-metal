#include "entropy_analyzer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Constants
#define SYMBOL_COUNT 256
#define LOG2_TABLE_SIZE 65536
#define MIN_DATA_LENGTH 16
#define DEFAULT_WINDOW_SIZE 256
#define DEFAULT_CONTEXT_LENGTH 8
#define DEFAULT_BLOCK_SIZE 64

// Utility macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SAFE_LOG2(x) ((x) > 0 ? log2f((float)(x)) : 0.0f)

// Configuration functions
EntropyAnalyzerConfig entropy_analyzer_config_default(void) {
    EntropyAnalyzerConfig config = {
        .window_size = DEFAULT_WINDOW_SIZE,
        .context_length = DEFAULT_CONTEXT_LENGTH,
        .block_size = DEFAULT_BLOCK_SIZE,
        .enable_conditional = true,
        .enable_variance = true,
        .enable_mutual_info = false  // Expensive, disabled by default
    };
    return config;
}

EntropyAnalyzerConfig entropy_analyzer_config_fast(void) {
    EntropyAnalyzerConfig config = {
        .window_size = 128,
        .context_length = 4,
        .block_size = 32,
        .enable_conditional = false,
        .enable_variance = false,
        .enable_mutual_info = false
    };
    return config;
}

EntropyAnalyzerConfig entropy_analyzer_config_accurate(void) {
    EntropyAnalyzerConfig config = {
        .window_size = 512,
        .context_length = 16,
        .block_size = 128,
        .enable_conditional = true,
        .enable_variance = true,
        .enable_mutual_info = true
    };
    return config;
}

bool entropy_analyzer_config_validate(const EntropyAnalyzerConfig *config) {
    if (!config) return false;
    
    if (config->window_size < 16 || config->window_size > 8192) return false;
    if (config->context_length < 1 || config->context_length > 64) return false;
    if (config->block_size < 8 || config->block_size > config->window_size) return false;
    
    return true;
}

// Core API functions
EntropyAnalyzer* entropy_analyzer_create(const EntropyAnalyzerConfig *config) {
    if (!config || !entropy_analyzer_config_validate(config)) {
        return NULL;
    }
    
    EntropyAnalyzer *analyzer = calloc(1, sizeof(EntropyAnalyzer));
    if (!analyzer) return NULL;
    
    analyzer->config = *config;
    
    // Allocate frequency tables
    analyzer->symbol_frequencies = calloc(SYMBOL_COUNT, sizeof(uint32_t));
    analyzer->context_frequencies = calloc(SYMBOL_COUNT, sizeof(uint32_t));
    
    if (!analyzer->symbol_frequencies || !analyzer->context_frequencies) {
        entropy_analyzer_destroy(analyzer);
        return NULL;
    }
    
    // Allocate conditional frequency table if needed
    if (config->enable_conditional) {
        analyzer->conditional_frequencies = malloc(SYMBOL_COUNT * sizeof(uint32_t*));
        if (!analyzer->conditional_frequencies) {
            entropy_analyzer_destroy(analyzer);
            return NULL;
        }
        
        for (int i = 0; i < SYMBOL_COUNT; i++) {
            analyzer->conditional_frequencies[i] = calloc(SYMBOL_COUNT, sizeof(uint32_t));
            if (!analyzer->conditional_frequencies[i]) {
                entropy_analyzer_destroy(analyzer);
                return NULL;
            }
        }
    }
    
    // Allocate analysis buffer
    analyzer->buffer_size = config->window_size * 2;  // Double buffering
    analyzer->analysis_buffer = malloc(analyzer->buffer_size);
    if (!analyzer->analysis_buffer) {
        entropy_analyzer_destroy(analyzer);
        return NULL;
    }
    
    // Precompute log2 table for performance
    if (entropy_analyzer_precompute_tables(analyzer) != ENTROPY_ANALYZER_SUCCESS) {
        entropy_analyzer_destroy(analyzer);
        return NULL;
    }
    
    return analyzer;
}

void entropy_analyzer_destroy(EntropyAnalyzer *analyzer) {
    if (!analyzer) return;
    
    free(analyzer->symbol_frequencies);
    free(analyzer->context_frequencies);
    free(analyzer->analysis_buffer);
    free(analyzer->precomputed_log2_table);
    
    if (analyzer->conditional_frequencies) {
        for (int i = 0; i < SYMBOL_COUNT; i++) {
            free(analyzer->conditional_frequencies[i]);
        }
        free(analyzer->conditional_frequencies);
    }
    
    free(analyzer);
}

// Performance optimization
EntropyAnalyzerResult entropy_analyzer_precompute_tables(EntropyAnalyzer *analyzer) {
    if (!analyzer) return ENTROPY_ANALYZER_ERROR_INVALID_PARAM;
    
    analyzer->precomputed_log2_table = malloc(LOG2_TABLE_SIZE * sizeof(float));
    if (!analyzer->precomputed_log2_table) {
        return ENTROPY_ANALYZER_ERROR_MEMORY_ALLOCATION;
    }
    
    // Precompute log2 values for fast entropy calculation
    for (int i = 0; i < LOG2_TABLE_SIZE; i++) {
        analyzer->precomputed_log2_table[i] = (i > 0) ? log2f((float)i) : 0.0f;
    }
    
    return ENTROPY_ANALYZER_SUCCESS;
}

// Fast log2 lookup
static inline float fast_log2(const EntropyAnalyzer *analyzer, uint32_t value) {
    if (value == 0) return 0.0f;
    if (value < LOG2_TABLE_SIZE) {
        return analyzer->precomputed_log2_table[value];
    }
    return log2f((float)value);
}

// Analysis functions
EntropyAnalyzerResult entropy_analyzer_process_data(
    EntropyAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length
) {
    if (!analyzer || !data || data_length < MIN_DATA_LENGTH) {
        return ENTROPY_ANALYZER_ERROR_INVALID_PARAM;
    }
    
    // Reset frequency tables
    memset(analyzer->symbol_frequencies, 0, SYMBOL_COUNT * sizeof(uint32_t));
    memset(analyzer->context_frequencies, 0, SYMBOL_COUNT * sizeof(uint32_t));
    
    if (analyzer->conditional_frequencies) {
        for (int i = 0; i < SYMBOL_COUNT; i++) {
            memset(analyzer->conditional_frequencies[i], 0, SYMBOL_COUNT * sizeof(uint32_t));
        }
    }
    
    // Build frequency tables
    for (size_t i = 0; i < data_length; i++) {
        analyzer->symbol_frequencies[data[i]]++;
        
        // Build conditional frequency table
        if (analyzer->config.enable_conditional && i >= analyzer->config.context_length) {
            uint8_t context = data[i - analyzer->config.context_length];
            uint8_t symbol = data[i];
            analyzer->conditional_frequencies[context][symbol]++;
            analyzer->context_frequencies[context]++;
        }
    }
    
    // Calculate metrics
    return entropy_analyzer_calculate_metrics(analyzer, data, data_length, &analyzer->current_metrics);
}

EntropyAnalyzerResult entropy_analyzer_calculate_metrics(
    EntropyAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    EntropyMetrics *metrics
) {
    if (!analyzer || !data || !metrics || data_length < MIN_DATA_LENGTH) {
        return ENTROPY_ANALYZER_ERROR_INVALID_PARAM;
    }
    
    memset(metrics, 0, sizeof(EntropyMetrics));
    
    // Calculate Shannon entropy
    metrics->shannon_entropy = calculate_shannon_entropy(
        analyzer->symbol_frequencies, SYMBOL_COUNT, (uint32_t)data_length);
    
    // Count unique symbols
    for (int i = 0; i < SYMBOL_COUNT; i++) {
        if (analyzer->symbol_frequencies[i] > 0) {
            metrics->unique_symbols++;
        }
    }
    
    // Calculate symbol distribution uniformity
    if (metrics->unique_symbols > 0) {
        float max_entropy = fast_log2(analyzer, (uint32_t)metrics->unique_symbols);
        metrics->symbol_distribution = (max_entropy > 0) ? metrics->shannon_entropy / max_entropy : 0.0f;
    }
    
    // Calculate conditional entropy if enabled
    if (analyzer->config.enable_conditional && analyzer->conditional_frequencies) {
        metrics->conditional_entropy = calculate_conditional_entropy_from_frequencies(
            (const uint32_t**)analyzer->conditional_frequencies,
            analyzer->context_frequencies,
            SYMBOL_COUNT,
            SYMBOL_COUNT
        );
    }
    
    // Calculate entropy variance if enabled
    if (analyzer->config.enable_variance) {
        size_t num_blocks = data_length / analyzer->config.block_size;
        if (num_blocks > 1) {
            float variance_sum = 0.0f;
            
            for (size_t block = 0; block < num_blocks; block++) {
                size_t block_start = block * analyzer->config.block_size;
                size_t block_end = MIN(block_start + analyzer->config.block_size, data_length);
                
                float block_entropy = entropy_analyzer_local_entropy(
                    data, data_length, block_start, block_end - block_start);
                
                float diff = block_entropy - metrics->shannon_entropy;
                variance_sum += diff * diff;
            }
            
            metrics->entropy_variance = variance_sum / num_blocks;
        }
    }
    
    // Estimate compression potential
    metrics->compression_potential = entropy_analyzer_compression_potential(metrics);
    
    // Update running average
    analyzer->analysis_count++;
    float alpha = 1.0f / analyzer->analysis_count;
    
    analyzer->running_average.shannon_entropy = 
        (1.0f - alpha) * analyzer->running_average.shannon_entropy + alpha * metrics->shannon_entropy;
    analyzer->running_average.entropy_variance = 
        (1.0f - alpha) * analyzer->running_average.entropy_variance + alpha * metrics->entropy_variance;
    
    return ENTROPY_ANALYZER_SUCCESS;
}

// Real-time analysis functions
float entropy_analyzer_local_entropy(
    const uint8_t *data,
    size_t data_length,
    size_t position,
    size_t context_size
) {
    if (!data || position >= data_length || context_size == 0) return 0.0f;
    
    size_t start = (position >= context_size) ? position - context_size : 0;
    size_t end = MIN(position + context_size, data_length);
    size_t length = end - start;
    
    if (length < 2) return 0.0f;
    
    // Count symbol frequencies in local context
    uint32_t frequencies[SYMBOL_COUNT] = {0};
    for (size_t i = start; i < end; i++) {
        frequencies[data[i]]++;
    }
    
    return calculate_shannon_entropy(frequencies, SYMBOL_COUNT, (uint32_t)length);
}

float entropy_analyzer_conditional_entropy(
    const uint8_t *data,
    size_t data_length,
    size_t context_length
) {
    if (!data || data_length < context_length + 1) return 0.0f;
    
    uint32_t context_freq[SYMBOL_COUNT] = {0};
    uint32_t conditional_freq[SYMBOL_COUNT][SYMBOL_COUNT] = {{0}};
    
    // Build conditional frequency tables
    for (size_t i = context_length; i < data_length; i++) {
        uint8_t context = data[i - context_length];
        uint8_t symbol = data[i];
        conditional_freq[context][symbol]++;
        context_freq[context]++;
    }
    
    return calculate_conditional_entropy_from_frequencies(
        (const uint32_t**)conditional_freq, context_freq, SYMBOL_COUNT, SYMBOL_COUNT);
}

// Statistical calculation functions
float calculate_shannon_entropy(const uint32_t *frequencies, size_t num_symbols, uint32_t total_count) {
    if (!frequencies || total_count == 0) return 0.0f;
    
    float entropy = 0.0f;
    for (size_t i = 0; i < num_symbols; i++) {
        if (frequencies[i] > 0) {
            float probability = (float)frequencies[i] / total_count;
            entropy -= probability * log2f(probability);
        }
    }
    
    return entropy;
}

float calculate_conditional_entropy_from_frequencies(
    const uint32_t **conditional_freq,
    const uint32_t *context_freq,
    size_t num_contexts,
    size_t num_symbols
) {
    if (!conditional_freq || !context_freq) return 0.0f;
    
    float conditional_entropy = 0.0f;
    uint32_t total_count = 0;
    
    // Calculate total count
    for (size_t i = 0; i < num_contexts; i++) {
        total_count += context_freq[i];
    }
    
    if (total_count == 0) return 0.0f;
    
    // Calculate conditional entropy
    for (size_t context = 0; context < num_contexts; context++) {
        if (context_freq[context] > 0) {
            float context_prob = (float)context_freq[context] / total_count;
            float context_entropy = calculate_shannon_entropy(
                conditional_freq[context], num_symbols, context_freq[context]);
            conditional_entropy += context_prob * context_entropy;
        }
    }
    
    return conditional_entropy;
}

// Utility functions
float entropy_analyzer_compression_potential(const EntropyMetrics *metrics) {
    if (!metrics) return 0.0f;
    
    // Estimate compression potential based on entropy
    float theoretical_limit = metrics->shannon_entropy / 8.0f;  // Convert to bytes
    
    // Adjust for symbol distribution and variance
    float distribution_factor = 1.0f - metrics->symbol_distribution;
    float variance_factor = 1.0f / (1.0f + metrics->entropy_variance);
    
    return theoretical_limit * distribution_factor * variance_factor;
}

EntropyMetrics entropy_analyzer_get_current_metrics(const EntropyAnalyzer *analyzer) {
    EntropyMetrics empty = {0};
    return analyzer ? analyzer->current_metrics : empty;
}

EntropyMetrics entropy_analyzer_get_average_metrics(const EntropyAnalyzer *analyzer) {
    EntropyMetrics empty = {0};
    return analyzer ? analyzer->running_average : empty;
}

void entropy_analyzer_reset(EntropyAnalyzer *analyzer) {
    if (!analyzer) return;
    
    memset(&analyzer->current_metrics, 0, sizeof(EntropyMetrics));
    memset(&analyzer->running_average, 0, sizeof(EntropyMetrics));
    analyzer->analysis_count = 0;
    analyzer->cache_valid = false;
}

bool entropy_analyzer_is_ready(const EntropyAnalyzer *analyzer) {
    return analyzer && analyzer->precomputed_log2_table;
}

// Content characterization
bool entropy_analyzer_is_random_data(const EntropyMetrics *metrics) {
    if (!metrics) return false;
    return metrics->shannon_entropy > 7.5f && metrics->symbol_distribution > 0.9f;
}

bool entropy_analyzer_is_structured_data(const EntropyMetrics *metrics) {
    if (!metrics) return false;
    return metrics->entropy_variance > 1.0f && metrics->symbol_distribution < 0.7f;
}

bool entropy_analyzer_is_text_data(const EntropyMetrics *metrics) {
    if (!metrics) return false;
    return metrics->shannon_entropy >= 4.0f && metrics->shannon_entropy <= 6.5f && 
           metrics->unique_symbols >= 20 && metrics->unique_symbols <= 100;
}

float entropy_analyzer_predict_compression_ratio(const EntropyMetrics *metrics) {
    if (!metrics) return 1.0f;
    
    // Basic compression ratio prediction based on entropy
    float entropy_ratio = metrics->shannon_entropy / 8.0f;
    
    // Adjust for various factors
    if (entropy_analyzer_is_random_data(metrics)) {
        return 0.95f;  // Very little compression possible
    } else if (entropy_analyzer_is_text_data(metrics)) {
        return entropy_ratio * 0.7f;  // Text compresses well
    } else if (entropy_analyzer_is_structured_data(metrics)) {
        return entropy_ratio * 0.5f;  // Structured data compresses very well
    }
    
    return entropy_ratio * 0.8f;  // General case
}

// Integration with prediction scoring
PredictionEntropyInfo entropy_analyzer_analyze_for_prediction(
    EntropyAnalyzer *analyzer,
    const uint8_t *context,
    size_t context_length,
    uint8_t candidate_byte
) {
    PredictionEntropyInfo info = {0};
    
    if (!analyzer || !context || context_length == 0) {
        return info;
    }
    
    // Calculate local entropy around the context
    info.entropy_score = entropy_analyzer_local_entropy(
        context, context_length, context_length - 1, 
        MIN(analyzer->config.context_length, context_length));
    
    // Calculate context relevance
    uint32_t candidate_count = 0;
    for (size_t i = 0; i < context_length; i++) {
        if (context[i] == candidate_byte) {
            candidate_count++;
        }
    }
    
    info.context_relevance = (float)candidate_count / context_length;
    
    // Estimate prediction confidence based on entropy and context
    info.prediction_confidence = (8.0f - info.entropy_score) / 8.0f * info.context_relevance;
    info.is_predictable = info.prediction_confidence > 0.3f;
    
    return info;
}
