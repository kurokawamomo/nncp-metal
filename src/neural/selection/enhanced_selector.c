#include "enhanced_selector.h"
#include "../prediction/entropy_analyzer.h"
#include "../prediction/pattern_matcher.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Constants
#define DEFAULT_ANALYSIS_WINDOW_SIZE 1024
#define DEFAULT_MIN_ANALYSIS_SIZE 64
#define DEFAULT_ENTROPY_WEIGHT 0.4f
#define DEFAULT_PATTERN_WEIGHT 0.3f
#define DEFAULT_PERFORMANCE_WEIGHT 0.3f
#define DEFAULT_CONFIDENCE_THRESHOLD 0.6f

// Hash function for cache validation
static uint32_t hash_data(const uint8_t *data, size_t length) {
    uint32_t hash = 5381;
    for (size_t i = 0; i < length; i++) {
        hash = ((hash << 5) + hash) + data[i];
    }
    return hash;
}

// Configuration functions
EnhancedSelectorConfig enhanced_selector_config_default(void) {
    EnhancedSelectorConfig config = {
        .analysis_window_size = DEFAULT_ANALYSIS_WINDOW_SIZE,
        .min_analysis_size = DEFAULT_MIN_ANALYSIS_SIZE,
        .enable_deep_analysis = true,
        .enable_ml_prediction = true,
        .entropy_weight = DEFAULT_ENTROPY_WEIGHT,
        .pattern_weight = DEFAULT_PATTERN_WEIGHT,
        .performance_weight = DEFAULT_PERFORMANCE_WEIGHT,
        .confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    };
    return config;
}

EnhancedSelectorConfig enhanced_selector_config_fast(void) {
    EnhancedSelectorConfig config = {
        .analysis_window_size = 512,
        .min_analysis_size = 32,
        .enable_deep_analysis = false,
        .enable_ml_prediction = false,
        .entropy_weight = 0.5f,
        .pattern_weight = 0.5f,
        .performance_weight = 0.0f,
        .confidence_threshold = 0.4f
    };
    return config;
}

EnhancedSelectorConfig enhanced_selector_config_accurate(void) {
    EnhancedSelectorConfig config = {
        .analysis_window_size = 2048,
        .min_analysis_size = 128,
        .enable_deep_analysis = true,
        .enable_ml_prediction = true,
        .entropy_weight = 0.3f,
        .pattern_weight = 0.3f,
        .performance_weight = 0.4f,
        .confidence_threshold = 0.8f
    };
    return config;
}

bool enhanced_selector_config_validate(const EnhancedSelectorConfig *config) {
    if (!config) return false;
    
    if (config->analysis_window_size < 32 || config->analysis_window_size > 8192) return false;
    if (config->min_analysis_size < 16 || config->min_analysis_size > config->analysis_window_size) return false;
    if (config->entropy_weight < 0.0f || config->entropy_weight > 1.0f) return false;
    if (config->pattern_weight < 0.0f || config->pattern_weight > 1.0f) return false;
    if (config->performance_weight < 0.0f || config->performance_weight > 1.0f) return false;
    if (config->confidence_threshold < 0.0f || config->confidence_threshold > 1.0f) return false;
    
    float total_weight = config->entropy_weight + config->pattern_weight + config->performance_weight;
    if (total_weight <= 0.0f || total_weight > 3.0f) return false;
    
    return true;
}

// Core API functions
EnhancedSelector* enhanced_selector_create(const EnhancedSelectorConfig *config) {
    if (!config || !enhanced_selector_config_validate(config)) {
        return NULL;
    }
    
    EnhancedSelector *selector = calloc(1, sizeof(EnhancedSelector));
    if (!selector) return NULL;
    
    selector->config = *config;
    
    // Create entropy analyzer
    EntropyAnalyzerConfig entropy_config = entropy_analyzer_config_default();
    entropy_config.window_size = config->analysis_window_size;
    entropy_config.enable_conditional = config->enable_deep_analysis;
    entropy_config.enable_variance = config->enable_deep_analysis;
    
    selector->entropy_analyzer = entropy_analyzer_create(&entropy_config);
    if (!selector->entropy_analyzer) {
        enhanced_selector_destroy(selector);
        return NULL;
    }
    
    // Create pattern matcher
    PatternMatcherConfig pattern_config = config->enable_deep_analysis ? 
        pattern_matcher_config_accurate() : pattern_matcher_config_fast();
    
    selector->pattern_matcher = pattern_matcher_create(&pattern_config);
    if (!selector->pattern_matcher) {
        enhanced_selector_destroy(selector);
        return NULL;
    }
    
    // Initialize performance models with simple heuristics
    // Transformer model (favors structured data with medium entropy)
    float transformer_model[16] = {
        0.3f, 0.2f, 0.4f, -0.1f,  // entropy, variance, patterns, randomness
        0.2f, 0.3f, -0.2f, 0.1f,  // text, binary, structured, compressed
        0.1f, 0.1f, 0.1f, 0.1f,   // complexity, regularity, distribution, potential
        0.0f, 0.0f, 0.0f, 0.0f    // reserved
    };
    memcpy(selector->transformer_performance_model, transformer_model, sizeof(transformer_model));
    
    // LSTM model (favors sequential data with good patterns)
    float lstm_model[16] = {
        0.2f, 0.1f, 0.5f, -0.2f,  // entropy, variance, patterns, randomness
        0.4f, -0.1f, 0.2f, 0.1f,  // text, binary, structured, compressed
        0.2f, 0.3f, 0.1f, 0.2f,   // complexity, regularity, distribution, potential
        0.0f, 0.0f, 0.0f, 0.0f    // reserved
    };
    memcpy(selector->lstm_performance_model, lstm_model, sizeof(lstm_model));
    
    selector->models_trained = false;  // Will be true after first updates
    
    return selector;
}

void enhanced_selector_destroy(EnhancedSelector *selector) {
    if (!selector) return;
    
    if (selector->entropy_analyzer) {
        entropy_analyzer_destroy((EntropyAnalyzer*)selector->entropy_analyzer);
    }
    
    if (selector->pattern_matcher) {
        pattern_matcher_destroy((PatternMatcher*)selector->pattern_matcher);
    }
    
    free(selector->text_classification_weights);
    free(selector->binary_classification_weights);
    free(selector);
}

// Content analysis functions
EnhancedSelectorResult enhanced_selector_analyze_content(
    EnhancedSelector *selector,
    const uint8_t *data,
    size_t data_length,
    ContentAnalysis *analysis
) {
    if (!selector || !data || !analysis || data_length < selector->config.min_analysis_size) {
        return ENHANCED_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    // Check cache
    uint32_t data_hash = hash_data(data, data_length);
    if (selector->cache_valid && selector->cache_data_hash == data_hash) {
        *analysis = selector->cached_analysis;
        return ENHANCED_SELECTOR_SUCCESS;
    }
    
    memset(analysis, 0, sizeof(ContentAnalysis));
    
    // Limit analysis size for performance
    size_t analysis_size = (data_length > selector->config.analysis_window_size) ? 
                          selector->config.analysis_window_size : data_length;
    
    // Entropy analysis
    EntropyAnalyzer *entropy_analyzer = (EntropyAnalyzer*)selector->entropy_analyzer;
    entropy_analyzer_process_data(entropy_analyzer, data, analysis_size);
    EntropyMetrics entropy_metrics = entropy_analyzer_get_current_metrics(entropy_analyzer);
    
    analysis->average_entropy = entropy_metrics.shannon_entropy;
    analysis->entropy_variance = entropy_metrics.entropy_variance;
    analysis->conditional_entropy = entropy_metrics.conditional_entropy;
    analysis->symbol_distribution_uniformity = entropy_metrics.symbol_distribution;
    analysis->compression_potential_estimate = entropy_metrics.compression_potential;
    
    // Pattern analysis
    PatternMatcher *pattern_matcher = (PatternMatcher*)selector->pattern_matcher;
    pattern_matcher_add_data(pattern_matcher, data, analysis_size);
    
    PatternMatcherStats pattern_stats = pattern_matcher_get_stats(pattern_matcher);
    analysis->pattern_density = (float)pattern_stats.total_patterns / analysis_size;
    analysis->repetition_ratio = pattern_stats.hit_ratio;
    
    // Content type classification
    analysis->has_random_data = entropy_analyzer_is_random_data(&entropy_metrics);
    analysis->has_structured_data = entropy_analyzer_is_structured_data(&entropy_metrics);
    analysis->has_text_patterns = entropy_analyzer_is_text_data(&entropy_metrics);
    analysis->has_binary_data = (!analysis->has_text_patterns && !analysis->has_structured_data && 
                                !analysis->has_random_data);
    
    // Calculate probabilities
    analysis->text_probability = analysis->has_text_patterns ? 0.8f : 0.2f;
    analysis->binary_probability = analysis->has_binary_data ? 0.8f : 0.2f;
    analysis->structured_probability = analysis->has_structured_data ? 0.8f : 0.2f;
    analysis->compressed_probability = analysis->has_random_data ? 0.8f : 0.2f;
    
    // Performance prediction using simple models
    if (selector->config.enable_ml_prediction) {
        float features[16] = {
            analysis->average_entropy / 8.0f,
            analysis->entropy_variance,
            analysis->pattern_density,
            analysis->has_random_data ? 1.0f : 0.0f,
            analysis->text_probability,
            analysis->binary_probability,
            analysis->structured_probability,
            analysis->compressed_probability,
            analysis->pattern_density,
            analysis->repetition_ratio,
            analysis->symbol_distribution_uniformity,
            analysis->compression_potential_estimate,
            0.0f, 0.0f, 0.0f, 0.0f
        };
        
        // Simple dot product prediction
        float transformer_score = 0.0f;
        float lstm_score = 0.0f;
        
        for (int i = 0; i < 16; i++) {
            transformer_score += features[i] * selector->transformer_performance_model[i];
            lstm_score += features[i] * selector->lstm_performance_model[i];
        }
        
        // Convert scores to compression ratios (0.0 to 1.0)
        analysis->transformer_predicted_ratio = 0.3f + 0.4f * (1.0f / (1.0f + expf(-transformer_score)));
        analysis->lstm_predicted_ratio = 0.3f + 0.4f * (1.0f / (1.0f + expf(-lstm_score)));
        
        // Calculate confidence based on score difference
        float score_diff = fabsf(transformer_score - lstm_score);
        analysis->confidence_level = fminf(0.95f, 0.5f + score_diff * 0.3f);
        
        analysis->transformer_confidence = transformer_score > lstm_score ? analysis->confidence_level : 1.0f - analysis->confidence_level;
        analysis->lstm_confidence = lstm_score > transformer_score ? analysis->confidence_level : 1.0f - analysis->confidence_level;
    } else {
        // Simple heuristic predictions
        if (analysis->has_text_patterns) {
            analysis->lstm_predicted_ratio = 0.4f;
            analysis->transformer_predicted_ratio = 0.5f;
        } else if (analysis->has_structured_data) {
            analysis->transformer_predicted_ratio = 0.3f;
            analysis->lstm_predicted_ratio = 0.4f;
        } else {
            analysis->transformer_predicted_ratio = 0.6f;
            analysis->lstm_predicted_ratio = 0.6f;
        }
        analysis->confidence_level = 0.6f;
        analysis->transformer_confidence = 0.6f;
        analysis->lstm_confidence = 0.6f;
    }
    
    // Cache the results
    selector->cached_analysis = *analysis;
    selector->cache_valid = true;
    selector->cache_data_hash = data_hash;
    
    selector->total_analyses++;
    selector->average_confidence = (selector->average_confidence * (selector->total_analyses - 1) + 
                                   analysis->confidence_level) / selector->total_analyses;
    
    return ENHANCED_SELECTOR_SUCCESS;
}

EnhancedSelectorResult enhanced_selector_quick_analysis(
    EnhancedSelector *selector,
    const uint8_t *data,
    size_t data_length,
    ContentAnalysis *analysis
) {
    if (!selector || !data || !analysis || data_length < selector->config.min_analysis_size) {
        return ENHANCED_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    memset(analysis, 0, sizeof(ContentAnalysis));
    
    // Quick entropy calculation
    size_t sample_size = (data_length > 256) ? 256 : data_length;
    analysis->average_entropy = entropy_analyzer_local_entropy(data, data_length, 0, sample_size);
    
    // Basic pattern detection
    uint32_t pattern_count = 0;
    for (size_t i = 0; i < sample_size - 2; i++) {
        for (size_t j = i + 2; j < sample_size; j++) {
            if (data[i] == data[j] && data[i+1] == data[j+1]) {
                pattern_count++;
                break;
            }
        }
    }
    analysis->pattern_density = (float)pattern_count / sample_size;
    
    // Quick classification
    analysis->has_text_patterns = (analysis->average_entropy >= 4.0f && analysis->average_entropy <= 6.5f);
    analysis->has_random_data = (analysis->average_entropy > 7.5f);
    analysis->has_structured_data = (analysis->pattern_density > 0.3f);
    analysis->has_binary_data = (!analysis->has_text_patterns && !analysis->has_random_data);
    
    // Quick predictions
    if (analysis->has_text_patterns) {
        analysis->lstm_predicted_ratio = 0.4f;
        analysis->transformer_predicted_ratio = 0.5f;
    } else if (analysis->has_structured_data) {
        analysis->transformer_predicted_ratio = 0.3f;
        analysis->lstm_predicted_ratio = 0.4f;
    } else {
        analysis->transformer_predicted_ratio = 0.6f;
        analysis->lstm_predicted_ratio = 0.6f;
    }
    
    analysis->confidence_level = 0.5f;
    return ENHANCED_SELECTOR_SUCCESS;
}

// Algorithm selection functions
AlgorithmRecommendation enhanced_selector_recommend_algorithm(
    EnhancedSelector *selector,
    const ContentAnalysis *analysis
) {
    if (!selector || !analysis) {
        return SELECTION_ERROR;
    }
    
    // Calculate weighted scores
    float transformer_score = 0.0f;
    float lstm_score = 0.0f;
    
    // Entropy-based scoring
    if (selector->config.entropy_weight > 0.0f) {
        float entropy_transformer = (analysis->average_entropy >= 4.0f && analysis->average_entropy <= 6.0f) ? 1.0f : 0.5f;
        float entropy_lstm = (analysis->average_entropy >= 3.0f && analysis->average_entropy <= 7.0f) ? 1.0f : 0.5f;
        
        transformer_score += entropy_transformer * selector->config.entropy_weight;
        lstm_score += entropy_lstm * selector->config.entropy_weight;
    }
    
    // Pattern-based scoring
    if (selector->config.pattern_weight > 0.0f) {
        float pattern_transformer = analysis->has_structured_data ? 1.0f : 0.6f;
        float pattern_lstm = analysis->has_text_patterns ? 1.0f : 0.6f;
        
        transformer_score += pattern_transformer * selector->config.pattern_weight;
        lstm_score += pattern_lstm * selector->config.pattern_weight;
    }
    
    // Performance-based scoring
    if (selector->config.performance_weight > 0.0f && selector->config.enable_ml_prediction) {
        float performance_transformer = 1.0f - analysis->transformer_predicted_ratio;
        float performance_lstm = 1.0f - analysis->lstm_predicted_ratio;
        
        transformer_score += performance_transformer * selector->config.performance_weight;
        lstm_score += performance_lstm * selector->config.performance_weight;
    }
    
    // Make recommendation
    float score_diff = fabsf(transformer_score - lstm_score);
    float confidence = fminf(0.95f, 0.5f + score_diff * 0.5f);
    
    if (confidence < selector->config.confidence_threshold) {
        return SELECTION_ANALYSIS_NEEDED;
    }
    
    if (transformer_score > lstm_score) {
        selector->transformer_selections++;
        return SELECTION_TRANSFORMER_FAVORED;
    } else {
        selector->lstm_selections++;
        return SELECTION_LSTM_FAVORED;
    }
}

EnhancedSelectorResult enhanced_selector_select_best_algorithm(
    EnhancedSelector *selector,
    const uint8_t *data,
    size_t data_length,
    AlgorithmRecommendation *recommendation,
    float *confidence
) {
    if (!selector || !data || !recommendation || !confidence) {
        return ENHANCED_SELECTOR_ERROR_INVALID_PARAM;
    }
    
    ContentAnalysis analysis;
    EnhancedSelectorResult result = enhanced_selector_analyze_content(
        selector, data, data_length, &analysis);
    
    if (result != ENHANCED_SELECTOR_SUCCESS) {
        return result;
    }
    
    *recommendation = enhanced_selector_recommend_algorithm(selector, &analysis);
    *confidence = analysis.confidence_level;
    
    return ENHANCED_SELECTOR_SUCCESS;
}

// Statistics
EnhancedSelectorStats enhanced_selector_get_stats(const EnhancedSelector *selector) {
    EnhancedSelectorStats stats = {0};
    
    if (selector) {
        stats.total_analyses = selector->total_analyses;
        stats.transformer_recommendations = selector->transformer_selections;
        stats.lstm_recommendations = selector->lstm_selections;
        stats.hybrid_recommendations = selector->hybrid_selections;
        stats.average_confidence = selector->average_confidence;
        stats.cache_hits = selector->cache_valid ? 1 : 0;  // Simplified
    }
    
    return stats;
}

void enhanced_selector_reset_stats(EnhancedSelector *selector) {
    if (!selector) return;
    
    selector->total_analyses = 0;
    selector->transformer_selections = 0;
    selector->lstm_selections = 0;
    selector->hybrid_selections = 0;
    selector->average_confidence = 0.0f;
}

// Content characterization utilities
bool enhanced_selector_is_text_content(const ContentAnalysis *analysis) {
    return analysis && analysis->has_text_patterns;
}

bool enhanced_selector_is_binary_content(const ContentAnalysis *analysis) {
    return analysis && analysis->has_binary_data;
}

bool enhanced_selector_is_structured_content(const ContentAnalysis *analysis) {
    return analysis && analysis->has_structured_data;
}

bool enhanced_selector_is_random_content(const ContentAnalysis *analysis) {
    return analysis && analysis->has_random_data;
}

// Cache management
void enhanced_selector_clear_cache(EnhancedSelector *selector) {
    if (!selector) return;
    selector->cache_valid = false;
    selector->cache_data_hash = 0;
}

bool enhanced_selector_is_cache_valid(const EnhancedSelector *selector, const uint8_t *data, size_t length) {
    if (!selector || !data) return false;
    
    uint32_t data_hash = hash_data(data, length);
    return selector->cache_valid && selector->cache_data_hash == data_hash;
}
