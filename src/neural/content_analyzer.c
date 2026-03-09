#include "content_analyzer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <assert.h>

// Constants
#define DEFAULT_MAX_ANALYSIS_SIZE 4096
#define DEFAULT_MIN_PATTERN_LENGTH 2
#define DEFAULT_MAX_PATTERN_LENGTH 16
#define DEFAULT_MAX_PATTERNS_TO_TRACK 100
#define DEFAULT_CONFIDENCE_THRESHOLD 0.7f

#define JSON_MARKERS "{}[]:,"
#define XML_MARKERS "<>/=\"'"
#define CSV_SEPARATORS ",;\t|"
#define NEWLINE_CHARS "\r\n"

// Utility functions
static float calculate_entropy(const uint32_t *frequencies, size_t total_count) {
    float entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (frequencies[i] > 0) {
            float probability = (float)frequencies[i] / total_count;
            entropy -= probability * log2f(probability);
        }
    }
    return entropy;
}

static float calculate_variance(const float *values, size_t count, float mean) {
    if (count < 2) return 0.0f;
    
    float variance = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        variance += diff * diff;
    }
    return variance / (count - 1);
}

static bool is_printable_text_char(uint8_t c) {
    return (c >= 32 && c <= 126) || c == '\t' || c == '\n' || c == '\r';
}

static bool has_json_structure(const uint8_t *data, size_t length) {
    size_t json_chars = 0;
    size_t total_chars = 0;
    
    for (size_t i = 0; i < length && i < 1024; i++) {
        if (strchr(JSON_MARKERS, data[i])) {
            json_chars++;
        }
        if (data[i] != 0) {
            total_chars++;
        }
    }
    
    return total_chars > 0 && (float)json_chars / total_chars > 0.1f;
}

static bool has_xml_structure(const uint8_t *data, size_t length) {
    size_t xml_chars = 0;
    size_t total_chars = 0;
    bool has_tag_pattern = false;
    
    for (size_t i = 0; i < length && i < 1024; i++) {
        if (strchr(XML_MARKERS, data[i])) {
            xml_chars++;
        }
        if (i < length - 1 && data[i] == '<' && (isalpha(data[i+1]) || data[i+1] == '/')) {
            has_tag_pattern = true;
        }
        if (data[i] != 0) {
            total_chars++;
        }
    }
    
    return has_tag_pattern && total_chars > 0 && (float)xml_chars / total_chars > 0.05f;
}

static bool has_csv_structure(const uint8_t *data, size_t length) {
    size_t separator_count = 0;
    size_t line_count = 0;
    
    for (size_t i = 0; i < length && i < 1024; i++) {
        if (strchr(CSV_SEPARATORS, data[i])) {
            separator_count++;
        }
        if (strchr(NEWLINE_CHARS, data[i])) {
            line_count++;
        }
    }
    
    return line_count > 1 && separator_count > line_count;
}

// Configuration functions
ContentAnalyzerConfig content_analyzer_config_default(void) {
    ContentAnalyzerConfig config = {
        .max_analysis_size = DEFAULT_MAX_ANALYSIS_SIZE,
        .min_pattern_length = DEFAULT_MIN_PATTERN_LENGTH,
        .max_pattern_length = DEFAULT_MAX_PATTERN_LENGTH,
        .max_patterns_to_track = DEFAULT_MAX_PATTERNS_TO_TRACK,
        .enable_language_detection = true,
        .enable_deep_analysis = true,
        .enable_structural_analysis = true,
        .confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    };
    return config;
}

ContentAnalyzerConfig content_analyzer_config_fast(void) {
    ContentAnalyzerConfig config = {
        .max_analysis_size = 1024,
        .min_pattern_length = 2,
        .max_pattern_length = 8,
        .max_patterns_to_track = 50,
        .enable_language_detection = false,
        .enable_deep_analysis = false,
        .enable_structural_analysis = true,
        .confidence_threshold = 0.5f
    };
    return config;
}

ContentAnalyzerConfig content_analyzer_config_thorough(void) {
    ContentAnalyzerConfig config = {
        .max_analysis_size = 8192,
        .min_pattern_length = 2,
        .max_pattern_length = 32,
        .max_patterns_to_track = 200,
        .enable_language_detection = true,
        .enable_deep_analysis = true,
        .enable_structural_analysis = true,
        .confidence_threshold = 0.8f
    };
    return config;
}

bool content_analyzer_config_validate(const ContentAnalyzerConfig *config) {
    if (!config) return false;
    
    if (config->max_analysis_size < 64 || config->max_analysis_size > 65536) return false;
    if (config->min_pattern_length < 1 || config->min_pattern_length > config->max_pattern_length) return false;
    if (config->max_pattern_length < config->min_pattern_length || config->max_pattern_length > 256) return false;
    if (config->max_patterns_to_track < 1 || config->max_patterns_to_track > 1000) return false;
    if (config->confidence_threshold < 0.0f || config->confidence_threshold > 1.0f) return false;
    
    return true;
}

// Core API functions
ContentAnalyzer* content_analyzer_create(const ContentAnalyzerConfig *config) {
    if (!config || !content_analyzer_config_validate(config)) {
        return NULL;
    }
    
    ContentAnalyzer *analyzer = calloc(1, sizeof(ContentAnalyzer));
    if (!analyzer) return NULL;
    
    analyzer->config = *config;
    
    // Initialize pattern cache
    analyzer->cache_capacity = config->max_patterns_to_track;
    analyzer->pattern_cache = calloc(analyzer->cache_capacity, sizeof(ContentPattern));
    if (!analyzer->pattern_cache) {
        content_analyzer_destroy(analyzer);
        return NULL;
    }
    
    // Initialize language models with simple byte frequency patterns
    if (config->enable_language_detection) {
        // English text model (higher frequency for common letters)
        for (int i = 0; i < 256; i++) {
            analyzer->language_models[LANGUAGE_ENGLISH][i] = 0.1f;
        }
        // Boost common English letters
        analyzer->language_models[LANGUAGE_ENGLISH]['e'] = 12.7f;
        analyzer->language_models[LANGUAGE_ENGLISH]['t'] = 9.1f;
        analyzer->language_models[LANGUAGE_ENGLISH]['a'] = 8.2f;
        analyzer->language_models[LANGUAGE_ENGLISH]['o'] = 7.5f;
        analyzer->language_models[LANGUAGE_ENGLISH]['i'] = 7.0f;
        analyzer->language_models[LANGUAGE_ENGLISH]['n'] = 6.7f;
        analyzer->language_models[LANGUAGE_ENGLISH]['s'] = 6.3f;
        analyzer->language_models[LANGUAGE_ENGLISH]['h'] = 6.1f;
        analyzer->language_models[LANGUAGE_ENGLISH]['r'] = 6.0f;
        analyzer->language_models[LANGUAGE_ENGLISH][' '] = 18.0f;
        
        // C programming model (higher frequency for C-specific characters)
        for (int i = 0; i < 256; i++) {
            analyzer->language_models[LANGUAGE_PROGRAMMING_C][i] = 0.1f;
        }
        analyzer->language_models[LANGUAGE_PROGRAMMING_C]['{'] = 5.0f;
        analyzer->language_models[LANGUAGE_PROGRAMMING_C]['}'] = 5.0f;
        analyzer->language_models[LANGUAGE_PROGRAMMING_C]['('] = 4.0f;
        analyzer->language_models[LANGUAGE_PROGRAMMING_C][')'] = 4.0f;
        analyzer->language_models[LANGUAGE_PROGRAMMING_C][';'] = 6.0f;
        analyzer->language_models[LANGUAGE_PROGRAMMING_C]['='] = 3.0f;
        analyzer->language_models[LANGUAGE_PROGRAMMING_C]['_'] = 3.0f;
        
        // JSON model
        for (int i = 0; i < 256; i++) {
            analyzer->language_models[LANGUAGE_DATA_JSON][i] = 0.1f;
        }
        analyzer->language_models[LANGUAGE_DATA_JSON]['{'] = 8.0f;
        analyzer->language_models[LANGUAGE_DATA_JSON]['}'] = 8.0f;
        analyzer->language_models[LANGUAGE_DATA_JSON]['['] = 6.0f;
        analyzer->language_models[LANGUAGE_DATA_JSON][']'] = 6.0f;
        analyzer->language_models[LANGUAGE_DATA_JSON]['"'] = 10.0f;
        analyzer->language_models[LANGUAGE_DATA_JSON][':'] = 7.0f;
        analyzer->language_models[LANGUAGE_DATA_JSON][','] = 9.0f;
        
        analyzer->models_initialized = true;
    }
    
    return analyzer;
}

void content_analyzer_destroy(ContentAnalyzer *analyzer) {
    if (!analyzer) return;
    
    if (analyzer->pattern_cache) {
        for (size_t i = 0; i < analyzer->cache_size; i++) {
            free(analyzer->pattern_cache[i].pattern_bytes);
        }
        free(analyzer->pattern_cache);
    }
    
    free(analyzer);
}

// Primary analysis function
ContentAnalyzerResult content_analyzer_analyze_content(
    ContentAnalyzer *analyzer,
    const uint8_t *data,
    size_t data_length,
    ContentCharacteristics *characteristics
) {
    if (!analyzer || !data || !characteristics || data_length == 0) {
        return CONTENT_ANALYZER_ERROR_INVALID_PARAM;
    }
    
    memset(characteristics, 0, sizeof(ContentCharacteristics));
    
    // Limit analysis size for performance
    size_t analysis_size = (data_length > analyzer->config.max_analysis_size) ? 
                          analyzer->config.max_analysis_size : data_length;
    
    characteristics->total_bytes = data_length;
    
    // Byte frequency analysis
    uint32_t byte_frequencies[256] = {0};
    size_t printable_count = 0;
    size_t whitespace_count = 0;
    size_t control_count = 0;
    size_t extended_count = 0;
    size_t line_count = 1;
    size_t word_count = 0;
    bool in_word = false;
    
    for (size_t i = 0; i < analysis_size; i++) {
        uint8_t c = data[i];
        byte_frequencies[c]++;
        
        if (is_printable_text_char(c)) {
            printable_count++;
            if (isalpha(c)) {
                if (!in_word) {
                    word_count++;
                    in_word = true;
                }
            } else {
                in_word = false;
            }
        } else if (isspace(c)) {
            whitespace_count++;
            in_word = false;
            if (c == '\n') line_count++;
        } else if (c < 32) {
            control_count++;
            in_word = false;
        } else {
            extended_count++;
            in_word = false;
        }
    }
    
    // Copy byte frequencies
    memcpy(characteristics->byte_frequencies, byte_frequencies, sizeof(byte_frequencies));
    
    // Count unique bytes
    size_t unique_bytes = 0;
    for (int i = 0; i < 256; i++) {
        if (byte_frequencies[i] > 0) {
            unique_bytes++;
        }
    }
    
    characteristics->unique_bytes = unique_bytes;
    characteristics->entropy = calculate_entropy(byte_frequencies, analysis_size);
    characteristics->printable_chars = printable_count;
    characteristics->whitespace_chars = whitespace_count;
    characteristics->control_chars = control_count;
    characteristics->extended_chars = extended_count;
    characteristics->line_count = line_count;
    characteristics->word_count = word_count;
    
    if (line_count > 0) {
        characteristics->average_line_length = (float)analysis_size / line_count;
    }
    
    // Text probability calculation
    if (analysis_size > 0) {
        characteristics->text_probability = (float)printable_count / analysis_size;
    }
    
    // Content type detection
    characteristics->has_json_markers = has_json_structure(data, analysis_size);
    characteristics->has_xml_markers = has_xml_structure(data, analysis_size);
    characteristics->has_csv_markers = has_csv_structure(data, analysis_size);
    
    // Determine primary content type
    if (characteristics->text_probability > 0.8f) {
        if (characteristics->has_json_markers) {
            characteristics->primary_type = CONTENT_TYPE_JSON;
        } else if (characteristics->has_xml_markers) {
            characteristics->primary_type = CONTENT_TYPE_XML;
        } else if (characteristics->has_csv_markers) {
            characteristics->primary_type = CONTENT_TYPE_CSV;
        } else if (word_count > 0 && (float)word_count / line_count > 3.0f) {
            characteristics->primary_type = CONTENT_TYPE_NATURAL_LANGUAGE;
        } else {
            characteristics->primary_type = CONTENT_TYPE_SOURCE_CODE;
        }
        characteristics->type_confidence = 0.8f;
    } else if (characteristics->entropy > 7.5f) {
        characteristics->primary_type = CONTENT_TYPE_COMPRESSED;
        characteristics->type_confidence = 0.7f;
    } else if (characteristics->entropy > 7.0f) {
        characteristics->primary_type = CONTENT_TYPE_RANDOM;
        characteristics->type_confidence = 0.6f;
    } else {
        characteristics->primary_type = CONTENT_TYPE_BINARY;
        characteristics->type_confidence = 0.6f;
    }
    
    // Language detection (simplified)
    if (analyzer->config.enable_language_detection && analyzer->models_initialized) {
        float best_score = 0.0f;
        LanguageType best_language = LANGUAGE_UNKNOWN;
        
        for (int lang = 1; lang < 16; lang++) {
            float score = 0.0f;
            for (int i = 0; i < 256; i++) {
                if (byte_frequencies[i] > 0) {
                    float freq = (float)byte_frequencies[i] / analysis_size;
                    score += freq * analyzer->language_models[lang][i];
                }
            }
            
            if (score > best_score) {
                best_score = score;
                best_language = lang;
            }
        }
        
        characteristics->detected_language = best_language;
        characteristics->language_confidence = fminf(0.9f, best_score / 10.0f);
    }
    
    // Pattern analysis (simplified)
    if (analyzer->config.enable_deep_analysis) {
        size_t pattern_count = 0;
        size_t repeated_sequences = 0;
        
        // Simple 2-byte pattern detection
        for (size_t i = 0; i < analysis_size - 2; i++) {
            for (size_t j = i + 2; j < analysis_size - 1; j++) {
                if (data[i] == data[j] && data[i+1] == data[j+1]) {
                    repeated_sequences++;
                    pattern_count++;
                    break;
                }
            }
            if (pattern_count >= 100) break; // Limit for performance
        }
        
        characteristics->repeated_sequences = repeated_sequences;
        characteristics->unique_patterns = pattern_count;
        if (analysis_size > 0) {
            characteristics->pattern_regularity = (float)repeated_sequences / analysis_size;
            characteristics->repetition_factor = (float)pattern_count / analysis_size;
        }
    }
    
    // Algorithm suitability assessment
    characteristics->transformer_suitability = content_analyzer_transformer_suitability(characteristics);
    characteristics->lstm_suitability = content_analyzer_lstm_suitability(characteristics);
    characteristics->hybrid_suitability = content_analyzer_hybrid_suitability(characteristics);
    
    // Compression estimate
    if (characteristics->entropy > 0) {
        characteristics->compression_ratio_estimate = characteristics->entropy / 8.0f;
    } else {
        characteristics->compression_ratio_estimate = 0.1f;
    }
    
    // Analysis confidence
    characteristics->analysis_confidence = fminf(0.95f, 
        (characteristics->type_confidence + characteristics->language_confidence) / 2.0f);
    
    // Update statistics
    analyzer->total_analyses++;
    analyzer->successful_analyses++;
    
    // Update content type counters
    switch (characteristics->primary_type) {
        case CONTENT_TYPE_NATURAL_LANGUAGE:
        case CONTENT_TYPE_SOURCE_CODE:
        case CONTENT_TYPE_DOCUMENTATION:
            analyzer->text_detections++;
            break;
        case CONTENT_TYPE_JSON:
        case CONTENT_TYPE_XML:
        case CONTENT_TYPE_CSV:
            analyzer->structured_detections++;
            break;
        case CONTENT_TYPE_COMPRESSED:
            analyzer->compressed_detections++;
            break;
        case CONTENT_TYPE_BINARY:
            analyzer->binary_detections++;
            break;
        default:
            break;
    }
    
    return CONTENT_ANALYZER_SUCCESS;
}

// Content type utilities
bool content_analyzer_is_text_data(const ContentCharacteristics *characteristics) {
    return characteristics && (
        characteristics->primary_type == CONTENT_TYPE_NATURAL_LANGUAGE ||
        characteristics->primary_type == CONTENT_TYPE_SOURCE_CODE ||
        characteristics->primary_type == CONTENT_TYPE_DOCUMENTATION ||
        characteristics->text_probability > 0.7f
    );
}

bool content_analyzer_is_structured_data(const ContentCharacteristics *characteristics) {
    return characteristics && (
        characteristics->primary_type == CONTENT_TYPE_JSON ||
        characteristics->primary_type == CONTENT_TYPE_XML ||
        characteristics->primary_type == CONTENT_TYPE_CSV ||
        characteristics->has_json_markers ||
        characteristics->has_xml_markers ||
        characteristics->has_csv_markers
    );
}

bool content_analyzer_is_binary_data(const ContentCharacteristics *characteristics) {
    return characteristics && (
        characteristics->primary_type == CONTENT_TYPE_BINARY ||
        characteristics->text_probability < 0.3f
    );
}

bool content_analyzer_is_compressed_data(const ContentCharacteristics *characteristics) {
    return characteristics && (
        characteristics->primary_type == CONTENT_TYPE_COMPRESSED ||
        characteristics->entropy > 7.5f
    );
}

bool content_analyzer_is_random_data(const ContentCharacteristics *characteristics) {
    return characteristics && (
        characteristics->primary_type == CONTENT_TYPE_RANDOM ||
        characteristics->entropy > 7.8f
    );
}

// Algorithm suitability assessment
float content_analyzer_transformer_suitability(const ContentCharacteristics *characteristics) {
    if (!characteristics) return 0.0f;
    
    float suitability = 0.5f; // Base score
    
    // Structured data favors Transformer
    if (content_analyzer_is_structured_data(characteristics)) {
        suitability += 0.3f;
    }
    
    // Medium entropy is good for Transformer
    if (characteristics->entropy >= 4.0f && characteristics->entropy <= 6.5f) {
        suitability += 0.2f;
    }
    
    // Pattern regularity helps
    if (characteristics->pattern_regularity > 0.1f) {
        suitability += 0.1f;
    }
    
    // High entropy hurts Transformer
    if (characteristics->entropy > 7.5f) {
        suitability -= 0.3f;
    }
    
    return fmaxf(0.0f, fminf(1.0f, suitability));
}

float content_analyzer_lstm_suitability(const ContentCharacteristics *characteristics) {
    if (!characteristics) return 0.0f;
    
    float suitability = 0.5f; // Base score
    
    // Text data favors LSTM
    if (content_analyzer_is_text_data(characteristics)) {
        suitability += 0.3f;
    }
    
    // Sequential patterns help LSTM
    if (characteristics->repetition_factor > 0.05f) {
        suitability += 0.2f;
    }
    
    // Lower entropy is good for LSTM
    if (characteristics->entropy >= 3.0f && characteristics->entropy <= 7.0f) {
        suitability += 0.2f;
    }
    
    // Very high entropy hurts LSTM
    if (characteristics->entropy > 7.8f) {
        suitability -= 0.4f;
    }
    
    return fmaxf(0.0f, fminf(1.0f, suitability));
}

float content_analyzer_hybrid_suitability(const ContentCharacteristics *characteristics) {
    if (!characteristics) return 0.0f;
    
    float transformer_score = content_analyzer_transformer_suitability(characteristics);
    float lstm_score = content_analyzer_lstm_suitability(characteristics);
    
    // Hybrid is suitable when both algorithms have moderate scores
    float score_diff = fabsf(transformer_score - lstm_score);
    float average_score = (transformer_score + lstm_score) / 2.0f;
    
    // Favor hybrid when scores are close and both are reasonable
    if (score_diff < 0.3f && average_score > 0.4f) {
        return fminf(1.0f, average_score + 0.2f);
    }
    
    return average_score * 0.8f;
}

// Statistics
ContentAnalyzerStats content_analyzer_get_stats(const ContentAnalyzer *analyzer) {
    ContentAnalyzerStats stats = {0};
    
    if (analyzer) {
        stats.total_analyses = analyzer->total_analyses;
        stats.successful_analyses = analyzer->successful_analyses;
        stats.text_detections = analyzer->text_detections;
        stats.binary_detections = analyzer->binary_detections;
        stats.structured_detections = analyzer->structured_detections;
        stats.compressed_detections = analyzer->compressed_detections;
        stats.average_analysis_time = analyzer->average_analysis_time;
        stats.patterns_detected = analyzer->cache_size;
        
        if (analyzer->total_analyses > 0) {
            stats.average_confidence = (float)analyzer->successful_analyses / analyzer->total_analyses;
        }
    }
    
    return stats;
}

void content_analyzer_reset_stats(ContentAnalyzer *analyzer) {
    if (!analyzer) return;
    
    analyzer->total_analyses = 0;
    analyzer->successful_analyses = 0;
    analyzer->text_detections = 0;
    analyzer->binary_detections = 0;
    analyzer->structured_detections = 0;
    analyzer->compressed_detections = 0;
    analyzer->average_analysis_time = 0.0f;
}

// Cache management
void content_analyzer_clear_pattern_cache(ContentAnalyzer *analyzer) {
    if (!analyzer) return;
    
    for (size_t i = 0; i < analyzer->cache_size; i++) {
        free(analyzer->pattern_cache[i].pattern_bytes);
        analyzer->pattern_cache[i].pattern_bytes = NULL;
    }
    analyzer->cache_size = 0;
}

size_t content_analyzer_get_cache_size(const ContentAnalyzer *analyzer) {
    return analyzer ? analyzer->cache_size : 0;
}
