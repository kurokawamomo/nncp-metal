#include "prediction_scorer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Default configuration values
#define DEFAULT_MAX_CANDIDATES 16
#define DEFAULT_MAX_CONTEXT_SIZE 1024
#define DEFAULT_PATTERN_BUFFER_SIZE 4096
#define DEFAULT_ENTROPY_WEIGHT 0.3f
#define DEFAULT_FREQUENCY_WEIGHT 0.25f
#define DEFAULT_RECENCY_WEIGHT 0.15f
#define DEFAULT_PATTERN_WEIGHT 0.3f

// Frequency table size (256 for all possible bytes)
#define FREQUENCY_TABLE_SIZE 256

// Recency tracking
#define RECENCY_WINDOW_SIZE 256

// Utility macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Internal helper functions
static void update_frequency_table(AdvancedPredictionScorer *scorer, const uint8_t *data, size_t length);
static void update_recency_scores(AdvancedPredictionScorer *scorer, const uint8_t *data, size_t length);
static float calculate_multi_factor_score(const AdvancedPredictionScorer *scorer, uint8_t candidate_byte);
static int compare_candidates(const void *a, const void *b);

// Configuration functions
PredictionScorerConfig prediction_scorer_config_default(void) {
    PredictionScorerConfig config = {
        .max_candidates = DEFAULT_MAX_CANDIDATES,
        .max_context_size = DEFAULT_MAX_CONTEXT_SIZE,
        .pattern_buffer_size = DEFAULT_PATTERN_BUFFER_SIZE,
        .entropy_weight = DEFAULT_ENTROPY_WEIGHT,
        .frequency_weight = DEFAULT_FREQUENCY_WEIGHT,
        .recency_weight = DEFAULT_RECENCY_WEIGHT,
        .pattern_weight = DEFAULT_PATTERN_WEIGHT
    };
    return config;
}

bool prediction_scorer_config_validate(const PredictionScorerConfig *config) {
    if (!config) return false;
    
    if (config->max_candidates == 0 || config->max_candidates > 256) return false;
    if (config->max_context_size == 0 || config->max_context_size > 65536) return false;
    if (config->pattern_buffer_size == 0 || config->pattern_buffer_size > 65536) return false;
    
    // Validate weights are reasonable
    float total_weight = config->entropy_weight + config->frequency_weight + 
                        config->recency_weight + config->pattern_weight;
    if (total_weight <= 0.0f || total_weight > 2.0f) return false;
    
    return true;
}

// Core API functions
AdvancedPredictionScorer* prediction_scorer_create(const PredictionScorerConfig *config) {
    if (!config || !prediction_scorer_config_validate(config)) {
        return NULL;
    }
    
    AdvancedPredictionScorer *scorer = calloc(1, sizeof(AdvancedPredictionScorer));
    if (!scorer) return NULL;
    
    // Set configuration
    scorer->entropy_weight = config->entropy_weight;
    scorer->frequency_weight = config->frequency_weight;
    scorer->recency_weight = config->recency_weight;
    scorer->pattern_weight = config->pattern_weight;
    scorer->max_candidates = config->max_candidates;
    scorer->max_context_size = config->max_context_size;
    
    // Allocate buffers
    scorer->pattern_buffer = malloc(config->pattern_buffer_size);
    scorer->frequency_table = calloc(FREQUENCY_TABLE_SIZE, sizeof(uint32_t));
    scorer->recency_scores = calloc(FREQUENCY_TABLE_SIZE, sizeof(uint32_t));
    scorer->candidates = malloc(config->max_candidates * sizeof(PredictionCandidate));
    scorer->context_buffer = malloc(config->max_context_size);
    
    if (!scorer->pattern_buffer || !scorer->frequency_table || !scorer->recency_scores ||
        !scorer->candidates || !scorer->context_buffer) {
        prediction_scorer_destroy(scorer);
        return NULL;
    }
    
    scorer->pattern_buffer_size = config->pattern_buffer_size;
    scorer->num_candidates = 0;
    scorer->context_size = 0;
    
    return scorer;
}

void prediction_scorer_destroy(AdvancedPredictionScorer *scorer) {
    if (!scorer) return;
    
    free(scorer->pattern_buffer);
    free(scorer->frequency_table);
    free(scorer->recency_scores);
    free(scorer->candidates);
    free(scorer->context_buffer);
    free(scorer);
}

// Context analysis functions
PredictionScorerResult prediction_scorer_analyze_context(
    AdvancedPredictionScorer *scorer,
    const uint8_t *context,
    size_t context_length
) {
    if (!scorer || !context || context_length == 0) {
        return PREDICTION_SCORER_ERROR_INVALID_PARAM;
    }
    
    // Update context buffer
    size_t copy_length = MIN(context_length, scorer->max_context_size);
    memcpy(scorer->context_buffer, context, copy_length);
    scorer->context_size = copy_length;
    
    // Update frequency and recency tracking
    update_frequency_table(scorer, context, context_length);
    update_recency_scores(scorer, context, context_length);
    
    // Update pattern buffer (circular buffer)
    size_t pattern_copy_length = MIN(context_length, scorer->pattern_buffer_size);
    if (context_length <= scorer->pattern_buffer_size) {
        memcpy(scorer->pattern_buffer, context, pattern_copy_length);
    } else {
        // Copy the most recent data
        memcpy(scorer->pattern_buffer, 
               context + context_length - scorer->pattern_buffer_size, 
               scorer->pattern_buffer_size);
    }
    
    return PREDICTION_SCORER_SUCCESS;
}

// Prediction generation functions
PredictionScorerResult prediction_scorer_generate_candidates(
    AdvancedPredictionScorer *scorer,
    const uint8_t *input_data,
    size_t input_size,
    size_t position
) {
    if (!scorer || !input_data || position >= input_size) {
        return PREDICTION_SCORER_ERROR_INVALID_PARAM;
    }
    
    scorer->num_candidates = 0;
    
    // Generate candidates based on frequency analysis
    // Sort frequency table to get most frequent bytes
    typedef struct {
        uint8_t byte_value;
        uint32_t frequency;
    } ByteFrequency;
    
    ByteFrequency freq_sorted[FREQUENCY_TABLE_SIZE];
    for (int i = 0; i < FREQUENCY_TABLE_SIZE; i++) {
        freq_sorted[i].byte_value = (uint8_t)i;
        freq_sorted[i].frequency = scorer->frequency_table[i];
    }
    
    // Simple selection sort for top candidates
    for (size_t i = 0; i < scorer->max_candidates && i < FREQUENCY_TABLE_SIZE; i++) {
        size_t max_idx = i;
        for (size_t j = i + 1; j < FREQUENCY_TABLE_SIZE; j++) {
            if (freq_sorted[j].frequency > freq_sorted[max_idx].frequency) {
                max_idx = j;
            }
        }
        
        // Swap
        ByteFrequency temp = freq_sorted[i];
        freq_sorted[i] = freq_sorted[max_idx];
        freq_sorted[max_idx] = temp;
        
        // Create candidate if frequency > 0
        if (freq_sorted[i].frequency > 0) {
            PredictionCandidate *candidate = &scorer->candidates[scorer->num_candidates];
            candidate->predicted_byte = freq_sorted[i].byte_value;
            candidate->frequency_rank = (uint32_t)i + 1;
            
            // Calculate multi-factor score
            float score = calculate_multi_factor_score(scorer, candidate->predicted_byte);
            candidate->confidence_score = score;
            
            // Calculate individual scores for debugging
            candidate->entropy_score = calculate_entropy_score(
                candidate->predicted_byte, scorer->context_buffer, scorer->context_size);
            candidate->pattern_match_length = 0; // TODO: Implement pattern matching
            candidate->recency_score = scorer->recency_scores[candidate->predicted_byte];
            
            scorer->num_candidates++;
        }
    }
    
    // Sort candidates by confidence score
    qsort(scorer->candidates, scorer->num_candidates, sizeof(PredictionCandidate), compare_candidates);
    
    return scorer->num_candidates > 0 ? PREDICTION_SCORER_SUCCESS : PREDICTION_SCORER_ERROR_NO_CANDIDATES;
}

PredictionScorerResult prediction_scorer_select_best(
    AdvancedPredictionScorer *scorer,
    PredictionCandidate *best_candidate
) {
    if (!scorer || !best_candidate) {
        return PREDICTION_SCORER_ERROR_INVALID_PARAM;
    }
    
    if (scorer->num_candidates == 0) {
        return PREDICTION_SCORER_ERROR_NO_CANDIDATES;
    }
    
    // Return the highest-scoring candidate (first after sorting)
    *best_candidate = scorer->candidates[0];
    
    return PREDICTION_SCORER_SUCCESS;
}

// Utility functions
PredictionScorerResult prediction_scorer_update_feedback(
    AdvancedPredictionScorer *scorer,
    uint8_t actual_byte,
    bool prediction_correct
) {
    if (!scorer) {
        return PREDICTION_SCORER_ERROR_INVALID_PARAM;
    }
    
    scorer->total_predictions++;
    if (prediction_correct) {
        scorer->successful_predictions++;
    }
    
    // Update frequency table with actual byte
    scorer->frequency_table[actual_byte]++;
    
    // Update average confidence (simple moving average)
    if (scorer->num_candidates > 0) {
        float current_confidence = scorer->candidates[0].confidence_score;
        scorer->average_confidence = (scorer->average_confidence * (scorer->total_predictions - 1) + 
                                    current_confidence) / scorer->total_predictions;
    }
    
    return PREDICTION_SCORER_SUCCESS;
}

float prediction_scorer_get_confidence(const AdvancedPredictionScorer *scorer) {
    if (!scorer || scorer->num_candidates == 0) return 0.0f;
    return scorer->candidates[0].confidence_score;
}

size_t prediction_scorer_get_candidate_count(const AdvancedPredictionScorer *scorer) {
    if (!scorer) return 0;
    return scorer->num_candidates;
}

// Statistics functions
PredictionScorerStats prediction_scorer_get_stats(const AdvancedPredictionScorer *scorer) {
    PredictionScorerStats stats = {0};
    
    if (scorer) {
        stats.total_predictions = scorer->total_predictions;
        stats.successful_predictions = scorer->successful_predictions;
        stats.success_rate = scorer->total_predictions > 0 ? 
            (float)scorer->successful_predictions / scorer->total_predictions : 0.0f;
        stats.average_confidence = scorer->average_confidence;
        stats.current_candidates = scorer->num_candidates;
        stats.context_size = scorer->context_size;
    }
    
    return stats;
}

void prediction_scorer_reset_stats(AdvancedPredictionScorer *scorer) {
    if (!scorer) return;
    
    scorer->total_predictions = 0;
    scorer->successful_predictions = 0;
    scorer->average_confidence = 0.0f;
}

// Internal scoring functions
float calculate_entropy_score(uint8_t candidate_byte, const uint8_t *context, size_t context_length) {
    if (!context || context_length == 0) return 0.0f;
    
    // Count occurrences of candidate byte in context
    size_t count = 0;
    for (size_t i = 0; i < context_length; i++) {
        if (context[i] == candidate_byte) {
            count++;
        }
    }
    
    if (count == 0) return 0.0f;
    
    // Calculate local probability
    float probability = (float)count / context_length;
    
    // Calculate entropy contribution (negative log probability)
    return -probability * log2f(probability);
}

float calculate_frequency_score(uint8_t candidate_byte, const uint32_t *frequency_table) {
    if (!frequency_table) return 0.0f;
    
    // Normalize frequency to [0, 1]
    uint32_t max_frequency = 0;
    for (int i = 0; i < FREQUENCY_TABLE_SIZE; i++) {
        if (frequency_table[i] > max_frequency) {
            max_frequency = frequency_table[i];
        }
    }
    
    if (max_frequency == 0) return 0.0f;
    
    return (float)frequency_table[candidate_byte] / max_frequency;
}

float calculate_pattern_score(uint8_t candidate_byte, const uint8_t *pattern_buffer, size_t buffer_size) {
    if (!pattern_buffer || buffer_size == 0) return 0.0f;
    
    // Simple pattern matching - look for candidate byte in recent patterns
    size_t matches = 0;
    for (size_t i = 0; i < buffer_size; i++) {
        if (pattern_buffer[i] == candidate_byte) {
            matches++;
        }
    }
    
    return (float)matches / buffer_size;
}

float calculate_recency_score(uint8_t candidate_byte, const uint32_t *recency_scores) {
    if (!recency_scores) return 0.0f;
    
    // Higher recency scores indicate more recent occurrence
    return (float)recency_scores[candidate_byte] / RECENCY_WINDOW_SIZE;
}

// Internal helper functions
static void update_frequency_table(AdvancedPredictionScorer *scorer, const uint8_t *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        scorer->frequency_table[data[i]]++;
    }
}

static void update_recency_scores(AdvancedPredictionScorer *scorer, const uint8_t *data, size_t length) {
    // Decay existing recency scores
    for (int i = 0; i < FREQUENCY_TABLE_SIZE; i++) {
        if (scorer->recency_scores[i] > 0) {
            scorer->recency_scores[i]--;
        }
    }
    
    // Update recency for new data (more recent = higher score)
    for (size_t i = 0; i < length; i++) {
        scorer->recency_scores[data[i]] = RECENCY_WINDOW_SIZE;
    }
}

static float calculate_multi_factor_score(const AdvancedPredictionScorer *scorer, uint8_t candidate_byte) {
    float entropy_score = calculate_entropy_score(candidate_byte, scorer->context_buffer, scorer->context_size);
    float frequency_score = calculate_frequency_score(candidate_byte, scorer->frequency_table);
    float pattern_score = calculate_pattern_score(candidate_byte, scorer->pattern_buffer, scorer->pattern_buffer_size);
    float recency_score = calculate_recency_score(candidate_byte, scorer->recency_scores);
    
    return (entropy_score * scorer->entropy_weight +
            frequency_score * scorer->frequency_weight +
            pattern_score * scorer->pattern_weight +
            recency_score * scorer->recency_weight);
}

static int compare_candidates(const void *a, const void *b) {
    const PredictionCandidate *candidate_a = (const PredictionCandidate *)a;
    const PredictionCandidate *candidate_b = (const PredictionCandidate *)b;
    
    // Sort by confidence score (descending)
    if (candidate_a->confidence_score > candidate_b->confidence_score) return -1;
    if (candidate_a->confidence_score < candidate_b->confidence_score) return 1;
    return 0;
}
