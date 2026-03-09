#ifndef PREDICTION_SCORER_H
#define PREDICTION_SCORER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct AdvancedPredictionScorer AdvancedPredictionScorer;
typedef struct PredictionCandidate PredictionCandidate;

// Prediction candidate structure
struct PredictionCandidate {
    uint8_t predicted_byte;
    float confidence_score;
    float entropy_score;
    uint32_t pattern_match_length;
    uint32_t frequency_rank;
    uint32_t recency_score;
};

// Advanced prediction scorer structure
struct AdvancedPredictionScorer {
    // Entropy-based scoring weights
    float entropy_weight;
    float frequency_weight;
    float recency_weight;
    float pattern_weight;
    
    // Pattern recognition buffers
    uint8_t *pattern_buffer;
    size_t pattern_buffer_size;
    uint32_t *frequency_table;
    uint32_t *recency_scores;
    
    // Prediction candidates
    PredictionCandidate *candidates;
    size_t num_candidates;
    size_t max_candidates;
    
    // Context analysis
    uint8_t *context_buffer;
    size_t context_size;
    size_t max_context_size;
    
    // Performance metrics
    uint64_t total_predictions;
    uint64_t successful_predictions;
    float average_confidence;
};

// Configuration structure
typedef struct {
    size_t max_candidates;
    size_t max_context_size;
    size_t pattern_buffer_size;
    float entropy_weight;
    float frequency_weight;
    float recency_weight;
    float pattern_weight;
} PredictionScorerConfig;

// Return codes
typedef enum {
    PREDICTION_SCORER_SUCCESS = 0,
    PREDICTION_SCORER_ERROR_INVALID_PARAM = -1,
    PREDICTION_SCORER_ERROR_MEMORY_ALLOCATION = -2,
    PREDICTION_SCORER_ERROR_BUFFER_FULL = -3,
    PREDICTION_SCORER_ERROR_NO_CANDIDATES = -4,
    PREDICTION_SCORER_ERROR_INVALID_CONTEXT = -5
} PredictionScorerResult;

// Core API functions
AdvancedPredictionScorer* prediction_scorer_create(const PredictionScorerConfig *config);
void prediction_scorer_destroy(AdvancedPredictionScorer *scorer);

// Context analysis functions
PredictionScorerResult prediction_scorer_analyze_context(
    AdvancedPredictionScorer *scorer,
    const uint8_t *context,
    size_t context_length
);

// Prediction generation functions
PredictionScorerResult prediction_scorer_generate_candidates(
    AdvancedPredictionScorer *scorer,
    const uint8_t *input_data,
    size_t input_size,
    size_t position
);

// Prediction selection functions
PredictionScorerResult prediction_scorer_select_best(
    AdvancedPredictionScorer *scorer,
    PredictionCandidate *best_candidate
);

// Utility functions
PredictionScorerResult prediction_scorer_update_feedback(
    AdvancedPredictionScorer *scorer,
    uint8_t actual_byte,
    bool prediction_correct
);

float prediction_scorer_get_confidence(const AdvancedPredictionScorer *scorer);
size_t prediction_scorer_get_candidate_count(const AdvancedPredictionScorer *scorer);

// Configuration helpers
PredictionScorerConfig prediction_scorer_config_default(void);
bool prediction_scorer_config_validate(const PredictionScorerConfig *config);

// Statistics and debugging
typedef struct {
    uint64_t total_predictions;
    uint64_t successful_predictions;
    float success_rate;
    float average_confidence;
    size_t current_candidates;
    size_t context_size;
} PredictionScorerStats;

PredictionScorerStats prediction_scorer_get_stats(const AdvancedPredictionScorer *scorer);
void prediction_scorer_reset_stats(AdvancedPredictionScorer *scorer);

// Internal scoring functions (exposed for testing)
float calculate_entropy_score(uint8_t candidate_byte, const uint8_t *context, size_t context_length);
float calculate_frequency_score(uint8_t candidate_byte, const uint32_t *frequency_table);
float calculate_pattern_score(uint8_t candidate_byte, const uint8_t *pattern_buffer, size_t buffer_size);
float calculate_recency_score(uint8_t candidate_byte, const uint32_t *recency_scores);

#ifdef __cplusplus
}
#endif

#endif // PREDICTION_SCORER_H
