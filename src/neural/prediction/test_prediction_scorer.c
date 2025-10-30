#include "prediction_scorer.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

// Simple test framework
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        printf("Running test: %s... ", #name); \
        if (test_##name()) { \
            tests_passed++; \
            printf("PASSED\n"); \
        } else { \
            printf("FAILED\n"); \
        } \
    } while(0)

// Test functions
static bool test_config_creation() {
    PredictionScorerConfig config = prediction_scorer_config_default();
    return prediction_scorer_config_validate(&config);
}

static bool test_scorer_creation() {
    PredictionScorerConfig config = prediction_scorer_config_default();
    AdvancedPredictionScorer *scorer = prediction_scorer_create(&config);
    
    bool success = (scorer != NULL);
    
    prediction_scorer_destroy(scorer);
    return success;
}

static bool test_context_analysis() {
    PredictionScorerConfig config = prediction_scorer_config_default();
    AdvancedPredictionScorer *scorer = prediction_scorer_create(&config);
    
    if (!scorer) return false;
    
    const char *test_data = "Hello, World! This is a test string for context analysis.";
    PredictionScorerResult result = prediction_scorer_analyze_context(
        scorer, (const uint8_t*)test_data, strlen(test_data));
    
    bool success = (result == PREDICTION_SCORER_SUCCESS);
    
    prediction_scorer_destroy(scorer);
    return success;
}

static bool test_candidate_generation() {
    PredictionScorerConfig config = prediction_scorer_config_default();
    AdvancedPredictionScorer *scorer = prediction_scorer_create(&config);
    
    if (!scorer) return false;
    
    const char *test_data = "Hello, World! This is a test string for prediction testing.";
    
    // Analyze context first
    prediction_scorer_analyze_context(scorer, (const uint8_t*)test_data, strlen(test_data));
    
    // Generate candidates
    PredictionScorerResult result = prediction_scorer_generate_candidates(
        scorer, (const uint8_t*)test_data, strlen(test_data), 10);
    
    bool success = (result == PREDICTION_SCORER_SUCCESS) && 
                   (prediction_scorer_get_candidate_count(scorer) > 0);
    
    prediction_scorer_destroy(scorer);
    return success;
}

static bool test_best_selection() {
    PredictionScorerConfig config = prediction_scorer_config_default();
    AdvancedPredictionScorer *scorer = prediction_scorer_create(&config);
    
    if (!scorer) return false;
    
    const char *test_data = "Hello, World! This is a test string for prediction testing.";
    
    // Analyze context and generate candidates
    prediction_scorer_analyze_context(scorer, (const uint8_t*)test_data, strlen(test_data));
    prediction_scorer_generate_candidates(scorer, (const uint8_t*)test_data, strlen(test_data), 10);
    
    // Select best candidate
    PredictionCandidate best;
    PredictionScorerResult result = prediction_scorer_select_best(scorer, &best);
    
    bool success = (result == PREDICTION_SCORER_SUCCESS) && 
                   (best.confidence_score >= 0.0f);
    
    prediction_scorer_destroy(scorer);
    return success;
}

static bool test_feedback_update() {
    PredictionScorerConfig config = prediction_scorer_config_default();
    AdvancedPredictionScorer *scorer = prediction_scorer_create(&config);
    
    if (!scorer) return false;
    
    // Test feedback update
    PredictionScorerResult result = prediction_scorer_update_feedback(scorer, 'A', true);
    
    bool success = (result == PREDICTION_SCORER_SUCCESS);
    
    prediction_scorer_destroy(scorer);
    return success;
}

static bool test_statistics() {
    PredictionScorerConfig config = prediction_scorer_config_default();
    AdvancedPredictionScorer *scorer = prediction_scorer_create(&config);
    
    if (!scorer) return false;
    
    // Update some feedback
    prediction_scorer_update_feedback(scorer, 'A', true);
    prediction_scorer_update_feedback(scorer, 'B', false);
    
    PredictionScorerStats stats = prediction_scorer_get_stats(scorer);
    
    bool success = (stats.total_predictions == 2) && 
                   (stats.successful_predictions == 1) && 
                   (stats.success_rate == 0.5f);
    
    prediction_scorer_destroy(scorer);
    return success;
}

static bool test_entropy_calculation() {
    const uint8_t context[] = "aaabbbccc";
    float entropy_a = calculate_entropy_score('a', context, sizeof(context) - 1);
    float entropy_z = calculate_entropy_score('z', context, sizeof(context) - 1);
    
    // 'a' should have some entropy score, 'z' should have 0
    return (entropy_a > 0.0f) && (entropy_z == 0.0f);
}

int main() {
    printf("=== Prediction Scorer Unit Tests ===\n\n");
    
    TEST(config_creation);
    TEST(scorer_creation);
    TEST(context_analysis);
    TEST(candidate_generation);
    TEST(best_selection);
    TEST(feedback_update);
    TEST(statistics);
    TEST(entropy_calculation);
    
    printf("\n=== Test Results ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    printf("Success rate: %.1f%%\n", (float)tests_passed / tests_run * 100.0f);
    
    return (tests_passed == tests_run) ? 0 : 1;
}
