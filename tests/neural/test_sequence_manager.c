#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "../../src/neural/sequence/sequence_manager.h"
#include "../../src/neural/tokenizer/bpe_tokenizer.h"

// Test framework
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("FAIL: %s - %s\n", __func__, message); \
            return 0; \
        } \
    } while(0)

#define TEST_PASS() \
    do { \
        printf("PASS: %s\n", __func__); \
        return 1; \
    } while(0)

// Test data
static const char* test_vocab_path = "test_vocab.txt";

// Helper function to create test token sequence
TokenSequence* create_test_sequence(uint32_t length) {
    TokenSequence* sequence = (TokenSequence*)calloc(1, sizeof(TokenSequence));
    if (!sequence) return NULL;
    
    sequence->tokens = (uint32_t*)malloc(length * sizeof(uint32_t));
    sequence->positions = (uint32_t*)malloc(length * sizeof(uint32_t));
    if (!sequence->tokens || !sequence->positions) {
        token_sequence_destroy(sequence);
        return NULL;
    }
    
    // Fill with test data
    for (uint32_t i = 0; i < length; i++) {
        sequence->tokens[i] = i % 100; // Cycle through token IDs
        sequence->positions[i] = i;
    }
    
    sequence->token_count = length;
    sequence->is_valid = true;
    sequence->confidence = 0.95f;
    sequence->oov_count = 0;
    
    return sequence;
}

// Test functions
int test_window_config_creation() {
    WindowConfig config;
    SequenceError error = window_config_create_default(&config);
    
    TEST_ASSERT(error == SEQ_SUCCESS, "Default config creation should succeed");
    TEST_ASSERT(config.min_window_size > 0, "Min window size should be positive");
    TEST_ASSERT(config.max_window_size > config.min_window_size, "Max should be greater than min");
    TEST_ASSERT(config.preferred_window_size >= config.min_window_size, "Preferred should be >= min");
    TEST_ASSERT(config.preferred_window_size <= config.max_window_size, "Preferred should be <= max");
    TEST_ASSERT(config.overlap_size < config.preferred_window_size, "Overlap should be less than window size");
    TEST_ASSERT(config.max_memory_mb > 0, "Memory limit should be positive");
    
    printf("  Default config: window_size=%u, overlap=%u, memory=%uMB\n",
           config.preferred_window_size, config.overlap_size, config.max_memory_mb);
    
    TEST_PASS();
}

int test_window_config_validation() {
    WindowConfig config;
    window_config_create_default(&config);
    
    // Test valid config
    SequenceError error = window_config_validate(&config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Valid config should pass validation");
    
    // Test invalid configs
    WindowConfig invalid_config = config;
    invalid_config.min_window_size = 0;
    error = window_config_validate(&invalid_config);
    TEST_ASSERT(error == SEQ_ERROR_WINDOW_TOO_SMALL, "Zero min window should fail");
    
    invalid_config = config;
    invalid_config.max_window_size = config.min_window_size - 1;
    error = window_config_validate(&invalid_config);
    TEST_ASSERT(error == SEQ_ERROR_WINDOW_TOO_SMALL, "Max < min should fail");
    
    invalid_config = config;
    invalid_config.overlap_size = config.preferred_window_size;
    error = window_config_validate(&invalid_config);
    TEST_ASSERT(error == SEQ_ERROR_OVERLAP_INVALID, "Overlap >= window size should fail");
    
    invalid_config = config;
    invalid_config.overlap_ratio = 0.6f;
    error = window_config_validate(&invalid_config);
    TEST_ASSERT(error == SEQ_ERROR_OVERLAP_INVALID, "Overlap ratio >= 0.5 should fail");
    
    TEST_PASS();
}

int test_sequence_manager_creation() {
    WindowConfig config;
    window_config_create_default(&config);
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    TEST_ASSERT(manager != NULL, "Manager should not be NULL");
    TEST_ASSERT(manager->cache_size > 0, "Cache should be allocated");
    TEST_ASSERT(manager->current_position == 0, "Initial position should be 0");
    TEST_ASSERT(!manager->is_processing, "Should not be processing initially");
    
    printf("  Created manager with cache_size=%u, max_checkpoints=%u\n",
           manager->cache_size, manager->max_checkpoints);
    
    sequence_manager_destroy(manager);
    TEST_PASS();
}

int test_sequence_manager_set_source() {
    WindowConfig config;
    window_config_create_default(&config);
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    
    // Create test sequence
    TokenSequence* sequence = create_test_sequence(1000);
    TEST_ASSERT(sequence != NULL, "Test sequence creation should succeed");
    
    error = sequence_manager_set_source(manager, sequence);
    TEST_ASSERT(error == SEQ_SUCCESS, "Setting source should succeed");
    TEST_ASSERT(manager->source_sequence == sequence, "Source should be set correctly");
    TEST_ASSERT(manager->total_length == 1000, "Total length should be correct");
    TEST_ASSERT(manager->current_position == 0, "Position should be reset");
    
    printf("  Set source sequence with %u tokens\n", manager->total_length);
    
    token_sequence_destroy(sequence);
    sequence_manager_destroy(manager);
    TEST_PASS();
}

int test_window_processing() {
    WindowConfig config;
    window_config_create_default(&config);
    config.min_window_size = 50;
    config.max_window_size = 500;
    config.preferred_window_size = 100;
    config.overlap_size = 25;
    config.max_memory_mb = 32;
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    
    // Create test sequence
    TokenSequence* sequence = create_test_sequence(1000);
    TEST_ASSERT(sequence != NULL, "Test sequence creation should succeed");
    
    error = sequence_manager_set_source(manager, sequence);
    TEST_ASSERT(error == SEQ_SUCCESS, "Setting source should succeed");
    
    // Process windows
    uint32_t window_count = 0;
    WindowContext* window = NULL;
    
    while (sequence_manager_has_next_window(manager)) {
        error = sequence_manager_get_next_window(manager, &window);
        TEST_ASSERT(error == SEQ_SUCCESS, "Getting next window should succeed");
        TEST_ASSERT(window != NULL, "Window should not be NULL");
        TEST_ASSERT(window->tokens != NULL, "Window tokens should not be NULL");
        TEST_ASSERT(window->tokens->token_count > 0, "Window should have tokens");
        TEST_ASSERT(window->reference_count == 1, "Reference count should be 1");
        
        printf("  Window %u: start=%u, size=%u, tokens=%u\n",
               window_count, window->boundary.start_position,
               window->boundary.window_size, window->tokens->token_count);
        
        // Release window
        error = sequence_manager_release_window(manager, window);
        TEST_ASSERT(error == SEQ_SUCCESS, "Releasing window should succeed");
        
        window_count++;
        window = NULL;
        
        // Limit iterations to avoid infinite loop
        if (window_count > 10) break;
    }
    
    TEST_ASSERT(window_count > 0, "Should have processed at least one window");
    printf("  Processed %u windows total\n", window_count);
    
    token_sequence_destroy(sequence);
    sequence_manager_destroy(manager);
    TEST_PASS();
}

int test_window_overlap() {
    WindowConfig config;
    window_config_create_default(&config);
    config.min_window_size = 50;
    config.max_window_size = 500;
    config.preferred_window_size = 100;
    config.overlap_size = 25;
    config.overlap_ratio = 0.0f; // Use absolute overlap_size instead of ratio
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    
    TokenSequence* sequence = create_test_sequence(300);
    error = sequence_manager_set_source(manager, sequence);
    TEST_ASSERT(error == SEQ_SUCCESS, "Setting source should succeed");
    
    // Get first window
    WindowContext* window1 = NULL;
    error = sequence_manager_get_next_window(manager, &window1);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting first window should succeed");
    TEST_ASSERT(window1->boundary.start_position == 0, "First window should start at 0");
    
    // Get second window
    WindowContext* window2 = NULL;
    error = sequence_manager_get_next_window(manager, &window2);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting second window should succeed");
    
    // Check overlap
    uint32_t expected_start = 100 - 25; // window_size - overlap_size
    TEST_ASSERT(window2->boundary.start_position == expected_start, 
                "Second window should start at correct overlap position");
    
    printf("  Window 1: start=%u, size=%u\n", 
           window1->boundary.start_position, window1->boundary.window_size);
    printf("  Window 2: start=%u, size=%u\n", 
           window2->boundary.start_position, window2->boundary.window_size);
    printf("  Overlap: %u tokens\n", 
           window1->boundary.end_position - window2->boundary.start_position);
    
    sequence_manager_release_window(manager, window1);
    sequence_manager_release_window(manager, window2);
    token_sequence_destroy(sequence);
    sequence_manager_destroy(manager);
    TEST_PASS();
}

int test_window_caching() {
    WindowConfig config;
    window_config_create_default(&config);
    config.min_window_size = 50;
    config.max_window_size = 500;
    config.preferred_window_size = 100;
    config.overlap_size = 25;
    config.max_memory_mb = 32; // Small cache for testing
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    
    TokenSequence* sequence = create_test_sequence(500);
    error = sequence_manager_set_source(manager, sequence);
    TEST_ASSERT(error == SEQ_SUCCESS, "Setting source should succeed");
    
    // Get window at position 0 (should cache)
    WindowContext* window1 = NULL;
    error = sequence_manager_get_window_at(manager, 0, &window1);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting window at position should succeed");
    TEST_ASSERT(window1->is_cached, "Window should be cached");
    
    // Get same window again (should hit cache)
    WindowContext* window2 = NULL;
    error = sequence_manager_get_window_at(manager, 0, &window2);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting cached window should succeed");
    TEST_ASSERT(window1 == window2, "Should return same cached window");
    TEST_ASSERT(window2->reference_count == 2, "Reference count should be 2");
    
    // Check statistics
    SequenceStats stats;
    error = sequence_manager_get_stats(manager, &stats);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting stats should succeed");
    TEST_ASSERT(stats.cache_hits > 0, "Should have cache hits");
    
    printf("  Cache hits: %u, Cache misses: %u\n", 
           stats.cache_hits, stats.cache_misses);
    
    sequence_manager_release_window(manager, window1);
    sequence_manager_release_window(manager, window2);
    token_sequence_destroy(sequence);
    sequence_manager_destroy(manager);
    TEST_PASS();
}

int test_checkpoint_resume() {
    WindowConfig config;
    window_config_create_default(&config);
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    
    TokenSequence* sequence = create_test_sequence(1000);
    error = sequence_manager_set_source(manager, sequence);
    TEST_ASSERT(error == SEQ_SUCCESS, "Setting source should succeed");
    
    // Process some windows
    WindowContext* window = NULL;
    for (int i = 0; i < 3; i++) {
        error = sequence_manager_get_next_window(manager, &window);
        TEST_ASSERT(error == SEQ_SUCCESS, "Getting window should succeed");
        sequence_manager_release_window(manager, window);
    }
    
    uint32_t checkpoint_position = manager->current_position;
    
    // Create checkpoint
    uint32_t checkpoint_id;
    error = sequence_manager_create_checkpoint(manager, &checkpoint_id);
    TEST_ASSERT(error == SEQ_SUCCESS, "Creating checkpoint should succeed");
    
    // Process more windows
    for (int i = 0; i < 2; i++) {
        error = sequence_manager_get_next_window(manager, &window);
        TEST_ASSERT(error == SEQ_SUCCESS, "Getting window should succeed");
        sequence_manager_release_window(manager, window);
    }
    
    uint32_t advanced_position = manager->current_position;
    TEST_ASSERT(advanced_position > checkpoint_position, "Should have advanced past checkpoint");
    
    // Resume from checkpoint
    error = sequence_manager_resume_from_checkpoint(manager, checkpoint_id);
    TEST_ASSERT(error == SEQ_SUCCESS, "Resuming from checkpoint should succeed");
    TEST_ASSERT(manager->current_position == checkpoint_position, "Should be back at checkpoint position");
    
    printf("  Checkpoint created at position %u\n", checkpoint_position);
    printf("  Advanced to position %u\n", advanced_position);
    printf("  Resumed back to position %u\n", manager->current_position);
    
    token_sequence_destroy(sequence);
    sequence_manager_destroy(manager);
    TEST_PASS();
}

int test_memory_management() {
    WindowConfig config;
    window_config_create_default(&config);
    config.max_memory_mb = 16; // Small memory limit for testing
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    
    TokenSequence* sequence = create_test_sequence(1000);
    error = sequence_manager_set_source(manager, sequence);
    TEST_ASSERT(error == SEQ_SUCCESS, "Setting source should succeed");
    
    // Get memory usage
    uint32_t memory_usage_mb;
    error = sequence_manager_get_memory_usage(manager, &memory_usage_mb);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting memory usage should succeed");
    TEST_ASSERT(memory_usage_mb > 0, "Memory usage should be positive");
    
    printf("  Initial memory usage: %u MB\n", memory_usage_mb);
    
    // Process several windows to fill cache
    WindowContext* windows[5];
    for (int i = 0; i < 5; i++) {
        error = sequence_manager_get_next_window(manager, &windows[i]);
        if (error == SEQ_SUCCESS) {
            printf("  Created window %d at position %u\n", i, windows[i]->boundary.start_position);
        }
    }
    
    // Check memory usage again
    uint32_t new_memory_usage_mb;
    error = sequence_manager_get_memory_usage(manager, &new_memory_usage_mb);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting memory usage should succeed");
    TEST_ASSERT(new_memory_usage_mb >= memory_usage_mb, "Memory usage should have increased");
    
    printf("  Memory usage after windows: %u MB\n", new_memory_usage_mb);
    
    // Clear cache
    error = sequence_manager_clear_cache(manager);
    TEST_ASSERT(error == SEQ_SUCCESS, "Clearing cache should succeed");
    
    // Check memory usage after clearing
    uint32_t cleared_memory_usage_mb;
    error = sequence_manager_get_memory_usage(manager, &cleared_memory_usage_mb);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting memory usage should succeed");
    
    printf("  Memory usage after clear: %u MB\n", cleared_memory_usage_mb);
    
    token_sequence_destroy(sequence);
    sequence_manager_destroy(manager);
    TEST_PASS();
}

int test_progress_tracking() {
    WindowConfig config;
    window_config_create_default(&config);
    config.min_window_size = 50;
    config.max_window_size = 500;
    config.preferred_window_size = 100;
    config.overlap_size = 25;
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    
    TokenSequence* sequence = create_test_sequence(500);
    error = sequence_manager_set_source(manager, sequence);
    TEST_ASSERT(error == SEQ_SUCCESS, "Setting source should succeed");
    
    // Check initial progress
    float progress;
    error = sequence_manager_get_progress(manager, &progress);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting progress should succeed");
    TEST_ASSERT(progress == 0.0f, "Initial progress should be 0.0");
    
    // Process half the windows
    WindowContext* window = NULL;
    while (sequence_manager_has_next_window(manager) && manager->current_position < 250) {
        error = sequence_manager_get_next_window(manager, &window);
        TEST_ASSERT(error == SEQ_SUCCESS, "Getting window should succeed");
        sequence_manager_release_window(manager, window);
    }
    
    // Check progress
    error = sequence_manager_get_progress(manager, &progress);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting progress should succeed");
    TEST_ASSERT(progress > 0.4f && progress < 0.6f, "Progress should be around 50%");
    
    printf("  Progress at position %u: %.1f%%\n", manager->current_position, progress * 100);
    
    token_sequence_destroy(sequence);
    sequence_manager_destroy(manager);
    TEST_PASS();
}

int test_adaptive_window_sizing() {
    WindowConfig config;
    window_config_create_default(&config);
    config.adaptive_sizing = true;
    config.min_window_size = 50;
    config.max_window_size = 200;
    config.preferred_window_size = 100;
    config.overlap_size = 25;
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    
    TokenSequence* sequence = create_test_sequence(300); // Small sequence
    error = sequence_manager_set_source(manager, sequence);
    TEST_ASSERT(error == SEQ_SUCCESS, "Setting source should succeed");
    
    // Get window near end of sequence
    WindowContext* window = NULL;
    error = sequence_manager_get_window_at(manager, 250, &window);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting window should succeed");
    
    // Window size should be adapted for remaining tokens
    TEST_ASSERT(window->boundary.window_size <= 50, "Window size should be adapted for remaining tokens");
    TEST_ASSERT(window->boundary.window_size > 0, "Window size should be positive");
    
    printf("  Adaptive window at position 250: size=%u (remaining=%u)\n",
           window->boundary.window_size, 300 - 250);
    
    sequence_manager_release_window(manager, window);
    token_sequence_destroy(sequence);
    sequence_manager_destroy(manager);
    TEST_PASS();
}

int test_statistics_collection() {
    WindowConfig config;
    window_config_create_default(&config);
    
    SequenceManager* manager = NULL;
    SequenceError error = sequence_manager_create(&manager, &config);
    TEST_ASSERT(error == SEQ_SUCCESS, "Manager creation should succeed");
    
    TokenSequence* sequence = create_test_sequence(500);
    error = sequence_manager_set_source(manager, sequence);
    TEST_ASSERT(error == SEQ_SUCCESS, "Setting source should succeed");
    
    // Process several windows
    WindowContext* window = NULL;
    for (int i = 0; i < 3; i++) {
        error = sequence_manager_get_next_window(manager, &window);
        TEST_ASSERT(error == SEQ_SUCCESS, "Getting window should succeed");
        sequence_manager_release_window(manager, window);
    }
    
    // Get statistics
    SequenceStats stats;
    error = sequence_manager_get_stats(manager, &stats);
    TEST_ASSERT(error == SEQ_SUCCESS, "Getting statistics should succeed");
    
    TEST_ASSERT(stats.total_windows == 3, "Should have processed 3 windows");
    TEST_ASSERT(stats.total_tokens > 0, "Should have processed tokens");
    TEST_ASSERT(stats.processing_time_ns > 0, "Should have recorded processing time");
    
    printf("  Statistics: windows=%u, tokens=%u, time=%llu ns\n",
           stats.total_windows, stats.total_tokens, stats.processing_time_ns);
    printf("  Cache: hits=%u, misses=%u\n", 
           stats.cache_hits, stats.cache_misses);
    
    token_sequence_destroy(sequence);
    sequence_manager_destroy(manager);
    TEST_PASS();
}

// Test runner
int run_test(const char* test_name, int (*test_func)()) {
    printf("Running %s...\n", test_name);
    int result = test_func();
    if (result) {
        printf("✓ %s passed\n\n", test_name);
    } else {
        printf("✗ %s failed\n\n", test_name);
    }
    return result;
}

int main() {
    printf("Sequence Manager Test Suite\n");
    printf("===========================\n\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    total_tests++; passed_tests += run_test("Window Config Creation", test_window_config_creation);
    total_tests++; passed_tests += run_test("Window Config Validation", test_window_config_validation);
    total_tests++; passed_tests += run_test("Sequence Manager Creation", test_sequence_manager_creation);
    total_tests++; passed_tests += run_test("Sequence Manager Set Source", test_sequence_manager_set_source);
    total_tests++; passed_tests += run_test("Window Processing", test_window_processing);
    total_tests++; passed_tests += run_test("Window Overlap", test_window_overlap);
    total_tests++; passed_tests += run_test("Window Caching", test_window_caching);
    total_tests++; passed_tests += run_test("Checkpoint Resume", test_checkpoint_resume);
    total_tests++; passed_tests += run_test("Memory Management", test_memory_management);
    total_tests++; passed_tests += run_test("Progress Tracking", test_progress_tracking);
    total_tests++; passed_tests += run_test("Adaptive Window Sizing", test_adaptive_window_sizing);
    total_tests++; passed_tests += run_test("Statistics Collection", test_statistics_collection);
    
    // Summary
    printf("Test Results\n");
    printf("============\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! Sequence Manager implementation is ready.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the implementation.\n");
        return 1;
    }
}
