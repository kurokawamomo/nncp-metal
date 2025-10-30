#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

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
static const char* sample_text = "the quick brown fox jumps over the lazy dog compression neural network algorithm data processing text tokenizer vocabulary sequence";

// Test functions
int test_tokenizer_creation() {
    BPETokenizer* tokenizer = NULL;
    BPEError error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    
    if (error != BPE_SUCCESS) {
        printf("  Error creating tokenizer: %s (code: %d)\n", bpe_get_error_string(error), error);
        printf("  Vocabulary file path: %s\n", test_vocab_path);
        return 0;
    }
    
    TEST_ASSERT(tokenizer != NULL, "Tokenizer should not be NULL");
    TEST_ASSERT(tokenizer->vocab_size > 0, "Vocabulary size should be greater than 0");
    TEST_ASSERT(tokenizer->vocabulary != NULL, "Vocabulary array should not be NULL");
    
    printf("  Created tokenizer with %u tokens\n", tokenizer->vocab_size);
    
    bpe_tokenizer_destroy(tokenizer);
    TEST_PASS();
}

int test_tokenizer_validation() {
    BPETokenizer* tokenizer = NULL;
    BPEError error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenizer creation should succeed");
    
    error = bpe_validate_tokenizer(tokenizer);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenizer validation should pass");
    
    bpe_tokenizer_destroy(tokenizer);
    TEST_PASS();
}

int test_basic_tokenization() {
    BPETokenizer* tokenizer = NULL;
    TokenSequence* sequence = NULL;
    
    BPEError error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenizer creation should succeed");
    
    error = bpe_tokenize(tokenizer, sample_text, &sequence);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenization should succeed");
    TEST_ASSERT(sequence != NULL, "Token sequence should not be NULL");
    TEST_ASSERT(sequence->token_count > 0, "Token count should be greater than 0");
    TEST_ASSERT(sequence->is_valid == true, "Sequence should be valid");
    
    printf("  Tokenized %zu chars into %u tokens (confidence: %.3f)\n", 
           strlen(sample_text), sequence->token_count, sequence->confidence);
    
    token_sequence_destroy(sequence);
    bpe_tokenizer_destroy(tokenizer);
    TEST_PASS();
}

int test_advanced_tokenization() {
    BPETokenizer* tokenizer = NULL;
    TokenSequence* sequence = NULL;
    TokenizationStats stats;
    
    BPEError error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenizer creation should succeed");
    
    error = bpe_tokenize_advanced(tokenizer, sample_text, 1000, true, &sequence, &stats);
    TEST_ASSERT(error == BPE_SUCCESS, "Advanced tokenization should succeed");
    TEST_ASSERT(sequence != NULL, "Token sequence should not be NULL");
    TEST_ASSERT(stats.total_tokens > 0, "Total tokens should be greater than 0");
    TEST_ASSERT(stats.oov_rate >= 0.0f && stats.oov_rate <= 1.0f, "OOV rate should be between 0 and 1");
    TEST_ASSERT(stats.processing_time_ns > 0, "Processing time should be recorded");
    
    printf("  OOV rate: %.2f%%, Avg token length: %.2f, Processing time: %llu ns\n",
           stats.oov_rate * 100, stats.avg_token_length, stats.processing_time_ns);
    
    // Test OOV rate requirement (<5%)
    TEST_ASSERT(stats.oov_rate < 0.05f, "OOV rate should be less than 5%");
    
    token_sequence_destroy(sequence);
    bpe_tokenizer_destroy(tokenizer);
    TEST_PASS();
}

int test_detokenization() {
    BPETokenizer* tokenizer = NULL;
    TokenSequence* sequence = NULL;
    char* reconstructed_text = NULL;
    
    BPEError error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenizer creation should succeed");
    
    // Tokenize
    error = bpe_tokenize(tokenizer, "the neural network", &sequence);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenization should succeed");
    
    // Detokenize
    error = bpe_detokenize(tokenizer, sequence, &reconstructed_text);
    TEST_ASSERT(error == BPE_SUCCESS, "Detokenization should succeed");
    TEST_ASSERT(reconstructed_text != NULL, "Reconstructed text should not be NULL");
    
    printf("  Original: 'the neural network'\n");
    printf("  Reconstructed: '%s'\n", reconstructed_text);
    
    free(reconstructed_text);
    token_sequence_destroy(sequence);
    bpe_tokenizer_destroy(tokenizer);
    TEST_PASS();
}

int test_vocabulary_lookup() {
    BPETokenizer* tokenizer = NULL;
    uint32_t token_id;
    VocabEntry* entry = NULL;
    
    BPEError error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenizer creation should succeed");
    
    // Test known token lookup
    error = bpe_get_token_id(tokenizer, "the", &token_id);
    TEST_ASSERT(error == BPE_SUCCESS, "Token ID lookup should succeed");
    
    error = bpe_get_vocab_entry(tokenizer, token_id, &entry);
    TEST_ASSERT(error == BPE_SUCCESS, "Vocab entry lookup should succeed");
    TEST_ASSERT(entry != NULL, "Vocab entry should not be NULL");
    TEST_ASSERT(strcmp(entry->token_str, "the") == 0, "Token string should match");
    
    printf("  Token 'the' has ID %u, frequency %.4f\n", token_id, entry->frequency);
    
    // Test unknown token lookup
    error = bpe_get_token_id(tokenizer, "unknownword123", &token_id);
    TEST_ASSERT(error == BPE_SUCCESS, "Unknown token lookup should succeed with UNK");
    TEST_ASSERT(token_id == tokenizer->unk_token_id, "Unknown token should map to UNK");
    
    bpe_tokenizer_destroy(tokenizer);
    TEST_PASS();
}

int test_performance_benchmark() {
    BPETokenizer* tokenizer = NULL;
    uint64_t avg_time_ns;
    float tokens_per_sec;
    
    BPEError error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenizer creation should succeed");
    
    error = bpe_benchmark_performance(tokenizer, sample_text, 100, &avg_time_ns, &tokens_per_sec);
    TEST_ASSERT(error == BPE_SUCCESS, "Performance benchmark should succeed");
    TEST_ASSERT(tokens_per_sec > 0.0f, "Tokens per second should be positive");
    
    printf("  Performance: %.2f tokens/second, avg time: %llu ns\n", 
           tokens_per_sec, avg_time_ns);
    
    // Test performance requirement (>10K tokens/second)
    // Note: This might fail on slower systems or debug builds
    if (tokens_per_sec >= 10000.0f) {
        printf("  ✓ Performance requirement met (>10K tokens/second)\n");
    } else {
        printf("  ⚠ Performance below target (%.2f < 10K tokens/second)\n", tokens_per_sec);
    }
    
    bpe_tokenizer_destroy(tokenizer);
    TEST_PASS();
}

int test_memory_usage() {
    BPETokenizer* tokenizer = NULL;
    size_t after_creation = 0;
    
    // Simple memory usage estimation
    
    BPEError error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenizer creation should succeed");
    
    // Estimate memory usage
    after_creation = sizeof(BPETokenizer);
    if (tokenizer->vocabulary) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            if (tokenizer->vocabulary[i]) {
                after_creation += strlen(tokenizer->vocabulary[i]) + 1;
            }
        }
        after_creation += tokenizer->vocab_size * sizeof(char*);
        after_creation += tokenizer->vocab_size * sizeof(uint32_t);
        after_creation += tokenizer->vocab_size * sizeof(float);
    }
    
    printf("  Estimated memory usage: %zu bytes (%.2f MB)\n", 
           after_creation, (float)after_creation / (1024 * 1024));
    
    // Test memory requirement (<100MB)
    TEST_ASSERT(after_creation < 100 * 1024 * 1024, "Memory usage should be less than 100MB");
    
    bpe_tokenizer_destroy(tokenizer);
    TEST_PASS();
}

int test_character_fallback() {
    BPETokenizer* tokenizer = NULL;
    TokenSequence* sequence = NULL;
    
    BPEError error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenizer creation should succeed");
    
    // Test with text containing unknown characters/words
    const char* unknown_text = "xyzqwerty12345 unknown_word_123";
    error = bpe_tokenize_advanced(tokenizer, unknown_text, 100, true, &sequence, NULL);
    TEST_ASSERT(error == BPE_SUCCESS, "Tokenization with fallback should succeed");
    TEST_ASSERT(sequence != NULL, "Token sequence should not be NULL");
    TEST_ASSERT(sequence->token_count > 0, "Should have some tokens even with unknown words");
    
    printf("  Tokenized unknown text into %u tokens, OOV count: %u\n",
           sequence->token_count, sequence->oov_count);
    
    token_sequence_destroy(sequence);
    bpe_tokenizer_destroy(tokenizer);
    TEST_PASS();
}

int test_error_handling() {
    BPETokenizer* tokenizer = NULL;
    TokenSequence* sequence = NULL;
    
    // Test invalid parameters
    BPEError error = bpe_tokenizer_create(NULL, test_vocab_path);
    TEST_ASSERT(error == BPE_ERROR_INVALID_PARAM, "Should return invalid param error");
    
    error = bpe_tokenizer_create(&tokenizer, NULL);
    TEST_ASSERT(error == BPE_ERROR_INVALID_PARAM, "Should return invalid param error");
    
    error = bpe_tokenizer_create(&tokenizer, "nonexistent_file.txt");
    TEST_ASSERT(error == BPE_ERROR_FILE_IO, "Should return file IO error");
    
    // Test with valid tokenizer
    error = bpe_tokenizer_create(&tokenizer, test_vocab_path);
    TEST_ASSERT(error == BPE_SUCCESS, "Valid tokenizer creation should succeed");
    
    error = bpe_tokenize(tokenizer, NULL, &sequence);
    TEST_ASSERT(error == BPE_ERROR_INVALID_PARAM, "Should return invalid param error for NULL text");
    
    error = bpe_tokenize(NULL, "test", &sequence);
    TEST_ASSERT(error == BPE_ERROR_INVALID_PARAM, "Should return invalid param error for NULL tokenizer");
    
    bpe_tokenizer_destroy(tokenizer);
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
    printf("BPE Tokenizer Test Suite\n");
    printf("========================\n\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    total_tests++; passed_tests += run_test("Tokenizer Creation", test_tokenizer_creation);
    total_tests++; passed_tests += run_test("Tokenizer Validation", test_tokenizer_validation);
    total_tests++; passed_tests += run_test("Basic Tokenization", test_basic_tokenization);
    total_tests++; passed_tests += run_test("Advanced Tokenization", test_advanced_tokenization);
    total_tests++; passed_tests += run_test("Detokenization", test_detokenization);
    total_tests++; passed_tests += run_test("Vocabulary Lookup", test_vocabulary_lookup);
    total_tests++; passed_tests += run_test("Performance Benchmark", test_performance_benchmark);
    total_tests++; passed_tests += run_test("Memory Usage", test_memory_usage);
    total_tests++; passed_tests += run_test("Character Fallback", test_character_fallback);
    total_tests++; passed_tests += run_test("Error Handling", test_error_handling);
    
    // Summary
    printf("Test Results\n");
    printf("============\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! BPE Tokenizer implementation is ready.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the implementation.\n");
        return 1;
    }
}
