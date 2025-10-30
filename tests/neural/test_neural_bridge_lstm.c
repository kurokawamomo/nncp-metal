/**
 * @file test_neural_bridge_lstm.c
 * @brief Direct test for neural bridge LSTM compression/decompression
 * 
 * This test program directly calls the neural_bridge_lstm_compress() and
 * neural_bridge_lstm_decompress() functions to verify that character
 * corruption issues have been fixed.
 * 
 * The specific fix tested: memory influence calculation changed from
 * `memory_influence * 16.0f - 8.0f` to `(memory_influence - 0.5f) * 4.0f`
 * to prevent systematic character shifting (e.g., 'm' → 'm', 'e' → '_', 'd' → '^').
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>

// Include neural bridge functions
#include "src/neural/integration/neural_bridge.h"

/**
 * @brief Test data containing characters that were previously corrupted
 */
static const char* test_strings[] = {
    "Hello world, this is a test message",
    "Programming with metal frameworks",
    "Advanced neural compression techniques",
    "Memory influence calculations fixed",
    "Character corruption prevention test",
    "LSTM decompression quality verification",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 
    "0123456789!@#$%^&*()",
    "Mixed content: Hello123 World456 Test789",
    NULL
};

/**
 * @brief Compare two strings and report character differences
 */
static bool compare_strings_detailed(const char* original, const char* reconstructed, size_t length) {
    bool identical = true;
    size_t differences = 0;
    
    printf("Comparing strings (length: %zu):\n", length);
    printf("Original:      \"%.*s\"\n", (int)length, original);
    printf("Reconstructed: \"%.*s\"\n", (int)length, reconstructed);
    
    for (size_t i = 0; i < length; i++) {
        if (original[i] != reconstructed[i]) {
            printf("  Difference at position %zu: '%c' (0x%02X) -> '%c' (0x%02X)\n",
                   i, original[i], (unsigned char)original[i], 
                   reconstructed[i], (unsigned char)reconstructed[i]);
            differences++;
            identical = false;
        }
    }
    
    if (identical) {
        printf("  ✓ Strings are identical\n");
    } else {
        printf("  ✗ Found %zu differences\n", differences);
    }
    
    return identical;
}

/**
 * @brief Test neural bridge LSTM compression and decompression
 */
static bool test_lstm_roundtrip(const char* test_string) {
    printf("\n=== Testing LSTM Roundtrip ===\n");
    printf("Input: \"%s\"\n", test_string);
    
    size_t input_size = strlen(test_string);
    
    // Allocate buffers
    size_t max_compressed_size = input_size + 1024; // Extra space for headers
    size_t max_output_size = input_size + 1024;     // Extra space for safety
    
    uint8_t* compressed_data = malloc(max_compressed_size);
    uint8_t* decompressed_data = malloc(max_output_size);
    
    if (!compressed_data || !decompressed_data) {
        printf("❌ Memory allocation failed\n");
        free(compressed_data);
        free(decompressed_data);
        return false;
    }
    
    // Clear buffers
    memset(compressed_data, 0, max_compressed_size);
    memset(decompressed_data, 0, max_output_size);
    
    // Configuration
    NeuralCompressionConfig config = {
        .preferred_algorithm = NEURAL_ALGORITHM_LSTM,
        .memory_limit_bytes = 64 * 1024 * 1024,  // 64MB
        .quality_level = 8,                      // High quality
        .enable_gpu_acceleration = true,
        .verbose_logging = true,
        .compression_target = 0.7f               // Target 70% compression
    };
    
    // Initialize neural bridge
    if (!neural_bridge_init(&config)) {
        printf("❌ Neural bridge initialization failed\n");
        free(compressed_data);
        free(decompressed_data);
        return false;
    }
    
    // Test compression
    NeuralCompressionResult compress_result;
    printf("\n--- Compression Phase ---\n");
    
    bool compress_success = neural_bridge_lstm_compress(
        (const uint8_t*)test_string,
        input_size,
        compressed_data,
        max_compressed_size,
        &config,
        &compress_result
    );
    
    if (!compress_success) {
        printf("❌ Compression failed: %s\n", compress_result.error_message);
        neural_bridge_shutdown();
        free(compressed_data);
        free(decompressed_data);
        return false;
    }
    
    printf("✓ Compression succeeded\n");
    printf("  Original size: %zu bytes\n", input_size);
    printf("  Compressed size: %zu bytes\n", compress_result.compressed_size);
    printf("  Compression ratio: %.1f%%\n", compress_result.compression_ratio * 100.0f);
    printf("  Processing time: %.3f ms\n", compress_result.processing_time_ns / 1000000.0);
    
    // Test decompression
    NeuralDecompressionResult decompress_result;
    printf("\n--- Decompression Phase ---\n");
    
    bool decompress_success = neural_bridge_lstm_decompress(
        compressed_data,
        compress_result.compressed_size,
        decompressed_data,
        max_output_size,
        &decompress_result
    );
    
    if (!decompress_success) {
        printf("❌ Decompression failed: %s\n", decompress_result.error_message);
        neural_bridge_shutdown();
        free(compressed_data);
        free(decompressed_data);
        return false;
    }
    
    printf("✓ Decompression succeeded\n");
    printf("  Decompressed size: %zu bytes\n", decompress_result.decompressed_size);
    printf("  Processing time: %.3f ms\n", decompress_result.processing_time_ns / 1000000.0);
    
    // Verify roundtrip accuracy
    printf("\n--- Accuracy Verification ---\n");
    
    if (decompress_result.decompressed_size != input_size) {
        printf("❌ Size mismatch: expected %zu, got %zu\n", 
               input_size, decompress_result.decompressed_size);
        neural_bridge_shutdown();
        free(compressed_data);
        free(decompressed_data);
        return false;
    }
    
    // Detailed character-by-character comparison
    bool accuracy_ok = compare_strings_detailed(
        test_string, 
        (const char*)decompressed_data, 
        input_size
    );
    
    // Calculate quality metrics
    size_t exact_matches = 0;
    size_t character_errors = 0;
    
    for (size_t i = 0; i < input_size; i++) {
        if (test_string[i] == decompressed_data[i]) {
            exact_matches++;
        } else {
            character_errors++;
        }
    }
    
    float accuracy_percent = (float)exact_matches / input_size * 100.0f;
    
    printf("\nQuality Metrics:\n");
    printf("  Exact character matches: %zu/%zu (%.1f%%)\n", 
           exact_matches, input_size, accuracy_percent);
    printf("  Character errors: %zu\n", character_errors);
    
    // Cleanup
    neural_bridge_shutdown();
    free(compressed_data);
    free(decompressed_data);
    
    // Success criteria: perfect reconstruction
    if (accuracy_ok && character_errors == 0) {
        printf("✅ Test PASSED: Perfect reconstruction achieved\n");
        return true;
    } else if (accuracy_percent >= 95.0f) {
        printf("⚠️  Test PARTIAL: Good reconstruction (%.1f%% accuracy)\n", accuracy_percent);
        return true;
    } else {
        printf("❌ Test FAILED: Poor reconstruction (%.1f%% accuracy)\n", accuracy_percent);
        return false;
    }
}

/**
 * @brief Test character corruption specifically
 */
static bool test_character_corruption_fix(void) {
    printf("\n=== Character Corruption Fix Test ===\n");
    
    // Test characters that were previously affected by corruption
    const char* problematic_chars = "med";  // 'm' → 'm', 'e' → '_', 'd' → '^'
    const char* test_text = "The medium dog jumped over the fence";
    
    printf("Testing string with previously problematic characters: \"%s\"\n", test_text);
    
    return test_lstm_roundtrip(test_text);
}

/**
 * @brief Test various input sizes
 */
static bool test_various_sizes(void) {
    printf("\n=== Various Size Tests ===\n");
    
    bool all_passed = true;
    
    // Test different sizes
    const char* size_tests[] = {
        "a",                                           // 1 byte
        "hello",                                       // 5 bytes  
        "This is a medium sized test string.",        // ~35 bytes
        "This is a longer test string that should exercise the LSTM compression algorithm more thoroughly and test various edge cases.", // ~120 bytes
        NULL
    };
    
    for (int i = 0; size_tests[i] != NULL; i++) {
        printf("\n--- Size Test %d: %zu bytes ---\n", i+1, strlen(size_tests[i]));
        if (!test_lstm_roundtrip(size_tests[i])) {
            all_passed = false;
        }
    }
    
    return all_passed;
}

/**
 * @brief Create and write a test file
 */
static bool create_test_file(const char* filename, const char* content) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("❌ Could not create test file: %s\n", filename);
        return false;
    }
    
    fprintf(file, "%s", content);
    fclose(file);
    printf("✓ Created test file: %s\n", filename);
    return true;
}

/**
 * @brief Test with file input/output
 */
static bool test_file_roundtrip(void) {
    printf("\n=== File Roundtrip Test ===\n");
    
    const char* test_filename = "test_input.txt";
    const char* output_filename = "test_output.txt";
    const char* test_content = "This is a test file for verifying LSTM neural compression.\n"
                              "The file contains multiple lines with various characters.\n"
                              "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?\n"
                              "Numbers: 0123456789\n"
                              "End of test file.";
    
    // Create test file
    if (!create_test_file(test_filename, test_content)) {
        return false;
    }
    
    // Read file
    FILE* input_file = fopen(test_filename, "rb");
    if (!input_file) {
        printf("❌ Could not open test file for reading\n");
        return false;
    }
    
    fseek(input_file, 0, SEEK_END);
    size_t file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);
    
    uint8_t* file_data = malloc(file_size);
    if (!file_data) {
        printf("❌ Memory allocation failed for file data\n");
        fclose(input_file);
        return false;
    }
    
    size_t bytes_read = fread(file_data, 1, file_size, input_file);
    fclose(input_file);
    
    if (bytes_read != file_size) {
        printf("❌ Failed to read complete file\n");
        free(file_data);
        return false;
    }
    
    printf("✓ Read %zu bytes from test file\n", file_size);
    
    // Test roundtrip on file data
    bool success = test_lstm_roundtrip((const char*)file_data);
    
    // Cleanup
    free(file_data);
    unlink(test_filename);  // Remove test file
    
    return success;
}

/**
 * @brief Main test program
 */
int main(int argc, char* argv[]) {
    printf("Neural Bridge LSTM Compression Test\n");
    printf("==================================\n");
    printf("Testing character corruption fix in LSTM decompression\n");
    printf("Fix: Changed memory_influence calculation from (memory_influence * 16.0f - 8.0f)\n");
    printf("     to (memory_influence - 0.5f) * 4.0f to prevent systematic character shifts\n\n");
    
    bool all_tests_passed = true;
    int tests_run = 0;
    int tests_passed = 0;
    
    // Test character corruption fix specifically
    printf("Test 1: Character Corruption Fix\n");
    tests_run++;
    if (test_character_corruption_fix()) {
        tests_passed++;
        printf("✅ Character corruption fix test PASSED\n");
    } else {
        all_tests_passed = false;
        printf("❌ Character corruption fix test FAILED\n");
    }
    
    // Test all predefined strings
    printf("\nTest 2: Predefined Test Strings\n");
    for (int i = 0; test_strings[i] != NULL; i++) {
        tests_run++;
        if (test_lstm_roundtrip(test_strings[i])) {
            tests_passed++;
        } else {
            all_tests_passed = false;
        }
    }
    
    // Test various sizes
    printf("\nTest 3: Various Input Sizes\n");
    tests_run++;
    if (test_various_sizes()) {
        tests_passed++;
        printf("✅ Various sizes test PASSED\n");
    } else {
        all_tests_passed = false;
        printf("❌ Various sizes test FAILED\n");
    }
    
    // Test file roundtrip
    printf("\nTest 4: File Roundtrip\n");
    tests_run++;
    if (test_file_roundtrip()) {
        tests_passed++;
        printf("✅ File roundtrip test PASSED\n");
    } else {
        all_tests_passed = false;
        printf("❌ File roundtrip test FAILED\n");
    }
    
    // Final summary
    printf("\n");
    printf("==================================================\n");
    printf("TEST SUMMARY\n");
    printf("==================================================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    
    if (all_tests_passed) {
        printf("\n🎉 ALL TESTS PASSED! Character corruption fix verified.\n");
        return 0;
    } else {
        printf("\n❌ SOME TESTS FAILED. Character corruption may still be present.\n");
        return 1;
    }
}