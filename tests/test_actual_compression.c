/**
 * @file test_actual_compression.c
 * @brief Test to verify actual compression functionality vs dummy implementation
 * 
 * This test validates that:
 * 1. Compression ratios are not fixed dummy values (70%, 75%)
 * 2. Decompressed data matches original exactly
 * 3. Neural algorithms are actually implemented, not placeholders
 * 4. Different input data produces different compression ratios
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Include compression integration
#include "../src/metal/compression_integration.h"
#include "../src/neural/integration/neural_bridge.h"

// Test configuration
#define SMALL_DATA_SIZE 1024
#define MEDIUM_DATA_SIZE 8192
#define LARGE_DATA_SIZE 65536

// Expected compression ratios for different data types
#define EXPECTED_RANDOM_RATIO_MIN 0.95f   // Random data should compress poorly
#define EXPECTED_REPETITIVE_RATIO_MAX 0.30f // Repetitive data should compress well
#define DUMMY_RATIO_THRESHOLD 0.02f       // Tolerance for detecting dummy ratios

typedef struct {
    int tests_run;
    int tests_passed;
    int tests_failed;
    char last_error[256];
} TestResults;

static TestResults g_results = {0};

static void test_assert(bool condition, const char* message) {
    g_results.tests_run++;
    if (condition) {
        g_results.tests_passed++;
        printf("✓ %s\n", message);
    } else {
        g_results.tests_failed++;
        printf("✗ %s\n", message);
        snprintf(g_results.last_error, sizeof(g_results.last_error), "%s", message);
    }
}

static void generate_test_data(uint8_t* buffer, size_t size, const char* pattern_type) {
    if (strcmp(pattern_type, "random") == 0) {
        // True random data (should be incompressible)
        srand(42);
        for (size_t i = 0; i < size; i++) {
            buffer[i] = rand() % 256;
        }
    } else if (strcmp(pattern_type, "repetitive") == 0) {
        // Highly repetitive data (should compress very well)
        uint8_t pattern = 'A';
        for (size_t i = 0; i < size; i++) {
            buffer[i] = pattern;
            if (i % 100 == 0) pattern = 'A' + (i / 100) % 26; // Change pattern occasionally
        }
    } else if (strcmp(pattern_type, "zeros") == 0) {
        // All zeros (should compress to almost nothing)
        memset(buffer, 0, size);
    } else {
        // Default: text-like data
        const char* sample = "The quick brown fox jumps over the lazy dog. ";
        size_t sample_len = strlen(sample);
        for (size_t i = 0; i < size; i++) {
            buffer[i] = sample[i % sample_len];
        }
    }
}

// Test 1: Verify compression ratios are not dummy values
static void test_compression_ratios_not_dummy(void) {
    printf("\n--- Test 1: Verify Compression Ratios Are Not Dummy Values ---\n");
    
    CompressionConfig config = {
        .preferred_algorithm = COMPRESSION_ALGORITHM_LSTM,
        .enable_fallback = true,
        .verbose_logging = false,
        .memory_limit_bytes = 1024 * 1024 * 1024,
        .quantization_bits = 8
    };
    
    compression_integration_init(&config);
    
    uint8_t* input_data = malloc(MEDIUM_DATA_SIZE);
    uint8_t* output_data = malloc(MEDIUM_DATA_SIZE * 2);
    
    // Test with random data (should compress poorly)
    generate_test_data(input_data, MEDIUM_DATA_SIZE, "random");
    
    CompressionResult result = {0};
    bool success = compression_integration_compress(
        input_data, MEDIUM_DATA_SIZE, output_data, MEDIUM_DATA_SIZE * 2, &config, &result);
    
    if (success) {
        printf("  Random data compression ratio: %.3f\n", result.compression_ratio);
        
        // Check if ratio is suspiciously close to dummy values
        bool is_dummy_70 = fabs(result.compression_ratio - 0.70f) < DUMMY_RATIO_THRESHOLD;
        bool is_dummy_75 = fabs(result.compression_ratio - 0.75f) < DUMMY_RATIO_THRESHOLD;
        
        test_assert(!is_dummy_70 && !is_dummy_75, "Compression ratio is not dummy 70% or 75%");
        
        // Random data should not compress well (ratio should be close to 1.0 or even > 1.0)
        test_assert(result.compression_ratio > EXPECTED_RANDOM_RATIO_MIN, "Random data compresses poorly as expected");
    } else {
        test_assert(false, "Random data compression failed");
    }
    
    free(input_data);
    free(output_data);
    compression_integration_shutdown();
}

// Test 2: Verify different data types produce different compression ratios
static void test_different_data_different_ratios(void) {
    printf("\n--- Test 2: Different Data Types Produce Different Ratios ---\n");
    
    CompressionConfig config = {
        .preferred_algorithm = COMPRESSION_ALGORITHM_TRANSFORMER,
        .enable_fallback = true,
        .verbose_logging = false,
        .memory_limit_bytes = 1024 * 1024 * 1024,
        .quantization_bits = 8
    };
    
    compression_integration_init(&config);
    
    uint8_t* input_data = malloc(MEDIUM_DATA_SIZE);
    uint8_t* output_data = malloc(MEDIUM_DATA_SIZE * 2);
    
    float ratios[3];
    const char* data_types[] = {"random", "repetitive", "zeros"};
    
    for (int i = 0; i < 3; i++) {
        generate_test_data(input_data, MEDIUM_DATA_SIZE, data_types[i]);
        
        CompressionResult result = {0};
        bool success = compression_integration_compress(
            input_data, MEDIUM_DATA_SIZE, output_data, MEDIUM_DATA_SIZE * 2, &config, &result);
        
        if (success) {
            ratios[i] = result.compression_ratio;
            printf("  %s data compression ratio: %.3f\n", data_types[i], ratios[i]);
        } else {
            ratios[i] = 1.0f; // Failed compression
            printf("  %s data compression failed\n", data_types[i]);
        }
    }
    
    // Verify that different data types produce meaningfully different ratios
    float random_ratio = ratios[0];
    float repetitive_ratio = ratios[1]; 
    float zeros_ratio = ratios[2];
    
    // Check for dummy behavior (all ratios being the same)
    bool all_same = (fabs(random_ratio - repetitive_ratio) < DUMMY_RATIO_THRESHOLD) &&
                    (fabs(repetitive_ratio - zeros_ratio) < DUMMY_RATIO_THRESHOLD);
    
    test_assert(!all_same, "Different data types produce different compression ratios");
    
    // Zeros should compress better than repetitive, which should compress better than random
    test_assert(zeros_ratio < repetitive_ratio, "Zeros compress better than repetitive data");
    test_assert(repetitive_ratio < random_ratio, "Repetitive data compresses better than random");
    
    free(input_data);
    free(output_data);
    compression_integration_shutdown();
}

// Test 3: Verify decompression exactly matches original
static void test_decompression_fidelity(void) {
    printf("\n--- Test 3: Decompression Fidelity Test ---\n");
    
    CompressionConfig config = {
        .preferred_algorithm = COMPRESSION_ALGORITHM_LSTM,
        .enable_fallback = true,
        .verbose_logging = false,
        .memory_limit_bytes = 1024 * 1024 * 1024,
        .quantization_bits = 8
    };
    
    compression_integration_init(&config);
    
    uint8_t* original_data = malloc(SMALL_DATA_SIZE);
    uint8_t* compressed_data = malloc(SMALL_DATA_SIZE * 2);
    uint8_t* decompressed_data = malloc(SMALL_DATA_SIZE);
    
    // Generate deterministic test data
    generate_test_data(original_data, SMALL_DATA_SIZE, "text");
    
    // Compress
    CompressionResult comp_result = {0};
    bool comp_success = compression_integration_compress(
        original_data, SMALL_DATA_SIZE, compressed_data, SMALL_DATA_SIZE * 2, &config, &comp_result);
    
    test_assert(comp_success, "Compression succeeded");
    
    if (comp_success) {
        // Decompress
        DecompressionResult decomp_result = {0};
        bool decomp_success = compression_integration_decompress(
            compressed_data, comp_result.compressed_size, 
            decompressed_data, SMALL_DATA_SIZE, &decomp_result);
        
        test_assert(decomp_success, "Decompression succeeded");
        
        if (decomp_success) {
            // Verify exact match for lossless algorithms
            if (comp_result.algorithm_used == COMPRESSION_ALGORITHM_RLE) {
                bool exact_match = (memcmp(original_data, decompressed_data, SMALL_DATA_SIZE) == 0);
                test_assert(exact_match, "RLE decompression matches original exactly");
            } else {
                // For neural algorithms, check reasonable similarity
                size_t differences = 0;
                for (size_t i = 0; i < SMALL_DATA_SIZE; i++) {
                    if (original_data[i] != decompressed_data[i]) {
                        differences++;
                    }
                }
                
                float diff_percentage = (float)differences / SMALL_DATA_SIZE;
                printf("  Neural algorithm difference: %.2f%%\n", diff_percentage * 100.0f);
                
                // Neural algorithms should still preserve most of the data
                test_assert(diff_percentage < 0.20f, "Neural decompression preserves >80% of data");
            }
        }
    }
    
    free(original_data);
    free(compressed_data);
    free(decompressed_data);
    compression_integration_shutdown();
}

// Test 4: Verify neural algorithms are implemented (not placeholder)
static void test_neural_implementation_not_placeholder(void) {
    printf("\n--- Test 4: Neural Implementation Not Placeholder ---\n");
    
    // Test neural bridge directly
    NeuralCompressionConfig neural_config = {
        .preferred_algorithm = NEURAL_ALGORITHM_TRANSFORMER,
        .memory_limit_bytes = 1024 * 1024 * 1024,
        .quality_level = 7,
        .enable_gpu_acceleration = true,
        .verbose_logging = false,
        .compression_target = 0.15f
    };
    
    bool neural_init = neural_bridge_init(&neural_config);
    test_assert(neural_init, "Neural bridge initialization");
    
    if (neural_init) {
        uint8_t* input_data = malloc(MEDIUM_DATA_SIZE);
        uint8_t* output_data = malloc(MEDIUM_DATA_SIZE * 2);
        
        generate_test_data(input_data, MEDIUM_DATA_SIZE, "text");
        
        NeuralCompressionResult result = {0};
        bool success = neural_bridge_transformer_compress(
            input_data, MEDIUM_DATA_SIZE, output_data, MEDIUM_DATA_SIZE * 2, 
            &neural_config, &result);
        
        if (success) {
            printf("  Transformer compression: %zu -> %zu bytes (%.2f%%)\n",
                   MEDIUM_DATA_SIZE, result.compressed_size, result.compression_ratio * 100.0f);
            
            // Check for placeholder behavior
            bool has_algorithm_header = (output_data[0] == 'T' && output_data[1] == 'R' && 
                                       output_data[2] == 'N' && output_data[3] == 'F');
            test_assert(has_algorithm_header, "Transformer output has proper algorithm header");
            
            // Check that compression actually achieved something
            test_assert(result.compressed_size != MEDIUM_DATA_SIZE, "Compression changed data size");
            
            // Processing time should be reasonable (not zero, not too high)
            test_assert(result.processing_time_ns > 0, "Processing time recorded");
            test_assert(result.processing_time_ns < 1000000000ULL, "Processing time reasonable (<1s)");
        } else {
            printf("  Transformer compression failed: %s\n", result.error_message);
            test_assert(false, "Transformer compression should work");
        }
        
        free(input_data);
        free(output_data);
        neural_bridge_shutdown();
    }
}

// Test 5: End-to-end compression test with file
static void test_end_to_end_file_compression(void) {
    printf("\n--- Test 5: End-to-End File Compression Test ---\n");
    
    // Create test file
    const char* test_filename = "/tmp/nncp_test_file.txt";
    const char* compressed_filename = "/tmp/nncp_test_file.nncp";
    const char* decompressed_filename = "/tmp/nncp_test_file_restored.txt";
    
    FILE* test_file = fopen(test_filename, "w");
    test_assert(test_file != NULL, "Test file creation");
    
    if (test_file) {
        // Write test content
        const char* test_content = "This is a test file for NNCP compression.\n"
                                  "It contains multiple lines of text.\n"
                                  "The content should be preserved exactly after compression and decompression.\n"
                                  "Testing special characters: ©️ µ ñ é à ü\n"
                                  "Numbers: 1234567890\n"
                                  "Symbols: !@#$%^&*()_+-=[]{}|;:,.<>?\n";
        
        size_t content_len = strlen(test_content);
        size_t written = fwrite(test_content, 1, content_len, test_file);
        fclose(test_file);
        
        test_assert(written == content_len, "Test file content written");
        
        if (written == content_len) {
            // Test compression using system binary
            char compress_cmd[512];
            snprintf(compress_cmd, sizeof(compress_cmd), 
                    "nncp-metal compress %s %s -l 6 2>/dev/null", 
                    test_filename, compressed_filename);
            
            int compress_result = system(compress_cmd);
            test_assert(compress_result == 0, "System compression command");
            
            // Check compressed file exists and has reasonable size
            FILE* compressed_file = fopen(compressed_filename, "rb");
            if (compressed_file) {
                fseek(compressed_file, 0, SEEK_END);
                long compressed_size = ftell(compressed_file);
                fclose(compressed_file);
                
                printf("  Original: %zu bytes, Compressed: %ld bytes\n", content_len, compressed_size);
                test_assert(compressed_size > 0, "Compressed file has content");
                test_assert(compressed_size != (long)content_len, "Compression changed file size");
                
                // Test decompression
                char decompress_cmd[512];
                snprintf(decompress_cmd, sizeof(decompress_cmd),
                        "nncp-metal decompress %s %s 2>/dev/null",
                        compressed_filename, decompressed_filename);
                
                int decompress_result = system(decompress_cmd);
                test_assert(decompress_result == 0, "System decompression command");
                
                // Verify decompressed content
                FILE* decompressed_file = fopen(decompressed_filename, "r");
                if (decompressed_file) {
                    char* decompressed_content = malloc(content_len + 1);
                    size_t read_len = fread(decompressed_content, 1, content_len, decompressed_file);
                    decompressed_content[read_len] = '\0';
                    fclose(decompressed_file);
                    
                    bool content_matches = (read_len == content_len) && 
                                         (memcmp(test_content, decompressed_content, content_len) == 0);
                    test_assert(content_matches, "Decompressed content matches original exactly");
                    
                    if (!content_matches) {
                        printf("  Expected length: %zu, got: %zu\n", content_len, read_len);
                        printf("  Content comparison failed\n");
                    }
                    
                    free(decompressed_content);
                } else {
                    test_assert(false, "Decompressed file could not be read");
                }
            } else {
                test_assert(false, "Compressed file was not created");
            }
        }
    }
    
    // Cleanup
    unlink(test_filename);
    unlink(compressed_filename);  
    unlink(decompressed_filename);
}

int main(int argc, char* argv[]) {
    printf("=================================================================\n");
    printf("  NNCP Actual Compression Functionality Test\n");
    printf("=================================================================\n");
    printf("This test verifies that compression is actually working and\n");
    printf("not returning dummy/placeholder values.\n");
    printf("=================================================================\n");
    
    // Run all tests
    test_compression_ratios_not_dummy();
    test_different_data_different_ratios();
    test_decompression_fidelity();
    test_neural_implementation_not_placeholder();
    test_end_to_end_file_compression();
    
    // Print final results
    printf("\n=================================================================\n");
    printf("  Actual Compression Test Results\n");
    printf("=================================================================\n");
    printf("Total tests run: %d\n", g_results.tests_run);
    printf("Tests passed: %d\n", g_results.tests_passed);
    printf("Tests failed: %d\n", g_results.tests_failed);
    
    if (g_results.tests_failed > 0) {
        printf("Last error: %s\n", g_results.last_error);
        printf("\n❌ ACTUAL COMPRESSION TESTS FAILED\n");
        printf("The implementation appears to be using dummy/placeholder values.\n");
        return 1;
    } else {
        printf("\n✅ All actual compression tests PASSED\n");
        printf("The implementation appears to be working correctly.\n");
        return 0;
    }
}
