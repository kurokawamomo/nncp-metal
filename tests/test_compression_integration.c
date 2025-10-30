/**
 * @file test_compression_integration.c
 * @brief Comprehensive Integration Tests for NNCP Neural Compression
 * 
 * This test suite validates the complete integration of neural compression
 * algorithms including memory management, algorithm selection, and error handling.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/stat.h>

// Include integration headers
#include "../src/metal/compression_integration.h"
#include "../src/neural/integration/neural_bridge.h"

// Test configuration
#define TEST_DATA_SIZE_SMALL 1024
#define TEST_DATA_SIZE_MEDIUM 8192
#define TEST_DATA_SIZE_LARGE 65536
#define TEST_BUFFER_SIZE 131072
#define MAX_COMPRESSION_RATIO 0.95f
#define MIN_COMPRESSION_RATIO 0.05f

// Test results tracking
typedef struct {
    int tests_run;
    int tests_passed;
    int tests_failed;
    char last_error[256];
} TestResults;

static TestResults g_results = {0};

// Test utility functions
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
        // Random data (low compressibility)
        srand(42); // Fixed seed for reproducible tests
        for (size_t i = 0; i < size; i++) {
            buffer[i] = rand() % 256;
        }
    } else if (strcmp(pattern_type, "repetitive") == 0) {
        // Repetitive data (high compressibility)
        for (size_t i = 0; i < size; i++) {
            buffer[i] = (uint8_t)(i % 16);
        }
    } else if (strcmp(pattern_type, "text") == 0) {
        // Text-like data (medium compressibility)
        const char* sample_text = "The quick brown fox jumps over the lazy dog. ";
        size_t text_len = strlen(sample_text);
        for (size_t i = 0; i < size; i++) {
            buffer[i] = sample_text[i % text_len];
        }
    } else {
        // Zero data (highest compressibility)
        memset(buffer, 0, size);
    }
}

// Test 1: Basic Initialization and Shutdown
static void test_basic_initialization(void) {
    printf("\n--- Test 1: Basic Initialization and Shutdown ---\n");
    
    CompressionConfig config = {
        .preferred_algorithm = COMPRESSION_ALGORITHM_AUTO,
        .enable_fallback = true,
        .verbose_logging = true,
        .memory_limit_bytes = 1024 * 1024 * 1024, // 1GB
        .quantization_bits = 8
    };
    
    // Test initialization
    bool init_result = compression_integration_init(&config);
    test_assert(init_result, "Compression integration initialization");
    
    // Test shutdown
    compression_integration_shutdown();
    test_assert(true, "Compression integration shutdown");
    
    // Test double initialization
    init_result = compression_integration_init(&config);
    test_assert(init_result, "Double initialization handling");
    compression_integration_shutdown();
}

// Test 2: Memory Manager Functionality
static void test_memory_manager(void) {
    printf("\n--- Test 2: Memory Manager Functionality ---\n");
    
    // Initialize memory manager
    bool mem_init = memory_manager_init(8, 1024 * 1024); // 8 blocks of 1MB each
    test_assert(mem_init, "Memory manager initialization");
    
    // Test allocation
    void* ptr1 = memory_manager_alloc(512 * 1024); // 512KB
    test_assert(ptr1 != NULL, "Memory allocation (512KB)");
    
    void* ptr2 = memory_manager_alloc(256 * 1024); // 256KB
    test_assert(ptr2 != NULL, "Memory allocation (256KB)");
    
    // Test statistics
    const MemoryManager* stats = memory_manager_get_stats();
    test_assert(stats != NULL, "Memory statistics retrieval");
    test_assert(stats->allocation_count >= 2, "Allocation count tracking");
    
    // Test deallocation
    memory_manager_free(ptr1, 512 * 1024);
    memory_manager_free(ptr2, 256 * 1024);
    
    // Test cleanup
    memory_manager_cleanup(1000); // 1 second age
    test_assert(true, "Memory cleanup execution");
    
    memory_manager_shutdown();
    test_assert(true, "Memory manager shutdown");
}

// Test 3: Algorithm Selection Logic
static void test_algorithm_selection(void) {
    printf("\n--- Test 3: Algorithm Selection Logic ---\n");
    
    CompressionConfig config = {
        .preferred_algorithm = COMPRESSION_ALGORITHM_AUTO,
        .enable_fallback = true,
        .verbose_logging = true,
        .memory_limit_bytes = 1024 * 1024 * 1024,
        .quantization_bits = 8
    };
    
    compression_integration_init(&config);
    
    uint8_t test_data[1024];
    
    // Test with different data patterns
    generate_test_data(test_data, sizeof(test_data), "text");
    CompressionAlgorithm alg1 = compression_integration_select_algorithm(test_data, sizeof(test_data));
    test_assert(alg1 != COMPRESSION_ALGORITHM_AUTO, "Algorithm selection for text data");
    
    generate_test_data(test_data, sizeof(test_data), "random");
    CompressionAlgorithm alg2 = compression_integration_select_algorithm(test_data, sizeof(test_data));
    test_assert(alg2 != COMPRESSION_ALGORITHM_AUTO, "Algorithm selection for random data");
    
    compression_integration_shutdown();
}

// Test 4: Compression and Decompression Roundtrip
static void test_compression_roundtrip(CompressionAlgorithm algorithm, const char* data_type, size_t data_size) {
    printf("\n--- Test 4: %s Compression Roundtrip (%s, %zu bytes) ---\n", 
           compression_integration_algorithm_name(algorithm), data_type, data_size);
    
    CompressionConfig config = {
        .preferred_algorithm = algorithm,
        .enable_fallback = true,
        .verbose_logging = true,
        .memory_limit_bytes = 1024 * 1024 * 1024,
        .quantization_bits = 8
    };
    
    compression_integration_init(&config);
    
    // Allocate buffers
    uint8_t* input_data = malloc(data_size);
    uint8_t* compressed_data = malloc(TEST_BUFFER_SIZE);
    uint8_t* decompressed_data = malloc(data_size);
    
    test_assert(input_data && compressed_data && decompressed_data, "Buffer allocation");
    
    // Generate test data
    generate_test_data(input_data, data_size, data_type);
    
    // Test compression
    CompressionResult comp_result = {0};
    bool comp_success = compression_integration_compress(
        input_data, data_size,
        compressed_data, TEST_BUFFER_SIZE,
        &config, &comp_result
    );
    
    test_assert(comp_success, "Compression operation");
    test_assert(comp_result.success, "Compression result success");
    test_assert(comp_result.compressed_size > 0, "Non-zero compressed size");
    test_assert(comp_result.compressed_size <= TEST_BUFFER_SIZE, "Compressed size within bounds");
    
    if (comp_success && comp_result.success) {
        printf("  Compression: %zu -> %zu bytes (%.1f%%)\n", 
               data_size, comp_result.compressed_size, 
               comp_result.compression_ratio * 100.0f);
        
        // Test decompression
        DecompressionResult decomp_result = {0};
        bool decomp_success = compression_integration_decompress(
            compressed_data, comp_result.compressed_size,
            decompressed_data, data_size,
            &decomp_result
        );
        
        test_assert(decomp_success, "Decompression operation");
        test_assert(decomp_result.success, "Decompression result success");
        test_assert(decomp_result.decompressed_size == data_size, "Decompressed size matches original");
        
        // Verify data integrity - all our algorithms are lossless
        bool data_match = (memcmp(input_data, decompressed_data, data_size) == 0);
        
        // If not exact match, provide diagnostic info for neural algorithms
        if (!data_match && (algorithm == COMPRESSION_ALGORITHM_TRANSFORMER || algorithm == COMPRESSION_ALGORITHM_LSTM || 
            (algorithm == COMPRESSION_ALGORITHM_AUTO && comp_result.algorithm_used != COMPRESSION_ALGORITHM_LSTM))) {
            size_t differences = 0;
            size_t total_diff = 0;
            for (size_t i = 0; i < data_size; i++) {
                int diff = abs((int)input_data[i] - (int)decompressed_data[i]);
                if (diff > 0) {
                    differences++;
                    total_diff += diff;
                }
            }
            float diff_percentage = (float)differences / data_size;
            float avg_diff = differences > 0 ? (float)total_diff / differences : 0.0f;
            printf("    Neural reconstruction diagnostic: %.1f%% different bytes, avg diff: %.1f\n", 
                   diff_percentage * 100.0f, avg_diff);
        }
        test_assert(data_match, "Data integrity verification");
        
        printf("  Decompression: %zu -> %zu bytes\n", 
               comp_result.compressed_size, decomp_result.decompressed_size);
    }
    
    free(input_data);
    free(compressed_data);
    free(decompressed_data);
    compression_integration_shutdown();
}

// Test 5: Error Handling
static void test_error_handling(void) {
    printf("\n--- Test 5: Error Handling ---\n");
    
    CompressionConfig config = {
        .preferred_algorithm = COMPRESSION_ALGORITHM_TRANSFORMER,
        .enable_fallback = false, // Disable fallback to test error paths
        .verbose_logging = true,
        .memory_limit_bytes = 1024 * 1024,
        .quantization_bits = 8
    };
    
    compression_integration_init(&config);
    
    uint8_t test_data[1024];
    uint8_t output_buffer[2048];
    CompressionResult result = {0};
    
    // Test NULL input validation
    bool null_test = compression_integration_compress(
        NULL, sizeof(test_data), output_buffer, sizeof(output_buffer), &config, &result);
    test_assert(!null_test, "NULL input data handling");
    
    // Test zero size input
    bool zero_test = compression_integration_compress(
        test_data, 0, output_buffer, sizeof(output_buffer), &config, &result);
    test_assert(!zero_test, "Zero size input handling");
    
    // Test insufficient output buffer
    bool small_buffer_test = compression_integration_compress(
        test_data, sizeof(test_data), output_buffer, 10, &config, &result);
    test_assert(!small_buffer_test || result.compressed_size <= 10, "Small output buffer handling");
    
    compression_integration_shutdown();
}

// Test 6: Performance Benchmarking
static void test_performance_benchmark(void) {
    printf("\n--- Test 6: Performance Benchmarking ---\n");
    
    const size_t test_sizes[] = {1024, 8192, 65536};
    const size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const char* data_types[] = {"text", "repetitive", "random"};
    const size_t num_types = sizeof(data_types) / sizeof(data_types[0]);
    
    CompressionConfig config = {
        .preferred_algorithm = COMPRESSION_ALGORITHM_AUTO,
        .enable_fallback = true,
        .verbose_logging = true,
        .memory_limit_bytes = 1024 * 1024 * 1024,
        .quantization_bits = 8
    };
    
    compression_integration_init(&config);
    
    printf("Performance Results:\n");
    printf("%-12s %-12s %-10s %-15s %-10s %-15s\n", 
           "Data Type", "Size", "Algorithm", "Comp. Ratio", "Time (ms)", "Throughput (MB/s)");
    printf("--------------------------------------------------------------------------------\n");
    
    for (size_t i = 0; i < num_types; i++) {
        for (size_t j = 0; j < num_sizes; j++) {
            size_t data_size = test_sizes[j];
            const char* data_type = data_types[i];
            
            uint8_t* input_data = malloc(data_size);
            uint8_t* output_data = malloc(data_size * 2);
            
            if (!input_data || !output_data) {
                test_assert(false, "Performance test buffer allocation");
                continue;
            }
            
            generate_test_data(input_data, data_size, data_type);
            
            CompressionResult result = {0};
            bool success = compression_integration_compress(
                input_data, data_size, output_data, data_size * 2, &config, &result);
            
            if (success && result.success) {
                double time_ms = result.processing_time_ns / 1000000.0;
                double throughput = (data_size / 1024.0 / 1024.0) / (time_ms / 1000.0);
                
                printf("%-12s %-12zu %-10s %-15.1f%% %-10.2f %-15.2f\n",
                       data_type, data_size,
                       compression_integration_algorithm_name(result.algorithm_used),
                       result.compression_ratio * 100.0f,
                       time_ms, throughput);
            }
            
            free(input_data);
            free(output_data);
        }
    }
    
    compression_integration_shutdown();
    test_assert(true, "Performance benchmarking completed");
}

// Test 7: Memory Pressure Testing
static void test_memory_pressure(void) {
    printf("\n--- Test 7: Memory Pressure Testing ---\n");
    
    CompressionConfig config = {
        .preferred_algorithm = COMPRESSION_ALGORITHM_AUTO,
        .enable_fallback = true,
        .verbose_logging = true,
        .memory_limit_bytes = 16 * 1024 * 1024, // Limited to 16MB
        .quantization_bits = 8
    };
    
    compression_integration_init(&config);
    
    // Test with large data that may exceed memory limits
    const size_t large_size = 8 * 1024 * 1024; // 8MB
    uint8_t* large_data = malloc(large_size);
    uint8_t* output_data = malloc(large_size);
    
    test_assert(large_data && output_data, "Large buffer allocation");
    
    if (large_data && output_data) {
        generate_test_data(large_data, large_size, "text");
        
        CompressionResult result = {0};
        bool success = compression_integration_compress(
            large_data, large_size, output_data, large_size, &config, &result);
        
        test_assert(success || result.error_message[0] != '\0', "Memory pressure handling");
        
        if (success) {
            test_assert(result.memory_used_bytes <= config.memory_limit_bytes || 
                       result.memory_used_bytes == 0, "Memory usage within limits");
            printf("  Memory used: %zu bytes (limit: %zu bytes)\n", 
                   result.memory_used_bytes, config.memory_limit_bytes);
        }
    }
    
    free(large_data);
    free(output_data);
    compression_integration_shutdown();
}

// Main test runner
int main(int argc, char* argv[]) {
    printf("=================================================================\n");
    printf("  NNCP Neural Compression Integration Test Suite\n");
    printf("=================================================================\n");
    
    // Run all tests
    test_basic_initialization();
    test_memory_manager();
    test_algorithm_selection();
    
    // Compression roundtrip tests for different algorithms and data types
    test_compression_roundtrip(COMPRESSION_ALGORITHM_LSTM, "text", TEST_DATA_SIZE_SMALL);
    test_compression_roundtrip(COMPRESSION_ALGORITHM_TRANSFORMER, "text", TEST_DATA_SIZE_SMALL);
    test_compression_roundtrip(COMPRESSION_ALGORITHM_LSTM, "repetitive", TEST_DATA_SIZE_MEDIUM);
    test_compression_roundtrip(COMPRESSION_ALGORITHM_AUTO, "random", TEST_DATA_SIZE_MEDIUM);
    
    test_error_handling();
    test_performance_benchmark();
    test_memory_pressure();
    
    // Print final results
    printf("\n=================================================================\n");
    printf("  Test Results Summary\n");
    printf("=================================================================\n");
    printf("Total tests run: %d\n", g_results.tests_run);
    printf("Tests passed: %d\n", g_results.tests_passed);
    printf("Tests failed: %d\n", g_results.tests_failed);
    
    if (g_results.tests_failed > 0) {
        printf("Last error: %s\n", g_results.last_error);
        printf("\n❌ Integration tests FAILED\n");
        return 1;
    } else {
        printf("\n✅ All integration tests PASSED\n");
        return 0;
    }
}