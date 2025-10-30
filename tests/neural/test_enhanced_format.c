#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

#include "../../src/neural/formats/enhanced_nncp_format.h"

// Test utilities
static void print_test_header(const char* test_name) {
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Testing: %s\n", test_name);
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
}

static void print_success(const char* test_name) {
    printf("✅ %s: PASSED\n", test_name);
}

static void print_error(const char* test_name, const char* error) {
    printf("❌ %s: FAILED - %s\n", test_name, error);
}

// Generate test data
static void generate_test_data(uint8_t* buffer, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buffer[i] = (uint8_t)(i % 256);
    }
}

// Test 1: Basic file creation and closing
static int test_file_creation(void) {
    print_test_header("File Creation and Closing");
    
    const char* test_filename = "test_enhanced.nncp";
    ENNCPFile* file = NULL;
    
    // Create file
    ENNCPError error = enncp_create_file(test_filename, &file, ENNCP_DATA_TYPE_BINARY);
    if (error != ENNCP_SUCCESS) {
        print_error("File Creation", enncp_get_error_string(error));
        return 0;
    }
    
    if (!file) {
        print_error("File Creation", "File handle is NULL");
        return 0;
    }
    
    // Verify metadata
    if (file->metadata.magic_number != ENNCP_MAGIC_NUMBER) {
        print_error("Magic Number", "Invalid magic number");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    if (file->metadata.data_type != ENNCP_DATA_TYPE_BINARY) {
        print_error("Data Type", "Data type not set correctly");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("File Creation");
    
    // Close file
    error = enncp_close_file(file);
    if (error != ENNCP_SUCCESS) {
        print_error("File Closing", enncp_get_error_string(error));
        unlink(test_filename);
        return 0;
    }
    
    print_success("File Closing");
    
    // Clean up
    unlink(test_filename);
    
    return 1;
}

// Test 2: Writing and reading data
static int test_write_read_data(void) {
    print_test_header("Data Writing and Reading");
    
    const char* test_filename = "test_data.nncp";
    ENNCPFile* file = NULL;
    
    // Create test data
    const size_t test_size = 8192;
    uint8_t* write_buffer = malloc(test_size);
    uint8_t* read_buffer = malloc(test_size);
    
    if (!write_buffer || !read_buffer) {
        print_error("Memory Allocation", "Failed to allocate buffers");
        free(write_buffer);
        free(read_buffer);
        return 0;
    }
    
    generate_test_data(write_buffer, test_size);
    
    // Create and write to file
    ENNCPError error = enncp_create_file(test_filename, &file, ENNCP_DATA_TYPE_BINARY);
    if (error != ENNCP_SUCCESS) {
        print_error("File Creation", enncp_get_error_string(error));
        free(write_buffer);
        free(read_buffer);
        return 0;
    }
    
    error = enncp_write_data(file, write_buffer, test_size, ENNCP_COMPRESSION_NONE);
    if (error != ENNCP_SUCCESS) {
        print_error("Data Writing", enncp_get_error_string(error));
        enncp_close_file(file);
        free(write_buffer);
        free(read_buffer);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Data Writing");
    
    // Close and reopen for reading
    enncp_close_file(file);
    
    error = enncp_open_file(test_filename, &file, true);
    if (error != ENNCP_SUCCESS) {
        print_error("File Opening", enncp_get_error_string(error));
        free(write_buffer);
        free(read_buffer);
        unlink(test_filename);
        return 0;
    }
    
    // Read data
    size_t bytes_read;
    error = enncp_read_data(file, read_buffer, test_size, &bytes_read);
    if (error != ENNCP_SUCCESS) {
        print_error("Data Reading", enncp_get_error_string(error));
        enncp_close_file(file);
        free(write_buffer);
        free(read_buffer);
        unlink(test_filename);
        return 0;
    }
    
    if (bytes_read != test_size) {
        print_error("Data Size", "Read size doesn't match written size");
        enncp_close_file(file);
        free(write_buffer);
        free(read_buffer);
        unlink(test_filename);
        return 0;
    }
    
    // Verify data
    if (memcmp(write_buffer, read_buffer, test_size) != 0) {
        print_error("Data Verification", "Read data doesn't match written data");
        enncp_close_file(file);
        free(write_buffer);
        free(read_buffer);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Data Reading and Verification");
    
    // Clean up
    enncp_close_file(file);
    free(write_buffer);
    free(read_buffer);
    unlink(test_filename);
    
    return 1;
}

// Test 3: Custom metadata fields
static int test_custom_metadata(void) {
    print_test_header("Custom Metadata Fields");
    
    const char* test_filename = "test_metadata.nncp";
    ENNCPFile* file = NULL;
    
    // Create file
    ENNCPError error = enncp_create_file(test_filename, &file, ENNCP_DATA_TYPE_TEXT);
    if (error != ENNCP_SUCCESS) {
        print_error("File Creation", enncp_get_error_string(error));
        return 0;
    }
    
    // Set custom fields
    error = enncp_set_custom_field(file, "Author", "NNCP Test Suite");
    if (error != ENNCP_SUCCESS) {
        print_error("Set Custom Field (Author)", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    error = enncp_set_custom_field(file, "Version", "1.0.0");
    if (error != ENNCP_SUCCESS) {
        print_error("Set Custom Field (Version)", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    error = enncp_set_custom_field(file, "Description", "Test file for enhanced NNCP format");
    if (error != ENNCP_SUCCESS) {
        print_error("Set Custom Field (Description)", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Setting Custom Fields");
    
    // Get custom fields
    char value[1024];
    error = enncp_get_custom_field(file, "Author", value, sizeof(value));
    if (error != ENNCP_SUCCESS) {
        print_error("Get Custom Field (Author)", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    if (strcmp(value, "NNCP Test Suite") != 0) {
        print_error("Field Value (Author)", "Unexpected value");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Getting Custom Fields");
    
    // List custom fields
    char field_names[10][ENNCP_MAX_FIELD_NAME_LEN];
    uint32_t num_fields;
    error = enncp_list_custom_fields(file, field_names, 10, &num_fields);
    if (error != ENNCP_SUCCESS) {
        print_error("List Custom Fields", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    if (num_fields != 3) {
        print_error("Number of Fields", "Expected 3 fields");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    printf("  Custom fields: ");
    for (uint32_t i = 0; i < num_fields; i++) {
        printf("%s ", field_names[i]);
    }
    printf("\n");
    
    print_success("Listing Custom Fields");
    
    // Remove a field
    error = enncp_remove_custom_field(file, "Version");
    if (error != ENNCP_SUCCESS) {
        print_error("Remove Custom Field", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    // Verify removal
    error = enncp_get_custom_field(file, "Version", value, sizeof(value));
    if (error != ENNCP_ERROR_FIELD_NOT_FOUND) {
        print_error("Field Removal", "Field was not removed");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Removing Custom Fields");
    
    // Clean up
    enncp_close_file(file);
    unlink(test_filename);
    
    return 1;
}

// Test 4: Version and compatibility
static int test_version_compatibility(void) {
    print_test_header("Version and Compatibility");
    
    const char* test_filename = "test_version.nncp";
    ENNCPFile* file = NULL;
    
    // Create file
    ENNCPError error = enncp_create_file(test_filename, &file, ENNCP_DATA_TYPE_BINARY);
    if (error != ENNCP_SUCCESS) {
        print_error("File Creation", enncp_get_error_string(error));
        return 0;
    }
    
    // Check version info
    uint32_t version = file->metadata.format_version;
    char version_string[64];
    error = enncp_get_version_string(version, version_string, sizeof(version_string));
    if (error != ENNCP_SUCCESS) {
        print_error("Version String", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    printf("  Format version: %s\n", version_string);
    printf("  Creator software: %s\n", file->metadata.creator_software);
    printf("  Creator platform: %s\n", file->metadata.creator_platform);
    
    print_success("Version Information");
    
    // Test version components
    uint32_t major = (version >> 24) & 0xFF;
    uint32_t minor = (version >> 16) & 0xFF;
    uint32_t patch = (version >> 8) & 0xFF;
    
    if (major != ENNCP_VERSION_MAJOR || minor != ENNCP_VERSION_MINOR || patch != ENNCP_VERSION_PATCH) {
        print_error("Version Components", "Version mismatch");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Version Component Validation");
    
    // Clean up
    enncp_close_file(file);
    unlink(test_filename);
    
    return 1;
}

// Test 5: File integrity validation
static int test_integrity_validation(void) {
    print_test_header("File Integrity Validation");
    
    const char* test_filename = "test_integrity.nncp";
    ENNCPFile* file = NULL;
    
    // Create file with test data
    ENNCPError error = enncp_create_file(test_filename, &file, ENNCP_DATA_TYPE_BINARY);
    if (error != ENNCP_SUCCESS) {
        print_error("File Creation", enncp_get_error_string(error));
        return 0;
    }
    
    // Write some data
    const char* test_data = "This is test data for integrity validation";
    error = enncp_write_data(file, test_data, strlen(test_data), ENNCP_COMPRESSION_NONE);
    if (error != ENNCP_SUCCESS) {
        print_error("Data Writing", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    // Validate integrity (level 0)
    error = enncp_validate_integrity(file, 0);
    if (error != ENNCP_SUCCESS) {
        print_error("Basic Validation", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Basic Integrity Validation");
    
    // Validate integrity (level 1)
    error = enncp_validate_integrity(file, 1);
    if (error != ENNCP_SUCCESS) {
        print_error("Full Validation", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Full Integrity Validation");
    
    // Close and reopen
    enncp_close_file(file);
    
    error = enncp_open_file(test_filename, &file, true);
    if (error != ENNCP_SUCCESS) {
        print_error("File Reopening", enncp_get_error_string(error));
        unlink(test_filename);
        return 0;
    }
    
    // Validate after reopening
    error = enncp_validate_integrity(file, 2);
    if (error != ENNCP_SUCCESS) {
        print_error("Deep Validation", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Deep Integrity Validation");
    
    // Clean up
    enncp_close_file(file);
    unlink(test_filename);
    
    return 1;
}

// Test 6: Quality metrics
static int test_quality_metrics(void) {
    print_test_header("Quality Metrics");
    
    const char* test_filename = "test_quality.nncp";
    ENNCPFile* file = NULL;
    
    // Create file
    ENNCPError error = enncp_create_file(test_filename, &file, ENNCP_DATA_TYPE_IMAGE);
    if (error != ENNCP_SUCCESS) {
        print_error("File Creation", enncp_get_error_string(error));
        return 0;
    }
    
    // Write data with compression info
    const size_t data_size = 10240;
    uint8_t* test_data = malloc(data_size);
    if (!test_data) {
        print_error("Memory Allocation", "Failed to allocate test data");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    generate_test_data(test_data, data_size);
    
    error = enncp_write_data(file, test_data, data_size, ENNCP_COMPRESSION_NEURAL_QUANTIZATION);
    if (error != ENNCP_SUCCESS) {
        print_error("Data Writing", enncp_get_error_string(error));
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    // Get quality metrics
    ENNCPQualityMetrics metrics;
    error = enncp_get_quality_metrics(file, &metrics);
    if (error != ENNCP_SUCCESS) {
        print_error("Get Quality Metrics", enncp_get_error_string(error));
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    printf("  Compression ratio: %.2f\n", metrics.compression_ratio);
    printf("  Quality score: %.3f\n", metrics.quality_score);
    printf("  Original size: %llu bytes\n", file->metadata.original_size);
    printf("  Compressed size: %llu bytes\n", file->metadata.compressed_size);
    
    print_success("Quality Metrics Retrieval");
    
    // Verify compression stages
    if (file->metadata.num_compression_stages > 0) {
        ENNCPCompressionStage* stage = &file->metadata.compression_stages[0];
        printf("  Compression algorithm: %s\n", 
               enncp_get_compression_name(stage->algorithm));
        printf("  Stage ratio: %.2f\n", stage->compression_ratio);
    }
    
    print_success("Compression Stage Information");
    
    // Clean up
    free(test_data);
    enncp_close_file(file);
    unlink(test_filename);
    
    return 1;
}

// Test 7: Seek and tell operations
static int test_seek_operations(void) {
    print_test_header("Seek and Tell Operations");
    
    const char* test_filename = "test_seek.nncp";
    ENNCPFile* file = NULL;
    
    // Create file with data
    ENNCPError error = enncp_create_file(test_filename, &file, ENNCP_DATA_TYPE_BINARY);
    if (error != ENNCP_SUCCESS) {
        print_error("File Creation", enncp_get_error_string(error));
        return 0;
    }
    
    // Write test data
    const size_t data_size = 1024;
    uint8_t* test_data = malloc(data_size);
    if (!test_data) {
        print_error("Memory Allocation", "Failed to allocate test data");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    for (size_t i = 0; i < data_size; i++) {
        test_data[i] = i & 0xFF;
    }
    
    error = enncp_write_data(file, test_data, data_size, ENNCP_COMPRESSION_NONE);
    if (error != ENNCP_SUCCESS) {
        print_error("Data Writing", enncp_get_error_string(error));
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    // Close and reopen for reading
    enncp_close_file(file);
    error = enncp_open_file(test_filename, &file, true);
    if (error != ENNCP_SUCCESS) {
        print_error("File Reopening", enncp_get_error_string(error));
        free(test_data);
        unlink(test_filename);
        return 0;
    }
    
    // Test tell at start
    uint64_t position;
    error = enncp_tell(file, &position);
    if (error != ENNCP_SUCCESS || position != 0) {
        print_error("Tell at Start", "Expected position 0");
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Tell at Start");
    
    // Seek to middle
    error = enncp_seek(file, data_size / 2, SEEK_SET);
    if (error != ENNCP_SUCCESS) {
        print_error("Seek to Middle", enncp_get_error_string(error));
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    error = enncp_tell(file, &position);
    if (error != ENNCP_SUCCESS || position != data_size / 2) {
        print_error("Tell at Middle", "Unexpected position");
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Seek to Middle");
    
    // Read and verify data at middle
    uint8_t buffer[16];
    size_t bytes_read;
    error = enncp_read_data(file, buffer, sizeof(buffer), &bytes_read);
    if (error != ENNCP_SUCCESS || bytes_read != sizeof(buffer)) {
        print_error("Read at Middle", "Read failed");
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    bool data_correct = true;
    for (size_t i = 0; i < sizeof(buffer); i++) {
        if (buffer[i] != ((data_size / 2 + i) & 0xFF)) {
            data_correct = false;
            break;
        }
    }
    
    if (!data_correct) {
        print_error("Data Verification", "Data mismatch after seek");
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Read After Seek");
    
    // Seek to end
    error = enncp_seek(file, 0, SEEK_END);
    if (error != ENNCP_SUCCESS) {
        print_error("Seek to End", enncp_get_error_string(error));
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    error = enncp_tell(file, &position);
    if (error != ENNCP_SUCCESS || position != data_size) {
        print_error("Tell at End", "Unexpected position");
        free(test_data);
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Seek to End");
    
    // Clean up
    free(test_data);
    enncp_close_file(file);
    unlink(test_filename);
    
    return 1;
}

// Test 8: Error handling
static int test_error_handling(void) {
    print_test_header("Error Handling");
    
    // Test NULL parameters
    ENNCPError error = enncp_create_file(NULL, NULL, ENNCP_DATA_TYPE_BINARY);
    if (error != ENNCP_ERROR_INVALID_PARAM) {
        print_error("NULL Parameters", "Expected INVALID_PARAM error");
        return 0;
    }
    
    print_success("NULL Parameter Handling");
    
    // Test invalid file operations
    ENNCPFile* file = NULL;
    error = enncp_open_file("nonexistent.nncp", &file, true);
    if (error != ENNCP_ERROR_FILE_IO) {
        print_error("Nonexistent File", "Expected FILE_IO error");
        return 0;
    }
    
    print_success("Nonexistent File Handling");
    
    // Test error string function
    const char* error_msg = enncp_get_error_string(ENNCP_ERROR_CHECKSUM_MISMATCH);
    if (!error_msg || strlen(error_msg) == 0) {
        print_error("Error String", "Invalid error message");
        return 0;
    }
    printf("  Example error message: '%s'\n", error_msg);
    
    print_success("Error String Function");
    
    // Test data type names
    const char* type_name = enncp_get_data_type_name(ENNCP_DATA_TYPE_NEURAL_NETWORK);
    if (!type_name || strcmp(type_name, "Neural Network") != 0) {
        print_error("Data Type Name", "Invalid data type name");
        return 0;
    }
    
    print_success("Data Type Name Function");
    
    // Test compression names
    const char* comp_name = enncp_get_compression_name(ENNCP_COMPRESSION_TRANSFORMER);
    if (!comp_name || strcmp(comp_name, "Transformer") != 0) {
        print_error("Compression Name", "Invalid compression name");
        return 0;
    }
    
    print_success("Compression Name Function");
    
    return 1;
}

// Test 9: Block size calculation
static int test_block_size_calculation(void) {
    print_test_header("Block Size Calculation");
    
    struct {
        uint64_t data_size;
        ENNCPDataType data_type;
        uint32_t expected_min;
        uint32_t expected_max;
        const char* description;
    } test_cases[] = {
        {1024, ENNCP_DATA_TYPE_TEXT, 8192, 16384, "Small text file"},
        {1048576, ENNCP_DATA_TYPE_IMAGE, 1048576, 1048576, "1MB image file"},
        {10485760, ENNCP_DATA_TYPE_VIDEO, 1048576, 1048576, "10MB video file"},
        {65536, ENNCP_DATA_TYPE_BINARY, 65536, 65536, "64KB binary file"},
        {512, ENNCP_DATA_TYPE_LOG, 8192, 16384, "Small log file"}
    };
    
    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        uint32_t block_size;
        ENNCPError error = enncp_calculate_optimal_block_size(test_cases[i].data_size,
                                                             test_cases[i].data_type,
                                                             &block_size);
        if (error != ENNCP_SUCCESS) {
            print_error("Block Size Calculation", enncp_get_error_string(error));
            return 0;
        }
        
        if (block_size < test_cases[i].expected_min || block_size > test_cases[i].expected_max) {
            printf("  %s: Block size %u outside expected range [%u, %u]\n",
                   test_cases[i].description, block_size,
                   test_cases[i].expected_min, test_cases[i].expected_max);
            print_error("Block Size Range", "Unexpected block size");
            return 0;
        }
        
        printf("  %s: %u bytes block size\n", test_cases[i].description, block_size);
    }
    
    print_success("Block Size Calculations");
    
    return 1;
}

// Test 10: Streaming mode
static int test_streaming_mode(void) {
    print_test_header("Streaming Mode");
    
    const char* test_filename = "test_streaming.nncp";
    ENNCPFile* file = NULL;
    
    // Create file
    ENNCPError error = enncp_create_file(test_filename, &file, ENNCP_DATA_TYPE_BINARY);
    if (error != ENNCP_SUCCESS) {
        print_error("File Creation", enncp_get_error_string(error));
        return 0;
    }
    
    // Enable streaming with 4KB chunks
    error = enncp_enable_streaming(file, 4096);
    if (error != ENNCP_SUCCESS) {
        print_error("Enable Streaming", enncp_get_error_string(error));
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    // Verify streaming settings
    if (!file->metadata.supports_streaming) {
        print_error("Streaming Flag", "Streaming not enabled");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    if (file->metadata.block_size != 4096) {
        print_error("Block Size", "Incorrect block size");
        enncp_close_file(file);
        unlink(test_filename);
        return 0;
    }
    
    print_success("Streaming Mode Configuration");
    
    // Clean up
    enncp_close_file(file);
    unlink(test_filename);
    
    return 1;
}

// Main test runner
int main(void) {
    printf("Enhanced NNCP Format Test Suite\n");
    printf("===============================\n");
    
    srand((unsigned int)time(NULL));
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    struct {
        const char* name;
        int (*test_func)(void);
    } tests[] = {
        {"File Creation", test_file_creation},
        {"Write/Read Data", test_write_read_data},
        {"Custom Metadata", test_custom_metadata},
        {"Version Compatibility", test_version_compatibility},
        {"Integrity Validation", test_integrity_validation},
        {"Quality Metrics", test_quality_metrics},
        {"Seek Operations", test_seek_operations},
        {"Error Handling", test_error_handling},
        {"Block Size Calculation", test_block_size_calculation},
        {"Streaming Mode", test_streaming_mode}
    };
    
    const int num_tests = sizeof(tests) / sizeof(tests[0]);
    
    for (int i = 0; i < num_tests; i++) {
        total_tests++;
        if (tests[i].test_func()) {
            passed_tests++;
        }
        
        // Small delay between tests
        usleep(50000); // 50ms
    }
    
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Test Summary\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    printf("Success rate: %.1f%%\n", (float)passed_tests / total_tests * 100.0f);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! Enhanced NNCP format is working correctly.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the output above.\n");
        return 1;
    }
}
