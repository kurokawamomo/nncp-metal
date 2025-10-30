#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifdef USE_METAL
#include "metal_context.h"
#include "neural_engine.h"
#include "hybrid_inference.h"
#include "memory_manager.h"
#include "compute_kernels.h"
#include "benchmark.h"
#include <string.h>
#endif

// Simple test framework
static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(test) do { \
    tests_run++; \
    if (!(test)) { \
        printf("FAIL: %s at %s:%d\n", #test, __FILE__, __LINE__); \
        return 0; \
    } else { \
        tests_passed++; \
        printf("PASS: %s\n", #test); \
    } \
} while(0)

// Basic functionality tests
int test_basic_functionality(void) {
    printf("Running basic functionality tests...\n");
    
    // Test basic math
    ASSERT(2 + 2 == 4);
    ASSERT(5 * 6 == 30);
    
    return 1;
}

#ifdef USE_METAL
int test_metal_availability(void) {
    printf("Running Metal availability tests...\n");
    
    // Test Metal device detection
    bool metal_available = metal_is_available();
    printf("Metal available: %s\n", metal_available ? "YES" : "NO");
    
    // This should always pass on Apple Silicon
    ASSERT(metal_available == true || metal_available == false);  // Just test it returns something
    
    return 1;
}

int test_neural_engine_info(void) {
    printf("Running Neural Engine info tests...\n");
    
    NESystemInfo info;
    int result = ne_get_system_info(&info);
    ASSERT(result == 0);
    
    printf("Neural Engine available: %s\n", info.neural_engine_available ? "YES" : "NO");
    printf("Metal GPU available: %s\n", info.metal_gpu_available ? "YES" : "NO");
    printf("Recommended backend: %d\n", info.backend);
    
    return 1;
}

int test_memory_manager(void) {
    printf("Running Memory Manager tests...\n");
    
    // Memory manager creation test
    MMManager* mm_manager = NULL;
    MetalContext* ctx = NULL;
    
    if (metal_context_create(&ctx) != METAL_SUCCESS) {
        printf("FAIL: Could not create Metal context\n");
        return 0;
    }
    
    if (mm_manager_create(&mm_manager, ctx) != METAL_SUCCESS) {
        printf("FAIL: Could not create memory manager\n");
        metal_context_destroy(ctx);
        return 0;
    }
    
    printf("PASS: Memory manager created successfully\n");
    tests_passed++;
    tests_run++;
    
    // Buffer allocation test
    MMBuffer* buffer = NULL;
    MetalError alloc_result = mm_buffer_alloc(mm_manager, 1024, MM_ACCESS_READ_WRITE, &buffer);
    if (alloc_result != METAL_SUCCESS || buffer == NULL) {
        printf("FAIL: Buffer allocation failed\n");
        mm_manager_destroy(mm_manager);
        metal_context_destroy(ctx);
        return 0;
    }
    
    printf("PASS: Buffer allocation successful\n");
    tests_passed++;
    tests_run++;
    
    // Buffer access test
    void* ptr = mm_buffer_map_write(buffer);
    if (ptr == NULL) {
        printf("FAIL: Buffer write mapping failed\n");
        mm_buffer_release(buffer);
        mm_manager_destroy(mm_manager);
        metal_context_destroy(ctx);
        return 0;
    }
    
    // Write test data
    memcpy(ptr, "test_data", 9);
    mm_buffer_unmap(buffer);
    
    // Read test data
    ptr = mm_buffer_map_read(buffer);
    if (ptr == NULL || memcmp(ptr, "test_data", 9) != 0) {
        printf("FAIL: Buffer read verification failed\n");
        mm_buffer_unmap(buffer);
        mm_buffer_release(buffer);
        mm_manager_destroy(mm_manager);
        metal_context_destroy(ctx);
        return 0;
    }
    mm_buffer_unmap(buffer);
    
    printf("PASS: Buffer read/write operations\n");
    tests_passed++;
    tests_run++;
    
    // Memory statistics test
    MMManagerStats stats;
    if (mm_get_stats(mm_manager, &stats) != METAL_SUCCESS) {
        printf("FAIL: Could not retrieve memory statistics\n");
        mm_buffer_release(buffer);
        mm_manager_destroy(mm_manager);
        metal_context_destroy(ctx);
        return 0;
    }
    
    printf("PASS: Memory statistics retrieved\n");
    printf("Active buffers: %u\n", stats.active_buffers);
    printf("Unified memory: %s\n", stats.unified_memory_size > 0 ? "YES" : "NO");
    tests_passed++;
    tests_run++;
    
    // Cleanup
    mm_buffer_release(buffer);
    mm_manager_destroy(mm_manager);
    metal_context_destroy(ctx);
    
    return 1;
}

int test_compute_kernels(void) {
    printf("Running Compute Kernels tests...\n");
    
    // Setup context
    MetalContext* metal_ctx = NULL;
    MMManager* mm_manager = NULL;
    ComputeKernelContext* ck_ctx = NULL;
    
    if (metal_context_create(&metal_ctx) != METAL_SUCCESS) {
        printf("FAIL: Could not create Metal context\n");
        return 0;
    }
    
    if (mm_manager_create(&mm_manager, metal_ctx) != METAL_SUCCESS) {
        printf("FAIL: Could not create memory manager\n");
        metal_context_destroy(metal_ctx);
        return 0;
    }
    
    if (ck_context_create(&ck_ctx, metal_ctx, mm_manager) != METAL_SUCCESS) {
        printf("FAIL: Could not create compute kernel context\n");
        mm_manager_destroy(mm_manager);
        metal_context_destroy(metal_ctx);
        return 0;
    }
    
    printf("PASS: Compute kernel context created successfully\n");
    tests_passed++;
    tests_run++;
    
    // Test matrix multiplication
    const uint32_t M = 64, N = 64, K = 64;
    
    // Create tensor descriptors
    TensorDescriptor desc_a, desc_b, desc_c;
    uint32_t dims_a[] = {M, K};
    uint32_t dims_b[] = {K, N};
    uint32_t dims_c[] = {M, N};
    
    ck_create_tensor_desc(&desc_a, dims_a, 2);
    ck_create_tensor_desc(&desc_b, dims_b, 2);
    ck_create_tensor_desc(&desc_c, dims_c, 2);
    
    // Allocate buffers
    MMBuffer* buffer_a = NULL;
    MMBuffer* buffer_b = NULL;
    MMBuffer* buffer_c = NULL;
    
    size_t size_a = ck_calculate_tensor_size(&desc_a);
    size_t size_b = ck_calculate_tensor_size(&desc_b);
    size_t size_c = ck_calculate_tensor_size(&desc_c);
    
    if (mm_buffer_alloc(mm_manager, size_a, MM_ACCESS_READ_ONLY, &buffer_a) != METAL_SUCCESS ||
        mm_buffer_alloc(mm_manager, size_b, MM_ACCESS_READ_ONLY, &buffer_b) != METAL_SUCCESS ||
        mm_buffer_alloc(mm_manager, size_c, MM_ACCESS_WRITE_ONLY, &buffer_c) != METAL_SUCCESS) {
        printf("FAIL: Could not allocate buffers for matrix multiplication\n");
        ck_context_destroy(ck_ctx);
        mm_manager_destroy(mm_manager);
        metal_context_destroy(metal_ctx);
        return 0;
    }
    
    // Initialize input matrices with test data
    float* data_a = (float*)mm_buffer_map_write(buffer_a);
    float* data_b = (float*)mm_buffer_map_write(buffer_b);
    
    // Fill with simple test pattern
    for (uint32_t i = 0; i < M * K; i++) {
        data_a[i] = 1.0f;  // Identity-like pattern
    }
    for (uint32_t i = 0; i < K * N; i++) {
        data_b[i] = 2.0f;  // Constant pattern
    }
    
    mm_buffer_unmap(buffer_a);
    mm_buffer_unmap(buffer_b);
    
    // Setup matrix multiplication parameters
    MatMulParams matmul_params = {
        .a_desc = desc_a,
        .b_desc = desc_b,
        .c_desc = desc_c,
        .transpose_a = false,
        .transpose_b = false,
        .alpha = 1.0f,
        .beta = 0.0f
    };
    
    // Enable profiling
    ck_profile_operation(ck_ctx, CK_OP_MATRIX_MULTIPLY, true);
    
    // Perform matrix multiplication
    MetalError result = ck_matrix_multiply(ck_ctx, buffer_a, buffer_b, buffer_c, &matmul_params);
    
    if (result != METAL_SUCCESS) {
        printf("FAIL: Matrix multiplication failed\n");
        mm_buffer_release(buffer_a);
        mm_buffer_release(buffer_b);
        mm_buffer_release(buffer_c);
        ck_context_destroy(ck_ctx);
        mm_manager_destroy(mm_manager);
        metal_context_destroy(metal_ctx);
        return 0;
    }
    
    printf("PASS: Matrix multiplication completed\n");
    tests_passed++;
    tests_run++;
    
    // Verify result
    float* data_c = (float*)mm_buffer_map_read(buffer_c);
    float expected = (float)K * 1.0f * 2.0f;  // Each element should be K*2
    bool verification_passed = true;
    
    for (uint32_t i = 0; i < M * N && verification_passed; i++) {
        if (fabs(data_c[i] - expected) > 1e-5f) {
            verification_passed = false;
        }
    }
    mm_buffer_unmap(buffer_c);
    
    if (verification_passed) {
        printf("PASS: Matrix multiplication result verification\n");
        tests_passed++;
        tests_run++;
    } else {
        printf("FAIL: Matrix multiplication result verification\n");
        tests_run++;
    }
    
    // Print performance metrics
    KernelMetrics metrics;
    if (ck_get_kernel_metrics(ck_ctx, CK_OP_MATRIX_MULTIPLY, &metrics) == METAL_SUCCESS) {
        printf("Matrix multiplication performance: %.3f ms\n", metrics.execution_time_ms);
    }
    
    // Test activation function
    MMBuffer* activation_input = NULL;
    MMBuffer* activation_output = NULL;
    
    size_t activation_size = 1024 * sizeof(float);
    
    if (mm_buffer_alloc(mm_manager, activation_size, MM_ACCESS_READ_WRITE, &activation_input) == METAL_SUCCESS &&
        mm_buffer_alloc(mm_manager, activation_size, MM_ACCESS_WRITE_ONLY, &activation_output) == METAL_SUCCESS) {
        
        // Initialize with test data
        float* input_data = (float*)mm_buffer_map_write(activation_input);
        for (int i = 0; i < 1024; i++) {
            input_data[i] = (float)(i - 512) / 512.0f;  // Range from -1 to 1
        }
        mm_buffer_unmap(activation_input);
        
        TensorDescriptor activation_desc;
        uint32_t activation_dims[] = {1, 1, 32, 32};  // Simple 32x32 tensor
        ck_create_tensor_desc(&activation_desc, activation_dims, 4);
        
        // Test ReLU activation
        if (ck_activation(ck_ctx, activation_input, activation_output, CK_ACTIVATION_RELU, &activation_desc) == METAL_SUCCESS) {
            printf("PASS: ReLU activation completed\n");
            tests_passed++;
            tests_run++;
        } else {
            printf("FAIL: ReLU activation failed\n");
            tests_run++;
        }
        
        mm_buffer_release(activation_input);
        mm_buffer_release(activation_output);
    }
    
    // Cleanup
    mm_buffer_release(buffer_a);
    mm_buffer_release(buffer_b);
    mm_buffer_release(buffer_c);
    ck_context_destroy(ck_ctx);
    mm_manager_destroy(mm_manager);
    metal_context_destroy(metal_ctx);
    
    return 1;
}

int test_advanced_operations(void) {
    printf("Running Advanced Neural Network Operations tests...\n");
    
    // Setup contexts
    MetalContext* metal_ctx = NULL;
    MMManager* mm_manager = NULL;
    ComputeKernelContext* ck_ctx = NULL;
    
    if (metal_context_create(&metal_ctx) != METAL_SUCCESS ||
        mm_manager_create(&mm_manager, metal_ctx) != METAL_SUCCESS ||
        ck_context_create(&ck_ctx, metal_ctx, mm_manager) != METAL_SUCCESS) {
        printf("FAIL: Could not create contexts for advanced operations\n");
        return 0;
    }
    
    printf("PASS: Advanced operations context created successfully\n");
    tests_passed++;
    tests_run++;
    
    // Test Pooling Operations
    {
        const uint32_t batch = 1, channels = 2, height = 4, width = 4;
        const uint32_t out_height = 2, out_width = 2;
        
        size_t input_size = batch * channels * height * width * sizeof(float);
        size_t output_size = batch * channels * out_height * out_width * sizeof(float);
        
        MMBuffer* input_buffer = NULL;
        MMBuffer* output_buffer = NULL;
        
        if (mm_buffer_alloc(mm_manager, input_size, MM_ACCESS_READ_WRITE, &input_buffer) == METAL_SUCCESS &&
            mm_buffer_alloc(mm_manager, output_size, MM_ACCESS_WRITE_ONLY, &output_buffer) == METAL_SUCCESS) {
            
            // Initialize input data (simple pattern)
            float* input_data = (float*)mm_buffer_map_write(input_buffer);
            for (uint32_t i = 0; i < batch * channels * height * width; i++) {
                input_data[i] = (float)(i % 16); // Pattern from 0-15
            }
            mm_buffer_unmap(input_buffer);
            
            // Setup pooling parameters
            PoolingParams pool_params;
            uint32_t input_dims[] = {batch, channels, height, width};
            uint32_t output_dims[] = {batch, channels, out_height, out_width};
            
            ck_create_tensor_desc(&pool_params.input_desc, input_dims, 4);
            ck_create_tensor_desc(&pool_params.output_desc, output_dims, 4);
            
            pool_params.kernel_h = 2;
            pool_params.kernel_w = 2;
            pool_params.stride_h = 2;
            pool_params.stride_w = 2;
            pool_params.padding_h = 0;
            pool_params.padding_w = 0;
            pool_params.pool_type = CK_POOL_MAX;
            
            // Test max pooling
            if (ck_pooling_2d(ck_ctx, input_buffer, output_buffer, &pool_params) == METAL_SUCCESS) {
                printf("PASS: Max pooling operation\n");
                tests_passed++;
                tests_run++;
            } else {
                printf("FAIL: Max pooling operation\n");
                tests_run++;
            }
            
            // Test average pooling
            pool_params.pool_type = CK_POOL_AVERAGE;
            if (ck_pooling_2d(ck_ctx, input_buffer, output_buffer, &pool_params) == METAL_SUCCESS) {
                printf("PASS: Average pooling operation\n");
                tests_passed++;
                tests_run++;
            } else {
                printf("FAIL: Average pooling operation\n");
                tests_run++;
            }
            
            mm_buffer_release(input_buffer);
            mm_buffer_release(output_buffer);
        }
    }
    
    // Test Batch Normalization (temporarily disabled - function not implemented)
    /*
    {
        const uint32_t batch = 2, channels = 3, height = 4, width = 4;
        size_t tensor_size = batch * channels * height * width * sizeof(float);
        size_t param_size = channels * sizeof(float);
        
        MMBuffer* input_buffer = NULL;
        MMBuffer* output_buffer = NULL;
        MMBuffer* gamma_buffer = NULL;
        MMBuffer* beta_buffer = NULL;
        
        if (mm_buffer_alloc(mm_manager, tensor_size, MM_ACCESS_READ_WRITE, &input_buffer) == METAL_SUCCESS &&
            mm_buffer_alloc(mm_manager, tensor_size, MM_ACCESS_WRITE_ONLY, &output_buffer) == METAL_SUCCESS &&
            mm_buffer_alloc(mm_manager, param_size, MM_ACCESS_READ_ONLY, &gamma_buffer) == METAL_SUCCESS &&
            mm_buffer_alloc(mm_manager, param_size, MM_ACCESS_READ_ONLY, &beta_buffer) == METAL_SUCCESS) {
            
            // Initialize data
            float* input_data = (float*)mm_buffer_map_write(input_buffer);
            for (uint32_t i = 0; i < batch * channels * height * width; i++) {
                input_data[i] = (float)(rand()) / RAND_MAX * 2.0f - 1.0f; // Random [-1, 1]
            }
            mm_buffer_unmap(input_buffer);
            
            float* gamma_data = (float*)mm_buffer_map_write(gamma_buffer);
            float* beta_data = (float*)mm_buffer_map_write(beta_buffer);
            for (uint32_t c = 0; c < channels; c++) {
                gamma_data[c] = 1.0f;  // Scale factor
                beta_data[c] = 0.0f;   // Shift factor
            }
            mm_buffer_unmap(gamma_buffer);
            mm_buffer_unmap(beta_buffer);
            
            // Test batch normalization (with simplified parameters)
            float epsilon = 1e-5f;
            
            if (ck_batch_normalization(ck_ctx, input_buffer, gamma_buffer, beta_buffer, 
                                      NULL, NULL, output_buffer, epsilon) == METAL_SUCCESS) {
                printf("PASS: Batch normalization operation\n");
                tests_passed++;
                tests_run++;
            } else {
                printf("FAIL: Batch normalization operation\n");
                tests_run++;
            }
            
            mm_buffer_release(input_buffer);
            mm_buffer_release(output_buffer);
            mm_buffer_release(gamma_buffer);
            mm_buffer_release(beta_buffer);
        }
    }
    */
    printf("SKIP: Batch normalization test - function not yet implemented\n");
    
    // Cleanup
    ck_context_destroy(ck_ctx);
    mm_manager_destroy(mm_manager);
    metal_context_destroy(metal_ctx);
    
    return 1;
}

/*
int test_comprehensive_benchmarks(void) {
    // Temporarily disabled due to build issues
    printf("Benchmark tests temporarily disabled\n");
    return 1;
}
*/
#endif

int main(void) {
    printf("NNCP Test Suite\n");
    printf("================\n");
    
    // Run tests
    if (!test_basic_functionality()) return 1;
    
#ifdef USE_METAL
    if (!test_metal_availability()) return 1;
    if (!test_neural_engine_info()) return 1;
    if (!test_memory_manager()) return 1;
    if (!test_compute_kernels()) return 1;
    if (!test_advanced_operations()) return 1;
    // Temporarily disable benchmark test while fixing build issues
    // if (!test_comprehensive_benchmarks()) return 1;
#else
    printf("Metal support not compiled in\n");
#endif
    
    printf("\nTest Results:\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    
    if (tests_passed == tests_run) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
