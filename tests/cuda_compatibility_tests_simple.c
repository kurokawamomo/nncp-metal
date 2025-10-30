/*
 * Simplified CUDA Compatibility Tests
 * 
 * Basic test suite for validating core CUDA compatibility components.
 */

// #include "cuda_compatibility_tests.h" // Removed - unused header
#include "../src/neural/config/cuda_profiles.h"
#include "../src/neural/compatibility/cuda_math_compat.h"
#include "../src/neural/error/cuda_error_handler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Simple test framework
typedef struct {
    const char* name;
    bool (*test_func)(void);
    bool passed;
    double execution_time_ms;
} SimpleTest;

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Basic profile tests
bool test_profile_parameters(void) {
    const CUDAProfile* default_profile = cuda_profile_get("default");
    if (!default_profile) return false;
    
    return default_profile->params.seg_len == 32;
}

bool test_profile_enwik8(void) {
    const CUDAProfile* enwik8_profile = cuda_profile_get("enwik8");
    if (!enwik8_profile) return false;
    
    return enwik8_profile->params.seg_len == 64;
}

bool test_profile_lstm(void) {
    const CUDAProfile* lstm_profile = cuda_profile_get("lstm");
    if (!lstm_profile) return false;
    
    return lstm_profile->params.seg_len == 20;
}

// Basic math compatibility tests
bool test_rng_deterministic(void) {
    CUDACompatRNG rng1, rng2;
    uint64_t seed = 12345;
    
    cuda_compat_rng_init(&rng1, seed);
    cuda_compat_rng_init(&rng2, seed);
    
    if (!rng1.is_initialized || !rng2.is_initialized) return false;
    
    // Test deterministic behavior
    for (int i = 0; i < 5; i++) {
        float val1 = cuda_compat_rng_normal(&rng1, 0.0f, 1.0f);
        float val2 = cuda_compat_rng_normal(&rng2, 0.0f, 1.0f);
        
        if (fabsf(val1 - val2) > 1e-6f) {
            return false;
        }
    }
    
    return true;
}

bool test_tensor_creation(void) {
    size_t shape[] = {32};
    CUDAMathConfig config = {0};
    config.enforce_deterministic = true;
    
    CUDACompatTensor* tensor = cuda_compat_tensor_create(shape, 1, &config);
    if (!tensor) return false;
    
    bool success = (tensor->shape != NULL) && 
                   (tensor->data != NULL) && 
                   (tensor->ndim == 1) && 
                   (tensor->shape[0] == 32);
    
    cuda_compat_tensor_free(tensor);
    return success;
}

// Basic error handling tests
bool test_error_handling(void) {
    // Create testing-optimized error config (non-fatal, quiet)
    CUDAErrorConfig* config = cuda_error_config_create_default();
    if (!config) return false;
    
    // Optimize for testing: disable abort on fatal errors and reduce verbosity
    config->abort_on_fatal_error = false;
    config->log_to_console = false;
    config->min_log_severity = CUDA_ERROR_SEVERITY_ERROR;
    
    bool init_ok = cuda_error_init(config);
    cuda_error_config_free(config);
    
    if (!init_ok) return false;
    
    // Test error setting and retrieval with non-fatal error
    CUDAErrorCode test_error = CUDA_ERROR_INVALID_VALUE; // Non-fatal error
    cuda_error_set(test_error, __FILE__, __LINE__, __FUNCTION__, "Test error");
    
    CUDAErrorCode retrieved = cuda_error_get_last();
    bool codes_match = (retrieved == test_error);
    
    const char* error_string = cuda_error_get_string(test_error);
    bool has_string = (error_string != NULL && strlen(error_string) > 0);
    
    cuda_error_shutdown();
    
    return codes_match && has_string;
}

// Main test execution
int main(void) {
    SimpleTest tests[] = {
        {"Profile Default Parameters", test_profile_parameters, false, 0.0},
        {"Profile Enwik8 Parameters", test_profile_enwik8, false, 0.0},
        {"Profile LSTM Parameters", test_profile_lstm, false, 0.0},
        {"RNG Deterministic Behavior", test_rng_deterministic, false, 0.0},
        {"Tensor Creation", test_tensor_creation, false, 0.0},
        {"Error Handling", test_error_handling, false, 0.0}
    };
    
    size_t num_tests = sizeof(tests) / sizeof(tests[0]);
    size_t passed = 0;
    
    printf("Running CUDA Compatibility Tests...\n");
    printf("=====================================\n");
    
    for (size_t i = 0; i < num_tests; i++) {
        printf("Running: %s... ", tests[i].name);
        
        double start_time = get_time_ms();
        tests[i].passed = tests[i].test_func();
        tests[i].execution_time_ms = get_time_ms() - start_time;
        
        if (tests[i].passed) {
            printf("PASS (%.2f ms)\n", tests[i].execution_time_ms);
            passed++;
        } else {
            printf("FAIL (%.2f ms)\n", tests[i].execution_time_ms);
        }
    }
    
    printf("=====================================\n");
    printf("Test Results: %zu/%zu passed (%.1f%%)\n", 
           passed, num_tests, 
           num_tests > 0 ? (double)passed / num_tests * 100.0 : 0.0);
    
    if (passed == num_tests) {
        printf("✅ ALL TESTS PASSED - CUDA COMPATIBILITY VERIFIED\n");
        return 0;
    } else {
        printf("❌ SOME TESTS FAILED - CUDA COMPATIBILITY ISSUES DETECTED\n");
        return 1;
    }
}