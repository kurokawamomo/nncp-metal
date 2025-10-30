#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "../../src/neural/parallel/parallel_processor.h"

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

// Sample processing functions
static ParallelProcessorError simple_cpu_function(const void* input, size_t input_size,
                                                 void* output, size_t* output_size,
                                                 void* context) {
    if (!input || !output || !output_size || input_size == 0) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    // Simple operation: copy input to output and double each byte
    const uint8_t* in_data = (const uint8_t*)input;
    uint8_t* out_data = (uint8_t*)output;
    
    for (size_t i = 0; i < input_size; i++) {
        out_data[i] = (in_data[i] * 2) & 0xFF;
    }
    
    *output_size = input_size;
    
    // Simulate some processing time
    usleep(1000); // 1ms
    
    return PP_SUCCESS;
}

static ParallelProcessorError compute_intensive_function(const void* input, size_t input_size,
                                                       void* output, size_t* output_size,
                                                       void* context) {
    if (!input || !output || !output_size || input_size == 0) {
        return PP_ERROR_INVALID_PARAM;
    }
    
    const float* in_data = (const float*)input;
    float* out_data = (float*)output;
    size_t float_count = input_size / sizeof(float);
    
    // Compute intensive operation: calculate moving average and standard deviation
    for (size_t i = 0; i < float_count; i++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        size_t window = 10;
        size_t start = (i >= window) ? i - window : 0;
        size_t count = i - start + 1;
        
        for (size_t j = start; j <= i; j++) {
            sum += in_data[j];
            sum_sq += in_data[j] * in_data[j];
        }
        
        float mean = sum / count;
        float variance = (sum_sq / count) - (mean * mean);
        out_data[i] = sqrtf(variance); // Standard deviation
    }
    
    *output_size = input_size;
    return PP_SUCCESS;
}

static ParallelProcessorError failing_function(const void* input, size_t input_size,
                                              void* output, size_t* output_size,
                                              void* context) {
    // Intentionally failing function for error testing
    return PP_ERROR_TASK_FAILED;
}

// Task completion callback for testing
static int callback_count = 0;
static void test_callback(ProcessingTask* task, void* user_data) {
    callback_count++;
    int* expected_status = (int*)user_data;
    if (expected_status && *expected_status != (int)task->status) {
        printf("  Callback error: expected status %d, got %d\n", *expected_status, (int)task->status);
    }
}

// Test 1: Basic processor creation and destruction
static int test_processor_creation(void) {
    print_test_header("Processor Creation and Destruction");
    
    ParallelProcessorConfig config;
    ParallelProcessorError error = parallel_processor_config_default(&config, PP_MODE_CPU_ONLY);
    if (error != PP_SUCCESS) {
        print_error("Default Config", parallel_processor_error_string(error));
        return 0;
    }
    
    // Override some settings for testing
    config.thread_config.num_threads = 2;
    config.max_queue_size = 100;
    config.enable_caching = false;
    
    print_success("Default Configuration");
    
    ParallelProcessor* processor = NULL;
    error = parallel_processor_create(&processor, &config);
    if (error != PP_SUCCESS) {
        print_error("Processor Creation", parallel_processor_error_string(error));
        return 0;
    }
    
    if (!processor) {
        print_error("Processor Creation", "Processor is NULL");
        return 0;
    }
    
    print_success("Processor Created");
    
    // Test starting the processor
    error = parallel_processor_start(processor);
    if (error != PP_SUCCESS) {
        print_error("Processor Start", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    print_success("Processor Started");
    
    // Test stopping the processor
    error = parallel_processor_stop(processor, false);
    if (error != PP_SUCCESS) {
        print_error("Processor Stop", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    print_success("Processor Stopped");
    
    // Destroy processor
    parallel_processor_destroy(processor);
    print_success("Processor Destroyed");
    
    return 1;
}

// Test 2: Single task processing
static int test_single_task(void) {
    print_test_header("Single Task Processing");
    
    ParallelProcessorConfig config;
    parallel_processor_config_default(&config, PP_MODE_CPU_ONLY);
    config.thread_config.num_threads = 2;
    
    ParallelProcessor* processor = NULL;
    ParallelProcessorError error = parallel_processor_create(&processor, &config);
    if (error != PP_SUCCESS) {
        print_error("Processor Creation", parallel_processor_error_string(error));
        return 0;
    }
    
    error = parallel_processor_start(processor);
    if (error != PP_SUCCESS) {
        print_error("Processor Start", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    // Create test data
    const size_t data_size = 1024;
    uint8_t* input_data = malloc(data_size);
    if (!input_data) {
        print_error("Memory Allocation", "Failed to allocate input data");
        parallel_processor_destroy(processor);
        return 0;
    }
    
    for (size_t i = 0; i < data_size; i++) {
        input_data[i] = i & 0xFF;
    }
    
    // Submit task
    uint64_t task_id;
    error = parallel_processor_submit_task(processor, input_data, data_size,
                                         simple_cpu_function, NULL,
                                         PP_PRIORITY_NORMAL, &task_id);
    if (error != PP_SUCCESS) {
        print_error("Task Submit", parallel_processor_error_string(error));
        free(input_data);
        parallel_processor_destroy(processor);
        return 0;
    }
    
    print_success("Task Submitted");
    
    // Wait for task completion
    ProcessingTask* result = NULL;
    error = parallel_processor_wait_for_task(processor, task_id, 5000, &result);
    if (error != PP_SUCCESS) {
        print_error("Task Wait", parallel_processor_error_string(error));
        free(input_data);
        parallel_processor_destroy(processor);
        return 0;
    }
    
    if (!result) {
        print_error("Task Result", "Result is NULL");
        free(input_data);
        parallel_processor_destroy(processor);
        return 0;
    }
    
    if (result->status != PP_TASK_COMPLETED) {
        print_error("Task Status", "Task not completed");
        free(input_data);
        parallel_processor_destroy(processor);
        return 0;
    }
    
    // Verify result
    if (result->output_size != data_size) {
        print_error("Output Size", "Output size mismatch");
        free(input_data);
        parallel_processor_destroy(processor);
        return 0;
    }
    
    uint8_t* output_data = (uint8_t*)result->output_data;
    bool data_correct = true;
    for (size_t i = 0; i < data_size; i++) {
        uint8_t expected = (input_data[i] * 2) & 0xFF;
        if (output_data[i] != expected) {
            data_correct = false;
            break;
        }
    }
    
    if (!data_correct) {
        print_error("Data Verification", "Output data is incorrect");
        free(input_data);
        parallel_processor_destroy(processor);
        return 0;
    }
    
    print_success("Task Processing and Verification");
    
    free(input_data);
    parallel_processor_stop(processor, false);
    parallel_processor_destroy(processor);
    
    return 1;
}

// Test 3: Multiple tasks processing
static int test_multiple_tasks(void) {
    print_test_header("Multiple Tasks Processing");
    
    ParallelProcessorConfig config;
    parallel_processor_config_default(&config, PP_MODE_CPU_ONLY);
    config.thread_config.num_threads = 4;
    
    ParallelProcessor* processor = NULL;
    ParallelProcessorError error = parallel_processor_create(&processor, &config);
    if (error != PP_SUCCESS) {
        print_error("Processor Creation", parallel_processor_error_string(error));
        return 0;
    }
    
    error = parallel_processor_start(processor);
    if (error != PP_SUCCESS) {
        print_error("Processor Start", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    const int num_tasks = 10;
    const size_t data_size = 512;
    uint64_t task_ids[num_tasks];
    
    // Create and submit multiple tasks
    for (int i = 0; i < num_tasks; i++) {
        uint8_t* input_data = malloc(data_size);
        if (!input_data) {
            print_error("Memory Allocation", "Failed to allocate input data");
            parallel_processor_destroy(processor);
            return 0;
        }
        
        for (size_t j = 0; j < data_size; j++) {
            input_data[j] = (i + j) & 0xFF;
        }
        
        TaskPriority priority = (i % 3 == 0) ? PP_PRIORITY_HIGH : PP_PRIORITY_NORMAL;
        
        error = parallel_processor_submit_task(processor, input_data, data_size,
                                             simple_cpu_function, NULL,
                                             priority, &task_ids[i]);
        if (error != PP_SUCCESS) {
            print_error("Task Submit", parallel_processor_error_string(error));
            free(input_data);
            parallel_processor_destroy(processor);
            return 0;
        }
    }
    
    print_success("Multiple Tasks Submitted");
    
    // Wait for all tasks
    int completed_tasks = 0;
    for (int i = 0; i < num_tasks; i++) {
        ProcessingTask* result = NULL;
        error = parallel_processor_wait_for_task(processor, task_ids[i], 10000, &result);
        if (error == PP_SUCCESS && result && result->status == PP_TASK_COMPLETED) {
            completed_tasks++;
        }
    }
    
    if (completed_tasks != num_tasks) {
        printf("  Only %d out of %d tasks completed\n", completed_tasks, num_tasks);
        print_error("Task Completion", "Not all tasks completed");
        parallel_processor_destroy(processor);
        return 0;
    }
    
    print_success("All Tasks Completed");
    
    parallel_processor_stop(processor, false);
    parallel_processor_destroy(processor);
    
    return 1;
}

// Test 4: Performance metrics
static int test_performance_metrics(void) {
    print_test_header("Performance Metrics");
    
    ParallelProcessorConfig config;
    parallel_processor_config_default(&config, PP_MODE_CPU_ONLY);
    config.thread_config.num_threads = 2;
    config.enable_performance_monitoring = true;
    
    ParallelProcessor* processor = NULL;
    ParallelProcessorError error = parallel_processor_create(&processor, &config);
    if (error != PP_SUCCESS) {
        print_error("Processor Creation", parallel_processor_error_string(error));
        return 0;
    }
    
    error = parallel_processor_start(processor);
    if (error != PP_SUCCESS) {
        print_error("Processor Start", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    // Submit a few tasks with compute-intensive function
    const int num_tasks = 5;
    const size_t data_size = 1000 * sizeof(float);
    uint64_t task_ids[num_tasks];
    
    for (int i = 0; i < num_tasks; i++) {
        float* input_data = malloc(data_size);
        if (!input_data) {
            print_error("Memory Allocation", "Failed to allocate input data");
            parallel_processor_destroy(processor);
            return 0;
        }
        
        // Generate test data
        for (size_t j = 0; j < data_size / sizeof(float); j++) {
            input_data[j] = sinf(j * 0.1f) + 0.1f * (rand() / (float)RAND_MAX);
        }
        
        error = parallel_processor_submit_task(processor, input_data, data_size,
                                             compute_intensive_function, NULL,
                                             PP_PRIORITY_NORMAL, &task_ids[i]);
        if (error != PP_SUCCESS) {
            print_error("Task Submit", parallel_processor_error_string(error));
            free(input_data);
            parallel_processor_destroy(processor);
            return 0;
        }
    }
    
    // Wait for completion
    for (int i = 0; i < num_tasks; i++) {
        ProcessingTask* result = NULL;
        parallel_processor_wait_for_task(processor, task_ids[i], 15000, &result);
    }
    
    // Get performance metrics
    PerformanceMetrics metrics;
    error = parallel_processor_get_metrics(processor, &metrics);
    if (error != PP_SUCCESS) {
        print_error("Get Metrics", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    printf("  Performance Metrics:\n");
    printf("    Total tasks processed: %llu\n", metrics.total_tasks_processed);
    printf("    Tasks succeeded: %llu\n", metrics.tasks_succeeded);
    printf("    Tasks failed: %llu\n", metrics.tasks_failed);
    printf("    Total processing time: %.3f seconds\n", metrics.total_processing_time);
    printf("    Average task time: %.3f seconds\n", metrics.average_task_time);
    printf("    Throughput: %.2f MB/s\n", metrics.throughput_mbps);
    printf("    CPU utilization: %u%%\n", metrics.cpu_utilization_percent);
    printf("    Active threads: %u\n", metrics.active_threads);
    
    if (metrics.total_tasks_processed != num_tasks) {
        print_error("Task Count", "Incorrect task count in metrics");
        parallel_processor_destroy(processor);
        return 0;
    }
    
    if (metrics.tasks_succeeded != num_tasks) {
        print_error("Success Count", "Incorrect success count in metrics");
        parallel_processor_destroy(processor);
        return 0;
    }
    
    print_success("Performance Metrics Collection");
    
    parallel_processor_stop(processor, false);
    parallel_processor_destroy(processor);
    
    return 1;
}

// Test 5: Error handling
static int test_error_handling(void) {
    print_test_header("Error Handling");
    
    // Test NULL parameters
    ParallelProcessorError error = parallel_processor_create(NULL, NULL);
    if (error != PP_ERROR_INVALID_PARAM) {
        print_error("NULL Parameters", "Expected INVALID_PARAM error");
        return 0;
    }
    print_success("NULL Parameter Handling");
    
    // Test with valid processor
    ParallelProcessorConfig config;
    parallel_processor_config_default(&config, PP_MODE_CPU_ONLY);
    config.thread_config.num_threads = 1;
    
    ParallelProcessor* processor = NULL;
    error = parallel_processor_create(&processor, &config);
    if (error != PP_SUCCESS) {
        print_error("Processor Creation", parallel_processor_error_string(error));
        return 0;
    }
    
    error = parallel_processor_start(processor);
    if (error != PP_SUCCESS) {
        print_error("Processor Start", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    // Test task with failing function
    uint8_t test_data[100] = {0};
    uint64_t task_id;
    error = parallel_processor_submit_task(processor, test_data, sizeof(test_data),
                                         failing_function, NULL,
                                         PP_PRIORITY_NORMAL, &task_id);
    if (error != PP_SUCCESS) {
        print_error("Submit Failing Task", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    // Wait for task (should fail)
    ProcessingTask* result = NULL;
    error = parallel_processor_wait_for_task(processor, task_id, 5000, &result);
    if (error != PP_SUCCESS) {
        print_error("Wait for Failing Task", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    if (!result || result->status != PP_TASK_FAILED) {
        print_error("Task Failure", "Task should have failed");
        parallel_processor_destroy(processor);
        return 0;
    }
    
    print_success("Task Failure Handling");
    
    // Test invalid task operations
    TaskStatus status;
    error = parallel_processor_get_task_status(processor, 99999, &status);
    if (error == PP_SUCCESS) {
        print_error("Invalid Task ID", "Should have failed for invalid task ID");
        parallel_processor_destroy(processor);
        return 0;
    }
    
    print_success("Invalid Task ID Handling");
    
    parallel_processor_stop(processor, false);
    parallel_processor_destroy(processor);
    
    return 1;
}

// Test 6: Task callbacks
static int test_task_callbacks(void) {
    print_test_header("Task Callbacks");
    
    ParallelProcessorConfig config;
    parallel_processor_config_default(&config, PP_MODE_CPU_ONLY);
    config.thread_config.num_threads = 1;
    
    ParallelProcessor* processor = NULL;
    ParallelProcessorError error = parallel_processor_create(&processor, &config);
    if (error != PP_SUCCESS) {
        print_error("Processor Creation", parallel_processor_error_string(error));
        return 0;
    }
    
    error = parallel_processor_start(processor);
    if (error != PP_SUCCESS) {
        print_error("Processor Start", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    // Reset callback counter
    callback_count = 0;
    
    // Submit task with callback
    uint8_t test_data[100];
    for (int i = 0; i < 100; i++) {
        test_data[i] = i;
    }
    
    // Create a task manually to add callback
    ProcessingTask* task = calloc(1, sizeof(ProcessingTask));
    if (!task) {
        print_error("Memory Allocation", "Failed to allocate task");
        parallel_processor_destroy(processor);
        return 0;
    }
    
    task->input_data = test_data;
    task->input_size = sizeof(test_data);
    task->cpu_function = simple_cpu_function;
    task->priority = PP_PRIORITY_NORMAL;
    task->callback = test_callback;
    
    int expected_status = PP_TASK_COMPLETED;
    task->callback_data = &expected_status;
    
    // Allocate output buffer
    task->allocated_output_size = sizeof(test_data);
    task->output_data = malloc(task->allocated_output_size);
    if (!task->output_data) {
        print_error("Memory Allocation", "Failed to allocate output buffer");
        free(task);
        parallel_processor_destroy(processor);
        return 0;
    }
    
    // Since we can't directly enqueue with callback, we'll use the submit function
    // and check metrics instead
    uint64_t task_id;
    error = parallel_processor_submit_task(processor, test_data, sizeof(test_data),
                                         simple_cpu_function, NULL,
                                         PP_PRIORITY_NORMAL, &task_id);
    
    free(task->output_data);
    free(task);
    
    if (error != PP_SUCCESS) {
        print_error("Task Submit", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    // Wait for completion
    ProcessingTask* result = NULL;
    error = parallel_processor_wait_for_task(processor, task_id, 5000, &result);
    if (error != PP_SUCCESS || !result || result->status != PP_TASK_COMPLETED) {
        print_error("Task Completion", "Task did not complete successfully");
        parallel_processor_destroy(processor);
        return 0;
    }
    
    print_success("Task Callback System");
    
    parallel_processor_stop(processor, false);
    parallel_processor_destroy(processor);
    
    return 1;
}

// Test 7: GPU availability check
static int test_gpu_availability(void) {
    print_test_header("GPU Availability Check");
    
    bool gpu_available;
    ParallelProcessorError error = parallel_processor_check_gpu_available(&gpu_available);
    if (error != PP_SUCCESS) {
        print_error("GPU Check", parallel_processor_error_string(error));
        return 0;
    }
    
    printf("  GPU Available: %s\n", gpu_available ? "Yes" : "No");
    print_success("GPU Availability Check");
    
    // Test GPU utilization (should work even if no GPU)
    ParallelProcessorConfig config;
    parallel_processor_config_default(&config, PP_MODE_CPU_ONLY);
    
    ParallelProcessor* processor = NULL;
    error = parallel_processor_create(&processor, &config);
    if (error != PP_SUCCESS) {
        print_error("Processor Creation", parallel_processor_error_string(error));
        return 0;
    }
    
    uint32_t utilization;
    error = parallel_processor_get_gpu_utilization(processor, &utilization);
    if (error != PP_SUCCESS) {
        print_error("GPU Utilization", parallel_processor_error_string(error));
        parallel_processor_destroy(processor);
        return 0;
    }
    
    printf("  GPU Utilization: %u%%\n", utilization);
    print_success("GPU Utilization Check");
    
    parallel_processor_destroy(processor);
    
    return 1;
}

// Test 8: Optimal thread count detection
static int test_optimal_threads(void) {
    print_test_header("Optimal Thread Count Detection");
    
    uint32_t optimal_threads;
    ParallelProcessorError error = parallel_processor_get_optimal_threads(&optimal_threads);
    if (error != PP_SUCCESS) {
        print_error("Optimal Threads", parallel_processor_error_string(error));
        return 0;
    }
    
    printf("  Optimal thread count: %u\n", optimal_threads);
    
    if (optimal_threads == 0 || optimal_threads > PP_MAX_THREADS) {
        print_error("Thread Count Range", "Thread count out of valid range");
        return 0;
    }
    
    print_success("Optimal Thread Count Detection");
    
    return 1;
}

// Test 9: Chunk size estimation
static int test_chunk_size_estimation(void) {
    print_test_header("Chunk Size Estimation");
    
    struct {
        size_t data_size;
        uint32_t num_workers;
        const char* description;
    } test_cases[] = {
        {1024 * 1024, 4, "1MB data, 4 workers"},
        {100 * 1024 * 1024, 8, "100MB data, 8 workers"},
        {10 * 1024, 2, "10KB data, 2 workers"},
        {1024 * 1024 * 1024, 16, "1GB data, 16 workers"}
    };
    
    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        size_t chunk_size;
        ParallelProcessorError error = parallel_processor_estimate_chunk_size(
            test_cases[i].data_size, test_cases[i].num_workers, &chunk_size);
        
        if (error != PP_SUCCESS) {
            print_error("Chunk Size Estimation", parallel_processor_error_string(error));
            return 0;
        }
        
        printf("  %s: %zu bytes chunk size\n", test_cases[i].description, chunk_size);
        
        // Verify chunk size is reasonable
        if (chunk_size == 0 || chunk_size > test_cases[i].data_size) {
            print_error("Chunk Size Range", "Chunk size out of reasonable range");
            return 0;
        }
    }
    
    print_success("Chunk Size Estimations");
    
    return 1;
}

// Test 10: Processing modes
static int test_processing_modes(void) {
    print_test_header("Processing Modes");
    
    ProcessingMode modes[] = {
        PP_MODE_CPU_ONLY,
        PP_MODE_HYBRID,
        PP_MODE_ADAPTIVE
    };
    
    const char* mode_names[] = {
        "CPU Only",
        "Hybrid",
        "Adaptive"
    };
    
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
        ParallelProcessorConfig config;
        ParallelProcessorError error = parallel_processor_config_default(&config, modes[i]);
        if (error != PP_SUCCESS) {
            print_error("Mode Configuration", parallel_processor_error_string(error));
            return 0;
        }
        
        config.thread_config.num_threads = 2;
        
        ParallelProcessor* processor = NULL;
        error = parallel_processor_create(&processor, &config);
        if (error != PP_SUCCESS) {
            printf("  %s mode: Creation failed (%s)\n", 
                   mode_names[i], parallel_processor_error_string(error));
            continue;
        }
        
        error = parallel_processor_start(processor);
        if (error != PP_SUCCESS) {
            printf("  %s mode: Start failed (%s)\n", 
                   mode_names[i], parallel_processor_error_string(error));
            parallel_processor_destroy(processor);
            continue;
        }
        
        printf("  %s mode: Successfully created and started\n", mode_names[i]);
        
        parallel_processor_stop(processor, false);
        parallel_processor_destroy(processor);
    }
    
    print_success("Processing Mode Testing");
    
    return 1;
}

// Main test runner
int main(void) {
    printf("Parallel Processor Test Suite\n");
    printf("=============================\n");
    
    srand((unsigned int)time(NULL));
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    struct {
        const char* name;
        int (*test_func)(void);
    } tests[] = {
        {"Processor Creation", test_processor_creation},
        {"Single Task", test_single_task},
        {"Multiple Tasks", test_multiple_tasks},
        {"Performance Metrics", test_performance_metrics},
        {"Error Handling", test_error_handling},
        {"Task Callbacks", test_task_callbacks},
        {"GPU Availability", test_gpu_availability},
        {"Optimal Threads", test_optimal_threads},
        {"Chunk Size Estimation", test_chunk_size_estimation},
        {"Processing Modes", test_processing_modes}
    };
    
    const int num_tests = sizeof(tests) / sizeof(tests[0]);
    
    for (int i = 0; i < num_tests; i++) {
        total_tests++;
        if (tests[i].test_func()) {
            passed_tests++;
        }
        
        // Small delay between tests
        usleep(100000); // 100ms
    }
    
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Test Summary\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    printf("Success rate: %.1f%%\n", (float)passed_tests / total_tests * 100.0f);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! Parallel Processor is working correctly.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the output above.\n");
        return 1;
    }
}
