int test_comprehensive_benchmarks(void) {
    printf("Running Comprehensive Benchmark Suite...\n");
    
    // Setup contexts
    MetalContext* metal_ctx = NULL;
    MMManager* mm_manager = NULL;
    ComputeKernelContext* ck_ctx = NULL;
    BenchmarkContext* bench_ctx = NULL;
    
    if (metal_context_create(&metal_ctx) != METAL_SUCCESS) {
        printf("FAIL: Could not create Metal context for benchmarks\n");
        return 0;
    }
    
    if (mm_manager_create(&mm_manager, metal_ctx) != METAL_SUCCESS) {
        printf("FAIL: Could not create memory manager for benchmarks\n");
        metal_context_destroy(metal_ctx);
        return 0;
    }
    
    if (ck_context_create(&ck_ctx, metal_ctx, mm_manager) != METAL_SUCCESS) {
        printf("FAIL: Could not create compute kernel context for benchmarks\n");
        mm_manager_destroy(mm_manager);
        metal_context_destroy(metal_ctx);
        return 0;
    }
    
    if (benchmark_context_create(&bench_ctx, metal_ctx, mm_manager, ck_ctx) != METAL_SUCCESS) {
        printf("FAIL: Could not create benchmark context\n");
        ck_context_destroy(ck_ctx);
        mm_manager_destroy(mm_manager);
        metal_context_destroy(metal_ctx);
        return 0;
    }
    
    printf("PASS: Benchmark context created successfully\n");
    tests_passed++;
    tests_run++;
    
    // Configure benchmark for quick testing
    BenchmarkConfig config;
    benchmark_get_config(bench_ctx, &config);
    config.warmup_iterations = 2;        // Reduced for testing
    config.measurement_iterations = 3;   // Reduced for testing
    config.verbose_output = true;
    config.timeout_seconds = 10.0;
    benchmark_set_config(bench_ctx, &config);
    
    printf("PASS: Benchmark configuration set\n");
    tests_passed++;
    tests_run++;
    
    // Test individual benchmark operations
    BenchmarkResult result;
    
    // Test matrix multiplication benchmark
    if (benchmark_matrix_multiply(bench_ctx, BENCH_SIZE_SMALL, &result) == METAL_SUCCESS) {
        printf("PASS: Matrix multiplication benchmark (%.2f ms, %.2f GFLOPS)\n", 
               result.execution_time_ms, result.gflops);
        tests_passed++;
        tests_run++;
    } else {
        printf("FAIL: Matrix multiplication benchmark\n");
        tests_run++;
    }
    
    // Test activation function benchmark
    if (benchmark_activation_functions(bench_ctx, BENCH_SIZE_SMALL, &result) == METAL_SUCCESS) {
        printf("PASS: Activation function benchmark (%.2f ms)\n", result.execution_time_ms);
        tests_passed++;
        tests_run++;
    } else {
        printf("FAIL: Activation function benchmark\n");
        tests_run++;
    }
    
    // Test memory operations benchmark
    if (benchmark_memory_operations(bench_ctx, BENCH_SIZE_SMALL, &result) == METAL_SUCCESS) {
        printf("PASS: Memory operations benchmark (%.2f ms, %.2f GB/s)\n", 
               result.execution_time_ms, result.memory_bandwidth_gbps);
        tests_passed++;
        tests_run++;
    } else {
        printf("FAIL: Memory operations benchmark\n");
        tests_run++;
    }
    
    // Cleanup
    benchmark_context_destroy(bench_ctx);
    ck_context_destroy(ck_ctx);
    mm_manager_destroy(mm_manager);
    metal_context_destroy(metal_ctx);
    
    return 1;
}