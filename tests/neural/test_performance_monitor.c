#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "../../src/neural/monitoring/performance_monitor.h"

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

// Custom metric function for testing
static PerformanceMonitorError test_custom_metric(void* context, double* value) {
    if (!value) return PM_ERROR_INVALID_PARAM;
    
    // Simple test metric: random value between 0-100
    *value = (rand() % 10000) / 100.0;
    return PM_SUCCESS;
}

// Alert callback for testing
static int alert_callback_count = 0;
static void test_alert_callback(const PerformanceAlert* alert, void* user_data) {
    alert_callback_count++;
    printf("  Alert received: %s (severity: %s, value: %.2f)\n",
           alert->message,
           performance_monitor_alert_severity_name(alert->severity),
           alert->current_value);
}

// Optimization callback for testing
static int optimization_callback_count = 0;
static PerformanceMonitorError test_optimization_callback(OptimizationStrategy strategy,
                                                         double current_value,
                                                         double target_value,
                                                         void* context) {
    optimization_callback_count++;
    printf("  Optimization triggered: %s (current: %.2f, target: %.2f)\n",
           performance_monitor_optimization_strategy_name(strategy),
           current_value, target_value);
    return PM_SUCCESS;
}

// Test 1: Basic monitor creation and destruction
static int test_monitor_creation(void) {
    print_test_header("Monitor Creation and Destruction");
    
    MonitoringConfig config;
    PerformanceMonitorError error = performance_monitor_config_default(&config, PM_MODE_ACTIVE);
    if (error != PM_SUCCESS) {
        print_error("Default Config", performance_monitor_error_string(error));
        return 0;
    }
    
    print_success("Default Configuration");
    
    PerformanceMonitor* monitor = NULL;
    error = performance_monitor_create(&monitor, &config);
    if (error != PM_SUCCESS) {
        print_error("Monitor Creation", performance_monitor_error_string(error));
        return 0;
    }
    
    if (!monitor) {
        print_error("Monitor Creation", "Monitor is NULL");
        return 0;
    }
    
    print_success("Monitor Created");
    
    // Test different monitoring modes
    const MonitoringMode modes[] = {
        PM_MODE_PASSIVE, PM_MODE_ACTIVE, PM_MODE_ADAPTIVE,
        PM_MODE_PREDICTIVE, PM_MODE_REAL_TIME
    };
    
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
        MonitoringConfig test_config;
        error = performance_monitor_config_default(&test_config, modes[i]);
        if (error != PM_SUCCESS) {
            print_error("Config Mode", "Failed to create config for mode");
            performance_monitor_destroy(monitor);
            return 0;
        }
        printf("  Mode %d configured: sampling=%dms, alerts=%s, optimization=%s\n",
               (int)modes[i], test_config.sampling_interval_ms,
               test_config.enable_alerts ? "enabled" : "disabled",
               test_config.enable_optimization ? "enabled" : "disabled");
    }
    
    print_success("Configuration Modes");
    
    // Destroy monitor
    performance_monitor_destroy(monitor);
    print_success("Monitor Destroyed");
    
    return 1;
}

// Test 2: Basic monitoring operations
static int test_basic_monitoring(void) {
    print_test_header("Basic Monitoring Operations");
    
    MonitoringConfig config;
    performance_monitor_config_default(&config, PM_MODE_PASSIVE);
    config.sampling_interval_ms = 100; // Fast sampling for testing
    
    PerformanceMonitor* monitor = NULL;
    PerformanceMonitorError error = performance_monitor_create(&monitor, &config);
    if (error != PM_SUCCESS) {
        print_error("Monitor Creation", performance_monitor_error_string(error));
        return 0;
    }
    
    // Start monitoring
    error = performance_monitor_start(monitor);
    if (error != PM_SUCCESS) {
        print_error("Monitor Start", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Monitor Started");
    
    // Record some test metrics
    for (int i = 0; i < 10; i++) {
        double cpu_value = 50.0 + (rand() % 20); // 50-70%
        double memory_value = 1024 * 1024 * 1024 + (rand() % (512 * 1024 * 1024)); // ~1-1.5GB
        double throughput_value = 100.0 + (rand() % 50); // 100-150 items/sec
        
        performance_monitor_record_metric(monitor, PM_METRIC_CPU_USAGE, cpu_value, 0);
        performance_monitor_record_metric(monitor, PM_METRIC_MEMORY_USAGE, memory_value, 0);
        performance_monitor_record_metric(monitor, PM_METRIC_THROUGHPUT, throughput_value, 0);
        
        usleep(50000); // 50ms delay
    }
    
    print_success("Metrics Recorded");
    
    // Get metric statistics
    MetricStatistics cpu_stats, memory_stats, throughput_stats;
    
    error = performance_monitor_get_metric_stats(monitor, PM_METRIC_CPU_USAGE, &cpu_stats);
    if (error != PM_SUCCESS) {
        print_error("CPU Stats", performance_monitor_error_string(error));
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    error = performance_monitor_get_metric_stats(monitor, PM_METRIC_MEMORY_USAGE, &memory_stats);
    if (error != PM_SUCCESS) {
        print_error("Memory Stats", performance_monitor_error_string(error));
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    error = performance_monitor_get_metric_stats(monitor, PM_METRIC_THROUGHPUT, &throughput_stats);
    if (error != PM_SUCCESS) {
        print_error("Throughput Stats", performance_monitor_error_string(error));
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    printf("  CPU Usage Stats:\n");
    printf("    Current: %.2f%%, Average: %.2f%%, Min: %.2f%%, Max: %.2f%%\n",
           cpu_stats.current, cpu_stats.average, cpu_stats.minimum, cpu_stats.maximum);
    printf("    Std Dev: %.2f, 95th percentile: %.2f, Samples: %u\n",
           cpu_stats.standard_deviation, cpu_stats.percentile_95, cpu_stats.sample_count);
    
    printf("  Memory Usage Stats:\n");
    printf("    Current: %.0f MB, Average: %.0f MB, Min: %.0f MB, Max: %.0f MB\n",
           memory_stats.current / (1024*1024), memory_stats.average / (1024*1024),
           memory_stats.minimum / (1024*1024), memory_stats.maximum / (1024*1024));
    
    printf("  Throughput Stats:\n");
    printf("    Current: %.2f items/s, Average: %.2f items/s, Min: %.2f items/s, Max: %.2f items/s\n",
           throughput_stats.current, throughput_stats.average, 
           throughput_stats.minimum, throughput_stats.maximum);
    
    if (cpu_stats.sample_count == 0) {
        print_error("Sample Count", "No samples recorded");
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Statistics Retrieved");
    
    // Stop monitoring
    error = performance_monitor_stop(monitor);
    if (error != PM_SUCCESS) {
        print_error("Monitor Stop", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Monitor Stopped");
    
    performance_monitor_destroy(monitor);
    
    return 1;
}

// Test 3: Custom metrics
static int test_custom_metrics(void) {
    print_test_header("Custom Metrics");
    
    MonitoringConfig config;
    performance_monitor_config_default(&config, PM_MODE_PASSIVE);
    
    PerformanceMonitor* monitor = NULL;
    PerformanceMonitorError error = performance_monitor_create(&monitor, &config);
    if (error != PM_SUCCESS) {
        print_error("Monitor Creation", performance_monitor_error_string(error));
        return 0;
    }
    
    // Add custom metric
    error = performance_monitor_add_custom_metric(monitor, "Test Metric", 
                                                 test_custom_metric, NULL);
    if (error != PM_SUCCESS) {
        print_error("Add Custom Metric", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Custom Metric Added");
    
    // Start monitoring to collect custom metrics
    error = performance_monitor_start(monitor);
    if (error != PM_SUCCESS) {
        print_error("Monitor Start", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    // Let it run for a short time to collect custom metrics
    sleep(1);
    
    // Get custom metric statistics
    MetricStatistics custom_stats;
    error = performance_monitor_get_metric_stats(monitor, PM_METRIC_CUSTOM, &custom_stats);
    if (error != PM_SUCCESS) {
        print_error("Custom Metric Stats", performance_monitor_error_string(error));
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    printf("  Custom Metric Stats:\n");
    printf("    Current: %.2f, Average: %.2f, Min: %.2f, Max: %.2f\n",
           custom_stats.current, custom_stats.average, 
           custom_stats.minimum, custom_stats.maximum);
    printf("    Samples: %u\n", custom_stats.sample_count);
    
    if (custom_stats.sample_count == 0) {
        print_error("Custom Samples", "No custom metric samples collected");
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Custom Metrics Collection");
    
    performance_monitor_stop(monitor);
    performance_monitor_destroy(monitor);
    
    return 1;
}

// Test 4: Alert system
static int test_alert_system(void) {
    print_test_header("Alert System");
    
    MonitoringConfig config;
    performance_monitor_config_default(&config, PM_MODE_ACTIVE);
    config.sampling_interval_ms = 100;
    config.alert_evaluation_interval_ms = 200;
    
    PerformanceMonitor* monitor = NULL;
    PerformanceMonitorError error = performance_monitor_create(&monitor, &config);
    if (error != PM_SUCCESS) {
        print_error("Monitor Creation", performance_monitor_error_string(error));
        return 0;
    }
    
    // Set alert callback
    alert_callback_count = 0;
    error = performance_monitor_set_alert_callback(monitor, test_alert_callback, NULL);
    if (error != PM_SUCCESS) {
        print_error("Set Alert Callback", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Alert Callback Set");
    
    // Add alert rule (this is simplified - in the full implementation we'd have proper alert rules)
    error = performance_monitor_add_alert_rule(monitor, PM_METRIC_CPU_USAGE, 80.0, 
                                              PM_ALERT_WARNING, "High CPU usage");
    if (error != PM_SUCCESS) {
        print_error("Add Alert Rule", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Alert Rule Added");
    
    // Start monitoring
    error = performance_monitor_start(monitor);
    if (error != PM_SUCCESS) {
        print_error("Monitor Start", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    // Record high CPU usage to trigger alert
    for (int i = 0; i < 5; i++) {
        performance_monitor_record_metric(monitor, PM_METRIC_CPU_USAGE, 95.0, 0);
        usleep(100000); // 100ms
    }
    
    // Wait for alert evaluation
    sleep(1);
    
    printf("  Alert callbacks received: %d\n", alert_callback_count);
    
    print_success("Alert System Testing");
    
    performance_monitor_stop(monitor);
    performance_monitor_destroy(monitor);
    
    return 1;
}

// Test 5: Optimization system
static int test_optimization_system(void) {
    print_test_header("Optimization System");
    
    MonitoringConfig config;
    performance_monitor_config_default(&config, PM_MODE_ADAPTIVE);
    config.sampling_interval_ms = 100;
    config.optimization_interval_ms = 300;
    
    PerformanceMonitor* monitor = NULL;
    PerformanceMonitorError error = performance_monitor_create(&monitor, &config);
    if (error != PM_SUCCESS) {
        print_error("Monitor Creation", performance_monitor_error_string(error));
        return 0;
    }
    
    // Set optimization callback
    optimization_callback_count = 0;
    error = performance_monitor_set_optimization_callback(monitor, test_optimization_callback, NULL);
    if (error != PM_SUCCESS) {
        print_error("Set Optimization Callback", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Optimization Callback Set");
    
    // Add optimization rule
    OptimizationRule rule;
    rule.trigger_metric = PM_METRIC_CPU_USAGE;
    rule.threshold_min = 20.0;
    rule.threshold_max = 80.0;
    rule.strategy = PM_OPTIMIZE_THREAD_COUNT;
    rule.adjustment_factor = 1.2;
    rule.cooldown_seconds = 1;
    rule.last_applied = 0;
    rule.enabled = true;
    snprintf(rule.description, sizeof(rule.description), "CPU usage optimization");
    
    error = performance_monitor_add_optimization_rule(monitor, &rule);
    if (error != PM_SUCCESS) {
        print_error("Add Optimization Rule", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Optimization Rule Added");
    
    // Start monitoring
    error = performance_monitor_start(monitor);
    if (error != PM_SUCCESS) {
        print_error("Monitor Start", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    // Record CPU usage that should trigger optimization
    for (int i = 0; i < 3; i++) {
        performance_monitor_record_metric(monitor, PM_METRIC_CPU_USAGE, 90.0, 0);
        usleep(150000); // 150ms
    }
    
    // Wait for optimization
    sleep(1);
    
    printf("  Optimization callbacks received: %d\n", optimization_callback_count);
    
    print_success("Optimization System Testing");
    
    performance_monitor_stop(monitor);
    performance_monitor_destroy(monitor);
    
    return 1;
}

// Test 6: Profiler system
static int test_profiler_system(void) {
    print_test_header("Profiler System");
    
    MonitoringConfig config;
    performance_monitor_config_default(&config, PM_MODE_PREDICTIVE);
    
    PerformanceMonitor* monitor = NULL;
    PerformanceMonitorError error = performance_monitor_create(&monitor, &config);
    if (error != PM_SUCCESS) {
        print_error("Monitor Creation", performance_monitor_error_string(error));
        return 0;
    }
    
    // Start monitoring
    error = performance_monitor_start(monitor);
    if (error != PM_SUCCESS) {
        print_error("Monitor Start", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    // Record profiler events
    const char* function_names[] = {
        "test_function_1", "test_function_2", "test_function_3"
    };
    
    for (int i = 0; i < 10; i++) {
        const char* func_name = function_names[i % 3];
        
        // Function enter
        error = performance_monitor_record_event(monitor, PM_EVENT_FUNCTION_ENTER, 
                                                func_name, NULL);
        if (error != PM_SUCCESS) {
            print_error("Record Event Enter", performance_monitor_error_string(error));
            performance_monitor_stop(monitor);
            performance_monitor_destroy(monitor);
            return 0;
        }
        
        usleep(1000); // Simulate function execution
        
        // Function exit
        error = performance_monitor_record_event(monitor, PM_EVENT_FUNCTION_EXIT, 
                                                func_name, NULL);
        if (error != PM_SUCCESS) {
            print_error("Record Event Exit", performance_monitor_error_string(error));
            performance_monitor_stop(monitor);
            performance_monitor_destroy(monitor);
            return 0;
        }
    }
    
    print_success("Profiler Events Recorded");
    
    // Get profiler events
    ProfilerEvent events[100];
    uint32_t num_events;
    error = performance_monitor_get_profiler_events(monitor, events, 100, &num_events);
    if (error != PM_SUCCESS) {
        print_error("Get Profiler Events", performance_monitor_error_string(error));
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    printf("  Profiler events retrieved: %u\n", num_events);
    
    if (num_events == 0) {
        print_error("Profiler Events", "No profiler events found");
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    // Show first few events
    for (uint32_t i = 0; i < num_events && i < 5; i++) {
        const char* event_type = (events[i].type == PM_EVENT_FUNCTION_ENTER) ? "ENTER" : "EXIT";
        printf("    Event %u: %s %s (thread: %u)\n", 
               i, event_type, events[i].function_name ? events[i].function_name : "NULL",
               events[i].thread_id);
    }
    
    print_success("Profiler Events Retrieved");
    
    performance_monitor_stop(monitor);
    performance_monitor_destroy(monitor);
    
    return 1;
}

// Test 7: System metrics
static int test_system_metrics(void) {
    print_test_header("System Metrics");
    
    double cpu_usage;
    uint64_t memory_usage;
    double gpu_usage;
    
    PerformanceMonitorError error = performance_monitor_get_system_metrics(&cpu_usage, 
                                                                          &memory_usage, 
                                                                          &gpu_usage);
    if (error != PM_SUCCESS) {
        print_error("Get System Metrics", performance_monitor_error_string(error));
        return 0;
    }
    
    printf("  System Metrics:\n");
    printf("    CPU Usage: %.2f%%\n", cpu_usage);
    printf("    Memory Usage: %.2f MB\n", memory_usage / (1024.0 * 1024.0));
    printf("    GPU Usage: %.2f%%\n", gpu_usage);
    
    // Validate ranges
    if (cpu_usage < 0.0 || cpu_usage > 100.0) {
        print_error("CPU Usage Range", "CPU usage out of valid range");
        return 0;
    }
    
    if (memory_usage == 0) {
        print_error("Memory Usage", "Memory usage is zero");
        return 0;
    }
    
    print_success("System Metrics Retrieved");
    
    return 1;
}

// Test 8: Performance summary
static int test_performance_summary(void) {
    print_test_header("Performance Summary");
    
    MonitoringConfig config;
    performance_monitor_config_default(&config, PM_MODE_ACTIVE);
    config.sampling_interval_ms = 50;
    
    PerformanceMonitor* monitor = NULL;
    PerformanceMonitorError error = performance_monitor_create(&monitor, &config);
    if (error != PM_SUCCESS) {
        print_error("Monitor Creation", performance_monitor_error_string(error));
        return 0;
    }
    
    // Start monitoring
    error = performance_monitor_start(monitor);
    if (error != PM_SUCCESS) {
        print_error("Monitor Start", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    // Record metrics for a short time
    for (int i = 0; i < 20; i++) {
        performance_monitor_record_metric(monitor, PM_METRIC_CPU_USAGE, 60.0 + (i % 10), 0);
        performance_monitor_record_metric(monitor, PM_METRIC_THROUGHPUT, 100.0 + (i % 5), 0);
        usleep(25000); // 25ms
    }
    
    // Get performance summary
    PerformanceSummary summary;
    error = performance_monitor_get_summary(monitor, &summary);
    if (error != PM_SUCCESS) {
        print_error("Get Summary", performance_monitor_error_string(error));
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    printf("  Performance Summary:\n");
    printf("    Monitoring time: %.3f seconds\n", summary.total_monitoring_time / 1000000.0);
    printf("    Total samples: %u\n", summary.total_samples);
    printf("    Total alerts: %u\n", summary.total_alerts);
    printf("    Optimizations applied: %u\n", summary.optimizations_applied);
    printf("    Active alerts: %u\n", summary.active_alerts);
    
    if (summary.total_samples == 0) {
        print_error("Sample Count", "No samples in summary");
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Performance Summary Retrieved");
    
    performance_monitor_stop(monitor);
    performance_monitor_destroy(monitor);
    
    return 1;
}

// Test 9: Utility functions
static int test_utility_functions(void) {
    print_test_header("Utility Functions");
    
    // Test error string function
    const char* error_msg = performance_monitor_error_string(PM_ERROR_THRESHOLD_EXCEEDED);
    if (!error_msg || strlen(error_msg) == 0) {
        print_error("Error String", "Invalid error message");
        return 0;
    }
    printf("  Example error message: '%s'\n", error_msg);
    
    print_success("Error String Function");
    
    // Test metric names
    const MetricType test_metrics[] = {
        PM_METRIC_CPU_USAGE, PM_METRIC_MEMORY_USAGE, PM_METRIC_GPU_USAGE,
        PM_METRIC_THROUGHPUT, PM_METRIC_LATENCY, PM_METRIC_QUEUE_SIZE
    };
    
    for (size_t i = 0; i < sizeof(test_metrics) / sizeof(test_metrics[0]); i++) {
        const char* metric_name = performance_monitor_metric_name(test_metrics[i]);
        if (!metric_name || strlen(metric_name) == 0) {
            print_error("Metric Name", "Invalid metric name");
            return 0;
        }
        printf("  Metric %d: %s\n", (int)test_metrics[i], metric_name);
    }
    
    print_success("Metric Names");
    
    // Test alert severity names
    const AlertSeverity test_severities[] = {
        PM_ALERT_INFO, PM_ALERT_WARNING, PM_ALERT_CRITICAL, PM_ALERT_EMERGENCY
    };
    
    for (size_t i = 0; i < sizeof(test_severities) / sizeof(test_severities[0]); i++) {
        const char* severity_name = performance_monitor_alert_severity_name(test_severities[i]);
        if (!severity_name || strlen(severity_name) == 0) {
            print_error("Severity Name", "Invalid severity name");
            return 0;
        }
        printf("  Severity %d: %s\n", (int)test_severities[i], severity_name);
    }
    
    print_success("Alert Severity Names");
    
    // Test optimization strategy names
    const OptimizationStrategy test_strategies[] = {
        PM_OPTIMIZE_THREAD_COUNT, PM_OPTIMIZE_BATCH_SIZE, PM_OPTIMIZE_MEMORY_POOL,
        PM_OPTIMIZE_COMPRESSION_LEVEL, PM_OPTIMIZE_CACHE_SIZE
    };
    
    for (size_t i = 0; i < sizeof(test_strategies) / sizeof(test_strategies[0]); i++) {
        const char* strategy_name = performance_monitor_optimization_strategy_name(test_strategies[i]);
        if (!strategy_name || strlen(strategy_name) == 0) {
            print_error("Strategy Name", "Invalid strategy name");
            return 0;
        }
        printf("  Strategy %d: %s\n", (int)test_strategies[i], strategy_name);
    }
    
    print_success("Optimization Strategy Names");
    
    return 1;
}

// Test 10: Stress testing
static int test_stress_testing(void) {
    print_test_header("Stress Testing");
    
    MonitoringConfig config;
    performance_monitor_config_default(&config, PM_MODE_REAL_TIME);
    config.sampling_interval_ms = 10; // Very fast sampling
    
    PerformanceMonitor* monitor = NULL;
    PerformanceMonitorError error = performance_monitor_create(&monitor, &config);
    if (error != PM_SUCCESS) {
        print_error("Monitor Creation", performance_monitor_error_string(error));
        return 0;
    }
    
    // Start monitoring
    error = performance_monitor_start(monitor);
    if (error != PM_SUCCESS) {
        print_error("Monitor Start", performance_monitor_error_string(error));
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    // Rapid metric recording
    const int num_iterations = 1000;
    uint64_t start_time = time(NULL);
    
    for (int i = 0; i < num_iterations; i++) {
        double cpu_value = 50.0 + (rand() % 40);
        double memory_value = 1024 * 1024 * 1024 + (rand() % (1024 * 1024 * 1024));
        double throughput_value = 100.0 + (rand() % 100);
        
        performance_monitor_record_metric(monitor, PM_METRIC_CPU_USAGE, cpu_value, i % 8);
        performance_monitor_record_metric(monitor, PM_METRIC_MEMORY_USAGE, memory_value, i % 8);
        performance_monitor_record_metric(monitor, PM_METRIC_THROUGHPUT, throughput_value, i % 8);
        
        if (i % 100 == 0) {
            printf("  Recorded %d metrics...\n", i * 3);
        }
    }
    
    uint64_t end_time = time(NULL);
    double duration = end_time - start_time;
    
    printf("  Stress test completed in %.2f seconds\n", duration);
    printf("  Recorded %d metrics total\n", num_iterations * 3);
    
    if (duration > 0) {
        printf("  Metrics per second: %.2f\n", (num_iterations * 3) / duration);
    }
    
    // Verify that statistics are still working
    MetricStatistics cpu_stats;
    error = performance_monitor_get_metric_stats(monitor, PM_METRIC_CPU_USAGE, &cpu_stats);
    if (error != PM_SUCCESS) {
        print_error("Stats After Stress", performance_monitor_error_string(error));
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    printf("  Final CPU stats: avg=%.2f, samples=%u\n", 
           cpu_stats.average, cpu_stats.sample_count);
    
    if (cpu_stats.sample_count == 0) {
        print_error("Stress Test", "No samples recorded during stress test");
        performance_monitor_stop(monitor);
        performance_monitor_destroy(monitor);
        return 0;
    }
    
    print_success("Stress Testing");
    
    performance_monitor_stop(monitor);
    performance_monitor_destroy(monitor);
    
    return 1;
}

// Main test runner
int main(void) {
    printf("Performance Monitor Test Suite\n");
    printf("==============================\n");
    
    srand((unsigned int)time(NULL));
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    struct {
        const char* name;
        int (*test_func)(void);
    } tests[] = {
        {"Monitor Creation", test_monitor_creation},
        {"Basic Monitoring", test_basic_monitoring},
        {"Custom Metrics", test_custom_metrics},
        {"Alert System", test_alert_system},
        {"Optimization System", test_optimization_system},
        {"Profiler System", test_profiler_system},
        {"System Metrics", test_system_metrics},
        {"Performance Summary", test_performance_summary},
        {"Utility Functions", test_utility_functions},
        {"Stress Testing", test_stress_testing}
    };
    
    const int num_tests = sizeof(tests) / sizeof(tests[0]);
    
    for (int i = 0; i < num_tests; i++) {
        total_tests++;
        if (tests[i].test_func()) {
            passed_tests++;
        }
        
        // Small delay between tests
        usleep(200000); // 200ms
    }
    
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Test Summary\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    printf("Success rate: %.1f%%\n", (float)passed_tests / total_tests * 100.0f);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed! Performance Monitor is working correctly.\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Please review the output above.\n");
        return 1;
    }
}
