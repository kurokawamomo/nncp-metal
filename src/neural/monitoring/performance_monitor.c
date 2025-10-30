#include "performance_monitor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#endif

// Helper functions
static uint64_t get_current_time_us(void);
static void* monitoring_thread_main(void* arg);
static void* alert_thread_main(void* arg);
static void* optimization_thread_main(void* arg);
static void update_metric_statistics(PerformanceMonitor* monitor, MetricType type);
static void evaluate_alerts(PerformanceMonitor* monitor);
static void apply_optimization_rules(PerformanceMonitor* monitor);
static double calculate_standard_deviation(const MetricSample* samples, uint32_t count, double mean);
static double calculate_percentile(const MetricSample* samples, uint32_t count, double percentile);
static PerformanceMonitorError get_system_cpu_usage(double* cpu_usage);
static PerformanceMonitorError get_system_memory_usage(uint64_t* memory_usage);
static PerformanceMonitorError get_system_gpu_usage(double* gpu_usage);

PerformanceMonitorError performance_monitor_create(PerformanceMonitor** monitor,
                                                   const MonitoringConfig* config) {
    if (!monitor || !config) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    // Allocate monitor
    PerformanceMonitor* pm = (PerformanceMonitor*)calloc(1, sizeof(PerformanceMonitor));
    if (!pm) {
        return PM_ERROR_MEMORY_ALLOCATION;
    }
    
    // Copy configuration
    memcpy(&pm->config, config, sizeof(MonitoringConfig));
    
    // Initialize mutexes and locks
    if (pthread_mutex_init(&pm->monitor_mutex, NULL) != 0) {
        free(pm);
        return PM_ERROR_INVALID_PARAM;
    }
    
    if (pthread_rwlock_init(&pm->metrics_lock, NULL) != 0) {
        pthread_mutex_destroy(&pm->monitor_mutex);
        free(pm);
        return PM_ERROR_INVALID_PARAM;
    }
    
    if (pthread_mutex_init(&pm->alerts_mutex, NULL) != 0) {
        pthread_rwlock_destroy(&pm->metrics_lock);
        pthread_mutex_destroy(&pm->monitor_mutex);
        free(pm);
        return PM_ERROR_INVALID_PARAM;
    }
    
    if (pthread_mutex_init(&pm->profiler_mutex, NULL) != 0) {
        pthread_mutex_destroy(&pm->alerts_mutex);
        pthread_rwlock_destroy(&pm->metrics_lock);
        pthread_mutex_destroy(&pm->monitor_mutex);
        free(pm);
        return PM_ERROR_INVALID_PARAM;
    }
    
    // Initialize sample storage
    for (int i = 0; i < PM_MAX_METRICS; i++) {
        pm->sample_counts[i] = 0;
        pm->sample_indices[i] = 0;
        
        // Initialize statistics
        pm->statistics[i].current = 0.0;
        pm->statistics[i].average = 0.0;
        pm->statistics[i].minimum = INFINITY;
        pm->statistics[i].maximum = -INFINITY;
        pm->statistics[i].standard_deviation = 0.0;
        pm->statistics[i].percentile_95 = 0.0;
        pm->statistics[i].percentile_99 = 0.0;
        pm->statistics[i].sample_count = 0;
        pm->statistics[i].last_updated = 0;
    }
    
    // Initialize alerts
    pm->num_alerts = 0;
    pm->next_alert_id = 1;
    pm->alert_callback = NULL;
    pm->alert_callback_data = NULL;
    
    // Initialize optimization
    pm->num_optimization_rules = 0;
    pm->optimization_callback = NULL;
    pm->optimization_callback_data = NULL;
    
    // Initialize profiler
    pm->profiler_event_count = 0;
    pm->profiler_event_index = 0;
    pm->profiler_overflow = false;
    
    // Initialize custom metrics
    pm->num_custom_metrics = 0;
    
    // Initialize summary
    memset(&pm->summary, 0, sizeof(PerformanceSummary));
    
    pm->is_running = false;
    pm->is_initialized = true;
    pm->threads_should_stop = false;
    
    *monitor = pm;
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_start(PerformanceMonitor* monitor) {
    if (!monitor || !monitor->is_initialized) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    if (monitor->is_running) {
        return PM_ERROR_MONITOR_ALREADY_RUNNING;
    }
    
    pthread_mutex_lock(&monitor->monitor_mutex);
    
    monitor->start_time = get_current_time_us();
    monitor->summary.monitoring_start_time = monitor->start_time;
    monitor->is_running = true;
    monitor->threads_should_stop = false;
    
    // Start monitoring thread
    if (pthread_create(&monitor->monitoring_thread, NULL, monitoring_thread_main, monitor) != 0) {
        monitor->is_running = false;
        pthread_mutex_unlock(&monitor->monitor_mutex);
        return PM_ERROR_INVALID_PARAM;
    }
    
    // Start alert thread if alerts are enabled
    if (monitor->config.enable_alerts) {
        if (pthread_create(&monitor->alert_thread, NULL, alert_thread_main, monitor) != 0) {
            monitor->threads_should_stop = true;
            pthread_join(monitor->monitoring_thread, NULL);
            monitor->is_running = false;
            pthread_mutex_unlock(&monitor->monitor_mutex);
            return PM_ERROR_INVALID_PARAM;
        }
    }
    
    // Start optimization thread if optimization is enabled
    if (monitor->config.enable_optimization) {
        if (pthread_create(&monitor->optimization_thread, NULL, optimization_thread_main, monitor) != 0) {
            monitor->threads_should_stop = true;
            pthread_join(monitor->monitoring_thread, NULL);
            if (monitor->config.enable_alerts) {
                pthread_join(monitor->alert_thread, NULL);
            }
            monitor->is_running = false;
            pthread_mutex_unlock(&monitor->monitor_mutex);
            return PM_ERROR_INVALID_PARAM;
        }
    }
    
    pthread_mutex_unlock(&monitor->monitor_mutex);
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_stop(PerformanceMonitor* monitor) {
    if (!monitor || !monitor->is_running) {
        return PM_ERROR_MONITOR_NOT_RUNNING;
    }
    
    pthread_mutex_lock(&monitor->monitor_mutex);
    
    monitor->threads_should_stop = true;
    
    // Wait for threads to finish
    pthread_join(monitor->monitoring_thread, NULL);
    
    if (monitor->config.enable_alerts) {
        pthread_join(monitor->alert_thread, NULL);
    }
    
    if (monitor->config.enable_optimization) {
        pthread_join(monitor->optimization_thread, NULL);
    }
    
    monitor->is_running = false;
    
    // Update summary
    uint64_t end_time = get_current_time_us();
    monitor->summary.total_monitoring_time = end_time - monitor->start_time;
    
    pthread_mutex_unlock(&monitor->monitor_mutex);
    
    return PM_SUCCESS;
}

void performance_monitor_destroy(PerformanceMonitor* monitor) {
    if (!monitor) {
        return;
    }
    
    // Stop monitoring if still running
    if (monitor->is_running) {
        performance_monitor_stop(monitor);
    }
    
    // Destroy synchronization primitives
    pthread_mutex_destroy(&monitor->profiler_mutex);
    pthread_mutex_destroy(&monitor->alerts_mutex);
    pthread_rwlock_destroy(&monitor->metrics_lock);
    pthread_mutex_destroy(&monitor->monitor_mutex);
    
    free(monitor);
}

PerformanceMonitorError performance_monitor_record_metric(PerformanceMonitor* monitor,
                                                         MetricType type,
                                                         double value,
                                                         uint32_t worker_id) {
    if (!monitor || type >= PM_MAX_METRICS) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    if (!monitor->is_running) {
        return PM_ERROR_MONITOR_NOT_RUNNING;
    }
    
    pthread_rwlock_wrlock(&monitor->metrics_lock);
    
    uint32_t index = monitor->sample_indices[type];
    MetricSample* sample = &monitor->samples[type][index];
    
    sample->type = type;
    sample->value = value;
    sample->timestamp = get_current_time_us();
    sample->worker_id = worker_id;
    sample->is_valid = true;
    
    // Update indices and counts
    monitor->sample_indices[type] = (index + 1) % PM_MAX_SAMPLES;
    if (monitor->sample_counts[type] < PM_MAX_SAMPLES) {
        monitor->sample_counts[type]++;
    }
    
    // Update statistics
    update_metric_statistics(monitor, type);
    
    // Update summary
    monitor->summary.total_samples++;
    
    pthread_rwlock_unlock(&monitor->metrics_lock);
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_get_metric_stats(PerformanceMonitor* monitor,
                                                            MetricType type,
                                                            MetricStatistics* stats) {
    if (!monitor || type >= PM_MAX_METRICS || !stats) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    pthread_rwlock_rdlock(&monitor->metrics_lock);
    memcpy(stats, &monitor->statistics[type], sizeof(MetricStatistics));
    pthread_rwlock_unlock(&monitor->metrics_lock);
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_add_custom_metric(PerformanceMonitor* monitor,
                                                             const char* name,
                                                             CustomMetricFunction metric_function,
                                                             void* context) {
    if (!monitor || !name || !metric_function) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    if (monitor->num_custom_metrics >= PM_MAX_CUSTOM_MONITORS) {
        return PM_ERROR_RESOURCE_EXHAUSTED;
    }
    
    uint32_t index = monitor->num_custom_metrics;
    monitor->custom_metrics[index] = metric_function;
    monitor->custom_contexts[index] = context;
    strncpy(monitor->custom_names[index], name, sizeof(monitor->custom_names[index]) - 1);
    monitor->custom_names[index][sizeof(monitor->custom_names[index]) - 1] = '\0';
    
    monitor->num_custom_metrics++;
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_add_alert_rule(PerformanceMonitor* monitor,
                                                          MetricType metric_type,
                                                          double threshold,
                                                          AlertSeverity severity,
                                                          const char* message) {
    if (!monitor || metric_type >= PM_MAX_METRICS || !message) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    if (monitor->num_alerts >= PM_MAX_ALERTS) {
        return PM_ERROR_RESOURCE_EXHAUSTED;
    }
    
    // For simplicity, we'll store alert rules as inactive alerts that get activated when triggered
    // In a real implementation, we'd have a separate alert rules structure
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_set_alert_callback(PerformanceMonitor* monitor,
                                                              AlertCallback callback,
                                                              void* user_data) {
    if (!monitor) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    monitor->alert_callback = callback;
    monitor->alert_callback_data = user_data;
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_add_optimization_rule(PerformanceMonitor* monitor,
                                                                 const OptimizationRule* rule) {
    if (!monitor || !rule) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    if (monitor->num_optimization_rules >= PM_MAX_OPTIMIZATION_RULES) {
        return PM_ERROR_RESOURCE_EXHAUSTED;
    }
    
    memcpy(&monitor->optimization_rules[monitor->num_optimization_rules], rule, sizeof(OptimizationRule));
    monitor->num_optimization_rules++;
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_set_optimization_callback(PerformanceMonitor* monitor,
                                                                     OptimizationCallback callback,
                                                                     void* user_data) {
    if (!monitor) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    monitor->optimization_callback = callback;
    monitor->optimization_callback_data = user_data;
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_record_event(PerformanceMonitor* monitor,
                                                        ProfilerEventType type,
                                                        const char* function_name,
                                                        void* user_data) {
    if (!monitor || !monitor->config.enable_profiler) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&monitor->profiler_mutex);
    
    if (monitor->profiler_event_count >= PM_MAX_PROFILER_EVENTS) {
        monitor->profiler_overflow = true;
        pthread_mutex_unlock(&monitor->profiler_mutex);
        return PM_ERROR_RESOURCE_EXHAUSTED;
    }
    
    uint32_t index = monitor->profiler_event_index;
    ProfilerEvent* event = &monitor->profiler_events[index];
    
    event->type = type;
    event->timestamp = get_current_time_us();
    event->thread_id = (uint32_t)pthread_self();
    event->function_name = function_name;
    event->address = NULL;
    event->size = 0;
    event->duration_us = 0;
    event->user_data = user_data;
    
    monitor->profiler_event_index = (index + 1) % PM_MAX_PROFILER_EVENTS;
    if (monitor->profiler_event_count < PM_MAX_PROFILER_EVENTS) {
        monitor->profiler_event_count++;
    }
    
    pthread_mutex_unlock(&monitor->profiler_mutex);
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_config_default(MonitoringConfig* config,
                                                          MonitoringMode mode) {
    if (!config) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(MonitoringConfig));
    
    config->mode = mode;
    
    // Set default sampling intervals based on mode
    switch (mode) {
        case PM_MODE_PASSIVE:
            config->sampling_interval_ms = 5000;  // 5 seconds
            config->enable_alerts = false;
            config->enable_optimization = false;
            config->enable_profiler = false;
            config->enable_prediction = false;
            break;
            
        case PM_MODE_ACTIVE:
            config->sampling_interval_ms = 1000;  // 1 second
            config->enable_alerts = true;
            config->enable_optimization = false;
            config->enable_profiler = false;
            config->enable_prediction = false;
            break;
            
        case PM_MODE_ADAPTIVE:
            config->sampling_interval_ms = 500;   // 500ms
            config->enable_alerts = true;
            config->enable_optimization = true;
            config->enable_profiler = false;
            config->enable_prediction = false;
            break;
            
        case PM_MODE_PREDICTIVE:
            config->sampling_interval_ms = 250;   // 250ms
            config->enable_alerts = true;
            config->enable_optimization = true;
            config->enable_profiler = true;
            config->enable_prediction = true;
            break;
            
        case PM_MODE_REAL_TIME:
            config->sampling_interval_ms = 100;   // 100ms
            config->enable_alerts = true;
            config->enable_optimization = true;
            config->enable_profiler = true;
            config->enable_prediction = true;
            break;
    }
    
    config->max_samples_per_metric = PM_MAX_SAMPLES;
    
    // Alert configuration
    config->alert_evaluation_interval_ms = config->sampling_interval_ms * 2;
    config->alert_suppression_timeout_ms = 30000;  // 30 seconds
    config->enable_alert_aggregation = true;
    
    // Optimization configuration
    config->optimization_interval_ms = config->sampling_interval_ms * 10;
    config->optimization_sensitivity = 0.5;
    config->conservative_optimization = (mode != PM_MODE_REAL_TIME);
    
    // Profiler configuration
    config->enable_function_profiling = config->enable_profiler;
    config->enable_memory_profiling = config->enable_profiler;
    config->enable_gpu_profiling = config->enable_profiler;
    config->profiler_buffer_size = PM_MAX_PROFILER_EVENTS;
    
    return PM_SUCCESS;
}

PerformanceMonitorError performance_monitor_get_summary(PerformanceMonitor* monitor,
                                                       PerformanceSummary* summary) {
    if (!monitor || !summary) {
        return PM_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&monitor->monitor_mutex);
    memcpy(summary, &monitor->summary, sizeof(PerformanceSummary));
    
    // Update derived fields
    if (monitor->is_running) {
        summary->total_monitoring_time = get_current_time_us() - monitor->start_time;
    }
    
    // Count active alerts
    pthread_mutex_lock(&monitor->alerts_mutex);
    summary->active_alerts = 0;
    for (uint32_t i = 0; i < monitor->num_alerts; i++) {
        if (monitor->alerts[i].is_active) {
            summary->active_alerts++;
        }
    }
    pthread_mutex_unlock(&monitor->alerts_mutex);
    
    pthread_mutex_unlock(&monitor->monitor_mutex);
    
    return PM_SUCCESS;
}

// Helper function implementations

static uint64_t get_current_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

static void* monitoring_thread_main(void* arg) {
    PerformanceMonitor* monitor = (PerformanceMonitor*)arg;
    
    while (!monitor->threads_should_stop) {
        // Collect system metrics
        double cpu_usage, gpu_usage;
        uint64_t memory_usage;
        
        if (get_system_cpu_usage(&cpu_usage) == PM_SUCCESS) {
            performance_monitor_record_metric(monitor, PM_METRIC_CPU_USAGE, cpu_usage, 0);
        }
        
        if (get_system_memory_usage(&memory_usage) == PM_SUCCESS) {
            performance_monitor_record_metric(monitor, PM_METRIC_MEMORY_USAGE, memory_usage, 0);
        }
        
        if (get_system_gpu_usage(&gpu_usage) == PM_SUCCESS) {
            performance_monitor_record_metric(monitor, PM_METRIC_GPU_USAGE, gpu_usage, 0);
        }
        
        // Collect custom metrics
        for (uint32_t i = 0; i < monitor->num_custom_metrics; i++) {
            double value;
            if (monitor->custom_metrics[i](monitor->custom_contexts[i], &value) == PM_SUCCESS) {
                performance_monitor_record_metric(monitor, PM_METRIC_CUSTOM, value, i);
            }
        }
        
        // Sleep for sampling interval
        usleep(monitor->config.sampling_interval_ms * 1000);
    }
    
    return NULL;
}

static void* alert_thread_main(void* arg) {
    PerformanceMonitor* monitor = (PerformanceMonitor*)arg;
    
    while (!monitor->threads_should_stop) {
        evaluate_alerts(monitor);
        
        // Sleep for alert evaluation interval
        usleep(monitor->config.alert_evaluation_interval_ms * 1000);
    }
    
    return NULL;
}

static void* optimization_thread_main(void* arg) {
    PerformanceMonitor* monitor = (PerformanceMonitor*)arg;
    
    while (!monitor->threads_should_stop) {
        apply_optimization_rules(monitor);
        
        // Sleep for optimization interval
        usleep(monitor->config.optimization_interval_ms * 1000);
    }
    
    return NULL;
}

static void update_metric_statistics(PerformanceMonitor* monitor, MetricType type) {
    MetricStatistics* stats = &monitor->statistics[type];
    uint32_t count = monitor->sample_counts[type];
    
    if (count == 0) return;
    
    MetricSample* samples = monitor->samples[type];
    
    // Update current value
    uint32_t latest_index = (monitor->sample_indices[type] + PM_MAX_SAMPLES - 1) % PM_MAX_SAMPLES;
    stats->current = samples[latest_index].value;
    
    // Calculate statistics
    double sum = 0.0;
    double min_val = INFINITY;
    double max_val = -INFINITY;
    
    uint32_t valid_samples = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (samples[i].is_valid) {
            double value = samples[i].value;
            sum += value;
            if (value < min_val) min_val = value;
            if (value > max_val) max_val = value;
            valid_samples++;
        }
    }
    
    if (valid_samples > 0) {
        stats->average = sum / valid_samples;
        stats->minimum = min_val;
        stats->maximum = max_val;
        stats->sample_count = valid_samples;
        stats->last_updated = get_current_time_us();
        
        // Calculate standard deviation
        stats->standard_deviation = calculate_standard_deviation(samples, count, stats->average);
        
        // Calculate percentiles
        stats->percentile_95 = calculate_percentile(samples, count, 0.95);
        stats->percentile_99 = calculate_percentile(samples, count, 0.99);
    }
}

static void evaluate_alerts(PerformanceMonitor* monitor) {
    // Simple alert evaluation - in a real implementation this would be more sophisticated
    pthread_rwlock_rdlock(&monitor->metrics_lock);
    
    // Check CPU usage
    MetricStatistics* cpu_stats = &monitor->statistics[PM_METRIC_CPU_USAGE];
    if (cpu_stats->current > 90.0) {
        // High CPU usage alert
        pthread_mutex_lock(&monitor->alerts_mutex);
        if (monitor->num_alerts < PM_MAX_ALERTS) {
            PerformanceAlert* alert = &monitor->alerts[monitor->num_alerts];
            alert->alert_id = monitor->next_alert_id++;
            alert->metric_type = PM_METRIC_CPU_USAGE;
            alert->severity = PM_ALERT_CRITICAL;
            alert->threshold_value = 90.0;
            alert->current_value = cpu_stats->current;
            alert->timestamp = get_current_time_us();
            alert->occurrence_count = 1;
            snprintf(alert->message, sizeof(alert->message), 
                    "High CPU usage: %.1f%% (threshold: %.1f%%)", 
                    cpu_stats->current, 90.0);
            alert->is_active = true;
            alert->is_suppressed = false;
            
            monitor->num_alerts++;
            monitor->summary.total_alerts++;
            
            // Call alert callback if set
            if (monitor->alert_callback) {
                monitor->alert_callback(alert, monitor->alert_callback_data);
            }
        }
        pthread_mutex_unlock(&monitor->alerts_mutex);
    }
    
    pthread_rwlock_unlock(&monitor->metrics_lock);
}

static void apply_optimization_rules(PerformanceMonitor* monitor) {
    if (!monitor->optimization_callback) {
        return;
    }
    
    uint64_t current_time = get_current_time_us();
    
    for (uint32_t i = 0; i < monitor->num_optimization_rules; i++) {
        OptimizationRule* rule = &monitor->optimization_rules[i];
        
        if (!rule->enabled) continue;
        
        // Check cooldown
        if (current_time - rule->last_applied < rule->cooldown_seconds * 1000000ULL) {
            continue;
        }
        
        pthread_rwlock_rdlock(&monitor->metrics_lock);
        MetricStatistics* stats = &monitor->statistics[rule->trigger_metric];
        
        bool should_optimize = false;
        double target_value = 0.0;
        
        if (stats->current < rule->threshold_min) {
            should_optimize = true;
            target_value = rule->threshold_min;
        } else if (stats->current > rule->threshold_max) {
            should_optimize = true;
            target_value = rule->threshold_max;
        }
        
        pthread_rwlock_unlock(&monitor->metrics_lock);
        
        if (should_optimize) {
            PerformanceMonitorError error = monitor->optimization_callback(
                rule->strategy, stats->current, target_value, 
                monitor->optimization_callback_data);
            
            if (error == PM_SUCCESS) {
                rule->last_applied = current_time;
                monitor->summary.optimizations_applied++;
            }
        }
    }
}

static double calculate_standard_deviation(const MetricSample* samples, uint32_t count, double mean) {
    if (count <= 1) return 0.0;
    
    double sum_sq_diff = 0.0;
    uint32_t valid_count = 0;
    
    for (uint32_t i = 0; i < count; i++) {
        if (samples[i].is_valid) {
            double diff = samples[i].value - mean;
            sum_sq_diff += diff * diff;
            valid_count++;
        }
    }
    
    if (valid_count <= 1) return 0.0;
    
    return sqrt(sum_sq_diff / (valid_count - 1));
}

static double calculate_percentile(const MetricSample* samples, uint32_t count, double percentile) {
    if (count == 0) return 0.0;
    
    // Simple percentile calculation - collect valid values and sort
    double values[PM_MAX_SAMPLES];
    uint32_t valid_count = 0;
    
    for (uint32_t i = 0; i < count && i < PM_MAX_SAMPLES; i++) {
        if (samples[i].is_valid) {
            values[valid_count++] = samples[i].value;
        }
    }
    
    if (valid_count == 0) return 0.0;
    
    // Simple bubble sort for small arrays
    for (uint32_t i = 0; i < valid_count - 1; i++) {
        for (uint32_t j = 0; j < valid_count - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                double temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    
    uint32_t index = (uint32_t)(percentile * (valid_count - 1));
    if (index >= valid_count) index = valid_count - 1;
    
    return values[index];
}

static PerformanceMonitorError get_system_cpu_usage(double* cpu_usage) {
    if (!cpu_usage) return PM_ERROR_INVALID_PARAM;
    
#ifdef __APPLE__
    // macOS implementation using mach
    static uint64_t last_idle = 0, last_total = 0;
    
    mach_port_t host_port = mach_host_self();
    natural_t processor_count;
    processor_info_array_t info_array;
    mach_msg_type_number_t info_count;
    
    kern_return_t error = host_processor_info(host_port, PROCESSOR_CPU_LOAD_INFO,
                                            &processor_count, &info_array, &info_count);
    
    if (error != KERN_SUCCESS) {
        *cpu_usage = 0.0;
        return PM_ERROR_INVALID_PARAM;
    }
    
    processor_cpu_load_info_t cpu_load_info = (processor_cpu_load_info_t)info_array;
    
    uint64_t total_ticks = 0;
    uint64_t idle_ticks = 0;
    
    for (natural_t i = 0; i < processor_count; i++) {
        total_ticks += cpu_load_info[i].cpu_ticks[CPU_STATE_USER] +
                      cpu_load_info[i].cpu_ticks[CPU_STATE_SYSTEM] +
                      cpu_load_info[i].cpu_ticks[CPU_STATE_IDLE] +
                      cpu_load_info[i].cpu_ticks[CPU_STATE_NICE];
        idle_ticks += cpu_load_info[i].cpu_ticks[CPU_STATE_IDLE];
    }
    
    vm_deallocate(mach_task_self(), (vm_address_t)info_array, info_count * sizeof(*info_array));
    
    if (last_total != 0) {
        uint64_t total_diff = total_ticks - last_total;
        uint64_t idle_diff = idle_ticks - last_idle;
        
        if (total_diff > 0) {
            *cpu_usage = ((double)(total_diff - idle_diff) / total_diff) * 100.0;
        } else {
            *cpu_usage = 0.0;
        }
    } else {
        *cpu_usage = 0.0;
    }
    
    last_total = total_ticks;
    last_idle = idle_ticks;
    
    return PM_SUCCESS;
#else
    // Linux implementation would go here
    *cpu_usage = 50.0; // Placeholder
    return PM_SUCCESS;
#endif
}

static PerformanceMonitorError get_system_memory_usage(uint64_t* memory_usage) {
    if (!memory_usage) return PM_ERROR_INVALID_PARAM;
    
#ifdef __APPLE__
    mach_port_t host_port = mach_host_self();
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    
    kern_return_t error = host_statistics64(host_port, HOST_VM_INFO64,
                                          (host_info64_t)&vm_stat, &count);
    
    if (error != KERN_SUCCESS) {
        *memory_usage = 0;
        return PM_ERROR_INVALID_PARAM;
    }
    
    // Calculate used memory (total - free - inactive)
    uint64_t page_size = 4096; // 4KB pages on most systems
    uint64_t used_pages = vm_stat.free_count + vm_stat.active_count + 
                         vm_stat.wire_count + vm_stat.compressor_page_count;
    
    *memory_usage = used_pages * page_size;
    
    return PM_SUCCESS;
#else
    // Linux implementation would read from /proc/meminfo
    *memory_usage = 1024 * 1024 * 1024; // Placeholder: 1GB
    return PM_SUCCESS;
#endif
}

static PerformanceMonitorError get_system_gpu_usage(double* gpu_usage) {
    if (!gpu_usage) return PM_ERROR_INVALID_PARAM;
    
    // GPU usage is platform and driver specific
    // For macOS with Metal, we would query GPU statistics
    // For now, return a placeholder value
    *gpu_usage = 0.0;
    return PM_SUCCESS;
}

// Utility functions

const char* performance_monitor_error_string(PerformanceMonitorError error) {
    switch (error) {
        case PM_SUCCESS: return "Success";
        case PM_ERROR_INVALID_PARAM: return "Invalid parameter";
        case PM_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case PM_ERROR_MONITOR_NOT_RUNNING: return "Monitor not running";
        case PM_ERROR_MONITOR_ALREADY_RUNNING: return "Monitor already running";
        case PM_ERROR_THRESHOLD_EXCEEDED: return "Threshold exceeded";
        case PM_ERROR_OPTIMIZATION_FAILED: return "Optimization failed";
        case PM_ERROR_PROFILER_ERROR: return "Profiler error";
        case PM_ERROR_RESOURCE_EXHAUSTED: return "Resource exhausted";
        case PM_ERROR_CONFIGURATION_INVALID: return "Configuration invalid";
        case PM_ERROR_ALERT_SUPPRESSED: return "Alert suppressed";
        case PM_ERROR_METRIC_NOT_FOUND: return "Metric not found";
        default: return "Unknown error";
    }
}

const char* performance_monitor_metric_name(MetricType type) {
    switch (type) {
        case PM_METRIC_CPU_USAGE: return "CPU Usage";
        case PM_METRIC_MEMORY_USAGE: return "Memory Usage";
        case PM_METRIC_GPU_USAGE: return "GPU Usage";
        case PM_METRIC_GPU_MEMORY: return "GPU Memory";
        case PM_METRIC_THROUGHPUT: return "Throughput";
        case PM_METRIC_LATENCY: return "Latency";
        case PM_METRIC_QUEUE_SIZE: return "Queue Size";
        case PM_METRIC_ERROR_RATE: return "Error Rate";
        case PM_METRIC_CACHE_HIT_RATE: return "Cache Hit Rate";
        case PM_METRIC_NETWORK_BANDWIDTH: return "Network Bandwidth";
        case PM_METRIC_DISK_IO: return "Disk I/O";
        case PM_METRIC_TEMPERATURE: return "Temperature";
        case PM_METRIC_POWER_CONSUMPTION: return "Power Consumption";
        case PM_METRIC_COMPRESSION_RATIO: return "Compression Ratio";
        case PM_METRIC_QUALITY_SCORE: return "Quality Score";
        case PM_METRIC_CUSTOM: return "Custom Metric";
        default: return "Unknown Metric";
    }
}

const char* performance_monitor_alert_severity_name(AlertSeverity severity) {
    switch (severity) {
        case PM_ALERT_INFO: return "Info";
        case PM_ALERT_WARNING: return "Warning";
        case PM_ALERT_CRITICAL: return "Critical";
        case PM_ALERT_EMERGENCY: return "Emergency";
        default: return "Unknown";
    }
}

const char* performance_monitor_optimization_strategy_name(OptimizationStrategy strategy) {
    switch (strategy) {
        case PM_OPTIMIZE_NONE: return "None";
        case PM_OPTIMIZE_THREAD_COUNT: return "Thread Count";
        case PM_OPTIMIZE_BATCH_SIZE: return "Batch Size";
        case PM_OPTIMIZE_MEMORY_POOL: return "Memory Pool";
        case PM_OPTIMIZE_COMPRESSION_LEVEL: return "Compression Level";
        case PM_OPTIMIZE_CACHE_SIZE: return "Cache Size";
        case PM_OPTIMIZE_GPU_UTILIZATION: return "GPU Utilization";
        case PM_OPTIMIZE_POWER_PROFILE: return "Power Profile";
        case PM_OPTIMIZE_QUALITY_SETTINGS: return "Quality Settings";
        default: return "Unknown Strategy";
    }
}

PerformanceMonitorError performance_monitor_get_system_metrics(double* cpu_usage,
                                                              uint64_t* memory_usage,
                                                              double* gpu_usage) {
    PerformanceMonitorError error = PM_SUCCESS;
    
    if (cpu_usage) {
        PerformanceMonitorError cpu_error = get_system_cpu_usage(cpu_usage);
        if (cpu_error != PM_SUCCESS) error = cpu_error;
    }
    
    if (memory_usage) {
        PerformanceMonitorError mem_error = get_system_memory_usage(memory_usage);
        if (mem_error != PM_SUCCESS && error == PM_SUCCESS) error = mem_error;
    }
    
    if (gpu_usage) {
        PerformanceMonitorError gpu_error = get_system_gpu_usage(gpu_usage);
        if (gpu_error != PM_SUCCESS && error == PM_SUCCESS) error = gpu_error;
    }
    
    return error;
}
