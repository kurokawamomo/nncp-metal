#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum limits
#define PM_MAX_METRICS               64
#define PM_MAX_ALERTS               32
#define PM_MAX_SAMPLES              1000
#define PM_MAX_WORKERS              16
#define PM_MAX_OPTIMIZATION_RULES    32
#define PM_MAX_PROFILER_EVENTS      10000
#define PM_MAX_CUSTOM_MONITORS      8

// Error codes
typedef enum {
    PM_SUCCESS = 0,
    PM_ERROR_INVALID_PARAM,
    PM_ERROR_MEMORY_ALLOCATION,
    PM_ERROR_MONITOR_NOT_RUNNING,
    PM_ERROR_MONITOR_ALREADY_RUNNING,
    PM_ERROR_THRESHOLD_EXCEEDED,
    PM_ERROR_OPTIMIZATION_FAILED,
    PM_ERROR_PROFILER_ERROR,
    PM_ERROR_RESOURCE_EXHAUSTED,
    PM_ERROR_CONFIGURATION_INVALID,
    PM_ERROR_ALERT_SUPPRESSED,
    PM_ERROR_METRIC_NOT_FOUND
} PerformanceMonitorError;

// Monitoring modes
typedef enum {
    PM_MODE_PASSIVE = 0,           // Passive monitoring only
    PM_MODE_ACTIVE,                // Active monitoring with alerts
    PM_MODE_ADAPTIVE,              // Adaptive monitoring with optimization
    PM_MODE_PREDICTIVE,            // Predictive monitoring with forecasting
    PM_MODE_REAL_TIME             // Real-time monitoring with immediate response
} MonitoringMode;

// Metric types
typedef enum {
    PM_METRIC_CPU_USAGE = 0,       // CPU utilization percentage
    PM_METRIC_MEMORY_USAGE,        // Memory usage in bytes
    PM_METRIC_GPU_USAGE,           // GPU utilization percentage
    PM_METRIC_GPU_MEMORY,          // GPU memory usage in bytes
    PM_METRIC_THROUGHPUT,          // Processing throughput (items/sec)
    PM_METRIC_LATENCY,             // Processing latency (milliseconds)
    PM_METRIC_QUEUE_SIZE,          // Task queue size
    PM_METRIC_ERROR_RATE,          // Error rate (errors/sec)
    PM_METRIC_CACHE_HIT_RATE,      // Cache hit rate percentage
    PM_METRIC_NETWORK_BANDWIDTH,   // Network bandwidth usage
    PM_METRIC_DISK_IO,             // Disk I/O operations per second
    PM_METRIC_TEMPERATURE,         // System temperature (Celsius)
    PM_METRIC_POWER_CONSUMPTION,   // Power consumption (watts)
    PM_METRIC_COMPRESSION_RATIO,   // Compression efficiency
    PM_METRIC_QUALITY_SCORE,       // Quality assessment score
    PM_METRIC_CUSTOM              // User-defined metric
} MetricType;

// Alert severity levels
typedef enum {
    PM_ALERT_INFO = 0,
    PM_ALERT_WARNING,
    PM_ALERT_CRITICAL,
    PM_ALERT_EMERGENCY
} AlertSeverity;

// Optimization strategies
typedef enum {
    PM_OPTIMIZE_NONE = 0,
    PM_OPTIMIZE_THREAD_COUNT,      // Adjust thread pool size
    PM_OPTIMIZE_BATCH_SIZE,        // Adjust batch processing size
    PM_OPTIMIZE_MEMORY_POOL,       // Adjust memory pool allocation
    PM_OPTIMIZE_COMPRESSION_LEVEL, // Adjust compression settings
    PM_OPTIMIZE_CACHE_SIZE,        // Adjust cache size
    PM_OPTIMIZE_GPU_UTILIZATION,   // Optimize GPU usage
    PM_OPTIMIZE_POWER_PROFILE,     // Adjust power consumption profile
    PM_OPTIMIZE_QUALITY_SETTINGS   // Adjust quality vs performance trade-off
} OptimizationStrategy;

// Profiler event types
typedef enum {
    PM_EVENT_FUNCTION_ENTER = 0,
    PM_EVENT_FUNCTION_EXIT,
    PM_EVENT_MEMORY_ALLOC,
    PM_EVENT_MEMORY_FREE,
    PM_EVENT_GPU_KERNEL_START,
    PM_EVENT_GPU_KERNEL_END,
    PM_EVENT_THREAD_CREATE,
    PM_EVENT_THREAD_DESTROY,
    PM_EVENT_CUSTOM
} ProfilerEventType;

// Forward declarations
typedef struct PerformanceMonitor PerformanceMonitor;
typedef struct MetricSample MetricSample;
typedef struct PerformanceAlert PerformanceAlert;
typedef struct OptimizationRule OptimizationRule;
typedef struct ProfilerEvent ProfilerEvent;
typedef struct MonitoringConfig MonitoringConfig;

// Custom metric function
typedef PerformanceMonitorError (*CustomMetricFunction)(void* context, double* value);

// Alert callback function
typedef void (*AlertCallback)(const PerformanceAlert* alert, void* user_data);

// Optimization callback function
typedef PerformanceMonitorError (*OptimizationCallback)(OptimizationStrategy strategy, 
                                                        double current_value,
                                                        double target_value,
                                                        void* context);

// Metric sample structure
struct MetricSample {
    MetricType type;               // Metric type
    double value;                  // Metric value
    uint64_t timestamp;            // Sample timestamp (microseconds)
    uint32_t worker_id;            // Worker/thread ID (if applicable)
    bool is_valid;                 // Whether sample is valid
};

// Performance alert structure
struct PerformanceAlert {
    uint32_t alert_id;             // Unique alert ID
    MetricType metric_type;        // Related metric
    AlertSeverity severity;        // Alert severity
    double threshold_value;        // Threshold that was exceeded
    double current_value;          // Current metric value
    uint64_t timestamp;            // Alert timestamp
    uint32_t occurrence_count;     // Number of occurrences
    char message[256];             // Alert message
    bool is_active;                // Whether alert is currently active
    bool is_suppressed;            // Whether alert is suppressed
};

// Optimization rule structure
struct OptimizationRule {
    MetricType trigger_metric;     // Metric that triggers optimization
    double threshold_min;          // Minimum threshold
    double threshold_max;          // Maximum threshold
    OptimizationStrategy strategy; // Optimization strategy to apply
    double adjustment_factor;      // How much to adjust (multiplier)
    uint32_t cooldown_seconds;     // Cooldown period between optimizations
    uint64_t last_applied;         // Last time this rule was applied
    bool enabled;                  // Whether rule is active
    char description[256];         // Rule description
};

// Profiler event structure
struct ProfilerEvent {
    ProfilerEventType type;        // Event type
    uint64_t timestamp;            // Event timestamp
    uint32_t thread_id;            // Thread ID
    const char* function_name;     // Function name (for function events)
    void* address;                 // Memory address (for memory events)
    size_t size;                   // Size (for memory events)
    uint64_t duration_us;          // Duration in microseconds (for timed events)
    void* user_data;               // Custom user data
};

// Metric statistics
typedef struct {
    double current;                // Current value
    double average;                // Average value
    double minimum;                // Minimum value
    double maximum;                // Maximum value
    double standard_deviation;     // Standard deviation
    double percentile_95;          // 95th percentile
    double percentile_99;          // 99th percentile
    uint32_t sample_count;         // Number of samples
    uint64_t last_updated;         // Last update timestamp
} MetricStatistics;

// Performance summary
typedef struct {
    uint64_t monitoring_start_time;    // When monitoring started
    uint64_t total_monitoring_time;    // Total monitoring time
    uint32_t total_samples;            // Total samples collected
    uint32_t total_alerts;             // Total alerts generated
    uint32_t optimizations_applied;    // Number of optimizations applied
    double average_cpu_usage;          // Average CPU usage
    double average_memory_usage;       // Average memory usage
    double average_throughput;         // Average throughput
    double average_latency;            // Average latency
    uint32_t active_alerts;            // Currently active alerts
} PerformanceSummary;

// Monitoring configuration
struct MonitoringConfig {
    MonitoringMode mode;               // Monitoring mode
    uint32_t sampling_interval_ms;     // Sampling interval in milliseconds
    uint32_t max_samples_per_metric;   // Maximum samples to keep per metric
    bool enable_alerts;                // Enable alert system
    bool enable_optimization;          // Enable auto-optimization
    bool enable_profiler;              // Enable built-in profiler
    bool enable_prediction;            // Enable predictive analytics
    
    // Alert configuration
    uint32_t alert_evaluation_interval_ms;  // How often to evaluate alerts
    uint32_t alert_suppression_timeout_ms;  // Alert suppression timeout
    bool enable_alert_aggregation;     // Aggregate similar alerts
    
    // Optimization configuration
    uint32_t optimization_interval_ms;     // How often to run optimization
    double optimization_sensitivity;       // Optimization sensitivity (0-1)
    bool conservative_optimization;        // Use conservative optimization
    
    // Profiler configuration
    bool enable_function_profiling;       // Profile function calls
    bool enable_memory_profiling;         // Profile memory allocation
    bool enable_gpu_profiling;            // Profile GPU operations
    uint32_t profiler_buffer_size;        // Profiler event buffer size
    
    void* user_context;               // User context for callbacks
};

// Main performance monitor structure
struct PerformanceMonitor {
    MonitoringConfig config;           // Configuration
    bool is_running;                   // Whether monitoring is active
    bool is_initialized;               // Initialization flag
    
    // Metrics storage
    MetricSample samples[PM_MAX_METRICS][PM_MAX_SAMPLES];  // Sample storage
    uint32_t sample_counts[PM_MAX_METRICS];                // Current sample counts
    uint32_t sample_indices[PM_MAX_METRICS];               // Current sample indices
    MetricStatistics statistics[PM_MAX_METRICS];           // Metric statistics
    
    // Custom metrics
    CustomMetricFunction custom_metrics[PM_MAX_CUSTOM_MONITORS];
    void* custom_contexts[PM_MAX_CUSTOM_MONITORS];
    char custom_names[PM_MAX_CUSTOM_MONITORS][64];
    uint32_t num_custom_metrics;
    
    // Alerts
    PerformanceAlert alerts[PM_MAX_ALERTS];               // Active alerts
    uint32_t num_alerts;                                  // Number of active alerts
    uint32_t next_alert_id;                               // Next alert ID
    AlertCallback alert_callback;                         // Alert callback
    void* alert_callback_data;                            // Alert callback data
    
    // Optimization
    OptimizationRule optimization_rules[PM_MAX_OPTIMIZATION_RULES];
    uint32_t num_optimization_rules;                      // Number of optimization rules
    OptimizationCallback optimization_callback;           // Optimization callback
    void* optimization_callback_data;                     // Optimization callback data
    
    // Profiler
    ProfilerEvent profiler_events[PM_MAX_PROFILER_EVENTS];  // Profiler events
    uint32_t profiler_event_count;                        // Current event count
    uint32_t profiler_event_index;                        // Current event index
    bool profiler_overflow;                               // Whether buffer overflowed
    
    // Worker threads
    pthread_t monitoring_thread;                          // Monitoring thread
    pthread_t alert_thread;                               // Alert evaluation thread
    pthread_t optimization_thread;                        // Optimization thread
    bool threads_should_stop;                             // Thread stop flag
    
    // Statistics
    PerformanceSummary summary;                           // Performance summary
    uint64_t start_time;                                  // Monitoring start time
    
    // Thread safety
    pthread_mutex_t monitor_mutex;                        // General mutex
    pthread_rwlock_t metrics_lock;                        // Metrics read-write lock
    pthread_mutex_t alerts_mutex;                         // Alerts mutex
    pthread_mutex_t profiler_mutex;                       // Profiler mutex
};

// Core API Functions

/**
 * Create and initialize performance monitor
 * @param monitor Pointer to store created monitor
 * @param config Monitoring configuration
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_create(PerformanceMonitor** monitor,
                                                   const MonitoringConfig* config);

/**
 * Start performance monitoring
 * @param monitor Performance monitor instance
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_start(PerformanceMonitor* monitor);

/**
 * Stop performance monitoring
 * @param monitor Performance monitor instance
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_stop(PerformanceMonitor* monitor);

/**
 * Destroy performance monitor and free resources
 * @param monitor Monitor to destroy
 */
void performance_monitor_destroy(PerformanceMonitor* monitor);

// Metric Collection Functions

/**
 * Record a metric sample
 * @param monitor Performance monitor instance
 * @param type Metric type
 * @param value Metric value
 * @param worker_id Worker ID (optional, use 0 if not applicable)
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_record_metric(PerformanceMonitor* monitor,
                                                         MetricType type,
                                                         double value,
                                                         uint32_t worker_id);

/**
 * Get current metric statistics
 * @param monitor Performance monitor instance
 * @param type Metric type
 * @param stats Output statistics
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_get_metric_stats(PerformanceMonitor* monitor,
                                                            MetricType type,
                                                            MetricStatistics* stats);

/**
 * Get recent metric samples
 * @param monitor Performance monitor instance
 * @param type Metric type
 * @param samples Output samples array
 * @param max_samples Maximum number of samples to return
 * @param num_samples Output number of samples returned
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_get_metric_samples(PerformanceMonitor* monitor,
                                                              MetricType type,
                                                              MetricSample* samples,
                                                              uint32_t max_samples,
                                                              uint32_t* num_samples);

/**
 * Add custom metric
 * @param monitor Performance monitor instance
 * @param name Metric name
 * @param metric_function Custom metric function
 * @param context Custom context
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_add_custom_metric(PerformanceMonitor* monitor,
                                                             const char* name,
                                                             CustomMetricFunction metric_function,
                                                             void* context);

// Alert System Functions

/**
 * Add alert rule
 * @param monitor Performance monitor instance
 * @param metric_type Metric to monitor
 * @param threshold Threshold value
 * @param severity Alert severity
 * @param message Alert message
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_add_alert_rule(PerformanceMonitor* monitor,
                                                          MetricType metric_type,
                                                          double threshold,
                                                          AlertSeverity severity,
                                                          const char* message);

/**
 * Set alert callback
 * @param monitor Performance monitor instance
 * @param callback Alert callback function
 * @param user_data User data for callback
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_set_alert_callback(PerformanceMonitor* monitor,
                                                              AlertCallback callback,
                                                              void* user_data);

/**
 * Get active alerts
 * @param monitor Performance monitor instance
 * @param alerts Output alerts array
 * @param max_alerts Maximum number of alerts to return
 * @param num_alerts Output number of alerts returned
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_get_active_alerts(PerformanceMonitor* monitor,
                                                             PerformanceAlert* alerts,
                                                             uint32_t max_alerts,
                                                             uint32_t* num_alerts);

/**
 * Suppress alert
 * @param monitor Performance monitor instance
 * @param alert_id Alert ID to suppress
 * @param suppress_duration_ms Suppression duration in milliseconds
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_suppress_alert(PerformanceMonitor* monitor,
                                                          uint32_t alert_id,
                                                          uint32_t suppress_duration_ms);

// Optimization Functions

/**
 * Add optimization rule
 * @param monitor Performance monitor instance
 * @param rule Optimization rule
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_add_optimization_rule(PerformanceMonitor* monitor,
                                                                 const OptimizationRule* rule);

/**
 * Set optimization callback
 * @param monitor Performance monitor instance
 * @param callback Optimization callback function
 * @param user_data User data for callback
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_set_optimization_callback(PerformanceMonitor* monitor,
                                                                     OptimizationCallback callback,
                                                                     void* user_data);

/**
 * Trigger manual optimization
 * @param monitor Performance monitor instance
 * @param strategy Optimization strategy to apply
 * @param target_value Target value for optimization
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_trigger_optimization(PerformanceMonitor* monitor,
                                                                OptimizationStrategy strategy,
                                                                double target_value);

// Profiler Functions

/**
 * Record profiler event
 * @param monitor Performance monitor instance
 * @param type Event type
 * @param function_name Function name (for function events)
 * @param user_data Custom user data
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_record_event(PerformanceMonitor* monitor,
                                                        ProfilerEventType type,
                                                        const char* function_name,
                                                        void* user_data);

/**
 * Get profiler events
 * @param monitor Performance monitor instance
 * @param events Output events array
 * @param max_events Maximum number of events to return
 * @param num_events Output number of events returned
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_get_profiler_events(PerformanceMonitor* monitor,
                                                               ProfilerEvent* events,
                                                               uint32_t max_events,
                                                               uint32_t* num_events);

/**
 * Clear profiler events
 * @param monitor Performance monitor instance
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_clear_profiler_events(PerformanceMonitor* monitor);

// Configuration and Reporting Functions

/**
 * Create default monitoring configuration
 * @param config Output configuration
 * @param mode Monitoring mode
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_config_default(MonitoringConfig* config,
                                                          MonitoringMode mode);

/**
 * Update monitoring configuration
 * @param monitor Performance monitor instance
 * @param config New configuration
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_update_config(PerformanceMonitor* monitor,
                                                         const MonitoringConfig* config);

/**
 * Get performance summary
 * @param monitor Performance monitor instance
 * @param summary Output performance summary
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_get_summary(PerformanceMonitor* monitor,
                                                       PerformanceSummary* summary);

/**
 * Generate performance report
 * @param monitor Performance monitor instance
 * @param report_buffer Output report buffer
 * @param buffer_size Buffer size
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_generate_report(PerformanceMonitor* monitor,
                                                           char* report_buffer,
                                                           size_t buffer_size);

/**
 * Reset monitoring statistics
 * @param monitor Performance monitor instance
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_reset_statistics(PerformanceMonitor* monitor);

// Utility Functions

/**
 * Get error string for error code
 * @param error Error code
 * @return Human-readable error string
 */
const char* performance_monitor_error_string(PerformanceMonitorError error);

/**
 * Get metric type name
 * @param type Metric type
 * @return Human-readable metric type name
 */
const char* performance_monitor_metric_name(MetricType type);

/**
 * Get alert severity name
 * @param severity Alert severity
 * @return Human-readable severity name
 */
const char* performance_monitor_alert_severity_name(AlertSeverity severity);

/**
 * Get optimization strategy name
 * @param strategy Optimization strategy
 * @return Human-readable strategy name
 */
const char* performance_monitor_optimization_strategy_name(OptimizationStrategy strategy);

/**
 * Get current system metrics
 * @param cpu_usage Output CPU usage percentage
 * @param memory_usage Output memory usage in bytes
 * @param gpu_usage Output GPU usage percentage (if available)
 * @return PM_SUCCESS on success, error code on failure
 */
PerformanceMonitorError performance_monitor_get_system_metrics(double* cpu_usage,
                                                              uint64_t* memory_usage,
                                                              double* gpu_usage);

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_MONITOR_H
