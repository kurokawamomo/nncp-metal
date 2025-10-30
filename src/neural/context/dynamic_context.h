/**
 * @file dynamic_context.h
 * @brief Dynamic Context Management for Neural Compression
 * 
 * This component implements adaptive context window management that adjusts
 * attention and memory window sizes based on content characteristics and
 * compression effectiveness for both Transformer and LSTM algorithms.
 */

#ifndef DYNAMIC_CONTEXT_H
#define DYNAMIC_CONTEXT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants and limits
#define DYNAMIC_CONTEXT_MIN_WINDOW_SIZE     1024    // 1KB minimum
#define DYNAMIC_CONTEXT_MAX_WINDOW_SIZE     65536   // 64KB maximum
#define DYNAMIC_CONTEXT_DEFAULT_WINDOW_SIZE 16384   // 16KB default
#define DYNAMIC_CONTEXT_MIN_MEMORY_SIZE     512     // 512 bytes minimum
#define DYNAMIC_CONTEXT_MAX_MEMORY_SIZE     8192    // 8KB maximum
#define DYNAMIC_CONTEXT_DEFAULT_MEMORY_SIZE 2048    // 2KB default
#define DYNAMIC_CONTEXT_ADAPTATION_HISTORY  32      // History for adaptation decisions
#define DYNAMIC_CONTEXT_MAX_CONTEXTS        8       // Maximum concurrent contexts

// Error codes
typedef enum {
    DYNAMIC_CONTEXT_SUCCESS = 0,
    DYNAMIC_CONTEXT_ERROR_INVALID_PARAM,
    DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION,
    DYNAMIC_CONTEXT_ERROR_BUFFER_OVERFLOW,
    DYNAMIC_CONTEXT_ERROR_INVALID_SIZE,
    DYNAMIC_CONTEXT_ERROR_ADAPTATION_FAILED,
    DYNAMIC_CONTEXT_ERROR_CONTEXT_NOT_FOUND,
    DYNAMIC_CONTEXT_ERROR_MEMORY_PRESSURE,
    DYNAMIC_CONTEXT_ERROR_SYSTEM_LIMIT,
    DYNAMIC_CONTEXT_ERROR_INVALID_STATE
} DynamicContextError;

// Context sizing strategies
typedef enum {
    DYNAMIC_CONTEXT_STRATEGY_CONSERVATIVE = 0,  // Conservative growth, prefer stability
    DYNAMIC_CONTEXT_STRATEGY_AGGRESSIVE,        // Aggressive adaptation, prefer efficiency
    DYNAMIC_CONTEXT_STRATEGY_BALANCED,          // Balanced approach (default)
    DYNAMIC_CONTEXT_STRATEGY_CONTENT_AWARE      // Content-driven adaptation
} ContextSizingStrategy;

// Content complexity levels
typedef enum {
    DYNAMIC_CONTEXT_COMPLEXITY_LOW = 0,     // Simple, repetitive content
    DYNAMIC_CONTEXT_COMPLEXITY_MEDIUM,      // Mixed content patterns
    DYNAMIC_CONTEXT_COMPLEXITY_HIGH,        // Complex, varied content
    DYNAMIC_CONTEXT_COMPLEXITY_VARIABLE     // Highly variable content
} ContentComplexity;

// Memory pressure levels
typedef enum {
    DYNAMIC_CONTEXT_MEMORY_PRESSURE_NONE = 0,   // Abundant memory available
    DYNAMIC_CONTEXT_MEMORY_PRESSURE_LOW,        // Some memory pressure
    DYNAMIC_CONTEXT_MEMORY_PRESSURE_MEDIUM,     // Moderate memory pressure
    DYNAMIC_CONTEXT_MEMORY_PRESSURE_HIGH        // High memory pressure
} MemoryPressureLevel;

// Performance metrics structure
typedef struct {
    float compression_ratio;            // Current compression ratio
    float compression_speed;            // Bytes per second
    float memory_efficiency;            // Memory usage efficiency (0-1)
    uint64_t processing_time_ns;        // Processing time in nanoseconds
    uint32_t prediction_accuracy;       // Prediction accuracy percentage
    uint32_t pattern_recognition_rate;  // Pattern recognition success rate
} DynamicContextMetrics;

// Adaptation history entry
typedef struct {
    uint32_t window_size;               // Window size used
    uint32_t memory_size;               // Memory size used
    DynamicContextMetrics metrics;      // Performance metrics achieved
    ContentComplexity complexity;       // Content complexity detected
    uint64_t timestamp;                 // Timestamp of measurement
    float effectiveness_score;          // Overall effectiveness score (0-1)
} AdaptationHistoryEntry;

// Dynamic context configuration
typedef struct {
    ContextSizingStrategy strategy;     // Sizing strategy to use
    uint32_t min_window_size;          // Minimum window size
    uint32_t max_window_size;          // Maximum window size
    uint32_t min_memory_size;          // Minimum memory size
    uint32_t max_memory_size;          // Maximum memory size
    float adaptation_threshold;         // Threshold for triggering adaptation
    float memory_pressure_threshold;   // Memory pressure threshold (0-1)
    uint32_t adaptation_interval;      // Adaptation interval in bytes processed
    bool enable_metal_acceleration;    // Enable Metal acceleration
    bool enable_predictive_sizing;     // Enable predictive sizing
    uint32_t history_window;           // History window for decisions
} DynamicContextConfig;

// Dynamic buffer structure
typedef struct {
    uint8_t* data;                     // Buffer data
    uint32_t capacity;                 // Buffer capacity
    uint32_t size;                     // Current data size
    uint32_t window_size;              // Current window size
    bool is_metal_backed;              // Metal buffer backing
    void* metal_buffer;                // Metal buffer reference
    pthread_mutex_t buffer_mutex;      // Buffer access mutex
} DynamicBuffer;

// Context effectiveness monitor
typedef struct {
    AdaptationHistoryEntry history[DYNAMIC_CONTEXT_ADAPTATION_HISTORY];
    uint32_t history_index;            // Current history index
    uint32_t history_count;            // Number of history entries
    float current_effectiveness;       // Current effectiveness score
    float baseline_effectiveness;      // Baseline effectiveness
    ContentComplexity detected_complexity; // Current content complexity
    MemoryPressureLevel memory_pressure;   // Current memory pressure
    uint64_t bytes_processed;          // Total bytes processed
    uint64_t last_adaptation_time;     // Last adaptation timestamp
} ContextEffectivenessMonitor;

// Main dynamic context manager structure
typedef struct {
    DynamicContextConfig config;       // Configuration
    bool is_initialized;               // Initialization flag
    uint32_t context_id;               // Unique context identifier
    
    // Dynamic buffers
    DynamicBuffer attention_buffer;    // Attention context buffer
    DynamicBuffer memory_buffer;       // Memory context buffer
    
    // Effectiveness monitoring
    ContextEffectivenessMonitor monitor; // Effectiveness monitor
    
    // Current state
    uint32_t current_window_size;      // Current attention window size
    uint32_t current_memory_size;      // Current memory size
    ContentComplexity current_complexity; // Current content complexity
    MemoryPressureLevel current_pressure; // Current memory pressure
    
    // Statistics
    uint64_t total_adaptations;        // Total adaptations performed
    uint64_t successful_adaptations;   // Successful adaptations
    uint64_t total_bytes_processed;    // Total bytes processed
    float average_effectiveness;       // Average effectiveness score
    
    // Thread safety
    pthread_mutex_t context_mutex;     // Context access mutex
    pthread_rwlock_t config_lock;      // Configuration read-write lock
} DynamicContextManager;

// Core API Functions

/**
 * Create and initialize dynamic context manager
 * @param manager Pointer to store created manager
 * @param config Dynamic context configuration
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_create(DynamicContextManager** manager,
                                          const DynamicContextConfig* config);

/**
 * Destroy dynamic context manager and free resources
 * @param manager Dynamic context manager to destroy
 */
void dynamic_context_destroy(DynamicContextManager* manager);

// Configuration Functions

/**
 * Create default dynamic context configuration
 * @param config Output configuration structure
 * @param strategy Sizing strategy to use
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_config_default(DynamicContextConfig* config,
                                                   ContextSizingStrategy strategy);

/**
 * Update dynamic context configuration
 * @param manager Dynamic context manager
 * @param config New configuration
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_update_config(DynamicContextManager* manager,
                                                  const DynamicContextConfig* config);

// Buffer Management Functions

/**
 * Get current attention buffer
 * @param manager Dynamic context manager
 * @param buffer Output buffer pointer
 * @param size Output buffer size
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_get_attention_buffer(DynamicContextManager* manager,
                                                        uint8_t** buffer,
                                                        uint32_t* size);

/**
 * Get current memory buffer
 * @param manager Dynamic context manager
 * @param buffer Output buffer pointer
 * @param size Output buffer size
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_get_memory_buffer(DynamicContextManager* manager,
                                                     uint8_t** buffer,
                                                     uint32_t* size);

/**
 * Resize attention buffer
 * @param manager Dynamic context manager
 * @param new_size New buffer size
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_resize_attention_buffer(DynamicContextManager* manager,
                                                           uint32_t new_size);

/**
 * Resize memory buffer
 * @param manager Dynamic context manager
 * @param new_size New buffer size
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_resize_memory_buffer(DynamicContextManager* manager,
                                                        uint32_t new_size);

// Adaptation Functions

/**
 * Update compression effectiveness metrics
 * @param manager Dynamic context manager
 * @param metrics Performance metrics
 * @param content_complexity Content complexity level
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_update_metrics(DynamicContextManager* manager,
                                                   const DynamicContextMetrics* metrics,
                                                   ContentComplexity content_complexity);

/**
 * Trigger adaptation based on current metrics
 * @param manager Dynamic context manager
 * @param force_adaptation Force adaptation regardless of thresholds
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_trigger_adaptation(DynamicContextManager* manager,
                                                       bool force_adaptation);

/**
 * Analyze content and recommend context sizes
 * @param manager Dynamic context manager
 * @param content Content data to analyze
 * @param content_size Size of content data
 * @param recommended_window_size Output recommended window size
 * @param recommended_memory_size Output recommended memory size
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_analyze_content(DynamicContextManager* manager,
                                                    const uint8_t* content,
                                                    uint32_t content_size,
                                                    uint32_t* recommended_window_size,
                                                    uint32_t* recommended_memory_size);

/**
 * Check and respond to memory pressure
 * @param manager Dynamic context manager
 * @param available_memory Available memory in bytes
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_handle_memory_pressure(DynamicContextManager* manager,
                                                           uint64_t available_memory);

// Monitoring Functions

/**
 * Get current effectiveness score
 * @param manager Dynamic context manager
 * @param effectiveness_score Output effectiveness score (0-1)
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_get_effectiveness(DynamicContextManager* manager,
                                                     float* effectiveness_score);

/**
 * Get adaptation history
 * @param manager Dynamic context manager
 * @param history Output history array
 * @param max_entries Maximum entries to return
 * @param num_entries Output number of entries returned
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_get_history(DynamicContextManager* manager,
                                                AdaptationHistoryEntry* history,
                                                uint32_t max_entries,
                                                uint32_t* num_entries);

/**
 * Get current context statistics
 * @param manager Dynamic context manager
 * @param total_adaptations Output total adaptations
 * @param success_rate Output adaptation success rate (0-1)
 * @param average_effectiveness Output average effectiveness
 * @param bytes_processed Output total bytes processed
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_get_statistics(DynamicContextManager* manager,
                                                   uint64_t* total_adaptations,
                                                   float* success_rate,
                                                   float* average_effectiveness,
                                                   uint64_t* bytes_processed);

// Utility Functions

/**
 * Get error string for error code
 * @param error Error code
 * @return Human-readable error string
 */
const char* dynamic_context_error_string(DynamicContextError error);

/**
 * Get strategy name
 * @param strategy Context sizing strategy
 * @return Human-readable strategy name
 */
const char* dynamic_context_strategy_name(ContextSizingStrategy strategy);

/**
 * Get complexity level name
 * @param complexity Content complexity level
 * @return Human-readable complexity name
 */
const char* dynamic_context_complexity_name(ContentComplexity complexity);

/**
 * Calculate recommended buffer sizes based on content analysis
 * @param content Content data
 * @param content_size Size of content
 * @param strategy Sizing strategy
 * @param window_size Output recommended window size
 * @param memory_size Output recommended memory size
 * @return DYNAMIC_CONTEXT_SUCCESS on success, error code on failure
 */
DynamicContextError dynamic_context_calculate_sizes(const uint8_t* content,
                                                    uint32_t content_size,
                                                    ContextSizingStrategy strategy,
                                                    uint32_t* window_size,
                                                    uint32_t* memory_size);

/**
 * Detect memory pressure level
 * @param available_memory Available memory in bytes
 * @param required_memory Required memory in bytes
 * @return Memory pressure level
 */
MemoryPressureLevel dynamic_context_detect_memory_pressure(uint64_t available_memory,
                                                           uint64_t required_memory);

#ifdef __cplusplus
}
#endif

#endif // DYNAMIC_CONTEXT_H
