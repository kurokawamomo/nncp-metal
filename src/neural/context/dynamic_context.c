/**
 * @file dynamic_context.c
 * @brief Dynamic Context Management Implementation
 * 
 * Implementation of adaptive context window management for neural compression
 * with intelligent sizing based on content characteristics and performance.
 */

#include "dynamic_context.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

// Internal constants
#define EFFECTIVENESS_DECAY_RATE        0.95f   // Decay rate for effectiveness averaging
#define ADAPTATION_COOLDOWN_NS          1000000000ULL // 1 second cooldown
#define MEMORY_PRESSURE_HYSTERESIS      0.1f    // Hysteresis for memory pressure detection
#define CONTENT_SAMPLE_SIZE             1024    // Sample size for content analysis
#define METAL_BUFFER_ALIGNMENT          64      // Metal buffer alignment

// Forward declarations for internal functions
static DynamicContextError allocate_dynamic_buffer(DynamicBuffer* buffer, uint32_t size, bool use_metal);
static DynamicContextError reallocate_dynamic_buffer(DynamicBuffer* buffer, uint32_t new_size);
static void free_dynamic_buffer(DynamicBuffer* buffer);
static ContentComplexity analyze_content_complexity(const uint8_t* content, uint32_t size);
static float calculate_effectiveness_score(const DynamicContextMetrics* metrics, ContentComplexity complexity);
static uint32_t recommend_window_size(ContentComplexity complexity, ContextSizingStrategy strategy, uint32_t min_size, uint32_t max_size);
static uint32_t recommend_memory_size(ContentComplexity complexity, ContextSizingStrategy strategy, uint32_t min_size, uint32_t max_size);
static bool should_trigger_adaptation(DynamicContextManager* manager, float current_effectiveness);
static uint64_t get_current_time_ns(void);
static uint64_t get_available_memory(void);

// Core API Implementation

DynamicContextError dynamic_context_create(DynamicContextManager** manager,
                                          const DynamicContextConfig* config) {
    if (!manager || !config) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    // Validate configuration parameters
    if (config->min_window_size < DYNAMIC_CONTEXT_MIN_WINDOW_SIZE ||
        config->max_window_size > DYNAMIC_CONTEXT_MAX_WINDOW_SIZE ||
        config->min_window_size > config->max_window_size ||
        config->min_memory_size < DYNAMIC_CONTEXT_MIN_MEMORY_SIZE ||
        config->max_memory_size > DYNAMIC_CONTEXT_MAX_MEMORY_SIZE ||
        config->min_memory_size > config->max_memory_size) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    // Allocate manager structure
    DynamicContextManager* mgr = calloc(1, sizeof(DynamicContextManager));
    if (!mgr) {
        return DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION;
    }

    // Initialize configuration
    mgr->config = *config;
    mgr->context_id = (uint32_t)time(NULL); // Simple ID generation
    mgr->is_initialized = false;

    // Initialize mutexes
    if (pthread_mutex_init(&mgr->context_mutex, NULL) != 0) {
        free(mgr);
        return DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION;
    }

    if (pthread_rwlock_init(&mgr->config_lock, NULL) != 0) {
        pthread_mutex_destroy(&mgr->context_mutex);
        free(mgr);
        return DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION;
    }

    // Initialize buffer mutexes
    if (pthread_mutex_init(&mgr->attention_buffer.buffer_mutex, NULL) != 0 ||
        pthread_mutex_init(&mgr->memory_buffer.buffer_mutex, NULL) != 0) {
        pthread_rwlock_destroy(&mgr->config_lock);
        pthread_mutex_destroy(&mgr->context_mutex);
        free(mgr);
        return DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION;
    }

    // Set initial sizes
    mgr->current_window_size = DYNAMIC_CONTEXT_DEFAULT_WINDOW_SIZE;
    mgr->current_memory_size = DYNAMIC_CONTEXT_DEFAULT_MEMORY_SIZE;
    mgr->current_complexity = DYNAMIC_CONTEXT_COMPLEXITY_MEDIUM;
    mgr->current_pressure = DYNAMIC_CONTEXT_MEMORY_PRESSURE_NONE;

    // Allocate initial buffers
    DynamicContextError error = allocate_dynamic_buffer(&mgr->attention_buffer, 
                                                       mgr->current_window_size,
                                                       config->enable_metal_acceleration);
    if (error != DYNAMIC_CONTEXT_SUCCESS) {
        pthread_mutex_destroy(&mgr->memory_buffer.buffer_mutex);
        pthread_mutex_destroy(&mgr->attention_buffer.buffer_mutex);
        pthread_rwlock_destroy(&mgr->config_lock);
        pthread_mutex_destroy(&mgr->context_mutex);
        free(mgr);
        return error;
    }

    error = allocate_dynamic_buffer(&mgr->memory_buffer, 
                                   mgr->current_memory_size,
                                   config->enable_metal_acceleration);
    if (error != DYNAMIC_CONTEXT_SUCCESS) {
        free_dynamic_buffer(&mgr->attention_buffer);
        pthread_mutex_destroy(&mgr->memory_buffer.buffer_mutex);
        pthread_mutex_destroy(&mgr->attention_buffer.buffer_mutex);
        pthread_rwlock_destroy(&mgr->config_lock);
        pthread_mutex_destroy(&mgr->context_mutex);
        free(mgr);
        return error;
    }

    // Initialize effectiveness monitor
    mgr->monitor.current_effectiveness = 0.5f; // Start with neutral effectiveness
    mgr->monitor.baseline_effectiveness = 0.5f;
    mgr->monitor.detected_complexity = DYNAMIC_CONTEXT_COMPLEXITY_MEDIUM;
    mgr->monitor.memory_pressure = DYNAMIC_CONTEXT_MEMORY_PRESSURE_NONE;
    mgr->monitor.last_adaptation_time = get_current_time_ns();

    mgr->is_initialized = true;
    *manager = mgr;

    return DYNAMIC_CONTEXT_SUCCESS;
}

void dynamic_context_destroy(DynamicContextManager* manager) {
    if (!manager) {
        return;
    }

    // Free buffers
    free_dynamic_buffer(&manager->attention_buffer);
    free_dynamic_buffer(&manager->memory_buffer);

    // Destroy mutexes
    pthread_mutex_destroy(&manager->memory_buffer.buffer_mutex);
    pthread_mutex_destroy(&manager->attention_buffer.buffer_mutex);
    pthread_rwlock_destroy(&manager->config_lock);
    pthread_mutex_destroy(&manager->context_mutex);

    // Free manager
    free(manager);
}

DynamicContextError dynamic_context_config_default(DynamicContextConfig* config,
                                                   ContextSizingStrategy strategy) {
    if (!config) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    memset(config, 0, sizeof(DynamicContextConfig));

    config->strategy = strategy;
    config->min_window_size = DYNAMIC_CONTEXT_MIN_WINDOW_SIZE;
    config->max_window_size = DYNAMIC_CONTEXT_MAX_WINDOW_SIZE;
    config->min_memory_size = DYNAMIC_CONTEXT_MIN_MEMORY_SIZE;
    config->max_memory_size = DYNAMIC_CONTEXT_MAX_MEMORY_SIZE;
    config->adaptation_threshold = 0.1f; // 10% improvement threshold
    config->memory_pressure_threshold = 0.8f; // 80% memory usage threshold
    config->adaptation_interval = 65536; // 64KB processing interval
    config->enable_metal_acceleration = true;
    config->enable_predictive_sizing = true;
    config->history_window = DYNAMIC_CONTEXT_ADAPTATION_HISTORY;

    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_update_config(DynamicContextManager* manager,
                                                  const DynamicContextConfig* config) {
    if (!manager || !config || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_rwlock_wrlock(&manager->config_lock);
    manager->config = *config;
    pthread_rwlock_unlock(&manager->config_lock);

    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_get_attention_buffer(DynamicContextManager* manager,
                                                        uint8_t** buffer,
                                                        uint32_t* size) {
    if (!manager || !buffer || !size || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&manager->attention_buffer.buffer_mutex);
    *buffer = manager->attention_buffer.data;
    *size = manager->attention_buffer.capacity;  // Return capacity, not current size
    pthread_mutex_unlock(&manager->attention_buffer.buffer_mutex);

    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_get_memory_buffer(DynamicContextManager* manager,
                                                     uint8_t** buffer,
                                                     uint32_t* size) {
    if (!manager || !buffer || !size || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&manager->memory_buffer.buffer_mutex);
    *buffer = manager->memory_buffer.data;
    *size = manager->memory_buffer.capacity;  // Return capacity, not current size
    pthread_mutex_unlock(&manager->memory_buffer.buffer_mutex);

    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_resize_attention_buffer(DynamicContextManager* manager,
                                                           uint32_t new_size) {
    if (!manager || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_rwlock_rdlock(&manager->config_lock);
    uint32_t min_size = manager->config.min_window_size;
    uint32_t max_size = manager->config.max_window_size;
    pthread_rwlock_unlock(&manager->config_lock);

    if (new_size < min_size || new_size > max_size) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_SIZE;
    }

    DynamicContextError error = reallocate_dynamic_buffer(&manager->attention_buffer, new_size);
    if (error == DYNAMIC_CONTEXT_SUCCESS) {
        manager->current_window_size = new_size;
    }

    return error;
}

DynamicContextError dynamic_context_resize_memory_buffer(DynamicContextManager* manager,
                                                        uint32_t new_size) {
    if (!manager || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_rwlock_rdlock(&manager->config_lock);
    uint32_t min_size = manager->config.min_memory_size;
    uint32_t max_size = manager->config.max_memory_size;
    pthread_rwlock_unlock(&manager->config_lock);

    if (new_size < min_size || new_size > max_size) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_SIZE;
    }

    DynamicContextError error = reallocate_dynamic_buffer(&manager->memory_buffer, new_size);
    if (error == DYNAMIC_CONTEXT_SUCCESS) {
        manager->current_memory_size = new_size;
    }

    return error;
}

DynamicContextError dynamic_context_update_metrics(DynamicContextManager* manager,
                                                   const DynamicContextMetrics* metrics,
                                                   ContentComplexity content_complexity) {
    if (!manager || !metrics || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&manager->context_mutex);

    // Calculate effectiveness score
    float effectiveness = calculate_effectiveness_score(metrics, content_complexity);
    
    // Update current effectiveness with decay
    manager->monitor.current_effectiveness = 
        manager->monitor.current_effectiveness * EFFECTIVENESS_DECAY_RATE +
        effectiveness * (1.0f - EFFECTIVENESS_DECAY_RATE);

    // Update detected complexity
    manager->monitor.detected_complexity = content_complexity;
    manager->current_complexity = content_complexity;

    // Add to history
    uint32_t index = manager->monitor.history_index;
    AdaptationHistoryEntry* entry = &manager->monitor.history[index];
    
    entry->window_size = manager->current_window_size;
    entry->memory_size = manager->current_memory_size;
    entry->metrics = *metrics;
    entry->complexity = content_complexity;
    entry->timestamp = get_current_time_ns();
    entry->effectiveness_score = effectiveness;

    manager->monitor.history_index = (index + 1) % DYNAMIC_CONTEXT_ADAPTATION_HISTORY;
    if (manager->monitor.history_count < DYNAMIC_CONTEXT_ADAPTATION_HISTORY) {
        manager->monitor.history_count++;
    }

    // Update statistics
    manager->total_bytes_processed += metrics->processing_time_ns > 0 ? 
        (uint64_t)(metrics->compression_speed * metrics->processing_time_ns / 1e9) : 0;
    
    manager->average_effectiveness = 
        manager->average_effectiveness * 0.9f + effectiveness * 0.1f;

    pthread_mutex_unlock(&manager->context_mutex);

    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_trigger_adaptation(DynamicContextManager* manager,
                                                       bool force_adaptation) {
    if (!manager || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&manager->context_mutex);

    uint64_t current_time = get_current_time_ns();
    float current_effectiveness = manager->monitor.current_effectiveness;

    // Check cooldown period
    if (!force_adaptation && 
        (current_time - manager->monitor.last_adaptation_time) < ADAPTATION_COOLDOWN_NS) {
        pthread_mutex_unlock(&manager->context_mutex);
        return DYNAMIC_CONTEXT_SUCCESS;
    }

    // Check if adaptation is needed
    if (!force_adaptation && !should_trigger_adaptation(manager, current_effectiveness)) {
        pthread_mutex_unlock(&manager->context_mutex);
        return DYNAMIC_CONTEXT_SUCCESS;
    }

    // Determine new sizes based on current effectiveness and complexity
    pthread_rwlock_rdlock(&manager->config_lock);
    ContextSizingStrategy strategy = manager->config.strategy;
    uint32_t min_window = manager->config.min_window_size;
    uint32_t max_window = manager->config.max_window_size;
    uint32_t min_memory = manager->config.min_memory_size;
    uint32_t max_memory = manager->config.max_memory_size;
    pthread_rwlock_unlock(&manager->config_lock);

    uint32_t new_window_size = recommend_window_size(manager->current_complexity, 
                                                    strategy, min_window, max_window);
    uint32_t new_memory_size = recommend_memory_size(manager->current_complexity, 
                                                    strategy, min_memory, max_memory);

    // Check memory pressure
    uint64_t available_memory = get_available_memory();
    uint64_t required_memory = new_window_size + new_memory_size;
    MemoryPressureLevel pressure = dynamic_context_detect_memory_pressure(available_memory, required_memory);
    
    if (pressure >= DYNAMIC_CONTEXT_MEMORY_PRESSURE_HIGH) {
        // Scale down sizes due to memory pressure
        new_window_size = (uint32_t)(new_window_size * 0.7f);
        new_memory_size = (uint32_t)(new_memory_size * 0.7f);
        new_window_size = new_window_size < min_window ? min_window : new_window_size;
        new_memory_size = new_memory_size < min_memory ? min_memory : new_memory_size;
    }

    manager->current_pressure = pressure;

    // Perform adaptations if sizes changed significantly
    bool adapted = false;
    DynamicContextError error = DYNAMIC_CONTEXT_SUCCESS;

    if (abs((int32_t)new_window_size - (int32_t)manager->current_window_size) > 
        (int32_t)(manager->current_window_size * 0.1f)) {
        pthread_mutex_unlock(&manager->context_mutex);
        error = dynamic_context_resize_attention_buffer(manager, new_window_size);
        pthread_mutex_lock(&manager->context_mutex);
        
        if (error == DYNAMIC_CONTEXT_SUCCESS) {
            adapted = true;
        }
    }

    if (error == DYNAMIC_CONTEXT_SUCCESS &&
        abs((int32_t)new_memory_size - (int32_t)manager->current_memory_size) > 
        (int32_t)(manager->current_memory_size * 0.1f)) {
        pthread_mutex_unlock(&manager->context_mutex);
        error = dynamic_context_resize_memory_buffer(manager, new_memory_size);
        pthread_mutex_lock(&manager->context_mutex);
        
        if (error == DYNAMIC_CONTEXT_SUCCESS) {
            adapted = true;
        }
    }

    // Update adaptation statistics
    manager->total_adaptations++;
    if (error == DYNAMIC_CONTEXT_SUCCESS && adapted) {
        manager->successful_adaptations++;
        manager->monitor.last_adaptation_time = current_time;
    }

    pthread_mutex_unlock(&manager->context_mutex);

    return error;
}

DynamicContextError dynamic_context_analyze_content(DynamicContextManager* manager,
                                                    const uint8_t* content,
                                                    uint32_t content_size,
                                                    uint32_t* recommended_window_size,
                                                    uint32_t* recommended_memory_size) {
    if (!manager || !content || !recommended_window_size || !recommended_memory_size || 
        !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    // Analyze content complexity
    uint32_t sample_size = content_size < CONTENT_SAMPLE_SIZE ? content_size : CONTENT_SAMPLE_SIZE;
    ContentComplexity complexity = analyze_content_complexity(content, sample_size);

    pthread_rwlock_rdlock(&manager->config_lock);
    ContextSizingStrategy strategy = manager->config.strategy;
    uint32_t min_window = manager->config.min_window_size;
    uint32_t max_window = manager->config.max_window_size;
    uint32_t min_memory = manager->config.min_memory_size;
    uint32_t max_memory = manager->config.max_memory_size;
    pthread_rwlock_unlock(&manager->config_lock);

    *recommended_window_size = recommend_window_size(complexity, strategy, min_window, max_window);
    *recommended_memory_size = recommend_memory_size(complexity, strategy, min_memory, max_memory);

    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_handle_memory_pressure(DynamicContextManager* manager,
                                                           uint64_t available_memory) {
    if (!manager || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    uint64_t current_usage = manager->current_window_size + manager->current_memory_size;
    MemoryPressureLevel pressure = dynamic_context_detect_memory_pressure(available_memory, current_usage);

    if (pressure != manager->current_pressure) {
        manager->current_pressure = pressure;
        
        // Trigger adaptation if memory pressure changed significantly
        if (pressure >= DYNAMIC_CONTEXT_MEMORY_PRESSURE_MEDIUM) {
            return dynamic_context_trigger_adaptation(manager, true);
        }
    }

    return DYNAMIC_CONTEXT_SUCCESS;
}

// Internal function implementations

static DynamicContextError allocate_dynamic_buffer(DynamicBuffer* buffer, uint32_t size, bool use_metal) {
    if (!buffer) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    memset(buffer, 0, sizeof(DynamicBuffer));

    if (use_metal) {
        // For Metal acceleration, use aligned allocation
        buffer->data = aligned_alloc(METAL_BUFFER_ALIGNMENT, size);
    } else {
        buffer->data = malloc(size);
    }

    if (!buffer->data) {
        return DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION;
    }

    buffer->capacity = size;
    buffer->size = 0;
    buffer->window_size = size;
    buffer->is_metal_backed = use_metal;
    buffer->metal_buffer = NULL; // Would be initialized with actual Metal buffer

    return DYNAMIC_CONTEXT_SUCCESS;
}

static DynamicContextError reallocate_dynamic_buffer(DynamicBuffer* buffer, uint32_t new_size) {
    if (!buffer || !buffer->data) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&buffer->buffer_mutex);

    uint8_t* new_data;
    if (buffer->is_metal_backed) {
        new_data = aligned_alloc(METAL_BUFFER_ALIGNMENT, new_size);
    } else {
        new_data = realloc(buffer->data, new_size);
    }

    if (!new_data) {
        pthread_mutex_unlock(&buffer->buffer_mutex);
        return DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION;
    }

    if (buffer->is_metal_backed) {
        // Copy existing data for Metal buffers
        uint32_t copy_size = buffer->size < new_size ? buffer->size : new_size;
        memcpy(new_data, buffer->data, copy_size);
        free(buffer->data);
    }

    buffer->data = new_data;
    buffer->capacity = new_size;
    buffer->window_size = new_size;
    if (buffer->size > new_size) {
        buffer->size = new_size;
    }

    pthread_mutex_unlock(&buffer->buffer_mutex);

    return DYNAMIC_CONTEXT_SUCCESS;
}

static void free_dynamic_buffer(DynamicBuffer* buffer) {
    if (!buffer) {
        return;
    }

    if (buffer->data) {
        free(buffer->data);
        buffer->data = NULL;
    }
    
    buffer->capacity = 0;
    buffer->size = 0;
    buffer->window_size = 0;
    buffer->is_metal_backed = false;
    buffer->metal_buffer = NULL;
}

static ContentComplexity analyze_content_complexity(const uint8_t* content, uint32_t size) {
    if (!content || size == 0) {
        return DYNAMIC_CONTEXT_COMPLEXITY_LOW;
    }

    // Calculate entropy and other complexity metrics
    uint32_t byte_counts[256] = {0};
    for (uint32_t i = 0; i < size; i++) {
        byte_counts[content[i]]++;
    }

    // Calculate entropy
    float entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (byte_counts[i] > 0) {
            float probability = (float)byte_counts[i] / size;
            entropy -= probability * log2f(probability);
        }
    }

    // Calculate pattern complexity
    uint32_t unique_bytes = 0;
    for (int i = 0; i < 256; i++) {
        if (byte_counts[i] > 0) {
            unique_bytes++;
        }
    }

    float complexity_score = entropy / 8.0f; // Normalize to 0-1

    // Classify complexity
    if (complexity_score < 0.3f || unique_bytes < 16) {
        return DYNAMIC_CONTEXT_COMPLEXITY_LOW;
    } else if (complexity_score < 0.6f || unique_bytes < 64) {
        return DYNAMIC_CONTEXT_COMPLEXITY_MEDIUM;
    } else if (complexity_score < 0.8f || unique_bytes < 128) {
        return DYNAMIC_CONTEXT_COMPLEXITY_HIGH;
    } else {
        return DYNAMIC_CONTEXT_COMPLEXITY_VARIABLE;
    }
}

static float calculate_effectiveness_score(const DynamicContextMetrics* metrics, 
                                         ContentComplexity complexity) {
    if (!metrics) {
        return 0.0f;
    }

    // Weight factors based on importance
    float compression_weight = 0.4f;
    float speed_weight = 0.2f;
    float memory_weight = 0.2f;
    float accuracy_weight = 0.2f;

    // Normalize compression ratio (lower is better, target 0.5 for 50% compression)
    float compression_score = 1.0f - (metrics->compression_ratio > 1.0f ? 1.0f : metrics->compression_ratio);
    
    // Normalize speed (higher is better, assume 1MB/s as baseline)
    float speed_score = metrics->compression_speed / 1000000.0f;
    speed_score = speed_score > 1.0f ? 1.0f : speed_score;

    // Memory efficiency score (0-1, higher is better)
    float memory_score = metrics->memory_efficiency;

    // Accuracy score (0-100%, normalize to 0-1)
    float accuracy_score = metrics->prediction_accuracy / 100.0f;

    // Calculate weighted score
    float score = compression_score * compression_weight +
                  speed_score * speed_weight +
                  memory_score * memory_weight +
                  accuracy_score * accuracy_weight;

    // Adjust for complexity
    switch (complexity) {
        case DYNAMIC_CONTEXT_COMPLEXITY_LOW:
            score *= 1.1f; // Boost for handling simple content well
            break;
        case DYNAMIC_CONTEXT_COMPLEXITY_HIGH:
        case DYNAMIC_CONTEXT_COMPLEXITY_VARIABLE:
            score *= 1.2f; // Boost for handling complex content
            break;
        default:
            break;
    }

    return score > 1.0f ? 1.0f : score;
}

static uint32_t recommend_window_size(ContentComplexity complexity, 
                                     ContextSizingStrategy strategy,
                                     uint32_t min_size, uint32_t max_size) {
    uint32_t base_size = DYNAMIC_CONTEXT_DEFAULT_WINDOW_SIZE;
    float multiplier = 1.0f;

    // Adjust based on complexity
    switch (complexity) {
        case DYNAMIC_CONTEXT_COMPLEXITY_LOW:
            multiplier = 0.7f;
            break;
        case DYNAMIC_CONTEXT_COMPLEXITY_MEDIUM:
            multiplier = 1.0f;
            break;
        case DYNAMIC_CONTEXT_COMPLEXITY_HIGH:
            multiplier = 1.4f;
            break;
        case DYNAMIC_CONTEXT_COMPLEXITY_VARIABLE:
            multiplier = 1.8f;
            break;
    }

    // Adjust based on strategy
    switch (strategy) {
        case DYNAMIC_CONTEXT_STRATEGY_CONSERVATIVE:
            multiplier *= 0.8f;
            break;
        case DYNAMIC_CONTEXT_STRATEGY_AGGRESSIVE:
            multiplier *= 1.3f;
            break;
        case DYNAMIC_CONTEXT_STRATEGY_BALANCED:
            multiplier *= 1.0f;
            break;
        case DYNAMIC_CONTEXT_STRATEGY_CONTENT_AWARE:
            multiplier *= 1.1f;
            break;
    }

    uint32_t recommended = (uint32_t)(base_size * multiplier);
    
    // Ensure within bounds
    if (recommended < min_size) recommended = min_size;
    if (recommended > max_size) recommended = max_size;

    return recommended;
}

static uint32_t recommend_memory_size(ContentComplexity complexity, 
                                     ContextSizingStrategy strategy,
                                     uint32_t min_size, uint32_t max_size) {
    uint32_t base_size = DYNAMIC_CONTEXT_DEFAULT_MEMORY_SIZE;
    float multiplier = 1.0f;

    // Memory needs scale differently than window size
    switch (complexity) {
        case DYNAMIC_CONTEXT_COMPLEXITY_LOW:
            multiplier = 0.6f;
            break;
        case DYNAMIC_CONTEXT_COMPLEXITY_MEDIUM:
            multiplier = 1.0f;
            break;
        case DYNAMIC_CONTEXT_COMPLEXITY_HIGH:
            multiplier = 1.6f;
            break;
        case DYNAMIC_CONTEXT_COMPLEXITY_VARIABLE:
            multiplier = 2.0f;
            break;
    }

    // Strategy adjustments
    switch (strategy) {
        case DYNAMIC_CONTEXT_STRATEGY_CONSERVATIVE:
            multiplier *= 0.9f;
            break;
        case DYNAMIC_CONTEXT_STRATEGY_AGGRESSIVE:
            multiplier *= 1.2f;
            break;
        case DYNAMIC_CONTEXT_STRATEGY_BALANCED:
            multiplier *= 1.0f;
            break;
        case DYNAMIC_CONTEXT_STRATEGY_CONTENT_AWARE:
            multiplier *= 1.15f;
            break;
    }

    uint32_t recommended = (uint32_t)(base_size * multiplier);
    
    // Ensure within bounds
    if (recommended < min_size) recommended = min_size;
    if (recommended > max_size) recommended = max_size;

    return recommended;
}

static bool should_trigger_adaptation(DynamicContextManager* manager, float current_effectiveness) {
    if (manager->monitor.history_count < 3) {
        return false; // Need some history
    }

    float threshold = manager->config.adaptation_threshold;
    float baseline = manager->monitor.baseline_effectiveness;

    // Check if current effectiveness significantly differs from baseline
    float improvement = current_effectiveness - baseline;
    if (improvement < -threshold) {
        return true; // Performance degraded
    }

    // Check if we can improve significantly
    float max_recent = 0.0f;
    uint32_t recent_entries = manager->monitor.history_count < 5 ? manager->monitor.history_count : 5;
    for (uint32_t i = 0; i < recent_entries; i++) {
        uint32_t index = (manager->monitor.history_index - 1 - i + DYNAMIC_CONTEXT_ADAPTATION_HISTORY) 
                        % DYNAMIC_CONTEXT_ADAPTATION_HISTORY;
        float score = manager->monitor.history[index].effectiveness_score;
        if (score > max_recent) {
            max_recent = score;
        }
    }

    return (max_recent - current_effectiveness) > threshold;
}

static uint64_t get_current_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static uint64_t get_available_memory(void) {
    // Simple approximation using system page size and available pages
    long page_size = sysconf(_SC_PAGESIZE);
    
#ifdef _SC_AVPHYS_PAGES
    long available_pages = sysconf(_SC_AVPHYS_PAGES);
    if (page_size > 0 && available_pages > 0) {
        return (uint64_t)page_size * available_pages;
    }
#else
    // For macOS and other systems without _SC_AVPHYS_PAGES
    // Use a conservative estimate based on total memory
    long total_pages = sysconf(_SC_PHYS_PAGES);
    if (page_size > 0 && total_pages > 0) {
        // Assume 70% of total memory is available
        return (uint64_t)(page_size * total_pages * 0.7);
    }
#endif
    
    return 1024 * 1024 * 1024; // Default to 1GB if can't determine
}

// Utility function implementations

const char* dynamic_context_error_string(DynamicContextError error) {
    switch (error) {
        case DYNAMIC_CONTEXT_SUCCESS:
            return "Success";
        case DYNAMIC_CONTEXT_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case DYNAMIC_CONTEXT_ERROR_BUFFER_OVERFLOW:
            return "Buffer overflow";
        case DYNAMIC_CONTEXT_ERROR_INVALID_SIZE:
            return "Invalid size";
        case DYNAMIC_CONTEXT_ERROR_ADAPTATION_FAILED:
            return "Adaptation failed";
        case DYNAMIC_CONTEXT_ERROR_CONTEXT_NOT_FOUND:
            return "Context not found";
        case DYNAMIC_CONTEXT_ERROR_MEMORY_PRESSURE:
            return "Memory pressure detected";
        case DYNAMIC_CONTEXT_ERROR_SYSTEM_LIMIT:
            return "System limit reached";
        case DYNAMIC_CONTEXT_ERROR_INVALID_STATE:
            return "Invalid state";
        default:
            return "Unknown error";
    }
}

const char* dynamic_context_strategy_name(ContextSizingStrategy strategy) {
    switch (strategy) {
        case DYNAMIC_CONTEXT_STRATEGY_CONSERVATIVE:
            return "Conservative";
        case DYNAMIC_CONTEXT_STRATEGY_AGGRESSIVE:
            return "Aggressive";
        case DYNAMIC_CONTEXT_STRATEGY_BALANCED:
            return "Balanced";
        case DYNAMIC_CONTEXT_STRATEGY_CONTENT_AWARE:
            return "Content-Aware";
        default:
            return "Unknown";
    }
}

const char* dynamic_context_complexity_name(ContentComplexity complexity) {
    switch (complexity) {
        case DYNAMIC_CONTEXT_COMPLEXITY_LOW:
            return "Low";
        case DYNAMIC_CONTEXT_COMPLEXITY_MEDIUM:
            return "Medium";
        case DYNAMIC_CONTEXT_COMPLEXITY_HIGH:
            return "High";
        case DYNAMIC_CONTEXT_COMPLEXITY_VARIABLE:
            return "Variable";
        default:
            return "Unknown";
    }
}

DynamicContextError dynamic_context_calculate_sizes(const uint8_t* content,
                                                    uint32_t content_size,
                                                    ContextSizingStrategy strategy,
                                                    uint32_t* window_size,
                                                    uint32_t* memory_size) {
    if (!content || !window_size || !memory_size) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    ContentComplexity complexity = analyze_content_complexity(content, 
        content_size < CONTENT_SAMPLE_SIZE ? content_size : CONTENT_SAMPLE_SIZE);

    *window_size = recommend_window_size(complexity, strategy,
                                        DYNAMIC_CONTEXT_MIN_WINDOW_SIZE,
                                        DYNAMIC_CONTEXT_MAX_WINDOW_SIZE);
    *memory_size = recommend_memory_size(complexity, strategy,
                                        DYNAMIC_CONTEXT_MIN_MEMORY_SIZE,
                                        DYNAMIC_CONTEXT_MAX_MEMORY_SIZE);

    return DYNAMIC_CONTEXT_SUCCESS;
}

MemoryPressureLevel dynamic_context_detect_memory_pressure(uint64_t available_memory,
                                                           uint64_t required_memory) {
    if (available_memory == 0) {
        return DYNAMIC_CONTEXT_MEMORY_PRESSURE_HIGH;
    }

    float usage_ratio = (float)required_memory / available_memory;

    if (usage_ratio < 0.5f) {
        return DYNAMIC_CONTEXT_MEMORY_PRESSURE_NONE;
    } else if (usage_ratio < 0.7f) {
        return DYNAMIC_CONTEXT_MEMORY_PRESSURE_LOW;
    } else if (usage_ratio < 0.9f) {
        return DYNAMIC_CONTEXT_MEMORY_PRESSURE_MEDIUM;
    } else {
        return DYNAMIC_CONTEXT_MEMORY_PRESSURE_HIGH;
    }
}

// Additional API function implementations (monitoring, statistics, etc.)

DynamicContextError dynamic_context_get_effectiveness(DynamicContextManager* manager,
                                                     float* effectiveness_score) {
    if (!manager || !effectiveness_score || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&manager->context_mutex);
    *effectiveness_score = manager->monitor.current_effectiveness;
    pthread_mutex_unlock(&manager->context_mutex);

    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_get_history(DynamicContextManager* manager,
                                                AdaptationHistoryEntry* history,
                                                uint32_t max_entries,
                                                uint32_t* num_entries) {
    if (!manager || !history || !num_entries || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&manager->context_mutex);
    
    uint32_t available = manager->monitor.history_count;
    uint32_t to_copy = available < max_entries ? available : max_entries;
    
    for (uint32_t i = 0; i < to_copy; i++) {
        uint32_t index = (manager->monitor.history_index - to_copy + i + DYNAMIC_CONTEXT_ADAPTATION_HISTORY) 
                        % DYNAMIC_CONTEXT_ADAPTATION_HISTORY;
        history[i] = manager->monitor.history[index];
    }
    
    *num_entries = to_copy;
    
    pthread_mutex_unlock(&manager->context_mutex);

    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_get_statistics(DynamicContextManager* manager,
                                                   uint64_t* total_adaptations,
                                                   float* success_rate,
                                                   float* average_effectiveness,
                                                   uint64_t* bytes_processed) {
    if (!manager || !total_adaptations || !success_rate || 
        !average_effectiveness || !bytes_processed || !manager->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&manager->context_mutex);
    
    *total_adaptations = manager->total_adaptations;
    *success_rate = manager->total_adaptations > 0 ? 
        (float)manager->successful_adaptations / manager->total_adaptations : 0.0f;
    *average_effectiveness = manager->average_effectiveness;
    *bytes_processed = manager->total_bytes_processed;
    
    pthread_mutex_unlock(&manager->context_mutex);

    return DYNAMIC_CONTEXT_SUCCESS;
}
