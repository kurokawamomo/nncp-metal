/*
 * ProgressiveCompressionEngine.h
 * 
 * Progressive Compression Engine for Neural Network Compression
 * Three-tier processing strategy (Basic → Extended → Premium)
 * Adaptive algorithm selection and resource optimization
 */

#ifndef PROGRESSIVE_COMPRESSION_ENGINE_H
#define PROGRESSIVE_COMPRESSION_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../validation/IntegrityValidator.h"
#include "../robustness/RobustCompressionEngine.h"
#include "../verification/PerformanceVerifier.h"

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct ProgressiveCompressionEngine ProgressiveCompressionEngine;

// Compression tiers for progressive processing
typedef enum {
    COMPRESSION_TIER_BASIC = 0,           // Basic compression (fast, lower ratio)
    COMPRESSION_TIER_EXTENDED,            // Extended compression (balanced)
    COMPRESSION_TIER_PREMIUM,             // Premium compression (slow, highest ratio)
    COMPRESSION_TIER_ADAPTIVE             // Adaptive tier selection
} CompressionTier;

// Processing strategy modes
typedef enum {
    STRATEGY_MODE_SPEED_OPTIMIZED = 0,    // Optimize for speed
    STRATEGY_MODE_RATIO_OPTIMIZED,        // Optimize for compression ratio
    STRATEGY_MODE_QUALITY_OPTIMIZED,      // Optimize for quality
    STRATEGY_MODE_BALANCED,               // Balanced performance
    STRATEGY_MODE_POWER_EFFICIENT,        // Power efficiency optimized
    STRATEGY_MODE_MEMORY_EFFICIENT,       // Memory usage optimized
    STRATEGY_MODE_ADAPTIVE_DYNAMIC        // Dynamic adaptation based on conditions
} StrategyMode;

// Resource allocation profiles
typedef enum {
    RESOURCE_PROFILE_MINIMAL = 0,         // Minimal resource usage
    RESOURCE_PROFILE_CONSERVATIVE,        // Conservative resource usage
    RESOURCE_PROFILE_BALANCED,            // Balanced resource allocation
    RESOURCE_PROFILE_AGGRESSIVE,          // Aggressive resource usage
    RESOURCE_PROFILE_MAXIMUM,             // Maximum resource utilization
    RESOURCE_PROFILE_ADAPTIVE            // Adaptive based on availability
} ResourceProfile;

// Neural network layer types for compression
typedef enum {
    LAYER_TYPE_DENSE = 0,                 // Dense/fully connected layers
    LAYER_TYPE_CONVOLUTIONAL,             // Convolutional layers
    LAYER_TYPE_ATTENTION,                 // Attention/transformer layers
    LAYER_TYPE_RECURRENT,                 // RNN/LSTM layers
    LAYER_TYPE_NORMALIZATION,             // Batch/layer normalization
    LAYER_TYPE_ACTIVATION,                // Activation layers
    LAYER_TYPE_POOLING,                   // Pooling layers
    LAYER_TYPE_RESIDUAL,                  // Residual connection layers
    LAYER_TYPE_EMBEDDING,                 // Embedding layers
    LAYER_TYPE_CUSTOM                     // Custom layer types
} LayerType;

// Adaptive selection criteria
typedef struct {
    float available_cpu_percentage;       // Available CPU resources (0-1)
    float available_memory_percentage;    // Available memory resources (0-1)
    float available_gpu_percentage;       // Available GPU resources (0-1)
    float current_thermal_state;          // Current thermal state (°C)
    float battery_level_percentage;       // Battery level (0-1, -1 if AC powered)
    float network_bandwidth_mbps;         // Available network bandwidth
    uint64_t target_completion_time_ms;   // Target completion time
    float quality_requirement;            // Required quality level (0-1)
    float compression_ratio_requirement;  // Required compression ratio
    bool real_time_constraint;            // Real-time processing constraint
    bool power_efficiency_priority;       // Power efficiency is priority
    bool memory_constraint_active;        // Memory constraints active
} AdaptiveSelectionCriteria;

// Tier configuration for each processing level
typedef struct {
    CompressionTier tier;                 // Compression tier
    float expected_compression_ratio;     // Expected compression ratio
    float expected_quality_score;         // Expected quality score (0-1)
    uint32_t estimated_processing_time_ms; // Estimated processing time
    float cpu_utilization_target;         // Target CPU utilization (0-1)
    float memory_utilization_target;      // Target memory utilization (0-1)
    float gpu_utilization_target;         // Target GPU utilization (0-1)
    bool uses_neural_engine;              // Uses Neural Engine acceleration
    bool uses_metal_gpu;                  // Uses Metal GPU acceleration
    bool uses_simd_optimization;          // Uses SIMD optimization
    uint32_t context_window_size;         // Context window size for processing
    uint32_t batch_size;                  // Processing batch size
    float precision_level;                // Numerical precision level (0-1)
    char tier_description[256];           // Human-readable tier description
} TierConfiguration;

// Progressive compression configuration
typedef struct {
    StrategyMode strategy_mode;           // Processing strategy mode
    ResourceProfile resource_profile;    // Resource allocation profile
    bool enable_tier_escalation;         // Enable automatic tier escalation
    bool enable_tier_fallback;           // Enable tier fallback on failure
    bool enable_adaptive_selection;      // Enable adaptive tier selection
    bool enable_parallel_processing;     // Enable parallel tier processing
    bool enable_progressive_refinement;  // Enable progressive quality refinement
    float tier_escalation_threshold;     // Quality threshold for tier escalation
    float tier_fallback_threshold;       // Performance threshold for fallback
    uint32_t max_processing_time_ms;     // Maximum allowed processing time
    uint32_t tier_timeout_ms;            // Timeout for individual tier processing
    float minimum_acceptable_quality;    // Minimum acceptable quality (0-1)
    float minimum_acceptable_ratio;      // Minimum acceptable compression ratio
    TierConfiguration basic_tier_config; // Basic tier configuration
    TierConfiguration extended_tier_config; // Extended tier configuration
    TierConfiguration premium_tier_config; // Premium tier configuration
} ProgressiveCompressionConfig;

// Layer compression metadata
typedef struct {
    LayerType layer_type;                 // Type of neural network layer
    uint32_t layer_index;                 // Index in the network
    char layer_name[64];                  // Layer name/identifier
    size_t original_size_bytes;           // Original layer size
    size_t compressed_size_bytes;         // Compressed layer size
    float compression_ratio;              // Achieved compression ratio
    float quality_score;                  // Quality score (0-1)
    CompressionTier tier_used;            // Tier used for compression
    uint64_t compression_time_ns;         // Time taken for compression
    bool compression_successful;          // Compression success status
    uint32_t error_count;                 // Number of errors encountered
    char compression_notes[256];          // Additional compression notes
} LayerCompressionMetadata;

// Progressive compression result
typedef struct {
    bool compression_successful;          // Overall compression success
    CompressionTier final_tier_used;      // Final tier used
    float overall_compression_ratio;      // Overall compression ratio achieved
    float overall_quality_score;         // Overall quality score (0-1)
    uint64_t total_processing_time_ns;    // Total processing time
    uint64_t tier_breakdown_time_ns[3];   // Time spent in each tier
    uint32_t layers_compressed;           // Number of layers compressed
    uint32_t layers_failed;               // Number of layers that failed
    uint32_t tier_escalations;            // Number of tier escalations
    uint32_t tier_fallbacks;              // Number of tier fallbacks
    size_t original_total_size;           // Original total size
    size_t compressed_total_size;         // Compressed total size
    float cpu_utilization_average;        // Average CPU utilization
    float memory_utilization_peak;        // Peak memory utilization
    float gpu_utilization_average;        // Average GPU utilization
    LayerCompressionMetadata* layer_metadata; // Per-layer metadata
    uint32_t layer_metadata_count;        // Number of layer metadata entries
    char compression_summary[1024];       // Human-readable summary
} ProgressiveCompressionResult;

// Adaptive algorithm selection result
typedef struct {
    CompressionTier recommended_tier;     // Recommended compression tier
    StrategyMode recommended_strategy;    // Recommended strategy mode
    ResourceProfile recommended_profile; // Recommended resource profile
    float confidence_score;               // Confidence in recommendation (0-1)
    float expected_performance_score;     // Expected performance score
    float expected_quality_score;         // Expected quality score
    uint32_t estimated_completion_time_ms; // Estimated completion time
    char selection_rationale[512];        // Rationale for selection
    bool requires_fallback_preparation;   // Requires fallback preparation
    bool real_time_feasible;              // Real-time processing feasible
} AdaptiveSelectionResult;

// Resource utilization tracking
typedef struct {
    uint64_t measurement_timestamp_ns;    // Measurement timestamp
    float cpu_utilization_percent;        // CPU utilization percentage
    float memory_utilization_percent;     // Memory utilization percentage
    float gpu_utilization_percent;        // GPU utilization percentage
    uint64_t memory_peak_usage_bytes;     // Peak memory usage
    uint64_t gpu_memory_usage_bytes;      // GPU memory usage
    float thermal_state_celsius;          // Thermal state
    float power_consumption_watts;        // Power consumption
    uint32_t active_threads;              // Number of active threads
    uint32_t gpu_command_buffers;         // Active GPU command buffers
    bool thermal_throttling_detected;     // Thermal throttling detected
    bool memory_pressure_detected;        // Memory pressure detected
} ResourceUtilizationSnapshot;

// Performance optimization metrics
typedef struct {
    float compression_throughput_mbps;    // Compression throughput (MB/s)
    float decompression_throughput_mbps;  // Decompression throughput (MB/s)
    float quality_per_second;             // Quality points per second
    float compression_efficiency;         // Compression efficiency ratio
    float resource_efficiency;            // Resource utilization efficiency
    float power_efficiency;               // Operations per watt
    uint32_t operations_per_second;       // Operations per second
    float latency_p50_ms;                 // 50th percentile latency
    float latency_p95_ms;                 // 95th percentile latency
    float latency_p99_ms;                 // 99th percentile latency
} PerformanceOptimizationMetrics;

// Error codes for progressive compression engine
typedef enum {
    PROGRESSIVE_ENGINE_SUCCESS = 0,
    PROGRESSIVE_ENGINE_ERROR_INVALID_PARAM,
    PROGRESSIVE_ENGINE_ERROR_MEMORY_ALLOCATION,
    PROGRESSIVE_ENGINE_ERROR_INITIALIZATION_FAILED,
    PROGRESSIVE_ENGINE_ERROR_TIER_PROCESSING_FAILED,
    PROGRESSIVE_ENGINE_ERROR_RESOURCE_EXHAUSTED,
    PROGRESSIVE_ENGINE_ERROR_TIMEOUT,
    PROGRESSIVE_ENGINE_ERROR_QUALITY_INSUFFICIENT,
    PROGRESSIVE_ENGINE_ERROR_ADAPTIVE_SELECTION_FAILED,
    PROGRESSIVE_ENGINE_ERROR_LAYER_PROCESSING_FAILED,
    PROGRESSIVE_ENGINE_ERROR_CONFIGURATION_INVALID,
    PROGRESSIVE_ENGINE_ERROR_HARDWARE_INCOMPATIBLE,
    PROGRESSIVE_ENGINE_ERROR_ALL_TIERS_FAILED
} ProgressiveEngineError;

// Core API Functions

/**
 * Create progressive compression engine instance
 * @param engine Pointer to store created engine
 * @param config Progressive compression configuration
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_create(ProgressiveCompressionEngine** engine,
                                                 const ProgressiveCompressionConfig* config);

/**
 * Initialize progressive compression engine
 * @param engine Progressive compression engine instance
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_initialize(ProgressiveCompressionEngine* engine);

/**
 * Configure tier settings for specific processing requirements
 * @param engine Progressive compression engine instance
 * @param tier Compression tier to configure
 * @param tier_config Tier configuration
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_configure_tier(ProgressiveCompressionEngine* engine,
                                                        CompressionTier tier,
                                                        const TierConfiguration* tier_config);

// Adaptive Selection and Strategy

/**
 * Perform adaptive algorithm selection based on current system state
 * @param engine Progressive compression engine instance
 * @param selection_criteria Current system and requirement criteria
 * @param selection_result Output adaptive selection result
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_adaptive_selection(ProgressiveCompressionEngine* engine,
                                                            const AdaptiveSelectionCriteria* selection_criteria,
                                                            AdaptiveSelectionResult* selection_result);

/**
 * Update strategy mode based on current conditions
 * @param engine Progressive compression engine instance
 * @param new_strategy_mode New strategy mode to apply
 * @param adaptation_reason Reason for strategy adaptation
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_update_strategy(ProgressiveCompressionEngine* engine,
                                                         StrategyMode new_strategy_mode,
                                                         const char* adaptation_reason);

/**
 * Optimize resource allocation based on current system state
 * @param engine Progressive compression engine instance
 * @param target_performance_score Target performance score (0-1)
 * @param resource_constraints Resource constraint information
 * @param optimization_successful Output boolean for optimization success
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_optimize_resources(ProgressiveCompressionEngine* engine,
                                                            float target_performance_score,
                                                            const void* resource_constraints,
                                                            bool* optimization_successful);

// Progressive Compression Operations

/**
 * Compress neural network model using progressive strategy
 * @param engine Progressive compression engine instance
 * @param model_data Neural network model data
 * @param model_size Size of model data
 * @param layer_info Array of layer information
 * @param layer_count Number of layers
 * @param compressed_data Output buffer for compressed data
 * @param compressed_buffer_size Size of compressed data buffer
 * @param compressed_size Output size of compressed data
 * @param compression_result Output compression result
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_compress_model(ProgressiveCompressionEngine* engine,
                                                        const void* model_data,
                                                        size_t model_size,
                                                        const void* layer_info,
                                                        uint32_t layer_count,
                                                        void* compressed_data,
                                                        size_t compressed_buffer_size,
                                                        size_t* compressed_size,
                                                        ProgressiveCompressionResult* compression_result);

/**
 * Compress single layer with progressive tier escalation
 * @param engine Progressive compression engine instance
 * @param layer_data Layer data to compress
 * @param layer_size Size of layer data
 * @param layer_type Type of neural network layer
 * @param starting_tier Starting compression tier
 * @param compressed_layer Output buffer for compressed layer
 * @param compressed_buffer_size Size of compressed layer buffer
 * @param compressed_layer_size Output size of compressed layer
 * @param layer_result Output layer compression metadata
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_compress_layer(ProgressiveCompressionEngine* engine,
                                                        const void* layer_data,
                                                        size_t layer_size,
                                                        LayerType layer_type,
                                                        CompressionTier starting_tier,
                                                        void* compressed_layer,
                                                        size_t compressed_buffer_size,
                                                        size_t* compressed_layer_size,
                                                        LayerCompressionMetadata* layer_result);

/**
 * Decompress neural network model with progressive validation
 * @param engine Progressive compression engine instance
 * @param compressed_data Compressed model data
 * @param compressed_size Size of compressed data
 * @param decompressed_data Output buffer for decompressed data
 * @param decompressed_buffer_size Size of decompressed data buffer
 * @param decompressed_size Output size of decompressed data
 * @param integrity_validation_result Output integrity validation result
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_decompress_model(ProgressiveCompressionEngine* engine,
                                                          const void* compressed_data,
                                                          size_t compressed_size,
                                                          void* decompressed_data,
                                                          size_t decompressed_buffer_size,
                                                          size_t* decompressed_size,
                                                          IntegrityValidationResult* integrity_validation_result);

/**
 * Perform progressive quality refinement
 * @param engine Progressive compression engine instance
 * @param initial_compressed_data Initial compressed data
 * @param initial_compressed_size Size of initial compressed data
 * @param target_quality_improvement Target quality improvement (0-1)
 * @param refinement_budget_ms Time budget for refinement
 * @param refined_compressed_data Output buffer for refined compressed data
 * @param refined_buffer_size Size of refined compressed data buffer
 * @param refined_compressed_size Output size of refined compressed data
 * @param refinement_successful Output boolean for refinement success
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_progressive_refinement(ProgressiveCompressionEngine* engine,
                                                                const void* initial_compressed_data,
                                                                size_t initial_compressed_size,
                                                                float target_quality_improvement,
                                                                uint32_t refinement_budget_ms,
                                                                void* refined_compressed_data,
                                                                size_t refined_buffer_size,
                                                                size_t* refined_compressed_size,
                                                                bool* refinement_successful);

// Resource Management and Monitoring

/**
 * Monitor resource utilization during processing
 * @param engine Progressive compression engine instance
 * @param monitoring_interval_ms Monitoring interval in milliseconds
 * @param monitoring_duration_ms Total monitoring duration
 * @param utilization_snapshots Output array of utilization snapshots
 * @param max_snapshots Maximum number of snapshots to store
 * @param snapshot_count Output number of snapshots collected
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_monitor_resources(ProgressiveCompressionEngine* engine,
                                                           uint32_t monitoring_interval_ms,
                                                           uint32_t monitoring_duration_ms,
                                                           ResourceUtilizationSnapshot* utilization_snapshots,
                                                           uint32_t max_snapshots,
                                                           uint32_t* snapshot_count);

/**
 * Calculate performance optimization metrics
 * @param engine Progressive compression engine instance
 * @param compression_result Compression result to analyze
 * @param utilization_snapshots Resource utilization snapshots
 * @param snapshot_count Number of utilization snapshots
 * @param optimization_metrics Output performance optimization metrics
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_calculate_metrics(ProgressiveCompressionEngine* engine,
                                                           const ProgressiveCompressionResult* compression_result,
                                                           const ResourceUtilizationSnapshot* utilization_snapshots,
                                                           uint32_t snapshot_count,
                                                           PerformanceOptimizationMetrics* optimization_metrics);

/**
 * Predict resource requirements for given input
 * @param engine Progressive compression engine instance
 * @param input_size Size of input data
 * @param layer_count Number of layers to process
 * @param target_tier Target compression tier
 * @param strategy_mode Processing strategy mode
 * @param predicted_cpu_usage Output predicted CPU usage (0-1)
 * @param predicted_memory_usage Output predicted memory usage (bytes)
 * @param predicted_processing_time Output predicted processing time (ms)
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_predict_requirements(ProgressiveCompressionEngine* engine,
                                                              size_t input_size,
                                                              uint32_t layer_count,
                                                              CompressionTier target_tier,
                                                              StrategyMode strategy_mode,
                                                              float* predicted_cpu_usage,
                                                              uint64_t* predicted_memory_usage,
                                                              uint32_t* predicted_processing_time);

// Configuration and Optimization

/**
 * Create default progressive compression configuration
 * @param config Output default configuration
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_create_default_config(ProgressiveCompressionConfig* config);

/**
 * Create speed-optimized configuration
 * @param config Output speed-optimized configuration
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_create_speed_config(ProgressiveCompressionConfig* config);

/**
 * Create quality-optimized configuration
 * @param config Output quality-optimized configuration
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_create_quality_config(ProgressiveCompressionConfig* config);

/**
 * Create power-efficient configuration
 * @param config Output power-efficient configuration
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_create_power_config(ProgressiveCompressionConfig* config);

/**
 * Validate progressive compression configuration
 * @param config Configuration to validate
 * @param is_valid Output boolean for configuration validity
 * @param validation_message Output validation message buffer
 * @param message_size Size of validation message buffer
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_validate_config(const ProgressiveCompressionConfig* config,
                                                         bool* is_valid,
                                                         char* validation_message,
                                                         size_t message_size);

/**
 * Auto-tune configuration based on system capabilities
 * @param engine Progressive compression engine instance
 * @param target_use_case Target use case description
 * @param performance_priority Performance vs quality priority (0-1)
 * @param tuned_config Output auto-tuned configuration
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_auto_tune_config(ProgressiveCompressionEngine* engine,
                                                          const char* target_use_case,
                                                          float performance_priority,
                                                          ProgressiveCompressionConfig* tuned_config);

/**
 * Benchmark tier performance on current system
 * @param engine Progressive compression engine instance
 * @param tier Tier to benchmark
 * @param test_data_size Size of test data for benchmarking
 * @param iterations Number of benchmark iterations
 * @param benchmark_throughput Output benchmark throughput (MB/s)
 * @param benchmark_quality Output benchmark quality score
 * @param benchmark_efficiency Output benchmark efficiency score
 * @return PROGRESSIVE_ENGINE_SUCCESS on success, error code on failure
 */
ProgressiveEngineError progressive_engine_benchmark_tier(ProgressiveCompressionEngine* engine,
                                                        CompressionTier tier,
                                                        size_t test_data_size,
                                                        uint32_t iterations,
                                                        float* benchmark_throughput,
                                                        float* benchmark_quality,
                                                        float* benchmark_efficiency);

/**
 * Destroy progressive compression engine and free resources
 * @param engine Progressive compression engine instance to destroy
 */
void progressive_engine_destroy(ProgressiveCompressionEngine* engine);

// Utility Functions

/**
 * Get error string for progressive engine error code
 * @param error_code ProgressiveEngineError code
 * @return Human-readable error message
 */
const char* progressive_engine_get_error_string(ProgressiveEngineError error_code);

/**
 * Get compression tier string
 * @param tier Compression tier enum
 * @return Human-readable tier name
 */
const char* progressive_engine_get_tier_string(CompressionTier tier);

/**
 * Get strategy mode string
 * @param strategy_mode Strategy mode enum
 * @return Human-readable strategy mode name
 */
const char* progressive_engine_get_strategy_string(StrategyMode strategy_mode);

/**
 * Get resource profile string
 * @param resource_profile Resource profile enum
 * @return Human-readable resource profile name
 */
const char* progressive_engine_get_profile_string(ResourceProfile resource_profile);

/**
 * Get layer type string
 * @param layer_type Layer type enum
 * @return Human-readable layer type name
 */
const char* progressive_engine_get_layer_type_string(LayerType layer_type);

/**
 * Calculate tier escalation score
 * @param current_quality Current quality score (0-1)
 * @param target_quality Target quality score (0-1)
 * @param resource_availability Current resource availability (0-1)
 * @return Escalation score (0-1)
 */
float progressive_engine_calculate_escalation_score(float current_quality,
                                                   float target_quality,
                                                   float resource_availability);

/**
 * Estimate tier processing time
 * @param tier Compression tier
 * @param data_size Size of data to process
 * @param layer_type Type of layer
 * @param system_performance_factor System performance factor (0-1)
 * @return Estimated processing time in milliseconds
 */
uint32_t progressive_engine_estimate_tier_time(CompressionTier tier,
                                               size_t data_size,
                                               LayerType layer_type,
                                               float system_performance_factor);

/**
 * Calculate compression efficiency score
 * @param compression_ratio Achieved compression ratio
 * @param processing_time_ms Processing time in milliseconds
 * @param quality_score Quality score (0-1)
 * @param resource_utilization Resource utilization (0-1)
 * @return Efficiency score (0-1)
 */
float progressive_engine_calculate_efficiency_score(float compression_ratio,
                                                    uint32_t processing_time_ms,
                                                    float quality_score,
                                                    float resource_utilization);

// Constants for progressive compression engine

// Tier performance characteristics
#define BASIC_TIER_EXPECTED_RATIO 0.5f              // Basic tier compression ratio
#define EXTENDED_TIER_EXPECTED_RATIO 0.3f           // Extended tier compression ratio
#define PREMIUM_TIER_EXPECTED_RATIO 0.2f            // Premium tier compression ratio

#define BASIC_TIER_EXPECTED_QUALITY 0.8f            // Basic tier quality score
#define EXTENDED_TIER_EXPECTED_QUALITY 0.9f         // Extended tier quality score
#define PREMIUM_TIER_EXPECTED_QUALITY 0.95f         // Premium tier quality score

// Processing time estimates (per MB on Apple Silicon M3)
#define BASIC_TIER_PROCESSING_TIME_MS_PER_MB 10     // Basic tier processing time
#define EXTENDED_TIER_PROCESSING_TIME_MS_PER_MB 50  // Extended tier processing time
#define PREMIUM_TIER_PROCESSING_TIME_MS_PER_MB 200  // Premium tier processing time

// Resource utilization targets
#define BASIC_TIER_CPU_TARGET 0.3f                  // Basic tier CPU target
#define EXTENDED_TIER_CPU_TARGET 0.6f               // Extended tier CPU target
#define PREMIUM_TIER_CPU_TARGET 0.9f                // Premium tier CPU target

#define BASIC_TIER_MEMORY_TARGET 0.2f               // Basic tier memory target
#define EXTENDED_TIER_MEMORY_TARGET 0.5f            // Extended tier memory target
#define PREMIUM_TIER_MEMORY_TARGET 0.8f             // Premium tier memory target

// Quality and performance thresholds
#define TIER_ESCALATION_QUALITY_THRESHOLD 0.85f    // Quality threshold for escalation
#define TIER_FALLBACK_PERFORMANCE_THRESHOLD 0.4f   // Performance threshold for fallback
#define MINIMUM_ACCEPTABLE_QUALITY 0.7f            // Minimum acceptable quality
#define MINIMUM_ACCEPTABLE_RATIO 0.1f              // Minimum acceptable compression ratio

// Adaptive selection parameters
#define ADAPTIVE_CONFIDENCE_THRESHOLD 0.8f         // Adaptive selection confidence threshold
#define RESOURCE_UTILIZATION_WARNING_THRESHOLD 0.9f // Resource utilization warning
#define THERMAL_THROTTLING_THRESHOLD 80.0f         // Thermal throttling threshold (°C)
#define BATTERY_LOW_THRESHOLD 0.2f                 // Battery low threshold
#define REAL_TIME_LATENCY_BUDGET_MS 100           // Real-time latency budget

// Performance optimization targets
#define TARGET_THROUGHPUT_MBPS 100.0f              // Target throughput (MB/s)
#define TARGET_LATENCY_P95_MS 50.0f                // Target 95th percentile latency
#define TARGET_RESOURCE_EFFICIENCY 0.8f            // Target resource efficiency
#define TARGET_POWER_EFFICIENCY 1000.0f            // Target power efficiency (ops/watt)

#ifdef __cplusplus
}
#endif

#endif // PROGRESSIVE_COMPRESSION_ENGINE_H
