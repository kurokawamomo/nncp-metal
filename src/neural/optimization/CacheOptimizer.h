/*
 * CacheOptimizer.h
 * 
 * Cache Efficiency Optimization Engine for Apple Silicon
 * L1/L2 cache-aligned data layout, sequential access patterns,
 * intelligent prefetching, and working set minimization
 */

#ifndef CACHE_OPTIMIZER_H
#define CACHE_OPTIMIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct CacheOptimizer CacheOptimizer;

// Apple Silicon cache hierarchy characteristics
typedef struct {
    size_t l1_cache_size_kb;           // L1 cache size (typically 128KB per core)
    size_t l2_cache_size_kb;           // L2 cache size (typically 12-24MB shared)
    size_t cache_line_size_bytes;      // Cache line size (typically 64 bytes)
    size_t l1_associativity;           // L1 cache associativity (typically 8-way)
    size_t l2_associativity;           // L2 cache associativity (typically 12-way)
    uint32_t prefetch_distance;        // Hardware prefetch distance
    uint32_t memory_bandwidth_gbps;    // Memory bandwidth (400+ GB/s for M3)
    uint32_t memory_latency_cycles;    // Memory access latency in cycles
    bool has_hardware_prefetcher;      // Hardware prefetcher availability
    bool supports_streaming_stores;    // Streaming store support
} AppleSiliconCacheInfo;

// Cache optimization strategy types
typedef enum {
    CACHE_STRATEGY_SEQUENTIAL = 0,     // Sequential access optimization
    CACHE_STRATEGY_BLOCKED,            // Block/tile-based access
    CACHE_STRATEGY_STREAMING,          // Streaming access for large data
    CACHE_STRATEGY_TEMPORAL_LOCALITY,  // Temporal locality optimization
    CACHE_STRATEGY_SPATIAL_LOCALITY,   // Spatial locality optimization
    CACHE_STRATEGY_ADAPTIVE,           // Adaptive strategy selection
    CACHE_STRATEGY_NEURAL_SPECIFIC     // Neural network-specific patterns
} CacheOptimizationStrategy;

// Memory access pattern analysis
typedef enum {
    ACCESS_PATTERN_SEQUENTIAL = 0,     // Sequential forward access
    ACCESS_PATTERN_REVERSE_SEQUENTIAL, // Sequential reverse access
    ACCESS_PATTERN_STRIDED,            // Strided access pattern
    ACCESS_PATTERN_RANDOM,             // Random access pattern
    ACCESS_PATTERN_BLOCKED,            // Block-wise access
    ACCESS_PATTERN_SPARSE,             // Sparse matrix access
    ACCESS_PATTERN_GATHER_SCATTER,     // Gather/scatter operations
    ACCESS_PATTERN_MIXED              // Mixed access patterns
} MemoryAccessPattern;

// Cache optimization configuration
typedef struct {
    bool enable_cache_alignment;       // Enable cache line alignment
    bool enable_prefetch_optimization; // Enable prefetch optimization
    bool enable_working_set_analysis;  // Enable working set analysis
    bool enable_access_pattern_detection; // Enable pattern detection
    bool enable_temporal_blocking;     // Enable temporal blocking
    bool enable_spatial_blocking;      // Enable spatial blocking
    bool enable_streaming_optimization; // Enable streaming optimization
    bool force_cache_friendly_layout;  // Force cache-friendly data layout
    uint32_t block_size_bytes;         // Block size for tiling (default: 64KB)
    uint32_t prefetch_distance_lines;  // Prefetch distance in cache lines
    uint32_t working_set_threshold_kb; // Working set size threshold
    float cache_miss_penalty_factor;   // Cache miss penalty weighting
    uint32_t optimization_level;       // Optimization aggressiveness (0-3)
} CacheOptimizationConfig;

// Data layout optimization parameters
typedef struct {
    size_t data_size_bytes;            // Total data size
    size_t element_size_bytes;         // Size of individual elements
    uint32_t dimensions;               // Number of dimensions
    uint32_t dim_sizes[8];             // Size of each dimension
    uint32_t access_strides[8];        // Access stride for each dimension
    MemoryAccessPattern primary_pattern; // Primary access pattern
    MemoryAccessPattern secondary_pattern; // Secondary access pattern
    float temporal_locality_factor;    // Temporal locality strength (0.0-1.0)
    float spatial_locality_factor;     // Spatial locality strength (0.0-1.0)
    bool is_read_mostly;               // Read-heavy workload
    bool is_write_mostly;              // Write-heavy workload
    bool requires_atomic_operations;   // Requires atomic operations
} DataLayoutParams;

// Cache performance metrics
typedef struct {
    uint64_t total_memory_accesses;    // Total memory operations
    uint64_t l1_cache_hits;            // L1 cache hits
    uint64_t l1_cache_misses;          // L1 cache misses
    uint64_t l2_cache_hits;            // L2 cache hits
    uint64_t l2_cache_misses;          // L2 cache misses
    uint64_t memory_accesses;          // Main memory accesses
    uint64_t prefetch_hits;            // Successful prefetches
    uint64_t prefetch_misses;          // Failed prefetches
    float l1_hit_rate;                 // L1 cache hit rate
    float l2_hit_rate;                 // L2 cache hit rate
    float overall_hit_rate;            // Overall cache hit rate
    uint64_t memory_bandwidth_used_mb; // Memory bandwidth utilization
    float cache_efficiency_score;      // Overall cache efficiency (0.0-1.0)
    uint64_t cycles_saved_by_optimization; // Estimated cycles saved
} CachePerformanceMetrics;

// Working set analysis results
typedef struct {
    size_t working_set_size_kb;        // Current working set size
    size_t peak_working_set_kb;        // Peak working set size
    uint32_t active_data_blocks;       // Number of active data blocks
    uint32_t hot_cache_lines;          // Number of hot cache lines
    uint32_t cold_cache_lines;         // Number of cold cache lines
    float working_set_locality;        // Working set locality factor
    float cache_pressure;              // Cache pressure indicator
    bool exceeds_l1_capacity;          // Working set exceeds L1 capacity
    bool exceeds_l2_capacity;          // Working set exceeds L2 capacity
    uint32_t recommended_block_size;   // Recommended tiling block size
} WorkingSetAnalysis;

// Prefetch strategy configuration
typedef struct {
    bool enable_hardware_prefetch;     // Use hardware prefetcher
    bool enable_software_prefetch;     // Use software prefetch hints
    bool enable_adaptive_distance;     // Adaptive prefetch distance
    uint32_t base_prefetch_distance;   // Base prefetch distance (cache lines)
    uint32_t max_prefetch_distance;    // Maximum prefetch distance
    uint32_t prefetch_degree;          // Number of parallel prefetches
    float confidence_threshold;        // Prefetch confidence threshold
    bool prefetch_on_cache_miss;       // Trigger prefetch on miss
    bool prefetch_on_pattern_detection; // Trigger on pattern detection
    uint32_t prefetch_buffer_size;     // Prefetch buffer size (cache lines)
} PrefetchStrategy;

// Cache optimization result
typedef struct {
    CacheOptimizationStrategy selected_strategy; // Selected optimization strategy
    uint32_t optimal_block_size_bytes; // Optimal block/tile size
    uint32_t optimal_prefetch_distance; // Optimal prefetch distance
    float estimated_performance_gain;  // Estimated performance improvement
    size_t memory_layout_overhead_bytes; // Memory overhead for optimization
    bool requires_data_reorganization; // Data needs to be reorganized
    uint32_t recommended_access_order[8]; // Recommended dimension access order
    size_t cache_aligned_size_bytes;   // Cache-aligned allocation size
    void* optimized_layout_hint;       // Hint for optimized memory layout
} CacheOptimizationResult;

// Error codes for cache optimization
typedef enum {
    CACHE_OPTIMIZER_SUCCESS = 0,
    CACHE_OPTIMIZER_ERROR_INVALID_PARAM,
    CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION,
    CACHE_OPTIMIZER_ERROR_UNSUPPORTED_PATTERN,
    CACHE_OPTIMIZER_ERROR_INSUFFICIENT_MEMORY,
    CACHE_OPTIMIZER_ERROR_CACHE_INFO_UNAVAILABLE,
    CACHE_OPTIMIZER_ERROR_OPTIMIZATION_FAILED,
    CACHE_OPTIMIZER_ERROR_ANALYSIS_FAILED,
    CACHE_OPTIMIZER_ERROR_HARDWARE_INCOMPATIBLE
} CacheOptimizerError;

// Core API Functions

/**
 * Create cache optimizer instance
 * @param optimizer Pointer to store created optimizer
 * @param config Cache optimization configuration
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_create(CacheOptimizer** optimizer,
                                          const CacheOptimizationConfig* config);

/**
 * Initialize cache optimizer with Apple Silicon cache information
 * @param optimizer Cache optimizer instance
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_initialize(CacheOptimizer* optimizer);

/**
 * Get Apple Silicon cache information
 * @param optimizer Cache optimizer instance
 * @param cache_info Output cache hierarchy information
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_get_cache_info(CacheOptimizer* optimizer,
                                                   AppleSiliconCacheInfo* cache_info);

// Data Layout Optimization

/**
 * Analyze and optimize data layout for cache efficiency
 * @param optimizer Cache optimizer instance
 * @param layout_params Data layout parameters
 * @param optimization_result Output optimization recommendations
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_optimize_data_layout(CacheOptimizer* optimizer,
                                                         const DataLayoutParams* layout_params,
                                                         CacheOptimizationResult* optimization_result);

/**
 * Create cache-aligned memory allocation
 * @param optimizer Cache optimizer instance
 * @param size Size of allocation in bytes
 * @param alignment_requirements Specific alignment requirements
 * @param aligned_ptr Output pointer to aligned memory
 * @param actual_size Output actual allocated size (may be larger)
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_allocate_aligned(CacheOptimizer* optimizer,
                                                     size_t size,
                                                     size_t alignment_requirements,
                                                     void** aligned_ptr,
                                                     size_t* actual_size);

/**
 * Reorganize data for optimal cache access patterns
 * @param optimizer Cache optimizer instance
 * @param source_data Source data to reorganize
 * @param data_size Size of data in bytes
 * @param layout_params Layout optimization parameters
 * @param reorganized_data Output reorganized data
 * @param new_layout_info Output new layout information
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_reorganize_data(CacheOptimizer* optimizer,
                                                    const void* source_data,
                                                    size_t data_size,
                                                    const DataLayoutParams* layout_params,
                                                    void** reorganized_data,
                                                    CacheOptimizationResult* new_layout_info);

// Access Pattern Optimization

/**
 * Detect memory access patterns from usage data
 * @param optimizer Cache optimizer instance
 * @param access_addresses Array of accessed memory addresses
 * @param access_count Number of memory accesses
 * @param access_timestamps Optional timestamps for temporal analysis
 * @param detected_pattern Output detected access pattern
 * @param pattern_confidence Output confidence in pattern detection
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_detect_access_pattern(CacheOptimizer* optimizer,
                                                          const void* const* access_addresses,
                                                          size_t access_count,
                                                          const uint64_t* access_timestamps,
                                                          MemoryAccessPattern* detected_pattern,
                                                          float* pattern_confidence);

/**
 * Optimize access order for given access pattern
 * @param optimizer Cache optimizer instance
 * @param access_pattern Target access pattern
 * @param data_dimensions Number of data dimensions
 * @param dimension_sizes Size of each dimension
 * @param current_order Current access order
 * @param optimized_order Output optimized access order
 * @param estimated_improvement Output estimated performance improvement
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_optimize_access_order(CacheOptimizer* optimizer,
                                                          MemoryAccessPattern access_pattern,
                                                          uint32_t data_dimensions,
                                                          const uint32_t* dimension_sizes,
                                                          const uint32_t* current_order,
                                                          uint32_t* optimized_order,
                                                          float* estimated_improvement);

/**
 * Generate cache-efficient loop tiling parameters
 * @param optimizer Cache optimizer instance
 * @param loop_nest_depth Number of nested loops
 * @param loop_bounds Upper bounds for each loop
 * @param data_access_strides Memory stride for each loop
 * @param tile_sizes Output optimal tile sizes for each loop level
 * @param tiling_strategy Output recommended tiling strategy
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_generate_tiling_params(CacheOptimizer* optimizer,
                                                          uint32_t loop_nest_depth,
                                                          const uint32_t* loop_bounds,
                                                          const size_t* data_access_strides,
                                                          uint32_t* tile_sizes,
                                                          CacheOptimizationStrategy* tiling_strategy);

// Prefetch Optimization

/**
 * Configure adaptive prefetch strategy
 * @param optimizer Cache optimizer instance
 * @param prefetch_config Prefetch strategy configuration
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_configure_prefetch(CacheOptimizer* optimizer,
                                                       const PrefetchStrategy* prefetch_config);

/**
 * Generate prefetch hints for given access pattern
 * @param optimizer Cache optimizer instance
 * @param current_address Current memory address being accessed
 * @param access_pattern Detected access pattern
 * @param prefetch_addresses Output array of addresses to prefetch
 * @param prefetch_count Maximum number of prefetch hints to generate
 * @param actual_prefetch_count Output actual number of prefetch hints generated
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_generate_prefetch_hints(CacheOptimizer* optimizer,
                                                           const void* current_address,
                                                           MemoryAccessPattern access_pattern,
                                                           const void** prefetch_addresses,
                                                           uint32_t prefetch_count,
                                                           uint32_t* actual_prefetch_count);

/**
 * Issue software prefetch instructions
 * @param optimizer Cache optimizer instance
 * @param prefetch_addresses Array of addresses to prefetch
 * @param prefetch_count Number of addresses to prefetch
 * @param prefetch_type Type of prefetch (read/write)
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_issue_prefetch(CacheOptimizer* optimizer,
                                                   const void* const* prefetch_addresses,
                                                   uint32_t prefetch_count,
                                                   bool prefetch_for_write);

// Working Set Analysis

/**
 * Analyze working set characteristics
 * @param optimizer Cache optimizer instance
 * @param data_addresses Array of data addresses in working set
 * @param data_sizes Array of data sizes for each address
 * @param access_frequencies Array of access frequencies
 * @param data_count Number of data items in working set
 * @param analysis_result Output working set analysis
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_analyze_working_set(CacheOptimizer* optimizer,
                                                        const void* const* data_addresses,
                                                        const size_t* data_sizes,
                                                        const uint32_t* access_frequencies,
                                                        size_t data_count,
                                                        WorkingSetAnalysis* analysis_result);

/**
 * Optimize working set for cache efficiency
 * @param optimizer Cache optimizer instance
 * @param current_working_set Current working set analysis
 * @param optimization_target Target optimization (L1 fit, L2 fit, etc.)
 * @param optimization_result Output optimization recommendations
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_optimize_working_set(CacheOptimizer* optimizer,
                                                         const WorkingSetAnalysis* current_working_set,
                                                         CacheOptimizationStrategy optimization_target,
                                                         CacheOptimizationResult* optimization_result);

/**
 * Minimize working set size through data compression and pruning
 * @param optimizer Cache optimizer instance
 * @param working_set_data Array of data pointers in working set
 * @param working_set_sizes Array of data sizes
 * @param working_set_priorities Array of data priorities (higher = more important)
 * @param data_count Number of data items
 * @param target_size_kb Target working set size in KB
 * @param minimized_set Output minimized working set
 * @param compression_ratio Output achieved compression ratio
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_minimize_working_set(CacheOptimizer* optimizer,
                                                         void* const* working_set_data,
                                                         const size_t* working_set_sizes,
                                                         const float* working_set_priorities,
                                                         size_t data_count,
                                                         size_t target_size_kb,
                                                         void*** minimized_set,
                                                         float* compression_ratio);

// Performance Monitoring and Analysis

/**
 * Begin cache performance monitoring
 * @param optimizer Cache optimizer instance
 * @param monitoring_interval_ms Monitoring interval in milliseconds
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_start_monitoring(CacheOptimizer* optimizer,
                                                     uint32_t monitoring_interval_ms);

/**
 * Get current cache performance metrics
 * @param optimizer Cache optimizer instance
 * @param metrics Output performance metrics
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_get_performance_metrics(CacheOptimizer* optimizer,
                                                           CachePerformanceMetrics* metrics);

/**
 * Stop cache performance monitoring
 * @param optimizer Cache optimizer instance
 * @param final_metrics Output final performance metrics
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_stop_monitoring(CacheOptimizer* optimizer,
                                                    CachePerformanceMetrics* final_metrics);

/**
 * Reset performance counters and statistics
 * @param optimizer Cache optimizer instance
 */
void cache_optimizer_reset_counters(CacheOptimizer* optimizer);

// Neural Network Specific Optimizations

/**
 * Optimize transformer attention mechanism cache access
 * @param optimizer Cache optimizer instance
 * @param sequence_length Input sequence length
 * @param d_model Model dimension
 * @param num_heads Number of attention heads
 * @param attention_optimization Output attention-specific optimization
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_optimize_attention_cache(CacheOptimizer* optimizer,
                                                            uint32_t sequence_length,
                                                            uint32_t d_model,
                                                            uint32_t num_heads,
                                                            CacheOptimizationResult* attention_optimization);

/**
 * Optimize feed-forward network cache access patterns
 * @param optimizer Cache optimizer instance
 * @param batch_size Batch size
 * @param input_dimension Input dimension
 * @param hidden_dimension Hidden dimension
 * @param ffn_optimization Output FFN-specific optimization
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_optimize_ffn_cache(CacheOptimizer* optimizer,
                                                       uint32_t batch_size,
                                                       uint32_t input_dimension,
                                                       uint32_t hidden_dimension,
                                                       CacheOptimizationResult* ffn_optimization);

/**
 * Optimize layer normalization cache access
 * @param optimizer Cache optimizer instance
 * @param vector_length Length of vectors to normalize
 * @param batch_size Number of vectors
 * @param layer_norm_optimization Output layer norm-specific optimization
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_optimize_layer_norm_cache(CacheOptimizer* optimizer,
                                                             uint32_t vector_length,
                                                             uint32_t batch_size,
                                                             CacheOptimizationResult* layer_norm_optimization);

// Configuration and Utility Functions

/**
 * Create default cache optimization configuration
 * @param config Output default configuration
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_create_default_config(CacheOptimizationConfig* config);

/**
 * Create Apple Silicon optimized configuration
 * @param config Output Apple Silicon optimized configuration
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_create_apple_silicon_config(CacheOptimizationConfig* config);

/**
 * Create aggressive cache optimization configuration
 * @param config Output aggressive optimization configuration
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_create_aggressive_config(CacheOptimizationConfig* config);

/**
 * Validate cache optimization configuration
 * @param config Configuration to validate
 * @param cache_info Available cache information for validation
 * @param is_valid Output boolean for validity
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_validate_config(const CacheOptimizationConfig* config,
                                                    const AppleSiliconCacheInfo* cache_info,
                                                    bool* is_valid);

/**
 * Estimate cache optimization benefits
 * @param optimizer Cache optimizer instance
 * @param layout_params Data layout parameters
 * @param current_performance Baseline performance metrics
 * @param estimated_improvement Output estimated improvement
 * @return CACHE_OPTIMIZER_SUCCESS on success, error code on failure
 */
CacheOptimizerError cache_optimizer_estimate_benefits(CacheOptimizer* optimizer,
                                                      const DataLayoutParams* layout_params,
                                                      const CachePerformanceMetrics* current_performance,
                                                      float* estimated_improvement);

/**
 * Destroy cache optimizer and free resources
 * @param optimizer Cache optimizer instance to destroy
 */
void cache_optimizer_destroy(CacheOptimizer* optimizer);

// Utility Functions

/**
 * Get error string for cache optimizer error code
 * @param error_code CacheOptimizerError code
 * @return Human-readable error message
 */
const char* cache_optimizer_get_error_string(CacheOptimizerError error_code);

/**
 * Get optimization strategy string
 * @param strategy Cache optimization strategy
 * @return Human-readable strategy name
 */
const char* cache_optimizer_get_strategy_string(CacheOptimizationStrategy strategy);

/**
 * Get access pattern string
 * @param pattern Memory access pattern
 * @return Human-readable pattern name
 */
const char* cache_optimizer_get_access_pattern_string(MemoryAccessPattern pattern);

/**
 * Calculate optimal alignment for given data size
 * @param data_size Size of data in bytes
 * @param cache_line_size Cache line size in bytes
 * @return Optimal alignment in bytes
 */
size_t cache_optimizer_calculate_optimal_alignment(size_t data_size, size_t cache_line_size);

/**
 * Calculate cache-aligned size for allocation
 * @param requested_size Requested allocation size
 * @param cache_line_size Cache line size
 * @return Cache-aligned allocation size
 */
size_t cache_optimizer_calculate_aligned_size(size_t requested_size, size_t cache_line_size);

/**
 * Check if address is cache-aligned
 * @param address Memory address to check
 * @param cache_line_size Cache line size
 * @return true if aligned, false otherwise
 */
bool cache_optimizer_is_cache_aligned(const void* address, size_t cache_line_size);

// Constants for Apple Silicon cache optimization
#define APPLE_SILICON_L1_CACHE_SIZE_KB 128        // L1 cache size per core
#define APPLE_SILICON_L2_CACHE_SIZE_KB 12288      // L2 cache size (12MB for M3)
#define APPLE_SILICON_CACHE_LINE_SIZE 64          // Cache line size in bytes
#define APPLE_SILICON_L1_ASSOCIATIVITY 8          // L1 cache associativity
#define APPLE_SILICON_L2_ASSOCIATIVITY 12         // L2 cache associativity
#define APPLE_SILICON_MEMORY_BANDWIDTH_GBPS 400   // Memory bandwidth (M3 Max)

// Optimization constants
#define CACHE_OPTIMIZER_DEFAULT_BLOCK_SIZE_KB 64  // Default blocking size
#define CACHE_OPTIMIZER_MAX_PREFETCH_DISTANCE 16 // Maximum prefetch distance
#define CACHE_OPTIMIZER_WORKING_SET_THRESHOLD_KB 96 // Working set threshold
#define CACHE_OPTIMIZER_ALIGNMENT_BYTES 64        // Default alignment
#define CACHE_OPTIMIZER_MIN_TILE_SIZE 32          // Minimum tile size

// Performance thresholds
#define CACHE_HIT_RATE_EXCELLENT 0.95f           // Excellent cache hit rate
#define CACHE_HIT_RATE_GOOD 0.85f                // Good cache hit rate
#define CACHE_HIT_RATE_ACCEPTABLE 0.75f          // Acceptable cache hit rate
#define CACHE_PRESSURE_HIGH 0.8f                 // High cache pressure threshold
#define WORKING_SET_EFFICIENCY_TARGET 0.9f       // Target working set efficiency

#ifdef __cplusplus
}
#endif

#endif // CACHE_OPTIMIZER_H
