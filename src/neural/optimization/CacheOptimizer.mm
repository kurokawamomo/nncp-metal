/*
 * CacheOptimizer.mm
 * 
 * Cache Efficiency Optimization Engine Implementation for Apple Silicon
 * L1/L2 cache-aligned data layout, sequential access patterns,
 * intelligent prefetching, and working set minimization
 */

#include "CacheOptimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <unistd.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#endif

// Internal structure for CacheOptimizer
struct CacheOptimizer {
    CacheOptimizationConfig config;         // Configuration settings
    AppleSiliconCacheInfo cache_info;       // Cache hierarchy information
    CachePerformanceMetrics metrics;        // Performance metrics
    WorkingSetAnalysis working_set;         // Current working set analysis
    PrefetchStrategy prefetch_strategy;     // Prefetch configuration
    
    // Cache monitoring state
    bool monitoring_active;                 // Monitoring active flag
    uint64_t monitoring_start_time;        // Monitoring start timestamp
    uint32_t monitoring_interval_ms;       // Monitoring interval
    
    // Access pattern detection
    MemoryAccessPattern detected_patterns[16]; // Recent access patterns
    float pattern_confidences[16];         // Pattern detection confidence
    uint32_t pattern_history_index;       // Pattern history circular index
    
    // Working set tracking
    void** tracked_addresses;             // Tracked memory addresses
    size_t* tracked_sizes;                // Sizes of tracked allocations
    uint32_t* access_counts;              // Access counts for each allocation
    uint64_t* last_access_times;          // Last access timestamps
    size_t max_tracked_allocations;       // Maximum tracked allocations
    size_t current_tracked_count;         // Current number of tracked allocations
    
    // Performance counters
    uint64_t total_cache_accesses;        // Total cache access attempts
    uint64_t successful_optimizations;    // Successful optimization count
    uint64_t failed_optimizations;        // Failed optimization count
    
    // Memory pools for optimization
    void* aligned_memory_pool;            // Pool for aligned allocations
    size_t memory_pool_size;              // Size of memory pool
    size_t memory_pool_used;              // Used portion of memory pool
    
    // Apple Silicon specific
    uint32_t cpu_core_count;              // Number of CPU cores
    uint32_t performance_core_count;      // Number of performance cores
    uint64_t memory_size_bytes;           // Total system memory
    
    bool initialized;                     // Initialization state
};

// Utility functions
static uint64_t get_current_time_microseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

static size_t get_system_memory_size(void) {
    size_t memory_size = 0;
    size_t size = sizeof(memory_size);
    if (sysctlbyname("hw.memsize", &memory_size, &size, NULL, 0) != 0) {
        return 8ULL * 1024 * 1024 * 1024; // Default to 8GB if detection fails
    }
    return memory_size;
}

static uint32_t get_cpu_core_count(void) {
    uint32_t core_count = 0;
    size_t size = sizeof(core_count);
    if (sysctlbyname("hw.ncpu", &core_count, &size, NULL, 0) != 0) {
        return 8; // Default to 8 cores if detection fails
    }
    return core_count;
}

static uint32_t get_performance_core_count(void) {
    uint32_t perf_core_count = 0;
    size_t size = sizeof(perf_core_count);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_core_count, &size, NULL, 0) != 0) {
        return get_cpu_core_count() / 2; // Estimate: half are performance cores
    }
    return perf_core_count;
}

// Cache line alignment utilities
static void* align_to_cache_line(void* ptr, size_t cache_line_size) {
    uintptr_t aligned = ((uintptr_t)ptr + cache_line_size - 1) & ~(cache_line_size - 1);
    return (void*)aligned;
}

static size_t align_size_to_cache_line(size_t size, size_t cache_line_size) {
    return ((size + cache_line_size - 1) / cache_line_size) * cache_line_size;
}

// Software prefetch hints (Apple Silicon specific)
static void software_prefetch_read(const void* addr) {
    __builtin_prefetch(addr, 0, 3); // Prefetch for read, high temporal locality
}

static void software_prefetch_write(const void* addr) {
    __builtin_prefetch(addr, 1, 3); // Prefetch for write, high temporal locality
}

// Core API Implementation

CacheOptimizerError cache_optimizer_create(CacheOptimizer** optimizer,
                                          const CacheOptimizationConfig* config) {
    if (!optimizer) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    *optimizer = (CacheOptimizer*)calloc(1, sizeof(CacheOptimizer));
    if (!*optimizer) {
        return CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION;
    }
    
    CacheOptimizer* opt = *optimizer;
    
    // Use provided config or create default
    if (config) {
        opt->config = *config;
    } else {
        cache_optimizer_create_default_config(&opt->config);
    }
    
    // Initialize tracking arrays
    opt->max_tracked_allocations = 1024; // Track up to 1024 allocations
    opt->tracked_addresses = (void**)calloc(opt->max_tracked_allocations, sizeof(void*));
    opt->tracked_sizes = (size_t*)calloc(opt->max_tracked_allocations, sizeof(size_t));
    opt->access_counts = (uint32_t*)calloc(opt->max_tracked_allocations, sizeof(uint32_t));
    opt->last_access_times = (uint64_t*)calloc(opt->max_tracked_allocations, sizeof(uint64_t));
    
    if (!opt->tracked_addresses || !opt->tracked_sizes || 
        !opt->access_counts || !opt->last_access_times) {
        free(opt->tracked_addresses);
        free(opt->tracked_sizes);
        free(opt->access_counts);
        free(opt->last_access_times);
        free(opt);
        *optimizer = NULL;
        return CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION;
    }
    
    // Allocate memory pool for aligned allocations
    opt->memory_pool_size = 64 * 1024 * 1024; // 64MB pool
    opt->aligned_memory_pool = aligned_alloc(APPLE_SILICON_CACHE_LINE_SIZE, opt->memory_pool_size);
    if (!opt->aligned_memory_pool) {
        opt->aligned_memory_pool = malloc(opt->memory_pool_size);
        if (!opt->aligned_memory_pool) {
            free(opt->tracked_addresses);
            free(opt->tracked_sizes);
            free(opt->access_counts);
            free(opt->last_access_times);
            free(opt);
            *optimizer = NULL;
            return CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    opt->initialized = false;
    opt->monitoring_active = false;
    opt->current_tracked_count = 0;
    opt->memory_pool_used = 0;
    opt->pattern_history_index = 0;
    
    // Initialize performance metrics
    memset(&opt->metrics, 0, sizeof(CachePerformanceMetrics));
    opt->metrics.cache_efficiency_score = 0.5f; // Start with neutral score
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_initialize(CacheOptimizer* optimizer) {
    if (!optimizer) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    // Detect Apple Silicon cache characteristics
    optimizer->cache_info.l1_cache_size_kb = APPLE_SILICON_L1_CACHE_SIZE_KB;
    optimizer->cache_info.l2_cache_size_kb = APPLE_SILICON_L2_CACHE_SIZE_KB;
    optimizer->cache_info.cache_line_size_bytes = APPLE_SILICON_CACHE_LINE_SIZE;
    optimizer->cache_info.l1_associativity = APPLE_SILICON_L1_ASSOCIATIVITY;
    optimizer->cache_info.l2_associativity = APPLE_SILICON_L2_ASSOCIATIVITY;
    optimizer->cache_info.prefetch_distance = 4; // Conservative prefetch distance
    optimizer->cache_info.memory_bandwidth_gbps = APPLE_SILICON_MEMORY_BANDWIDTH_GBPS;
    optimizer->cache_info.memory_latency_cycles = 200; // Estimated memory latency
    optimizer->cache_info.has_hardware_prefetcher = true;
    optimizer->cache_info.supports_streaming_stores = true;
    
    // Get system information
    optimizer->cpu_core_count = get_cpu_core_count();
    optimizer->performance_core_count = get_performance_core_count();
    optimizer->memory_size_bytes = get_system_memory_size();
    
    // Initialize prefetch strategy
    optimizer->prefetch_strategy.enable_hardware_prefetch = true;
    optimizer->prefetch_strategy.enable_software_prefetch = true;
    optimizer->prefetch_strategy.enable_adaptive_distance = true;
    optimizer->prefetch_strategy.base_prefetch_distance = 2;
    optimizer->prefetch_strategy.max_prefetch_distance = 8;
    optimizer->prefetch_strategy.prefetch_degree = 2;
    optimizer->prefetch_strategy.confidence_threshold = 0.7f;
    optimizer->prefetch_strategy.prefetch_on_cache_miss = true;
    optimizer->prefetch_strategy.prefetch_on_pattern_detection = true;
    optimizer->prefetch_strategy.prefetch_buffer_size = 16;
    
    // Initialize working set analysis
    memset(&optimizer->working_set, 0, sizeof(WorkingSetAnalysis));
    optimizer->working_set.working_set_locality = 0.5f;
    optimizer->working_set.recommended_block_size = CACHE_OPTIMIZER_DEFAULT_BLOCK_SIZE_KB * 1024;
    
    optimizer->initialized = true;
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_get_cache_info(CacheOptimizer* optimizer,
                                                   AppleSiliconCacheInfo* cache_info) {
    if (!optimizer || !cache_info || !optimizer->initialized) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    *cache_info = optimizer->cache_info;
    return CACHE_OPTIMIZER_SUCCESS;
}

// Data Layout Optimization

CacheOptimizerError cache_optimizer_optimize_data_layout(CacheOptimizer* optimizer,
                                                         const DataLayoutParams* layout_params,
                                                         CacheOptimizationResult* optimization_result) {
    if (!optimizer || !layout_params || !optimization_result || !optimizer->initialized) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    // Initialize result structure
    memset(optimization_result, 0, sizeof(CacheOptimizationResult));
    
    size_t data_size = layout_params->data_size_bytes;
    size_t element_size = layout_params->element_size_bytes;
    size_t cache_line_size = optimizer->cache_info.cache_line_size_bytes;
    size_t l1_capacity = optimizer->cache_info.l1_cache_size_kb * 1024;
    size_t l2_capacity = optimizer->cache_info.l2_cache_size_kb * 1024;
    
    // Determine optimal strategy based on data characteristics
    if (data_size <= l1_capacity / 4) {
        // Small data: optimize for L1 cache
        optimization_result->selected_strategy = CACHE_STRATEGY_TEMPORAL_LOCALITY;
        optimization_result->optimal_block_size_bytes = cache_line_size * 2;
    } else if (data_size <= l2_capacity / 2) {
        // Medium data: optimize for L2 cache with blocking
        optimization_result->selected_strategy = CACHE_STRATEGY_BLOCKED;
        optimization_result->optimal_block_size_bytes = 32 * 1024; // 32KB blocks
    } else {
        // Large data: use streaming strategy
        optimization_result->selected_strategy = CACHE_STRATEGY_STREAMING;
        optimization_result->optimal_block_size_bytes = 64 * 1024; // 64KB blocks
    }
    
    // Analyze access pattern and adjust strategy
    if (layout_params->primary_pattern == ACCESS_PATTERN_SEQUENTIAL) {
        optimization_result->optimal_prefetch_distance = 4;
        if (optimization_result->selected_strategy == CACHE_STRATEGY_TEMPORAL_LOCALITY) {
            optimization_result->selected_strategy = CACHE_STRATEGY_SEQUENTIAL;
        }
    } else if (layout_params->primary_pattern == ACCESS_PATTERN_STRIDED) {
        optimization_result->optimal_prefetch_distance = 2;
        optimization_result->selected_strategy = CACHE_STRATEGY_SPATIAL_LOCALITY;
    } else if (layout_params->primary_pattern == ACCESS_PATTERN_RANDOM) {
        optimization_result->optimal_prefetch_distance = 1;
        optimization_result->selected_strategy = CACHE_STRATEGY_BLOCKED;
    }
    
    // Calculate cache-aligned size
    optimization_result->cache_aligned_size_bytes = 
        align_size_to_cache_line(data_size, cache_line_size);
    optimization_result->memory_layout_overhead_bytes = 
        optimization_result->cache_aligned_size_bytes - data_size;
    
    // Determine if reorganization is needed
    optimization_result->requires_data_reorganization = 
        (layout_params->spatial_locality_factor < 0.5f) ||
        (optimization_result->selected_strategy == CACHE_STRATEGY_BLOCKED);
    
    // Calculate estimated performance gain
    float cache_miss_reduction = 0.3f; // Conservative estimate
    if (optimization_result->selected_strategy == CACHE_STRATEGY_SEQUENTIAL) {
        cache_miss_reduction = 0.5f;
    } else if (optimization_result->selected_strategy == CACHE_STRATEGY_STREAMING) {
        cache_miss_reduction = 0.4f;
    }
    
    float baseline_hit_rate = 0.7f;
    float optimized_hit_rate = fminf(0.95f, baseline_hit_rate + cache_miss_reduction);
    optimization_result->estimated_performance_gain = 
        (optimized_hit_rate - baseline_hit_rate) * 3.0f; // 3x penalty for cache miss
    
    // Generate recommended access order for multi-dimensional data
    if (layout_params->dimensions > 1) {
        for (uint32_t i = 0; i < layout_params->dimensions && i < 8; i++) {
            // Prioritize dimensions with better spatial locality
            if (layout_params->access_strides[i] == element_size) {
                optimization_result->recommended_access_order[i] = i; // Inner dimension
            } else {
                optimization_result->recommended_access_order[i] = 
                    layout_params->dimensions - 1 - i; // Reverse for cache efficiency
            }
        }
    }
    
    optimizer->successful_optimizations++;
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_allocate_aligned(CacheOptimizer* optimizer,
                                                     size_t size,
                                                     size_t alignment_requirements,
                                                     void** aligned_ptr,
                                                     size_t* actual_size) {
    if (!optimizer || !aligned_ptr || !actual_size || size == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    size_t cache_line_size = optimizer->cache_info.cache_line_size_bytes;
    size_t alignment = (alignment_requirements > 0) ? alignment_requirements : cache_line_size;
    size_t aligned_size = align_size_to_cache_line(size, alignment);
    
    // Try to allocate from memory pool first
    if (optimizer->memory_pool_used + aligned_size <= optimizer->memory_pool_size) {
        void* pool_ptr = (char*)optimizer->aligned_memory_pool + optimizer->memory_pool_used;
        void* aligned_pool_ptr = align_to_cache_line(pool_ptr, alignment);
        
        // Check if alignment caused us to exceed pool capacity
        size_t pool_offset = (char*)aligned_pool_ptr - (char*)optimizer->aligned_memory_pool;
        if (pool_offset + aligned_size <= optimizer->memory_pool_size) {
            *aligned_ptr = aligned_pool_ptr;
            *actual_size = aligned_size;
            optimizer->memory_pool_used = pool_offset + aligned_size;
            
            // Track this allocation
            if (optimizer->current_tracked_count < optimizer->max_tracked_allocations) {
                size_t idx = optimizer->current_tracked_count++;
                optimizer->tracked_addresses[idx] = *aligned_ptr;
                optimizer->tracked_sizes[idx] = aligned_size;
                optimizer->access_counts[idx] = 0;
                optimizer->last_access_times[idx] = get_current_time_microseconds();
            }
            
            return CACHE_OPTIMIZER_SUCCESS;
        }
    }
    
    // Fall back to system aligned allocation
    *aligned_ptr = aligned_alloc(alignment, aligned_size);
    if (!*aligned_ptr) {
        // Try regular malloc as last resort
        void* raw_ptr = malloc(aligned_size + alignment);
        if (!raw_ptr) {
            return CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION;
        }
        *aligned_ptr = align_to_cache_line(raw_ptr, alignment);
    }
    
    *actual_size = aligned_size;
    
    // Track this allocation
    if (optimizer->current_tracked_count < optimizer->max_tracked_allocations) {
        size_t idx = optimizer->current_tracked_count++;
        optimizer->tracked_addresses[idx] = *aligned_ptr;
        optimizer->tracked_sizes[idx] = aligned_size;
        optimizer->access_counts[idx] = 0;
        optimizer->last_access_times[idx] = get_current_time_microseconds();
    }
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_reorganize_data(CacheOptimizer* optimizer,
                                                    const void* source_data,
                                                    size_t data_size,
                                                    const DataLayoutParams* layout_params,
                                                    void** reorganized_data,
                                                    CacheOptimizationResult* new_layout_info) {
    if (!optimizer || !source_data || !layout_params || !reorganized_data || 
        !new_layout_info || data_size == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    // Allocate reorganized data buffer
    size_t aligned_size = 0;
    CacheOptimizerError error = cache_optimizer_allocate_aligned(optimizer, data_size,
                                                                0, reorganized_data, &aligned_size);
    if (error != CACHE_OPTIMIZER_SUCCESS) {
        return error;
    }
    
    // Determine reorganization strategy
    if (layout_params->primary_pattern == ACCESS_PATTERN_BLOCKED ||
        layout_params->dimensions > 1) {
        
        // Block-based reorganization for multi-dimensional data
        size_t element_size = layout_params->element_size_bytes;
        size_t cache_line_size = optimizer->cache_info.cache_line_size_bytes;
        size_t elements_per_cache_line = cache_line_size / element_size;
        
        if (elements_per_cache_line > 1 && layout_params->dimensions == 2) {
            // 2D blocking optimization
            uint32_t rows = layout_params->dim_sizes[0];
            uint32_t cols = layout_params->dim_sizes[1];
            uint32_t block_rows = (uint32_t)sqrt(elements_per_cache_line);
            uint32_t block_cols = elements_per_cache_line / block_rows;
            
            const char* src = (const char*)source_data;
            char* dst = (char*)*reorganized_data;
            size_t dst_offset = 0;
            
            for (uint32_t br = 0; br < rows; br += block_rows) {
                for (uint32_t bc = 0; bc < cols; bc += block_cols) {
                    for (uint32_t r = br; r < fminf(br + block_rows, rows); r++) {
                        for (uint32_t c = bc; c < fminf(bc + block_cols, cols); c++) {
                            size_t src_offset = (r * cols + c) * element_size;
                            memcpy(dst + dst_offset, src + src_offset, element_size);
                            dst_offset += element_size;
                        }
                    }
                }
            }
        } else {
            // Simple cache-line aligned copy
            memcpy(*reorganized_data, source_data, data_size);
        }
    } else {
        // Standard copy for sequential or other patterns
        memcpy(*reorganized_data, source_data, data_size);
    }
    
    // Generate layout optimization result
    error = cache_optimizer_optimize_data_layout(optimizer, layout_params, new_layout_info);
    if (error != CACHE_OPTIMIZER_SUCCESS) {
        free(*reorganized_data);
        *reorganized_data = NULL;
        return error;
    }
    
    new_layout_info->requires_data_reorganization = false; // Already reorganized
    new_layout_info->optimized_layout_hint = *reorganized_data;
    
    return CACHE_OPTIMIZER_SUCCESS;
}

// Access Pattern Optimization

CacheOptimizerError cache_optimizer_detect_access_pattern(CacheOptimizer* optimizer,
                                                          const void* const* access_addresses,
                                                          size_t access_count,
                                                          const uint64_t* access_timestamps,
                                                          MemoryAccessPattern* detected_pattern,
                                                          float* pattern_confidence) {
    if (!optimizer || !access_addresses || !detected_pattern || 
        !pattern_confidence || access_count < 2) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    *pattern_confidence = 0.0f;
    *detected_pattern = ACCESS_PATTERN_RANDOM;
    
    // Analyze address deltas to detect patterns
    ptrdiff_t* deltas = (ptrdiff_t*)malloc((access_count - 1) * sizeof(ptrdiff_t));
    if (!deltas) {
        return CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION;
    }
    
    for (size_t i = 1; i < access_count; i++) {
        deltas[i - 1] = (char*)access_addresses[i] - (char*)access_addresses[i - 1];
    }
    
    // Check for sequential pattern
    uint32_t sequential_count = 0;
    uint32_t reverse_sequential_count = 0;
    uint32_t stride_consistent_count = 0;
    ptrdiff_t common_stride = 0;
    
    for (size_t i = 0; i < access_count - 1; i++) {
        ptrdiff_t delta = deltas[i];
        
        if (delta > 0 && delta <= 128) { // Forward sequential within 128 bytes
            sequential_count++;
        } else if (delta < 0 && delta >= -128) { // Reverse sequential
            reverse_sequential_count++;
        }
        
        // Check for strided pattern
        if (i == 0) {
            common_stride = delta;
            stride_consistent_count = 1;
        } else if (abs(delta - common_stride) <= 8) { // Allow small variations
            stride_consistent_count++;
        }
    }
    
    size_t total_deltas = access_count - 1;
    float sequential_ratio = (float)sequential_count / total_deltas;
    float reverse_ratio = (float)reverse_sequential_count / total_deltas;
    float stride_ratio = (float)stride_consistent_count / total_deltas;
    
    // Determine dominant pattern
    if (sequential_ratio > 0.7f) {
        *detected_pattern = ACCESS_PATTERN_SEQUENTIAL;
        *pattern_confidence = sequential_ratio;
    } else if (reverse_ratio > 0.7f) {
        *detected_pattern = ACCESS_PATTERN_REVERSE_SEQUENTIAL;
        *pattern_confidence = reverse_ratio;
    } else if (stride_ratio > 0.6f && abs(common_stride) > 1) {
        *detected_pattern = ACCESS_PATTERN_STRIDED;
        *pattern_confidence = stride_ratio;
    } else if (sequential_ratio + reverse_ratio > 0.4f) {
        *detected_pattern = ACCESS_PATTERN_MIXED;
        *pattern_confidence = sequential_ratio + reverse_ratio;
    } else {
        *detected_pattern = ACCESS_PATTERN_RANDOM;
        *pattern_confidence = 1.0f - fmaxf(sequential_ratio, fmaxf(reverse_ratio, stride_ratio));
    }
    
    // Store pattern in history for learning
    optimizer->detected_patterns[optimizer->pattern_history_index] = *detected_pattern;
    optimizer->pattern_confidences[optimizer->pattern_history_index] = *pattern_confidence;
    optimizer->pattern_history_index = (optimizer->pattern_history_index + 1) % 16;
    
    free(deltas);
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_optimize_access_order(CacheOptimizer* optimizer,
                                                          MemoryAccessPattern access_pattern,
                                                          uint32_t data_dimensions,
                                                          const uint32_t* dimension_sizes,
                                                          const uint32_t* current_order,
                                                          uint32_t* optimized_order,
                                                          float* estimated_improvement) {
    if (!optimizer || !dimension_sizes || !current_order || !optimized_order || 
        !estimated_improvement || data_dimensions == 0 || data_dimensions > 8) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    *estimated_improvement = 0.0f;
    
    // Copy current order as starting point
    for (uint32_t i = 0; i < data_dimensions; i++) {
        optimized_order[i] = current_order[i];
    }
    
    // Optimize based on access pattern
    switch (access_pattern) {
        case ACCESS_PATTERN_SEQUENTIAL:
        case ACCESS_PATTERN_REVERSE_SEQUENTIAL:
            // For sequential patterns, optimize for spatial locality
            // Innermost loop should access contiguous memory
            if (data_dimensions >= 2) {
                // Swap dimensions to put smallest stride first
                uint32_t min_dim = 0;
                uint32_t max_dim = 0;
                for (uint32_t i = 1; i < data_dimensions; i++) {
                    if (dimension_sizes[i] < dimension_sizes[min_dim]) {
                        min_dim = i;
                    }
                    if (dimension_sizes[i] > dimension_sizes[max_dim]) {
                        max_dim = i;
                    }
                }
                optimized_order[0] = min_dim; // Innermost
                optimized_order[data_dimensions - 1] = max_dim; // Outermost
                *estimated_improvement = 0.2f;
            }
            break;
            
        case ACCESS_PATTERN_STRIDED:
            // For strided patterns, optimize for regular access
            // Order dimensions by stride alignment with cache lines
            if (data_dimensions >= 2) {
                // Simple heuristic: reverse order for strided access
                for (uint32_t i = 0; i < data_dimensions; i++) {
                    optimized_order[i] = data_dimensions - 1 - i;
                }
                *estimated_improvement = 0.15f;
            }
            break;
            
        case ACCESS_PATTERN_BLOCKED:
            // For blocked patterns, optimize for temporal locality
            // Order by block-friendly dimensions
            if (data_dimensions >= 2) {
                // Prioritize dimensions that fit well in cache blocks
                size_t cache_line_size = optimizer->cache_info.cache_line_size_bytes;
                for (uint32_t i = 0; i < data_dimensions; i++) {
                    uint32_t best_dim = i;
                    uint32_t best_fit = dimension_sizes[i] % (cache_line_size / 4); // Assume 4-byte elements
                    
                    for (uint32_t j = i + 1; j < data_dimensions; j++) {
                        uint32_t fit = dimension_sizes[j] % (cache_line_size / 4);
                        if (fit < best_fit) {
                            best_fit = fit;
                            best_dim = j;
                        }
                    }
                    
                    if (best_dim != i) {
                        // Swap
                        uint32_t temp = optimized_order[i];
                        optimized_order[i] = optimized_order[best_dim];
                        optimized_order[best_dim] = temp;
                    }
                }
                *estimated_improvement = 0.25f;
            }
            break;
            
        case ACCESS_PATTERN_RANDOM:
        case ACCESS_PATTERN_MIXED:
            // For random/mixed patterns, optimize for cache utilization
            // Keep current order but estimate minimal improvement
            *estimated_improvement = 0.05f;
            break;
            
        default:
            break;
    }
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_generate_tiling_params(CacheOptimizer* optimizer,
                                                          uint32_t loop_nest_depth,
                                                          const uint32_t* loop_bounds,
                                                          const size_t* data_access_strides,
                                                          uint32_t* tile_sizes,
                                                          CacheOptimizationStrategy* tiling_strategy) {
    if (!optimizer || !loop_bounds || !data_access_strides || !tile_sizes || 
        !tiling_strategy || loop_nest_depth == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    size_t l1_capacity = optimizer->cache_info.l1_cache_size_kb * 1024;
    size_t cache_line_size = optimizer->cache_info.cache_line_size_bytes;
    
    // Calculate working set size for current loop bounds
    size_t working_set_size = 1;
    for (uint32_t i = 0; i < loop_nest_depth; i++) {
        working_set_size += loop_bounds[i] * data_access_strides[i];
    }
    
    // Determine tiling strategy based on working set size
    if (working_set_size <= l1_capacity / 4) {
        *tiling_strategy = CACHE_STRATEGY_TEMPORAL_LOCALITY;
        // Small working set: no tiling needed
        for (uint32_t i = 0; i < loop_nest_depth; i++) {
            tile_sizes[i] = loop_bounds[i];
        }
    } else if (working_set_size <= l1_capacity) {
        *tiling_strategy = CACHE_STRATEGY_BLOCKED;
        // Medium working set: tile for L1
        size_t target_tile_size = l1_capacity / (loop_nest_depth * 2);
        for (uint32_t i = 0; i < loop_nest_depth; i++) {
            uint32_t max_tile = (uint32_t)(target_tile_size / data_access_strides[i]);
            tile_sizes[i] = fminf(loop_bounds[i], fmaxf(1, max_tile));
            
            // Align to cache line boundaries for innermost dimension
            if (i == loop_nest_depth - 1) {
                uint32_t elements_per_line = (uint32_t)(cache_line_size / data_access_strides[i]);
                if (elements_per_line > 1) {
                    tile_sizes[i] = ((tile_sizes[i] + elements_per_line - 1) / 
                                    elements_per_line) * elements_per_line;
                    tile_sizes[i] = fminf(tile_sizes[i], loop_bounds[i]);
                }
            }
        }
    } else {
        *tiling_strategy = CACHE_STRATEGY_STREAMING;
        // Large working set: use streaming tiles
        size_t l2_capacity = optimizer->cache_info.l2_cache_size_kb * 1024;
        size_t target_tile_size = l2_capacity / (loop_nest_depth * 4);
        for (uint32_t i = 0; i < loop_nest_depth; i++) {
            uint32_t max_tile = (uint32_t)(target_tile_size / data_access_strides[i]);
            tile_sizes[i] = fminf(loop_bounds[i], fmaxf(32, max_tile)); // Minimum 32 elements
        }
    }
    
    return CACHE_OPTIMIZER_SUCCESS;
}

// Prefetch Optimization

CacheOptimizerError cache_optimizer_configure_prefetch(CacheOptimizer* optimizer,
                                                       const PrefetchStrategy* prefetch_config) {
    if (!optimizer || !prefetch_config) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    optimizer->prefetch_strategy = *prefetch_config;
    
    // Validate and adjust prefetch parameters for Apple Silicon
    if (optimizer->prefetch_strategy.max_prefetch_distance > 16) {
        optimizer->prefetch_strategy.max_prefetch_distance = 16;
    }
    if (optimizer->prefetch_strategy.prefetch_degree > 4) {
        optimizer->prefetch_strategy.prefetch_degree = 4;
    }
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_generate_prefetch_hints(CacheOptimizer* optimizer,
                                                           const void* current_address,
                                                           MemoryAccessPattern access_pattern,
                                                           const void** prefetch_addresses,
                                                           uint32_t prefetch_count,
                                                           uint32_t* actual_prefetch_count) {
    if (!optimizer || !current_address || !prefetch_addresses || 
        !actual_prefetch_count || prefetch_count == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    *actual_prefetch_count = 0;
    size_t cache_line_size = optimizer->cache_info.cache_line_size_bytes;
    uint32_t distance = optimizer->prefetch_strategy.base_prefetch_distance;
    
    // Adjust prefetch distance based on access pattern
    switch (access_pattern) {
        case ACCESS_PATTERN_SEQUENTIAL:
            distance = optimizer->prefetch_strategy.base_prefetch_distance * 2;
            break;
        case ACCESS_PATTERN_STRIDED:
            distance = optimizer->prefetch_strategy.base_prefetch_distance;
            break;
        case ACCESS_PATTERN_RANDOM:
            distance = 1; // Minimal prefetching for random access
            break;
        default:
            break;
    }
    
    distance = fminf(distance, optimizer->prefetch_strategy.max_prefetch_distance);
    
    // Generate prefetch addresses
    for (uint32_t i = 0; i < fminf(prefetch_count, distance); i++) {
        const char* prefetch_addr = (const char*)current_address + 
                                   (i + 1) * cache_line_size;
        prefetch_addresses[*actual_prefetch_count] = prefetch_addr;
        (*actual_prefetch_count)++;
    }
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_issue_prefetch(CacheOptimizer* optimizer,
                                                   const void* const* prefetch_addresses,
                                                   uint32_t prefetch_count,
                                                   bool prefetch_for_write) {
    if (!optimizer || !prefetch_addresses || prefetch_count == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    if (!optimizer->prefetch_strategy.enable_software_prefetch) {
        return CACHE_OPTIMIZER_SUCCESS; // Silently skip if disabled
    }
    
    // Issue software prefetch instructions
    for (uint32_t i = 0; i < prefetch_count; i++) {
        if (prefetch_addresses[i]) {
            if (prefetch_for_write) {
                software_prefetch_write(prefetch_addresses[i]);
            } else {
                software_prefetch_read(prefetch_addresses[i]);
            }
        }
    }
    
    // Update metrics
    optimizer->metrics.prefetch_hits += prefetch_count; // Assume all succeed
    
    return CACHE_OPTIMIZER_SUCCESS;
}

// Working Set Analysis

CacheOptimizerError cache_optimizer_analyze_working_set(CacheOptimizer* optimizer,
                                                        const void* const* data_addresses,
                                                        const size_t* data_sizes,
                                                        const uint32_t* access_frequencies,
                                                        size_t data_count,
                                                        WorkingSetAnalysis* analysis_result) {
    if (!optimizer || !data_addresses || !data_sizes || !access_frequencies || 
        !analysis_result || data_count == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    memset(analysis_result, 0, sizeof(WorkingSetAnalysis));
    
    size_t total_working_set = 0;
    size_t hot_data_size = 0;
    size_t cold_data_size = 0;
    uint32_t total_accesses = 0;
    uint32_t hot_threshold = 0;
    
    // Calculate total accesses and determine hot/cold threshold
    for (size_t i = 0; i < data_count; i++) {
        total_accesses += access_frequencies[i];
        total_working_set += data_sizes[i];
    }
    
    hot_threshold = total_accesses / (data_count * 4); // Top 25% are considered hot
    
    // Analyze hot vs cold data
    for (size_t i = 0; i < data_count; i++) {
        if (access_frequencies[i] >= hot_threshold) {
            hot_data_size += data_sizes[i];
            analysis_result->hot_cache_lines += 
                (uint32_t)((data_sizes[i] + optimizer->cache_info.cache_line_size_bytes - 1) / 
                          optimizer->cache_info.cache_line_size_bytes);
        } else {
            cold_data_size += data_sizes[i];
            analysis_result->cold_cache_lines += 
                (uint32_t)((data_sizes[i] + optimizer->cache_info.cache_line_size_bytes - 1) / 
                          optimizer->cache_info.cache_line_size_bytes);
        }
    }
    
    analysis_result->working_set_size_kb = total_working_set / 1024;
    analysis_result->peak_working_set_kb = analysis_result->working_set_size_kb;
    analysis_result->active_data_blocks = (uint32_t)data_count;
    
    // Calculate locality factor based on hot/cold ratio
    if (total_working_set > 0) {
        analysis_result->working_set_locality = (float)hot_data_size / total_working_set;
    }
    
    // Determine cache pressure
    size_t l1_capacity = optimizer->cache_info.l1_cache_size_kb * 1024;
    size_t l2_capacity = optimizer->cache_info.l2_cache_size_kb * 1024;
    
    analysis_result->exceeds_l1_capacity = (total_working_set > l1_capacity);
    analysis_result->exceeds_l2_capacity = (total_working_set > l2_capacity);
    
    if (analysis_result->exceeds_l2_capacity) {
        analysis_result->cache_pressure = 1.0f;
    } else if (analysis_result->exceeds_l1_capacity) {
        analysis_result->cache_pressure = (float)total_working_set / l2_capacity;
    } else {
        analysis_result->cache_pressure = (float)total_working_set / l1_capacity;
    }
    
    // Recommend block size based on working set characteristics
    if (total_working_set <= l1_capacity / 2) {
        analysis_result->recommended_block_size = 16 * 1024; // 16KB blocks
    } else if (total_working_set <= l2_capacity / 2) {
        analysis_result->recommended_block_size = 64 * 1024; // 64KB blocks
    } else {
        analysis_result->recommended_block_size = 256 * 1024; // 256KB blocks
    }
    
    // Update optimizer state
    optimizer->working_set = *analysis_result;
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_optimize_working_set(CacheOptimizer* optimizer,
                                                         const WorkingSetAnalysis* current_working_set,
                                                         CacheOptimizationStrategy optimization_target,
                                                         CacheOptimizationResult* optimization_result) {
    if (!optimizer || !current_working_set || !optimization_result) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    memset(optimization_result, 0, sizeof(CacheOptimizationResult));
    optimization_result->selected_strategy = optimization_target;
    
    size_t l1_capacity = optimizer->cache_info.l1_cache_size_kb * 1024;
    size_t l2_capacity = optimizer->cache_info.l2_cache_size_kb * 1024;
    size_t working_set_size = current_working_set->working_set_size_kb * 1024;
    
    switch (optimization_target) {
        case CACHE_STRATEGY_TEMPORAL_LOCALITY:
            // Optimize for L1 cache fit
            if (working_set_size > l1_capacity) {
                optimization_result->optimal_block_size_bytes = 
                    (uint32_t)(l1_capacity / current_working_set->active_data_blocks);
                optimization_result->requires_data_reorganization = true;
                optimization_result->estimated_performance_gain = 0.4f;
            } else {
                optimization_result->optimal_block_size_bytes = 16 * 1024;
                optimization_result->estimated_performance_gain = 0.2f;
            }
            break;
            
        case CACHE_STRATEGY_SPATIAL_LOCALITY:
            // Optimize for spatial access patterns
            optimization_result->optimal_block_size_bytes = 32 * 1024;
            optimization_result->optimal_prefetch_distance = 4;
            optimization_result->estimated_performance_gain = 0.3f;
            break;
            
        case CACHE_STRATEGY_BLOCKED:
            // Use blocking to fit in L2
            if (working_set_size > l2_capacity) {
                optimization_result->optimal_block_size_bytes = 
                    (uint32_t)(l2_capacity / (current_working_set->active_data_blocks * 2));
            } else {
                optimization_result->optimal_block_size_bytes = 64 * 1024;
            }
            optimization_result->requires_data_reorganization = true;
            optimization_result->estimated_performance_gain = 0.5f;
            break;
            
        case CACHE_STRATEGY_STREAMING:
            // Use streaming for large working sets
            optimization_result->optimal_block_size_bytes = 256 * 1024;
            optimization_result->optimal_prefetch_distance = 8;
            optimization_result->estimated_performance_gain = 0.25f;
            break;
            
        default:
            optimization_result->optimal_block_size_bytes = 
                current_working_set->recommended_block_size;
            optimization_result->estimated_performance_gain = 0.15f;
            break;
    }
    
    // Calculate memory overhead
    optimization_result->memory_layout_overhead_bytes = 
        optimization_result->optimal_block_size_bytes * 
        current_working_set->active_data_blocks * 0.1f; // 10% overhead estimate
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_minimize_working_set(CacheOptimizer* optimizer,
                                                         void* const* working_set_data,
                                                         const size_t* working_set_sizes,
                                                         const float* working_set_priorities,
                                                         size_t data_count,
                                                         size_t target_size_kb,
                                                         void*** minimized_set,
                                                         float* compression_ratio) {
    if (!optimizer || !working_set_data || !working_set_sizes || !working_set_priorities ||
        !minimized_set || !compression_ratio || data_count == 0 || target_size_kb == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    // Calculate current working set size
    size_t current_size = 0;
    for (size_t i = 0; i < data_count; i++) {
        current_size += working_set_sizes[i];
    }
    
    size_t target_size_bytes = target_size_kb * 1024;
    
    if (current_size <= target_size_bytes) {
        // Already within target, return original set
        *minimized_set = working_set_data;
        *compression_ratio = 1.0f;
        return CACHE_OPTIMIZER_SUCCESS;
    }
    
    // Create priority-sorted indices
    typedef struct {
        size_t index;
        float priority;
        size_t size;
    } PriorityItem;
    
    PriorityItem* items = (PriorityItem*)malloc(data_count * sizeof(PriorityItem));
    if (!items) {
        return CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION;
    }
    
    for (size_t i = 0; i < data_count; i++) {
        items[i].index = i;
        items[i].priority = working_set_priorities[i];
        items[i].size = working_set_sizes[i];
    }
    
    // Sort by priority (descending)
    for (size_t i = 0; i < data_count - 1; i++) {
        for (size_t j = i + 1; j < data_count; j++) {
            if (items[j].priority > items[i].priority) {
                PriorityItem temp = items[i];
                items[i] = items[j];
                items[j] = temp;
            }
        }
    }
    
    // Select items that fit within target size
    void** minimized = (void**)malloc(data_count * sizeof(void*));
    if (!minimized) {
        free(items);
        return CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION;
    }
    
    size_t selected_size = 0;
    size_t selected_count = 0;
    
    for (size_t i = 0; i < data_count && selected_size < target_size_bytes; i++) {
        if (selected_size + items[i].size <= target_size_bytes) {
            minimized[selected_count] = working_set_data[items[i].index];
            selected_size += items[i].size;
            selected_count++;
        }
    }
    
    // Reallocate to exact size
    void** final_set = (void**)realloc(minimized, selected_count * sizeof(void*));
    if (!final_set && selected_count > 0) {
        free(minimized);
        free(items);
        return CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION;
    }
    
    *minimized_set = final_set;
    *compression_ratio = (float)selected_size / current_size;
    
    free(items);
    return CACHE_OPTIMIZER_SUCCESS;
}

// Performance Monitoring

CacheOptimizerError cache_optimizer_start_monitoring(CacheOptimizer* optimizer,
                                                     uint32_t monitoring_interval_ms) {
    if (!optimizer || monitoring_interval_ms == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    if (optimizer->monitoring_active) {
        return CACHE_OPTIMIZER_SUCCESS; // Already monitoring
    }
    
    optimizer->monitoring_active = true;
    optimizer->monitoring_start_time = get_current_time_microseconds();
    optimizer->monitoring_interval_ms = monitoring_interval_ms;
    
    // Reset counters for fresh monitoring session
    memset(&optimizer->metrics, 0, sizeof(CachePerformanceMetrics));
    optimizer->metrics.cache_efficiency_score = 0.5f;
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_get_performance_metrics(CacheOptimizer* optimizer,
                                                           CachePerformanceMetrics* metrics) {
    if (!optimizer || !metrics) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    *metrics = optimizer->metrics;
    
    // Calculate derived metrics if monitoring is active
    if (optimizer->monitoring_active) {
        uint64_t current_time = get_current_time_microseconds();
        uint64_t monitoring_duration = current_time - optimizer->monitoring_start_time;
        
        if (optimizer->metrics.total_memory_accesses > 0) {
            metrics->l1_hit_rate = (float)optimizer->metrics.l1_cache_hits / 
                                  optimizer->metrics.total_memory_accesses;
            metrics->l2_hit_rate = (float)optimizer->metrics.l2_cache_hits / 
                                  optimizer->metrics.total_memory_accesses;
            metrics->overall_hit_rate = (metrics->l1_hit_rate + metrics->l2_hit_rate);
            
            // Calculate cache efficiency score
            if (metrics->overall_hit_rate >= CACHE_HIT_RATE_EXCELLENT) {
                metrics->cache_efficiency_score = 1.0f;
            } else if (metrics->overall_hit_rate >= CACHE_HIT_RATE_GOOD) {
                metrics->cache_efficiency_score = 0.8f;
            } else if (metrics->overall_hit_rate >= CACHE_HIT_RATE_ACCEPTABLE) {
                metrics->cache_efficiency_score = 0.6f;
            } else {
                metrics->cache_efficiency_score = 0.4f;
            }
        }
        
        // Estimate cycles saved by optimization
        float cache_miss_penalty = 200.0f; // Estimated cycles for memory access
        uint64_t cache_misses = optimizer->metrics.l1_cache_misses + 
                               optimizer->metrics.l2_cache_misses;
        float hit_rate_improvement = fmaxf(0.0f, metrics->overall_hit_rate - 0.7f);
        metrics->cycles_saved_by_optimization = 
            (uint64_t)(hit_rate_improvement * cache_misses * cache_miss_penalty);
    }
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_stop_monitoring(CacheOptimizer* optimizer,
                                                    CachePerformanceMetrics* final_metrics) {
    if (!optimizer) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    if (!optimizer->monitoring_active) {
        return CACHE_OPTIMIZER_SUCCESS; // Not monitoring
    }
    
    // Get final metrics
    if (final_metrics) {
        cache_optimizer_get_performance_metrics(optimizer, final_metrics);
    }
    
    optimizer->monitoring_active = false;
    return CACHE_OPTIMIZER_SUCCESS;
}

void cache_optimizer_reset_counters(CacheOptimizer* optimizer) {
    if (!optimizer) {
        return;
    }
    
    memset(&optimizer->metrics, 0, sizeof(CachePerformanceMetrics));
    optimizer->metrics.cache_efficiency_score = 0.5f;
    optimizer->total_cache_accesses = 0;
    optimizer->successful_optimizations = 0;
    optimizer->failed_optimizations = 0;
    
    // Reset access tracking
    for (size_t i = 0; i < optimizer->current_tracked_count; i++) {
        optimizer->access_counts[i] = 0;
        optimizer->last_access_times[i] = get_current_time_microseconds();
    }
}

// Neural Network Specific Optimizations

CacheOptimizerError cache_optimizer_optimize_attention_cache(CacheOptimizer* optimizer,
                                                            uint32_t sequence_length,
                                                            uint32_t d_model,
                                                            uint32_t num_heads,
                                                            CacheOptimizationResult* attention_optimization) {
    if (!optimizer || !attention_optimization || sequence_length == 0 || 
        d_model == 0 || num_heads == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    memset(attention_optimization, 0, sizeof(CacheOptimizationResult));
    
    // Calculate attention memory requirements
    size_t head_dimension = d_model / num_heads;
    size_t qkv_size = sequence_length * d_model * sizeof(float); // Q, K, V matrices
    size_t attention_weights_size = num_heads * sequence_length * sequence_length * sizeof(float);
    size_t total_attention_memory = qkv_size * 3 + attention_weights_size;
    
    size_t l1_capacity = optimizer->cache_info.l1_cache_size_kb * 1024;
    size_t l2_capacity = optimizer->cache_info.l2_cache_size_kb * 1024;
    size_t cache_line_size = optimizer->cache_info.cache_line_size_bytes;
    
    if (total_attention_memory <= l1_capacity / 2) {
        // Small attention: optimize for L1
        attention_optimization->selected_strategy = CACHE_STRATEGY_TEMPORAL_LOCALITY;
        attention_optimization->optimal_block_size_bytes = sequence_length * head_dimension * sizeof(float);
        attention_optimization->estimated_performance_gain = 0.3f;
    } else if (total_attention_memory <= l2_capacity / 2) {
        // Medium attention: use blocking
        attention_optimization->selected_strategy = CACHE_STRATEGY_BLOCKED;
        uint32_t block_size = (uint32_t)sqrtf(l1_capacity / (num_heads * sizeof(float)));
        attention_optimization->optimal_block_size_bytes = block_size * block_size * sizeof(float);
        attention_optimization->estimated_performance_gain = 0.4f;
    } else {
        // Large attention: use streaming with Flash Attention pattern
        attention_optimization->selected_strategy = CACHE_STRATEGY_STREAMING;
        attention_optimization->optimal_block_size_bytes = 64 * 1024; // 64KB blocks
        attention_optimization->estimated_performance_gain = 0.5f;
    }
    
    // Configure prefetching for attention pattern
    attention_optimization->optimal_prefetch_distance = 4;
    
    // Recommend memory layout
    attention_optimization->cache_aligned_size_bytes = 
        align_size_to_cache_line(total_attention_memory, cache_line_size);
    attention_optimization->memory_layout_overhead_bytes = 
        attention_optimization->cache_aligned_size_bytes - total_attention_memory;
    
    // For attention, reorganization can help significantly
    attention_optimization->requires_data_reorganization = (sequence_length > 512);
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_optimize_ffn_cache(CacheOptimizer* optimizer,
                                                       uint32_t batch_size,
                                                       uint32_t input_dimension,
                                                       uint32_t hidden_dimension,
                                                       CacheOptimizationResult* ffn_optimization) {
    if (!optimizer || !ffn_optimization || batch_size == 0 || 
        input_dimension == 0 || hidden_dimension == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    memset(ffn_optimization, 0, sizeof(CacheOptimizationResult));
    
    // Calculate FFN memory requirements
    size_t input_size = batch_size * input_dimension * sizeof(float);
    size_t weight_size = input_dimension * hidden_dimension * sizeof(float);
    size_t output_size = batch_size * hidden_dimension * sizeof(float);
    size_t total_ffn_memory = input_size + weight_size + output_size;
    
    size_t l2_capacity = optimizer->cache_info.l2_cache_size_kb * 1024;
    size_t cache_line_size = optimizer->cache_info.cache_line_size_bytes;
    
    if (total_ffn_memory <= l2_capacity / 4) {
        // Small FFN: optimize for cache reuse
        ffn_optimization->selected_strategy = CACHE_STRATEGY_TEMPORAL_LOCALITY;
        ffn_optimization->optimal_block_size_bytes = 32 * 1024;
        ffn_optimization->estimated_performance_gain = 0.25f;
    } else {
        // Large FFN: use blocking for matrix multiplication
        ffn_optimization->selected_strategy = CACHE_STRATEGY_BLOCKED;
        // Optimize for cache-friendly matrix multiplication
        uint32_t block_dim = (uint32_t)sqrtf(l2_capacity / (4 * sizeof(float)));
        ffn_optimization->optimal_block_size_bytes = block_dim * block_dim * sizeof(float);
        ffn_optimization->estimated_performance_gain = 0.4f;
    }
    
    // FFN benefits from spatial prefetching
    ffn_optimization->optimal_prefetch_distance = 2;
    
    // Cache alignment is important for matrix operations
    ffn_optimization->cache_aligned_size_bytes = 
        align_size_to_cache_line(total_ffn_memory, cache_line_size);
    ffn_optimization->memory_layout_overhead_bytes = 
        ffn_optimization->cache_aligned_size_bytes - total_ffn_memory;
    
    // Data reorganization helps for large matrices
    ffn_optimization->requires_data_reorganization = (hidden_dimension > 1024);
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_optimize_layer_norm_cache(CacheOptimizer* optimizer,
                                                             uint32_t vector_length,
                                                             uint32_t batch_size,
                                                             CacheOptimizationResult* layer_norm_optimization) {
    if (!optimizer || !layer_norm_optimization || vector_length == 0 || batch_size == 0) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    memset(layer_norm_optimization, 0, sizeof(CacheOptimizationResult));
    
    // Calculate layer normalization memory requirements
    size_t vector_size = vector_length * sizeof(float);
    size_t weights_size = vector_length * 2 * sizeof(float); // gamma + beta
    size_t total_norm_memory = batch_size * vector_size + weights_size;
    
    size_t l1_capacity = optimizer->cache_info.l1_cache_size_kb * 1024;
    size_t cache_line_size = optimizer->cache_info.cache_line_size_bytes;
    
    if (total_norm_memory <= l1_capacity) {
        // Layer norm fits in L1: optimize for temporal locality
        layer_norm_optimization->selected_strategy = CACHE_STRATEGY_TEMPORAL_LOCALITY;
        layer_norm_optimization->optimal_block_size_bytes = vector_size;
        layer_norm_optimization->estimated_performance_gain = 0.2f;
    } else {
        // Large layer norm: process in cache-friendly chunks
        layer_norm_optimization->selected_strategy = CACHE_STRATEGY_SEQUENTIAL;
        uint32_t elements_per_cache_line = cache_line_size / sizeof(float);
        layer_norm_optimization->optimal_block_size_bytes = 
            elements_per_cache_line * cache_line_size;
        layer_norm_optimization->estimated_performance_gain = 0.15f;
    }
    
    // Layer norm benefits from sequential prefetching
    layer_norm_optimization->optimal_prefetch_distance = 1;
    
    // Cache alignment for vector operations
    layer_norm_optimization->cache_aligned_size_bytes = 
        align_size_to_cache_line(total_norm_memory, cache_line_size);
    layer_norm_optimization->memory_layout_overhead_bytes = 
        layer_norm_optimization->cache_aligned_size_bytes - total_norm_memory;
    
    // Layer norm rarely needs reorganization (already vector-friendly)
    layer_norm_optimization->requires_data_reorganization = false;
    
    return CACHE_OPTIMIZER_SUCCESS;
}

// Configuration Functions

CacheOptimizerError cache_optimizer_create_default_config(CacheOptimizationConfig* config) {
    if (!config) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    memset(config, 0, sizeof(CacheOptimizationConfig));
    
    config->enable_cache_alignment = true;
    config->enable_prefetch_optimization = true;
    config->enable_working_set_analysis = true;
    config->enable_access_pattern_detection = true;
    config->enable_temporal_blocking = true;
    config->enable_spatial_blocking = true;
    config->enable_streaming_optimization = false; // Conservative default
    config->force_cache_friendly_layout = false; // Conservative default
    config->block_size_bytes = CACHE_OPTIMIZER_DEFAULT_BLOCK_SIZE_KB * 1024;
    config->prefetch_distance_lines = 2;
    config->working_set_threshold_kb = CACHE_OPTIMIZER_WORKING_SET_THRESHOLD_KB;
    config->cache_miss_penalty_factor = 3.0f;
    config->optimization_level = 1; // Moderate optimization
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_create_apple_silicon_config(CacheOptimizationConfig* config) {
    if (!config) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    cache_optimizer_create_default_config(config);
    
    // Apple Silicon specific optimizations
    config->enable_streaming_optimization = true;
    config->force_cache_friendly_layout = true;
    config->prefetch_distance_lines = 4; // Aggressive prefetching
    config->working_set_threshold_kb = 96; // Based on L1 capacity
    config->optimization_level = 2; // Aggressive optimization
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_create_aggressive_config(CacheOptimizationConfig* config) {
    if (!config) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    cache_optimizer_create_apple_silicon_config(config);
    
    // Maximum optimization settings
    config->force_cache_friendly_layout = true;
    config->prefetch_distance_lines = 8;
    config->optimization_level = 3;
    config->cache_miss_penalty_factor = 5.0f; // Higher penalty weight
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_validate_config(const CacheOptimizationConfig* config,
                                                    const AppleSiliconCacheInfo* cache_info,
                                                    bool* is_valid) {
    if (!config || !cache_info || !is_valid) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    *is_valid = true;
    
    // Validate configuration parameters
    if (config->block_size_bytes == 0 || config->block_size_bytes > 1024 * 1024) {
        *is_valid = false; // Block size too small or too large
    }
    if (config->prefetch_distance_lines > 16) {
        *is_valid = false; // Prefetch distance too large
    }
    if (config->working_set_threshold_kb > cache_info->l2_cache_size_kb) {
        *is_valid = false; // Working set threshold larger than L2
    }
    if (config->cache_miss_penalty_factor <= 0.0f) {
        *is_valid = false; // Invalid penalty factor
    }
    if (config->optimization_level > 3) {
        *is_valid = false; // Invalid optimization level
    }
    
    return CACHE_OPTIMIZER_SUCCESS;
}

CacheOptimizerError cache_optimizer_estimate_benefits(CacheOptimizer* optimizer,
                                                      const DataLayoutParams* layout_params,
                                                      const CachePerformanceMetrics* current_performance,
                                                      float* estimated_improvement) {
    if (!optimizer || !layout_params || !estimated_improvement) {
        return CACHE_OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    *estimated_improvement = 0.0f;
    
    // Analyze potential for improvement based on access patterns and data size
    float pattern_improvement = 0.0f;
    switch (layout_params->primary_pattern) {
        case ACCESS_PATTERN_SEQUENTIAL:
            pattern_improvement = 0.3f;
            break;
        case ACCESS_PATTERN_STRIDED:
            pattern_improvement = 0.25f;
            break;
        case ACCESS_PATTERN_BLOCKED:
            pattern_improvement = 0.4f;
            break;
        case ACCESS_PATTERN_RANDOM:
            pattern_improvement = 0.1f;
            break;
        default:
            pattern_improvement = 0.15f;
            break;
    }
    
    // Adjust based on locality factors
    float locality_factor = (layout_params->spatial_locality_factor + 
                           layout_params->temporal_locality_factor) / 2.0f;
    pattern_improvement *= (1.0f - locality_factor); // More improvement if low locality
    
    // Adjust based on current cache performance
    if (current_performance) {
        float current_hit_rate = current_performance->overall_hit_rate;
        if (current_hit_rate < CACHE_HIT_RATE_ACCEPTABLE) {
            pattern_improvement *= 1.5f; // More room for improvement
        } else if (current_hit_rate > CACHE_HIT_RATE_EXCELLENT) {
            pattern_improvement *= 0.5f; // Less room for improvement
        }
    }
    
    // Consider data size impact
    size_t l1_capacity = optimizer->cache_info.l1_cache_size_kb * 1024;
    size_t l2_capacity = optimizer->cache_info.l2_cache_size_kb * 1024;
    
    if (layout_params->data_size_bytes > l2_capacity) {
        pattern_improvement *= 1.2f; // Large data benefits more from optimization
    } else if (layout_params->data_size_bytes < l1_capacity / 4) {
        pattern_improvement *= 0.8f; // Small data has less optimization potential
    }
    
    *estimated_improvement = fminf(0.8f, pattern_improvement); // Cap at 80% improvement
    
    return CACHE_OPTIMIZER_SUCCESS;
}

void cache_optimizer_destroy(CacheOptimizer* optimizer) {
    if (!optimizer) {
        return;
    }
    
    // Free tracking arrays
    free(optimizer->tracked_addresses);
    free(optimizer->tracked_sizes);
    free(optimizer->access_counts);
    free(optimizer->last_access_times);
    
    // Free memory pool
    free(optimizer->aligned_memory_pool);
    
    // Free main structure
    free(optimizer);
}

// Utility Functions

const char* cache_optimizer_get_error_string(CacheOptimizerError error_code) {
    switch (error_code) {
        case CACHE_OPTIMIZER_SUCCESS:
            return "Operation completed successfully";
        case CACHE_OPTIMIZER_ERROR_INVALID_PARAM:
            return "Invalid parameter provided";
        case CACHE_OPTIMIZER_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case CACHE_OPTIMIZER_ERROR_UNSUPPORTED_PATTERN:
            return "Unsupported access pattern";
        case CACHE_OPTIMIZER_ERROR_INSUFFICIENT_MEMORY:
            return "Insufficient memory for optimization";
        case CACHE_OPTIMIZER_ERROR_CACHE_INFO_UNAVAILABLE:
            return "Cache information unavailable";
        case CACHE_OPTIMIZER_ERROR_OPTIMIZATION_FAILED:
            return "Cache optimization failed";
        case CACHE_OPTIMIZER_ERROR_ANALYSIS_FAILED:
            return "Cache analysis failed";
        case CACHE_OPTIMIZER_ERROR_HARDWARE_INCOMPATIBLE:
            return "Hardware incompatible with optimization";
        default:
            return "Unknown error";
    }
}

const char* cache_optimizer_get_strategy_string(CacheOptimizationStrategy strategy) {
    switch (strategy) {
        case CACHE_STRATEGY_SEQUENTIAL:
            return "Sequential Access Optimization";
        case CACHE_STRATEGY_BLOCKED:
            return "Block/Tile-based Optimization";
        case CACHE_STRATEGY_STREAMING:
            return "Streaming Optimization";
        case CACHE_STRATEGY_TEMPORAL_LOCALITY:
            return "Temporal Locality Optimization";
        case CACHE_STRATEGY_SPATIAL_LOCALITY:
            return "Spatial Locality Optimization";
        case CACHE_STRATEGY_ADAPTIVE:
            return "Adaptive Strategy Selection";
        case CACHE_STRATEGY_NEURAL_SPECIFIC:
            return "Neural Network Specific Optimization";
        default:
            return "Unknown Strategy";
    }
}

const char* cache_optimizer_get_access_pattern_string(MemoryAccessPattern pattern) {
    switch (pattern) {
        case ACCESS_PATTERN_SEQUENTIAL:
            return "Sequential Forward";
        case ACCESS_PATTERN_REVERSE_SEQUENTIAL:
            return "Sequential Reverse";
        case ACCESS_PATTERN_STRIDED:
            return "Strided Access";
        case ACCESS_PATTERN_RANDOM:
            return "Random Access";
        case ACCESS_PATTERN_BLOCKED:
            return "Block-wise Access";
        case ACCESS_PATTERN_SPARSE:
            return "Sparse Matrix Access";
        case ACCESS_PATTERN_GATHER_SCATTER:
            return "Gather/Scatter Operations";
        case ACCESS_PATTERN_MIXED:
            return "Mixed Access Patterns";
        default:
            return "Unknown Pattern";
    }
}

size_t cache_optimizer_calculate_optimal_alignment(size_t data_size, size_t cache_line_size) {
    if (data_size <= cache_line_size) {
        return cache_line_size;
    } else if (data_size <= cache_line_size * 2) {
        return cache_line_size;
    } else if (data_size <= cache_line_size * 4) {
        return cache_line_size * 2;
    } else {
        return cache_line_size * 4; // Maximum alignment
    }
}

size_t cache_optimizer_calculate_aligned_size(size_t requested_size, size_t cache_line_size) {
    return ((requested_size + cache_line_size - 1) / cache_line_size) * cache_line_size;
}

bool cache_optimizer_is_cache_aligned(const void* address, size_t cache_line_size) {
    return ((uintptr_t)address % cache_line_size) == 0;
}
