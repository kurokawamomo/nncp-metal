#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#ifdef USE_METAL

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "metal_context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Memory pool types for different usage patterns
typedef enum {
    MM_POOL_SMALL = 0,      // < 1MB, frequent allocations
    MM_POOL_MEDIUM = 1,     // 1MB - 16MB, model weights
    MM_POOL_LARGE = 2,      // > 16MB, large tensors
    MM_POOL_TRANSIENT = 3,  // Temporary computation buffers
    MM_POOL_COUNT = 4
} MMPoolType;

// Memory allocation strategy
typedef enum {
    MM_STRATEGY_UNIFIED = 0,    // Use unified memory (Apple Silicon)
    MM_STRATEGY_DISCRETE = 1,   // Separate CPU/GPU memory
    MM_STRATEGY_AUTO = 2        // Auto-select based on size and usage
} MMStrategy;

// Buffer access pattern hints
typedef enum {
    MM_ACCESS_READ_ONLY = 0,    // Data written once, read many times
    MM_ACCESS_WRITE_ONLY = 1,   // Data written frequently, rarely read
    MM_ACCESS_READ_WRITE = 2,   // Data accessed frequently from both CPU/GPU
    MM_ACCESS_STREAMING = 3     // Large sequential access pattern
} MMAccessPattern;

// Enhanced buffer structure
typedef struct MMBuffer {
    // Basic buffer info
    void* cpu_ptr;              // CPU-accessible pointer
    void* gpu_handle;           // Metal buffer handle (opaque)
    size_t size;                // Buffer size in bytes
    size_t alignment;           // Memory alignment requirements
    
    // Memory management info
    MMPoolType pool_type;       // Which pool this buffer belongs to
    MMStrategy strategy;        // Memory allocation strategy used
    MMAccessPattern access_pattern; // Access pattern hint
    bool is_unified;            // Uses unified memory
    bool is_pooled;             // Allocated from pool vs direct
    
    // Reference counting and lifecycle
    uint32_t ref_count;         // Reference counter
    uint64_t alloc_id;          // Unique allocation ID
    double last_access_time;    // Timestamp of last access
    
    // Performance tracking
    size_t read_count;          // Number of read operations
    size_t write_count;         // Number of write operations
    double total_access_time;   // Cumulative access time
    
    // Internal pool management
    struct MMBuffer* next;      // Next buffer in pool free list
    struct MMBuffer* prev;      // Previous buffer for doubly-linked list
} MMBuffer;

// Memory pool statistics
typedef struct {
    size_t total_size;          // Total pool size
    size_t allocated_size;      // Currently allocated
    size_t free_size;           // Available for allocation
    size_t peak_usage;          // Peak memory usage
    uint32_t alloc_count;       // Number of allocations
    uint32_t free_count;        // Number of deallocations
    uint32_t fragmentation_score; // 0-100, higher = more fragmented
} MMPoolStats;

// Global memory manager statistics
typedef struct {
    MMPoolStats pools[MM_POOL_COUNT];  // Per-pool statistics
    size_t total_system_memory;        // Total system memory
    size_t total_gpu_memory;           // Total GPU memory
    size_t unified_memory_size;        // Available unified memory
    uint32_t active_buffers;           // Currently active buffers
    double avg_allocation_time;        // Average allocation time (ms)
    double avg_access_time;            // Average access time (ms)
} MMManagerStats;

// Forward declaration
typedef struct MMManager MMManager;

// Core memory manager functions
MetalError mm_manager_create(MMManager** manager, MetalContext* metal_context);
void mm_manager_destroy(MMManager* manager);

// Buffer allocation and management
MetalError mm_buffer_alloc(MMManager* manager, size_t size, 
                          MMAccessPattern access_pattern, MMBuffer** buffer);
MetalError mm_buffer_alloc_with_data(MMManager* manager, const void* data, size_t size,
                                    MMAccessPattern access_pattern, MMBuffer** buffer);
MetalError mm_buffer_realloc(MMManager* manager, MMBuffer* buffer, size_t new_size);
void mm_buffer_retain(MMBuffer* buffer);
void mm_buffer_release(MMBuffer* buffer);

// Data transfer operations (optimized for unified memory)
MetalError mm_buffer_copy_to_gpu(MMBuffer* buffer);
MetalError mm_buffer_copy_from_gpu(MMBuffer* buffer);
MetalError mm_buffer_sync(MMBuffer* buffer);  // Sync unified memory

// Buffer access and mapping
void* mm_buffer_map_read(MMBuffer* buffer);
void* mm_buffer_map_write(MMBuffer* buffer);
void* mm_buffer_map_read_write(MMBuffer* buffer);
void mm_buffer_unmap(MMBuffer* buffer);

// Memory pool management
MetalError mm_pool_configure(MMManager* manager, MMPoolType pool_type, 
                            size_t initial_size, size_t max_size);
MetalError mm_pool_defragment(MMManager* manager, MMPoolType pool_type);
MetalError mm_pool_trim(MMManager* manager, MMPoolType pool_type);

// Performance and monitoring
MetalError mm_get_stats(MMManager* manager, MMManagerStats* stats);
MetalError mm_get_pool_stats(MMManager* manager, MMPoolType pool_type, MMPoolStats* stats);
void mm_print_stats(MMManager* manager);
void mm_print_buffer_info(MMBuffer* buffer);

// Auto-optimization
MetalError mm_optimize_pools(MMManager* manager);
MetalError mm_suggest_strategy(MMManager* manager, size_t size, 
                              MMAccessPattern access_pattern, MMStrategy* strategy);

// Debugging and profiling
void mm_enable_profiling(MMManager* manager, bool enable);
void mm_dump_allocation_trace(MMManager* manager);
MetalError mm_validate_heap(MMManager* manager);

// Utility functions
const char* mm_pool_type_string(MMPoolType pool_type);
const char* mm_strategy_string(MMStrategy strategy);
const char* mm_access_pattern_string(MMAccessPattern access_pattern);

#ifdef __cplusplus
}
#endif

#endif /* USE_METAL */

#endif /* MEMORY_MANAGER_H */
