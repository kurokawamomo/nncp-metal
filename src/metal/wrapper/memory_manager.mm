#ifdef USE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <sys/time.h>

#include "memory_manager.h"
#include "metal_context.h"

// Memory pool configuration
static const size_t MM_POOL_SIZES[MM_POOL_COUNT] = {
    1024 * 1024,        // Small: 1MB
    16 * 1024 * 1024,   // Medium: 16MB  
    256 * 1024 * 1024,  // Large: 256MB
    64 * 1024 * 1024    // Transient: 64MB
};

static const size_t MM_POOL_MAX_SIZES[MM_POOL_COUNT] = {
    16 * 1024 * 1024,   // Small: 16MB max
    256 * 1024 * 1024,  // Medium: 256MB max
    1024 * 1024 * 1024, // Large: 1GB max
    128 * 1024 * 1024   // Transient: 128MB max
};

// Internal memory pool structure
typedef struct MMPool {
    MMPoolType type;
    size_t total_size;
    size_t allocated_size;
    size_t max_size;
    MMBuffer* free_list;     // Free buffer list
    MMBuffer* used_list;     // Used buffer list
    uint32_t alloc_count;
    uint32_t free_count;
    size_t peak_usage;
    void* pool_base;         // Base pointer for pool memory
    id<MTLBuffer> metal_pool_buffer; // Metal buffer for entire pool
} MMPool;

// Internal memory manager structure
struct MMManager {
    MetalContext* metal_context;
    id<MTLDevice> device;
    MMPool pools[MM_POOL_COUNT];
    uint64_t next_alloc_id;
    bool profiling_enabled;
    bool unified_memory_available;
    size_t total_system_memory;
    size_t total_gpu_memory;
    uint32_t active_buffers;
    NSMutableArray* allocation_trace;
};

// Utility functions
static double get_current_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

static MMPoolType determine_pool_type(size_t size) {
    if (size < 1024 * 1024) {
        return MM_POOL_SMALL;
    } else if (size < 16 * 1024 * 1024) {
        return MM_POOL_MEDIUM;
    } else {
        return MM_POOL_LARGE;
    }
}

static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Pool management
static MetalError init_pool(MMManager* manager, MMPoolType pool_type) {
    MMPool* pool = &manager->pools[pool_type];
    pool->type = pool_type;
    pool->total_size = MM_POOL_SIZES[pool_type];
    pool->max_size = MM_POOL_MAX_SIZES[pool_type];
    pool->allocated_size = 0;
    pool->alloc_count = 0;
    pool->free_count = 0;
    pool->peak_usage = 0;
    pool->free_list = NULL;
    pool->used_list = NULL;
    
    // Create Metal buffer for the pool
    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (manager->unified_memory_available) {
        options |= MTLResourceCPUCacheModeDefaultCache;
    }
    
    pool->metal_pool_buffer = [manager->device newBufferWithLength:pool->total_size 
                                                           options:options];
    if (!pool->metal_pool_buffer) {
        return METAL_ERROR_BUFFER_ALLOCATION;
    }
    
    pool->pool_base = [pool->metal_pool_buffer contents];
    return METAL_SUCCESS;
}

static void cleanup_pool(MMPool* pool) {
    // Free all buffers in the pool
    MMBuffer* current = pool->used_list;
    while (current) {
        MMBuffer* next = current->next;
        free(current);
        current = next;
    }
    
    current = pool->free_list;
    while (current) {
        MMBuffer* next = current->next;
        free(current);
        current = next;
    }
    
    pool->metal_pool_buffer = nil; // ARC will handle cleanup
}

// Core memory manager functions
MetalError mm_manager_create(MMManager** manager, MetalContext* metal_context) {
    if (!manager || !metal_context) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    MMManager* mgr = (MMManager*)calloc(1, sizeof(MMManager));
    if (!mgr) {
        return METAL_ERROR_OUT_OF_MEMORY;
    }
    
    mgr->metal_context = metal_context;
    mgr->device = MTLCreateSystemDefaultDevice();
    mgr->next_alloc_id = 1;
    mgr->profiling_enabled = false;
    mgr->active_buffers = 0;
    
    // Check for unified memory support
    mgr->unified_memory_available = mgr->device.hasUnifiedMemory;
    
    // Get system memory info
    mgr->total_system_memory = [NSProcessInfo processInfo].physicalMemory;
    if (@available(macOS 13.0, *)) {
        mgr->total_gpu_memory = mgr->device.recommendedMaxWorkingSetSize;
    } else {
        mgr->total_gpu_memory = mgr->total_system_memory / 4; // Conservative estimate
    }
    
    // Initialize allocation trace
    mgr->allocation_trace = [[NSMutableArray alloc] init];
    
    // Initialize memory pools
    for (int i = 0; i < MM_POOL_COUNT; i++) {
        MetalError result = init_pool(mgr, (MMPoolType)i);
        if (result != METAL_SUCCESS) {
            mm_manager_destroy(mgr);
            return result;
        }
    }
    
    *manager = mgr;
    return METAL_SUCCESS;
}

void mm_manager_destroy(MMManager* manager) {
    if (!manager) return;
    
    // Cleanup all pools
    for (int i = 0; i < MM_POOL_COUNT; i++) {
        cleanup_pool(&manager->pools[i]);
    }
    
    manager->allocation_trace = nil; // ARC cleanup
    manager->device = nil; // ARC cleanup
    free(manager);
}

// Buffer allocation - simplified implementation
MetalError mm_buffer_alloc(MMManager* manager, size_t size, 
                          MMAccessPattern access_pattern, MMBuffer** buffer) {
    if (!manager || !buffer || size == 0) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    // Determine pool type and alignment
    MMPoolType pool_type = determine_pool_type(size);
    size_t aligned_size = align_size(size, 16); // 16-byte alignment
    
    // For now, create a simple buffer (pool allocation will be implemented later)
    MMBuffer* buf = (MMBuffer*)calloc(1, sizeof(MMBuffer));
    if (!buf) {
        return METAL_ERROR_OUT_OF_MEMORY;
    }
    
    // Create Metal buffer
    MTLResourceOptions options = MTLResourceStorageModeShared;
    if (manager->unified_memory_available) {
        options |= MTLResourceCPUCacheModeDefaultCache;
    }
    
    id<MTLBuffer> metal_buffer = [manager->device newBufferWithLength:aligned_size
                                                              options:options];
    if (!metal_buffer) {
        free(buf);
        return METAL_ERROR_BUFFER_ALLOCATION;
    }
    
    // Initialize buffer structure
    buf->cpu_ptr = [metal_buffer contents];
    buf->gpu_handle = (__bridge void*)metal_buffer;
    buf->size = size;
    buf->alignment = 16;
    buf->pool_type = pool_type;
    buf->strategy = manager->unified_memory_available ? MM_STRATEGY_UNIFIED : MM_STRATEGY_DISCRETE;
    buf->access_pattern = access_pattern;
    buf->is_unified = manager->unified_memory_available;
    buf->is_pooled = false; // Direct allocation for now
    buf->ref_count = 1;
    buf->alloc_id = manager->next_alloc_id++;
    buf->last_access_time = get_current_time();
    buf->read_count = 0;
    buf->write_count = 0;
    buf->total_access_time = 0.0;
    buf->next = NULL;
    buf->prev = NULL;
    
    // Update statistics
    manager->active_buffers++;
    manager->pools[pool_type].allocated_size += aligned_size;
    manager->pools[pool_type].alloc_count++;
    
    if (manager->profiling_enabled) {
        NSString* trace = [NSString stringWithFormat:@"ALLOC: ID=%llu, Size=%zu, Pool=%d, Pattern=%d",
                          buf->alloc_id, size, pool_type, access_pattern];
        [manager->allocation_trace addObject:trace];
    }
    
    *buffer = buf;
    return METAL_SUCCESS;
}

MetalError mm_buffer_alloc_with_data(MMManager* manager, const void* data, size_t size,
                                    MMAccessPattern access_pattern, MMBuffer** buffer) {
    MetalError result = mm_buffer_alloc(manager, size, access_pattern, buffer);
    if (result != METAL_SUCCESS) {
        return result;
    }
    
    // Copy data to buffer
    memcpy((*buffer)->cpu_ptr, data, size);
    (*buffer)->write_count++;
    
    return METAL_SUCCESS;
}

void mm_buffer_retain(MMBuffer* buffer) {
    if (buffer) {
        buffer->ref_count++;
    }
}

void mm_buffer_release(MMBuffer* buffer) {
    if (!buffer) return;
    
    buffer->ref_count--;
    if (buffer->ref_count == 0) {
        // Buffer should be deallocated
        // For now, just mark as available for cleanup
        // Full implementation would return to pool
        if (buffer->gpu_handle) {
            // Release Metal buffer via ARC
            CFRelease(buffer->gpu_handle);
        }
        free(buffer);
    }
}

// Simplified data transfer operations
MetalError mm_buffer_copy_to_gpu(MMBuffer* buffer) {
    if (!buffer) return METAL_ERROR_INVALID_PARAMETER;
    
    // For unified memory, this is a no-op
    if (buffer->is_unified) {
        return METAL_SUCCESS;
    }
    
    // For discrete memory, would implement actual copy
    // Placeholder implementation
    buffer->last_access_time = get_current_time();
    buffer->write_count++;
    return METAL_SUCCESS;
}

MetalError mm_buffer_copy_from_gpu(MMBuffer* buffer) {
    if (!buffer) return METAL_ERROR_INVALID_PARAMETER;
    
    // For unified memory, this is a no-op
    if (buffer->is_unified) {
        return METAL_SUCCESS;
    }
    
    // For discrete memory, would implement actual copy
    buffer->last_access_time = get_current_time();
    buffer->read_count++;
    return METAL_SUCCESS;
}

MetalError mm_buffer_sync(MMBuffer* buffer) {
    if (!buffer) return METAL_ERROR_INVALID_PARAMETER;
    
    // Synchronize unified memory
    if (buffer->is_unified) {
        // Memory barrier for unified memory
        __sync_synchronize();
    }
    
    buffer->last_access_time = get_current_time();
    return METAL_SUCCESS;
}

// Buffer mapping operations
void* mm_buffer_map_read(MMBuffer* buffer) {
    if (!buffer) return NULL;
    
    buffer->last_access_time = get_current_time();
    buffer->read_count++;
    return buffer->cpu_ptr;
}

void* mm_buffer_map_write(MMBuffer* buffer) {
    if (!buffer) return NULL;
    
    buffer->last_access_time = get_current_time();
    buffer->write_count++;
    return buffer->cpu_ptr;
}

void* mm_buffer_map_read_write(MMBuffer* buffer) {
    if (!buffer) return NULL;
    
    buffer->last_access_time = get_current_time();
    buffer->read_count++;
    buffer->write_count++;
    return buffer->cpu_ptr;
}

void mm_buffer_unmap(MMBuffer* buffer) {
    if (!buffer) return;
    
    // For unified memory, ensure cache coherency
    if (buffer->is_unified) {
        __sync_synchronize();
    }
}

// Statistics and monitoring
MetalError mm_get_stats(MMManager* manager, MMManagerStats* stats) {
    if (!manager || !stats) {
        return METAL_ERROR_INVALID_PARAMETER;
    }
    
    memset(stats, 0, sizeof(MMManagerStats));
    
    // Copy pool statistics
    for (int i = 0; i < MM_POOL_COUNT; i++) {
        MMPool* pool = &manager->pools[i];
        stats->pools[i].total_size = pool->total_size;
        stats->pools[i].allocated_size = pool->allocated_size;
        stats->pools[i].free_size = pool->total_size - pool->allocated_size;
        stats->pools[i].peak_usage = pool->peak_usage;
        stats->pools[i].alloc_count = pool->alloc_count;
        stats->pools[i].free_count = pool->free_count;
        stats->pools[i].fragmentation_score = 0; // Simplified
    }
    
    // Global statistics
    stats->total_system_memory = manager->total_system_memory;
    stats->total_gpu_memory = manager->total_gpu_memory;
    stats->unified_memory_size = manager->unified_memory_available ? manager->total_system_memory : 0;
    stats->active_buffers = manager->active_buffers;
    stats->avg_allocation_time = 0.1; // Placeholder
    stats->avg_access_time = 0.01; // Placeholder
    
    return METAL_SUCCESS;
}

void mm_print_stats(MMManager* manager) {
    if (!manager) return;
    
    MMManagerStats stats;
    if (mm_get_stats(manager, &stats) != METAL_SUCCESS) {
        return;
    }
    
    printf("\n=== Memory Manager Statistics ===\n");
    printf("System Memory: %.2f MB\n", stats.total_system_memory / (1024.0 * 1024.0));
    printf("GPU Memory: %.2f MB\n", stats.total_gpu_memory / (1024.0 * 1024.0));
    printf("Unified Memory: %s\n", stats.unified_memory_size > 0 ? "YES" : "NO");
    printf("Active Buffers: %u\n", stats.active_buffers);
    
    printf("\nPool Statistics:\n");
    for (int i = 0; i < MM_POOL_COUNT; i++) {
        printf("  %s Pool:\n", mm_pool_type_string((MMPoolType)i));
        printf("    Total: %.2f MB\n", stats.pools[i].total_size / (1024.0 * 1024.0));
        printf("    Allocated: %.2f MB\n", stats.pools[i].allocated_size / (1024.0 * 1024.0));
        printf("    Allocations: %u\n", stats.pools[i].alloc_count);
    }
}

// Utility functions
const char* mm_pool_type_string(MMPoolType pool_type) {
    switch (pool_type) {
        case MM_POOL_SMALL: return "Small";
        case MM_POOL_MEDIUM: return "Medium";
        case MM_POOL_LARGE: return "Large";
        case MM_POOL_TRANSIENT: return "Transient";
        default: return "Unknown";
    }
}

const char* mm_strategy_string(MMStrategy strategy) {
    switch (strategy) {
        case MM_STRATEGY_UNIFIED: return "Unified";
        case MM_STRATEGY_DISCRETE: return "Discrete";
        case MM_STRATEGY_AUTO: return "Auto";
        default: return "Unknown";
    }
}

const char* mm_access_pattern_string(MMAccessPattern access_pattern) {
    switch (access_pattern) {
        case MM_ACCESS_READ_ONLY: return "ReadOnly";
        case MM_ACCESS_WRITE_ONLY: return "WriteOnly";
        case MM_ACCESS_READ_WRITE: return "ReadWrite";
        case MM_ACCESS_STREAMING: return "Streaming";
        default: return "Unknown";
    }
}

// Simplified optimization functions
MetalError mm_optimize_pools(MMManager* manager) {
    if (!manager) return METAL_ERROR_INVALID_PARAMETER;
    
    // Placeholder for pool optimization logic
    return METAL_SUCCESS;
}

MetalError mm_suggest_strategy(MMManager* manager, size_t size, 
                              MMAccessPattern access_pattern, MMStrategy* strategy) {
    if (!manager || !strategy) return METAL_ERROR_INVALID_PARAMETER;
    
    // Simple strategy suggestion
    if (manager->unified_memory_available) {
        *strategy = MM_STRATEGY_UNIFIED;
    } else {
        *strategy = MM_STRATEGY_DISCRETE;
    }
    
    return METAL_SUCCESS;
}

// Profiling and debugging
void mm_enable_profiling(MMManager* manager, bool enable) {
    if (manager) {
        manager->profiling_enabled = enable;
    }
}

void mm_dump_allocation_trace(MMManager* manager) {
    if (!manager) return;
    
    printf("\n=== Allocation Trace ===\n");
    for (NSString* entry in manager->allocation_trace) {
        printf("%s\n", [entry UTF8String]);
    }
}

#endif /* USE_METAL */
