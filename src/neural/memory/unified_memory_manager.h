/*
 * unified_memory_manager.h
 *
 * Unified Memory Management System for Apple Silicon
 * Implements shared memory pools using MTLResourceStorageModeShared
 * Provides direct CPU-GPU access and memory leak detection
 */

#ifndef UNIFIED_MEMORY_MANAGER_H
#define UNIFIED_MEMORY_MANAGER_H

#import <Metal/Metal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Memory pool types
typedef enum {
    UNIFIED_POOL_SMALL = 0,    // < 4KB buffers
    UNIFIED_POOL_MEDIUM,       // 4KB - 1MB buffers
    UNIFIED_POOL_LARGE,        // > 1MB buffers
    UNIFIED_POOL_WEIGHTS,      // Model weights (persistent)
    UNIFIED_POOL_TEMPORARY,    // Scratch space (frame/op lifetime)
    UNIFIED_POOL_COUNT
} UnifiedPoolType;

// Memory allocation tracking info
typedef struct {
    size_t size;
    UnifiedPoolType pool_type;
    const char* label;
    uint64_t timestamp;
    void* cpu_ptr;
} AllocationInfo;

// Memory statistics
typedef struct {
    size_t total_allocated;
    size_t total_used;
    size_t peak_memory;
    size_t pool_usage[UNIFIED_POOL_COUNT];
    uint32_t active_allocations;
    uint32_t leak_count;
} UnifiedMemoryStats;

// Opaque manager handle
typedef struct UnifiedMemoryManager UnifiedMemoryManager;

// Core API

/**
 * Create and initialize the Unified Memory Manager
 * @param device Metal device to use for allocations (must support unified memory)
 * @return Pointer to manager instance or NULL on failure
 */
UnifiedMemoryManager* unified_memory_manager_create(id<MTLDevice> device);

/**
 * Destroy the manager and free all resources
 * Reports memory leaks if any allocations remain active
 */
void unified_memory_manager_destroy(UnifiedMemoryManager* manager);

/**
 * Allocate unified memory accessible by both CPU and GPU
 * @param manager Manager instance
 * @param size Size in bytes
 * @param pool_type Target pool hint
 * @param label Debug label for tracking
 * @return Pointer to CPU-accessible memory (GPU buffer can be retrieved via _get_buffer)
 */
void* unified_memory_alloc(UnifiedMemoryManager* manager, size_t size, UnifiedPoolType pool_type, const char* label);

/**
 * Free unified memory
 * @param manager Manager instance
 * @param ptr CPU pointer to memory to free
 */
void unified_memory_free(UnifiedMemoryManager* manager, void* ptr);

/**
 * Get the underlying Metal buffer for a CPU pointer
 * @param manager Manager instance
 * @param ptr CPU pointer
 * @return id<MTLBuffer> or nil if not found
 */
id<MTLBuffer> unified_memory_get_buffer(UnifiedMemoryManager* manager, void* ptr);

/**
 * Get the offset within the Metal buffer for a CPU pointer
 * @param manager Manager instance
 * @param ptr CPU pointer
 * @return Offset in bytes, or 0 if not found
 */
size_t unified_memory_get_offset(UnifiedMemoryManager* manager, void* ptr);

/**
 * Get current memory statistics
 */
void unified_memory_get_stats(UnifiedMemoryManager* manager, UnifiedMemoryStats* stats);

/**
 * Print detailed memory report including potential leaks
 */
void unified_memory_print_report(UnifiedMemoryManager* manager);

#ifdef __cplusplus
}
#endif

#endif // UNIFIED_MEMORY_MANAGER_H
