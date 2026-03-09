/*
 * unified_memory_manager.mm
 *
 * Implementation of Unified Memory Management System
 * Leverages Apple Silicon's Unified Memory Architecture (UMA)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "unified_memory_manager.h"
#include "memory_pool_optimizer.h"
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <iostream>

// Internal tracking structure
struct AllocationEntry {
    id<MTLBuffer> buffer;
    size_t size;
    size_t offset; // Offset within the buffer
    UnifiedPoolType pool_type;
    std::string label;
    uint64_t timestamp;
};

// Manager implementation details
struct UnifiedMemoryManager {
    id<MTLDevice> device;
    std::mutex mutex;
    std::map<void*, AllocationEntry> allocations;
    UnifiedMemoryStats stats;
    bool is_initialized;
    
    // Memory pool optimizer
    MemoryPoolOptimizer* optimizer;
};

extern "C" {

UnifiedMemoryManager* unified_memory_manager_create(id<MTLDevice> device) {
    if (!device) return NULL;
    
    UnifiedMemoryManager* manager = new UnifiedMemoryManager();
    manager->device = device;
    manager->is_initialized = true;
    
    // Initialize optimizer
    manager->optimizer = new MemoryPoolOptimizer(device);
    
    // Initialize stats
    manager->stats = {0};
    
    printf("[UnifiedMemory] Initialized with device: %s\n", [device.name UTF8String]);
    
    // Check for unified memory support (Apple Silicon)
    if (@available(macOS 11.0, *)) {
        if (![device hasUnifiedMemory]) {
            printf("[UnifiedMemory] Warning: Device does not support unified memory. Performance may be degraded.\n");
        }
    }
    
    return manager;
}

void unified_memory_manager_destroy(UnifiedMemoryManager* manager) {
    if (!manager) return;
    
    std::lock_guard<std::mutex> lock(manager->mutex);
    
    // Check for leaks
    if (!manager->allocations.empty()) {
        printf("[UnifiedMemory] WARNING: %zu allocations leaking at destroy time:\n", manager->allocations.size());
        for (const auto& pair : manager->allocations) {
            printf("  - Leak: %p (%zu bytes) [%s]\n", 
                   pair.first, pair.second.size, pair.second.label.c_str());
        }
    }
    
    delete manager->optimizer;
    manager->allocations.clear();
    delete manager;
    printf("[UnifiedMemory] Destroyed.\n");
}

void* unified_memory_alloc(UnifiedMemoryManager* manager, size_t size, UnifiedPoolType pool_type, const char* label) {
    if (!manager || size == 0) return NULL;
    
    std::lock_guard<std::mutex> lock(manager->mutex);
    
    // Use optimizer to get buffer
    PoolAllocation alloc = manager->optimizer->getBuffer(size);
    if (!alloc.buffer) {
        printf("[UnifiedMemory] Error: Failed to allocate %zu bytes\n", size);
        return NULL;
    }
    
    void* cpu_ptr = alloc.cpu_ptr;
    
    // Register allocation
    AllocationEntry entry;
    entry.buffer = alloc.buffer;
    entry.size = alloc.size; // Use allocated size (bucket size)
    entry.offset = alloc.offset;
    entry.pool_type = pool_type;
    entry.label = label ? std::string(label) : "unnamed";
    entry.timestamp = (uint64_t)time(NULL);
    
    manager->allocations[cpu_ptr] = entry;
    
    // Update stats
    manager->stats.total_allocated += entry.size;
    manager->stats.total_used += entry.size;
    manager->stats.pool_usage[pool_type] += entry.size;
    manager->stats.active_allocations++;
    
    if (manager->stats.total_used > manager->stats.peak_memory) {
        manager->stats.peak_memory = manager->stats.total_used;
    }
    
    return cpu_ptr;
}

void unified_memory_free(UnifiedMemoryManager* manager, void* ptr) {
    if (!manager || !ptr) return;
    
    std::lock_guard<std::mutex> lock(manager->mutex);
    
    auto it = manager->allocations.find(ptr);
    if (it == manager->allocations.end()) {
        printf("[UnifiedMemory] Error: Attempt to free unknown pointer %p\n", ptr);
        return;
    }
    
    AllocationEntry& entry = it->second;
    
    // Update stats
    manager->stats.total_used -= entry.size;
    manager->stats.pool_usage[entry.pool_type] -= entry.size;
    manager->stats.active_allocations--;
    
    // Return to optimizer pool
    PoolAllocation alloc;
    alloc.buffer = entry.buffer;
    alloc.offset = entry.offset;
    alloc.cpu_ptr = ptr;
    alloc.size = entry.size;
    
    manager->optimizer->recycleBuffer(alloc);
    
    manager->allocations.erase(it);
}

id<MTLBuffer> unified_memory_get_buffer(UnifiedMemoryManager* manager, void* ptr) {
    if (!manager || !ptr) return nil;
    
    std::lock_guard<std::mutex> lock(manager->mutex);
    
    auto it = manager->allocations.find(ptr);
    if (it != manager->allocations.end()) {
        return it->second.buffer;
    }
    
    return nil;
}

// New API to get buffer with offset
size_t unified_memory_get_offset(UnifiedMemoryManager* manager, void* ptr) {
    if (!manager || !ptr) return 0;
    
    std::lock_guard<std::mutex> lock(manager->mutex);
    
    auto it = manager->allocations.find(ptr);
    if (it != manager->allocations.end()) {
        return it->second.offset;
    }
    
    return 0;
}

void unified_memory_get_stats(UnifiedMemoryManager* manager, UnifiedMemoryStats* stats) {
    if (!manager || !stats) return;
    
    std::lock_guard<std::mutex> lock(manager->mutex);
    *stats = manager->stats;
    
    stats->leak_count = 0;
}

void unified_memory_print_report(UnifiedMemoryManager* manager) {
    if (!manager) return;
    
    std::lock_guard<std::mutex> lock(manager->mutex);
    
    printf("=== Unified Memory Manager Report ===\n");
    printf("Total Allocated: %.2f MB\n", manager->stats.total_allocated / (1024.0 * 1024.0));
    printf("Current Used:    %.2f MB\n", manager->stats.total_used / (1024.0 * 1024.0));
    printf("Peak Memory:     %.2f MB\n", manager->stats.peak_memory / (1024.0 * 1024.0));
    printf("Active Allocations: %u\n", manager->stats.active_allocations);
    
    printf("\nPool Usage:\n");
    const char* pool_names[] = {"Small", "Medium", "Large", "Weights", "Temporary"};
    for (int i = 0; i < UNIFIED_POOL_COUNT; i++) {
        printf("  %s: %.2f MB\n", pool_names[i], manager->stats.pool_usage[i] / (1024.0 * 1024.0));
    }
    
    // Print optimizer stats
    manager->optimizer->printStats();
    
    if (!manager->allocations.empty()) {
        printf("\nActive Allocations:\n");
        int count = 0;
        for (const auto& pair : manager->allocations) {
            if (count++ > 10) {
                printf("  ... (and %zu more)\n", manager->allocations.size() - 10);
                break;
            }
            printf("  - %p: %zu bytes (Offset: %zu) [%s]\n", 
                   pair.first, pair.second.size, pair.second.offset, pair.second.label.c_str());
        }
    }
    printf("=====================================\n");
}

} // extern "C"
