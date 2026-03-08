/*
 * memory_pool_optimizer.h
 *
 * Memory Pool Optimization Engine for Unified Memory Manager
 * Handles size-based pooling, fragmentation prevention, and dynamic resizing
 */

#ifndef MEMORY_POOL_OPTIMIZER_H
#define MEMORY_POOL_OPTIMIZER_H

#import <Metal/Metal.h>
#include <vector>
#include <map>
#include <mutex>

// Pool bucket sizes (powers of 2 usually, or specific sizes)
struct PoolBucket {
    size_t size_class;
    std::vector<id<MTLBuffer>> free_buffers;
    size_t total_count;
    size_t hit_count;
    size_t miss_count;
};

// Allocation result containing buffer and offset
struct PoolAllocation {
    id<MTLBuffer> buffer;
    size_t offset;
    void* cpu_ptr;
    size_t size; // Requested/Allocated size
};

// Forward declaration
class SubAllocator;

class MemoryPoolOptimizer {
public:
    MemoryPoolOptimizer(id<MTLDevice> device);
    ~MemoryPoolOptimizer();

    // Get a buffer from the pool or allocate new one
    PoolAllocation getBuffer(size_t size);

    // Return a buffer to the pool
    void recycleBuffer(PoolAllocation allocation);

    // Maintenance
    void trim(float keep_ratio = 0.5f);
    void reset();

    // Stats
    void printStats();

private:
    id<MTLDevice> device;
    std::mutex pool_mutex;
    
    // Map size_class -> Bucket (for large allocations)
    std::map<size_t, PoolBucket> buckets;
    
    // Map size_class -> SubAllocator (for small allocations)
    std::map<size_t, SubAllocator*> sub_allocators;
    
    // Helper to find best fit bucket size
    size_t getBucketSize(size_t requested_size);
    
    // Helper to check if size is small enough for sub-allocation
    bool isSmallAlloc(size_t size);
};

#endif // MEMORY_POOL_OPTIMIZER_H
