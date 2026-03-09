/*
 * memory_pool_optimizer.mm
 *
 * Implementation of Memory Pool Optimization Engine
 */

#include "memory_pool_optimizer.h"
#include <cmath>
#include <iostream>
#include <vector>

// --- SubAllocator Implementation ---

struct Slab {
    id<MTLBuffer> buffer;
    void* cpu_base;
    std::vector<size_t> free_indices;
    size_t capacity_count;
};

class SubAllocator {
public:
    SubAllocator(id<MTLDevice> dev, size_t item_sz, size_t slab_sz = 1024 * 1024) 
        : device(dev), item_size(item_sz), slab_size(slab_sz) {
        alloc_count = 0;
    }
        
    PoolAllocation allocate() {
        alloc_count++;
        // Find slab with free space
        for (auto& slab : slabs) {
            if (!slab.free_indices.empty()) {
                size_t idx = slab.free_indices.back();
                slab.free_indices.pop_back();
                size_t offset = idx * item_size;
                return {slab.buffer, offset, (char*)slab.cpu_base + offset, item_size};
            }
        }
        
        // Create new slab
        Slab new_slab;
        MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
        new_slab.buffer = [device newBufferWithLength:slab_size options:options];
        if (!new_slab.buffer) {
            return {nil, 0, NULL, 0};
        }
        new_slab.cpu_base = [new_slab.buffer contents];
        new_slab.capacity_count = slab_size / item_size;
        
        // Initialize free list
        new_slab.free_indices.reserve(new_slab.capacity_count);
        for (size_t i = 0; i < new_slab.capacity_count; i++) {
            new_slab.free_indices.push_back(new_slab.capacity_count - 1 - i);
        }
        
        // Allocate first one
        size_t idx = new_slab.free_indices.back();
        new_slab.free_indices.pop_back();
        
        slabs.push_back(new_slab);
        
        size_t offset = idx * item_size;
        return {new_slab.buffer, offset, (char*)new_slab.cpu_base + offset, item_size};
    }
    
    void deallocate(PoolAllocation alloc) {
        // Find which slab it belongs to
        for (auto& slab : slabs) {
            if (slab.buffer == alloc.buffer) {
                size_t idx = alloc.offset / item_size;
                slab.free_indices.push_back(idx);
                return;
            }
        }
        // If not found, it might be a logic error or different allocator?
        // Should not happen if logic is correct.
    }
    
    void reset() {
        slabs.clear(); // ARC releases buffers
        alloc_count = 0;
    }
    
    size_t getTotalAllocated() const { return alloc_count; }
    size_t getSlabCount() const { return slabs.size(); }
    size_t getItemSize() const { return item_size; }

private:
    id<MTLDevice> device;
    size_t item_size;
    size_t slab_size;
    std::vector<Slab> slabs;
    size_t alloc_count;
};

// --- MemoryPoolOptimizer Implementation ---

MemoryPoolOptimizer::MemoryPoolOptimizer(id<MTLDevice> dev) : device(dev) {
    // Initialize buckets for large sizes (4KB to 64MB)
    size_t size = 4096;
    while (size <= 64 * 1024 * 1024) {
        buckets[size] = {size, {}, 0, 0, 0};
        size *= 2;
    }
    
    // Initialize sub-allocators for small sizes (64B to 2048B)
    // Using 1MB slabs
    size_t small_sizes[] = {64, 128, 256, 512, 1024, 2048};
    for (size_t s : small_sizes) {
        sub_allocators[s] = new SubAllocator(dev, s, 1024 * 1024);
    }
}

MemoryPoolOptimizer::~MemoryPoolOptimizer() {
    reset();
    for (auto& pair : sub_allocators) {
        delete pair.second;
    }
}

bool MemoryPoolOptimizer::isSmallAlloc(size_t size) {
    return size <= 2048;
}

size_t MemoryPoolOptimizer::getBucketSize(size_t requested_size) {
    if (isSmallAlloc(requested_size)) {
        // Round up to nearest sub-allocator size
        size_t size = 64;
        while (size < requested_size) {
            size *= 2;
        }
        return size;
    }
    
    // For large allocs
    size_t size = 4096;
    while (size < requested_size) {
        size *= 2;
        if (size > 64 * 1024 * 1024) {
            return requested_size;
        }
    }
    return size;
}

PoolAllocation MemoryPoolOptimizer::getBuffer(size_t size) {
    size_t bucketSize = getBucketSize(size);
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    if (isSmallAlloc(size)) {
        auto it = sub_allocators.find(bucketSize);
        if (it != sub_allocators.end()) {
            return it->second->allocate();
        }
    }
    
    // Large allocation logic
    auto it = buckets.find(bucketSize);
    if (it != buckets.end()) {
        PoolBucket& bucket = it->second;
        if (!bucket.free_buffers.empty()) {
            id<MTLBuffer> buffer = bucket.free_buffers.back();
            bucket.free_buffers.pop_back();
            bucket.hit_count++;
            return {buffer, 0, [buffer contents], bucketSize};
        }
        bucket.miss_count++;
        bucket.total_count++;
        MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
        id<MTLBuffer> buf = [device newBufferWithLength:bucketSize options:options];
        return {buf, 0, [buf contents], bucketSize};
    }
    
    // Non-pooled size
    MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
    id<MTLBuffer> buf = [device newBufferWithLength:size options:options];
    return {buf, 0, [buf contents], size};
}

void MemoryPoolOptimizer::recycleBuffer(PoolAllocation allocation) {
    if (!allocation.buffer) return;
    
    size_t size = allocation.size;
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    if (isSmallAlloc(size)) {
        // Re-calculate size class to find correct allocator
        size_t bucketSize = getBucketSize(size);
        auto it = sub_allocators.find(bucketSize);
        if (it != sub_allocators.end()) {
            it->second->deallocate(allocation);
        }
        return;
    }
    
    // Large buffer recycle
    // Note: allocation.size might be exact size, but we need bucket size
    // If it came from a bucket, allocation.size should match bucket size
    // or we use buffer.length
    
    size_t capacity = allocation.buffer.length;
    
    auto it = buckets.find(capacity);
    if (it != buckets.end()) {
        it->second.free_buffers.push_back(allocation.buffer);
    }
}

void MemoryPoolOptimizer::trim(float keep_ratio) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    for (auto& pair : buckets) {
        PoolBucket& bucket = pair.second;
        size_t current = bucket.free_buffers.size();
        if (current > 0) {
            size_t keep = (size_t)(current * keep_ratio);
            if (keep < 1 && keep_ratio > 0) keep = 1;
            
            while (bucket.free_buffers.size() > keep) {
                bucket.free_buffers.pop_back();
            }
        }
    }
    
    // Note: SubAllocators are harder to trim because slabs contain mixed active/free blocks.
    // We could free fully empty slabs, but that requires tracking active count per slab.
    // For now, we leave them.
}

void MemoryPoolOptimizer::reset() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    for (auto& pair : buckets) {
        pair.second.free_buffers.clear();
        pair.second.total_count = 0;
        pair.second.hit_count = 0;
        pair.second.miss_count = 0;
    }
    for (auto& pair : sub_allocators) {
        pair.second->reset();
    }
}

void MemoryPoolOptimizer::printStats() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    printf("--- Memory Pool Stats ---\n");
    
    printf("Small Allocations (Slabs):\n");
    for (const auto& pair : sub_allocators) {
        printf("  Size %4zu B: %zu slabs, %zu total allocs\n", 
               pair.second->getItemSize(), pair.second->getSlabCount(), pair.second->getTotalAllocated());
    }
    
    printf("Large Allocations (Buckets):\n");
    for (const auto& pair : buckets) {
        const PoolBucket& bucket = pair.second;
        if (bucket.total_count > 0 || !bucket.free_buffers.empty()) {
            printf("  Bucket %8zu B: %4zu free, %6zu total allocs, %6zu hits, %6zu misses\n", 
                   bucket.size_class, bucket.free_buffers.size(), bucket.total_count, bucket.hit_count, bucket.miss_count);
        }
    }
    printf("-------------------------\n");
}
