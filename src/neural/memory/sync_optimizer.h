/*
 * sync_optimizer.h
 *
 * CPU-GPU Synchronization Optimization Engine
 * Manages command buffers, tracks buffer dependencies, and minimizes CPU blocking
 * Implements Dependency Graph tracking and Deadlock Detection
 */

#ifndef SYNC_OPTIMIZER_H
#define SYNC_OPTIMIZER_H

#import <Metal/Metal.h>
#include <mutex>
#include <vector>
#include <map>
#include <set>
#include <string>

// Node in the dependency graph representing a Command Buffer
struct DependencyNode {
    uint64_t id;
    std::string label;
    uint64_t timestamp;
    std::set<uint64_t> dependencies; // IDs of nodes this node waits on
    bool is_committed;
    bool is_completed;
};

class SyncOptimizer {
public:
    SyncOptimizer(id<MTLDevice> device, id<MTLCommandQueue> queue);
    ~SyncOptimizer();

    // Get a command buffer for encoding work
    id<MTLCommandBuffer> getCommandBuffer(const char* label = "Unnamed");

    // Commit a command buffer
    // wait: if true, blocks CPU until completion (avoid if possible)
    void commitCommandBuffer(id<MTLCommandBuffer> buffer, bool wait = false);

    // Register buffer usage for dependency tracking
    // Call this before encoding commands that use these buffers
    void recordRead(id<MTLCommandBuffer> cmdBuffer, id<MTLBuffer> buffer);
    void recordWrite(id<MTLCommandBuffer> cmdBuffer, id<MTLBuffer> buffer);

    // Check if a buffer is currently in use by GPU
    bool isBufferInUse(id<MTLBuffer> buffer);

    // Wait for a specific buffer to be free (GPU finished using it)
    void waitForBuffer(id<MTLBuffer> buffer);

    // Wait for all pending commands to complete (Sync point)
    void waitForAll();

    // Helper to fence memory between encoders if needed
    void memoryBarrier(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer);

    // Deadlock detection: Checks for cycles in the dependency graph
    bool detectDeadlock();
    
    // Print the current dependency graph
    void printDependencyGraph();

private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    std::mutex sync_mutex;
    
    // Active command buffers
    std::vector<id<MTLCommandBuffer>> pending_buffers;
    
    // Dependency Graph: CommandBuffer ID -> Node
    std::map<uint64_t, DependencyNode> dependency_graph;
    
    // Map MTLCommandBuffer pointer to unique ID
    std::map<id<MTLCommandBuffer>, uint64_t> cmd_to_id;
    uint64_t next_cmd_id = 1;

    // Dependency tracking: Buffer -> List of CommandBuffers using it
    struct BufferState {
        uint64_t last_writer_id; // ID of the command buffer that last wrote
        std::set<uint64_t> current_reader_ids; // IDs of command buffers currently reading
    };
    
    std::map<id<MTLBuffer>, BufferState> buffer_dependencies;
    
    // Helpers
    uint64_t getCmdId(id<MTLCommandBuffer> cmd);
    void addDependency(uint64_t waiter_id, uint64_t signaler_id);
    void cleanupCompletedBuffers();
    bool hasCycle(uint64_t node_id, std::set<uint64_t>& visited, std::set<uint64_t>& recursion_stack);
};

#endif // SYNC_OPTIMIZER_H
