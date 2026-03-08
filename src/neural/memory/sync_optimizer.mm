/*
 * sync_optimizer.mm
 *
 * Implementation of CPU-GPU Synchronization Optimization
 * Includes dependency tracking and deadlock detection
 */

#include "sync_optimizer.h"
#include <iostream>
#include <algorithm>
#include <sstream>

SyncOptimizer::SyncOptimizer(id<MTLDevice> dev, id<MTLCommandQueue> queue) 
    : device(dev), commandQueue(queue) {
}

SyncOptimizer::~SyncOptimizer() {
    waitForAll();
}

uint64_t SyncOptimizer::getCmdId(id<MTLCommandBuffer> cmd) {
    if (!cmd) return 0;
    auto it = cmd_to_id.find(cmd);
    if (it != cmd_to_id.end()) {
        return it->second;
    }
    return 0;
}

id<MTLCommandBuffer> SyncOptimizer::getCommandBuffer(const char* label) {
    std::lock_guard<std::mutex> lock(sync_mutex);
    cleanupCompletedBuffers();
    
    // Create new command buffer
    id<MTLCommandBuffer> buffer = [commandQueue commandBuffer];
    if (!buffer) {
        printf("[SyncOptimizer] Error: Failed to create command buffer\n");
        return nil;
    }
    
    // Register node in graph
    uint64_t id = next_cmd_id++;
    cmd_to_id[buffer] = id;
    
    DependencyNode node;
    node.id = id;
    node.label = label ? std::string(label) : "Unnamed";
    node.timestamp = (uint64_t)time(NULL);
    node.is_committed = false;
    node.is_completed = false;
    
    dependency_graph[id] = node;
    
    return buffer;
}

void SyncOptimizer::addDependency(uint64_t waiter_id, uint64_t signaler_id) {
    if (waiter_id == 0 || signaler_id == 0 || waiter_id == signaler_id) return;
    
    // Check if signaler is already completed, if so, no dependency needed
    auto it = dependency_graph.find(signaler_id);
    if (it == dependency_graph.end() || it->second.is_completed) return;
    
    dependency_graph[waiter_id].dependencies.insert(signaler_id);
}

void SyncOptimizer::commitCommandBuffer(id<MTLCommandBuffer> buffer, bool wait) {
    if (!buffer) return;

    {
        std::lock_guard<std::mutex> lock(sync_mutex);
        uint64_t id = getCmdId(buffer);
        if (id > 0) {
            dependency_graph[id].is_committed = true;
        }
        
        if (!wait) {
            pending_buffers.push_back(buffer);
        }
    }
    
    // Add completion handler to update graph status
    // Note: We need to be careful with capturing 'this' or members in block
    // For simplicity in this C++ wrapper, we rely on polling/cleanup for status updates
    // or we could use a weak ref if we had a proper ObjC wrapper.
    
    [buffer commit];
    
    if (wait) {
        [buffer waitUntilCompleted];
        
        std::lock_guard<std::mutex> lock(sync_mutex);
        uint64_t id = getCmdId(buffer);
        if (id > 0) {
            dependency_graph[id].is_completed = true;
        }
        cleanupCompletedBuffers();
    }
}

void SyncOptimizer::recordRead(id<MTLCommandBuffer> cmdBuffer, id<MTLBuffer> buffer) {
    if (!cmdBuffer || !buffer) return;
    
    std::lock_guard<std::mutex> lock(sync_mutex);
    uint64_t cmd_id = getCmdId(cmdBuffer);
    if (cmd_id == 0) return; // Should not happen if created via getCommandBuffer
    
    BufferState& state = buffer_dependencies[buffer];
    
    // Read depends on last Write
    if (state.last_writer_id > 0) {
        addDependency(cmd_id, state.last_writer_id);
    }
    
    state.current_reader_ids.insert(cmd_id);
}

void SyncOptimizer::recordWrite(id<MTLCommandBuffer> cmdBuffer, id<MTLBuffer> buffer) {
    if (!cmdBuffer || !buffer) return;
    
    std::lock_guard<std::mutex> lock(sync_mutex);
    uint64_t cmd_id = getCmdId(cmdBuffer);
    if (cmd_id == 0) return;
    
    BufferState& state = buffer_dependencies[buffer];
    
    // Write depends on last Write (WAW)
    if (state.last_writer_id > 0) {
        addDependency(cmd_id, state.last_writer_id);
    }
    
    // Write depends on all current Reads (WAR)
    for (uint64_t reader_id : state.current_reader_ids) {
        addDependency(cmd_id, reader_id);
    }
    
    // Update state: Clear readers, set new writer
    state.current_reader_ids.clear();
    state.last_writer_id = cmd_id;
}

bool SyncOptimizer::isBufferInUse(id<MTLBuffer> buffer) {
    if (!buffer) return false;
    
    std::lock_guard<std::mutex> lock(sync_mutex);
    cleanupCompletedBuffers(); // Refresh status
    
    auto it = buffer_dependencies.find(buffer);
    if (it == buffer_dependencies.end()) return false;
    
    const BufferState& state = it->second;
    
    // Check writer
    if (state.last_writer_id > 0) {
        auto node_it = dependency_graph.find(state.last_writer_id);
        if (node_it != dependency_graph.end() && !node_it->second.is_completed) {
            return true;
        }
    }
    
    // Check readers
    for (uint64_t reader_id : state.current_reader_ids) {
        auto node_it = dependency_graph.find(reader_id);
        if (node_it != dependency_graph.end() && !node_it->second.is_completed) {
            return true;
        }
    }
    
    return false;
}

void SyncOptimizer::waitForBuffer(id<MTLBuffer> buffer) {
    if (!buffer) return;
    
    // This is a CPU wait. We should find which command buffers touch this buffer and wait for them.
    std::vector<id<MTLCommandBuffer>> buffers_to_wait;
    
    {
        std::lock_guard<std::mutex> lock(sync_mutex);
        cleanupCompletedBuffers();
        
        auto it = buffer_dependencies.find(buffer);
        if (it != buffer_dependencies.end()) {
            const BufferState& state = it->second;
            
            // Collect writer
            if (state.last_writer_id > 0) {
                // Find the MTLCommandBuffer object (slow reverse lookup, but robust)
                for (auto& pair : cmd_to_id) {
                    if (pair.second == state.last_writer_id) {
                        buffers_to_wait.push_back(pair.first);
                        break;
                    }
                }
            }
            
            // Collect readers
            for (uint64_t reader_id : state.current_reader_ids) {
                for (auto& pair : cmd_to_id) {
                    if (pair.second == reader_id) {
                        buffers_to_wait.push_back(pair.first);
                        break;
                    }
                }
            }
        }
    }
    
    for (id<MTLCommandBuffer> cmd : buffers_to_wait) {
        if (cmd.status < MTLCommandBufferStatusCompleted) {
            [cmd waitUntilCompleted];
        }
    }
}

void SyncOptimizer::waitForAll() {
    std::vector<id<MTLCommandBuffer>> buffers_to_wait;
    
    {
        std::lock_guard<std::mutex> lock(sync_mutex);
        buffers_to_wait = pending_buffers;
    }
    
    for (id<MTLCommandBuffer> buffer : buffers_to_wait) {
        if (buffer.status < MTLCommandBufferStatusCompleted) {
            [buffer waitUntilCompleted];
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(sync_mutex);
        pending_buffers.clear();
        buffer_dependencies.clear();
        dependency_graph.clear();
        cmd_to_id.clear();
        next_cmd_id = 1;
    }
}

void SyncOptimizer::memoryBarrier(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> buffer) {
    if (!encoder) return;
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

// DFS for cycle detection
bool SyncOptimizer::hasCycle(uint64_t node_id, std::set<uint64_t>& visited, std::set<uint64_t>& recursion_stack) {
    visited.insert(node_id);
    recursion_stack.insert(node_id);
    
    DependencyNode& node = dependency_graph[node_id];
    
    for (uint64_t dep_id : node.dependencies) {
        // If dependency is already completed, it doesn't cause a deadlock
        if (dependency_graph[dep_id].is_completed) continue;
        
        if (recursion_stack.count(dep_id)) {
            return true; // Cycle detected
        }
        if (!visited.count(dep_id)) {
            if (hasCycle(dep_id, visited, recursion_stack)) {
                return true;
            }
        }
    }
    
    recursion_stack.erase(node_id);
    return false;
}

bool SyncOptimizer::detectDeadlock() {
    std::lock_guard<std::mutex> lock(sync_mutex);
    cleanupCompletedBuffers();
    
    std::set<uint64_t> visited;
    std::set<uint64_t> recursion_stack;
    
    for (auto& pair : dependency_graph) {
        uint64_t id = pair.first;
        if (!pair.second.is_completed && !visited.count(id)) {
            if (hasCycle(id, visited, recursion_stack)) {
                printf("[SyncOptimizer] DEADLOCK DETECTED! Cycle found involving CmdBuffer ID %llu\n", id);
                return true;
            }
        }
    }
    
    return false;
}

void SyncOptimizer::printDependencyGraph() {
    std::lock_guard<std::mutex> lock(sync_mutex);
    cleanupCompletedBuffers();
    
    printf("=== Dependency Graph ===\n");
    for (const auto& pair : dependency_graph) {
        const DependencyNode& node = pair.second;
        if (!node.is_completed) {
            printf("Node %llu [%s] (Status: %s)\n", 
                   node.id, node.label.c_str(), node.is_committed ? "Committed" : "Pending");
            if (!node.dependencies.empty()) {
                printf("  Depends on: ");
                for (uint64_t dep : node.dependencies) {
                    printf("%llu ", dep);
                }
                printf("\n");
            }
        }
    }
    printf("========================\n");
}

void SyncOptimizer::cleanupCompletedBuffers() {
    // 1. Update status of pending buffers
    auto it = pending_buffers.begin();
    while (it != pending_buffers.end()) {
        if ((*it).status >= MTLCommandBufferStatusCompleted) {
            // Mark in graph
            uint64_t id = getCmdId(*it);
            if (id > 0) {
                dependency_graph[id].is_completed = true;
            }
            it = pending_buffers.erase(it);
        } else {
            ++it;
        }
    }
    
    // 2. Prune graph - remove completed nodes that are no longer dependencies for active nodes
    // This is complex, so for now we just keep the graph growing until waitForAll or explicit reset
    // To prevent infinite growth, we can remove nodes that are completed AND have no incoming edges from active nodes
    // But for this task, simple status update is enough.
}
