/*
 * pipeline_optimizer.mm
 *
 * Implementation of Pipeline Efficiency Optimizer
 */

#include "pipeline_optimizer.h"
#include <iostream>
#include <stack>

PipelineOptimizer::PipelineOptimizer(AsyncExecutionManager* asyncMgr) 
    : asyncManager(asyncMgr), nextNodeId(1) {
}

PipelineOptimizer::~PipelineOptimizer() {
}

uint64_t PipelineOptimizer::addNode(std::string name, PipelineNodeType type, 
                                  std::function<void(id<MTLCommandBuffer>)> executionBlock) {
    std::lock_guard<std::mutex> lock(graphMutex);
    
    PipelineNode node;
    node.id = nextNodeId++;
    node.name = name;
    node.type = type;
    node.executionBlock = executionBlock;
    node.estimatedCost = 1.0f; // Default cost
    node.isScheduled = false;
    node.isCompleted = false;
    
    nodes[node.id] = node;
    return node.id;
}

void PipelineOptimizer::addDependency(uint64_t producerId, uint64_t consumerId) {
    std::lock_guard<std::mutex> lock(graphMutex);
    
    if (nodes.find(producerId) == nodes.end() || nodes.find(consumerId) == nodes.end()) {
        printf("[PipelineOptimizer] Error: Invalid node IDs for dependency %llu -> %llu\n", producerId, consumerId);
        return;
    }
    
    nodes[consumerId].inputs.push_back(producerId);
    nodes[producerId].outputs.push_back(consumerId);
}

bool PipelineOptimizer::validateGraph() {
    std::lock_guard<std::mutex> lock(graphMutex);
    
    // Cycle detection using DFS
    std::map<uint64_t, bool> visited;
    std::map<uint64_t, bool> recursionStack;
    
    for (auto& pair : nodes) {
        visited[pair.first] = false;
        recursionStack[pair.first] = false;
    }
    
    std::function<bool(uint64_t)> hasCycle = [&](uint64_t nodeId) -> bool {
        visited[nodeId] = true;
        recursionStack[nodeId] = true;
        
        for (uint64_t childId : nodes[nodeId].outputs) {
            if (!visited[childId]) {
                if (hasCycle(childId)) return true;
            } else if (recursionStack[childId]) {
                return true;
            }
        }
        
        recursionStack[nodeId] = false;
        return false;
    };
    
    for (auto& pair : nodes) {
        if (!visited[pair.first]) {
            if (hasCycle(pair.first)) {
                printf("[PipelineOptimizer] Error: Cycle detected in pipeline graph.\n");
                return false;
            }
        }
    }
    
    return true;
}

void PipelineOptimizer::resetGraph() {
    std::lock_guard<std::mutex> lock(graphMutex);
    for (auto& pair : nodes) {
        pair.second.isScheduled = false;
        pair.second.isCompleted = false;
    }
    readyNodes.clear();
    runningNodes.clear();
    completedNodes.clear();
}

void PipelineOptimizer::executeGraph() {
    // Initial scheduling: Find nodes with no pending inputs
    // In a real scenario, this might be called repeatedly or run in a loop.
    // Here we trigger the first wave.
    
    {
        std::lock_guard<std::mutex> lock(graphMutex);
        
        // Reset if needed? Assuming executeGraph is called on fresh/reset graph.
        
        for (auto& pair : nodes) {
            PipelineNode& node = pair.second;
            if (node.inputs.empty()) {
                readyNodes.insert(node.id);
            }
        }
    }
    
    scheduleReadyNodes();
}

void PipelineOptimizer::scheduleReadyNodes() {
    std::lock_guard<std::mutex> lock(graphMutex);
    
    if (readyNodes.empty() && runningNodes.empty() && completedNodes.size() < nodes.size()) {
        // Deadlock or logic error?
        // If we have nodes but none are ready or running, and not all complete.
        // This could happen if graph has cycle (but validateGraph checks that).
        return;
    }
    
    auto it = readyNodes.begin();
    while (it != readyNodes.end()) {
        uint64_t nodeId = *it;
        PipelineNode& node = nodes[nodeId];
        
        if (!node.isScheduled) {
            node.isScheduled = true;
            runningNodes.insert(nodeId);
            
            // Schedule to AsyncManager
            // We use a callback to notify completion and trigger next steps
            // Note: We capture 'this' which is dangerous if optimizer is destroyed.
            // Assuming optimizer lives long enough.
            
            asyncManager->scheduleTask(ExecutionPriority::Normal, 
                node.executionBlock,
                [this, nodeId]() {
                    this->onNodeCompleted(nodeId);
                }
            );
        }
        
        it = readyNodes.erase(it);
    }
}

void PipelineOptimizer::onNodeCompleted(uint64_t nodeId) {
    // This is called from AsyncManager's completion thread
    // We need to update graph state and schedule dependents
    
    std::lock_guard<std::mutex> lock(graphMutex);
    
    runningNodes.erase(nodeId);
    completedNodes.insert(nodeId);
    nodes[nodeId].isCompleted = true;
    
    // Check dependents
    for (uint64_t childId : nodes[nodeId].outputs) {
        PipelineNode& child = nodes[childId];
        
        // Check if all inputs of child are complete
        bool allInputsDone = true;
        for (uint64_t inputId : child.inputs) {
            if (!nodes[inputId].isCompleted) {
                allInputsDone = false;
                break;
            }
        }
        
        if (allInputsDone && !child.isScheduled) {
            readyNodes.insert(childId);
        }
    }
    
    // Release lock before calling scheduleReadyNodes to avoid recursion issues if we were to inline execute
    // But scheduleReadyNodes takes lock. So we must unlock first.
    // However, scheduleReadyNodes is safe to call from here (different thread context usually, or re-entrant lock?)
    // std::mutex is NOT re-entrant.
    // So we unlock here.
    lock.unlock();
    
    scheduleReadyNodes();
}

void PipelineOptimizer::printGraphStats() {
    std::lock_guard<std::mutex> lock(graphMutex);
    printf("=== Pipeline Graph Stats ===\n");
    printf("Total Nodes: %zu\n", nodes.size());
    printf("Completed: %zu\n", completedNodes.size());
    printf("Running: %zu\n", runningNodes.size());
    printf("Pending: %zu\n", nodes.size() - completedNodes.size() - runningNodes.size());
    printf("============================\n");
}
