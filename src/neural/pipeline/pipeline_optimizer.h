/*
 * pipeline_optimizer.h
 *
 * Pipeline Efficiency Optimizer
 * Analyzes layer dependencies to maximize parallel execution.
 * Models the neural network as a Directed Acyclic Graph (DAG) and schedules
 * tasks to the AsyncExecutionManager based on dependency analysis.
 */

#ifndef PIPELINE_OPTIMIZER_H
#define PIPELINE_OPTIMIZER_H

#include "async_execution_manager.h"
#include <vector>
#include <string>
#include <map>
#include <set>

// Pipeline Node Type
enum class PipelineNodeType {
    Input,
    Compute,
    Transfer,
    Output
};

// Pipeline Node
struct PipelineNode {
    uint64_t id;
    std::string name;
    PipelineNodeType type;
    
    // Dependencies
    std::vector<uint64_t> inputs;  // Nodes that must complete before this one
    std::vector<uint64_t> outputs; // Nodes that depend on this one
    
    // Execution Logic
    std::function<void(id<MTLCommandBuffer>)> executionBlock;
    
    // Metadata
    float estimatedCost; // Estimated execution time in ms
    bool isScheduled;
    bool isCompleted;
};

class PipelineOptimizer {
public:
    PipelineOptimizer(AsyncExecutionManager* asyncMgr);
    ~PipelineOptimizer();

    // Graph Construction
    uint64_t addNode(std::string name, PipelineNodeType type, 
                     std::function<void(id<MTLCommandBuffer>)> executionBlock);
    
    void addDependency(uint64_t producerId, uint64_t consumerId);

    // Analysis
    // Returns true if the graph is valid (DAG)
    bool validateGraph();
    
    // Execution
    // Schedules ready nodes to AsyncExecutionManager
    void executeGraph();
    
    // Reset graph state for re-execution
    void resetGraph();

    // Stats
    void printGraphStats();

private:
    AsyncExecutionManager* asyncManager;
    
    std::map<uint64_t, PipelineNode> nodes;
    uint64_t nextNodeId;
    
    // Execution State
    std::set<uint64_t> readyNodes;
    std::set<uint64_t> runningNodes;
    std::set<uint64_t> completedNodes;
    std::mutex graphMutex;
    
    // Internal helpers
    void scheduleReadyNodes();
    void onNodeCompleted(uint64_t nodeId);
};

#endif // PIPELINE_OPTIMIZER_H
