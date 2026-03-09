/*
 * async_execution_manager.h
 *
 * Async Execution Manager
 * Manages concurrent execution of multiple command buffers and scheduling.
 * Integrates with SyncOptimizer for resource conflict avoidance.
 */

#ifndef ASYNC_EXECUTION_MANAGER_H
#define ASYNC_EXECUTION_MANAGER_H

#import <Metal/Metal.h>
#include <functional>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <set>
#include "sync_optimizer.h"

// Task Priority
enum class ExecutionPriority {
    Low,
    Normal,
    High,
    Critical
};

// Execution Task
struct ExecutionTask {
    uint64_t taskId;
    ExecutionPriority priority;
    std::function<void(id<MTLCommandBuffer>)> encodingBlock;
    std::function<void(void)> completionBlock;
    uint64_t timestamp;
    
    // Comparison for priority queue (Higher priority first)
    bool operator<(const ExecutionTask& other) const {
        if (priority != other.priority) {
            return priority < other.priority;
        }
        return timestamp > other.timestamp; // FIFO for same priority
    }
};

class AsyncExecutionManager {
public:
    AsyncExecutionManager(id<MTLDevice> device, SyncOptimizer* syncOpt);
    ~AsyncExecutionManager();

    // Schedule a task for execution
    // Returns a task ID
    uint64_t scheduleTask(ExecutionPriority priority, 
                          std::function<void(id<MTLCommandBuffer>)> encodingBlock,
                          std::function<void(void)> completionBlock = nullptr);

    // Wait for a specific task to complete
    void waitForTask(uint64_t taskId);

    // Wait for all tasks to complete
    void waitForAll();

    // Start/Stop the execution loop
    void start();
    void stop();

private:
    id<MTLDevice> device;
    SyncOptimizer* syncOptimizer;
    
    std::priority_queue<ExecutionTask> taskQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    
    std::thread executionThread;
    bool isRunning;
    uint64_t nextTaskId;
    
    // Track completed tasks
    std::set<uint64_t> completedTasks;
    std::mutex completionMutex;
    std::condition_variable completionCondition;
    
    void executionLoop();
};

#endif // ASYNC_EXECUTION_MANAGER_H
