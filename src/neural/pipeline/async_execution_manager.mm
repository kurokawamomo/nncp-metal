/*
 * async_execution_manager.mm
 *
 * Implementation of Async Execution Manager
 */

#include "async_execution_manager.h"
#include <iostream>
#include <chrono>

AsyncExecutionManager::AsyncExecutionManager(id<MTLDevice> dev, SyncOptimizer* syncOpt) 
    : device(dev), syncOptimizer(syncOpt), isRunning(false), nextTaskId(1) {
    start();
}

AsyncExecutionManager::~AsyncExecutionManager() {
    stop();
}

void AsyncExecutionManager::start() {
    if (isRunning) return;
    isRunning = true;
    executionThread = std::thread(&AsyncExecutionManager::executionLoop, this);
}

void AsyncExecutionManager::stop() {
    if (!isRunning) return;
    
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        isRunning = false;
    }
    queueCondition.notify_all();
    
    if (executionThread.joinable()) {
        executionThread.join();
    }
}

uint64_t AsyncExecutionManager::scheduleTask(ExecutionPriority priority, 
                                           std::function<void(id<MTLCommandBuffer>)> encodingBlock,
                                           std::function<void(void)> completionBlock) {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    ExecutionTask task;
    task.taskId = nextTaskId++;
    task.priority = priority;
    task.encodingBlock = encodingBlock;
    task.completionBlock = completionBlock;
    task.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    taskQueue.push(task);
    queueCondition.notify_one();
    
    return task.taskId;
}

void AsyncExecutionManager::waitForTask(uint64_t taskId) {
    std::unique_lock<std::mutex> lock(completionMutex);
    completionCondition.wait(lock, [this, taskId] {
        return completedTasks.count(taskId) > 0;
    });
}

void AsyncExecutionManager::waitForAll() {
    // Simple implementation: wait until queue is empty AND all submitted tasks are done.
    // This is a bit tricky because queue empty doesn't mean GPU is done.
    // We can rely on SyncOptimizer::waitForAll() but that waits for EVERYTHING.
    // Or we can just wait until our queue is empty and the current task finishes.
    
    // 1. Wait for queue to drain
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCondition.wait(lock, [this] {
            return taskQueue.empty() && isRunning;
        });
    }
    
    // 2. Wait for GPU
    if (syncOptimizer) {
        syncOptimizer->waitForAll();
    }
}

void AsyncExecutionManager::executionLoop() {
    while (true) {
        ExecutionTask task;
        
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondition.wait(lock, [this] {
                return !taskQueue.empty() || !isRunning;
            });
            
            if (!isRunning && taskQueue.empty()) {
                return;
            }
            
            task = taskQueue.top();
            taskQueue.pop();
        }
        
        // Get Command Buffer from SyncOptimizer
        // The label helps with debugging in Xcode GPU Frame Capture
        std::string label = "Task_" + std::to_string(task.taskId);
        id<MTLCommandBuffer> cmd = syncOptimizer->getCommandBuffer(label.c_str());
        
        if (cmd) {
            // Execute user encoding block
            if (task.encodingBlock) {
                task.encodingBlock(cmd);
            }
            
            // Add completion handler for task tracking
            // We need to capture task.id and completionBlock by value
            uint64_t tid = task.taskId;
            auto cb = task.completionBlock;
            
            // Note: We need to be careful about 'this' lifetime if manager is destroyed before callback.
            // But waitForAll usually handles it.
            // For robustness, we assume manager outlives tasks or we use shared_ptr.
            // Here we use a raw pointer but in a real system we'd use weak_ptr.
            
            [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
                if (cb) {
                    cb();
                }
                
                // Notify completion
                // We need a way to access the manager instance safely.
                // Since this is a C++ class, we can't easily capture 'this' safely without shared_from_this.
                // For this implementation, we assume the manager is alive.
                // Or we can use a static map or similar.
                // Let's assume the user calls waitForTask on the instance.
                // We'll use a lambda that captures 'this'.
                // Ideally, AsyncExecutionManager should be a shared_ptr.
            }];
            
            // Commit via SyncOptimizer (async)
            syncOptimizer->commitCommandBuffer(cmd, false);
            
            // We mark it as "completed" in our tracking only when GPU finishes?
            // Or when submitted?
            // waitForTask usually implies waiting for GPU execution.
            // So we should update completedTasks inside the completion handler.
            // However, accessing member variables from ObjC block requires care.
            
            // Workaround: We'll just track submission here, and let SyncOptimizer handle the GPU wait.
            // But waitForTask needs to know when SPECIFIC task is done.
            // Let's use a separate mechanism:
            // We can block in waitForTask on the command buffer itself if we stored it?
            // But we don't store it.
            // Let's update the set in the handler.
            // We need to ensure thread safety.
            
            [cmd addCompletedHandler:^(id<MTLCommandBuffer> c) {
                 std::lock_guard<std::mutex> lock(this->completionMutex);
                 this->completedTasks.insert(tid);
                 this->completionCondition.notify_all();
            }];
        }
    }
}
