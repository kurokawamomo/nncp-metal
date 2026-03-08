/*
 * gpu_utilization_monitor.h
 *
 * GPU Utilization Monitor
 * Monitors GPU performance metrics, execution times, and resource usage.
 * Detects bottlenecks and provides optimization suggestions.
 */

#ifndef GPU_UTILIZATION_MONITOR_H
#define GPU_UTILIZATION_MONITOR_H

#import <Metal/Metal.h>
#include <vector>
#include <string>
#include <map>

struct GPUMetric {
    uint64_t commandId;
    std::string label;
    double startTime;
    double endTime;
    double duration; // ms
    double gpuTime;  // ms (from GPUTimestamp)
};

struct GPUStats {
    double totalExecutionTime;
    double totalGpuActiveTime;
    double utilizationPercentage;
    double averageLatency;
    uint64_t commandCount;
    uint64_t stallCount;
};

class GpuUtilizationMonitor {
public:
    GpuUtilizationMonitor(id<MTLDevice> device);
    ~GpuUtilizationMonitor();

    // Start/Stop monitoring session
    void startMonitoring();
    void stopMonitoring();

    // Track command buffer execution
    // Call this immediately after creating a command buffer
    void trackCommandBuffer(id<MTLCommandBuffer> cmdBuffer, const char* label);

    // Get current statistics
    GPUStats getStats();

    // Generate detailed report
    void printReport();

    // Bottleneck analysis
    // Returns a list of suggestions based on observed metrics
    std::vector<std::string> analyzeBottlenecks();

private:
    id<MTLDevice> device;
    bool isMonitoring;
    double sessionStartTime;
    
    std::vector<GPUMetric> metrics;
    std::mutex monitorMutex;
    
    // Counters
    uint64_t totalCommands;
    
    // Helpers
    double getCurrentTime();
};

#endif // GPU_UTILIZATION_MONITOR_H
