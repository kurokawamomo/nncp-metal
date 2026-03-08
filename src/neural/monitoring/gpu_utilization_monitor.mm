/*
 * gpu_utilization_monitor.mm
 *
 * Implementation of GPU Utilization Monitor
 */

#include "gpu_utilization_monitor.h"
#include <iostream>
#include <sys/time.h>
#include <algorithm>
#include <iomanip>

GpuUtilizationMonitor::GpuUtilizationMonitor(id<MTLDevice> dev) 
    : device(dev), isMonitoring(false), totalCommands(0) {
}

GpuUtilizationMonitor::~GpuUtilizationMonitor() {
    stopMonitoring();
}

double GpuUtilizationMonitor::getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

void GpuUtilizationMonitor::startMonitoring() {
    std::lock_guard<std::mutex> lock(monitorMutex);
    isMonitoring = true;
    sessionStartTime = getCurrentTime();
    metrics.clear();
    totalCommands = 0;
    printf("[GpuMonitor] Monitoring started.\n");
}

void GpuUtilizationMonitor::stopMonitoring() {
    std::lock_guard<std::mutex> lock(monitorMutex);
    if (!isMonitoring) return;
    isMonitoring = false;
    printf("[GpuMonitor] Monitoring stopped.\n");
}

void GpuUtilizationMonitor::trackCommandBuffer(id<MTLCommandBuffer> cmdBuffer, const char* label) {
    if (!isMonitoring || !cmdBuffer) return;
    
    std::string labelStr = label ? label : "Unnamed";
    uint64_t cmdId = ++totalCommands;
    double submitTime = getCurrentTime();
    
    // Use completion handler to capture timing
    // Note: We capture 'this' assuming monitor lives longer than command buffer.
    // In production, use weak_ptr or shared_ptr.
    
    [cmdBuffer addCompletedHandler:^(id<MTLCommandBuffer> c) {
        if (c.status == MTLCommandBufferStatusCompleted) {
            double completeTime = this->getCurrentTime();
            double gpuDuration = 0.0;
            
            // Try to get GPU timings if available (requires .gpuStartTime and .gpuEndTime)
            // These properties are available on macOS 10.15+ / iOS 14+
            if (@available(macOS 10.15, iOS 14.0, *)) {
                gpuDuration = (c.GPUEndTime - c.GPUStartTime);
            } else {
                // Fallback to CPU wall clock (inaccurate for GPU util)
                gpuDuration = completeTime - submitTime; 
            }
            
            std::lock_guard<std::mutex> lock(this->monitorMutex);
            if (this->isMonitoring) {
                GPUMetric m;
                m.commandId = cmdId;
                m.label = labelStr;
                m.startTime = submitTime;
                m.endTime = completeTime;
                m.duration = (completeTime - submitTime) * 1000.0; // ms
                m.gpuTime = gpuDuration * 1000.0; // ms
                this->metrics.push_back(m);
            }
        }
    }];
}

GPUStats GpuUtilizationMonitor::getStats() {
    std::lock_guard<std::mutex> lock(monitorMutex);
    
    GPUStats stats = {0};
    stats.commandCount = metrics.size();
    
    if (metrics.empty()) return stats;
    
    double totalGpu = 0.0;
    double totalDuration = 0.0;
    double minStart = metrics[0].startTime;
    double maxEnd = metrics[0].endTime;
    
    for (const auto& m : metrics) {
        totalGpu += m.gpuTime;
        totalDuration += m.duration;
        if (m.startTime < minStart) minStart = m.startTime;
        if (m.endTime > maxEnd) maxEnd = m.endTime;
    }
    
    double sessionDuration = (isMonitoring ? getCurrentTime() : maxEnd) - sessionStartTime;
    if (sessionDuration < 0.001) sessionDuration = 0.001;
    
    stats.totalExecutionTime = sessionDuration * 1000.0;
    stats.totalGpuActiveTime = totalGpu;
    // Utilization: Sum of GPU times / Wall time. 
    // Note: If parallel execution happens, sum can exceed wall time (util > 100%).
    // This is desirable to see parallelism.
    // To get "busy %", we would need to merge intervals.
    // For now, simple ratio.
    stats.utilizationPercentage = (totalGpu / (sessionDuration * 1000.0)) * 100.0;
    stats.averageLatency = totalDuration / metrics.size();
    
    return stats;
}

std::vector<std::string> GpuUtilizationMonitor::analyzeBottlenecks() {
    std::lock_guard<std::mutex> lock(monitorMutex);
    std::vector<std::string> suggestions;
    
    if (metrics.empty()) {
        suggestions.push_back("No data collected.");
        return suggestions;
    }
    
    double totalGpu = 0.0;
    double totalCpuWait = 0.0;
    
    for (const auto& m : metrics) {
        totalGpu += m.gpuTime;
        // CPU Wait = Duration - GPU Time (roughly time spent in queue or scheduling)
        double wait = m.duration - m.gpuTime;
        if (wait > 0) totalCpuWait += wait;
    }
    
    double avgGpu = totalGpu / metrics.size();
    double avgWait = totalCpuWait / metrics.size();
    
    if (avgWait > avgGpu) {
        suggestions.push_back("High CPU/Queue Latency: Command buffers are spending more time waiting than executing.");
        suggestions.push_back("  -> Suggestion: Check for dependencies blocking execution or excessive CPU-GPU synchronization.");
    }
    
    if (avgGpu < 0.1) { // < 100us
        suggestions.push_back("Tiny Kernels Detected: Average GPU execution time is very low.");
        suggestions.push_back("  -> Suggestion: Batch small operations or fuse kernels to reduce launch overhead.");
    }
    
    // Check for gaps
    // Sort by start time
    std::vector<GPUMetric> sorted = metrics;
    std::sort(sorted.begin(), sorted.end(), [](const GPUMetric& a, const GPUMetric& b) {
        return a.startTime < b.startTime;
    });
    
    double idleTime = 0.0;
    if (sorted.size() > 1) {
        for (size_t i = 0; i < sorted.size() - 1; i++) {
            double gap = sorted[i+1].startTime - sorted[i].endTime;
            if (gap > 0.001) { // > 1ms gap
                idleTime += gap;
            }
        }
    }
    
    if (idleTime > 100.0) { // > 100ms total idle
        suggestions.push_back("GPU Idle Gaps: Significant idle time detected between commands.");
        suggestions.push_back("  -> Suggestion: Use double buffering or pipeline more batches to keep GPU busy.");
    }
    
    return suggestions;
}

void GpuUtilizationMonitor::printReport() {
    GPUStats stats = getStats();
    std::vector<std::string> suggestions = analyzeBottlenecks();
    
    std::cout << "\n=== GPU Utilization Report ===\n";
    std::cout << "Total Commands: " << stats.commandCount << "\n";
    std::cout << "Session Duration: " << std::fixed << std::setprecision(2) << stats.totalExecutionTime << " ms\n";
    std::cout << "Total GPU Time:   " << stats.totalGpuActiveTime << " ms\n";
    std::cout << "Utilization:      " << stats.utilizationPercentage << " %\n";
    std::cout << "Avg Latency:      " << stats.averageLatency << " ms\n";
    
    std::cout << "\n--- Optimization Suggestions ---\n";
    for (const auto& s : suggestions) {
        std::cout << "- " << s << "\n";
    }
    std::cout << "==============================\n";
}
