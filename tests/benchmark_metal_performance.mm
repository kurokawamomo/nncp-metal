/*
 * benchmark_metal_performance.mm
 *
 * Performance Benchmark for Metal-Optimized NNCP
 * Targets: enwik8 (100MB)
 * Metrics: Throughput, Latency, GPU Utilization
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include "../src/neural/flow/layer_flow_optimizer.h"
#include "../src/neural/monitoring/gpu_utilization_monitor.h"

// Configuration
const size_t BATCH_SIZE = 4096; // High batch size for GPU efficiency
const size_t SEQ_LEN = 64;      // Context length
const size_t HIDDEN_SIZE = 128; // Small model for speed test
const size_t LAYERS = 4;
const size_t HEADS = 4;
const size_t VOCAB_SIZE = 256;

void print_progress(size_t current, size_t total, double start_time) {
    double now = (double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;
    double elapsed = now - start_time;
    double speed = (double)current / (1024.0 * 1024.0) / elapsed;
    double progress = (double)current / total * 100.0;
    
    printf("\rProgress: %.1f%% (%.2f MB/s) - Time: %.1fs", progress, speed, elapsed);
    fflush(stdout);
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        printf("=== NNCP Metal Performance Benchmark ===\n");
        
        // 1. Setup Metal Device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("Error: No Metal device found.\n");
            return -1;
        }
        printf("Device: %s\n", [device.name UTF8String]);
        
        // 2. Initialize Monitors
        GpuUtilizationMonitor monitor(device);
        monitor.startMonitoring();
        
        // 3. Initialize Flow Optimizer
        FlowOptimizerContext* ctx = flow_optimizer_create(device);
        if (!ctx) {
            printf("Error: Failed to create Flow Optimizer.\n");
            return -1;
        }
        
        // 4. Configure Transformer
        GPUTransformerConfig config;
        config.num_layers = LAYERS;
        config.hidden_size = HIDDEN_SIZE;
        config.num_heads = HEADS;
        config.ffn_size = HIDDEN_SIZE * 4;
        config.context_length = SEQ_LEN;
        config.vocab_size = VOCAB_SIZE;
        
        if (!flow_optimizer_setup_transformer(ctx, config)) {
            printf("Error: Failed to setup Transformer.\n");
            return -1;
        }
        
        // 5. Load Data (enwik8)
        const char* filename = "enwik8";
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            printf("Warning: enwik8 not found, creating dummy 100MB data.\n");
            // Create dummy data
        }
        
        size_t fileSize = file.is_open() ? file.tellg() : 100 * 1024 * 1024;
        std::vector<uint8_t> data(fileSize);
        if (file.is_open()) {
            file.seekg(0, std::ios::beg);
            file.read((char*)data.data(), fileSize);
            file.close();
        } else {
            // Fill with random data
            for (size_t i = 0; i < fileSize; i++) data[i] = rand() % 256;
        }
        
        printf("Data Size: %.2f MB\n", fileSize / (1024.0 * 1024.0));
        printf("Batch Size: %zu\n", BATCH_SIZE);
        
        // 6. Execution Loop
        printf("Starting Benchmark...\n");
        auto start = std::chrono::steady_clock::now();
        double start_time_sec = (double)std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count() / 1000.0;
        
        size_t processed = 0;
        std::vector<int32_t> batch_input(BATCH_SIZE * SEQ_LEN);
        float* output_logits = nullptr;
        
        // Pre-fill batch input (simplified: just copying chunks)
        // In real NNCP, context slides. Here we test throughput.
        
        int iterations = 0;
        while (processed < fileSize) {
            // Fill batch
            // For benchmark, we just reuse the buffer or fill with dummy
            // to avoid CPU bottleneck.
            // But let's do a simple copy to simulate data movement.
            size_t chunk = std::min(BATCH_SIZE * SEQ_LEN, fileSize - processed);
            for (size_t i = 0; i < chunk; i++) {
                batch_input[i] = data[processed + i];
            }
            
            // Execute
            // We use the async API
            bool queued = flow_optimizer_execute_batch(ctx, 
                                                      batch_input.data(), 
                                                      BATCH_SIZE, 
                                                      SEQ_LEN, 
                                                      FLOW_ENGINE_TRANSFORMER, 
                                                      &output_logits);
            
            if (!queued) {
                printf("\nError: Failed to queue batch.\n");
                break;
            }
            
            processed += chunk;
            iterations++;
            
            // Sync every N batches to keep memory in check (or rely on pool)
            // Phase 4 pipeline optimizer should handle this, but we force sync occasionally
            // to measure latency.
            if (iterations % 10 == 0) {
                flow_optimizer_sync(ctx);
                print_progress(processed, fileSize, start_time_sec);
            }
        }
        
        // Final Sync
        flow_optimizer_sync(ctx);
        
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        printf("\n\n=== Benchmark Results ===\n");
        printf("Total Time: %.2f s\n", diff.count());
        printf("Throughput: %.2f MB/s\n", fileSize / (1024.0 * 1024.0) / diff.count());
        printf("Tokens/sec: %.2f k/s\n", (fileSize / 1000.0) / diff.count());
        
        monitor.stopMonitoring();
        monitor.printReport();
        
        flow_optimizer_destroy(ctx);
    }
    return 0;
}
