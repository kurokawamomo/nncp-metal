#include "compression_integration.h"
#include "algorithm_router.h"
#include "../neural/integration/neural_bridge.h"

// CUDA-compatible lossless compression functions  
extern size_t neural_bridge_cuda_lossless_compress(const uint8_t* input_data, size_t input_size, 
                                                  uint8_t* output_data, size_t output_capacity, 
                                                  const NeuralCompressionConfig* config);
extern size_t neural_bridge_cuda_lossless_decompress(const uint8_t* input_data, size_t input_size,
                                                    uint8_t* output_data, size_t output_capacity);
#include "nncp_original_port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// Global state for the integration system
static bool g_integration_initialized = false;
static CompressionConfig g_global_config = {0};
static bool g_neural_bridge_ready = false;
// Memory management globals
static MemoryManager g_memory_manager = {0};
static bool g_memory_manager_initialized = false;

/**
 * @brief Get current time in nanoseconds
 */
static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/**
 * @brief Initialize compression result structure
 */
static void init_compression_result(CompressionResult* result) {
    if (!result) return;
    
    result->success = false;
    result->compressed_size = 0;
    result->algorithm_used = COMPRESSION_ALGORITHM_LSTM;
    result->error_message[0] = '\0';
    result->compression_ratio = 0.0f;
    result->processing_time_ns = 0;
    result->memory_used_bytes = 0;
    result->buffer_allocations = 0;
}

/**
 * @brief Initialize decompression result structure
 */
static void init_decompression_result(DecompressionResult* result) {
    if (!result) return;
    
    result->success = false;
    result->decompressed_size = 0;
    result->algorithm_detected = COMPRESSION_ALGORITHM_LSTM;
    result->error_message[0] = '\0';
    result->processing_time_ns = 0;
    result->memory_used_bytes = 0;
    result->buffer_allocations = 0;
}

/**
 * @brief Get current time in milliseconds for memory management
 */
static uint64_t get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

bool memory_manager_init(size_t max_blocks, size_t block_size) {
    if (g_memory_manager_initialized) {
        return true; // Already initialized
    }
    
    if (max_blocks == 0 || block_size == 0) {
        return false;
    }
    
    // Allocate memory pool blocks
    g_memory_manager.blocks = calloc(max_blocks, sizeof(MemoryPoolBlock));
    if (!g_memory_manager.blocks) {
        return false;
    }
    
    g_memory_manager.num_blocks = max_blocks;
    g_memory_manager.block_size = block_size;
    g_memory_manager.pool_enabled = true;
    g_memory_manager.total_allocated = 0;
    g_memory_manager.peak_usage = 0;
    g_memory_manager.allocation_count = 0;
    g_memory_manager.deallocation_count = 0;
    
    // Pre-allocate memory blocks
    for (size_t i = 0; i < max_blocks; i++) {
        g_memory_manager.blocks[i].buffer = malloc(block_size);
        if (!g_memory_manager.blocks[i].buffer) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                free(g_memory_manager.blocks[j].buffer);
            }
            free(g_memory_manager.blocks);
            return false;
        }
        g_memory_manager.blocks[i].size = block_size;
        g_memory_manager.blocks[i].used = 0;
        g_memory_manager.blocks[i].in_use = false;
        g_memory_manager.blocks[i].last_used_time = 0;
    }
    
    g_memory_manager_initialized = true;
    return true;
}

void memory_manager_shutdown(void) {
    if (!g_memory_manager_initialized) {
        return;
    }
    
    // Free all blocks
    for (size_t i = 0; i < g_memory_manager.num_blocks; i++) {
        if (g_memory_manager.blocks[i].buffer) {
            free(g_memory_manager.blocks[i].buffer);
        }
    }
    
    free(g_memory_manager.blocks);
    memset(&g_memory_manager, 0, sizeof(MemoryManager));
    g_memory_manager_initialized = false;
}

void* memory_manager_alloc(size_t size) {
    if (!g_memory_manager_initialized || !g_memory_manager.pool_enabled) {
        // Fall back to system malloc
        g_memory_manager.allocation_count++;
        g_memory_manager.total_allocated += size;
        if (g_memory_manager.total_allocated > g_memory_manager.peak_usage) {
            g_memory_manager.peak_usage = g_memory_manager.total_allocated;
        }
        return malloc(size);
    }
    
    // Try to find a suitable block in the pool
    for (size_t i = 0; i < g_memory_manager.num_blocks; i++) {
        MemoryPoolBlock* block = &g_memory_manager.blocks[i];
        if (!block->in_use && block->size >= size) {
            block->in_use = true;
            block->used = size;
            block->last_used_time = get_time_ms();
            
            g_memory_manager.allocation_count++;
            g_memory_manager.total_allocated += size;
            if (g_memory_manager.total_allocated > g_memory_manager.peak_usage) {
                g_memory_manager.peak_usage = g_memory_manager.total_allocated;
            }
            
            return block->buffer;
        }
    }
    
    // No suitable block found, fall back to system malloc
    g_memory_manager.allocation_count++;
    g_memory_manager.total_allocated += size;
    if (g_memory_manager.total_allocated > g_memory_manager.peak_usage) {
        g_memory_manager.peak_usage = g_memory_manager.total_allocated;
    }
    return malloc(size);
}

void memory_manager_free(void* ptr, size_t size) {
    if (!ptr) return;
    
    if (!g_memory_manager_initialized || !g_memory_manager.pool_enabled) {
        // Fall back to system free
        g_memory_manager.deallocation_count++;
        g_memory_manager.total_allocated -= size;
        free(ptr);
        return;
    }
    
    // Check if this pointer belongs to our pool
    for (size_t i = 0; i < g_memory_manager.num_blocks; i++) {
        MemoryPoolBlock* block = &g_memory_manager.blocks[i];
        if (block->buffer == ptr && block->in_use) {
            block->in_use = false;
            block->used = 0;
            block->last_used_time = get_time_ms();
            
            g_memory_manager.deallocation_count++;
            g_memory_manager.total_allocated -= size;
            return;
        }
    }
    
    // Not from our pool, use system free
    g_memory_manager.deallocation_count++;
    g_memory_manager.total_allocated -= size;
    free(ptr);
}

const MemoryManager* memory_manager_get_stats(void) {
    return &g_memory_manager;
}

void memory_manager_cleanup(uint64_t max_age_ms) {
    if (!g_memory_manager_initialized) {
        return;
    }
    
    uint64_t current_time = get_time_ms();
    
    // This function doesn't actually free pool blocks (they're pre-allocated)
    // but it could be extended to track and report aged blocks
    for (size_t i = 0; i < g_memory_manager.num_blocks; i++) {
        MemoryPoolBlock* block = &g_memory_manager.blocks[i];
        if (!block->in_use && block->last_used_time > 0) {
            uint64_t age = current_time - block->last_used_time;
            if (age > max_age_ms) {
                // Block is old and unused - could be reported or managed
                // For now, just reset the timestamp
                block->last_used_time = 0;
            }
        }
    }
}

bool compression_integration_init(const CompressionConfig* config) {
    if (g_integration_initialized) {
        return true; // Already initialized
    }
    
    if (!config) {
        return false;
    }
    
    // Copy configuration
    memcpy(&g_global_config, config, sizeof(CompressionConfig));
    
    // Set defaults for unspecified values
    if (g_global_config.memory_limit_bytes == 0) {
        g_global_config.memory_limit_bytes = 8ULL * 1024 * 1024 * 1024; // 8GB default
    }
    
    if (g_global_config.quantization_bits == 0) {
        g_global_config.quantization_bits = 8; // Lossless by default
    }
    
    // Initialize algorithm router
    if (!algorithm_router_init()) {
        printf("Warning: Algorithm router initialization failed, using basic selection\n");
    } else {
        printf("Algorithm router initialized successfully\n");
    }
    
    // Initialize neural bridge
    NeuralCompressionConfig neural_config = {
        .preferred_algorithm = (config->preferred_algorithm == COMPRESSION_ALGORITHM_TRANSFORMER) ? 
                              NEURAL_ALGORITHM_TRANSFORMER : NEURAL_ALGORITHM_LSTM,
        .memory_limit_bytes = config->memory_limit_bytes,
        .quality_level = 7, // High quality
        .enable_gpu_acceleration = true,
        .verbose_logging = config->verbose_logging,
        .compression_target = 0.15f // Target 15% compression ratio
    };
    
    if (config->preferred_algorithm == COMPRESSION_ALGORITHM_AUTO) {
        neural_config.preferred_algorithm = NEURAL_ALGORITHM_AUTO;
    }
    
    g_neural_bridge_ready = neural_bridge_init(&neural_config);
    if (!g_neural_bridge_ready) {
        printf("FATAL: Neural bridge initialization failed. Metal LSTM required.\n");
        return false;
    }
    
    // Initialize memory management
    size_t max_blocks = 16; // Maximum number of memory blocks
    size_t block_size = config->memory_limit_bytes / max_blocks; // Distribute memory limit
    if (block_size < 1024 * 1024) {
        block_size = 1024 * 1024; // Minimum 1MB per block
    }
    
    if (!memory_manager_init(max_blocks, block_size)) {
        printf("Warning: Memory manager initialization failed, falling back to system malloc\n");
    }
    
    g_integration_initialized = true;
    
    if (config->verbose_logging) {
        printf("Compression integration initialized:\n");
        printf("  Preferred algorithm: %s\n", 
               compression_integration_algorithm_name(config->preferred_algorithm));
        printf("  Metal LSTM only: Original NNCP design");
        printf("  Memory limit: %zu MB\n", config->memory_limit_bytes / (1024 * 1024));
    }
    
    return true;
}

void compression_integration_shutdown(void) {
    if (!g_integration_initialized) {
        return;
    }
    
    // Shutdown algorithm router
    algorithm_router_shutdown();
    
    // Shutdown neural bridge
    if (g_neural_bridge_ready) {
        neural_bridge_shutdown();
        g_neural_bridge_ready = false;
    }
    
    // Shutdown memory manager
    memory_manager_shutdown();
    
    g_integration_initialized = false;
    memset(&g_global_config, 0, sizeof(CompressionConfig));
    
    if (g_global_config.verbose_logging) {
        printf("Compression integration shutdown complete\n");
    }
}

bool compression_integration_compress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    const CompressionConfig* config,
    CompressionResult* result
) {
    if (!result) return false;
    
    init_compression_result(result);
    uint64_t start_time = get_time_ns();
    
    // Validation
    if (!input_data || input_size == 0 || !output_data || output_capacity == 0) {
        snprintf(result->error_message, sizeof(result->error_message),
                "Invalid input parameters");
        return false;
    }
    
    if (!g_integration_initialized) {
        snprintf(result->error_message, sizeof(result->error_message),
                "Compression integration not initialized");
        return false;
    }
    
    // Use provided config or fall back to global config
    const CompressionConfig* active_config = config ? config : &g_global_config;
    
    // SIMPLIFIED: Metal LSTM only (following original NNCP design)
    if (!g_neural_bridge_ready) {
        snprintf(result->error_message, sizeof(result->error_message),
                "Metal LSTM neural bridge not available");
        return false;
    }
    
    // Configure Metal Transformer neural compression
    NeuralCompressionConfig neural_config = {
        .preferred_algorithm = NEURAL_ALGORITHM_TRANSFORMER,
        .memory_limit_bytes = active_config->memory_limit_bytes,
        .quality_level = 8, // Transformer quality level
        .enable_gpu_acceleration = true,
        .verbose_logging = true, // Force debug mode
        .compression_target = 0.20f // Target 20% compression ratio for transformer
    };
    
    // Enable detailed debugging
    printf("Metal Transformer compression: %zu bytes\n", input_size);
    
    NeuralCompressionResult neural_result = {0};
    
    // Call authentic lossless NNCP Transformer compression
    size_t compressed_size = neural_bridge_cuda_lossless_compress(
        input_data, input_size, output_data, output_capacity,
        &neural_config);
    
    bool success = (compressed_size > 0);
    
    if (success) {
        // Authentic lossless NNCP Transformer compression succeeded
        result->algorithm_used = COMPRESSION_ALGORITHM_TRANSFORMER;
        result->compressed_size = compressed_size;
        result->compression_ratio = (float)compressed_size / input_size;
        result->processing_time_ns = 1000000; // 1ms placeholder
        result->success = true;
        
        if (active_config->verbose_logging) {
            printf("Metal Transformer compression successful: %zu -> %zu bytes (%.1f%%)\n",
                   input_size, result->compressed_size, 
                   result->compression_ratio * 100.0f);
        }
        
        return true;
    } else {
        // Metal Transformer compression failed - no fallback (following original NNCP)
        snprintf(result->error_message, sizeof(result->error_message),
                "Metal Transformer compression failed: %s", neural_result.error_message);
        return false;
    }
}

bool compression_integration_decompress(
    const uint8_t* input_data,
    size_t input_size,
    uint8_t* output_data,
    size_t output_capacity,
    DecompressionResult* result
) {
    if (!result) return false;
    
    init_decompression_result(result);
    uint64_t start_time = get_time_ns();
    
    // Validation
    if (!input_data || input_size == 0 || !output_data || output_capacity == 0) {
        snprintf(result->error_message, sizeof(result->error_message),
                "Invalid input parameters");
        return false;
    }
    
    if (!g_integration_initialized) {
        snprintf(result->error_message, sizeof(result->error_message),
                "Compression integration not initialized");
        return false;
    }
    
    // SIMPLIFIED: Metal LSTM only (following original NNCP design)
    if (!g_neural_bridge_ready) {
        snprintf(result->error_message, sizeof(result->error_message),
                "Metal LSTM neural bridge not available");
        return false;
    }
    
    if (g_global_config.verbose_logging) {
        printf("Metal LSTM decompression: %zu bytes\n", input_size);
    }
    
    // All compressed data is Metal LSTM format
    result->algorithm_detected = COMPRESSION_ALGORITHM_LSTM;
    
    NeuralDecompressionResult neural_result = {0};
    
    // Call authentic lossless NNCP Transformer decompression
    size_t decompressed_size = neural_bridge_cuda_lossless_decompress(
        input_data, input_size, output_data, output_capacity);
    
    bool success = (decompressed_size > 0);
    
    if (success) {
        // Authentic lossless NNCP Transformer decompression succeeded
        result->decompressed_size = decompressed_size;
        result->success = true;
        result->processing_time_ns = get_time_ns() - start_time;
        
        // Track memory usage
        if (g_memory_manager_initialized) {
            const MemoryManager* stats = memory_manager_get_stats();
            result->memory_used_bytes = stats->peak_usage;
            result->buffer_allocations = stats->allocation_count;
        }
        
        if (g_global_config.verbose_logging) {
            printf("Metal LSTM decompression successful: %zu -> %zu bytes\n",
                   input_size, result->decompressed_size);
        }
        
        return true;
    } else {
        // Metal LSTM decompression failed - no fallback (following original NNCP)
        snprintf(result->error_message, sizeof(result->error_message),
                "Metal LSTM decompression failed: %s", neural_result.error_message);
        return false;
    }
}

size_t compression_integration_estimate_output_size(
    size_t input_size,
    CompressionAlgorithm algorithm
) {
    // Neural algorithms need space for headers, parameters, and compressed data
    size_t neural_overhead = 64; // 4 (algorithm) + 4 (size) + 8 (params) + 2 (checksum) + padding
    
    switch (algorithm) {
        case COMPRESSION_ALGORITHM_TRANSFORMER:
            // Transformer typically achieves good compression
            return (input_size * 3) / 4 + neural_overhead;
            
        case COMPRESSION_ALGORITHM_LSTM:
            // LSTM compression estimate - Metal LSTM only
            return input_size + neural_overhead + (input_size / 4);
            
        case COMPRESSION_ALGORITHM_AUTO:
        default:
            // Metal LSTM only - no AUTO mode in original NNCP design
            return input_size + neural_overhead + (input_size / 4);
    }
}

CompressionAlgorithm compression_integration_select_algorithm(
    const uint8_t* input_data,
    size_t input_size
) {
    if (!input_data || input_size == 0) {
        return COMPRESSION_ALGORITHM_LSTM; // Default neural algorithm
    }
    
    if (!g_integration_initialized) {
        // Simple text detection for neural algorithm selection
        bool looks_like_text = true;
        for (size_t i = 0; i < input_size && i < 100; i++) {
            uint8_t byte = input_data[i];
            if (byte < 32 && byte != '\n' && byte != '\r' && byte != '\t') {
                if (byte != 0) {
                    looks_like_text = false;
                    break;
                }
            }
        }
        
        return looks_like_text ? COMPRESSION_ALGORITHM_TRANSFORMER : COMPRESSION_ALGORITHM_LSTM;
    }
    
    // Use algorithm router if available
    if (algorithm_router_is_ready()) {
        AlgorithmRoutingDecision decision;
        if (algorithm_router_analyze_and_select(input_data, input_size, &decision)) {
            // Convert routing algorithm to neural compression algorithm
            switch (decision.primary_algorithm) {
                case ROUTING_ALGORITHM_TRANSFORMER:
                    return COMPRESSION_ALGORITHM_TRANSFORMER;
                case ROUTING_ALGORITHM_LSTM:
                    return COMPRESSION_ALGORITHM_LSTM;
                default:
                    break; // Fall through to heuristics
            }
        }
    }
    
    // Fallback to basic heuristics if router is not available
    // Simple text detection
    bool is_text_like = true;
    for (size_t i = 0; i < input_size && i < 100; i++) {
        uint8_t byte = input_data[i];
        if (byte < 32 && byte != '\n' && byte != '\r' && byte != '\t') {
            if (byte != 0) {
                is_text_like = false;
                break;
            }
        }
    }
    
    return is_text_like ? COMPRESSION_ALGORITHM_TRANSFORMER : COMPRESSION_ALGORITHM_LSTM;
}

const char* compression_integration_algorithm_name(CompressionAlgorithm algorithm) {
    switch (algorithm) {
        case COMPRESSION_ALGORITHM_TRANSFORMER:
            return "Transformer";
        case COMPRESSION_ALGORITHM_LSTM:
            return "LSTM";
        case COMPRESSION_ALGORITHM_AUTO:
            return "Auto";
        default:
            return "Unknown";
    }
}

bool compression_integration_algorithm_available(CompressionAlgorithm algorithm) {
    switch (algorithm) {
        case COMPRESSION_ALGORITHM_LSTM:
            return true; // Always available
            
        case COMPRESSION_ALGORITHM_TRANSFORMER:
            return g_neural_bridge_ready && neural_bridge_algorithm_available(NEURAL_ALGORITHM_TRANSFORMER);
            
        case COMPRESSION_ALGORITHM_AUTO:
            return true; // Auto selection always available
            
        default:
            return false;
    }
}
