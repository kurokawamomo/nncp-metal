/*
 * AdaptiveMemoryManager.h
 * 
 * Unified Memory Management System for CUDA enwik8 compatible architecture
 * Optimized for Apple Silicon unified memory architecture
 * Zone-based memory layout with CPU-GPU zero-copy buffers
 */

#ifndef ADAPTIVE_MEMORY_MANAGER_H
#define ADAPTIVE_MEMORY_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct AdaptiveMemoryManager AdaptiveMemoryManager;

// Memory zone types for organized layout
typedef enum {
    MEMORY_ZONE_WEIGHTS = 0,        // Model weights and parameters
    MEMORY_ZONE_CONTEXT,            // Context and embedding buffers
    MEMORY_ZONE_WORKSPACE,          // Temporary computation workspace
    MEMORY_ZONE_INTERMEDIATE,       // Intermediate layer activations
    MEMORY_ZONE_ATTENTION,          // Attention mechanism buffers
    MEMORY_ZONE_FFN,                // Feed-forward network buffers
    MEMORY_ZONE_COUNT               // Total number of zones
} MemoryZoneType;

// Memory allocation strategies
typedef enum {
    MEMORY_STRATEGY_EAGER = 0,      // Allocate all memory upfront
    MEMORY_STRATEGY_LAZY,           // Allocate memory as needed
    MEMORY_STRATEGY_ADAPTIVE,       // Dynamically adjust based on usage
    MEMORY_STRATEGY_OPTIMIZED       // Apple Silicon optimized allocation
} MemoryStrategy;

// Error codes
typedef enum {
    MEMORY_MANAGER_SUCCESS = 0,
    MEMORY_MANAGER_ERROR_INVALID_PARAM,
    MEMORY_MANAGER_ERROR_ALLOCATION_FAILED,
    MEMORY_MANAGER_ERROR_ZONE_NOT_FOUND,
    MEMORY_MANAGER_ERROR_INSUFFICIENT_MEMORY,
    MEMORY_MANAGER_ERROR_DEVICE_NOT_FOUND,
    MEMORY_MANAGER_ERROR_BUFFER_CREATION_FAILED,
    MEMORY_MANAGER_ERROR_ZONE_ALREADY_EXISTS,
    MEMORY_MANAGER_ERROR_ALIGNMENT_FAILED
} MemoryManagerError;

// Memory zone configuration
typedef struct {
    MemoryZoneType zone_type;       // Type of memory zone
    size_t initial_size_mb;         // Initial allocation size in MB
    size_t max_size_mb;             // Maximum allowed size in MB
    uint32_t alignment_bytes;       // Memory alignment requirement
    bool allow_cpu_access;          // Allow CPU direct access
    bool allow_gpu_access;          // Allow GPU direct access
    bool is_persistent;             // Keep allocated between operations
    float growth_factor;            // Growth factor when expanding (e.g., 1.5x)
} MemoryZoneConfig;

// Memory usage statistics
typedef struct {
    size_t total_allocated_bytes;   // Total allocated memory
    size_t total_used_bytes;        // Total used memory
    size_t peak_usage_bytes;        // Peak memory usage recorded
    size_t zone_allocated[MEMORY_ZONE_COUNT];  // Per-zone allocation
    size_t zone_used[MEMORY_ZONE_COUNT];       // Per-zone usage
    uint32_t allocation_count;      // Number of active allocations
    uint32_t deallocation_count;    // Total deallocations performed
    float fragmentation_ratio;      // Memory fragmentation (0.0-1.0)
} MemoryUsageStats;

// CUDA enwik8 specific memory requirements
typedef struct {
    // Model weights memory requirements
    size_t transformer_weights_mb;   // 768-dim Transformer weights
    size_t attention_weights_mb;     // 16-head attention weights  
    size_t ffn_weights_mb;          // 3072-dim FFN weights
    
    // Context buffer requirements  
    size_t context_64_mb;           // 64-token context buffers
    size_t context_512_mb;          // 512-token context buffers
    size_t context_1024_mb;         // 1024-token context buffers
    size_t context_2048_mb;         // 2048-token context buffers
    
    // Workspace requirements
    size_t intermediate_workspace_mb; // Intermediate computation space
    size_t attention_workspace_mb;   // Attention computation workspace
    size_t ffn_workspace_mb;         // FFN computation workspace
    
    // Total estimated requirements
    size_t total_minimum_mb;        // Minimum required memory
    size_t total_recommended_mb;    // Recommended memory for performance
    size_t total_optimal_mb;        // Optimal memory for all features
} CUDAEnwik8MemoryRequirements;

// Core API Functions

/**
 * Create adaptive memory manager with CUDA enwik8 optimized configuration
 * @param manager Pointer to store created memory manager context
 * @param strategy Memory allocation strategy to use
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_create(AdaptiveMemoryManager** manager,
                                         MemoryStrategy strategy);

/**
 * Initialize memory zones for CUDA enwik8 architecture
 * Configures zones for weights, context, workspace, and intermediate buffers
 * @param manager Memory manager context
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_initialize_cuda_enwik8_zones(AdaptiveMemoryManager* manager);

/**
 * Allocate memory buffer in specified zone with Apple Silicon optimization
 * @param manager Memory manager context
 * @param zone_type Target memory zone
 * @param size_bytes Size in bytes to allocate
 * @param alignment_bytes Memory alignment requirement (0 for default)
 * @param buffer_id Output buffer identifier
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_allocate(AdaptiveMemoryManager* manager,
                                           MemoryZoneType zone_type,
                                           size_t size_bytes,
                                           uint32_t alignment_bytes,
                                           uint32_t* buffer_id);

/**
 * Get CPU-accessible pointer to allocated buffer (unified memory)
 * @param manager Memory manager context
 * @param buffer_id Buffer identifier from allocation
 * @param cpu_ptr Output CPU pointer
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_get_cpu_pointer(AdaptiveMemoryManager* manager,
                                                  uint32_t buffer_id,
                                                  void** cpu_ptr);

#ifdef __OBJC__
/**
 * Get Metal GPU buffer for allocated memory (zero-copy on Apple Silicon)
 * @param manager Memory manager context
 * @param buffer_id Buffer identifier from allocation
 * @param metal_buffer Output Metal buffer
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_get_metal_buffer(AdaptiveMemoryManager* manager,
                                                   uint32_t buffer_id,
                                                   id<MTLBuffer>* metal_buffer);
#endif

/**
 * Deallocate memory buffer
 * @param manager Memory manager context  
 * @param buffer_id Buffer identifier to deallocate
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_deallocate(AdaptiveMemoryManager* manager,
                                             uint32_t buffer_id);

/**
 * Get current memory usage statistics
 * @param manager Memory manager context
 * @param stats Output memory usage statistics
 */
void memory_manager_get_usage_stats(AdaptiveMemoryManager* manager,
                                   MemoryUsageStats* stats);

/**
 * Get CUDA enwik8 memory requirements estimation
 * @param requirements Output memory requirements structure
 */
void memory_manager_get_cuda_enwik8_requirements(CUDAEnwik8MemoryRequirements* requirements);

/**
 * Destroy memory manager and free all resources
 * @param manager Memory manager context to destroy
 */
void memory_manager_destroy(AdaptiveMemoryManager* manager);

// Zone Management Functions

/**
 * Configure memory zone parameters
 * @param manager Memory manager context
 * @param zone_config Zone configuration parameters
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_configure_zone(AdaptiveMemoryManager* manager,
                                                 const MemoryZoneConfig* zone_config);

/**
 * Get zone configuration
 * @param manager Memory manager context
 * @param zone_type Zone type to query
 * @param zone_config Output zone configuration
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_get_zone_config(AdaptiveMemoryManager* manager,
                                                  MemoryZoneType zone_type,
                                                  MemoryZoneConfig* zone_config);

/**
 * Resize memory zone (if strategy allows)
 * @param manager Memory manager context
 * @param zone_type Zone type to resize
 * @param new_size_mb New size in megabytes
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_resize_zone(AdaptiveMemoryManager* manager,
                                              MemoryZoneType zone_type,
                                              size_t new_size_mb);

/**
 * Defragment memory zone to reduce fragmentation
 * @param manager Memory manager context
 * @param zone_type Zone type to defragment (or all zones if zone_type >= MEMORY_ZONE_COUNT)
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_defragment_zone(AdaptiveMemoryManager* manager,
                                                  MemoryZoneType zone_type);

// Utility Functions

/**
 * Get error string for error code
 * @param error_code MemoryManagerError code
 * @return Human-readable error message
 */
const char* memory_manager_get_error_string(MemoryManagerError error_code);

/**
 * Convert zone type to string
 * @param zone_type Memory zone type
 * @return Human-readable zone type name
 */
const char* memory_manager_zone_type_to_string(MemoryZoneType zone_type);

/**
 * Check if current system has sufficient memory for CUDA enwik8 requirements
 * @param min_requirement_mb Minimum memory requirement in MB
 * @param has_sufficient_memory Output boolean for sufficient memory check
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_check_system_requirements(size_t min_requirement_mb,
                                                           bool* has_sufficient_memory);

/**
 * Estimate memory usage for given configuration
 * @param num_layers Number of Transformer layers (e.g., 12)
 * @param hidden_size Hidden dimension size (e.g., 768)
 * @param ffn_size FFN dimension size (e.g., 3072)
 * @param num_heads Number of attention heads (e.g., 16)
 * @param max_context_length Maximum context length (e.g., 2048)
 * @param estimated_mb Output estimated memory usage in MB
 * @return MEMORY_MANAGER_SUCCESS on success, error code on failure
 */
MemoryManagerError memory_manager_estimate_usage(uint32_t num_layers,
                                                 uint32_t hidden_size,
                                                 uint32_t ffn_size,
                                                 uint32_t num_heads,
                                                 uint32_t max_context_length,
                                                 size_t* estimated_mb);

// CUDA enwik8 specific constants
#define CUDA_ENWIK8_HIDDEN_SIZE 768
#define CUDA_ENWIK8_FFN_SIZE 3072
#define CUDA_ENWIK8_NUM_HEADS 16
#define CUDA_ENWIK8_NUM_LAYERS 12
#define CUDA_ENWIK8_MAX_CONTEXT 2048

// Memory zone size constants (in MB)
#define CUDA_ENWIK8_WEIGHTS_ZONE_MB 512     // Transformer model weights
#define CUDA_ENWIK8_CONTEXT_ZONE_MB 256     // Context buffers
#define CUDA_ENWIK8_WORKSPACE_ZONE_MB 1024  // Computation workspace
#define CUDA_ENWIK8_INTERMEDIATE_ZONE_MB 512 // Intermediate activations
#define CUDA_ENWIK8_ATTENTION_ZONE_MB 256   // Attention computation
#define CUDA_ENWIK8_FFN_ZONE_MB 512         // FFN computation

// Total memory requirements
#define CUDA_ENWIK8_MIN_MEMORY_MB 2048      // 2GB minimum
#define CUDA_ENWIK8_RECOMMENDED_MEMORY_MB 4096  // 4GB recommended  
#define CUDA_ENWIK8_OPTIMAL_MEMORY_MB 8192      // 8GB optimal

#ifdef __cplusplus
}
#endif

#endif // ADAPTIVE_MEMORY_MANAGER_H
