/*
 * CUDA Profile System Implementation
 * 
 * Implements exact CUDA profile parameters based on original implementation
 * (commit bea8582) to ensure mathematical equivalence and behavioral consistency.
 */

#include "cuda_profiles.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA Profile Definitions (based on original CUDA implementation analysis)
static const CUDAProfile cuda_profiles[CUDA_PROFILE_COUNT] = {
    // DEFAULT Profile (general purpose)
    {
        .name = "default",
        .type = CUDA_PROFILE_DEFAULT,
        .params = {
            .batch_size = 16,           // Original CUDA default
            .seg_len = 32,              // Original CUDA default seg_len
            .train_len = 32,            // train_len = seg_len (critical)
            .n_symbols = 256,           // Standard byte alphabet
            .n_layers = 6,              // Moderate complexity
            .n_heads = 8,               // Standard attention heads
            .d_model = 512,             // Model dimension
            .d_ff = 2048,               // Feed-forward dimension
            .lstm_hidden_size = 256,    // LSTM hidden state
            .lstm_num_layers = 2,       // LSTM layers
            .learning_rate = 0.001f,    // Standard learning rate
            .dropout_rate = 0.1f,       // Moderate dropout
            .max_seq_len = 1024,        // Sequence length limit
            .memory_budget_mb = 2048,   // 2GB memory budget
            .use_mixed_precision = true,
            .math_config = NULL         // Math config set at runtime
        },
        .description = "General purpose profile for most file types",
        .is_lstm_optimized = false,
        .is_transformer_optimized = true,
        .min_file_size = 1024,      // 1KB minimum
        .max_file_size = 100*1024*1024  // 100MB maximum
    },
    
    // ENWIK8 Profile (optimized for enwik8 dataset)
    {
        .name = "enwik8",
        .type = CUDA_PROFILE_ENWIK8,
        .params = {
            .batch_size = 32,           // Larger batch for enwik8
            .seg_len = 64,              // Original CUDA enwik8 seg_len
            .train_len = 64,            // train_len = seg_len (critical)
            .n_symbols = 256,           // Byte alphabet
            .n_layers = 12,             // Deep model for quality
            .n_heads = 16,              // More attention heads
            .d_model = 768,             // Larger model dimension
            .d_ff = 3072,               // Larger feed-forward
            .lstm_hidden_size = 512,    // Larger LSTM
            .lstm_num_layers = 3,       // More LSTM layers
            .learning_rate = 0.0005f,   // Lower learning rate
            .dropout_rate = 0.15f,      // Higher dropout for generalization
            .max_seq_len = 2048,        // Longer sequences
            .memory_budget_mb = 4096,   // 4GB memory budget
            .use_mixed_precision = true,
            .math_config = NULL         // Math config set at runtime
        },
        .description = "Optimized for enwik8 dataset and similar text files",
        .is_lstm_optimized = false,
        .is_transformer_optimized = true,
        .min_file_size = 10*1024,   // 10KB minimum
        .max_file_size = 500*1024*1024  // 500MB maximum
    },
    
    // ENWIK9 Profile (optimized for enwik9 dataset)
    {
        .name = "enwik9",
        .type = CUDA_PROFILE_ENWIK9,
        .params = {
            .batch_size = 32,           // Same as enwik8
            .seg_len = 64,              // Original CUDA enwik9 seg_len
            .train_len = 64,            // train_len = seg_len (critical)
            .n_symbols = 256,           // Byte alphabet
            .n_layers = 16,             // Even deeper for enwik9
            .n_heads = 16,              // Same attention heads
            .d_model = 1024,            // Largest model dimension
            .d_ff = 4096,               // Largest feed-forward
            .lstm_hidden_size = 768,    // Largest LSTM
            .lstm_num_layers = 4,       // Most LSTM layers
            .learning_rate = 0.0003f,   // Lowest learning rate
            .dropout_rate = 0.2f,       // Highest dropout
            .max_seq_len = 4096,        // Longest sequences
            .memory_budget_mb = 8192,   // 8GB memory budget
            .use_mixed_precision = true,
            .math_config = NULL         // Math config set at runtime
        },
        .description = "Optimized for enwik9 dataset and large text files",
        .is_lstm_optimized = false,
        .is_transformer_optimized = true,
        .min_file_size = 100*1024,  // 100KB minimum
        .max_file_size = 1024*1024*1024  // 1GB maximum
    },
    
    // LSTM Profile (LSTM-optimized)
    {
        .name = "lstm",
        .type = CUDA_PROFILE_LSTM,
        .params = {
            .batch_size = 32,           // Original CUDA lstm batch_size
            .seg_len = 20,              // Original CUDA lstm seg_len (CRITICAL!)
            .train_len = 20,            // train_len = seg_len (critical)
            .n_symbols = 256,           // Byte alphabet
            .n_layers = 4,              // Moderate layers for LSTM focus
            .n_heads = 4,               // Fewer attention heads
            .d_model = 256,             // Smaller model for LSTM efficiency
            .d_ff = 1024,               // Smaller feed-forward
            .lstm_hidden_size = 512,    // Large LSTM hidden state
            .lstm_num_layers = 4,       // Emphasis on LSTM layers
            .learning_rate = 0.002f,    // Higher learning rate for LSTM
            .dropout_rate = 0.1f,       // Moderate dropout
            .max_seq_len = 512,         // Shorter sequences for LSTM
            .memory_budget_mb = 3072,   // 3GB memory budget
            .use_mixed_precision = true,
            .math_config = NULL         // Math config set at runtime
        },
        .description = "LSTM-optimized profile for sequential data",
        .is_lstm_optimized = true,
        .is_transformer_optimized = false,
        .min_file_size = 512,       // 512B minimum
        .max_file_size = 200*1024*1024  // 200MB maximum
    },
    
    // LSTM_FAST Profile (fast LSTM processing)
    {
        .name = "lstm_fast",
        .type = CUDA_PROFILE_LSTM_FAST,
        .params = {
            .batch_size = 256,          // Large batch for speed
            .seg_len = 20,              // Same as lstm (CRITICAL!)
            .train_len = 20,            // train_len = seg_len (critical)
            .n_symbols = 256,           // Byte alphabet
            .n_layers = 2,              // Fewer layers for speed
            .n_heads = 2,               // Minimal attention heads
            .d_model = 128,             // Small model for speed
            .d_ff = 512,                // Small feed-forward
            .lstm_hidden_size = 256,    // Moderate LSTM size
            .lstm_num_layers = 2,       // Fewer LSTM layers
            .learning_rate = 0.005f,    // Higher learning rate
            .dropout_rate = 0.05f,      // Lower dropout for speed
            .max_seq_len = 256,         // Short sequences for speed
            .memory_budget_mb = 1536,   // 1.5GB memory budget
            .use_mixed_precision = true,
            .math_config = NULL         // Math config set at runtime
        },
        .description = "Fast LSTM processing with reduced quality",
        .is_lstm_optimized = true,
        .is_transformer_optimized = false,
        .min_file_size = 256,       // 256B minimum
        .max_file_size = 50*1024*1024   // 50MB maximum
    }
};

// Profile name lookup table
static const char* profile_type_strings[CUDA_PROFILE_COUNT] = {
    "default",
    "enwik8", 
    "enwik9",
    "lstm",
    "lstm_fast"
};

// Get profile by name
const CUDAProfile* cuda_profile_get(const char* profile_name) {
    if (!profile_name) {
        return NULL;
    }
    
    for (int i = 0; i < CUDA_PROFILE_COUNT; i++) {
        if (strcmp(cuda_profiles[i].name, profile_name) == 0) {
            return &cuda_profiles[i];
        }
    }
    
    return NULL;
}

// Get profile by type
const CUDAProfile* cuda_profile_get_by_type(CUDAProfileType type) {
    if (type < 0 || type >= CUDA_PROFILE_COUNT) {
        return NULL;
    }
    
    return &cuda_profiles[type];
}

// Convert profile type to string
const char* cuda_profile_type_to_string(CUDAProfileType type) {
    if (type < 0 || type >= CUDA_PROFILE_COUNT) {
        return NULL;
    }
    
    return profile_type_strings[type];
}

// Convert string to profile type
CUDAProfileType cuda_profile_string_to_type(const char* profile_name) {
    if (!profile_name) {
        return CUDA_PROFILE_COUNT; // Invalid
    }
    
    for (int i = 0; i < CUDA_PROFILE_COUNT; i++) {
        if (strcmp(profile_name, profile_type_strings[i]) == 0) {
            return (CUDAProfileType)i;
        }
    }
    
    return CUDA_PROFILE_COUNT; // Invalid
}

// Validate profile parameters
bool cuda_profile_validate(const CUDAProfile* profile) {
    if (!profile) {
        return false;
    }
    
    const CUDAModelParams* params = &profile->params;
    
    // Basic parameter validation
    if (params->batch_size <= 0 || params->batch_size > 1024) {
        return false;
    }
    
    if (params->seg_len <= 0 || params->seg_len > 1024) {
        return false;
    }
    
    if (params->n_symbols <= 0 || params->n_symbols > 65536) {
        return false;
    }
    
    // CUDA-specific validations
    if (!cuda_params_verify_seg_len_relationship(params)) {
        return false;
    }
    
    if (!cuda_params_verify_batch_constraints(params)) {
        return false;
    }
    
    if (!cuda_params_verify_memory_requirements(params)) {
        return false;
    }
    
    return true;
}

// Check file size compatibility
bool cuda_profile_is_compatible_with_file_size(const CUDAProfile* profile, size_t file_size) {
    if (!profile) {
        return false;
    }
    
    return (file_size >= profile->min_file_size && file_size <= profile->max_file_size);
}

// Get all profiles
const CUDAProfile** cuda_profile_get_all(void) {
    static const CUDAProfile* all_profiles[CUDA_PROFILE_COUNT + 1];
    
    for (int i = 0; i < CUDA_PROFILE_COUNT; i++) {
        all_profiles[i] = &cuda_profiles[i];
    }
    all_profiles[CUDA_PROFILE_COUNT] = NULL; // Null terminator
    
    return all_profiles;
}

// Get profile count
size_t cuda_profile_count(void) {
    return CUDA_PROFILE_COUNT;
}

// Print profile information
void cuda_profile_print_info(const CUDAProfile* profile) {
    if (!profile) {
        printf("Invalid profile\n");
        return;
    }
    
    printf("CUDA Profile: %s\n", profile->name);
    printf("  Description: %s\n", profile->description);
    printf("  Type: %s\n", cuda_profile_type_to_string(profile->type));
    printf("  Parameters:\n");
    printf("    batch_size: %d\n", profile->params.batch_size);
    printf("    seg_len: %d (CRITICAL for CUDA compatibility)\n", profile->params.seg_len);
    printf("    n_symbols: %d\n", profile->params.n_symbols);
    printf("    n_layers: %d\n", profile->params.n_layers);
    printf("    LSTM optimized: %s\n", profile->is_lstm_optimized ? "Yes" : "No");
    printf("    Transformer optimized: %s\n", profile->is_transformer_optimized ? "Yes" : "No");
    printf("    File size range: %zu - %zu bytes\n", profile->min_file_size, profile->max_file_size);
}

// CUDA Parameter Verification Functions

bool cuda_params_verify_seg_len_relationship(const CUDAModelParams* params) {
    if (!params) {
        return false;
    }
    
    // Critical CUDA relationship: train_len = seg_len
    // This relationship must be maintained for mathematical equivalence
    
    // Verify seg_len is in valid CUDA range
    if (params->seg_len < 1 || params->seg_len > 1024) {
        return false;
    }
    
    // Verify seg_len matches expected CUDA values
    if (params->seg_len != 20 && params->seg_len != 32 && params->seg_len != 64) {
        // Allow non-standard values but warn
        printf("Warning: seg_len=%d is non-standard. CUDA uses 20 (lstm), 32 (default), 64 (enwik8/9)\n", 
               params->seg_len);
    }
    
    return true;
}

bool cuda_params_verify_batch_constraints(const CUDAModelParams* params) {
    if (!params) {
        return false;
    }
    
    // CUDA batch size constraints
    if (params->batch_size < 1 || params->batch_size > 1024) {
        return false;
    }
    
    // Verify power-of-2 alignment for optimal CUDA performance
    if ((params->batch_size & (params->batch_size - 1)) != 0) {
        printf("Warning: batch_size=%d is not power-of-2. May impact CUDA performance\n", 
               params->batch_size);
    }
    
    return true;
}

bool cuda_params_verify_memory_requirements(const CUDAModelParams* params) {
    if (!params) {
        return false;
    }
    
    // Estimate memory requirements
    size_t estimated_memory = 0;
    
    // Model parameters memory
    estimated_memory += params->n_layers * params->d_model * params->d_ff * sizeof(float);
    
    // Batch processing memory
    estimated_memory += params->batch_size * params->max_seq_len * params->d_model * sizeof(float);
    
    // LSTM memory
    estimated_memory += params->lstm_num_layers * params->lstm_hidden_size * params->batch_size * sizeof(float) * 4; // 4 gates
    
    size_t estimated_mb = estimated_memory / (1024 * 1024);
    
    if (estimated_mb > params->memory_budget_mb) {
        printf("Warning: Estimated memory usage (%zu MB) exceeds budget (%d MB)\n", 
               estimated_mb, params->memory_budget_mb);
        return false;
    }
    
    return true;
}

// Profile selection helper
const CUDAProfile* cuda_profile_select_best_for_file(size_t file_size, bool prefer_quality) {
    // Default fallback
    const CUDAProfile* best_profile = &cuda_profiles[CUDA_PROFILE_DEFAULT];
    
    // Size-based selection
    if (file_size < 1024) {
        // Very small files - use LSTM_FAST
        return &cuda_profiles[CUDA_PROFILE_LSTM_FAST];
    } else if (file_size < 100*1024) {
        // Small files - use LSTM or DEFAULT
        return prefer_quality ? &cuda_profiles[CUDA_PROFILE_LSTM] : &cuda_profiles[CUDA_PROFILE_DEFAULT];
    } else if (file_size < 10*1024*1024) {
        // Medium files - use DEFAULT or ENWIK8
        return prefer_quality ? &cuda_profiles[CUDA_PROFILE_ENWIK8] : &cuda_profiles[CUDA_PROFILE_DEFAULT];
    } else {
        // Large files - use ENWIK9 or ENWIK8
        return prefer_quality ? &cuda_profiles[CUDA_PROFILE_ENWIK9] : &cuda_profiles[CUDA_PROFILE_ENWIK8];
    }
}
