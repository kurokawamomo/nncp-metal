/*
 * CUDA Profile System
 * 
 * Implements exact parameter matching for original CUDA profiles to ensure
 * mathematical equivalence and behavioral consistency.
 * 
 * Based on original CUDA NNCP implementation (commit bea8582)
 */

#ifndef CUDA_PROFILES_H
#define CUDA_PROFILES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Profile Types (matching original implementation)
typedef enum {
    CUDA_PROFILE_DEFAULT = 0,
    CUDA_PROFILE_ENWIK8,
    CUDA_PROFILE_ENWIK9, 
    CUDA_PROFILE_LSTM,
    CUDA_PROFILE_LSTM_FAST,
    CUDA_PROFILE_COUNT
} CUDAProfileType;

// Include CUDAMathConfig definition
#include "../compatibility/cuda_math_compat.h"

// Core CUDA Model Parameters (from original libnc.h)
typedef struct {
    // Basic model configuration
    int32_t batch_size;          // Batch size for processing
    int32_t seg_len;             // Segment length (critical for CUDA compatibility)
    int32_t train_len;           // Training length (must equal seg_len for CUDA compatibility)
    int32_t n_symbols;           // Number of symbols in alphabet
    int32_t n_layers;            // Number of layers in the model
    int32_t n_heads;             // Number of attention heads
    int32_t d_model;             // Model dimension
    int32_t d_ff;                // Feed-forward dimension
    
    // LSTM specific parameters
    int32_t lstm_hidden_size;    // LSTM hidden state size
    int32_t lstm_num_layers;     // Number of LSTM layers
    
    // Training and inference parameters
    float learning_rate;         // Learning rate for training
    float dropout_rate;          // Dropout rate
    int32_t max_seq_len;         // Maximum sequence length
    
    // Memory and performance settings
    int32_t memory_budget_mb;    // Memory budget in MB
    bool use_mixed_precision;   // Use FP16/FP32 mixed precision
    
    // Math configuration
    CUDAMathConfig* math_config;  // Mathematical computation configuration
} CUDAModelParams;

// CUDA Profile Definition
typedef struct {
    const char* name;           // Profile name (e.g., "default", "lstm")
    CUDAProfileType type;       // Profile type enum
    CUDAModelParams params;     // Model parameters
    
    // Profile metadata
    const char* description;    // Human-readable description
    bool is_lstm_optimized;     // Whether this profile is LSTM-optimized
    bool is_transformer_optimized; // Whether this profile is Transformer-optimized
    
    // Validation constraints
    size_t min_file_size;       // Minimum recommended file size
    size_t max_file_size;       // Maximum recommended file size
} CUDAProfile;

// Profile Management Functions
const CUDAProfile* cuda_profile_get(const char* profile_name);
const CUDAProfile* cuda_profile_get_by_type(CUDAProfileType type);
const char* cuda_profile_type_to_string(CUDAProfileType type);
CUDAProfileType cuda_profile_string_to_type(const char* profile_name);

// Profile Validation
bool cuda_profile_validate(const CUDAProfile* profile);
bool cuda_profile_is_compatible_with_file_size(const CUDAProfile* profile, size_t file_size);

// Profile Information
const CUDAProfile** cuda_profile_get_all(void);
size_t cuda_profile_count(void);
void cuda_profile_print_info(const CUDAProfile* profile);

// CUDA Parameter Verification (ensures exact CUDA compatibility)
bool cuda_params_verify_seg_len_relationship(const CUDAModelParams* params);
bool cuda_params_verify_batch_constraints(const CUDAModelParams* params);
bool cuda_params_verify_memory_requirements(const CUDAModelParams* params);

// Profile Selection Helper
const CUDAProfile* cuda_profile_select_best_for_file(size_t file_size, bool prefer_quality);

#ifdef __cplusplus
}
#endif

#endif // CUDA_PROFILES_H
