#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for BPE tokenizer
typedef enum {
    BPE_SUCCESS = 0,
    BPE_ERROR_INVALID_PARAM,
    BPE_ERROR_MEMORY_ALLOCATION,
    BPE_ERROR_FILE_IO,
    BPE_ERROR_INVALID_VOCABULARY,
    BPE_ERROR_TOKENIZATION_FAILED,
    BPE_ERROR_INSUFFICIENT_BUFFER
} BPEError;

// BPE Tokenizer structure
typedef struct {
    uint32_t vocab_size;        // Number of tokens in vocabulary
    char** vocabulary;          // Array of vocabulary strings
    uint32_t* token_ids;        // Token ID mappings
    float* token_frequencies;   // Token frequency statistics
    uint32_t max_sequence_length; // Maximum sequence length supported
    uint32_t max_token_length;  // Maximum individual token length
    bool case_sensitive;        // Case sensitivity setting
    char* unk_token;           // Unknown token string
    uint32_t unk_token_id;     // Unknown token ID
} BPETokenizer;

// Token sequence structure
typedef struct {
    uint32_t* tokens;          // Array of token IDs
    uint32_t token_count;      // Number of tokens in sequence
    uint32_t* positions;       // Original character positions
    float confidence;          // Tokenization confidence score (0.0-1.0)
    uint32_t oov_count;        // Number of out-of-vocabulary tokens
    bool is_valid;             // Sequence validity flag
} TokenSequence;

// Tokenization statistics
typedef struct {
    uint32_t total_tokens;     // Total tokens processed
    uint32_t oov_tokens;       // Out-of-vocabulary tokens
    float oov_rate;            // OOV rate (0.0-1.0)
    uint64_t processing_time_ns; // Processing time in nanoseconds
    uint32_t max_merge_steps;  // Maximum merge steps used
    float avg_token_length;    // Average token length
} TokenizationStats;

// BPE merge rule structure for internal processing
typedef struct {
    char* first;               // First part of merge
    char* second;              // Second part of merge
    char* merged;              // Result of merge
    uint32_t priority;         // Merge priority (lower = higher priority)
    float frequency;           // Merge frequency in training data
} BPEMergeRule;

// Vocabulary entry structure
typedef struct {
    char* token_str;           // Token string
    uint32_t token_id;         // Token ID
    float frequency;           // Token frequency
    bool is_special;           // Special token flag (e.g., UNK, BOS, EOS)
    uint32_t token_length;     // Token string length
} VocabEntry;

// Core API Functions

/**
 * Create a new BPE tokenizer from vocabulary file
 * @param tokenizer Pointer to store created tokenizer
 * @param vocab_file Path to vocabulary file
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_tokenizer_create(BPETokenizer** tokenizer, const char* vocab_file);

/**
 * Create tokenizer from vocabulary data in memory
 * @param tokenizer Pointer to store created tokenizer
 * @param vocab_data Vocabulary data buffer
 * @param vocab_size Size of vocabulary data
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_tokenizer_create_from_data(BPETokenizer** tokenizer, 
                                       const char* vocab_data, 
                                       size_t vocab_size);

/**
 * Tokenize text string into token sequence
 * @param tokenizer BPE tokenizer instance
 * @param text Input text string
 * @param sequence Pointer to store token sequence
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_tokenize(BPETokenizer* tokenizer, 
                     const char* text, 
                     TokenSequence** sequence);

/**
 * Tokenize text with advanced options
 * @param tokenizer BPE tokenizer instance
 * @param text Input text string
 * @param max_tokens Maximum number of tokens to generate
 * @param enable_fallback Enable character-level fallback
 * @param sequence Pointer to store token sequence
 * @param stats Pointer to store tokenization statistics (can be NULL)
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_tokenize_advanced(BPETokenizer* tokenizer,
                              const char* text,
                              uint32_t max_tokens,
                              bool enable_fallback,
                              TokenSequence** sequence,
                              TokenizationStats* stats);

/**
 * Convert token sequence back to text
 * @param tokenizer BPE tokenizer instance
 * @param sequence Token sequence to convert
 * @param text Pointer to store reconstructed text
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_detokenize(BPETokenizer* tokenizer, 
                       TokenSequence* sequence, 
                       char** text);

/**
 * Get vocabulary entry by token ID
 * @param tokenizer BPE tokenizer instance
 * @param token_id Token ID to lookup
 * @param entry Pointer to store vocabulary entry
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_get_vocab_entry(BPETokenizer* tokenizer,
                            uint32_t token_id,
                            VocabEntry** entry);

/**
 * Get token ID by string
 * @param tokenizer BPE tokenizer instance
 * @param token_str Token string to lookup
 * @param token_id Pointer to store token ID
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_get_token_id(BPETokenizer* tokenizer,
                         const char* token_str,
                         uint32_t* token_id);

/**
 * Calculate tokenization confidence score
 * @param tokenizer BPE tokenizer instance
 * @param sequence Token sequence
 * @param confidence Pointer to store confidence score
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_calculate_confidence(BPETokenizer* tokenizer,
                                 TokenSequence* sequence,
                                 float* confidence);

/**
 * Validate tokenizer integrity
 * @param tokenizer BPE tokenizer instance
 * @return BPE_SUCCESS if valid, error code otherwise
 */
BPEError bpe_validate_tokenizer(BPETokenizer* tokenizer);

/**
 * Get tokenizer statistics
 * @param tokenizer BPE tokenizer instance
 * @param vocab_size Pointer to store vocabulary size
 * @param max_token_length Pointer to store maximum token length
 * @param special_token_count Pointer to store special token count
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_get_tokenizer_info(BPETokenizer* tokenizer,
                               uint32_t* vocab_size,
                               uint32_t* max_token_length,
                               uint32_t* special_token_count);

// Memory Management Functions

/**
 * Create empty token sequence
 * @param sequence Pointer to store created sequence
 * @param initial_capacity Initial capacity for tokens
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError token_sequence_create(TokenSequence** sequence, uint32_t initial_capacity);

/**
 * Resize token sequence capacity
 * @param sequence Token sequence to resize
 * @param new_capacity New capacity
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError token_sequence_resize(TokenSequence* sequence, uint32_t new_capacity);

/**
 * Clone token sequence
 * @param source Source token sequence
 * @param destination Pointer to store cloned sequence
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError token_sequence_clone(TokenSequence* source, TokenSequence** destination);

/**
 * Free token sequence memory
 * @param sequence Token sequence to free
 */
void token_sequence_destroy(TokenSequence* sequence);

/**
 * Free BPE tokenizer memory
 * @param tokenizer Tokenizer to free
 */
void bpe_tokenizer_destroy(BPETokenizer* tokenizer);

// Utility Functions

/**
 * Get error message string
 * @param error_code BPE error code
 * @return Human-readable error message
 */
const char* bpe_get_error_string(BPEError error_code);

/**
 * Set case sensitivity
 * @param tokenizer BPE tokenizer instance
 * @param case_sensitive Case sensitivity setting
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_set_case_sensitivity(BPETokenizer* tokenizer, bool case_sensitive);

/**
 * Load merge rules from file (for advanced tokenization)
 * @param tokenizer BPE tokenizer instance
 * @param merge_file Path to merge rules file
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_load_merge_rules(BPETokenizer* tokenizer, const char* merge_file);

/**
 * Benchmark tokenization performance
 * @param tokenizer BPE tokenizer instance
 * @param test_text Test text for benchmarking
 * @param iterations Number of iterations to run
 * @param avg_time_ns Pointer to store average time per iteration
 * @param tokens_per_sec Pointer to store tokens per second
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_benchmark_performance(BPETokenizer* tokenizer,
                                  const char* test_text,
                                  uint32_t iterations,
                                  uint64_t* avg_time_ns,
                                  float* tokens_per_sec);

// Integration with existing memory management
#ifdef USE_METAL
// Forward declaration for Metal memory manager
typedef struct MemoryManager MemoryManager;
#endif

#ifdef USE_METAL
/**
 * Create BPE tokenizer using Metal memory manager
 * @param tokenizer Pointer to store created tokenizer
 * @param vocab_file Path to vocabulary file
 * @param memory_manager Metal memory manager instance
 * @return BPE_SUCCESS on success, error code on failure
 */
BPEError bpe_tokenizer_create_with_metal_memory(BPETokenizer** tokenizer,
                                               const char* vocab_file,
                                               MemoryManager* memory_manager);
#endif

#ifdef __cplusplus
}
#endif

#endif // BPE_TOKENIZER_H
