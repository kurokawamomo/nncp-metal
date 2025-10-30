#include "bpe_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <assert.h>

// Internal constants
#define BPE_DEFAULT_VOCAB_SIZE 50000
#define BPE_DEFAULT_MAX_TOKEN_LENGTH 256
#define BPE_DEFAULT_MAX_SEQUENCE_LENGTH 16384
#define BPE_UNK_TOKEN "<unk>"
#define BPE_MAX_MERGE_ITERATIONS 1000
#define BPE_MIN_CONFIDENCE_THRESHOLD 0.1f

// Internal helper structures
typedef struct {
    char* word;
    uint32_t frequency;
    char** subwords;
    uint32_t subword_count;
} BPEWord;

typedef struct {
    char* text;
    uint32_t* positions;
    uint32_t char_count;
} TextSegment;

// Internal function declarations
static BPEError parse_vocabulary_file(const char* vocab_file, BPETokenizer* tokenizer);
static BPEError parse_vocabulary_data(const char* vocab_data, size_t data_size, BPETokenizer* tokenizer);
static BPEError initialize_tokenizer_defaults(BPETokenizer* tokenizer);
static BPEError tokenize_with_bpe(BPETokenizer* tokenizer, const char* text, 
                                 TokenSequence* sequence, bool enable_fallback);
static BPEError apply_bpe_merges(BPETokenizer* tokenizer, char** words, uint32_t word_count);
static BPEError character_level_fallback(BPETokenizer* tokenizer, const char* text, 
                                        TokenSequence* sequence);
static float calculate_sequence_confidence(BPETokenizer* tokenizer, TokenSequence* sequence);
static uint32_t hash_string(const char* str);
static int compare_tokens(const void* a, const void* b);
static BPEError resize_token_array(TokenSequence* sequence, uint32_t new_size);

// Error message strings
static const char* error_messages[] = {
    "Success",
    "Invalid parameter",
    "Memory allocation failed", 
    "File I/O error",
    "Invalid vocabulary format",
    "Tokenization failed",
    "Insufficient buffer space"
};

const char* bpe_get_error_string(BPEError error_code) {
    if (error_code >= 0 && error_code < sizeof(error_messages) / sizeof(error_messages[0])) {
        return error_messages[error_code];
    }
    return "Unknown error";
}

BPEError bpe_tokenizer_create(BPETokenizer** tokenizer, const char* vocab_file) {
    if (!tokenizer || !vocab_file) {
        return BPE_ERROR_INVALID_PARAM;
    }
    
    // Allocate tokenizer structure
    *tokenizer = (BPETokenizer*)calloc(1, sizeof(BPETokenizer));
    if (!*tokenizer) {
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize defaults
    BPEError error = initialize_tokenizer_defaults(*tokenizer);
    if (error != BPE_SUCCESS) {
        free(*tokenizer);
        *tokenizer = NULL;
        return error;
    }
    
    // Parse vocabulary file
    error = parse_vocabulary_file(vocab_file, *tokenizer);
    if (error != BPE_SUCCESS) {
        bpe_tokenizer_destroy(*tokenizer);
        *tokenizer = NULL;
        return error;
    }
    
    return BPE_SUCCESS;
}

BPEError bpe_tokenizer_create_from_data(BPETokenizer** tokenizer, 
                                       const char* vocab_data, 
                                       size_t vocab_size) {
    if (!tokenizer || !vocab_data || vocab_size == 0) {
        return BPE_ERROR_INVALID_PARAM;
    }
    
    // Allocate tokenizer structure
    *tokenizer = (BPETokenizer*)calloc(1, sizeof(BPETokenizer));
    if (!*tokenizer) {
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize defaults
    BPEError error = initialize_tokenizer_defaults(*tokenizer);
    if (error != BPE_SUCCESS) {
        free(*tokenizer);
        *tokenizer = NULL;
        return error;
    }
    
    // Parse vocabulary data
    error = parse_vocabulary_data(vocab_data, vocab_size, *tokenizer);
    if (error != BPE_SUCCESS) {
        bpe_tokenizer_destroy(*tokenizer);
        *tokenizer = NULL;
        return error;
    }
    
    return BPE_SUCCESS;
}

static BPEError initialize_tokenizer_defaults(BPETokenizer* tokenizer) {
    tokenizer->vocab_size = 0;
    tokenizer->vocabulary = NULL;
    tokenizer->token_ids = NULL;
    tokenizer->token_frequencies = NULL;
    tokenizer->max_sequence_length = BPE_DEFAULT_MAX_SEQUENCE_LENGTH;
    tokenizer->max_token_length = BPE_DEFAULT_MAX_TOKEN_LENGTH;
    tokenizer->case_sensitive = true;
    
    // Allocate and set unknown token
    tokenizer->unk_token = strdup(BPE_UNK_TOKEN);
    if (!tokenizer->unk_token) {
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    tokenizer->unk_token_id = 0; // Will be set during vocabulary parsing
    
    return BPE_SUCCESS;
}

static BPEError parse_vocabulary_file(const char* vocab_file, BPETokenizer* tokenizer) {
    FILE* file = fopen(vocab_file, "r");
    if (!file) {
        return BPE_ERROR_FILE_IO;
    }
    
    // Count lines to determine vocabulary size
    char line[1024];
    uint32_t line_count = 0;
    while (fgets(line, sizeof(line), file)) {
        line_count++;
    }
    
    if (line_count == 0) {
        fclose(file);
        return BPE_ERROR_INVALID_VOCABULARY;
    }
    
    // Allocate vocabulary arrays
    tokenizer->vocab_size = line_count;
    tokenizer->vocabulary = (char**)calloc(line_count, sizeof(char*));
    tokenizer->token_ids = (uint32_t*)calloc(line_count, sizeof(uint32_t));
    tokenizer->token_frequencies = (float*)calloc(line_count, sizeof(float));
    
    if (!tokenizer->vocabulary || !tokenizer->token_ids || !tokenizer->token_frequencies) {
        fclose(file);
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Parse vocabulary entries
    rewind(file);
    uint32_t token_id = 0;
    while (fgets(line, sizeof(line), file) && token_id < line_count) {
        // Remove newline
        line[strcspn(line, "\n")] = 0;
        
        // Parse line format: "number→token frequency" or "token frequency"
        char* first_part = strtok(line, "\t");
        char* freq_str = strtok(NULL, "\t");
        
        // Check if first part contains "→" (indicating number→token format)
        char* arrow_pos = strstr(first_part, "→");
        char* token_str;
        if (arrow_pos) {
            token_str = arrow_pos + 3; // Skip "→" (3 bytes for UTF-8)
        } else {
            token_str = first_part;
        }
        
        if (!token_str) {
            continue;
        }
        
        // Store token
        tokenizer->vocabulary[token_id] = strdup(token_str);
        if (!tokenizer->vocabulary[token_id]) {
            fclose(file);
            return BPE_ERROR_MEMORY_ALLOCATION;
        }
        
        tokenizer->token_ids[token_id] = token_id;
        tokenizer->token_frequencies[token_id] = freq_str ? atof(freq_str) : 1.0f;
        
        // Check for unknown token
        if (strcmp(token_str, tokenizer->unk_token) == 0) {
            tokenizer->unk_token_id = token_id;
        }
        
        token_id++;
    }
    
    fclose(file);
    
    // Validate that we have an unknown token
    if (tokenizer->unk_token_id == 0 && tokenizer->vocab_size > 0) {
        // If no explicit UNK token, use the first token as UNK
        tokenizer->unk_token_id = 0;
    }
    
    return BPE_SUCCESS;
}

static BPEError parse_vocabulary_data(const char* vocab_data, size_t data_size, BPETokenizer* tokenizer) {
    // Create a copy of the data to work with
    char* data_copy = (char*)malloc(data_size + 1);
    if (!data_copy) {
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    memcpy(data_copy, vocab_data, data_size);
    data_copy[data_size] = '\0';
    
    // Count lines
    uint32_t line_count = 0;
    for (size_t i = 0; i < data_size; i++) {
        if (data_copy[i] == '\n') {
            line_count++;
        }
    }
    
    if (line_count == 0) {
        free(data_copy);
        return BPE_ERROR_INVALID_VOCABULARY;
    }
    
    // Allocate vocabulary arrays
    tokenizer->vocab_size = line_count;
    tokenizer->vocabulary = (char**)calloc(line_count, sizeof(char*));
    tokenizer->token_ids = (uint32_t*)calloc(line_count, sizeof(uint32_t));
    tokenizer->token_frequencies = (float*)calloc(line_count, sizeof(float));
    
    if (!tokenizer->vocabulary || !tokenizer->token_ids || !tokenizer->token_frequencies) {
        free(data_copy);
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Parse lines
    char* line = strtok(data_copy, "\n");
    uint32_t token_id = 0;
    
    while (line && token_id < line_count) {
        char* first_part = strtok(line, "\t");
        char* freq_str = strtok(NULL, "\t");
        
        // Check if first part contains "→" (indicating number→token format)
        char* arrow_pos = strstr(first_part, "→");
        char* token_str;
        if (arrow_pos) {
            token_str = arrow_pos + 3; // Skip "→" (3 bytes for UTF-8)
        } else {
            token_str = first_part;
        }
        
        if (token_str) {
            tokenizer->vocabulary[token_id] = strdup(token_str);
            if (!tokenizer->vocabulary[token_id]) {
                free(data_copy);
                return BPE_ERROR_MEMORY_ALLOCATION;
            }
            
            tokenizer->token_ids[token_id] = token_id;
            tokenizer->token_frequencies[token_id] = freq_str ? atof(freq_str) : 1.0f;
            
            if (strcmp(token_str, tokenizer->unk_token) == 0) {
                tokenizer->unk_token_id = token_id;
            }
            
            token_id++;
        }
        
        line = strtok(NULL, "\n");
    }
    
    free(data_copy);
    return BPE_SUCCESS;
}

BPEError bpe_tokenize(BPETokenizer* tokenizer, const char* text, TokenSequence** sequence) {
    return bpe_tokenize_advanced(tokenizer, text, tokenizer->max_sequence_length, 
                                true, sequence, NULL);
}

BPEError bpe_tokenize_advanced(BPETokenizer* tokenizer,
                              const char* text,
                              uint32_t max_tokens,
                              bool enable_fallback,
                              TokenSequence** sequence,
                              TokenizationStats* stats) {
    if (!tokenizer || !text || !sequence) {
        return BPE_ERROR_INVALID_PARAM;
    }
    
    clock_t start_time = clock();
    
    // Create token sequence
    BPEError error = token_sequence_create(sequence, max_tokens);
    if (error != BPE_SUCCESS) {
        return error;
    }
    
    // Initialize stats if provided
    if (stats) {
        memset(stats, 0, sizeof(TokenizationStats));
        stats->max_merge_steps = 0;
    }
    
    // Perform tokenization
    error = tokenize_with_bpe(tokenizer, text, *sequence, enable_fallback);
    if (error != BPE_SUCCESS) {
        token_sequence_destroy(*sequence);
        *sequence = NULL;
        return error;
    }
    
    // Calculate confidence score
    (*sequence)->confidence = calculate_sequence_confidence(tokenizer, *sequence);
    
    // Update statistics
    if (stats) {
        stats->total_tokens = (*sequence)->token_count;
        stats->oov_tokens = (*sequence)->oov_count;
        stats->oov_rate = (float)stats->oov_tokens / stats->total_tokens;
        stats->processing_time_ns = (uint64_t)((clock() - start_time) * 1000000000 / CLOCKS_PER_SEC);
        
        // Calculate average token length
        float total_length = 0;
        for (uint32_t i = 0; i < (*sequence)->token_count; i++) {
            uint32_t token_id = (*sequence)->tokens[i];
            if (token_id < tokenizer->vocab_size) {
                total_length += strlen(tokenizer->vocabulary[token_id]);
            }
        }
        stats->avg_token_length = total_length / stats->total_tokens;
    }
    
    return BPE_SUCCESS;
}

static BPEError tokenize_with_bpe(BPETokenizer* tokenizer, const char* text, 
                                 TokenSequence* sequence, bool enable_fallback) {
    if (!text || strlen(text) == 0) {
        sequence->token_count = 0;
        sequence->confidence = 1.0f;
        return BPE_SUCCESS;
    }
    
    size_t text_len = strlen(text);
    uint32_t token_count = 0;
    uint32_t oov_count = 0;
    
    // Simple word-based tokenization with subword splitting
    char* text_copy = strdup(text);
    if (!text_copy) {
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Convert to lowercase if not case sensitive
    if (!tokenizer->case_sensitive) {
        for (size_t i = 0; i < text_len; i++) {
            text_copy[i] = tolower(text_copy[i]);
        }
    }
    
    // Split by whitespace and punctuation
    char* word = strtok(text_copy, " \t\n\r.,!?;:()[]{}\"'");
    uint32_t char_position = 0;
    
    while (word && token_count < sequence->token_count) {
        // Find word in vocabulary
        uint32_t token_id = tokenizer->unk_token_id;
        bool found = false;
        
        // Direct vocabulary lookup
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            if (strcmp(word, tokenizer->vocabulary[i]) == 0) {
                token_id = i;
                found = true;
                break;
            }
        }
        
        // If not found, try subword splitting
        if (!found && enable_fallback) {
            // Simple subword splitting: try progressively shorter prefixes
            size_t word_len = strlen(word);
            for (size_t len = word_len; len > 0 && len >= 2; len--) {
                char temp_char = word[len];
                word[len] = '\0';
                
                for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
                    if (strcmp(word, tokenizer->vocabulary[i]) == 0) {
                        token_id = i;
                        found = true;
                        break;
                    }
                }
                
                word[len] = temp_char;
                if (found) break;
            }
        }
        
        // Character-level fallback for completely unknown words
        if (!found && enable_fallback) {
            BPEError fallback_error = character_level_fallback(tokenizer, word, sequence);
            if (fallback_error == BPE_SUCCESS) {
                // Character fallback handled, continue with next word
                word = strtok(NULL, " \t\n\r.,!?;:()[]{}\"'");
                continue;
            }
        }
        
        // Add token to sequence
        if (token_count < sequence->token_count) {
            sequence->tokens[token_count] = token_id;
            sequence->positions[token_count] = char_position;
            
            if (!found) {
                oov_count++;
            }
            
            token_count++;
        }
        
        char_position += strlen(word) + 1; // +1 for separator
        word = strtok(NULL, " \t\n\r.,!?;:()[]{}\"'");
    }
    
    sequence->token_count = token_count;
    sequence->oov_count = oov_count;
    sequence->is_valid = true;
    
    free(text_copy);
    return BPE_SUCCESS;
}

static BPEError character_level_fallback(BPETokenizer* tokenizer, const char* text, 
                                        TokenSequence* sequence) {
    // For simplicity, just use UNK token for character fallback
    // In a full implementation, this would split into character tokens
    (void)text; // Suppress unused parameter warning
    
    // This function is called when we need fallback, not when checking capacity
    sequence->tokens[sequence->token_count] = tokenizer->unk_token_id;
    sequence->positions[sequence->token_count] = 0;
    sequence->oov_count++;
    sequence->token_count++;
    
    return BPE_SUCCESS;
}

static float calculate_sequence_confidence(BPETokenizer* tokenizer, TokenSequence* sequence) {
    if (sequence->token_count == 0) {
        return 1.0f;
    }
    
    float confidence = 1.0f - ((float)sequence->oov_count / sequence->token_count);
    
    // Additional confidence factors
    float avg_frequency = 0.0f;
    for (uint32_t i = 0; i < sequence->token_count; i++) {
        uint32_t token_id = sequence->tokens[i];
        if (token_id < tokenizer->vocab_size) {
            avg_frequency += tokenizer->token_frequencies[token_id];
        }
    }
    avg_frequency /= sequence->token_count;
    
    // Normalize frequency component (assuming max frequency of 1.0)
    confidence = (confidence + (avg_frequency * 0.1f)) / 1.1f;
    
    return confidence > 1.0f ? 1.0f : (confidence < 0.0f ? 0.0f : confidence);
}

BPEError bpe_detokenize(BPETokenizer* tokenizer, TokenSequence* sequence, char** text) {
    if (!tokenizer || !sequence || !text) {
        return BPE_ERROR_INVALID_PARAM;
    }
    
    if (sequence->token_count == 0) {
        *text = strdup("");
        return *text ? BPE_SUCCESS : BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Estimate output size
    size_t estimated_size = 0;
    for (uint32_t i = 0; i < sequence->token_count; i++) {
        uint32_t token_id = sequence->tokens[i];
        if (token_id < tokenizer->vocab_size) {
            estimated_size += strlen(tokenizer->vocabulary[token_id]) + 1; // +1 for space
        }
    }
    
    *text = (char*)malloc(estimated_size + 1);
    if (!*text) {
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    (*text)[0] = '\0';
    
    // Reconstruct text
    for (uint32_t i = 0; i < sequence->token_count; i++) {
        uint32_t token_id = sequence->tokens[i];
        if (token_id < tokenizer->vocab_size) {
            if (i > 0) {
                strcat(*text, " ");
            }
            strcat(*text, tokenizer->vocabulary[token_id]);
        }
    }
    
    return BPE_SUCCESS;
}

BPEError bpe_get_vocab_entry(BPETokenizer* tokenizer, uint32_t token_id, VocabEntry** entry) {
    if (!tokenizer || !entry || token_id >= tokenizer->vocab_size) {
        return BPE_ERROR_INVALID_PARAM;
    }
    
    static VocabEntry vocab_entry; // Static for simplicity
    vocab_entry.token_str = tokenizer->vocabulary[token_id];
    vocab_entry.token_id = token_id;
    vocab_entry.frequency = tokenizer->token_frequencies[token_id];
    vocab_entry.is_special = (token_id == tokenizer->unk_token_id);
    vocab_entry.token_length = strlen(tokenizer->vocabulary[token_id]);
    
    *entry = &vocab_entry;
    return BPE_SUCCESS;
}

BPEError bpe_get_token_id(BPETokenizer* tokenizer, const char* token_str, uint32_t* token_id) {
    if (!tokenizer || !token_str || !token_id) {
        return BPE_ERROR_INVALID_PARAM;
    }
    
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        if (strcmp(tokenizer->vocabulary[i], token_str) == 0) {
            *token_id = i;
            return BPE_SUCCESS;
        }
    }
    
    *token_id = tokenizer->unk_token_id;
    return BPE_SUCCESS; // Return success but with UNK token
}

BPEError token_sequence_create(TokenSequence** sequence, uint32_t initial_capacity) {
    if (!sequence || initial_capacity == 0) {
        return BPE_ERROR_INVALID_PARAM;
    }
    
    *sequence = (TokenSequence*)calloc(1, sizeof(TokenSequence));
    if (!*sequence) {
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    (*sequence)->tokens = (uint32_t*)malloc(initial_capacity * sizeof(uint32_t));
    (*sequence)->positions = (uint32_t*)malloc(initial_capacity * sizeof(uint32_t));
    
    if (!(*sequence)->tokens || !(*sequence)->positions) {
        token_sequence_destroy(*sequence);
        *sequence = NULL;
        return BPE_ERROR_MEMORY_ALLOCATION;
    }
    
    (*sequence)->token_count = initial_capacity; // Capacity, not actual count
    (*sequence)->confidence = 0.0f;
    (*sequence)->oov_count = 0;
    (*sequence)->is_valid = false;
    
    return BPE_SUCCESS;
}

void token_sequence_destroy(TokenSequence* sequence) {
    if (sequence) {
        free(sequence->tokens);
        free(sequence->positions);
        free(sequence);
    }
}

void bpe_tokenizer_destroy(BPETokenizer* tokenizer) {
    if (tokenizer) {
        if (tokenizer->vocabulary) {
            for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
                free(tokenizer->vocabulary[i]);
            }
            free(tokenizer->vocabulary);
        }
        
        free(tokenizer->token_ids);
        free(tokenizer->token_frequencies);
        free(tokenizer->unk_token);
        free(tokenizer);
    }
}

BPEError bpe_validate_tokenizer(BPETokenizer* tokenizer) {
    if (!tokenizer) {
        return BPE_ERROR_INVALID_PARAM;
    }
    
    if (tokenizer->vocab_size == 0 || !tokenizer->vocabulary || 
        !tokenizer->token_ids || !tokenizer->token_frequencies) {
        return BPE_ERROR_INVALID_VOCABULARY;
    }
    
    // Check for valid UNK token
    if (tokenizer->unk_token_id >= tokenizer->vocab_size) {
        return BPE_ERROR_INVALID_VOCABULARY;
    }
    
    return BPE_SUCCESS;
}

BPEError bpe_benchmark_performance(BPETokenizer* tokenizer,
                                  const char* test_text,
                                  uint32_t iterations,
                                  uint64_t* avg_time_ns,
                                  float* tokens_per_sec) {
    if (!tokenizer || !test_text || iterations == 0 || !avg_time_ns || !tokens_per_sec) {
        return BPE_ERROR_INVALID_PARAM;
    }
    
    uint64_t total_time = 0;
    uint32_t total_tokens = 0;
    
    for (uint32_t i = 0; i < iterations; i++) {
        TokenSequence* sequence;
        TokenizationStats stats;
        
        BPEError error = bpe_tokenize_advanced(tokenizer, test_text, 
                                              tokenizer->max_sequence_length,
                                              true, &sequence, &stats);
        if (error != BPE_SUCCESS) {
            return error;
        }
        
        total_time += stats.processing_time_ns;
        total_tokens += stats.total_tokens;
        
        token_sequence_destroy(sequence);
    }
    
    *avg_time_ns = total_time / iterations;
    *tokens_per_sec = (float)total_tokens / (total_time / 1000000000.0f);
    
    return BPE_SUCCESS;
}

#ifdef USE_METAL
// Integration with Metal memory manager placeholder
BPEError bpe_tokenizer_create_with_metal_memory(BPETokenizer** tokenizer,
                                               const char* vocab_file,
                                               MemoryManager* memory_manager) {
    // For now, use standard memory allocation
    // TODO: Integrate with Metal memory manager for vocabulary storage
    return bpe_tokenizer_create(tokenizer, vocab_file);
}
#endif
