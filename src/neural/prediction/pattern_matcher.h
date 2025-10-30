#ifndef PATTERN_MATCHER_H
#define PATTERN_MATCHER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct PatternMatcher PatternMatcher;
typedef struct PatternMatch PatternMatch;
typedef struct PatternInfo PatternInfo;

// Pattern match structure
struct PatternMatch {
    size_t position;           // Position in data where pattern was found
    size_t length;             // Length of the matched pattern
    uint32_t frequency;        // How many times this pattern appears
    uint32_t last_seen;        // Last position where this pattern was seen
    float score;               // Pattern relevance score
    uint8_t *pattern_data;     // The actual pattern bytes
};

// Pattern information for cache management
struct PatternInfo {
    uint8_t *pattern;          // Pattern bytes
    size_t length;             // Pattern length
    uint32_t frequency;        // Frequency count
    uint32_t last_access;      // Last access time (for LRU)
    uint32_t first_seen;       // First occurrence position
    uint32_t last_seen;        // Most recent occurrence position
    float average_distance;    // Average distance between occurrences
    bool is_active;            // Whether pattern is actively tracked
};

// Pattern matcher configuration
typedef struct {
    size_t min_pattern_length; // Minimum pattern length (default: 2)
    size_t max_pattern_length; // Maximum pattern length (default: 64)
    size_t max_patterns;       // Maximum patterns to track (default: 1024)
    size_t cache_size;         // Pattern cache size (default: 4096)
    size_t search_window;      // Search window size (default: 1024)
    float frequency_threshold; // Minimum frequency to keep pattern (default: 2.0)
    float score_threshold;     // Minimum score to report pattern (default: 0.1)
    bool enable_lru_eviction;  // Enable LRU cache eviction
    bool enable_overlapping;   // Allow overlapping pattern matches
} PatternMatcherConfig;

// Pattern matcher structure
struct PatternMatcher {
    PatternMatcherConfig config;
    
    // Pattern cache management
    PatternInfo *pattern_cache;
    size_t cache_capacity;
    size_t cache_count;
    uint32_t access_counter;
    
    // Search optimization
    uint8_t *search_buffer;
    size_t buffer_capacity;
    size_t buffer_length;
    
    // Hash table for fast pattern lookup
    uint32_t *pattern_hash_table;
    size_t hash_table_size;
    
    // Statistics
    uint64_t total_searches;
    uint64_t total_matches;
    uint64_t patterns_added;
    uint64_t patterns_evicted;
    
    // Performance optimization
    uint32_t *skip_table;      // Boyer-Moore skip table
    bool skip_table_valid;
};

// Return codes
typedef enum {
    PATTERN_MATCHER_SUCCESS = 0,
    PATTERN_MATCHER_ERROR_INVALID_PARAM = -1,
    PATTERN_MATCHER_ERROR_MEMORY_ALLOCATION = -2,
    PATTERN_MATCHER_ERROR_BUFFER_FULL = -3,
    PATTERN_MATCHER_ERROR_PATTERN_TOO_LONG = -4,
    PATTERN_MATCHER_ERROR_PATTERN_TOO_SHORT = -5,
    PATTERN_MATCHER_ERROR_CACHE_FULL = -6
} PatternMatcherResult;

// Core API functions
PatternMatcher* pattern_matcher_create(const PatternMatcherConfig *config);
void pattern_matcher_destroy(PatternMatcher *matcher);

// Pattern detection and matching
PatternMatcherResult pattern_matcher_add_data(
    PatternMatcher *matcher,
    const uint8_t *data,
    size_t data_length
);

PatternMatcherResult pattern_matcher_find_patterns(
    PatternMatcher *matcher,
    const uint8_t *data,
    size_t data_length,
    PatternMatch **matches,
    size_t *num_matches
);

PatternMatcherResult pattern_matcher_search_pattern(
    PatternMatcher *matcher,
    const uint8_t *pattern,
    size_t pattern_length,
    const uint8_t *data,
    size_t data_length,
    size_t **positions,
    size_t *num_positions
);

// Pattern scoring and relevance
float pattern_matcher_calculate_score(
    PatternMatcher *matcher,
    const PatternInfo *pattern,
    size_t current_position
);

PatternMatcherResult pattern_matcher_get_best_patterns(
    PatternMatcher *matcher,
    size_t max_patterns,
    PatternInfo **best_patterns,
    size_t *num_patterns
);

// Cache management
PatternMatcherResult pattern_matcher_add_pattern(
    PatternMatcher *matcher,
    const uint8_t *pattern,
    size_t pattern_length,
    size_t position
);

PatternMatcherResult pattern_matcher_remove_pattern(
    PatternMatcher *matcher,
    const uint8_t *pattern,
    size_t pattern_length
);

void pattern_matcher_clear_cache(PatternMatcher *matcher);

// Pattern lookup and validation
bool pattern_matcher_has_pattern(
    PatternMatcher *matcher,
    const uint8_t *pattern,
    size_t pattern_length
);

PatternInfo* pattern_matcher_get_pattern_info(
    PatternMatcher *matcher,
    const uint8_t *pattern,
    size_t pattern_length
);

// Statistics and debugging
typedef struct {
    size_t total_patterns;
    size_t active_patterns;
    uint64_t total_searches;
    uint64_t total_matches;
    float hit_ratio;
    size_t cache_usage;
    size_t memory_usage;
} PatternMatcherStats;

PatternMatcherStats pattern_matcher_get_stats(const PatternMatcher *matcher);
void pattern_matcher_reset_stats(PatternMatcher *matcher);

// Configuration helpers
PatternMatcherConfig pattern_matcher_config_default(void);
PatternMatcherConfig pattern_matcher_config_fast(void);      // Fast, less memory
PatternMatcherConfig pattern_matcher_config_accurate(void);  // Accurate, more memory
bool pattern_matcher_config_validate(const PatternMatcherConfig *config);

// Utility functions
uint32_t pattern_matcher_hash(const uint8_t *data, size_t length);
bool pattern_matcher_patterns_equal(
    const uint8_t *pattern1, size_t length1,
    const uint8_t *pattern2, size_t length2
);

// Advanced pattern analysis
typedef struct {
    float repetition_score;    // How repetitive the pattern is
    float uniqueness_score;    // How unique the pattern is
    float predictability;      // How predictable the pattern is
    size_t optimal_length;     // Optimal pattern length for matching
} PatternAnalysis;

PatternAnalysis pattern_matcher_analyze_pattern(
    PatternMatcher *matcher,
    const uint8_t *pattern,
    size_t pattern_length
);

// Integration with prediction scoring
typedef struct {
    float pattern_score;       // Pattern relevance score
    size_t match_length;       // Length of best matching pattern
    size_t match_distance;     // Distance to last match
    bool has_strong_match;     // Whether a strong pattern match exists
    uint32_t pattern_frequency;// Frequency of best matching pattern
} PredictionPatternInfo;

PredictionPatternInfo pattern_matcher_analyze_for_prediction(
    PatternMatcher *matcher,
    const uint8_t *context,
    size_t context_length,
    uint8_t candidate_byte
);

// Efficient string search algorithms
size_t* pattern_matcher_boyer_moore_search(
    const uint8_t *text,
    size_t text_length,
    const uint8_t *pattern,
    size_t pattern_length,
    size_t *num_matches
);

size_t* pattern_matcher_kmp_search(
    const uint8_t *text,
    size_t text_length,
    const uint8_t *pattern,
    size_t pattern_length,
    size_t *num_matches
);

// Memory management helpers
void pattern_matcher_compact_cache(PatternMatcher *matcher);
void pattern_matcher_evict_lru_patterns(PatternMatcher *matcher, size_t num_to_evict);

#ifdef __cplusplus
}
#endif

#endif // PATTERN_MATCHER_H
