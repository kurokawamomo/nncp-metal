#include "pattern_matcher.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Constants
#define DEFAULT_MIN_PATTERN_LENGTH 2
#define DEFAULT_MAX_PATTERN_LENGTH 64
#define DEFAULT_MAX_PATTERNS 1024
#define DEFAULT_CACHE_SIZE 4096
#define DEFAULT_SEARCH_WINDOW 1024
#define DEFAULT_FREQUENCY_THRESHOLD 2.0f
#define DEFAULT_SCORE_THRESHOLD 0.1f
#define HASH_TABLE_SIZE 4096
#define HASH_MULTIPLIER 31

// Utility macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Internal helper functions

// Configuration functions
PatternMatcherConfig pattern_matcher_config_default(void) {
    PatternMatcherConfig config = {
        .min_pattern_length = DEFAULT_MIN_PATTERN_LENGTH,
        .max_pattern_length = DEFAULT_MAX_PATTERN_LENGTH,
        .max_patterns = DEFAULT_MAX_PATTERNS,
        .cache_size = DEFAULT_CACHE_SIZE,
        .search_window = DEFAULT_SEARCH_WINDOW,
        .frequency_threshold = DEFAULT_FREQUENCY_THRESHOLD,
        .score_threshold = DEFAULT_SCORE_THRESHOLD,
        .enable_lru_eviction = true,
        .enable_overlapping = false
    };
    return config;
}

PatternMatcherConfig pattern_matcher_config_fast(void) {
    PatternMatcherConfig config = {
        .min_pattern_length = 3,
        .max_pattern_length = 16,
        .max_patterns = 256,
        .cache_size = 1024,
        .search_window = 512,
        .frequency_threshold = 3.0f,
        .score_threshold = 0.2f,
        .enable_lru_eviction = true,
        .enable_overlapping = false
    };
    return config;
}

PatternMatcherConfig pattern_matcher_config_accurate(void) {
    PatternMatcherConfig config = {
        .min_pattern_length = 2,
        .max_pattern_length = 128,
        .max_patterns = 2048,
        .cache_size = 8192,
        .search_window = 2048,
        .frequency_threshold = 1.5f,
        .score_threshold = 0.05f,
        .enable_lru_eviction = true,
        .enable_overlapping = true
    };
    return config;
}

bool pattern_matcher_config_validate(const PatternMatcherConfig *config) {
    if (!config) return false;
    
    if (config->min_pattern_length < 1 || config->min_pattern_length > 16) return false;
    if (config->max_pattern_length < config->min_pattern_length || 
        config->max_pattern_length > 256) return false;
    if (config->max_patterns < 16 || config->max_patterns > 8192) return false;
    if (config->cache_size < 256 || config->cache_size > 65536) return false;
    if (config->search_window < 64 || config->search_window > 8192) return false;
    if (config->frequency_threshold < 1.0f || config->frequency_threshold > 10.0f) return false;
    if (config->score_threshold < 0.0f || config->score_threshold > 1.0f) return false;
    
    return true;
}

// Core API functions
PatternMatcher* pattern_matcher_create(const PatternMatcherConfig *config) {
    if (!config || !pattern_matcher_config_validate(config)) {
        return NULL;
    }
    
    PatternMatcher *matcher = calloc(1, sizeof(PatternMatcher));
    if (!matcher) return NULL;
    
    matcher->config = *config;
    matcher->hash_table_size = HASH_TABLE_SIZE;
    
    // Allocate pattern cache
    matcher->cache_capacity = config->max_patterns;
    matcher->pattern_cache = calloc(matcher->cache_capacity, sizeof(PatternInfo));
    if (!matcher->pattern_cache) {
        pattern_matcher_destroy(matcher);
        return NULL;
    }
    
    // Allocate search buffer
    matcher->buffer_capacity = config->cache_size;
    matcher->search_buffer = malloc(matcher->buffer_capacity);
    if (!matcher->search_buffer) {
        pattern_matcher_destroy(matcher);
        return NULL;
    }
    
    // Allocate hash table
    matcher->pattern_hash_table = calloc(matcher->hash_table_size, sizeof(uint32_t));
    if (!matcher->pattern_hash_table) {
        pattern_matcher_destroy(matcher);
        return NULL;
    }
    
    // Allocate skip table for Boyer-Moore
    matcher->skip_table = malloc(256 * sizeof(uint32_t));
    if (!matcher->skip_table) {
        pattern_matcher_destroy(matcher);
        return NULL;
    }
    
    matcher->access_counter = 1;
    return matcher;
}

void pattern_matcher_destroy(PatternMatcher *matcher) {
    if (!matcher) return;
    
    if (matcher->pattern_cache) {
        for (size_t i = 0; i < matcher->cache_count; i++) {
            free(matcher->pattern_cache[i].pattern);
        }
        free(matcher->pattern_cache);
    }
    
    free(matcher->search_buffer);
    free(matcher->pattern_hash_table);
    free(matcher->skip_table);
    free(matcher);
}

// Hash function for patterns
uint32_t pattern_matcher_hash(const uint8_t *data, size_t length) {
    if (!data || length == 0) return 0;
    
    uint32_t hash = 0;
    for (size_t i = 0; i < length; i++) {
        hash = hash * HASH_MULTIPLIER + data[i];
    }
    return hash;
}

// Pattern equality check
bool pattern_matcher_patterns_equal(
    const uint8_t *pattern1, size_t length1,
    const uint8_t *pattern2, size_t length2
) {
    if (length1 != length2) return false;
    return memcmp(pattern1, pattern2, length1) == 0;
}

// Pattern management
PatternMatcherResult pattern_matcher_add_pattern(
    PatternMatcher *matcher,
    const uint8_t *pattern,
    size_t pattern_length,
    size_t position
) {
    if (!matcher || !pattern || pattern_length < matcher->config.min_pattern_length ||
        pattern_length > matcher->config.max_pattern_length) {
        return PATTERN_MATCHER_ERROR_INVALID_PARAM;
    }
    
    // Check if pattern already exists
    for (size_t i = 0; i < matcher->cache_count; i++) {
        if (pattern_matcher_patterns_equal(pattern, pattern_length,
                                          matcher->pattern_cache[i].pattern,
                                          matcher->pattern_cache[i].length)) {
            // Update existing pattern
            matcher->pattern_cache[i].frequency++;
            matcher->pattern_cache[i].last_access = matcher->access_counter++;
            matcher->pattern_cache[i].last_seen = (uint32_t)position;
            
            // Update average distance
            if (matcher->pattern_cache[i].frequency > 1) {
                float new_distance = (float)(position - matcher->pattern_cache[i].first_seen) / 
                                   (matcher->pattern_cache[i].frequency - 1);
                matcher->pattern_cache[i].average_distance = new_distance;
            }
            
            return PATTERN_MATCHER_SUCCESS;
        }
    }
    
    // Add new pattern
    if (matcher->cache_count >= matcher->cache_capacity) {
        if (matcher->config.enable_lru_eviction) {
            pattern_matcher_evict_lru_patterns(matcher, 1);
        } else {
            return PATTERN_MATCHER_ERROR_CACHE_FULL;
        }
    }
    
    // Allocate memory for pattern
    uint8_t *pattern_copy = malloc(pattern_length);
    if (!pattern_copy) {
        return PATTERN_MATCHER_ERROR_MEMORY_ALLOCATION;
    }
    memcpy(pattern_copy, pattern, pattern_length);
    
    // Add to cache
    PatternInfo *info = &matcher->pattern_cache[matcher->cache_count];
    info->pattern = pattern_copy;
    info->length = pattern_length;
    info->frequency = 1;
    info->last_access = matcher->access_counter++;
    info->first_seen = (uint32_t)position;
    info->last_seen = (uint32_t)position;
    info->average_distance = 0.0f;
    info->is_active = true;
    
    matcher->cache_count++;
    matcher->patterns_added++;
    
    return PATTERN_MATCHER_SUCCESS;
}

// Pattern detection - Optimized version
PatternMatcherResult pattern_matcher_add_data(
    PatternMatcher *matcher,
    const uint8_t *data,
    size_t data_length
) {
    if (!matcher || !data || data_length == 0) {
        return PATTERN_MATCHER_ERROR_INVALID_PARAM;
    }
    
    // Limit pattern extraction to avoid O(n³) complexity
    size_t max_patterns_to_extract = 100;  // Reasonable limit
    size_t patterns_extracted = 0;
    
    // Extract patterns of various lengths, but with limits
    for (size_t len = matcher->config.min_pattern_length; 
         len <= MIN(matcher->config.max_pattern_length, data_length) && 
         len <= 8; len++) {  // Limit max length for performance
        
        // Sample positions instead of every position
        size_t step = MAX(1, data_length / 50);  // Sample every 1/50th
        
        for (size_t pos = 0; pos <= data_length - len && 
                              patterns_extracted < max_patterns_to_extract; 
             pos += step) {
            
            PatternMatcherResult result = pattern_matcher_add_pattern(
                matcher, &data[pos], len, pos);
            
            patterns_extracted++;
            
            if (result != PATTERN_MATCHER_SUCCESS && 
                result != PATTERN_MATCHER_ERROR_CACHE_FULL) {
                return result;
            }
            
            if (result == PATTERN_MATCHER_ERROR_CACHE_FULL) {
                break;  // Stop if cache is full
            }
        }
    }
    
    return PATTERN_MATCHER_SUCCESS;
}

// Pattern scoring
float pattern_matcher_calculate_score(
    PatternMatcher *matcher,
    const PatternInfo *pattern,
    size_t current_position
) {
    if (!matcher || !pattern) return 0.0f;
    
    // Base score from frequency
    float frequency_score = logf((float)pattern->frequency + 1.0f) / 10.0f;
    
    // Recency score
    uint32_t distance = (current_position > pattern->last_seen) ? 
                       (uint32_t)(current_position - pattern->last_seen) : 0;
    float recency_score = 1.0f / (1.0f + distance / 100.0f);
    
    // Length score (longer patterns are more valuable)
    float length_score = (float)pattern->length / matcher->config.max_pattern_length;
    
    // Regularity score (patterns that appear at regular intervals)
    float regularity_score = 1.0f;
    if (pattern->average_distance > 0.0f) {
        regularity_score = 1.0f / (1.0f + fabsf(pattern->average_distance - 
                          (float)(pattern->last_seen - pattern->first_seen) / 
                          MAX(1, pattern->frequency - 1)) / pattern->average_distance);
    }
    
    return (frequency_score * 0.4f + recency_score * 0.3f + 
            length_score * 0.2f + regularity_score * 0.1f);
}

// Boyer-Moore implementation

size_t* pattern_matcher_boyer_moore_search(
    const uint8_t *text,
    size_t text_length,
    const uint8_t *pattern,
    size_t pattern_length,
    size_t *num_matches
) {
    if (!text || !pattern || !num_matches || text_length < pattern_length) {
        *num_matches = 0;
        return NULL;
    }
    
    // Build skip table
    uint32_t skip_table[256];
    for (int i = 0; i < 256; i++) {
        skip_table[i] = (uint32_t)pattern_length;
    }
    
    for (size_t i = 0; i < pattern_length - 1; i++) {
        skip_table[pattern[i]] = (uint32_t)(pattern_length - 1 - i);
    }
    
    // Search for matches
    size_t *matches = NULL;
    size_t match_capacity = 0;
    *num_matches = 0;
    
    size_t i = pattern_length - 1;
    while (i < text_length) {
        size_t j = pattern_length - 1;
        
        // Compare pattern from right to left
        while (j < pattern_length && text[i] == pattern[j]) {
            if (j == 0) {
                // Match found
                if (*num_matches >= match_capacity) {
                    match_capacity = match_capacity ? match_capacity * 2 : 16;
                    size_t *new_matches = realloc(matches, match_capacity * sizeof(size_t));
                    if (!new_matches) {
                        free(matches);
                        *num_matches = 0;
                        return NULL;
                    }
                    matches = new_matches;
                }
                matches[(*num_matches)++] = i - pattern_length + 1;
                break;
            }
            i--;
            j--;
        }
        
        // Skip based on mismatch character
        if (j == pattern_length || i >= text_length) {
            i += 1;
        } else {
            size_t skip = skip_table[text[i]];
            i += MAX(1, skip - (pattern_length - 1 - j));
        }
    }
    
    return matches;
}

// Pattern search
PatternMatcherResult pattern_matcher_search_pattern(
    PatternMatcher *matcher,
    const uint8_t *pattern,
    size_t pattern_length,
    const uint8_t *data,
    size_t data_length,
    size_t **positions,
    size_t *num_positions
) {
    if (!matcher || !pattern || !data || !positions || !num_positions) {
        return PATTERN_MATCHER_ERROR_INVALID_PARAM;
    }
    
    matcher->total_searches++;
    
    *positions = pattern_matcher_boyer_moore_search(
        data, data_length, pattern, pattern_length, num_positions);
    
    if (*num_positions > 0) {
        matcher->total_matches += *num_positions;
    }
    
    return PATTERN_MATCHER_SUCCESS;
}

// Find patterns in data
PatternMatcherResult pattern_matcher_find_patterns(
    PatternMatcher *matcher,
    const uint8_t *data,
    size_t data_length,
    PatternMatch **matches,
    size_t *num_matches
) {
    if (!matcher || !data || !matches || !num_matches) {
        return PATTERN_MATCHER_ERROR_INVALID_PARAM;
    }
    
    *matches = NULL;
    *num_matches = 0;
    
    size_t match_capacity = 0;
    
    // Search for each cached pattern
    for (size_t i = 0; i < matcher->cache_count; i++) {
        PatternInfo *pattern = &matcher->pattern_cache[i];
        if (!pattern->is_active) continue;
        
        size_t *positions;
        size_t num_positions;
        
        PatternMatcherResult result = pattern_matcher_search_pattern(
            matcher, pattern->pattern, pattern->length, 
            data, data_length, &positions, &num_positions);
        
        if (result != PATTERN_MATCHER_SUCCESS) continue;
        
        // Add matches with scores above threshold
        for (size_t j = 0; j < num_positions; j++) {
            float score = pattern_matcher_calculate_score(matcher, pattern, positions[j]);
            
            if (score >= matcher->config.score_threshold) {
                // Expand matches array if needed
                if (*num_matches >= match_capacity) {
                    match_capacity = match_capacity ? match_capacity * 2 : 32;
                    PatternMatch *new_matches = realloc(*matches, 
                                                       match_capacity * sizeof(PatternMatch));
                    if (!new_matches) {
                        free(*matches);
                        free(positions);
                        return PATTERN_MATCHER_ERROR_MEMORY_ALLOCATION;
                    }
                    *matches = new_matches;
                }
                
                // Create match
                PatternMatch *match = &(*matches)[*num_matches];
                match->position = positions[j];
                match->length = pattern->length;
                match->frequency = pattern->frequency;
                match->last_seen = pattern->last_seen;
                match->score = score;
                
                // Copy pattern data
                match->pattern_data = malloc(pattern->length);
                if (match->pattern_data) {
                    memcpy(match->pattern_data, pattern->pattern, pattern->length);
                }
                
                (*num_matches)++;
            }
        }
        
        free(positions);
    }
    
    return PATTERN_MATCHER_SUCCESS;
}

// Cache management
void pattern_matcher_evict_lru_patterns(PatternMatcher *matcher, size_t num_to_evict) {
    if (!matcher || num_to_evict == 0) return;
    
    for (size_t evicted = 0; evicted < num_to_evict && matcher->cache_count > 0; evicted++) {
        // Find LRU pattern
        size_t lru_index = 0;
        uint32_t oldest_access = matcher->pattern_cache[0].last_access;
        
        for (size_t i = 1; i < matcher->cache_count; i++) {
            if (matcher->pattern_cache[i].last_access < oldest_access) {
                oldest_access = matcher->pattern_cache[i].last_access;
                lru_index = i;
            }
        }
        
        // Free pattern memory
        free(matcher->pattern_cache[lru_index].pattern);
        
        // Move last pattern to fill gap
        if (lru_index < matcher->cache_count - 1) {
            matcher->pattern_cache[lru_index] = matcher->pattern_cache[matcher->cache_count - 1];
        }
        
        matcher->cache_count--;
        matcher->patterns_evicted++;
    }
}

void pattern_matcher_clear_cache(PatternMatcher *matcher) {
    if (!matcher) return;
    
    for (size_t i = 0; i < matcher->cache_count; i++) {
        free(matcher->pattern_cache[i].pattern);
    }
    
    matcher->cache_count = 0;
    matcher->access_counter = 1;
}

// Statistics
PatternMatcherStats pattern_matcher_get_stats(const PatternMatcher *matcher) {
    PatternMatcherStats stats = {0};
    
    if (matcher) {
        stats.total_patterns = matcher->cache_count;
        stats.total_searches = matcher->total_searches;
        stats.total_matches = matcher->total_matches;
        stats.hit_ratio = matcher->total_searches > 0 ? 
            (float)matcher->total_matches / matcher->total_searches : 0.0f;
        stats.cache_usage = matcher->cache_count;
        
        // Count active patterns
        for (size_t i = 0; i < matcher->cache_count; i++) {
            if (matcher->pattern_cache[i].is_active) {
                stats.active_patterns++;
            }
        }
        
        // Estimate memory usage
        stats.memory_usage = sizeof(PatternMatcher) + 
                           matcher->cache_capacity * sizeof(PatternInfo) +
                           matcher->buffer_capacity +
                           matcher->hash_table_size * sizeof(uint32_t) +
                           256 * sizeof(uint32_t);
        
        for (size_t i = 0; i < matcher->cache_count; i++) {
            stats.memory_usage += matcher->pattern_cache[i].length;
        }
    }
    
    return stats;
}

void pattern_matcher_reset_stats(PatternMatcher *matcher) {
    if (!matcher) return;
    
    matcher->total_searches = 0;
    matcher->total_matches = 0;
    matcher->patterns_added = 0;
    matcher->patterns_evicted = 0;
}

// Integration with prediction scoring
PredictionPatternInfo pattern_matcher_analyze_for_prediction(
    PatternMatcher *matcher,
    const uint8_t *context,
    size_t context_length,
    uint8_t candidate_byte
) {
    PredictionPatternInfo info = {0};
    
    if (!matcher || !context || context_length == 0) {
        return info;
    }
    
    // Create candidate pattern by appending candidate byte to context
    size_t max_pattern_length = MIN(context_length + 1, matcher->config.max_pattern_length);
    uint8_t *candidate_pattern = malloc(max_pattern_length);
    if (!candidate_pattern) return info;
    
    // Try different pattern lengths ending with candidate byte
    float best_score = 0.0f;
    size_t best_length = 0;
    
    for (size_t len = matcher->config.min_pattern_length; 
         len <= MIN(max_pattern_length, context_length + 1); len++) {
        
        if (len <= context_length) {
            memcpy(candidate_pattern, &context[context_length - len + 1], len - 1);
            candidate_pattern[len - 1] = candidate_byte;
        } else {
            memcpy(candidate_pattern, context, context_length);
            candidate_pattern[context_length] = candidate_byte;
        }
        
        // Check if this pattern exists in cache
        for (size_t i = 0; i < matcher->cache_count; i++) {
            PatternInfo *pattern = &matcher->pattern_cache[i];
            if (pattern_matcher_patterns_equal(candidate_pattern, len,
                                              pattern->pattern, pattern->length)) {
                
                float score = pattern_matcher_calculate_score(matcher, pattern, context_length);
                if (score > best_score) {
                    best_score = score;
                    best_length = len;
                    info.pattern_frequency = pattern->frequency;
                    info.match_distance = context_length > pattern->last_seen ? 
                                        context_length - pattern->last_seen : 0;
                }
            }
        }
    }
    
    free(candidate_pattern);
    
    info.pattern_score = best_score;
    info.match_length = best_length;
    info.has_strong_match = best_score > matcher->config.score_threshold;
    
    return info;
}
