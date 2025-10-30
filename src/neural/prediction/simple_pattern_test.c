#include "pattern_matcher.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("=== Simple Pattern Matcher Test ===\n");
    
    // Test 1: Configuration
    PatternMatcherConfig config = pattern_matcher_config_fast();
    printf("Config creation: %s\n", 
           pattern_matcher_config_validate(&config) ? "PASS" : "FAIL");
    
    // Test 2: Creation
    PatternMatcher *matcher = pattern_matcher_create(&config);
    printf("Matcher creation: %s\n", matcher ? "PASS" : "FAIL");
    
    if (!matcher) return 1;
    
    // Test 3: Simple pattern addition
    const uint8_t pattern[] = "abc";
    PatternMatcherResult result = pattern_matcher_add_pattern(
        matcher, pattern, 3, 0);
    printf("Pattern addition: %s\n", 
           (result == PATTERN_MATCHER_SUCCESS) ? "PASS" : "FAIL");
    
    // Test 4: Boyer-Moore search
    const uint8_t text[] = "abcdefabc";
    size_t num_matches;
    size_t *positions = pattern_matcher_boyer_moore_search(
        text, 9, pattern, 3, &num_matches);
    printf("Boyer-Moore search: %s (found %zu matches)\n", 
           (positions && num_matches == 2) ? "PASS" : "FAIL", num_matches);
    
    free(positions);
    
    // Test 5: Statistics
    PatternMatcherStats stats = pattern_matcher_get_stats(matcher);
    printf("Statistics: patterns=%zu, memory=%zu\n", 
           stats.total_patterns, stats.memory_usage);
    
    pattern_matcher_destroy(matcher);
    printf("All basic tests completed.\n");
    
    return 0;
}
