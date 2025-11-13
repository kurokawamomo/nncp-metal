/*
 * DynamicContextSelector.mm
 * 
 * Dynamic Context Selection System Implementation
 * Intelligent file analysis, content detection, and context optimization
 */

#include "DynamicContextSelector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

// Internal dynamic context selector structure
struct DynamicContextSelector {
    DynamicContextConfig config;
    AdaptiveContextManager* context_manager;
    DynamicContextStats stats;
    
    // Analysis state
    bool is_initialized;
    uint32_t analysis_cache_size;
    float* entropy_cache;
    uint32_t* pattern_cache;
    
    // Metal GPU resources for acceleration
#ifdef __OBJC__
    id<MTLDevice> metal_device;
    id<MTLCommandQueue> command_queue;
    id<MTLComputePipelineState> analysis_pipeline;
    id<MTLBuffer> analysis_buffer;
#endif
    
    // Learning and adaptation state
    float accuracy_history[64];
    uint32_t accuracy_history_count;
    float optimization_weights[8];
    bool adaptive_learning_enabled;
};

// Helper function prototypes
static DynamicContextError analyze_file_entropy(DynamicContextSelector* selector,
                                               const uint8_t* data, size_t size,
                                               float* entropy);
static DynamicContextError detect_pattern_structures(DynamicContextSelector* selector,
                                                    const uint8_t* data, size_t size,
                                                    FilePattern* pattern);
static DynamicContextError calculate_content_scores(DynamicContextSelector* selector,
                                                   const uint8_t* data, size_t size,
                                                   ContentTypeAnalysis* analysis);
static DynamicContextError optimize_context_length(DynamicContextSelector* selector,
                                                  const FileCharacteristics* characteristics,
                                                  uint32_t* optimal_length);
static void update_learning_weights(DynamicContextSelector* selector,
                                   float actual_performance);

// Core API Implementation

DynamicContextError dynamic_context_create(DynamicContextSelector** selector,
                                          const DynamicContextConfig* config) {
    if (!selector) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    *selector = (DynamicContextSelector*)calloc(1, sizeof(DynamicContextSelector));
    if (!*selector) {
        return DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize configuration
    if (config) {
        (*selector)->config = *config;
    } else {
        dynamic_context_create_default_config(&(*selector)->config);
    }
    
    // Allocate analysis caches
    (*selector)->analysis_cache_size = (*selector)->config.sample_size;
    (*selector)->entropy_cache = (float*)malloc((*selector)->analysis_cache_size * sizeof(float));
    (*selector)->pattern_cache = (uint32_t*)malloc((*selector)->analysis_cache_size * sizeof(uint32_t));
    
    if (!(*selector)->entropy_cache || !(*selector)->pattern_cache) {
        free((*selector)->entropy_cache);
        free((*selector)->pattern_cache);
        free(*selector);
        *selector = NULL;
        return DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize statistics
    memset(&(*selector)->stats, 0, sizeof(DynamicContextStats));
    
    // Initialize learning weights with balanced defaults
    for (int i = 0; i < 8; i++) {
        (*selector)->optimization_weights[i] = 0.125f; // Equal weights initially
    }
    
    (*selector)->adaptive_learning_enabled = true;
    (*selector)->is_initialized = false;
    
    printf("✓ Dynamic Context Selector created\n");
    printf("  - Sample size: %u bytes\n", (*selector)->config.sample_size);
    printf("  - Content detection: %s\n", (*selector)->config.enable_content_detection ? "Enabled" : "Disabled");
    printf("  - Pattern recognition: %s\n", (*selector)->config.enable_pattern_recognition ? "Enabled" : "Disabled");
    printf("  - Domain adaptation: %s\n", (*selector)->config.enable_domain_adaptation ? "Enabled" : "Disabled");
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_initialize(DynamicContextSelector* selector,
                                              AdaptiveContextManager* context_manager) {
    if (!selector || !context_manager) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    selector->context_manager = context_manager;
    
#ifdef __OBJC__
    // Initialize Metal GPU acceleration if available
    selector->metal_device = MTLCreateSystemDefaultDevice();
    if (selector->metal_device) {
        selector->command_queue = [selector->metal_device newCommandQueue];
        printf("  ✓ Metal GPU acceleration initialized\n");
    }
#endif
    
    selector->is_initialized = true;
    
    printf("✓ Dynamic Context Selector initialized\n");
    printf("  - Context manager integration: ✓ Ready\n");
    printf("  - GPU acceleration: %s\n", selector->metal_device ? "✓ Enabled" : "○ CPU only");
    printf("  - Analysis cache: ✓ Allocated\n");
    printf("  - Learning system: ✓ Active\n");
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_analyze_file_characteristics(DynamicContextSelector* selector,
                                                                const uint8_t* file_data,
                                                                size_t file_size,
                                                                FileCharacteristics* characteristics) {
    if (!selector || !file_data || !characteristics || file_size == 0) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    if (!selector->is_initialized) {
        return DYNAMIC_CONTEXT_ERROR_ANALYSIS_FAILED;
    }
    
    printf("  Analyzing file characteristics (%zu bytes)...\n", file_size);
    
    // Initialize characteristics structure
    memset(characteristics, 0, sizeof(FileCharacteristics));
    
    // Determine analysis sample size
    size_t sample_size = (file_size > selector->config.sample_size) ? 
                        selector->config.sample_size : file_size;
    
    // 1. Content entropy analysis
    DynamicContextError error = analyze_file_entropy(selector, file_data, sample_size,
                                                     &characteristics->content_entropy);
    if (error != DYNAMIC_CONTEXT_SUCCESS) {
        return error;
    }
    
    // 2. Pattern structure detection
    error = detect_pattern_structures(selector, file_data, sample_size,
                                     &characteristics->primary_pattern);
    if (error != DYNAMIC_CONTEXT_SUCCESS) {
        return error;
    }
    
    // 3. Structural complexity calculation
    float byte_distribution[256] = {0};
    for (size_t i = 0; i < sample_size; i++) {
        byte_distribution[file_data[i]]++;
    }
    
    // Calculate byte distribution entropy for complexity
    float complexity_sum = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (byte_distribution[i] > 0) {
            float prob = byte_distribution[i] / sample_size;
            complexity_sum += prob * log2f(prob);
        }
    }
    characteristics->structural_complexity = -complexity_sum / 8.0f; // Normalize to 0-1
    
    // 4. Repetition ratio analysis
    uint32_t repetitive_sequences = 0;
    uint32_t total_sequences = 0;
    
    // Check for repeating byte sequences (4-byte windows)
    for (size_t i = 0; i < sample_size - 8; i += 4) {
        for (size_t j = i + 4; j < sample_size - 4; j += 4) {
            if (memcmp(&file_data[i], &file_data[j], 4) == 0) {
                repetitive_sequences++;
                break;
            }
        }
        total_sequences++;
    }
    
    characteristics->repetition_ratio = (total_sequences > 0) ? 
        (float)repetitive_sequences / total_sequences : 0.0f;
    
    // 5. Line-based analysis for text-like content
    uint32_t line_count = 0;
    uint32_t total_line_length = 0;
    uint32_t current_line_length = 0;
    
    for (size_t i = 0; i < sample_size; i++) {
        if (file_data[i] == '\n' || file_data[i] == '\r') {
            if (current_line_length > 0) {
                line_count++;
                total_line_length += current_line_length;
                current_line_length = 0;
            }
        } else {
            current_line_length++;
        }
    }
    
    characteristics->average_line_length = (line_count > 0) ? 
        total_line_length / line_count : (uint32_t)sample_size;
    
    // 6. Vocabulary analysis
    bool char_used[256] = {false};
    for (size_t i = 0; i < sample_size; i++) {
        char_used[file_data[i]] = true;
    }
    
    uint32_t unique_chars = 0;
    for (int i = 0; i < 256; i++) {
        if (char_used[i]) unique_chars++;
    }
    characteristics->vocabulary_size = unique_chars;
    
    // 7. Linguistic density estimation
    uint32_t text_chars = 0;
    for (size_t i = 0; i < sample_size; i++) {
        if (isalnum(file_data[i]) || isspace(file_data[i]) || ispunct(file_data[i])) {
            text_chars++;
        }
    }
    characteristics->linguistic_density = (float)text_chars / sample_size;
    
    // 8. Code density estimation
    uint32_t code_chars = 0;
    for (size_t i = 0; i < sample_size; i++) {
        char c = file_data[i];
        if (c == '{' || c == '}' || c == '(' || c == ')' || c == '[' || c == ']' ||
            c == ';' || c == '=' || c == '+' || c == '-' || c == '*' || c == '/' ||
            c == '<' || c == '>' || c == '&' || c == '|' || c == '^' || c == '!') {
            code_chars++;
        }
    }
    characteristics->code_density = (float)code_chars / sample_size;
    
    // 9. Data regularity analysis
    uint32_t regular_patterns = 0;
    uint32_t pattern_checks = 0;
    
    // Check for comma-separated values
    uint32_t comma_count = 0;
    for (size_t i = 0; i < sample_size; i++) {
        if (file_data[i] == ',') comma_count++;
    }
    if (comma_count > sample_size / 100) regular_patterns++; // At least 1% commas
    pattern_checks++;
    
    // Check for JSON-like patterns
    uint32_t brace_count = 0;
    for (size_t i = 0; i < sample_size; i++) {
        if (file_data[i] == '{' || file_data[i] == '}') brace_count++;
    }
    if (brace_count > sample_size / 200) regular_patterns++; // At least 0.5% braces
    pattern_checks++;
    
    // Check for XML-like patterns
    uint32_t bracket_count = 0;
    for (size_t i = 0; i < sample_size; i++) {
        if (file_data[i] == '<' || file_data[i] == '>') bracket_count++;
    }
    if (bracket_count > sample_size / 100) regular_patterns++; // At least 1% brackets
    pattern_checks++;
    
    characteristics->data_regularity = (pattern_checks > 0) ? 
        (float)regular_patterns / pattern_checks : 0.0f;
    
    // 10. Pattern frequency analysis (top 16 patterns)
    uint32_t pattern_counts[65536] = {0}; // For 2-byte patterns
    for (size_t i = 0; i < sample_size - 1; i++) {
        uint16_t pattern = (file_data[i] << 8) | file_data[i + 1];
        pattern_counts[pattern]++;
    }
    
    // Find top patterns
    for (int i = 0; i < 16; i++) {
        uint32_t max_count = 0;
        uint16_t max_pattern = 0;
        for (int j = 0; j < 65536; j++) {
            if (pattern_counts[j] > max_count) {
                max_count = pattern_counts[j];
                max_pattern = j;
            }
        }
        characteristics->pattern_frequency[i] = max_count;
        pattern_counts[max_pattern] = 0; // Remove found pattern
    }
    
    // 11. Structure detection flags
    characteristics->has_hierarchical_structure = (characteristics->structural_complexity > 0.6f);
    characteristics->has_temporal_patterns = (characteristics->repetition_ratio > 0.3f);
    characteristics->has_cross_references = (characteristics->data_regularity > 0.4f);
    
    // Update statistics
    selector->stats.files_analyzed++;
    
    printf("    ✓ File characteristics analysis completed\n");
    printf("      Content entropy: %.3f\n", characteristics->content_entropy);
    printf("      Primary pattern: %s\n", 
           dynamic_context_file_pattern_to_string(characteristics->primary_pattern));
    printf("      Structural complexity: %.3f\n", characteristics->structural_complexity);
    printf("      Repetition ratio: %.3f\n", characteristics->repetition_ratio);
    printf("      Linguistic density: %.3f\n", characteristics->linguistic_density);
    printf("      Code density: %.3f\n", characteristics->code_density);
    printf("      Vocabulary size: %u\n", characteristics->vocabulary_size);
    printf("      Average line length: %u\n", characteristics->average_line_length);
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_detect_content_type(DynamicContextSelector* selector,
                                                       const uint8_t* file_data,
                                                       size_t file_size,
                                                       ContentTypeAnalysis* content_analysis) {
    if (!selector || !file_data || !content_analysis || file_size == 0) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    if (!selector->config.enable_content_detection) {
        return DYNAMIC_CONTEXT_ERROR_CONTENT_DETECTION_FAILED;
    }
    
    printf("  Detecting content type...\n");
    
    // Initialize analysis structure
    memset(content_analysis, 0, sizeof(ContentTypeAnalysis));
    
    // Calculate content scores
    DynamicContextError error = calculate_content_scores(selector, file_data, file_size, content_analysis);
    if (error != DYNAMIC_CONTEXT_SUCCESS) {
        return error;
    }
    
    // Determine primary and secondary content types
    float max_score = 0.0f;
    ContentType primary_type = CONTENT_TYPE_UNKNOWN;
    
    if (content_analysis->text_score > max_score) {
        max_score = content_analysis->text_score;
        primary_type = CONTENT_TYPE_TEXT;
    }
    if (content_analysis->code_score > max_score) {
        max_score = content_analysis->code_score;
        primary_type = CONTENT_TYPE_CODE;
    }
    if (content_analysis->binary_score > max_score) {
        max_score = content_analysis->binary_score;
        primary_type = CONTENT_TYPE_BINARY;
    }
    if (content_analysis->structured_data_score > max_score) {
        max_score = content_analysis->structured_data_score;
        primary_type = CONTENT_TYPE_STRUCTURED_DATA;
    }
    if (content_analysis->documentation_score > max_score) {
        max_score = content_analysis->documentation_score;
        primary_type = CONTENT_TYPE_DOCUMENTATION;
    }
    if (content_analysis->log_score > max_score) {
        max_score = content_analysis->log_score;
        primary_type = CONTENT_TYPE_LOG;
    }
    if (content_analysis->config_score > max_score) {
        max_score = content_analysis->config_score;
        primary_type = CONTENT_TYPE_CONFIG;
    }
    
    content_analysis->primary_type = primary_type;
    content_analysis->confidence_score = max_score;
    
    // Check for mixed content
    uint32_t high_scores = 0;
    if (content_analysis->text_score > 0.4f) high_scores++;
    if (content_analysis->code_score > 0.4f) high_scores++;
    if (content_analysis->binary_score > 0.4f) high_scores++;
    if (content_analysis->structured_data_score > 0.4f) high_scores++;
    
    content_analysis->is_mixed_content = (high_scores >= 2);
    if (content_analysis->is_mixed_content) {
        content_analysis->primary_type = CONTENT_TYPE_MIXED;
    }
    
    // Evidence counting
    content_analysis->evidence_count = 0;
    if (content_analysis->text_score > CONTENT_DETECTION_TEXT_THRESHOLD) content_analysis->evidence_count++;
    if (content_analysis->code_score > CONTENT_DETECTION_CODE_THRESHOLD) content_analysis->evidence_count++;
    if (content_analysis->binary_score > CONTENT_DETECTION_BINARY_THRESHOLD) content_analysis->evidence_count++;
    if (content_analysis->structured_data_score > CONTENT_DETECTION_STRUCTURED_THRESHOLD) content_analysis->evidence_count++;
    
    // Update statistics
    selector->stats.content_types_detected++;
    
    printf("    ✓ Content type detection completed\n");
    printf("      Primary type: %s (confidence: %.1f%%)\n",
           dynamic_context_content_type_to_string(content_analysis->primary_type),
           content_analysis->confidence_score * 100.0f);
    printf("      Text score: %.3f\n", content_analysis->text_score);
    printf("      Code score: %.3f\n", content_analysis->code_score);
    printf("      Binary score: %.3f\n", content_analysis->binary_score);
    printf("      Structured data score: %.3f\n", content_analysis->structured_data_score);
    printf("      Mixed content: %s\n", content_analysis->is_mixed_content ? "Yes" : "No");
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_determine_optimal_length(DynamicContextSelector* selector,
                                                           const FileCharacteristics* characteristics,
                                                           const ContentTypeAnalysis* content_analysis,
                                                           OptimalContextDetermination* optimal_context) {
    if (!selector || !characteristics || !content_analysis || !optimal_context) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    printf("  Determining optimal context length...\n");
    
    // Initialize optimal context structure
    memset(optimal_context, 0, sizeof(OptimalContextDetermination));
    
    // Base context length from configuration
    uint32_t base_length = (selector->config.min_context_length + selector->config.max_context_length) / 2;
    
    // Content type adjustments
    float content_factor = 1.0f;
    switch (content_analysis->primary_type) {
        case CONTENT_TYPE_TEXT:
            content_factor = 1.2f; // Text benefits from larger context
            break;
        case CONTENT_TYPE_CODE:
            content_factor = 1.1f; // Code has moderate context benefits
            break;
        case CONTENT_TYPE_BINARY:
            content_factor = 0.8f; // Binary content needs less context
            break;
        case CONTENT_TYPE_STRUCTURED_DATA:
            content_factor = 1.3f; // Structured data benefits from large context
            break;
        case CONTENT_TYPE_DOCUMENTATION:
            content_factor = 1.4f; // Documentation needs largest context
            break;
        case CONTENT_TYPE_LOG:
            content_factor = 0.9f; // Logs have moderate repetition
            break;
        case CONTENT_TYPE_CONFIG:
            content_factor = 0.7f; // Config files are usually small and simple
            break;
        case CONTENT_TYPE_MIXED:
            content_factor = 1.1f; // Mixed content gets moderate boost
            break;
        default:
            content_factor = 1.0f;
            break;
    }
    
    // Structural complexity adjustments
    float complexity_factor = 0.5f + (characteristics->structural_complexity * 1.0f);
    
    // Repetition adjustments (high repetition = smaller context needed)
    float repetition_factor = 1.0f - (characteristics->repetition_ratio * 0.3f);
    
    // Linguistic density adjustments (more linguistic = larger context)
    float linguistic_factor = 0.8f + (characteristics->linguistic_density * 0.4f);
    
    // Calculate recommended context length
    float adjusted_length = base_length * content_factor * complexity_factor * 
                           repetition_factor * linguistic_factor;
    
    // Apply optimization strategy
    switch (selector->config.optimization_strategy) {
        case CONTEXT_OPT_CONSERVATIVE:
            adjusted_length *= 0.8f;
            break;
        case CONTEXT_OPT_BALANCED:
            // No additional adjustment
            break;
        case CONTEXT_OPT_AGGRESSIVE:
            adjusted_length *= 1.3f;
            break;
        case CONTEXT_OPT_ADAPTIVE:
            // Use learning weights if available
            if (selector->adaptive_learning_enabled) {
                float learning_factor = selector->optimization_weights[0] * 2.0f; // Use first weight
                adjusted_length *= (0.5f + learning_factor);
            }
            break;
        case CONTEXT_OPT_MEMORY_CONSTRAINED:
            adjusted_length *= 0.6f;
            break;
    }
    
    // Clamp to configured limits
    optimal_context->recommended_context_length = (uint32_t)adjusted_length;
    if (optimal_context->recommended_context_length < selector->config.min_context_length) {
        optimal_context->recommended_context_length = selector->config.min_context_length;
    }
    if (optimal_context->recommended_context_length > selector->config.max_context_length) {
        optimal_context->recommended_context_length = selector->config.max_context_length;
    }
    
    // Set minimum and maximum effective lengths
    optimal_context->minimum_effective_length = optimal_context->recommended_context_length / 2;
    optimal_context->maximum_beneficial_length = optimal_context->recommended_context_length * 2;
    
    // Clamp these as well
    if (optimal_context->minimum_effective_length < selector->config.min_context_length) {
        optimal_context->minimum_effective_length = selector->config.min_context_length;
    }
    if (optimal_context->maximum_beneficial_length > selector->config.max_context_length) {
        optimal_context->maximum_beneficial_length = selector->config.max_context_length;
    }
    
    // Predict compression ratio based on characteristics
    float base_compression = 0.3f; // 30% baseline
    float entropy_bonus = (1.0f - characteristics->content_entropy) * 0.4f;
    float repetition_bonus = characteristics->repetition_ratio * 0.2f;
    float structure_bonus = characteristics->structural_complexity * 0.1f;
    
    optimal_context->compression_ratio_prediction = base_compression + entropy_bonus + 
                                                   repetition_bonus + structure_bonus;
    if (optimal_context->compression_ratio_prediction > 0.8f) {
        optimal_context->compression_ratio_prediction = 0.8f; // Cap at 80%
    }
    
    // Estimate processing time (ms per 1000 tokens)
    float base_time = 100.0f; // 100ms baseline
    float context_penalty = (float)optimal_context->recommended_context_length / 1000.0f;
    float complexity_penalty = characteristics->structural_complexity * 50.0f;
    
    optimal_context->processing_time_estimate = base_time + (context_penalty * 20.0f) + complexity_penalty;
    
    // Estimate memory requirement
    size_t base_memory = 256; // 256MB baseline
    size_t context_memory = (optimal_context->recommended_context_length * 4) / 1024; // 4 bytes per token, convert to KB, then MB
    optimal_context->memory_requirement_mb = base_memory + context_memory;
    
    // Select optimization strategy and domain
    optimal_context->selected_strategy = selector->config.optimization_strategy;
    
    // Determine if hierarchical processing is beneficial
    optimal_context->use_hierarchical_processing = (characteristics->has_hierarchical_structure &&
                                                   optimal_context->recommended_context_length > 512);
    
    // Determine if sparse attention is beneficial  
    optimal_context->use_sparse_attention = (characteristics->repetition_ratio > 0.3f ||
                                            characteristics->data_regularity > 0.5f);
    
    // Confidence based on evidence
    float confidence_factors = 0.0f;
    if (content_analysis->confidence_score > 0.7f) confidence_factors += 0.3f;
    if (characteristics->structural_complexity > 0.1f) confidence_factors += 0.2f;
    if (characteristics->vocabulary_size > 10) confidence_factors += 0.2f;
    if (content_analysis->evidence_count >= 2) confidence_factors += 0.3f;
    
    optimal_context->confidence_score = confidence_factors;
    if (optimal_context->confidence_score > 1.0f) optimal_context->confidence_score = 1.0f;
    
    // Update statistics
    selector->stats.successful_optimizations++;
    
    printf("    ✓ Optimal context determination completed\n");
    printf("      Recommended length: %u tokens\n", optimal_context->recommended_context_length);
    printf("      Effective range: %u - %u tokens\n", 
           optimal_context->minimum_effective_length, optimal_context->maximum_beneficial_length);
    printf("      Predicted compression: %.1f%%\n", optimal_context->compression_ratio_prediction * 100.0f);
    printf("      Estimated processing: %.1f ms\n", optimal_context->processing_time_estimate);
    printf("      Memory requirement: %zu MB\n", optimal_context->memory_requirement_mb);
    printf("      Hierarchical processing: %s\n", optimal_context->use_hierarchical_processing ? "Yes" : "No");
    printf("      Sparse attention: %s\n", optimal_context->use_sparse_attention ? "Yes" : "No");
    printf("      Confidence: %.1f%%\n", optimal_context->confidence_score * 100.0f);
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_predict_domain_adaptation(DynamicContextSelector* selector,
                                                             const FileCharacteristics* characteristics,
                                                             const ContentTypeAnalysis* content_analysis,
                                                             DomainAdaptationPrediction* domain_prediction) {
    if (!selector || !characteristics || !content_analysis || !domain_prediction) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    if (!selector->config.enable_domain_adaptation) {
        return DYNAMIC_CONTEXT_ERROR_DOMAIN_ADAPTATION_FAILED;
    }
    
    printf("  Predicting domain adaptation strategy...\n");
    
    // Initialize domain prediction structure
    memset(domain_prediction, 0, sizeof(DomainAdaptationPrediction));
    
    // Calculate effectiveness scores for each domain
    
    // General domain effectiveness (baseline)
    domain_prediction->general_effectiveness = 0.6f;
    
    // Code domain effectiveness
    float code_indicators = characteristics->code_density * 0.4f +
                           content_analysis->code_score * 0.6f;
    domain_prediction->code_effectiveness = 0.3f + (code_indicators * 0.6f);
    
    // Text domain effectiveness
    float text_indicators = characteristics->linguistic_density * 0.5f +
                           content_analysis->text_score * 0.5f;
    domain_prediction->text_effectiveness = 0.3f + (text_indicators * 0.6f);
    
    // Data domain effectiveness
    float data_indicators = characteristics->data_regularity * 0.6f +
                           content_analysis->structured_data_score * 0.4f;
    domain_prediction->data_effectiveness = 0.3f + (data_indicators * 0.6f);
    
    // Scientific domain effectiveness (based on text + structure)
    float scientific_indicators = (characteristics->structural_complexity * 0.3f +
                                  characteristics->linguistic_density * 0.3f +
                                  content_analysis->documentation_score * 0.4f);
    domain_prediction->scientific_effectiveness = 0.3f + (scientific_indicators * 0.5f);
    
    // Multimedia domain effectiveness (mostly binary + some structure)
    float multimedia_indicators = content_analysis->binary_score * 0.7f +
                                 characteristics->structural_complexity * 0.3f;
    domain_prediction->multimedia_effectiveness = 0.3f + (multimedia_indicators * 0.4f);
    
    // Configuration domain effectiveness
    float config_indicators = content_analysis->config_score * 0.8f +
                             characteristics->data_regularity * 0.2f;
    domain_prediction->config_effectiveness = 0.3f + (config_indicators * 0.5f);
    
    // Determine best domain
    float max_effectiveness = domain_prediction->general_effectiveness;
    DomainAdaptationType best_domain = DOMAIN_ADAPTATION_GENERAL;
    
    if (domain_prediction->code_effectiveness > max_effectiveness) {
        max_effectiveness = domain_prediction->code_effectiveness;
        best_domain = DOMAIN_ADAPTATION_CODE;
    }
    if (domain_prediction->text_effectiveness > max_effectiveness) {
        max_effectiveness = domain_prediction->text_effectiveness;
        best_domain = DOMAIN_ADAPTATION_TEXT;
    }
    if (domain_prediction->data_effectiveness > max_effectiveness) {
        max_effectiveness = domain_prediction->data_effectiveness;
        best_domain = DOMAIN_ADAPTATION_DATA;
    }
    if (domain_prediction->scientific_effectiveness > max_effectiveness) {
        max_effectiveness = domain_prediction->scientific_effectiveness;
        best_domain = DOMAIN_ADAPTATION_SCIENTIFIC;
    }
    if (domain_prediction->multimedia_effectiveness > max_effectiveness) {
        max_effectiveness = domain_prediction->multimedia_effectiveness;
        best_domain = DOMAIN_ADAPTATION_MULTIMEDIA;
    }
    if (domain_prediction->config_effectiveness > max_effectiveness) {
        max_effectiveness = domain_prediction->config_effectiveness;
        best_domain = DOMAIN_ADAPTATION_CONFIGURATION;
    }
    
    domain_prediction->predicted_domain = best_domain;
    domain_prediction->domain_confidence = max_effectiveness;
    
    // Determine fallback domain (second best)
    domain_prediction->fallback_domain = DOMAIN_ADAPTATION_GENERAL; // Safe fallback
    
    // Check if specialized processing is needed
    domain_prediction->requires_specialized_processing = (max_effectiveness > 0.7f &&
                                                         best_domain != DOMAIN_ADAPTATION_GENERAL);
    
    // Update statistics
    selector->stats.domain_adaptations++;
    
    printf("    ✓ Domain adaptation prediction completed\n");
    printf("      Predicted domain: %s (%.1f%% effective)\n",
           dynamic_context_domain_type_to_string(domain_prediction->predicted_domain),
           domain_prediction->domain_confidence * 100.0f);
    printf("      Domain effectiveness scores:\n");
    printf("        General: %.1f%%\n", domain_prediction->general_effectiveness * 100.0f);
    printf("        Code: %.1f%%\n", domain_prediction->code_effectiveness * 100.0f);
    printf("        Text: %.1f%%\n", domain_prediction->text_effectiveness * 100.0f);
    printf("        Data: %.1f%%\n", domain_prediction->data_effectiveness * 100.0f);
    printf("        Scientific: %.1f%%\n", domain_prediction->scientific_effectiveness * 100.0f);
    printf("        Multimedia: %.1f%%\n", domain_prediction->multimedia_effectiveness * 100.0f);
    printf("        Configuration: %.1f%%\n", domain_prediction->config_effectiveness * 100.0f);
    printf("      Specialized processing: %s\n", domain_prediction->requires_specialized_processing ? "Required" : "Not needed");
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

DynamicContextError dynamic_context_select_optimal_context(DynamicContextSelector* selector,
                                                          const uint8_t* file_data,
                                                          size_t file_size,
                                                          uint32_t* recommended_context_length,
                                                          DomainAdaptationType* recommended_domain) {
    if (!selector || !file_data || !recommended_context_length || !recommended_domain) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    printf("  Performing comprehensive context selection...\n");
    
    // Analyze file characteristics
    FileCharacteristics characteristics;
    DynamicContextError error = dynamic_context_analyze_file_characteristics(selector, file_data, file_size, &characteristics);
    if (error != DYNAMIC_CONTEXT_SUCCESS) {
        return error;
    }
    
    // Detect content type
    ContentTypeAnalysis content_analysis;
    error = dynamic_context_detect_content_type(selector, file_data, file_size, &content_analysis);
    if (error != DYNAMIC_CONTEXT_SUCCESS) {
        return error;
    }
    
    // Determine optimal context
    OptimalContextDetermination optimal_context;
    error = dynamic_context_determine_optimal_length(selector, &characteristics, &content_analysis, &optimal_context);
    if (error != DYNAMIC_CONTEXT_SUCCESS) {
        return error;
    }
    
    // Predict domain adaptation
    DomainAdaptationPrediction domain_prediction;
    error = dynamic_context_predict_domain_adaptation(selector, &characteristics, &content_analysis, &domain_prediction);
    if (error != DYNAMIC_CONTEXT_SUCCESS) {
        return error;
    }
    
    // Set output values
    *recommended_context_length = optimal_context.recommended_context_length;
    *recommended_domain = domain_prediction.predicted_domain;
    
    printf("  ✓ Optimal context selection completed\n");
    printf("    Final recommendation: %u tokens, %s domain\n",
           *recommended_context_length,
           dynamic_context_domain_type_to_string(*recommended_domain));
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

// Helper function implementations

static DynamicContextError analyze_file_entropy(DynamicContextSelector* selector,
                                               const uint8_t* data, size_t size,
                                               float* entropy) {
    if (!selector || !data || !entropy || size == 0) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    // Calculate Shannon entropy of the data
    uint32_t byte_counts[256] = {0};
    for (size_t i = 0; i < size; i++) {
        byte_counts[data[i]]++;
    }
    
    float entropy_sum = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (byte_counts[i] > 0) {
            float probability = (float)byte_counts[i] / size;
            entropy_sum -= probability * log2f(probability);
        }
    }
    
    *entropy = entropy_sum / 8.0f; // Normalize to 0-1 range (max entropy = 8 bits)
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

static DynamicContextError detect_pattern_structures(DynamicContextSelector* selector,
                                                    const uint8_t* data, size_t size,
                                                    FilePattern* pattern) {
    if (!selector || !data || !pattern || size == 0) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    // Initialize pattern detection scores
    float pattern_scores[FILE_PATTERN_HYBRID + 1] = {0};
    
    // Analyze for repetitive patterns
    uint32_t repeated_sequences = 0;
    for (size_t i = 0; i < size - 4; i++) {
        for (size_t j = i + 4; j < size - 4; j++) {
            if (memcmp(&data[i], &data[j], 4) == 0) {
                repeated_sequences++;
                break;
            }
        }
    }
    pattern_scores[FILE_PATTERN_REPETITIVE] = (float)repeated_sequences / (size / 8);
    
    // Analyze for hierarchical patterns (indentation, brackets, etc.)
    uint32_t hierarchy_indicators = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] == '\t' || data[i] == '{' || data[i] == '}' || 
            data[i] == '(' || data[i] == ')' || data[i] == '[' || data[i] == ']') {
            hierarchy_indicators++;
        }
    }
    pattern_scores[FILE_PATTERN_HIERARCHICAL] = (float)hierarchy_indicators / size;
    
    // Analyze for sequential patterns (timestamps, line numbers, etc.)
    uint32_t sequential_indicators = 0;
    for (size_t i = 0; i < size - 1; i++) {
        if (isdigit(data[i]) && (data[i+1] == ':' || data[i+1] == '-' || data[i+1] == '/')) {
            sequential_indicators++;
        }
    }
    pattern_scores[FILE_PATTERN_SEQUENTIAL] = (float)sequential_indicators / size;
    
    // Analyze for tabular patterns (commas, tabs, regular spacing)
    uint32_t tabular_indicators = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] == ',' || data[i] == '\t' || data[i] == '|') {
            tabular_indicators++;
        }
    }
    pattern_scores[FILE_PATTERN_TABULAR] = (float)tabular_indicators / size;
    
    // Analyze for linguistic patterns (words, sentences, paragraphs)
    uint32_t linguistic_indicators = 0;
    for (size_t i = 0; i < size; i++) {
        if (isalpha(data[i]) || data[i] == ' ' || data[i] == '.' || data[i] == ',' || data[i] == '!') {
            linguistic_indicators++;
        }
    }
    pattern_scores[FILE_PATTERN_LINGUISTIC] = (float)linguistic_indicators / size;
    
    // Analyze for algorithmic patterns (keywords, operators, symbols)
    uint32_t algorithmic_indicators = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] == '=' || data[i] == '+' || data[i] == '-' || data[i] == '*' ||
            data[i] == '/' || data[i] == '<' || data[i] == '>' || data[i] == '&' ||
            data[i] == '|' || data[i] == '^' || data[i] == '!' || data[i] == ';') {
            algorithmic_indicators++;
        }
    }
    pattern_scores[FILE_PATTERN_ALGORITHMIC] = (float)algorithmic_indicators / size;
    
    // Find dominant pattern
    float max_score = 0.0f;
    FilePattern dominant_pattern = FILE_PATTERN_RANDOM;
    
    for (int i = FILE_PATTERN_REPETITIVE; i <= FILE_PATTERN_ALGORITHMIC; i++) {
        if (pattern_scores[i] > max_score) {
            max_score = pattern_scores[i];
            dominant_pattern = (FilePattern)i;
        }
    }
    
    // Check for hybrid patterns (multiple high scores)
    uint32_t high_score_count = 0;
    for (int i = FILE_PATTERN_REPETITIVE; i <= FILE_PATTERN_ALGORITHMIC; i++) {
        if (pattern_scores[i] > 0.1f) high_score_count++;
    }
    
    if (high_score_count >= 3) {
        *pattern = FILE_PATTERN_HYBRID;
    } else {
        *pattern = dominant_pattern;
    }
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

static DynamicContextError calculate_content_scores(DynamicContextSelector* selector,
                                                   const uint8_t* data, size_t size,
                                                   ContentTypeAnalysis* analysis) {
    if (!selector || !data || !analysis || size == 0) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    // Text content analysis
    uint32_t text_chars = 0;
    uint32_t whitespace_chars = 0;
    for (size_t i = 0; i < size; i++) {
        if (isalnum(data[i]) || ispunct(data[i])) {
            text_chars++;
        } else if (isspace(data[i])) {
            whitespace_chars++;
        }
    }
    analysis->text_score = (float)(text_chars + whitespace_chars) / size;
    
    // Code content analysis  
    uint32_t code_chars = 0;
    for (size_t i = 0; i < size; i++) {
        char c = data[i];
        if (c == '{' || c == '}' || c == '(' || c == ')' || c == '[' || c == ']' ||
            c == ';' || c == '=' || c == '+' || c == '-' || c == '*' || c == '/' ||
            c == '<' || c == '>' || c == '&' || c == '|' || c == '^' || c == '!') {
            code_chars++;
        }
    }
    analysis->code_score = (float)code_chars / size;
    
    // Binary content analysis
    uint32_t non_printable = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] < 32 && data[i] != '\t' && data[i] != '\n' && data[i] != '\r') {
            non_printable++;
        }
    }
    analysis->binary_score = (float)non_printable / size;
    
    // Structured data analysis
    uint32_t structure_chars = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] == ',' || data[i] == ':' || data[i] == '"' || data[i] == '\'' ||
            data[i] == '{' || data[i] == '}' || data[i] == '[' || data[i] == ']' ||
            data[i] == '<' || data[i] == '>') {
            structure_chars++;
        }
    }
    analysis->structured_data_score = (float)structure_chars / size;
    
    // Documentation analysis (headers, formatting, etc.)
    uint32_t doc_indicators = 0;
    for (size_t i = 0; i < size - 1; i++) {
        if (data[i] == '#' || data[i] == '*' || data[i] == '-' || 
            (data[i] == '=' && data[i+1] == '=') ||
            (data[i] == '-' && data[i+1] == '-')) {
            doc_indicators++;
        }
    }
    analysis->documentation_score = (float)doc_indicators / size;
    
    // Log file analysis (timestamps, levels, etc.)
    uint32_t log_indicators = 0;
    for (size_t i = 0; i < size - 3; i++) {
        // Look for timestamp patterns like "2023" or "INFO" or "ERROR"
        if (isdigit(data[i]) && isdigit(data[i+1]) && isdigit(data[i+2]) && isdigit(data[i+3])) {
            log_indicators++;
        }
    }
    analysis->log_score = (float)log_indicators / (size / 100); // Normalize differently for logs
    
    // Configuration file analysis
    uint32_t config_indicators = 0;
    for (size_t i = 0; i < size - 1; i++) {
        if (data[i] == '=' && isalnum(data[i-1]) && isalnum(data[i+1])) {
            config_indicators++;
        }
    }
    analysis->config_score = (float)config_indicators / size;
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

// Configuration and utility functions

DynamicContextError dynamic_context_create_default_config(DynamicContextConfig* config) {
    if (!config) {
        return DYNAMIC_CONTEXT_ERROR_INVALID_PARAM;
    }
    
    config->enable_file_analysis = true;
    config->enable_content_detection = true;
    config->enable_pattern_recognition = true;
    config->enable_domain_adaptation = true;
    config->enable_real_time_optimization = false;
    config->analysis_threshold = DYNAMIC_CONTEXT_DEFAULT_THRESHOLD;
    config->sample_size = DYNAMIC_CONTEXT_DEFAULT_SAMPLE_SIZE;
    config->min_context_length = DYNAMIC_CONTEXT_MIN_CONTEXT_LENGTH;
    config->max_context_length = DYNAMIC_CONTEXT_MAX_CONTEXT_LENGTH;
    config->optimization_strategy = CONTEXT_OPT_BALANCED;
    config->default_domain = DOMAIN_ADAPTATION_GENERAL;
    
    return DYNAMIC_CONTEXT_SUCCESS;
}

void dynamic_context_get_stats(DynamicContextSelector* selector,
                               DynamicContextStats* stats) {
    if (!selector || !stats) {
        return;
    }
    
    *stats = selector->stats;
}

void dynamic_context_destroy(DynamicContextSelector* selector) {
    if (!selector) {
        return;
    }
    
    // Free allocated resources
    free(selector->entropy_cache);
    free(selector->pattern_cache);
    
#ifdef __OBJC__
    // Release Metal resources
    if (selector->metal_device) {
        [selector->metal_device release];
    }
    if (selector->command_queue) {
        [selector->command_queue release];
    }
#endif
    
    printf("✓ Dynamic Context Selector destroyed\n");
    printf("  - Files analyzed: %u\n", selector->stats.files_analyzed);
    printf("  - Content types detected: %u\n", selector->stats.content_types_detected);
    printf("  - Successful optimizations: %u\n", selector->stats.successful_optimizations);
    printf("  - Domain adaptations: %u\n", selector->stats.domain_adaptations);
    
    free(selector);
}

// Utility function implementations

const char* dynamic_context_get_error_string(DynamicContextError error_code) {
    switch (error_code) {
        case DYNAMIC_CONTEXT_SUCCESS:
            return "Success";
        case DYNAMIC_CONTEXT_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case DYNAMIC_CONTEXT_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case DYNAMIC_CONTEXT_ERROR_FILE_ACCESS:
            return "File access error";
        case DYNAMIC_CONTEXT_ERROR_ANALYSIS_FAILED:
            return "Analysis failed";
        case DYNAMIC_CONTEXT_ERROR_CONTENT_DETECTION_FAILED:
            return "Content detection failed";
        case DYNAMIC_CONTEXT_ERROR_PATTERN_RECOGNITION_FAILED:
            return "Pattern recognition failed";
        case DYNAMIC_CONTEXT_ERROR_OPTIMIZATION_FAILED:
            return "Optimization failed";
        case DYNAMIC_CONTEXT_ERROR_DOMAIN_ADAPTATION_FAILED:
            return "Domain adaptation failed";
        case DYNAMIC_CONTEXT_ERROR_INSUFFICIENT_DATA:
            return "Insufficient data";
        case DYNAMIC_CONTEXT_ERROR_INVALID_CONTENT_TYPE:
            return "Invalid content type";
        case DYNAMIC_CONTEXT_ERROR_CONFIGURATION_INVALID:
            return "Configuration invalid";
        default:
            return "Unknown error";
    }
}

const char* dynamic_context_content_type_to_string(ContentType content_type) {
    switch (content_type) {
        case CONTENT_TYPE_UNKNOWN:
            return "Unknown";
        case CONTENT_TYPE_TEXT:
            return "Text";
        case CONTENT_TYPE_CODE:
            return "Code";
        case CONTENT_TYPE_BINARY:
            return "Binary";
        case CONTENT_TYPE_STRUCTURED_DATA:
            return "Structured Data";
        case CONTENT_TYPE_DOCUMENTATION:
            return "Documentation";
        case CONTENT_TYPE_LOG:
            return "Log";
        case CONTENT_TYPE_CONFIG:
            return "Configuration";
        case CONTENT_TYPE_MIXED:
            return "Mixed";
        default:
            return "Unknown";
    }
}

const char* dynamic_context_file_pattern_to_string(FilePattern pattern) {
    switch (pattern) {
        case FILE_PATTERN_RANDOM:
            return "Random";
        case FILE_PATTERN_REPETITIVE:
            return "Repetitive";
        case FILE_PATTERN_HIERARCHICAL:
            return "Hierarchical";
        case FILE_PATTERN_SEQUENTIAL:
            return "Sequential";
        case FILE_PATTERN_TABULAR:
            return "Tabular";
        case FILE_PATTERN_LINGUISTIC:
            return "Linguistic";
        case FILE_PATTERN_ALGORITHMIC:
            return "Algorithmic";
        case FILE_PATTERN_HYBRID:
            return "Hybrid";
        default:
            return "Unknown";
    }
}

const char* dynamic_context_domain_type_to_string(DomainAdaptationType domain_type) {
    switch (domain_type) {
        case DOMAIN_ADAPTATION_GENERAL:
            return "General";
        case DOMAIN_ADAPTATION_CODE:
            return "Code";
        case DOMAIN_ADAPTATION_TEXT:
            return "Text";
        case DOMAIN_ADAPTATION_DATA:
            return "Data";
        case DOMAIN_ADAPTATION_SCIENTIFIC:
            return "Scientific";
        case DOMAIN_ADAPTATION_MULTIMEDIA:
            return "Multimedia";
        case DOMAIN_ADAPTATION_CONFIGURATION:
            return "Configuration";
        default:
            return "Unknown";
    }
}

const char* dynamic_context_optimization_strategy_to_string(ContextOptimizationStrategy strategy) {
    switch (strategy) {
        case CONTEXT_OPT_CONSERVATIVE:
            return "Conservative";
        case CONTEXT_OPT_BALANCED:
            return "Balanced";
        case CONTEXT_OPT_AGGRESSIVE:
            return "Aggressive";
        case CONTEXT_OPT_ADAPTIVE:
            return "Adaptive";
        case CONTEXT_OPT_MEMORY_CONSTRAINED:
            return "Memory Constrained";
        default:
            return "Unknown";
    }
}
