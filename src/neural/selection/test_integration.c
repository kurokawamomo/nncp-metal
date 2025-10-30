#include "content_analyzer.h"
#include "enhanced_selector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("=== Content Analyzer Integration Test ===\n");
    
    // Create content analyzer
    ContentAnalyzerConfig content_config = content_analyzer_config_default();
    ContentAnalyzer *content_analyzer = content_analyzer_create(&content_config);
    
    // Create enhanced selector
    EnhancedSelectorConfig selector_config = enhanced_selector_config_default();
    EnhancedSelector *enhanced_selector = enhanced_selector_create(&selector_config);
    
    if (!content_analyzer || !enhanced_selector) {
        printf("FAILED: Could not create analyzers\n");
        return 1;
    }
    
    // Test with different data types
    const char *text_data = "This is a test document with natural language content for analysis.";
    const char *json_data = "{\"name\":\"test\",\"data\":[1,2,3],\"nested\":{\"key\":\"value\"}}";
    const uint8_t binary_data[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
    
    ContentCharacteristics text_chars, json_chars, binary_chars;
    ContentAnalysis text_analysis, json_analysis, binary_analysis;
    
    // Analyze with content analyzer
    printf("\n--- Content Analyzer Results ---\n");
    
    content_analyzer_analyze_content(content_analyzer, (const uint8_t*)text_data, strlen(text_data), &text_chars);
    printf("Text content: type=%d, entropy=%.2f, text_prob=%.2f\n", 
           text_chars.primary_type, text_chars.entropy, text_chars.text_probability);
    
    content_analyzer_analyze_content(content_analyzer, (const uint8_t*)json_data, strlen(json_data), &json_chars);
    printf("JSON content: type=%d, entropy=%.2f, has_json=%s\n", 
           json_chars.primary_type, json_chars.entropy, json_chars.has_json_markers ? "yes" : "no");
    
    content_analyzer_analyze_content(content_analyzer, binary_data, sizeof(binary_data), &binary_chars);
    printf("Binary content: type=%d, entropy=%.2f, text_prob=%.2f\n", 
           binary_chars.primary_type, binary_chars.entropy, binary_chars.text_probability);
    
    // Analyze with enhanced selector
    printf("\n--- Enhanced Selector Results ---\n");
    
    enhanced_selector_analyze_content(enhanced_selector, (const uint8_t*)text_data, strlen(text_data), &text_analysis);
    printf("Text analysis: entropy=%.2f, has_text=%s, transformer_ratio=%.2f, lstm_ratio=%.2f\n",
           text_analysis.average_entropy, text_analysis.has_text_patterns ? "yes" : "no",
           text_analysis.transformer_predicted_ratio, text_analysis.lstm_predicted_ratio);
    
    enhanced_selector_analyze_content(enhanced_selector, (const uint8_t*)json_data, strlen(json_data), &json_analysis);
    printf("JSON analysis: entropy=%.2f, has_structured=%s, transformer_ratio=%.2f, lstm_ratio=%.2f\n",
           json_analysis.average_entropy, json_analysis.has_structured_data ? "yes" : "no",
           json_analysis.transformer_predicted_ratio, json_analysis.lstm_predicted_ratio);
    
    enhanced_selector_analyze_content(enhanced_selector, binary_data, sizeof(binary_data), &binary_analysis);
    printf("Binary analysis: entropy=%.2f, has_binary=%s, transformer_ratio=%.2f, lstm_ratio=%.2f\n",
           binary_analysis.average_entropy, binary_analysis.has_binary_data ? "yes" : "no",
           binary_analysis.transformer_predicted_ratio, binary_analysis.lstm_predicted_ratio);
    
    // Test algorithm recommendations
    printf("\n--- Algorithm Recommendations ---\n");
    
    AlgorithmRecommendation text_rec = enhanced_selector_recommend_algorithm(enhanced_selector, &text_analysis);
    AlgorithmRecommendation json_rec = enhanced_selector_recommend_algorithm(enhanced_selector, &json_analysis);
    AlgorithmRecommendation binary_rec = enhanced_selector_recommend_algorithm(enhanced_selector, &binary_analysis);
    
    printf("Text recommendation: %d\n", text_rec);
    printf("JSON recommendation: %d\n", json_rec);
    printf("Binary recommendation: %d\n", binary_rec);
    
    // Compare suitability scores
    printf("\n--- Algorithm Suitability Comparison ---\n");
    
    float text_transformer_suit = content_analyzer_transformer_suitability(&text_chars);
    float text_lstm_suit = content_analyzer_lstm_suitability(&text_chars);
    printf("Text: Transformer=%.2f, LSTM=%.2f\n", text_transformer_suit, text_lstm_suit);
    
    float json_transformer_suit = content_analyzer_transformer_suitability(&json_chars);
    float json_lstm_suit = content_analyzer_lstm_suitability(&json_chars);
    printf("JSON: Transformer=%.2f, LSTM=%.2f\n", json_transformer_suit, json_lstm_suit);
    
    float binary_transformer_suit = content_analyzer_transformer_suitability(&binary_chars);
    float binary_lstm_suit = content_analyzer_lstm_suitability(&binary_chars);
    printf("Binary: Transformer=%.2f, LSTM=%.2f\n", binary_transformer_suit, binary_lstm_suit);
    
    // Cleanup
    content_analyzer_destroy(content_analyzer);
    enhanced_selector_destroy(enhanced_selector);
    
    printf("\n=== Integration Test PASSED ===\n");
    return 0;
}
