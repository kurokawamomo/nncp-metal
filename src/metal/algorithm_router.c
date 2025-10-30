#include "algorithm_router.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// DataCharacteristics structure is defined in header file

// Global state
static bool g_router_initialized = false;

// Algorithm name mapping
static const char* g_algorithm_names[] = {
    [ROUTING_ALGORITHM_RLE] = "RLE",
    [ROUTING_ALGORITHM_TRANSFORMER] = "Transformer",
    [ROUTING_ALGORITHM_LSTM] = "LSTM",
    [ROUTING_ALGORITHM_AUTO] = "Auto"
};

bool algorithm_router_init(void) {
    if (g_router_initialized) {
        return true;
    }
    
    g_router_initialized = true;
    return true;
}

void algorithm_router_shutdown(void) {
    if (!g_router_initialized) {
        return;
    }
    
    g_router_initialized = false;
}

bool algorithm_router_analyze_and_select(
    const uint8_t* input_data,
    size_t input_size,
    AlgorithmRoutingDecision* decision
) {
    if (!decision || !input_data || input_size == 0) {
        return false;
    }
    
    if (!g_router_initialized) {
        return false;
    }
    
    // Initialize decision structure
    memset(decision, 0, sizeof(AlgorithmRoutingDecision));
    
    // Simple data analysis
    DataCharacteristics chars = {0};
    chars.data_size = input_size;
    chars.is_text = true;
    chars.is_binary = false;
    
    // Simple text detection
    for (size_t i = 0; i < input_size && i < 100; i++) {
        uint8_t byte = input_data[i];
        if (byte < 32 && byte != '\n' && byte != '\r' && byte != '\t') {
            if (byte != 0) {
                chars.is_text = false;
                chars.is_binary = true;
                break;
            }
        }
    }
    
    decision->characteristics = chars;
    
    // Algorithm selection logic
    if (chars.is_text) {
        decision->primary_algorithm = ROUTING_ALGORITHM_TRANSFORMER;
        decision->fallback_algorithm = ROUTING_ALGORITHM_LSTM;
        decision->confidence_score = 0.8f;
        strncpy(decision->reasoning, "Text data detected - using Transformer", 
                sizeof(decision->reasoning) - 1);
    } else {
        decision->primary_algorithm = ROUTING_ALGORITHM_LSTM;
        decision->fallback_algorithm = ROUTING_ALGORITHM_RLE;
        decision->confidence_score = 0.7f;
        strncpy(decision->reasoning, "Binary data detected - using LSTM", 
                sizeof(decision->reasoning) - 1);
    }
    
    return true;
}

bool algorithm_router_is_ready(void) {
    return g_router_initialized;
}

const char* algorithm_router_algorithm_name(RoutingAlgorithm algorithm) {
    if (algorithm >= 0 && algorithm < (sizeof(g_algorithm_names) / sizeof(g_algorithm_names[0]))) {
        return g_algorithm_names[algorithm];
    }
    return "Unknown";
}

RoutingAlgorithm algorithm_router_map_phase2c_algorithm(CompressionAlgorithm phase2c_algorithm) {
    switch (phase2c_algorithm) {
        case COMPRESSION_ALGORITHM_TRANSFORMER:
            return ROUTING_ALGORITHM_TRANSFORMER;
        case COMPRESSION_ALGORITHM_LSTM:
            return ROUTING_ALGORITHM_LSTM;
        case COMPRESSION_ALGORITHM_AUTO:
            return ROUTING_ALGORITHM_AUTO;
        default:
            return ROUTING_ALGORITHM_RLE;
    }
}

void algorithm_router_apply_heuristics(
    const DataCharacteristics* characteristics,
    AlgorithmRoutingDecision* decision
) {
    // Simple heuristics implementation
    if (!characteristics || !decision) {
        return;
    }
    
    // If data is very small, prefer RLE for speed
    if (characteristics->data_size < 1024) {
        decision->primary_algorithm = ROUTING_ALGORITHM_RLE;
        decision->confidence_score = 0.9f;
        strncpy(decision->reasoning, "Small data - RLE for speed", 
                sizeof(decision->reasoning) - 1);
    }
}
