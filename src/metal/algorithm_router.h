#ifndef ALGORITHM_ROUTER_H
#define ALGORITHM_ROUTER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "compression_integration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Data characteristics structure
typedef struct DataCharacteristics {
    uint64_t data_size;
    float entropy;
    float mean;
    float variance;
    int data_type;
    bool is_text;
    bool is_binary;
    float repetition_ratio;
} DataCharacteristics;

// Map between integration layer and Phase 2C algorithm types
typedef enum {
    ROUTING_ALGORITHM_RLE = 0,
    ROUTING_ALGORITHM_TRANSFORMER,
    ROUTING_ALGORITHM_LSTM,
    ROUTING_ALGORITHM_AUTO
} RoutingAlgorithm;

/**
 * @brief Algorithm routing decision structure
 */
typedef struct {
    RoutingAlgorithm primary_algorithm;     ///< Primary algorithm recommendation
    RoutingAlgorithm fallback_algorithm;    ///< Fallback algorithm if primary fails
    float confidence_score;                  ///< Confidence in the decision (0-1)
    DataCharacteristics characteristics;     ///< Analyzed data characteristics
    char reasoning[256];                     ///< Human-readable reasoning for decision
} AlgorithmRoutingDecision;

/**
 * @brief Initialize the algorithm router with Phase 2C compression selector
 * @return true if initialization succeeded, false otherwise
 */
bool algorithm_router_init(void);

/**
 * @brief Shutdown the algorithm router and free resources
 */
void algorithm_router_shutdown(void);

/**
 * @brief Analyze data and make algorithm routing decision
 * @param input_data Input data buffer
 * @param input_size Size of input data
 * @param decision Output routing decision
 * @return true if analysis succeeded, false otherwise
 */
bool algorithm_router_analyze_and_select(
    const uint8_t* input_data,
    size_t input_size,
    AlgorithmRoutingDecision* decision
);

/**
 * @brief Convert Phase 2C algorithm enum to routing algorithm enum
 * @param phase2c_algorithm Phase 2C compression algorithm
 * @return Corresponding routing algorithm
 */
RoutingAlgorithm algorithm_router_map_phase2c_algorithm(CompressionAlgorithm phase2c_algorithm);

/**
 * @brief Get human-readable name for routing algorithm
 * @param algorithm Routing algorithm enum
 * @return String name of the algorithm
 */
const char* algorithm_router_algorithm_name(RoutingAlgorithm algorithm);

/**
 * @brief Apply heuristic overrides for specific data patterns
 * @param characteristics Data characteristics from analysis
 * @param decision Decision to potentially override
 */
void algorithm_router_apply_heuristics(
    const DataCharacteristics* characteristics,
    AlgorithmRoutingDecision* decision
);

/**
 * @brief Check if router is initialized and ready
 * @return true if router is ready, false otherwise
 */
bool algorithm_router_is_ready(void);

#ifdef __cplusplus
}
#endif

#endif // ALGORITHM_ROUTER_H
