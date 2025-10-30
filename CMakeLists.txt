cmake_minimum_required(VERSION 3.20)
project(nncp VERSION 1.0.0 LANGUAGES C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build configuration
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
option(USE_METAL "Enable Metal support on macOS" OFF)
option(BUILD_TESTS "Build test suite" ON)

# Platform detection
if(APPLE)
    enable_language(OBJC)
    
    # Check for Apple Silicon
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(USE_METAL ON CACHE BOOL "Enable Metal support" FORCE)
        message(STATUS "Apple Silicon detected - enabling Metal support")
    else()
        message(STATUS "Intel Mac detected - Metal support optional")
    endif()
endif()

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpointer-arith")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -DDEBUG")
endif()

# Version information
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/version.h"
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/include
)

# Legacy core sources (not used in Metal version)
# set(NNCP_CORE_SOURCES
#     nncp.c
#     arith.c
#     preprocess.c
#     cp_utils.c
#     cmdopt.c
#     cutils.c
# )

# Metal support
if(USE_METAL)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(METALPERFORMANCESHADERSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    
    # Metal shader sources
    set(METAL_SOURCES
        src/metal/compute/matrix_ops.metal
        src/metal/compute/neural_net.metal
        src/metal/compute/preprocessing.metal
    )
    
    # Compile Metal shaders (skip if metal toolchain not available)
    find_program(METAL_COMPILER xcrun)
    execute_process(COMMAND xcrun -sdk macosx metal --version
                   RESULT_VARIABLE METAL_AVAILABLE
                   OUTPUT_QUIET ERROR_QUIET)
    if(METAL_COMPILER AND METAL_AVAILABLE EQUAL 0)
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/default.metallib
            COMMAND xcrun -sdk macosx metal ${METAL_SOURCES} -o ${CMAKE_BINARY_DIR}/default.metallib
            DEPENDS ${METAL_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Compiling Metal shaders"
        )
        
        add_custom_target(metal_shaders ALL DEPENDS ${CMAKE_BINARY_DIR}/default.metallib)
        set(METAL_SHADERS_AVAILABLE TRUE)
    else()
        message(WARNING "Metal compiler not found, skipping shader compilation")
        add_custom_target(metal_shaders)
        set(METAL_SHADERS_AVAILABLE FALSE)
    endif()
    
    # Metal wrapper sources
    set(METAL_WRAPPER_SOURCES
        src/metal/wrapper/metal_context.mm
        src/metal/wrapper/metal_buffers.mm
        src/metal/wrapper/metal_kernels.mm
        src/metal/wrapper/neural_engine.mm
        src/metal/wrapper/memory_manager.mm
        src/metal/wrapper/compute_kernels.mm
        # src/metal/wrapper/benchmark.mm
    )
    
    # Neural LSTM sources
    set(NEURAL_LSTM_SOURCES
        src/neural/ops/nncp_lstm_metal.mm
        src/neural/ops/nncp_lstm_prediction.mm
        src/neural/ops/nncp_lstm_state_manager.mm
        src/neural/lstm/nncp_lstm_metal_enhanced.mm
    )
    
    # LibNC Metal sources
    set(LIBNC_METAL_SOURCES
        src/libnc/metal/nc_metal.c
        src/libnc/metal/nc_metal_device.c
    )
    
    list(APPEND NNCP_CORE_SOURCES ${METAL_WRAPPER_SOURCES} ${LIBNC_METAL_SOURCES} ${NEURAL_LSTM_SOURCES}
        src/neural/config/cuda_profiles.c
        src/neural/compatibility/cuda_math_compat.c
        src/neural/validation/cuda_parameter_validator.c
        src/neural/handlers/small_file_handler.c
        src/neural/integration/neural_bridge.mm
        src/neural/integration/neural_bridge_lossless_cuda.mm
        src/neural/integration/neural_bridge_enhanced.c
        src/neural/quality/quality_validator.c
        src/neural/prediction/prediction_scorer.c
        src/neural/prediction/entropy_analyzer.c
        src/neural/prediction/pattern_matcher.c
        src/neural/context/dynamic_context.c
        src/neural/selection/enhanced_selector.c
        src/neural/selection/content_analyzer.c
    )
endif()



# Metal-specific NNCP executable
if(USE_METAL)
    set(NNCP_METAL_SOURCES
        src/metal/nncp_metal.c
        src/metal/compression_integration.c
        src/metal/algorithm_router.c
        src/metal/compression_selector_stub.c
        src/metal/nncp_original_port.c
        src/neural/config/cuda_profiles.c
        src/neural/compatibility/cuda_math_compat.c
        src/neural/validation/cuda_parameter_validator.c
        src/neural/handlers/small_file_handler.c
        src/neural/integration/neural_bridge.mm
        src/neural/integration/neural_bridge_lossless_cuda.mm
        src/neural/integration/neural_bridge_enhanced.c
        src/neural/quality/quality_validator.c
        src/neural/prediction/prediction_scorer.c
        src/neural/prediction/entropy_analyzer.c
        src/neural/prediction/pattern_matcher.c
        src/neural/context/dynamic_context.c
        src/neural/selection/enhanced_selector.c
        src/neural/selection/content_analyzer.c
        ${METAL_WRAPPER_SOURCES}
        ${NEURAL_LSTM_SOURCES}
    )
    
    add_executable(nncp-metal ${NNCP_METAL_SOURCES})
    
    target_link_libraries(nncp-metal 
        m pthread
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        ${METALPERFORMANCESHADERSGRAPH_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
    )
    
    target_compile_definitions(nncp-metal PRIVATE USE_METAL=1)
    add_dependencies(nncp-metal metal_shaders)
    
    # NNCP Diagnostics tool (simplified to avoid type conflicts)
    add_executable(nncp-diagnostics
        src/metal/nncp_diagnostics.c
        src/metal/compression_integration.c
        src/metal/algorithm_router.c
        src/neural/integration/neural_bridge.mm
        src/neural/integration/neural_bridge_lossless_cuda.mm
        src/neural/quality/quality_validator.c
        src/neural/prediction/prediction_scorer.c
        src/neural/prediction/entropy_analyzer.c
        src/neural/prediction/pattern_matcher.c
        src/neural/context/dynamic_context.c
        src/neural/selection/enhanced_selector.c
        src/neural/selection/content_analyzer.c
        src/neural/config/cuda_profiles.c
        src/neural/compatibility/cuda_math_compat.c
        src/neural/error/cuda_error_handler.c
        ${NEURAL_LSTM_SOURCES}
    )
    
    target_link_libraries(nncp-diagnostics
        m pthread
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        ${METALPERFORMANCESHADERSGRAPH_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
    )
    
    target_compile_definitions(nncp-diagnostics PRIVATE USE_METAL=1)
    

    
    # Comprehensive Integration Test Suite
    add_executable(test-compression-integration
        tests/test_compression_integration.c
        src/metal/compression_integration.c
        src/metal/algorithm_router.c
        src/metal/compression_selector_stub.c
        src/metal/nncp_original_port.c
        src/neural/integration/neural_bridge.mm
        src/neural/integration/neural_bridge_lossless_cuda.mm
        src/neural/quality/quality_validator.c
        src/neural/prediction/prediction_scorer.c
        src/neural/prediction/entropy_analyzer.c
        src/neural/prediction/pattern_matcher.c
        src/neural/context/dynamic_context.c
        src/neural/selection/enhanced_selector.c
        src/neural/selection/content_analyzer.c
        src/neural/config/cuda_profiles.c
        src/neural/compatibility/cuda_math_compat.c
        src/neural/error/cuda_error_handler.c
        ${METAL_WRAPPER_SOURCES}
        ${NEURAL_LSTM_SOURCES}
    )
    
    target_link_libraries(test-compression-integration
        m pthread
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        ${METALPERFORMANCESHADERSGRAPH_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
    )
    
    target_compile_definitions(test-compression-integration PRIVATE USE_METAL=1)
    

    
    # CUDA Compatibility Tests (simplified version)
    add_executable(cuda-compatibility-tests
        tests/cuda_compatibility_tests_simple.c
        src/neural/config/cuda_profiles.c
        src/neural/compatibility/cuda_math_compat.c
        src/neural/error/cuda_error_handler.c
    )
    
    target_link_libraries(cuda-compatibility-tests
        m pthread
        ${METAL_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    
    target_compile_definitions(cuda-compatibility-tests PRIVATE USE_METAL=1)



    # Install Metal NNCP and diagnostics
    install(TARGETS nncp-metal nncp-diagnostics test-compression-integration DESTINATION bin)
    

endif()



# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "NNCP Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Metal support: ${USE_METAL}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
