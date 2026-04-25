cmake_minimum_required(VERSION 3.20)
project(nncp VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
option(USE_METAL "Enable Metal support on macOS" OFF)

if(APPLE)
    enable_language(OBJC)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(USE_METAL ON CACHE BOOL "Enable Metal support" FORCE)
        message(STATUS "Apple Silicon detected - enabling Metal support")
    else()
        message(STATUS "Intel Mac detected - Metal support optional")
    endif()
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpointer-arith")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -DDEBUG")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/src/version.h"
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neural
)

if(USE_METAL)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(METALPERFORMANCESHADERSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

    # Metal shader sources
    set(METAL_SOURCES
        src/metal/compute/matrix_ops.metal
        src/metal/compute/neural_net.metal
        src/metal/compute/preprocessing.metal
        src/metal/compute/transformer_linear_amx_bf16.metal
    )

    find_program(METAL_COMPILER xcrun)
    execute_process(COMMAND xcrun -sdk macosx metal --version
                   RESULT_VARIABLE METAL_AVAILABLE
                   OUTPUT_QUIET ERROR_QUIET)
    if(METAL_COMPILER AND METAL_AVAILABLE EQUAL 0)
        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/default.metallib
            COMMAND xcrun -sdk macosx metal -std=metal3.1 ${METAL_SOURCES} -o ${CMAKE_BINARY_DIR}/default.metallib
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

    # Metal wrapper sources (src/metal/wrapper/ + flattened src/neural/)
    set(METAL_WRAPPER_SOURCES
        src/metal/wrapper/metal_context.mm
        src/metal/wrapper/metal_buffers.mm
        src/metal/wrapper/metal_kernels.mm
        src/metal/wrapper/neural_engine.mm
        src/metal/wrapper/memory_manager.mm
        src/metal/wrapper/compute_kernels.mm
        src/neural/unified_memory_manager.mm
        src/neural/memory_pool_optimizer.mm
        src/neural/sync_optimizer.mm
        src/neural/gpu_native_transformer.mm
        src/neural/gpu_native_lstm.mm
        src/neural/layer_flow_optimizer.mm
        src/neural/mps_transformer_graph.mm
        src/neural/mps_lstm_graph.mm
        src/neural/mps_matrix_optimizer.mm
        src/neural/async_execution_manager.mm
        src/neural/online_trainer.mm
    )

    # Neural LSTM sources (flattened src/neural/)
    set(NEURAL_LSTM_SOURCES
        src/neural/nncp_lstm_metal.mm
        src/neural/nncp_lstm_prediction.mm
        src/neural/nncp_lstm_state_manager.mm
        src/neural/nncp_lstm_metal_enhanced.mm
    )

    set(NNCP_METAL_SOURCES
        src/metal/nncp_metal.c
        src/metal/compression_integration.c
        src/neural/neural_weights.c
        src/neural/cuda_profiles.c
        src/neural/cuda_math_compat.c
        src/neural/cuda_parameter_validator.c
        src/neural/small_file_handler.c
        src/neural/neural_bridge.mm
        src/neural/neural_bridge_lossless_cuda.mm
        src/neural/neural_bridge_enhanced.c
        src/neural/quality_validator.c
        src/neural/prediction_scorer.c
        src/neural/entropy_analyzer.c
        src/neural/pattern_matcher.c
        src/neural/dynamic_context.c
        src/neural/enhanced_selector.c
        src/neural/content_analyzer.c
        src/neural/preprocess.c
        ${METAL_WRAPPER_SOURCES}
        ${NEURAL_LSTM_SOURCES}
    )

    add_executable(nncp ${NNCP_METAL_SOURCES})

    target_link_libraries(nncp
        m pthread
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        ${METALPERFORMANCESHADERSGRAPH_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
    )

    target_compile_definitions(nncp PRIVATE USE_METAL=1)
    # Phase M-2 Part 1: Metal backward path (opt-in). When ON, wires Metal
    # kernels for loss/layer backward via metal_bw_* helpers in online_trainer.mm.
    option(NNCP_METAL_BW "Enable Metal backward compute path (Phase M-2)" OFF)
    if(NNCP_METAL_BW)
        target_compile_definitions(nncp PRIVATE NNCP_METAL_BW=1)
        message(STATUS "  Metal-BW (Phase M-2): enabled")
    endif()
    add_dependencies(nncp metal_shaders)

    add_custom_command(TARGET nncp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/default.metallib
            ${CMAKE_SOURCE_DIR}/default.metallib
        COMMENT "Copying default.metallib to project root"
    )

    install(TARGETS nncp DESTINATION bin)

    # Phase M-1 backward kernel verification harness
    add_executable(bw_verify src/metal/bw_verify.mm)
    target_link_libraries(bw_verify
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    add_dependencies(bw_verify metal_shaders)
    add_custom_command(TARGET bw_verify POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/default.metallib
            $<TARGET_FILE_DIR:bw_verify>/default.metallib
    )

    # Phase M-2 Part 1 loss-backward verification
    add_executable(m2_verify src/metal/m2_verify.mm)
    target_link_libraries(m2_verify
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    add_dependencies(m2_verify metal_shaders)
    add_custom_command(TARGET m2_verify POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/default.metallib
            $<TARGET_FILE_DIR:m2_verify>/default.metallib
    )
endif()

message(STATUS "")
message(STATUS "NNCP Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Metal support: ${USE_METAL}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
