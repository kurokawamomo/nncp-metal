#!/bin/bash
# Simple build and test script for Neural Bridge LSTM test

set -e  # Exit on any error

echo "Neural Bridge LSTM Test Builder"
echo "==============================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This test requires macOS with Metal support"
    exit 1
fi

# Check if clang is available
if ! command -v clang &> /dev/null; then
    echo "❌ clang compiler not found"
    exit 1
fi

# Compiler settings
CC=clang
CFLAGS="-std=c11 -Wall -Wextra -O2 -g"
FRAMEWORKS="-framework Metal -framework MetalKit -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation -framework CoreML"
INCLUDES="-I. -Isrc/metal/include"

# Source files
TEST_SOURCES="test_neural_bridge_lstm.c src/neural/integration/neural_bridge.c src/neural/quality/quality_validator.c"
TEST_TARGET="test_neural_bridge_lstm"

echo "Building Neural Bridge LSTM Test..."
echo "Compiler: $CC"
echo "Flags: $CFLAGS"
echo "Sources: $TEST_SOURCES"

# Build the test
$CC $CFLAGS $INCLUDES -DUSE_METAL=1 $TEST_SOURCES $FRAMEWORKS -lm -lpthread -o $TEST_TARGET

if [ $? -eq 0 ]; then
    echo "✅ Build successful: $TEST_TARGET"
else
    echo "❌ Build failed"
    exit 1
fi

# Check if user wants to run the test immediately
if [ "$1" = "run" ] || [ "$1" = "test" ]; then
    echo ""
    echo "Running Neural Bridge LSTM Test..."
    echo "=================================="
    ./$TEST_TARGET
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 All tests completed successfully!"
    else
        echo ""
        echo "❌ Some tests failed."
        exit 1
    fi
elif [ "$1" = "quick" ]; then
    echo ""
    echo "Running quick test (filtered output)..."
    echo "======================================="
    ./$TEST_TARGET | grep -E "(PASSED|FAILED|corruption|accuracy|✅|❌|🎉)"
else
    echo ""
    echo "✅ Build complete. To run the test:"
    echo "   ./$TEST_TARGET"
    echo ""
    echo "Or use:"
    echo "   ./build_and_test.sh run    # Build and run full test"
    echo "   ./build_and_test.sh quick  # Build and run with filtered output"
    echo ""
    echo "To clean up:"
    echo "   rm -f $TEST_TARGET test_input.txt test_output.txt"
fi