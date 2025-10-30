# Makefile for Neural Bridge LSTM Test
# Simple compilation without CMake for quick testing

# Compiler and flags
CC = clang
CFLAGS = -std=c11 -Wall -Wextra -O2 -g
OBJC_FLAGS = -std=c11 -Wall -Wextra -O2 -g

# macOS Metal frameworks
FRAMEWORKS = -framework Metal -framework MetalKit -framework MetalPerformanceShaders \
             -framework MetalPerformanceShadersGraph -framework Foundation -framework CoreML

# Include directories
INCLUDES = -I. -Isrc/metal/include

# Source files for the test
TEST_SOURCES = tests/neural/test_neural_bridge_lstm.c \
               src/neural/integration/neural_bridge.c \
               src/neural/quality/quality_validator.c

# Target executable
TEST_TARGET = test_neural_bridge_lstm

# Default target
all: $(TEST_TARGET)

# Build the test executable
$(TEST_TARGET): $(TEST_SOURCES)
	@echo "Building Neural Bridge LSTM Test..."
	$(CC) $(CFLAGS) $(INCLUDES) -DUSE_METAL=1 $(TEST_SOURCES) $(FRAMEWORKS) -lm -lpthread -o $(TEST_TARGET)
	@echo "✓ Build complete: $(TEST_TARGET)"

# Run the test
test: $(TEST_TARGET)
	@echo "Running Neural Bridge LSTM Test..."
	./$(TEST_TARGET)

# Clean build artifacts
clean:
	rm -f $(TEST_TARGET) test_input.txt test_output.txt

# Create a simple test data file
create-testdata:
	@echo "Creating test data file..."
	@echo "This is a test file with various characters that were previously affected by character corruption." > test_data.txt
	@echo "The medium dog jumped over the fence." >> test_data.txt
	@echo "Programming with metal frameworks requires careful memory management." >> test_data.txt
	@echo "abcdefghijklmnopqrstuvwxyz" >> test_data.txt
	@echo "ABCDEFGHIJKLMNOPQRSTUVWXYZ" >> test_data.txt
	@echo "0123456789!@#$%^&*()" >> test_data.txt
	@echo "✓ Created test_data.txt"

# Quick test with just the LSTM functions
quick-test: $(TEST_TARGET)
	@echo "Running quick test with character corruption verification..."
	./$(TEST_TARGET) | grep -E "(PASSED|FAILED|corruption|accuracy)"

# Help target
help:
	@echo "Neural Bridge LSTM Test Makefile"
	@echo "Available targets:"
	@echo "  all           - Build the test executable"
	@echo "  test          - Build and run the test"
	@echo "  quick-test    - Run test with filtered output"
	@echo "  create-testdata - Create a test data file"
	@echo "  clean         - Remove build artifacts"
	@echo "  help          - Show this help message"

.PHONY: all test quick-test clean create-testdata help