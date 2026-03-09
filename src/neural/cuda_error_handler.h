/*
 * CUDA-Compatible Error Handling System
 * 
 * Implements error handling system that matches original CUDA behavior
 * and error codes to ensure compatibility and consistent error reporting.
 * 
 * Based on original CUDA NNCP implementation error patterns.
 */

#ifndef CUDA_ERROR_HANDLER_H
#define CUDA_ERROR_HANDLER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA-compatible error codes (matching original CUDA implementation)
typedef enum {
    // Success codes
    CUDA_ERROR_SUCCESS = 0,                     // Operation completed successfully
    
    // Memory errors (1-99)
    CUDA_ERROR_OUT_OF_MEMORY = 1,              // Out of memory
    CUDA_ERROR_MEMORY_ALLOCATION_FAILED = 2,   // Memory allocation failed
    CUDA_ERROR_INVALID_MEMORY_ACCESS = 3,      // Invalid memory access
    CUDA_ERROR_MEMORY_COPY_FAILED = 4,         // Memory copy operation failed
    CUDA_ERROR_HOST_MEMORY_ALLOC_FAILED = 5,   // Host memory allocation failed
    CUDA_ERROR_DEVICE_MEMORY_ALLOC_FAILED = 6, // Device memory allocation failed
    
    // Parameter validation errors (100-199)
    CUDA_ERROR_INVALID_VALUE = 100,            // Invalid parameter value
    CUDA_ERROR_INVALID_HANDLE = 101,           // Invalid handle or context
    CUDA_ERROR_INVALID_CONFIGURATION = 102,    // Invalid configuration
    CUDA_ERROR_INVALID_DIMENSION = 103,        // Invalid dimension
    CUDA_ERROR_INVALID_MEMORY_MAPPING = 104,   // Invalid memory mapping
    CUDA_ERROR_INVALID_FILTER_SETTING = 105,   // Invalid filter setting
    CUDA_ERROR_INVALID_NORM_SETTING = 106,     // Invalid normalization setting
    CUDA_ERROR_INVALID_TEXTURE = 107,          // Invalid texture reference
    CUDA_ERROR_ARRAY_IS_MAPPED = 108,          // Array is mapped
    CUDA_ERROR_ALREADY_MAPPED = 109,           // Resource already mapped
    CUDA_ERROR_NO_BINARY_FOR_GPU = 110,        // No binary for GPU
    CUDA_ERROR_ALREADY_ACQUIRED = 111,         // Resource already acquired
    
    // Execution errors (200-299)
    CUDA_ERROR_INVALID_CONTEXT = 200,          // Invalid context
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 201,  // Context already current
    CUDA_ERROR_MAP_FAILED = 202,               // Map operation failed
    CUDA_ERROR_UNMAP_FAILED = 203,             // Unmap operation failed
    CUDA_ERROR_NO_ARRAY_AVAILABLE = 204,       // No array available
    CUDA_ERROR_INVALID_SYMBOL = 205,           // Invalid symbol
    CUDA_ERROR_INVALID_HOST_POINTER = 206,     // Invalid host pointer
    CUDA_ERROR_INVALID_DEVICE_POINTER = 207,   // Invalid device pointer
    CUDA_ERROR_INVALID_INSTRUCTION = 208,      // Invalid instruction
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 209,    // Invalid address space
    CUDA_ERROR_INVALID_PC = 210,               // Invalid program counter
    CUDA_ERROR_LAUNCH_FAILED = 211,            // Kernel launch failed
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 212,  // Launch exceeded resources
    CUDA_ERROR_LAUNCH_TIMEOUT = 213,           // Launch timeout
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 214, // Incompatible texturing mode
    
    // Hardware/driver errors (300-399)
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 300,   // Peer access already enabled
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 301,       // Peer access not enabled
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 302,        // Primary context active
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 303,          // Context is destroyed
    CUDA_ERROR_ASSERT = 304,                        // Device assert triggered
    CUDA_ERROR_TOO_MANY_PEERS = 305,               // Too many peer devices
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 306, // Host memory already registered
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 307,    // Host memory not registered
    CUDA_ERROR_HARDWARE_STACK_ERROR = 308,          // Hardware stack error
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 309,           // Illegal instruction
    CUDA_ERROR_MISALIGNED_ADDRESS = 310,           // Misaligned address
    CUDA_ERROR_INVALID_ADDRESS_SPACE_311 = 311,     // Invalid address space (alt)
    CUDA_ERROR_INVALID_PC_312 = 312,               // Invalid PC (alt)
    CUDA_ERROR_ILLEGAL_ADDRESS = 313,              // Illegal address
    CUDA_ERROR_INVALID_PTX = 314,                  // Invalid PTX
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 315,     // Invalid graphics context
    CUDA_ERROR_NV_LINK_UNCORRECTABLE = 316,        // NVLink uncorrectable error
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 317,       // JIT compiler not found
    
    // File/stream errors (400-499)
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 400,      // Unsupported PTX version
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 401,     // JIT compilation disabled
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 402,    // Unsupported execution affinity
    CUDA_ERROR_FILE_NOT_FOUND = 403,               // File not found
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 404, // Symbol not found
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 405,    // Shared object init failed
    CUDA_ERROR_OPERATING_SYSTEM = 406,             // Operating system error
    CUDA_ERROR_INVALID_RESOURCE_HANDLE = 407,      // Invalid resource handle
    CUDA_ERROR_RESOURCE_REQUIRED = 408,            // Resource required
    CUDA_ERROR_ECC_UNCORRECTABLE = 409,            // ECC uncorrectable error
    
    // Compression-specific errors (500-599)
    CUDA_ERROR_COMPRESSION_FAILED = 500,           // Compression operation failed
    CUDA_ERROR_DECOMPRESSION_FAILED = 501,         // Decompression operation failed
    CUDA_ERROR_INVALID_COMPRESSED_DATA = 502,      // Invalid compressed data format
    CUDA_ERROR_COMPRESSION_BUFFER_TOO_SMALL = 503, // Compression buffer too small
    CUDA_ERROR_UNSUPPORTED_COMPRESSION_FORMAT = 504, // Unsupported compression format
    CUDA_ERROR_COMPRESSION_QUALITY_TOO_LOW = 505,  // Compression quality below threshold
    CUDA_ERROR_LSTM_PROCESSING_FAILED = 506,       // LSTM processing failed
    CUDA_ERROR_TRANSFORMER_PROCESSING_FAILED = 507, // Transformer processing failed
    CUDA_ERROR_NEURAL_NETWORK_ERROR = 508,         // General neural network error
    CUDA_ERROR_INCOMPATIBLE_PROFILE = 509,         // Incompatible CUDA profile
    CUDA_ERROR_SEGMENT_LENGTH_MISMATCH = 510,      // seg_len != train_len mismatch
    CUDA_ERROR_PRECISION_TOLERANCE_EXCEEDED = 511, // Precision tolerance exceeded
    
    // System-level errors (600-699)
    CUDA_ERROR_NOT_READY = 600,                    // Device not ready
    CUDA_ERROR_NOT_INITIALIZED = 601,              // CUDA not initialized
    CUDA_ERROR_NO_DEVICE = 602,                    // No CUDA device available
    CUDA_ERROR_INSUFFICIENT_DRIVER = 603,          // Insufficient driver version
    CUDA_ERROR_STUB_LIBRARY = 604,                 // Stub library error
    CUDA_ERROR_DEVICE_UNINITIALIZED = 605,         // Device uninitialized
    CUDA_ERROR_NO_KERNEL_IMAGE_FOR_DEVICE = 606,   // No kernel image for device
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 607, // Cooperative launch too large
    CUDA_ERROR_NOT_PERMITTED = 608,                // Operation not permitted
    CUDA_ERROR_NOT_SUPPORTED = 609,                // Operation not supported
    CUDA_ERROR_SYSTEM_NOT_READY = 610,             // System not ready
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 611,       // System driver mismatch
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 612, // Compatibility not supported
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 613,     // Stream capture unmatched
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 614,   // Stream capture invalidated
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 615,         // Stream capture merge error
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 616,      // Stream capture unjoined
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 617,     // Stream capture isolation
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 618,      // Stream capture implicit
    CUDA_ERROR_CAPTURED_EVENT = 619,               // Captured event error
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 620,  // Stream capture wrong thread
    
    // Unknown/undefined errors (700+)
    CUDA_ERROR_UNKNOWN = 999                       // Unknown error
} CUDAErrorCode;

// Error severity levels (CUDA-compatible)
typedef enum {
    CUDA_ERROR_SEVERITY_INFO = 0,          // Informational
    CUDA_ERROR_SEVERITY_WARNING,           // Warning - operation can continue
    CUDA_ERROR_SEVERITY_ERROR,             // Error - operation failed
    CUDA_ERROR_SEVERITY_FATAL              // Fatal - system state compromised
} CUDAErrorSeverity;

// Forward declaration for self-referencing pointer
typedef struct CUDAErrorInfo CUDAErrorInfo;

// Error context information
struct CUDAErrorInfo {
    CUDAErrorCode error_code;              // Primary error code
    CUDAErrorSeverity severity;            // Error severity
    
    // Location information
    const char* file;                      // Source file where error occurred
    int line;                              // Line number where error occurred
    const char* function;                  // Function where error occurred
    
    // Error details
    char message[256];                     // Detailed error message
    char context[128];                     // Additional context information
    
    // CUDA-specific information
    int cuda_device_id;                    // CUDA device ID (-1 if not applicable)
    const char* cuda_operation;           // CUDA operation that failed
    size_t cuda_memory_used;              // Memory usage at time of error
    
    // System information
    uint64_t timestamp;                    // Error timestamp (nanoseconds)
    uint32_t thread_id;                    // Thread ID where error occurred
    uint32_t process_id;                   // Process ID
    
    // Error chain support
    CUDAErrorInfo* previous_error;  // Previous error in chain
};

// Error callback function type
typedef void (*CUDAErrorCallback)(const CUDAErrorInfo* error_info, void* user_data);

// Global error handling configuration
typedef struct {
    bool enable_error_logging;             // Enable automatic error logging
    bool enable_stack_trace;               // Enable stack trace capture
    bool abort_on_fatal_error;             // Abort process on fatal errors
    bool chain_errors;                     // Chain multiple errors together
    
    // Callbacks
    CUDAErrorCallback error_callback;      // User error callback
    void* callback_user_data;              // User data for callback
    
    // Logging configuration
    const char* log_file_path;            // Path to error log file
    bool log_to_console;                  // Log errors to console
    CUDAErrorSeverity min_log_severity;   // Minimum severity to log
    
    // Performance settings
    size_t max_error_chain_length;        // Maximum error chain length
    bool enable_error_caching;            // Cache error strings for performance
} CUDAErrorConfig;

// Global error state
typedef struct {
    CUDAErrorCode last_error;             // Last error that occurred
    CUDAErrorInfo* error_chain;           // Chain of errors
    size_t error_count;                   // Total error count
    uint64_t last_error_timestamp;        // Timestamp of last error
    
    // Thread-local error states (simplified)
    CUDAErrorCode thread_local_errors[64]; // Per-thread last errors
    
    // Statistics
    size_t total_errors[4];               // Count by severity
    size_t memory_errors;                 // Count of memory-related errors
    size_t compression_errors;            // Count of compression-related errors
    size_t cuda_compatibility_errors;    // Count of CUDA compatibility errors
} CUDAErrorState;

// Error handling functions
bool cuda_error_init(const CUDAErrorConfig* config);
void cuda_error_shutdown(void);
bool cuda_error_is_initialized(void);

// Error reporting functions
CUDAErrorCode cuda_error_set(CUDAErrorCode error_code, 
                             const char* file, 
                             int line, 
                             const char* function,
                             const char* format, ...);

CUDAErrorCode cuda_error_set_with_context(CUDAErrorCode error_code,
                                         const char* file,
                                         int line, 
                                         const char* function,
                                         const char* context,
                                         const char* format, ...);

// Error querying functions
CUDAErrorCode cuda_error_get_last(void);
const char* cuda_error_get_string(CUDAErrorCode error_code);
const char* cuda_error_get_description(CUDAErrorCode error_code);
CUDAErrorSeverity cuda_error_get_severity(CUDAErrorCode error_code);

// Error information retrieval
const CUDAErrorInfo* cuda_error_get_info(void);
const CUDAErrorInfo* cuda_error_get_chain(void);
size_t cuda_error_get_chain_length(void);

// Error state management
void cuda_error_clear(void);
void cuda_error_clear_chain(void);
bool cuda_error_has_error(void);
bool cuda_error_has_fatal_error(void);

// Error checking macros (CUDA-style)
#define CUDA_CHECK(call) do { \
    CUDAErrorCode _err = (call); \
    if (_err != CUDA_ERROR_SUCCESS) { \
        cuda_error_set(_err, __FILE__, __LINE__, __FUNCTION__, \
                      "CUDA operation failed: %s", cuda_error_get_string(_err)); \
        return _err; \
    } \
} while(0)

#define CUDA_CHECK_RETURN(call, return_value) do { \
    CUDAErrorCode _err = (call); \
    if (_err != CUDA_ERROR_SUCCESS) { \
        cuda_error_set(_err, __FILE__, __LINE__, __FUNCTION__, \
                      "CUDA operation failed: %s", cuda_error_get_string(_err)); \
        return return_value; \
    } \
} while(0)

#define CUDA_CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        cuda_error_set(CUDA_ERROR_INVALID_VALUE, __FILE__, __LINE__, __FUNCTION__, \
                      "Null pointer: %s", #ptr); \
        return CUDA_ERROR_INVALID_VALUE; \
    } \
} while(0)

#define CUDA_CHECK_NULL_RETURN(ptr, return_value) do { \
    if ((ptr) == NULL) { \
        cuda_error_set(CUDA_ERROR_INVALID_VALUE, __FILE__, __LINE__, __FUNCTION__, \
                      "Null pointer: %s", #ptr); \
        return return_value; \
    } \
} while(0)

// Memory error helpers
#define CUDA_CHECK_MEMORY_ALLOC(ptr, size) do { \
    if ((ptr) == NULL) { \
        cuda_error_set(CUDA_ERROR_OUT_OF_MEMORY, __FILE__, __LINE__, __FUNCTION__, \
                      "Memory allocation failed: %zu bytes", (size_t)(size)); \
        return CUDA_ERROR_OUT_OF_MEMORY; \
    } \
} while(0)

// Compression-specific error helpers
#define CUDA_CHECK_COMPRESSION_PROFILE(profile) do { \
    if ((profile) == NULL) { \
        cuda_error_set(CUDA_ERROR_INCOMPATIBLE_PROFILE, __FILE__, __LINE__, __FUNCTION__, \
                      "Invalid CUDA profile"); \
        return CUDA_ERROR_INCOMPATIBLE_PROFILE; \
    } \
} while(0)

#define CUDA_CHECK_SEGMENT_LENGTH(train_len, seg_len) do { \
    if ((train_len) != (seg_len)) { \
        cuda_error_set(CUDA_ERROR_SEGMENT_LENGTH_MISMATCH, __FILE__, __LINE__, __FUNCTION__, \
                      "train_len (%d) != seg_len (%d)", (int)(train_len), (int)(seg_len)); \
        return CUDA_ERROR_SEGMENT_LENGTH_MISMATCH; \
    } \
} while(0)

// Statistics and diagnostics
void cuda_error_print_statistics(void);
void cuda_error_print_chain(void);
bool cuda_error_export_log(const char* filename);

// Configuration helpers
CUDAErrorConfig* cuda_error_config_create_default(void);
CUDAErrorConfig* cuda_error_config_create_production(void);
CUDAErrorConfig* cuda_error_config_create_debug(void);
void cuda_error_config_free(CUDAErrorConfig* config);

// Thread-local error handling (simplified)
CUDAErrorCode cuda_error_get_thread_local(void);
void cuda_error_set_thread_local(CUDAErrorCode error_code);

// Error recovery and retry mechanisms
typedef bool (*CUDAErrorRecoveryFunc)(CUDAErrorCode error_code, int retry_count, void* user_data);

bool cuda_error_set_recovery_callback(CUDAErrorRecoveryFunc callback, void* user_data);
bool cuda_error_retry_operation(CUDAErrorRecoveryFunc operation, 
                               void* operation_data,
                               int max_retries,
                               CUDAErrorCode* final_error);

// Compatibility with legacy error handling
bool cuda_error_is_cuda_compatible_error(CUDAErrorCode error_code);
CUDAErrorCode cuda_error_translate_from_legacy(int legacy_error_code);
int cuda_error_translate_to_legacy(CUDAErrorCode cuda_error_code);

#ifdef __cplusplus
}
#endif

#endif // CUDA_ERROR_HANDLER_H