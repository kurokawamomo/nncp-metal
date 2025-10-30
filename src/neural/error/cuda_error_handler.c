/*
 * CUDA-Compatible Error Handling System Implementation
 * 
 * Provides error handling system that matches original CUDA behavior
 * and error codes for consistent error reporting and debugging.
 */

#include "cuda_error_handler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <assert.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#endif

// Global error state
static CUDAErrorState g_error_state = {0};
static CUDAErrorConfig g_error_config = {0};
static bool g_error_handler_initialized = false;
static pthread_mutex_t g_error_mutex = PTHREAD_MUTEX_INITIALIZER;

// Error message strings (matching CUDA error messages)
static const char* cuda_error_strings[] = {
    // Success
    [CUDA_ERROR_SUCCESS] = "no error",
    
    // Memory errors (1-99)
    [CUDA_ERROR_OUT_OF_MEMORY] = "out of memory",
    [CUDA_ERROR_MEMORY_ALLOCATION_FAILED] = "memory allocation failed",
    [CUDA_ERROR_INVALID_MEMORY_ACCESS] = "invalid memory access",
    [CUDA_ERROR_MEMORY_COPY_FAILED] = "memory copy failed",
    [CUDA_ERROR_HOST_MEMORY_ALLOC_FAILED] = "host memory allocation failed",
    [CUDA_ERROR_DEVICE_MEMORY_ALLOC_FAILED] = "device memory allocation failed",
    
    // Parameter validation errors (100-199)
    [CUDA_ERROR_INVALID_VALUE] = "invalid argument",
    [CUDA_ERROR_INVALID_HANDLE] = "invalid resource handle",
    [CUDA_ERROR_INVALID_CONFIGURATION] = "invalid configuration",
    [CUDA_ERROR_INVALID_DIMENSION] = "invalid dimension",
    [CUDA_ERROR_INVALID_MEMORY_MAPPING] = "invalid memory mapping",
    [CUDA_ERROR_INVALID_FILTER_SETTING] = "invalid filter setting",
    [CUDA_ERROR_INVALID_NORM_SETTING] = "invalid norm setting",
    [CUDA_ERROR_INVALID_TEXTURE] = "invalid texture reference",
    [CUDA_ERROR_ARRAY_IS_MAPPED] = "array is mapped",
    [CUDA_ERROR_ALREADY_MAPPED] = "already mapped",
    [CUDA_ERROR_NO_BINARY_FOR_GPU] = "no binary for GPU",
    [CUDA_ERROR_ALREADY_ACQUIRED] = "already acquired",
    
    // Execution errors (200-299)
    [CUDA_ERROR_INVALID_CONTEXT] = "invalid device context",
    [CUDA_ERROR_CONTEXT_ALREADY_CURRENT] = "context already current",
    [CUDA_ERROR_MAP_FAILED] = "map failed",
    [CUDA_ERROR_UNMAP_FAILED] = "unmap failed",
    [CUDA_ERROR_NO_ARRAY_AVAILABLE] = "no array available",
    [CUDA_ERROR_INVALID_SYMBOL] = "invalid device symbol",
    [CUDA_ERROR_INVALID_HOST_POINTER] = "invalid host pointer",
    [CUDA_ERROR_INVALID_DEVICE_POINTER] = "invalid device pointer",
    [CUDA_ERROR_INVALID_INSTRUCTION] = "invalid instruction",
    [CUDA_ERROR_INVALID_ADDRESS_SPACE] = "invalid address space",
    [CUDA_ERROR_INVALID_PC] = "invalid program counter",
    [CUDA_ERROR_LAUNCH_FAILED] = "unspecified launch failure",
    [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES] = "too many resources requested for launch",
    [CUDA_ERROR_LAUNCH_TIMEOUT] = "the launch timed out and was terminated",
    [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING] = "incompatible texturing mode",
    
    // Hardware/driver errors (300-399)
    [CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED] = "peer access already enabled",
    [CUDA_ERROR_PEER_ACCESS_NOT_ENABLED] = "peer access not enabled",
    [CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE] = "primary context active",
    [CUDA_ERROR_CONTEXT_IS_DESTROYED] = "context is destroyed",
    [CUDA_ERROR_ASSERT] = "device assert triggered",
    [CUDA_ERROR_TOO_MANY_PEERS] = "too many peers",
    [CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED] = "host memory already registered",
    [CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED] = "host memory not registered",
    [CUDA_ERROR_HARDWARE_STACK_ERROR] = "hardware stack error",
    [CUDA_ERROR_ILLEGAL_INSTRUCTION] = "illegal instruction",
    [CUDA_ERROR_MISALIGNED_ADDRESS] = "misaligned address",
    [CUDA_ERROR_INVALID_ADDRESS_SPACE_311] = "invalid address space",
    [CUDA_ERROR_INVALID_PC_312] = "invalid program counter",
    [CUDA_ERROR_ILLEGAL_ADDRESS] = "illegal address",
    [CUDA_ERROR_INVALID_PTX] = "invalid PTX",
    [CUDA_ERROR_INVALID_GRAPHICS_CONTEXT] = "invalid graphics context",
    [CUDA_ERROR_NV_LINK_UNCORRECTABLE] = "NVLink uncorrectable error",
    [CUDA_ERROR_JIT_COMPILER_NOT_FOUND] = "JIT compiler not found",
    
    // File/stream errors (400-499)
    [CUDA_ERROR_UNSUPPORTED_PTX_VERSION] = "unsupported PTX version",
    [CUDA_ERROR_JIT_COMPILATION_DISABLED] = "JIT compilation disabled",
    [CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY] = "unsupported execution affinity",
    [CUDA_ERROR_FILE_NOT_FOUND] = "file not found",
    [CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND] = "shared object symbol not found",
    [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED] = "shared object initialization failed",
    [CUDA_ERROR_OPERATING_SYSTEM] = "OS call failed",
    [CUDA_ERROR_INVALID_RESOURCE_HANDLE] = "invalid resource handle",
    [CUDA_ERROR_RESOURCE_REQUIRED] = "resource required",
    [CUDA_ERROR_ECC_UNCORRECTABLE] = "ECC uncorrectable error",
    
    // Compression-specific errors (500-599)
    [CUDA_ERROR_COMPRESSION_FAILED] = "compression operation failed",
    [CUDA_ERROR_DECOMPRESSION_FAILED] = "decompression operation failed",
    [CUDA_ERROR_INVALID_COMPRESSED_DATA] = "invalid compressed data format",
    [CUDA_ERROR_COMPRESSION_BUFFER_TOO_SMALL] = "compression buffer too small",
    [CUDA_ERROR_UNSUPPORTED_COMPRESSION_FORMAT] = "unsupported compression format",
    [CUDA_ERROR_COMPRESSION_QUALITY_TOO_LOW] = "compression quality below threshold",
    [CUDA_ERROR_LSTM_PROCESSING_FAILED] = "LSTM processing failed",
    [CUDA_ERROR_TRANSFORMER_PROCESSING_FAILED] = "Transformer processing failed",
    [CUDA_ERROR_NEURAL_NETWORK_ERROR] = "neural network error",
    [CUDA_ERROR_INCOMPATIBLE_PROFILE] = "incompatible CUDA profile",
    [CUDA_ERROR_SEGMENT_LENGTH_MISMATCH] = "segment length mismatch (train_len != seg_len)",
    [CUDA_ERROR_PRECISION_TOLERANCE_EXCEEDED] = "precision tolerance exceeded",
    
    // System-level errors (600-699)
    [CUDA_ERROR_NOT_READY] = "device not ready",
    [CUDA_ERROR_NOT_INITIALIZED] = "initialization error",
    [CUDA_ERROR_NO_DEVICE] = "no CUDA-capable device is detected",
    [CUDA_ERROR_INSUFFICIENT_DRIVER] = "driver version insufficient for CUDA runtime",
    [CUDA_ERROR_STUB_LIBRARY] = "CUDA driver is a stub library",
    [CUDA_ERROR_DEVICE_UNINITIALIZED] = "device uninitialized",
    [CUDA_ERROR_NO_KERNEL_IMAGE_FOR_DEVICE] = "no kernel image available for device",
    [CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE] = "cooperative launch too large",
    [CUDA_ERROR_NOT_PERMITTED] = "operation not permitted",
    [CUDA_ERROR_NOT_SUPPORTED] = "operation not supported",
    [CUDA_ERROR_SYSTEM_NOT_READY] = "system not yet initialized",
    [CUDA_ERROR_SYSTEM_DRIVER_MISMATCH] = "system has unsupported driver",
    [CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE] = "operation not supported on device",
    [CUDA_ERROR_STREAM_CAPTURE_UNMATCHED] = "stream capture unmatched",
    [CUDA_ERROR_STREAM_CAPTURE_INVALIDATED] = "stream capture invalidated",
    [CUDA_ERROR_STREAM_CAPTURE_MERGE] = "stream capture merge",
    [CUDA_ERROR_STREAM_CAPTURE_UNJOINED] = "stream capture unjoined",
    [CUDA_ERROR_STREAM_CAPTURE_ISOLATION] = "stream capture isolation",
    [CUDA_ERROR_STREAM_CAPTURE_IMPLICIT] = "stream capture implicit",
    [CUDA_ERROR_CAPTURED_EVENT] = "captured event",
    [CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD] = "stream capture wrong thread",
    
    // Unknown/undefined errors
    [CUDA_ERROR_UNKNOWN] = "unknown error"
};

// Error severity mapping
static CUDAErrorSeverity cuda_error_severities[] = {
    // Success
    [CUDA_ERROR_SUCCESS] = CUDA_ERROR_SEVERITY_INFO,
    
    // Memory errors - typically fatal
    [CUDA_ERROR_OUT_OF_MEMORY] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_MEMORY_ALLOCATION_FAILED] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_INVALID_MEMORY_ACCESS] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_MEMORY_COPY_FAILED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_HOST_MEMORY_ALLOC_FAILED] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_DEVICE_MEMORY_ALLOC_FAILED] = CUDA_ERROR_SEVERITY_FATAL,
    
    // Parameter validation errors - typically errors
    [CUDA_ERROR_INVALID_VALUE] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_HANDLE] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_CONFIGURATION] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_DIMENSION] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_MEMORY_MAPPING] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_FILTER_SETTING] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_NORM_SETTING] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_TEXTURE] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_ARRAY_IS_MAPPED] = CUDA_ERROR_SEVERITY_WARNING,
    [CUDA_ERROR_ALREADY_MAPPED] = CUDA_ERROR_SEVERITY_WARNING,
    [CUDA_ERROR_NO_BINARY_FOR_GPU] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_ALREADY_ACQUIRED] = CUDA_ERROR_SEVERITY_WARNING,
    
    // Execution errors - typically errors or fatal
    [CUDA_ERROR_INVALID_CONTEXT] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_CONTEXT_ALREADY_CURRENT] = CUDA_ERROR_SEVERITY_WARNING,
    [CUDA_ERROR_MAP_FAILED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_UNMAP_FAILED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_NO_ARRAY_AVAILABLE] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_SYMBOL] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_HOST_POINTER] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_DEVICE_POINTER] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_INSTRUCTION] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_INVALID_ADDRESS_SPACE] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_INVALID_PC] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_LAUNCH_FAILED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_LAUNCH_TIMEOUT] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING] = CUDA_ERROR_SEVERITY_ERROR,
    
    // Compression errors - context dependent
    [CUDA_ERROR_COMPRESSION_FAILED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_DECOMPRESSION_FAILED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INVALID_COMPRESSED_DATA] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_COMPRESSION_BUFFER_TOO_SMALL] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_UNSUPPORTED_COMPRESSION_FORMAT] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_COMPRESSION_QUALITY_TOO_LOW] = CUDA_ERROR_SEVERITY_WARNING,
    [CUDA_ERROR_LSTM_PROCESSING_FAILED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_TRANSFORMER_PROCESSING_FAILED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_NEURAL_NETWORK_ERROR] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_INCOMPATIBLE_PROFILE] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_SEGMENT_LENGTH_MISMATCH] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_PRECISION_TOLERANCE_EXCEEDED] = CUDA_ERROR_SEVERITY_WARNING,
    
    // System errors - typically fatal
    [CUDA_ERROR_NOT_READY] = CUDA_ERROR_SEVERITY_WARNING,
    [CUDA_ERROR_NOT_INITIALIZED] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_NO_DEVICE] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_INSUFFICIENT_DRIVER] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_STUB_LIBRARY] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_DEVICE_UNINITIALIZED] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_NO_KERNEL_IMAGE_FOR_DEVICE] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_NOT_PERMITTED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_NOT_SUPPORTED] = CUDA_ERROR_SEVERITY_ERROR,
    [CUDA_ERROR_SYSTEM_NOT_READY] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_SYSTEM_DRIVER_MISMATCH] = CUDA_ERROR_SEVERITY_FATAL,
    [CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE] = CUDA_ERROR_SEVERITY_ERROR,
    
    // Default to error severity
    [CUDA_ERROR_UNKNOWN] = CUDA_ERROR_SEVERITY_ERROR
};

// Utility functions
static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint32_t get_thread_id(void) {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return (uint32_t)pthread_self();
#endif
}

static uint32_t get_process_id(void) {
#ifdef _WIN32
    return GetCurrentProcessId();
#else
    return getpid();
#endif
}

// Configuration functions
CUDAErrorConfig* cuda_error_config_create_default(void) {
    CUDAErrorConfig* config = (CUDAErrorConfig*)calloc(1, sizeof(CUDAErrorConfig));
    if (!config) return NULL;
    
    config->enable_error_logging = true;
    config->enable_stack_trace = false;
    config->abort_on_fatal_error = false;
    config->chain_errors = true;
    config->error_callback = NULL;
    config->callback_user_data = NULL;
    config->log_file_path = NULL;
    config->log_to_console = true;
    config->min_log_severity = CUDA_ERROR_SEVERITY_WARNING;
    config->max_error_chain_length = 10;
    config->enable_error_caching = true;
    
    return config;
}

CUDAErrorConfig* cuda_error_config_create_production(void) {
    CUDAErrorConfig* config = cuda_error_config_create_default();
    if (!config) return NULL;
    
    config->enable_stack_trace = false;
    config->abort_on_fatal_error = true;
    config->log_to_console = false;
    config->min_log_severity = CUDA_ERROR_SEVERITY_ERROR;
    
    return config;
}

CUDAErrorConfig* cuda_error_config_create_debug(void) {
    CUDAErrorConfig* config = cuda_error_config_create_default();
    if (!config) return NULL;
    
    config->enable_stack_trace = true;
    config->abort_on_fatal_error = false;
    config->log_to_console = true;
    config->min_log_severity = CUDA_ERROR_SEVERITY_INFO;
    config->max_error_chain_length = 50;
    
    return config;
}

void cuda_error_config_free(CUDAErrorConfig* config) {
    if (config) {
        free(config);
    }
}

// Initialization and shutdown
bool cuda_error_init(const CUDAErrorConfig* config) {
    if (g_error_handler_initialized) {
        return true;
    }
    
    pthread_mutex_lock(&g_error_mutex);
    
    // Initialize global state
    memset(&g_error_state, 0, sizeof(CUDAErrorState));
    
    // Copy configuration
    if (config) {
        g_error_config = *config;
    } else {
        CUDAErrorConfig* default_config = cuda_error_config_create_default();
        if (default_config) {
            g_error_config = *default_config;
            cuda_error_config_free(default_config);
        }
    }
    
    // Initialize error state
    g_error_state.last_error = CUDA_ERROR_SUCCESS;
    g_error_state.error_chain = NULL;
    g_error_state.error_count = 0;
    g_error_state.last_error_timestamp = 0;
    
    // Initialize statistics
    memset(g_error_state.total_errors, 0, sizeof(g_error_state.total_errors));
    g_error_state.memory_errors = 0;
    g_error_state.compression_errors = 0;
    g_error_state.cuda_compatibility_errors = 0;
    
    g_error_handler_initialized = true;
    
    pthread_mutex_unlock(&g_error_mutex);
    
    return true;
}

void cuda_error_shutdown(void) {
    if (!g_error_handler_initialized) {
        return;
    }
    
    pthread_mutex_lock(&g_error_mutex);
    
    // Clear error chain
    cuda_error_clear_chain();
    
    // Reset state
    memset(&g_error_state, 0, sizeof(CUDAErrorState));
    memset(&g_error_config, 0, sizeof(CUDAErrorConfig));
    
    g_error_handler_initialized = false;
    
    pthread_mutex_unlock(&g_error_mutex);
}

bool cuda_error_is_initialized(void) {
    return g_error_handler_initialized;
}

// Error reporting
CUDAErrorCode cuda_error_set(CUDAErrorCode error_code, 
                             const char* file, 
                             int line, 
                             const char* function,
                             const char* format, ...) {
    if (!g_error_handler_initialized) {
        return error_code;
    }
    
    pthread_mutex_lock(&g_error_mutex);
    
    // Create error info
    CUDAErrorInfo* error_info = (CUDAErrorInfo*)calloc(1, sizeof(CUDAErrorInfo));
    if (!error_info) {
        pthread_mutex_unlock(&g_error_mutex);
        return error_code;
    }
    
    error_info->error_code = error_code;
    error_info->severity = cuda_error_get_severity(error_code);
    error_info->file = file;
    error_info->line = line;
    error_info->function = function;
    error_info->cuda_device_id = -1;  // Not applicable for Metal
    error_info->cuda_operation = "Metal operation";
    error_info->cuda_memory_used = 0;  // Could be enhanced
    error_info->timestamp = get_timestamp_ns();
    error_info->thread_id = get_thread_id();
    error_info->process_id = get_process_id();
    error_info->previous_error = NULL;
    
    // Format message
    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(error_info->message, sizeof(error_info->message), format, args);
        va_end(args);
    } else {
        strncpy(error_info->message, cuda_error_get_string(error_code), 
                sizeof(error_info->message) - 1);
    }
    
    // Update global state
    g_error_state.last_error = error_code;
    g_error_state.last_error_timestamp = error_info->timestamp;
    g_error_state.error_count++;
    
    // Update statistics
    if (error_info->severity < 4) {
        g_error_state.total_errors[error_info->severity]++;
    }
    
    // Category-specific tracking
    if (error_code >= CUDA_ERROR_OUT_OF_MEMORY && error_code <= CUDA_ERROR_DEVICE_MEMORY_ALLOC_FAILED) {
        g_error_state.memory_errors++;
    } else if (error_code >= CUDA_ERROR_COMPRESSION_FAILED && error_code <= CUDA_ERROR_PRECISION_TOLERANCE_EXCEEDED) {
        g_error_state.compression_errors++;
    } else if (error_code == CUDA_ERROR_INCOMPATIBLE_PROFILE || error_code == CUDA_ERROR_SEGMENT_LENGTH_MISMATCH) {
        g_error_state.cuda_compatibility_errors++;
    }
    
    // Add to error chain if enabled
    if (g_error_config.chain_errors) {
        if (g_error_state.error_chain) {
            error_info->previous_error = g_error_state.error_chain;
        }
        g_error_state.error_chain = error_info;
        
        // Limit chain length
        size_t chain_length = 1;
        CUDAErrorInfo* current = error_info->previous_error;
        CUDAErrorInfo* prev = NULL;
        
        while (current && chain_length < g_error_config.max_error_chain_length) {
            prev = current;
            current = current->previous_error;
            chain_length++;
        }
        
        // Truncate chain if too long
        if (current && prev) {
            prev->previous_error = NULL;
            // Free remaining errors
            while (current) {
                CUDAErrorInfo* next = current->previous_error;
                free(current);
                current = next;
            }
        }
    } else {
        // Free the error info if not chaining
        free(error_info);
    }
    
    // Logging
    if (g_error_config.enable_error_logging && 
        error_info->severity >= g_error_config.min_log_severity) {
        
        if (g_error_config.log_to_console) {
            const char* severity_str = "UNKNOWN";
            switch (error_info->severity) {
                case CUDA_ERROR_SEVERITY_INFO: severity_str = "INFO"; break;
                case CUDA_ERROR_SEVERITY_WARNING: severity_str = "WARNING"; break;
                case CUDA_ERROR_SEVERITY_ERROR: severity_str = "ERROR"; break;
                case CUDA_ERROR_SEVERITY_FATAL: severity_str = "FATAL"; break;
            }
            
            fprintf(stderr, "[CUDA %s] %s:%d in %s(): %s (error %d)\n",
                   severity_str, file ? file : "unknown", line, 
                   function ? function : "unknown", error_info->message, error_code);
        }
    }
    
    // Call user callback if set
    if (g_error_config.error_callback && g_error_config.chain_errors) {
        g_error_config.error_callback(error_info, g_error_config.callback_user_data);
    }
    
    // Abort on fatal error if configured
    if (g_error_config.abort_on_fatal_error && error_info->severity == CUDA_ERROR_SEVERITY_FATAL) {
        fprintf(stderr, "[CUDA FATAL] Aborting due to fatal error: %s\n", error_info->message);
        abort();
    }
    
    pthread_mutex_unlock(&g_error_mutex);
    
    return error_code;
}

CUDAErrorCode cuda_error_set_with_context(CUDAErrorCode error_code,
                                         const char* file,
                                         int line, 
                                         const char* function,
                                         const char* context,
                                         const char* format, ...) {
    if (!g_error_handler_initialized) {
        return error_code;
    }
    
    char message[256];
    if (format) {
        va_list args;
        va_start(args, format);
        vsnprintf(message, sizeof(message), format, args);
        va_end(args);
    } else {
        strncpy(message, cuda_error_get_string(error_code), sizeof(message) - 1);
    }
    
    // Add context to message
    if (context) {
        char full_message[512];
        snprintf(full_message, sizeof(full_message), "%s (context: %s)", message, context);
        return cuda_error_set(error_code, file, line, function, "%s", full_message);
    } else {
        return cuda_error_set(error_code, file, line, function, "%s", message);
    }
}

// Error querying functions
CUDAErrorCode cuda_error_get_last(void) {
    if (!g_error_handler_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    return g_error_state.last_error;
}

const char* cuda_error_get_string(CUDAErrorCode error_code) {
    if (error_code >= 0 && error_code < sizeof(cuda_error_strings) / sizeof(cuda_error_strings[0])) {
        const char* msg = cuda_error_strings[error_code];
        return msg ? msg : "unknown error";
    }
    return "unknown error";
}

const char* cuda_error_get_description(CUDAErrorCode error_code) {
    // For now, same as get_string. Could be enhanced with more detailed descriptions.
    return cuda_error_get_string(error_code);
}

CUDAErrorSeverity cuda_error_get_severity(CUDAErrorCode error_code) {
    if (error_code >= 0 && error_code < sizeof(cuda_error_severities) / sizeof(cuda_error_severities[0])) {
        return cuda_error_severities[error_code];
    }
    return CUDA_ERROR_SEVERITY_ERROR;
}

// Error information retrieval
const CUDAErrorInfo* cuda_error_get_info(void) {
    if (!g_error_handler_initialized) {
        return NULL;
    }
    
    return g_error_state.error_chain;
}

const CUDAErrorInfo* cuda_error_get_chain(void) {
    if (!g_error_handler_initialized) {
        return NULL;
    }
    
    return g_error_state.error_chain;
}

size_t cuda_error_get_chain_length(void) {
    if (!g_error_handler_initialized) {
        return 0;
    }
    
    size_t length = 0;
    const CUDAErrorInfo* current = g_error_state.error_chain;
    while (current) {
        length++;
        current = current->previous_error;
    }
    
    return length;
}

// Error state management
void cuda_error_clear(void) {
    if (!g_error_handler_initialized) {
        return;
    }
    
    pthread_mutex_lock(&g_error_mutex);
    g_error_state.last_error = CUDA_ERROR_SUCCESS;
    pthread_mutex_unlock(&g_error_mutex);
}

void cuda_error_clear_chain(void) {
    if (!g_error_handler_initialized) {
        return;
    }
    
    pthread_mutex_lock(&g_error_mutex);
    
    CUDAErrorInfo* current = g_error_state.error_chain;
    while (current) {
        CUDAErrorInfo* next = current->previous_error;
        free(current);
        current = next;
    }
    
    g_error_state.error_chain = NULL;
    
    pthread_mutex_unlock(&g_error_mutex);
}

bool cuda_error_has_error(void) {
    if (!g_error_handler_initialized) {
        return false;
    }
    
    return g_error_state.last_error != CUDA_ERROR_SUCCESS;
}

bool cuda_error_has_fatal_error(void) {
    if (!g_error_handler_initialized) {
        return false;
    }
    
    const CUDAErrorInfo* current = g_error_state.error_chain;
    while (current) {
        if (current->severity == CUDA_ERROR_SEVERITY_FATAL) {
            return true;
        }
        current = current->previous_error;
    }
    
    return false;
}

// Statistics and diagnostics
void cuda_error_print_statistics(void) {
    if (!g_error_handler_initialized) {
        printf("CUDA error handler not initialized\n");
        return;
    }
    
    printf("CUDA Error Handler Statistics:\n");
    printf("  Total errors: %zu\n", g_error_state.error_count);
    printf("  Info messages: %zu\n", g_error_state.total_errors[CUDA_ERROR_SEVERITY_INFO]);
    printf("  Warnings: %zu\n", g_error_state.total_errors[CUDA_ERROR_SEVERITY_WARNING]);
    printf("  Errors: %zu\n", g_error_state.total_errors[CUDA_ERROR_SEVERITY_ERROR]);
    printf("  Fatal errors: %zu\n", g_error_state.total_errors[CUDA_ERROR_SEVERITY_FATAL]);
    printf("  Memory errors: %zu\n", g_error_state.memory_errors);
    printf("  Compression errors: %zu\n", g_error_state.compression_errors);
    printf("  CUDA compatibility errors: %zu\n", g_error_state.cuda_compatibility_errors);
    printf("  Current error chain length: %zu\n", cuda_error_get_chain_length());
    printf("  Last error: %s (%d)\n", 
           cuda_error_get_string(g_error_state.last_error), g_error_state.last_error);
}

void cuda_error_print_chain(void) {
    if (!g_error_handler_initialized) {
        printf("CUDA error handler not initialized\n");
        return;
    }
    
    printf("CUDA Error Chain:\n");
    
    const CUDAErrorInfo* current = g_error_state.error_chain;
    int index = 0;
    
    while (current) {
        const char* severity_str = "UNKNOWN";
        switch (current->severity) {
            case CUDA_ERROR_SEVERITY_INFO: severity_str = "INFO"; break;
            case CUDA_ERROR_SEVERITY_WARNING: severity_str = "WARNING"; break;
            case CUDA_ERROR_SEVERITY_ERROR: severity_str = "ERROR"; break;
            case CUDA_ERROR_SEVERITY_FATAL: severity_str = "FATAL"; break;
        }
        
        printf("  [%d] %s: %s (error %d)\n", index, severity_str, current->message, current->error_code);
        printf("      Location: %s:%d in %s()\n", 
               current->file ? current->file : "unknown", 
               current->line, 
               current->function ? current->function : "unknown");
        printf("      Timestamp: %llu ns\n", (unsigned long long)current->timestamp);
        
        current = current->previous_error;
        index++;
    }
    
    if (index == 0) {
        printf("  (no errors in chain)\n");
    }
}

// Compatibility functions
bool cuda_error_is_cuda_compatible_error(CUDAErrorCode error_code) {
    // All error codes are designed to be CUDA-compatible
    return (error_code >= CUDA_ERROR_SUCCESS && error_code <= CUDA_ERROR_UNKNOWN);
}

// Thread-local error handling (simplified implementation)
CUDAErrorCode cuda_error_get_thread_local(void) {
    if (!g_error_handler_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    // Simplified: just return global last error
    // In a full implementation, this would use thread-local storage
    return g_error_state.last_error;
}

void cuda_error_set_thread_local(CUDAErrorCode error_code) {
    if (!g_error_handler_initialized) {
        return;
    }
    
    // Simplified: just set global last error
    // In a full implementation, this would use thread-local storage
    g_error_state.last_error = error_code;
}