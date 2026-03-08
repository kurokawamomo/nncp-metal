# Metal Graph Optimization Report

**Target File:** `src/neural/mps_optimized/mps_transformer_graph.mm`

## Findings

### 1. Weight Tensor Data Caching (High Priority)
*   **Observation:** The `mps_transformer_execute` function creates new `MPSGraphTensorData` wrappers for all model weights (embeddings, QKV, FFN, etc.) on *every* execution call.
*   **Impact:** This introduces unnecessary CPU overhead for object allocation and Metal resource binding during every inference step.
*   **Recommendation:** Cache `MPSGraphTensorData` objects within the `MPSTransformerContext`. These should be created once when weights are set (in `mps_transformer_set_weights`) or lazily on the first run, and then reused.

### 2. Executable Caching for Dynamic Shapes (Critical)
*   **Observation:** The code compiles the `MPSGraph` only once (`if (!ctx->executable)`). The input placeholders are created with fixed shapes based on the `batch_size` and `seq_len` of the *first* call.
*   **Impact:** If `mps_transformer_execute` is called subsequently with different dimensions (e.g., a different batch size or sequence length), the cached executable will be incompatible, likely leading to a crash or incorrect results.
*   **Recommendation:** Implement an `NSMutableDictionary` cache for `MPSGraphExecutable` objects, keyed by a string representing the input shape (e.g., `"{batch}x{seq}"`). This ensures correct execution for varying input sizes while maintaining the performance benefits of compilation caching.

### 3. Input Data Transfer Overhead
*   **Observation:** Input data is wrapped in `NSData` (`dataWithBytes`) which implies a CPU copy before being handed to Metal.
*   **Impact:** Additional latency due to memory copying.
*   **Recommendation:** Where possible, use `initWithBytesNoCopy:length:deallocator:` if the input pointer lifetime is guaranteed, or ideally, update the API to accept `id<MTLBuffer>` directly for zero-copy transfer from Unified Memory.

### 4. Synchronous Execution Blocking (Throughput Bottleneck)
*   **Observation:** The function calls `[resultData.mpsndarray readBytes:output_data strideBytes:NULL]` at the end of execution. This method blocks the CPU thread until the GPU computation is complete and data is copied back.
*   **Impact:** This prevents CPU-GPU overlap (pipelining). The CPU sits idle while the GPU works, and vice-versa.
*   **Recommendation:** 
    *   If the consumer can handle it, return the `MPSGraphTensorData` or `id<MTLBuffer>` directly instead of copying to a CPU float array.
    *   If CPU access is required, investigate using `MPSGraph`'s asynchronous execution APIs or `synchronizeResource:` on the underlying buffer only when data is actually needed, allowing the CPU to prepare the next batch in parallel.

## Proposed Action Plan

1.  Modify `MPSTransformerContext` to include:
    *   `NSMutableDictionary<NSString*, MPSGraphExecutable*>* executableCache`
    *   `NSMutableDictionary<NSString*, MPSGraphTensorData*>* weightCache`
2.  Update `mps_transformer_set_weights` to clear or update `weightCache`.
3.  Refactor `mps_transformer_execute` to:
    *   Construct a cache key from `batch_size` and `seq_len`.
    *   Check `executableCache` for an existing executable.
    *   If missing, compile a new graph specialized for these shapes and store it.
    *   Use cached `MPSGraphTensorData` for weights instead of recreating them.
    *   Use `initWithBytesNoCopy` for input tensor creation.
