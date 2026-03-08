# Current Performance Bottleneck Analysis (nncp-metal)

**Observed Symptom:** Decompression speed is extremely slow (~100 B/s).

**Root Cause Identification:**

The bottleneck is **Latency-Bound Execution** due to byte-by-byte CPU-GPU synchronization.

### 1. The "Byte-by-Byte" Round-Trip Loop
In `src/neural/integration/neural_bridge_lossless_cuda.mm`, the `neural_bridge_cuda_lossless_decompress` function implements the following loop:

```c
while (decoded_bytes < output_capacity ...) {
    // ...
    // 1. Dispatch GPU Work
    bool success = flow_optimizer_execute_batch(..., input_batch, ...); 
    
    // 2. CPU Reads Result (Implicit Synchronization)
    float* current_logits = output_logits + (offset_in_chunk * vocab_size);
    // ... arithmetic decoding ...
}
```

*   **Logic:** Decompression is autoregressive. To decode byte `t`, we need the probability distribution $P(x_t | x_{0...t-1})$. This requires running the Transformer.
*   **Synchronization:** `flow_optimizer_execute_batch` eventually calls `gpu_transformer_predict_batch`. Even if this dispatch is asynchronous, the very next lines of code on the CPU access `output_logits` (Unified Memory). To read valid data, the CPU *must* wait for the GPU to finish writing.
*   **Latency:** The overhead of submitting a command buffer to the GPU, executing even a small kernel, and signaling completion back to the CPU is typically **0.1ms to 10ms** depending on the driver and system load.
    *   At 10ms latency: 1 byte / 0.01s = **100 Bytes/second**.
    *   This matches the observed performance exactly.

### 2. Inefficient Re-computation (Lack of KV Cache)
In `src/neural/engines/gpu_native_transformer.mm`, the `gpu_transformer_predict_batch` function re-executes the entire Transformer for the full sequence length (up to 64) for *every* step.

```objectivec
// In gpu_transformer_encode_batch
// ...
// QKV Projections for ALL tokens in sequence
dispatch_linear(..., input_count, ...); 
// Attention for ALL tokens
// ...
```

*   It does not cache the Key/Value vectors from previous steps.
*   While this increases computational load ($O(N^2)$ instead of $O(N)$), it is secondary to the latency bottleneck. Even with zero computation time, the driver overhead would limit speed to ~1-2 KB/s.

### 3. Single-Stream Processing
The current implementation processes a single stream of data (`batch_size = 1`).
*   GPUs rely on massive parallelism to hide latency.
*   Processing one byte of one stream utilizes < 1% of the GPU.

## Conclusion
The architecture is fundamentally limited by the **serial dependency** of the arithmetic decoder combined with the **high latency** of CPU-GPU roundtrips.

## Recommendations for Optimization

1.  **Multi-Stream Parallelism (Highest Impact):**
    *   Modify the compression format to split the file into $N$ independent chunks (e.g., $N=64$ or $128$).
    *   Process all $N$ chunks in parallel during decompression.
    *   This allows sending a batch of $N$ queries to the GPU at once.
    *   Expected Speedup: Linear with batch size (e.g., 64x - 128x).

2.  **KV Caching (Medium Impact):**
    *   Implement a stateful Transformer context that retains `K` and `V` matrices.
    *   Only compute the embedding and QKV for the *newest* token.
    *   Attend to the cached keys/values.

3.  **CPU-Side Optimization (Low Latency Fallback):**
    *   For very small models or extremely latency-sensitive loops, a highly optimized CPU implementation (SIMD/AMX) might beat the GPU due to zero synchronization overhead.
