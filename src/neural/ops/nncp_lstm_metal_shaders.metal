#include <metal_stdlib>
using namespace metal;

// Constants matching CUDA implementation
constant float LAYER_NORM_EPS = 1e-5f;
constant uint MAX_THREADS_PER_THREADGROUP = 1024;

// NNCP LSTM Compute Shader Parameters
struct NNCPLSTMParams {
    uint n_cells;           // Hidden state dimension (default: 352)
    uint n_cells2;          // Cell state dimension (default: 352)
    uint n_symbols;         // Vocabulary size (default: 256)
    uint n_streams;         // Batch size
    uint seg_len;           // Sequence length
    uint mat_count;         // Number of gate matrices (3 for CLAMPED)
    uint layer_idx;         // Current layer index
    bool use_layer_norm;    // Enable layer normalization
    float layer_norm_eps;   // Layer norm epsilon
};

// Memory layout matching CUDA implementation
struct NNCPLSTMBuffers {
    // Weight matrices (concatenated for efficiency)
    device float* u_weights;        // Recurrent weights [n_cells2*mat_count, n_cells]
    device float* w_weights;        // Dense weights [n_cells2*mat_count, n_inputs]
    device float* ws_weights;       // Sparse embedding [n_cells2*mat_count, n_symbols]
    
    // Bias vectors per gate
    device float* b_forget;         // Forget gate bias [n_cells2]
    device float* b_input;          // Input gate bias [n_cells2]
    device float* b_output;         // Output gate bias [n_cells2]
    
    // Layer normalization parameters
    device float* g_forget;         // Forget gate scale [n_cells2]
    device float* g_input;          // Input gate scale [n_cells2]
    device float* g_output;         // Output gate scale [n_cells2]
    
    // State buffers
    device float* h_states;         // Hidden states [n_cells, n_streams]
    device float* c_states;         // Cell states [n_cells2, n_streams]
    device float* h_prev;           // Previous hidden states
    device float* c_prev;           // Previous cell states
    
    // Temporary computation buffers
    device float* gate_inputs;      // Gate inputs [n_cells2*mat_count, n_streams]
    device float* gate_outputs;     // Gate outputs [n_cells2, n_streams] per gate
    device float* temp_buffer;      // General purpose temporary buffer
};

// Sparse embedding lookup kernel (nc_get_col equivalent)
kernel void nncp_lstm_sparse_lookup(
    device const uint* input_symbols [[buffer(0)]],      // Input symbols [n_streams]
    device const float* ws_weights [[buffer(1)]],        // Embedding weights [n_cells2*mat_count, n_symbols]
    device float* output [[buffer(2)]],                  // Output embeddings [n_cells2*mat_count, n_streams]
    constant NNCPLSTMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint stream_idx = gid.x;
    uint weight_row = gid.y;
    
    if (stream_idx >= params.n_streams || weight_row >= params.n_cells2 * params.mat_count) {
        return;
    }
    
    uint symbol = input_symbols[stream_idx];
    if (symbol >= params.n_symbols) {
        output[weight_row * params.n_streams + stream_idx] = 0.0f;
        return;
    }
    
    // Sparse lookup: ws_weights[weight_row, symbol]
    uint ws_index = weight_row * params.n_symbols + symbol;
    output[weight_row * params.n_streams + stream_idx] = ws_weights[ws_index];
}

// Matrix multiplication kernel (nc_matmul equivalent)
kernel void nncp_lstm_matrix_multiply(
    device const float* weights [[buffer(0)]],           // Weight matrix [rows, cols]
    device const float* input [[buffer(1)]],            // Input vector [cols, n_streams]
    device float* output [[buffer(2)]],                 // Output vector [rows, n_streams]
    constant uint& rows [[buffer(3)]],                  // Matrix rows
    constant uint& cols [[buffer(4)]],                  // Matrix columns
    constant uint& n_streams [[buffer(5)]],             // Number of streams
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint stream = gid.y;
    
    if (row >= rows || stream >= n_streams) {
        return;
    }
    
    float sum = 0.0f;
    for (uint col = 0; col < cols; col++) {
        float weight = weights[row * cols + col];
        float input_val = input[col * n_streams + stream];
        sum += weight * input_val;
    }
    
    output[row * n_streams + stream] = sum;
}

// RMS Layer normalization kernel
kernel void nncp_lstm_rms_layer_norm(
    device const float* input [[buffer(0)]],            // Input [n_cells2, n_streams]
    device const float* scale [[buffer(1)]],            // Scale parameters [n_cells2]
    device float* output [[buffer(2)]],                 // Output [n_cells2, n_streams]
    constant NNCPLSTMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint stream_idx = gid.x;
    
    if (stream_idx >= params.n_streams) {
        return;
    }
    
    // Calculate RMS for this stream
    float sum_squares = 0.0f;
    for (uint i = 0; i < params.n_cells2; i++) {
        float val = input[i * params.n_streams + stream_idx];
        sum_squares += val * val;
    }
    
    float rms = sqrt(sum_squares / float(params.n_cells2) + params.layer_norm_eps);
    
    // Apply normalization and scaling
    for (uint i = 0; i < params.n_cells2; i++) {
        float normalized = input[i * params.n_streams + stream_idx] / rms;
        float scaled = normalized * scale[i];
        output[i * params.n_streams + stream_idx] = scaled;
    }
}

// Sigmoid activation function
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Tanh activation function
inline float tanh_activation(float x) {
    return tanh(x);
}

// Critical nc_lstm_clamped operation (CUDA implementation)
inline float nc_lstm_clamped(float c_prev, float input_node, float forget_gate, float input_gate) {
    // CUDA reference implementation:
    // return cp * fg + min(1 - fg, ig) * in
    // For CLAMPED type, input_gate is effectively not used, so we use min(1-fg, 1.0)
    float effective_input_gate = min(1.0f - forget_gate, 1.0f);
    return c_prev * forget_gate + effective_input_gate * input_node;
}

// Main LSTM cell computation kernel
kernel void nncp_lstm_cell_forward(
    device const float* gate_inputs [[buffer(0)]],      // Gate inputs [n_cells2*mat_count, n_streams]
    device const float* c_prev [[buffer(1)]],           // Previous cell state [n_cells2, n_streams]
    device const float* b_forget [[buffer(2)]],         // Forget bias [n_cells2]
    device const float* b_input [[buffer(3)]],          // Input bias [n_cells2]
    device const float* b_output [[buffer(4)]],         // Output bias [n_cells2]
    device const float* g_forget [[buffer(5)]],         // Forget scale [n_cells2] (optional)
    device const float* g_input [[buffer(6)]],          // Input scale [n_cells2] (optional)
    device const float* g_output [[buffer(7)]],         // Output scale [n_cells2] (optional)
    device float* c_new [[buffer(8)]],                  // New cell state [n_cells2, n_streams]
    device float* h_new [[buffer(9)]],                  // New hidden state [n_cells2, n_streams]
    constant NNCPLSTMParams& params [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint cell_idx = gid.x;
    uint stream_idx = gid.y;
    
    if (cell_idx >= params.n_cells2 || stream_idx >= params.n_streams) {
        return;
    }
    
    uint linear_idx = cell_idx * params.n_streams + stream_idx;
    
    // Extract gate inputs (split operation)
    float forget_raw = gate_inputs[0 * params.n_cells2 * params.n_streams + linear_idx];
    float input_raw = gate_inputs[1 * params.n_cells2 * params.n_streams + linear_idx];
    float output_raw = gate_inputs[2 * params.n_cells2 * params.n_streams + linear_idx];
    
    // Apply layer normalization if enabled
    if (params.use_layer_norm) {
        // RMS normalization per gate
        forget_raw = forget_raw * g_forget[cell_idx];
        input_raw = input_raw * g_input[cell_idx];
        output_raw = output_raw * g_output[cell_idx];
    }
    
    // Add biases
    forget_raw += b_forget[cell_idx];
    input_raw += b_input[cell_idx];
    output_raw += b_output[cell_idx];
    
    // Apply activations
    float forget_gate = sigmoid(forget_raw);
    float input_node = tanh_activation(input_raw);
    float output_gate = sigmoid(output_raw);
    
    // LSTM cell state update using nc_lstm_clamped
    float c_prev_val = c_prev[linear_idx];
    float c_new_val = nc_lstm_clamped(c_prev_val, input_node, forget_gate, 1.0f);
    
    // Hidden state computation
    float h_new_val = output_gate * c_new_val;
    
    // Store results
    c_new[linear_idx] = c_new_val;
    h_new[linear_idx] = h_new_val;
}

// Gate input computation kernel (combines recurrent and input contributions)
kernel void nncp_lstm_gate_inputs(
    device const float* h_prev [[buffer(0)]],           // Previous hidden state [n_cells, n_streams]
    device const float* u_weights [[buffer(1)]],        // Recurrent weights [n_cells2*mat_count, n_cells]
    device const float* sparse_input [[buffer(2)]],     // Sparse embedding [n_cells2*mat_count, n_streams]
    device const float* dense_input [[buffer(3)]],      // Dense input [n_cells2*mat_count, n_streams] (optional)
    device float* gate_inputs [[buffer(4)]],            // Output gate inputs [n_cells2*mat_count, n_streams]
    constant NNCPLSTMParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint gate_cell_idx = gid.x;  // Combined gate and cell index
    uint stream_idx = gid.y;
    
    if (gate_cell_idx >= params.n_cells2 * params.mat_count || stream_idx >= params.n_streams) {
        return;
    }
    
    uint linear_idx = gate_cell_idx * params.n_streams + stream_idx;
    
    // Recurrent contribution: u * h_prev
    float recurrent_sum = 0.0f;
    for (uint h_idx = 0; h_idx < params.n_cells; h_idx++) {
        float weight = u_weights[gate_cell_idx * params.n_cells + h_idx];
        float hidden = h_prev[h_idx * params.n_streams + stream_idx];
        recurrent_sum += weight * hidden;
    }
    
    // Add sparse embedding contribution
    float sparse_contrib = sparse_input[linear_idx];
    
    // Add dense input contribution (if available)
    float dense_contrib = 0.0f;
    if (dense_input != nullptr) {
        dense_contrib = dense_input[linear_idx];
    }
    
    // Combine all contributions
    gate_inputs[linear_idx] = recurrent_sum + sparse_contrib + dense_contrib;
}

// Utility kernel for tensor operations
kernel void nncp_lstm_tensor_add(
    device const float* input_a [[buffer(0)]],          // First input tensor
    device const float* input_b [[buffer(1)]],          // Second input tensor
    device float* output [[buffer(2)]],                 // Output tensor
    constant uint& size [[buffer(3)]],                  // Tensor size
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) {
        return;
    }
    
    output[gid] = input_a[gid] + input_b[gid];
}

// Utility kernel for clearing buffers
kernel void nncp_lstm_clear_buffer(
    device float* buffer [[buffer(0)]],                 // Buffer to clear
    constant uint& size [[buffer(1)]],                  // Buffer size
    constant float& value [[buffer(2)]],                // Clear value
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) {
        return;
    }
    
    buffer[gid] = value;
}

// State copy kernel (for h0/c0 management)
kernel void nncp_lstm_copy_state(
    device const float* source [[buffer(0)]],           // Source state
    device float* destination [[buffer(1)]],            // Destination state
    constant uint& size [[buffer(2)]],                  // State size
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) {
        return;
    }
    
    destination[gid] = source[gid];
}

// Sequence processing kernel (full timestep through one layer)
kernel void nncp_lstm_process_timestep(
    device const uint* input_symbols [[buffer(0)]],     // Input symbols [n_streams, seg_len]
    device const float* dense_input [[buffer(1)]],      // Dense input from prev layer (optional)
    device float* h_states [[buffer(2)]],               // Hidden states [n_cells, n_streams]
    device float* c_states [[buffer(3)]],               // Cell states [n_cells2, n_streams]
    device const float* u_weights [[buffer(4)]],        // Recurrent weights
    device const float* w_weights [[buffer(5)]],        // Dense weights (optional)
    device const float* ws_weights [[buffer(6)]],       // Sparse weights
    device const float* biases [[buffer(7)]],           // All biases concatenated
    device const float* layer_norm_params [[buffer(8)]], // Layer norm parameters (optional)
    device float* temp_buffers [[buffer(9)]],           // Temporary buffers
    constant NNCPLSTMParams& params [[buffer(10)]],
    constant uint& timestep [[buffer(11)]],             // Current timestep
    uint2 gid [[thread_position_in_grid]]
) {
    uint stream_idx = gid.x;
    
    if (stream_idx >= params.n_streams) {
        return;
    }
    
    // This kernel would orchestrate the full timestep computation
    // by calling the individual kernels in sequence:
    // 1. Sparse lookup
    // 2. Matrix multiply (recurrent)
    // 3. Add dense input (if available)
    // 4. Layer normalization (if enabled)
    // 5. LSTM cell computation
    // 6. Store new states
    
    // Note: In a real implementation, this would be broken down into
    // separate kernel dispatches to avoid complexity and ensure proper
    // memory synchronization between stages
}

// Output projection kernel (final layer processing)
kernel void nncp_lstm_output_projection(
    device const float* layer_outputs [[buffer(0)]],    // Concatenated layer outputs
    device const float* fc_weights [[buffer(1)]],       // Output projection weights
    device const float* fc_bias [[buffer(2)]],          // Output projection bias (optional)
    device float* logits [[buffer(3)]],                 // Output logits [n_streams*seg_len, n_symbols]
    constant NNCPLSTMParams& params [[buffer(4)]],
    constant uint& n_total_cells [[buffer(5)]],         // Total cells from all layers
    uint2 gid [[thread_position_in_grid]]
) {
    uint output_idx = gid.x;   // Output symbol index
    uint sequence_idx = gid.y; // Sequence position (stream*seg_len + timestep)
    
    if (output_idx >= params.n_symbols || sequence_idx >= params.n_streams * params.seg_len) {
        return;
    }
    
    // Matrix multiply: fc_weights * layer_outputs
    float logit = 0.0f;
    for (uint i = 0; i < n_total_cells; i++) {
        float weight = fc_weights[output_idx * n_total_cells + i];
        float input_val = layer_outputs[i * params.n_streams * params.seg_len + sequence_idx];
        logit += weight * input_val;
    }
    
    // Add bias if available
    if (fc_bias != nullptr) {
        logit += fc_bias[output_idx];
    }
    
    // Store result
    logits[sequence_idx * params.n_symbols + output_idx] = logit;
}

// Debug and validation kernel
kernel void nncp_lstm_validate_tensors(
    device const float* tensor [[buffer(0)]],           // Tensor to validate
    device uint* error_flags [[buffer(1)]],             // Error flags output
    constant uint& size [[buffer(2)]],                  // Tensor size
    constant float& min_val [[buffer(3)]],              // Minimum allowed value
    constant float& max_val [[buffer(4)]],              // Maximum allowed value
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) {
        return;
    }
    
    float val = tensor[gid];
    
    // Check for NaN
    if (isnan(val)) {
        atomic_fetch_or_explicit((device atomic_uint*)&error_flags[0], 1, memory_order_relaxed);
    }
    
    // Check for infinity
    if (isinf(val)) {
        atomic_fetch_or_explicit((device atomic_uint*)&error_flags[1], 1, memory_order_relaxed);
    }
    
    // Check for out of range
    if (val < min_val || val > max_val) {
        atomic_fetch_or_explicit((device atomic_uint*)&error_flags[2], 1, memory_order_relaxed);
    }
}
