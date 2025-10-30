#include <metal_stdlib>
using namespace metal;

// Transformer attention mechanism
kernel void transformer_attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= seq_len || gid.y >= seq_len) return;
    
    // Compute attention scores (Q * K^T)
    float score = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        score += query[gid.x * head_dim + i] * key[gid.y * head_dim + i];
    }
    score *= scale;
    
    // Note: This is a simplified version. Full implementation would require
    // proper softmax across the sequence dimension and then multiplication with V
    float exp_score = exp(score);
    
    // Compute output (simplified - should be done after softmax)
    for (uint i = 0; i < head_dim; i++) {
        atomic_fetch_add_explicit(
            (device atomic<float>*)&output[gid.x * head_dim + i],
            exp_score * value[gid.y * head_dim + i],
            memory_order_relaxed
        );
    }
}

// Layer normalization
kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    // Note: This is a simplified per-element version
    // Full implementation would require parallel reduction for mean/variance
    
    // For now, assume mean and variance are pre-computed or use approximation
    float mean = 0.0f;
    float var = 1.0f;  // Simplified
    
    // Normalize and scale
    float normalized = (input[gid] - mean) / sqrt(var + eps);
    output[gid] = gamma[gid] * normalized + beta[gid];
}

// Feed-forward network layer
kernel void feed_forward(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& input_size [[buffer(4)]],
    constant uint& output_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_size) return;
    
    float sum = bias[gid];
    for (uint i = 0; i < input_size; i++) {
        sum += input[i] * weights[gid * input_size + i];
    }
    output[gid] = sum;
}

// LSTM cell computation
kernel void lstm_cell(
    device const float* input [[buffer(0)]],
    device const float* hidden_state [[buffer(1)]],
    device const float* cell_state [[buffer(2)]],
    device const float* weights_ih [[buffer(3)]],  // input-to-hidden weights
    device const float* weights_hh [[buffer(4)]],  // hidden-to-hidden weights
    device const float* bias [[buffer(5)]],
    device float* new_hidden [[buffer(6)]],
    device float* new_cell [[buffer(7)]],
    constant uint& input_size [[buffer(8)]],
    constant uint& hidden_size [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= hidden_size) return;
    
    // Compute gates: input, forget, cell, output
    float input_gate = 0.0f, forget_gate = 0.0f, cell_gate = 0.0f, output_gate = 0.0f;
    
    // Input-to-hidden transformation
    for (uint i = 0; i < input_size; i++) {
        input_gate += input[i] * weights_ih[gid * input_size + i];
        forget_gate += input[i] * weights_ih[(hidden_size + gid) * input_size + i];
        cell_gate += input[i] * weights_ih[(2 * hidden_size + gid) * input_size + i];
        output_gate += input[i] * weights_ih[(3 * hidden_size + gid) * input_size + i];
    }
    
    // Hidden-to-hidden transformation
    for (uint i = 0; i < hidden_size; i++) {
        input_gate += hidden_state[i] * weights_hh[gid * hidden_size + i];
        forget_gate += hidden_state[i] * weights_hh[(hidden_size + gid) * hidden_size + i];
        cell_gate += hidden_state[i] * weights_hh[(2 * hidden_size + gid) * hidden_size + i];
        output_gate += hidden_state[i] * weights_hh[(3 * hidden_size + gid) * hidden_size + i];
    }
    
    // Add bias and apply activations
    input_gate = 1.0f / (1.0f + exp(-(input_gate + bias[gid])));  // sigmoid
    forget_gate = 1.0f / (1.0f + exp(-(forget_gate + bias[hidden_size + gid])));  // sigmoid
    cell_gate = tanh(cell_gate + bias[2 * hidden_size + gid]);  // tanh
    output_gate = 1.0f / (1.0f + exp(-(output_gate + bias[3 * hidden_size + gid])));  // sigmoid
    
    // Update cell state
    float new_cell_val = forget_gate * cell_state[gid] + input_gate * cell_gate;
    new_cell[gid] = new_cell_val;
    
    // Update hidden state
    new_hidden[gid] = output_gate * tanh(new_cell_val);
}

// Position encoding for transformer
kernel void positional_encoding(
    device float* encoding [[buffer(0)]],
    constant uint& seq_len [[buffer(1)]],
    constant uint& d_model [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= seq_len || gid.y >= d_model) return;
    
    uint pos = gid.x;
    uint i = gid.y;
    
    if (i % 2 == 0) {
        // Even indices: sin
        float angle = pos / pow(10000.0f, (float)(i) / d_model);
        encoding[pos * d_model + i] = sin(angle);
    } else {
        // Odd indices: cos
        float angle = pos / pow(10000.0f, (float)(i - 1) / d_model);
        encoding[pos * d_model + i] = cos(angle);
    }
}

// Embedding lookup
kernel void embedding_lookup(
    device const uint* input_ids [[buffer(0)]],
    device const float* embedding_table [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant uint& embedding_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= embedding_dim) return;
    
    uint token_id = input_ids[gid.y];
    if (token_id >= vocab_size) return;
    
    output[gid.y * embedding_dim + gid.x] = embedding_table[token_id * embedding_dim + gid.x];
}
