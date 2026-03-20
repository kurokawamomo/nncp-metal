/*
 * online_trainer.h
 *
 * Online learning for the NNCP Metal Transformer.
 * Both the compress and decompress sides perform identical weight updates in
 * the same order, so no side-channel is needed to transmit weights.
 *
 * Usage pattern:
 *   OnlineTrainer* tr = online_trainer_create(device, ctx, 1e-4f, input_size);
 *   online_trainer_reset_session(tr, true);   // deterministic init
 *   for each segment (n_streams * seg_len tokens):
 *     float probs[256 * n_streams * seg_len];
 *     mps_transformer_execute_batch(..., probs);
 *     online_trainer_train_segment_batch(tr, seg_inputs, seg_targets, n_streams, seg_len);
 *   online_trainer_destroy(tr);
 */

#pragma once

#ifdef __OBJC__
#  import <Metal/Metal.h>
#  import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

#include "mps_transformer_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OnlineTrainer OnlineTrainer;

/**
 * Create an OnlineTrainer bound to the given context.
 * The trainer does NOT own ctx; ctx must outlive the trainer.
 * lr                : initial learning rate (e.g. 1e-4).
 * total_input_bytes : original file size in bytes; used to compute lr_decay_steps
 *                     (= total_input_bytes / seg_len).  Pass 0 for the default
 *                     156 250-step fallback (≈ 5 MB, original NNCP default profile).
 */
OnlineTrainer* online_trainer_create(id<MTLDevice>          device,
                                     MPSTransformerContext* ctx,
                                     float                  lr,
                                     size_t                 total_input_bytes);

/**
 * Force-flush any remaining buffered samples.
 * Call at the end of each chunk / session to drain the buffer.
 * A no-op when the buffer is empty.
 */
void online_trainer_flush(OnlineTrainer* trainer);

/**
 * Reset weights to a deterministic initial state.
 * Call at the start of every compress/decompress session so both sides
 * begin from identical weights.
 *
 * deterministic_init = true  → small fixed values (same every call)
 * deterministic_init = false → leave weights unchanged (for fine-tuning)
 */
void online_trainer_reset_session(OnlineTrainer* trainer, bool deterministic_init);

void online_trainer_destroy(OnlineTrainer* trainer);

/**
 * Segment-level training: run ONE backward pass over a complete segment.
 *
 * seg_inputs  : [n_streams * seg_len] context tokens (stream-major: inputs[s*T+t])
 * seg_targets : [n_streams * seg_len] true next-byte targets (same layout)
 * n_streams   : number of independent streams (must equal NUM_STREAMS = 16)
 * seg_len     : tokens per stream (must equal SEG_LEN = 32)
 *
 * Uses a causal Transformer training graph so token at position t attends to
 * positions 0..t within its stream (matching original NNCP seg_len semantics).
 * Falls back to batch training if dimensions don't match or graph unavailable.
 */
bool online_trainer_train_segment_batch(OnlineTrainer* trainer,
                                         const int32_t* seg_inputs,
                                         const int32_t* seg_targets,
                                         int            n_streams,
                                         int            seg_len);

#ifdef __cplusplus
}
#endif
