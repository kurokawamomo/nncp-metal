/*
 * online_trainer.h
 *
 * Online learning for the NNCP Metal Transformer.
 * Both the compress and decompress sides perform identical weight updates in
 * the same order, so no side-channel is needed to transmit weights.
 *
 * Usage pattern:
 *   OnlineTrainer* tr = online_trainer_create(device, ctx, 1e-3f);
 *   online_trainer_reset_session(tr, true);   // deterministic init
 *   for each byte b:
 *     float probs[256];
 *     mps_transformer_execute(ctx, &prev_tok, 1, 1, probs);
 *     // encode/decode b using probs ...
 *     online_trainer_step(tr, prev_tok, b);   // update weights
 *   online_trainer_destroy(tr);
 */

#pragma once

#ifdef __OBJC__
#  import <Metal/Metal.h>
#  import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

#include "../mps_optimized/mps_transformer_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OnlineTrainer OnlineTrainer;

/**
 * Create an OnlineTrainer bound to the given context.
 * The trainer does NOT own ctx; ctx must outlive the trainer.
 * lr : SGD learning rate (e.g. 1e-3).
 */
OnlineTrainer* online_trainer_create(id<MTLDevice>          device,
                                     MPSTransformerContext* ctx,
                                     float                  lr);

/**
 * Perform one training step.
 *
 * input_token : the token that was fed to the model this step (int32, 0-255).
 * true_byte   : the correct next byte (0-255).
 *
 * Internally runs: forward → cross_entropy_loss → gradients → SGD update.
 * Returns false on GPU/graph error.
 */
bool online_trainer_step(OnlineTrainer* trainer,
                         int32_t        input_token,
                         int            true_byte);

/**
 * Buffered step: accumulates (input_token, true_byte) pairs in an internal
 * buffer of size TRAIN_BATCH_SIZE (= 8).  When the buffer is full, a single
 * batched backward pass is executed automatically (one GPU call for 8 bytes
 * instead of 8 separate calls).
 *
 * Compress and decompress must call this in exactly the same order so that
 * weight updates remain symmetric.
 */
void online_trainer_step_buffered(OnlineTrainer* trainer,
                                   int32_t        input_token,
                                   int32_t        true_byte);

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

#ifdef __cplusplus
}
#endif
