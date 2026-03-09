/*
 * mps_matrix_optimizer.h
 *
 * MPS Matrix Multiplication Optimizer
 * Optimizes matrix operations for Apple Silicon GPU characteristics
 * Handles Threadgroup sizing and Batch Matrix Multiplication tuning
 */

#ifndef MPS_MATRIX_OPTIMIZER_H
#define MPS_MATRIX_OPTIMIZER_H

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MatrixDimensions {
    uint32_t M;
    uint32_t N;
    uint32_t K;
} MatrixDimensions;

typedef struct MPSMatrixOptimizerContext MPSMatrixOptimizerContext;

/**
 * Create Matrix Optimizer Context
 */
MPSMatrixOptimizerContext* mps_matrix_optimizer_create(id<MTLDevice> device);

/**
 * Destroy Context
 */
void mps_matrix_optimizer_destroy(MPSMatrixOptimizerContext* ctx);

/**
 * Perform Optimized Matrix Multiplication
 * C = alpha * A * B + beta * C
 * 
 * @param ctx Optimizer context
 * @param commandBuffer Command buffer to encode into
 * @param matrixA Left matrix [M x K]
 * @param matrixB Right matrix [K x N]
 * @param matrixC Result matrix [M x N]
 * @param dims Matrix dimensions
 * @param transposeA Transpose A
 * @param transposeB Transpose B
 * @param alpha Scalar multiplier
 * @param beta Scalar adder
 */
void mps_matrix_multiply(MPSMatrixOptimizerContext* ctx,
                        id<MTLCommandBuffer> commandBuffer,
                        id<MTLBuffer> matrixA,
                        id<MTLBuffer> matrixB,
                        id<MTLBuffer> matrixC,
                        MatrixDimensions dims,
                        bool transposeA,
                        bool transposeB,
                        float alpha,
                        float beta);

/**
 * Perform Batch Matrix Multiplication
 * 
 * @param batchCount Number of matrices in batch
 */
void mps_matrix_batch_multiply(MPSMatrixOptimizerContext* ctx,
                              id<MTLCommandBuffer> commandBuffer,
                              id<MTLBuffer> matrixA,
                              id<MTLBuffer> matrixB,
                              id<MTLBuffer> matrixC,
                              MatrixDimensions dims,
                              size_t batchCount,
                              bool transposeA,
                              bool transposeB,
                              float alpha,
                              float beta);

#ifdef __cplusplus
}
#endif

#endif // MPS_MATRIX_OPTIMIZER_H
