/*
 * mps_matrix_optimizer.mm
 *
 * Implementation of MPS Matrix Optimizer
 */

#import "mps_matrix_optimizer.h"
#import <Foundation/Foundation.h>

struct MPSMatrixOptimizerContext {
    id<MTLDevice> device;
    
    // Cache for matrix multiplication kernels
    // Key: Configuration hash, Value: MPSMatrixMultiplication object
    NSMutableDictionary* kernelCache;
};

MPSMatrixOptimizerContext* mps_matrix_optimizer_create(id<MTLDevice> device) {
    if (!device) return NULL;
    
    MPSMatrixOptimizerContext* ctx = new MPSMatrixOptimizerContext();
    ctx->device = device;
    ctx->kernelCache = [NSMutableDictionary dictionary];
    
    return ctx;
}

void mps_matrix_optimizer_destroy(MPSMatrixOptimizerContext* ctx) {
    if (ctx) {
        delete ctx;
    }
}

// Helper to create or retrieve cached matrix multiplication kernel
static MPSMatrixMultiplication* get_matmul_kernel(MPSMatrixOptimizerContext* ctx,
                                                 MatrixDimensions dims,
                                                 bool transposeA,
                                                 bool transposeB,
                                                 float alpha,
                                                 float beta) {
    NSString* key = [NSString stringWithFormat:@"%u-%u-%u-%d-%d-%.2f-%.2f", 
                     dims.M, dims.N, dims.K, transposeA, transposeB, alpha, beta];
    
    MPSMatrixMultiplication* kernel = ctx->kernelCache[key];
    if (!kernel) {
        kernel = [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device
                                                   transposeLeft:transposeA
                                                  transposeRight:transposeB
                                                      resultRows:dims.M
                                                   resultColumns:dims.N
                                                 interiorColumns:dims.K
                                                           alpha:alpha
                                                            beta:beta];
        ctx->kernelCache[key] = kernel;
    }
    return kernel;
}

void mps_matrix_multiply(MPSMatrixOptimizerContext* ctx,
                        id<MTLCommandBuffer> commandBuffer,
                        id<MTLBuffer> bufferA,
                        id<MTLBuffer> bufferB,
                        id<MTLBuffer> bufferC,
                        MatrixDimensions dims,
                        bool transposeA,
                        bool transposeB,
                        float alpha,
                        float beta) {
    if (!ctx || !commandBuffer) return;
    
    MPSMatrixMultiplication* kernel = get_matmul_kernel(ctx, dims, transposeA, transposeB, alpha, beta);
    
    // Create matrix descriptors
    // Note: Stride calculation assumes dense packing. 
    // Apple Silicon optimal stride is often aligned to 64 bytes.
    NSUInteger strideA = transposeA ? dims.M * sizeof(float) : dims.K * sizeof(float);
    NSUInteger strideB = transposeB ? dims.K * sizeof(float) : dims.N * sizeof(float);
    NSUInteger strideC = dims.N * sizeof(float);
    
    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufferA
                                             descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:transposeA ? dims.K : dims.M
                                                                                              columns:transposeA ? dims.M : dims.K
                                                                                             rowBytes:strideA
                                                                                             dataType:MPSDataTypeFloat32]];
                                                                                             
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufferB
                                             descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:transposeB ? dims.N : dims.K
                                                                                              columns:transposeB ? dims.K : dims.N
                                                                                             rowBytes:strideB
                                                                                             dataType:MPSDataTypeFloat32]];
                                                                                             
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufferC
                                             descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:dims.M
                                                                                              columns:dims.N
                                                                                             rowBytes:strideC
                                                                                             dataType:MPSDataTypeFloat32]];
    
    [kernel encodeToCommandBuffer:commandBuffer
                       leftMatrix:matA
                      rightMatrix:matB
                     resultMatrix:matC];
}

void mps_matrix_batch_multiply(MPSMatrixOptimizerContext* ctx,
                              id<MTLCommandBuffer> commandBuffer,
                              id<MTLBuffer> bufferA,
                              id<MTLBuffer> bufferB,
                              id<MTLBuffer> bufferC,
                              MatrixDimensions dims,
                              size_t batchCount,
                              bool transposeA,
                              bool transposeB,
                              float alpha,
                              float beta) {
    if (!ctx || !commandBuffer) return;
    
    // For batch multiplication, MPS has separate APIs or we can loop.
    // Ideally use MPSMatrixMultiplication with batch count if supported by specific API versions,
    // or use simple loop which MPS driver optimizes.
    // Apple Silicon optimizes command encoding, so loop overhead is low.
    
    MPSMatrixMultiplication* kernel = get_matmul_kernel(ctx, dims, transposeA, transposeB, alpha, beta);
    
    NSUInteger strideA = transposeA ? dims.M * sizeof(float) : dims.K * sizeof(float);
    NSUInteger strideB = transposeB ? dims.K * sizeof(float) : dims.N * sizeof(float);
    NSUInteger strideC = dims.N * sizeof(float);
    
    NSUInteger batchStrideA = dims.M * dims.K * sizeof(float);
    NSUInteger batchStrideB = dims.K * dims.N * sizeof(float);
    NSUInteger batchStrideC = dims.M * dims.N * sizeof(float);
    
    // If matrices are shared (broadcast), set stride to 0.
    // Assuming distinct matrices for now.
    
    for (size_t i = 0; i < batchCount; i++) {
        // Recreating MPSMatrix objects in loop is slightly heavy (descriptor alloc).
        // Optimization: Reuse descriptors or use lower-level API if available.
        // For this phase, simple loop is acceptable.
        
        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufferA
                                                     offset:i * batchStrideA
                                                 descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:transposeA ? dims.K : dims.M
                                                                                                  columns:transposeA ? dims.M : dims.K
                                                                                                 rowBytes:strideA
                                                                                                 dataType:MPSDataTypeFloat32]];
        
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufferB
                                                     offset:i * batchStrideB
                                                 descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:transposeB ? dims.N : dims.K
                                                                                                  columns:transposeB ? dims.K : dims.N
                                                                                                 rowBytes:strideB
                                                                                                 dataType:MPSDataTypeFloat32]];
        
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufferC
                                                     offset:i * batchStrideC
                                                 descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:dims.M
                                                                                                  columns:dims.N
                                                                                                 rowBytes:strideC
                                                                                                 dataType:MPSDataTypeFloat32]];
        
        [kernel encodeToCommandBuffer:commandBuffer
                           leftMatrix:matA
                          rightMatrix:matB
                         resultMatrix:matC];
    }
}
