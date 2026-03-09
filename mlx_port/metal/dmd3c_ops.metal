// DMD3C Custom Metal Kernels
// Ports of the CUDA BpOps kernels for MLX Metal backend
//
// Two kernels:
// 1. dist_kernel4: K-nearest neighbor search (k=4) on sparse 2D points
// 2. conv2d_local: Position-dependent (spatially-varying) convolution

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Kernel 1: K-NN Distance Search (k=4)
// ============================================================================
// For each pixel (h,w), finds the 4 nearest sparse depth points.
// Outputs the neighbor indices and (dx, dy) offsets.
//
// Pc:     [B, 2, M]     - x,y coordinates of M valid sparse points
// IPCnum: [B, 2, 4, N]  - output: dx,dy offsets to 4 nearest neighbors
// args:   [B, 4, N]     - output: indices of 4 nearest neighbors
// N = H * W
kernel void dist_kernel4(
    device const float* Pc       [[buffer(0)]],
    device float*       IPCnum   [[buffer(1)]],
    device int*         args     [[buffer(2)]],
    constant int&       B        [[buffer(3)]],
    constant int&       M        [[buffer(4)]],
    constant int&       H        [[buffer(5)]],
    constant int&       W        [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int w = tid.x;
    int h = tid.y;
    int b = tid.z;

    if (h >= H || w >= W || b >= B) return;

    const int num = 4;
    const int Cc = 2;
    int N = H * W;

    // Track 4 best distances and their indices
    float best[4];
    int argbest[4];
    for (int i = 0; i < num; i++) {
        best[i] = 1e20;
        argbest[i] = 0;
    }

    // Linear scan through all M sparse points
    for (int m = 0; m < M; m++) {
        float dx = Pc[b * Cc * M + 0 * M + m] - float(w);
        float dy = Pc[b * Cc * M + 1 * M + m] - float(h);
        float dist = dx * dx + dy * dy;

        // Insertion sort into top-4
        for (int i = 0; i < num; i++) {
            if (best[i] >= dist) {
                for (int j = num - 1; j > i; j--) {
                    best[j] = best[j - 1];
                    argbest[j] = argbest[j - 1];
                }
                best[i] = dist;
                argbest[i] = m;
                break;
            }
        }
    }

    // Write outputs
    for (int i = 0; i < num; i++) {
        if (best[i] >= 1e20 && i > 0) {
            argbest[i] = argbest[i - 1];
        }
        args[b * num * N + i * N + h * W + w] = argbest[i];
        IPCnum[b * Cc * num * N + 0 * num * N + i * N + h * W + w] =
            Pc[b * Cc * M + 0 * M + argbest[i]] - float(w);
        IPCnum[b * Cc * num * N + 1 * num * N + i * N + h * W + w] =
            Pc[b * Cc * M + 1 * M + argbest[i]] - float(h);
    }
}


// ============================================================================
// Kernel 2: Position-Dependent (Local) Convolution - Forward
// ============================================================================
// Applies a spatially-varying KxK convolution where each spatial position
// has its own kernel weights.
//
// x: [B, C, H, W]          - input features
// y: [B, C*K*K, H, W]      - per-position kernel weights
// z: [B, C, H, W]          - output (same size as input)
kernel void conv2d_local_forward(
    device const float* x     [[buffer(0)]],
    device const float* y     [[buffer(1)]],
    device float*       z     [[buffer(2)]],
    constant int&       B     [[buffer(3)]],
    constant int&       C     [[buffer(4)]],
    constant int&       H     [[buffer(5)]],
    constant int&       W     [[buffer(6)]],
    constant int&       K     [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int col = tid.x;   // W dimension
    int row = tid.y;   // H dimension
    int ch  = tid.z;   // C dimension

    if (row >= H || col >= W || ch >= C) return;

    int half_k = (K - 1) / 2;

    for (int b = 0; b < B; b++) {
        float result = 0.0;

        for (int i = -half_k; i <= half_k; i++) {
            for (int j = -half_k; j <= half_k; j++) {
                int r = row + i;
                int c = col + j;

                if (r < 0 || r >= H || c < 0 || c >= W) continue;

                // x layout: [B, C, H, W]
                float x_val = x[b * C * H * W + ch * H * W + r * W + c];

                // y layout: [B, C*K*K, H, W] where K*K is indexed as (i+half_k)*K + (j+half_k)
                int ki = i + half_k;
                int kj = j + half_k;
                float y_val = y[b * C * K * K * H * W
                               + ch * K * K * H * W
                               + ki * K * H * W
                               + kj * H * W
                               + r * W + c];

                result += x_val * y_val;
            }
        }

        z[b * C * H * W + ch * H * W + row * W + col] = result;
    }
}
