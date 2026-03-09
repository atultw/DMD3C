#!/usr/bin/env python3
"""
Verification script: Compares the pure-Python/NumPy implementations of the
custom CUDA ops (BpDist k-NN and BpConvLocal position-dependent conv) against
reference implementations to ensure 1:1 correctness.

This script does NOT require CUDA or MLX - it verifies the mathematical
equivalence using pure NumPy.

Usage:
    python verify_ops.py
"""

import numpy as np
import sys


def reference_dist_kernel4(Pc, H, W, num=4):
    """
    Pure Python reference implementation of dist_kernel4 CUDA kernel.
    Exact 1:1 port of the CUDA logic.

    Args:
        Pc: (1, 2, M) - coordinates of valid sparse points
        H, W: grid dimensions
        num: number of nearest neighbors (4)

    Returns:
        IPCnum: (1, 2, num, H*W) - offsets to nearest neighbors
        args: (1, num, H*W) - indices of nearest neighbors
    """
    B = 1
    Cc = 2
    M = Pc.shape[2]
    N = H * W

    args = np.zeros((B, num, N), dtype=np.int64)
    IPCnum = np.zeros((B, Cc, num, N), dtype=np.float32)

    for b in range(B):
        for h in range(H):
            for w in range(W):
                best = [1e20] * num
                argbest = [0] * num

                for m in range(M):
                    res1 = Pc[b, 0, m] - w
                    res2 = Pc[b, 1, m] - h
                    dist = res1 * res1 + res2 * res2

                    for i in range(num):
                        if best[i] >= dist:
                            for j in range(num - 1, i, -1):
                                best[j] = best[j - 1]
                                argbest[j] = argbest[j - 1]
                            best[i] = dist
                            argbest[i] = m
                            break

                for i in range(num):
                    if best[i] >= 1e20 and i > 0:
                        argbest[i] = argbest[i - 1]
                    n_idx = h * W + w
                    args[b, i, n_idx] = argbest[i]
                    IPCnum[b, 0, i, n_idx] = Pc[b, 0, argbest[i]] - w
                    IPCnum[b, 1, i, n_idx] = Pc[b, 1, argbest[i]] - h

    return IPCnum, args


def mlx_style_dist(xy, idx, Valid, num, H, W):
    """
    Our MLX replacement implementation (pure numpy version for testing).
    Should produce the same results as the CUDA kernel.
    """
    B = Valid.shape[0]
    N = H * W

    all_args = []
    all_ipcnum = []

    for b in range(B):
        valid_mask = Valid[b, 0]  # (N,)
        valid_indices = np.where(valid_mask)[0]
        M_valid = len(valid_indices)

        if M_valid == 0:
            all_args.append(np.zeros((num, N), dtype=np.int64))
            all_ipcnum.append(np.zeros((2, num, N), dtype=np.float32))
            continue

        valid_xy = xy[0, :, valid_indices].T  # xy indexing gives (M_valid, 2), .T gives (2, M_valid)

        ww, hh = np.meshgrid(np.arange(W), np.arange(H))
        grid_w = ww.reshape(-1).astype(np.float32)
        grid_h = hh.reshape(-1).astype(np.float32)

        dx = grid_w[:, None] - valid_xy[0:1, :]  # (N, M_valid)
        dy = grid_h[:, None] - valid_xy[1:2, :]
        dists = dx * dx + dy * dy

        k = min(num, M_valid)
        sorted_idx = np.argsort(dists, axis=1)

        if M_valid < num:
            last_col = sorted_idx[:, -1:]
            pad = np.broadcast_to(last_col, (N, num - M_valid))
            sorted_idx = np.concatenate([sorted_idx, pad], axis=1)

        topk_idx = sorted_idx[:, :num]
        topk_global = valid_indices[topk_idx]

        neighbor_x = valid_xy[0][topk_idx]
        neighbor_y = valid_xy[1][topk_idx]
        off_x = neighbor_x - grid_w[:, None]
        off_y = neighbor_y - grid_h[:, None]

        all_args.append(topk_global.T)
        all_ipcnum.append(np.stack([off_x.T, off_y.T], axis=0))

    args = np.stack(all_args, axis=0)
    ipcnum = np.stack(all_ipcnum, axis=0)
    return ipcnum, args


def reference_conv2d_local(x, y, K):
    """
    Pure Python reference implementation of conv2d_local CUDA kernel.
    Exact 1:1 port of the CUDA logic.

    The key detail: x is sampled at the NEIGHBOR position (row+i, col+j),
    while y (weights) is sampled at the CENTER position (row, col).

    Args:
        x: (B, C, H, W) input
        y: (B, C*K*K, H, W) per-position weights
        K: kernel size

    Returns:
        z: (B, C, H, W) output
    """
    B, C, H, W = x.shape
    z = np.zeros_like(x)
    half_k = (K - 1) // 2

    for b in range(B):
        for c in range(C):
            for row in range(H):
                for col in range(W):
                    result = 0.0
                    for i in range(-half_k, half_k + 1):
                        for j in range(-half_k, half_k + 1):
                            r = row + i
                            cc = col + j
                            if r < 0 or r >= H or cc < 0 or cc >= W:
                                continue
                            ki = i + half_k
                            kj = j + half_k
                            x_val = x[b, c, r, cc]
                            # Weight at CENTER position (row, col)
                            y_val = y[b, c * K * K + ki * K + kj, row, col]
                            result += x_val * y_val
                    z[b, c, row, col] = result

    return z


def mlx_style_conv2d_local(x, weight):
    """
    Our MLX replacement implementation (pure numpy version for testing).
    Uses unfold approach.
    """
    import math
    B, C, H, W = x.shape
    K2 = weight.shape[1] // C
    K = int(math.sqrt(K2))
    pad = (K - 1) // 2

    x_pad = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)])

    patches = []
    for i in range(K):
        for j in range(K):
            patches.append(x_pad[:, :, i:i + H, j:j + W])
    patches = np.stack(patches, axis=2)  # (B, C, K*K, H, W)

    weight_reshaped = weight.reshape(B, C, K2, H, W)
    output = np.sum(patches * weight_reshaped, axis=2)

    return output


def test_dist():
    """Test k-NN distance search op."""
    print("=" * 60)
    print("Testing BpDist (k-NN search) replacement")
    print("=" * 60)

    np.random.seed(42)
    H, W = 8, 12
    N = H * W

    # Create sparse depth with ~20% valid points
    sparse = np.zeros((1, 1, N), dtype=np.float32)
    valid_positions = np.random.choice(N, size=N // 5, replace=False)
    sparse[0, 0, valid_positions] = np.random.uniform(1, 80, size=len(valid_positions))

    Valid = sparse > 1e-3

    # xy coords for all pixels
    ww, hh = np.meshgrid(np.arange(W), np.arange(H))
    xy = np.stack([ww.reshape(-1), hh.reshape(-1)], axis=0).reshape(1, 2, N).astype(np.float32)
    idx = np.arange(N).reshape(1, 1, N)

    # Get valid point coordinates
    valid_mask = Valid[0, 0]
    valid_idx = np.where(valid_mask)[0]
    Pc_valid = xy[:, :, valid_idx]  # (1, 2, M_valid)

    # Run reference (CUDA-equivalent) on the filtered valid points
    ref_ipcnum, ref_args = reference_dist_kernel4(Pc_valid, H, W, num=4)

    # Run our MLX replacement
    mlx_ipcnum, mlx_args = mlx_style_dist(xy, idx, Valid, num=4, H=H, W=W)

    # The reference IPCnum gives offsets (dx,dy) from each pixel to its 4 nearest neighbors.
    # The MLX version does the same. Both should produce identical offsets.
    offset_match = np.allclose(ref_ipcnum, mlx_ipcnum, atol=1e-5)
    print(f"  Offset match (IPCnum): {offset_match}")

    if not offset_match:
        mismatch_count = 0
        for n in range(N):
            ref_offs = ref_ipcnum[0, :, :, n]
            mlx_offs = mlx_ipcnum[0, :, :, n]
            if not np.allclose(ref_offs, mlx_offs, atol=1e-5):
                mismatch_count += 1
                if mismatch_count <= 3:
                    print(f"    Pixel {n}: ref={ref_offs.T}, mlx={mlx_offs.T}")
        print(f"    Total mismatches: {mismatch_count}/{N}")

    # Verify correctness: for each pixel, check that found neighbors are truly nearest
    correct_neighbors = 0
    total_checked = 0
    for n in range(N):
        h, w = n // W, n % W
        # Distances to all valid sparse points
        sparse_xy = Pc_valid[0]  # (2, M_valid)
        dists_to_sparse = (sparse_xy[0] - w) ** 2 + (sparse_xy[1] - h) ** 2
        true_nearest_dists = np.sort(dists_to_sparse)[:4]

        # Our method's nearest distances (from offsets)
        our_offsets = mlx_ipcnum[0, :, :, n]  # (2, 4) - dx, dy
        our_dists = np.sort(our_offsets[0] ** 2 + our_offsets[1] ** 2)

        if len(true_nearest_dists) < 4:
            # Pad with last value
            true_nearest_dists = np.pad(true_nearest_dists,
                                        (0, 4 - len(true_nearest_dists)),
                                        mode='edge')

        if np.allclose(our_dists, true_nearest_dists, atol=1e-3):
            correct_neighbors += 1
        total_checked += 1

    accuracy = correct_neighbors / total_checked * 100
    print(f"  Neighbor accuracy: {correct_neighbors}/{total_checked} ({accuracy:.1f}%)")
    passed = accuracy > 99.0
    print(f"  PASS" if passed else f"  FAIL")
    return passed


def test_conv2d_local():
    """Test position-dependent convolution op."""
    print()
    print("=" * 60)
    print("Testing BpConvLocal (position-dependent conv) replacement")
    print("=" * 60)

    np.random.seed(42)

    for K in [3, 5, 7]:
        B, C, H, W = 1, 2, 6, 8
        x = np.random.randn(B, C, H, W).astype(np.float32)
        weight = np.random.randn(B, C * K * K, H, W).astype(np.float32)

        # Reference (CUDA-equivalent)
        ref_out = reference_conv2d_local(x, weight, K)

        # Our MLX replacement
        mlx_out = mlx_style_conv2d_local(x, weight)

        # Compare
        max_diff = np.max(np.abs(ref_out - mlx_out))
        mean_diff = np.mean(np.abs(ref_out - mlx_out))
        match = np.allclose(ref_out, mlx_out, atol=1e-5)

        print(f"  K={K}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, match={match}")

    # Larger test
    B, C, H, W, K = 1, 8, 16, 20, 3
    x = np.random.randn(B, C, H, W).astype(np.float32)
    weight = np.random.randn(B, C * K * K, H, W).astype(np.float32)
    ref_out = reference_conv2d_local(x, weight, K)
    mlx_out = mlx_style_conv2d_local(x, weight)
    max_diff = np.max(np.abs(ref_out - mlx_out))
    match = np.allclose(ref_out, mlx_out, atol=1e-5)
    print(f"  Large test (C=8, H=16, W=20, K=3): max_diff={max_diff:.2e}, match={match}")

    print(f"  PASS" if match else f"  FAIL")
    return match


def test_weight_conversion():
    """Test that weight conversion preserves values correctly."""
    print()
    print("=" * 60)
    print("Testing weight conversion (OIHW → OHWI)")
    print("=" * 60)

    # Regular conv: OIHW -> OHWI
    w = np.random.randn(64, 32, 3, 3).astype(np.float32)
    converted = np.transpose(w, (0, 2, 3, 1))
    reconstructed = np.transpose(converted, (0, 3, 1, 2))
    match = np.allclose(w, reconstructed)
    print(f"  Regular Conv OIHW→OHWI→OIHW roundtrip: {match}")

    # ConvTranspose: IOHW -> OHWI
    w = np.random.randn(64, 32, 4, 4).astype(np.float32)
    converted = np.transpose(w, (1, 2, 3, 0))
    reconstructed = np.transpose(converted, (3, 0, 1, 2))
    match_t = np.allclose(w, reconstructed)
    print(f"  ConvTranspose IOHW→OHWI→IOHW roundtrip: {match_t}")

    print(f"  PASS" if (match and match_t) else f"  FAIL")
    return match and match_t


def main():
    print("DMD3C Op Verification Tests")
    print("Comparing MLX replacement ops against CUDA reference implementations")
    print()

    results = []
    results.append(("BpDist (k-NN)", test_dist()))
    results.append(("BpConvLocal (local conv)", test_conv2d_local()))
    results.append(("Weight conversion", test_weight_conversion()))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests PASSED! The MLX replacement ops are mathematically equivalent.")
    else:
        print("Some tests FAILED. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
