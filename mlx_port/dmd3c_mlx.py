"""
DMD3C MLX Model Implementation

A complete 1:1 port of the DMD3C depth completion model from PyTorch to MLX.
This replaces the CUDA-dependent BpOps with pure MLX/numpy operations suitable
for Apple Silicon inference.

The two CUDA custom ops are replaced as follows:
- BpDist (k-NN search): Pure MLX distance computation + topk
- BpConvLocal (position-dependent convolution): MLX unfold + element-wise multiply + sum

Usage:
    import mlx.core as mx
    from dmd3c_mlx import DMD3CModel

    model = DMD3CModel()
    model.load_weights("mlx_weights/dmd3c.safetensors")
    depth = model.predict(image, sparse_depth, K_cam)
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ============================================================================
# Pure MLX replacements for CUDA custom ops
# ============================================================================

def bpdist_mlx(xy: mx.array, idx: mx.array, Valid: mx.array,
               num: int, H: int, W: int) -> Tuple[mx.array, mx.array]:
    """
    Pure MLX replacement for BpDist CUDA kernel.
    Finds the k=4 nearest sparse depth points for each pixel.

    Args:
        xy: (1, 2, M) - x,y coordinates of all pixels (used as base grid)
        idx: (1, 1, N) - indices of all pixels
        Valid: (B, 1, N) - boolean mask of valid sparse depth pixels
        num: number of neighbors (4)
        H, W: spatial dimensions

    Returns:
        IPCnum: (B, 2, num, N) - dx,dy offsets to nearest neighbors
        args: (B, num, N) - global indices of nearest neighbors
    """
    B = Valid.shape[0]
    N = H * W

    args_list = []
    IPCnum_list = []

    # Create grid of all pixel coordinates (shared across batches)
    ww = mx.arange(W)
    hh = mx.arange(H)
    grid_w, grid_h = mx.meshgrid(ww, hh, indexing='xy')  # both (H, W)
    grid_w = grid_w.reshape(-1).astype(mx.float32)  # (N,)
    grid_h = grid_h.reshape(-1).astype(mx.float32)  # (N,)

    for b in range(B):
        valid_mask = Valid[b, 0]  # (N,)

        # Get coordinates of valid (sparse) points
        valid_indices = mx.where(valid_mask)[0]  # indices into the N-length array
        M_valid = valid_indices.shape[0]

        if M_valid == 0:
            # No valid points - return zeros
            args_b = mx.zeros((num, N), dtype=mx.int32)
            IPCnum_b = mx.zeros((2, num, N), dtype=mx.float32)
            args_list.append(args_b)
            IPCnum_list.append(IPCnum_b)
            continue

        # valid point xy coords: extract from xy
        # xy is (1, 2, N) where N = H*W (all pixels)
        # Gather valid point coordinates - shape should be (2, M_valid)
        valid_x = xy[0, 0, valid_indices]  # (M_valid,)
        valid_y = xy[0, 1, valid_indices]  # (M_valid,)
        valid_xy = mx.stack([valid_x, valid_y], axis=0)  # (2, M_valid)

        # Compute distances: for each of N pixels, distance to each of M_valid sparse pts
        # pixel coords: (N, 1), sparse coords: (1, M_valid)
        dx = grid_w[:, None] - valid_xy[0:1, :]  # (N, M_valid)
        dy = grid_h[:, None] - valid_xy[1:2, :]  # (N, M_valid)
        dists = dx * dx + dy * dy  # (N, M_valid)

        # Find top-k nearest (smallest distances)
        k = min(num, M_valid)
        # MLX doesn't have topk for smallest, so use negative + topk or argsort
        if M_valid <= num:
            # If fewer valid points than k, just use all of them
            sorted_idx = mx.argsort(dists, axis=1)  # (N, M_valid)
            # Pad to num
            if M_valid < num:
                last_col = sorted_idx[:, -1:]  # (N, 1)
                pad = mx.broadcast_to(last_col, (N, num - M_valid))
                sorted_idx = mx.concatenate([sorted_idx, pad], axis=1)
            topk_idx = sorted_idx[:, :num]  # (N, num)
        else:
            # Use argpartition for efficiency if available, else argsort
            sorted_idx = mx.argsort(dists, axis=1)  # (N, M_valid)
            topk_idx = sorted_idx[:, :num]  # (N, num)

        # Map back to global indices
        topk_global = valid_indices[topk_idx]  # (N, num)

        # Compute offsets (dx, dy) for each neighbor
        # neighbor xy coords
        neighbor_x = valid_xy[0][topk_idx]  # (N, num)
        neighbor_y = valid_xy[1][topk_idx]  # (N, num)
        off_x = neighbor_x - grid_w[:, None]  # (N, num)
        off_y = neighbor_y - grid_h[:, None]  # (N, num)

        # Reshape to (num, N) and (2, num, N)
        args_b = topk_global.T  # (num, N)
        IPCnum_b = mx.stack([off_x.T, off_y.T], axis=0)  # (2, num, N)

        args_list.append(args_b)
        IPCnum_list.append(IPCnum_b)

    args = mx.stack(args_list, axis=0)  # (B, num, N)
    IPCnum = mx.stack(IPCnum_list, axis=0)  # (B, 2, num, N)

    return IPCnum, args


def conv2d_local_mlx(x: mx.array, weight: mx.array) -> mx.array:
    """
    Pure MLX replacement for BpConvLocal CUDA kernel.
    Position-dependent (spatially-varying) convolution.

    Args:
        x: (B, C, H, W) - input feature map
        weight: (B, C*K*K, H, W) - per-position convolution kernels

    Returns:
        output: (B, C, H, W) - convolved output
    """
    B, C, H, W = x.shape
    K2 = weight.shape[1] // C
    K = int(math.sqrt(K2))
    pad = (K - 1) // 2

    # Pad input
    x_pad = mx.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)])  # (B, C, H+2p, W+2p)

    # Unfold: extract KxK patches for each spatial position
    # Result should be (B, C, K*K, H, W)
    patches = []
    for i in range(K):
        for j in range(K):
            patches.append(x_pad[:, :, i:i + H, j:j + W])
    patches = mx.stack(patches, axis=2)  # (B, C, K*K, H, W)

    # Reshape weight to (B, C, K*K, H, W)
    weight_reshaped = weight.reshape(B, C, K2, H, W)

    # Element-wise multiply and sum over kernel dimension
    output = mx.sum(patches * weight_reshaped, axis=2)  # (B, C, H, W)

    return output


# ============================================================================
# Network Building Blocks
# ============================================================================

class Conv2dBnRelu(nn.Module):
    """Conv2d + BatchNorm + ReLU block (corresponds to Basic2d in PyTorch)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False,
                 use_bn: bool = True, act: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=not use_bn if not bias else bias)
        self.bn = nn.BatchNorm(out_ch) if use_bn else None
        self.act = act

    def __call__(self, x):
        # MLX conv2d expects NHWC
        x = self.conv(x)
        if self.bn is not None:
            # BatchNorm operates on last dim in MLX (NHWC, so C is last)
            x = self.bn(x)
        if self.act == "relu":
            x = nn.relu(x)
        elif self.act == "gelu":
            x = nn.gelu(x)
        elif self.act == "sigmoid":
            x = mx.sigmoid(x)
        elif self.act == "identity":
            pass
        return x


class Conv2dTransposeBnRelu(nn.Module):
    """ConvTranspose2d + BatchNorm + ReLU (corresponds to Basic2dTrans)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 4,
                 stride: int = 2, padding: int = 1, use_bn: bool = True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=not use_bn)
        self.bn = nn.BatchNorm(out_ch) if use_bn else None

    def __call__(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = nn.relu(x)
        return x


class BasicBlock(nn.Module):
    """ResNet BasicBlock with DropPath."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample=None, drop_path: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.downsample = downsample
        self.drop_path_rate = drop_path

    def __call__(self, x):
        identity = x
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        # DropPath: at inference, this is identity
        out = out + identity
        out = nn.relu(out)
        return out


class DownsampleBlock(nn.Module):
    """1x1 conv + BN for residual downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm(out_ch)

    def __call__(self, x):
        return self.bn(self.conv(x))


class UpCC(nn.Module):
    """Upsample + Concatenate + Conv block."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.upf = Conv2dTransposeBnRelu(in_ch, out_ch)
        self.conv = Conv2dBnRelu(mid_ch + out_ch, out_ch, kernel_size=3, padding=1)

    def __call__(self, x, y):
        out = self.upf(x)
        out = mx.concatenate([out, y], axis=-1)  # NHWC: concat on channel dim
        out = self.conv(out)
        return out


class GenKernel(nn.Module):
    """Generates spatially-varying convolution kernels for CSPN."""

    def __init__(self, in_ch: int, pk: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.conv0 = Conv2dBnRelu(in_ch, in_ch, kernel_size=3, padding=1, act="relu")
        self.conv1 = Conv2dBnRelu(in_ch, pk * pk - 1, kernel_size=3, padding=1, act="identity")

    def __call__(self, fout):
        weight = self.conv1(self.conv0(fout))
        weight_sum = mx.sum(mx.abs(weight), axis=-1, keepdims=True)
        weight = weight / (weight_sum + self.eps)
        weight_mid = 1.0 - mx.sum(weight, axis=-1, keepdims=True)
        half = weight.shape[-1] // 2
        weight_pre = weight[..., :half]
        weight_post = weight[..., half:]
        weight = mx.concatenate([weight_pre, weight_mid, weight_post], axis=-1)
        return weight


class CSPN(nn.Module):
    """
    CSPN++ implementation - Convolutional Spatial Propagation Network.
    Uses position-dependent convolution for iterative depth refinement.
    """

    def __init__(self, in_ch: int, pt: int, eps: float = 1e-6):
        super().__init__()
        self.pt = pt
        self.weight3x3 = GenKernel(in_ch, 3, eps=eps)
        self.weight5x5 = GenKernel(in_ch, 5, eps=eps)
        self.weight7x7 = GenKernel(in_ch, 7, eps=eps)
        self.convmask0 = Conv2dBnRelu(in_ch, in_ch, kernel_size=3, padding=1, act="relu")
        self.convmask1 = Conv2dBnRelu(in_ch, 3, kernel_size=3, padding=1, use_bn=False, bias=True, act="sigmoid")
        self.convck0 = Conv2dBnRelu(in_ch, in_ch, kernel_size=3, padding=1, act="relu")
        self.convck1 = Conv2dBnRelu(in_ch, 3, kernel_size=3, padding=1, use_bn=False, bias=True, act="identity")
        self.convct0 = Conv2dBnRelu(in_ch + 3, in_ch, kernel_size=3, padding=1, act="relu")
        self.convct1 = Conv2dBnRelu(in_ch, 3, kernel_size=3, padding=1, use_bn=False, bias=True, act="identity")

    def _apply_local_conv(self, hn_nchw, weight_nhwc):
        """Apply position-dependent convolution.
        hn_nchw: (B, C, H, W) in NCHW format (for the local conv op)
        weight_nhwc: (B, H, W, K*K) in NHWC format from GenKernel output
        Returns: (B, C, H, W) in NCHW
        """
        B, C, H, W = hn_nchw.shape
        K2 = weight_nhwc.shape[-1]
        # weight needs to be (B, C*K*K, H, W) for conv2d_local_mlx
        # GenKernel outputs per-channel weights? No - it outputs (B, H, W, K*K) shared across channels
        # In the original PyTorch code, bpconvlocal takes (B,C,H,W) and (B,C*K*K,H,W)
        # But GenKernel outputs (B, K*K, H, W) in PyTorch - the same kernel for all channels
        # Need to expand: repeat K*K weights C times

        # weight_nhwc is (B, H, W, K*K), convert to (B, K*K, H, W)
        weight_nchw = mx.transpose(weight_nhwc, (0, 3, 1, 2))
        # Expand to (B, C*K*K, H, W) by repeating for each channel
        weight_expanded = mx.repeat(weight_nchw[:, None, :, :, :], C, axis=1)  # (B, C, K*K, H, W)
        weight_expanded = weight_expanded.reshape(B, C * K2, H, W)

        return conv2d_local_mlx(hn_nchw, weight_expanded)

    def __call__(self, fout, hn, h0):
        """
        fout: (B, H, W, C) - features in NHWC
        hn: (B, H, W, 1) - current depth estimate in NHWC
        h0: (B, H, W, 1) - sparse depth (guidance) in NHWC
        Returns: (B, H, W, 1) refined depth in NHWC
        """
        weight3x3 = self.weight3x3(fout)   # (B, H, W, 9)
        weight5x5 = self.weight5x5(fout)   # (B, H, W, 25)
        weight7x7 = self.weight7x7(fout)   # (B, H, W, 49)

        mask_all = self.convmask1(self.convmask0(fout))  # (B, H, W, 3)
        h0_valid = (h0 > 1e-3).astype(mx.float32)
        mask_all = mask_all * h0_valid

        mask3x3 = mask_all[..., 0:1]
        mask5x5 = mask_all[..., 1:2]
        mask7x7 = mask_all[..., 2:3]

        conf_all = mx.softmax(self.convck1(self.convck0(fout)), axis=-1)  # (B, H, W, 3)
        conf3x3 = conf_all[..., 0:1]
        conf5x5 = conf_all[..., 1:2]
        conf7x7 = conf_all[..., 2:3]

        # Convert to NCHW for local conv
        B, Hh, Ww, _ = hn.shape
        hn3x3_nchw = mx.transpose(hn, (0, 3, 1, 2))
        hn5x5_nchw = mx.transpose(hn, (0, 3, 1, 2))
        hn7x7_nchw = mx.transpose(hn, (0, 3, 1, 2))
        h0_nchw_val = mx.transpose(h0, (0, 3, 1, 2))

        # Masks in NCHW
        mask3x3_nchw = mx.transpose(mask3x3, (0, 3, 1, 2))
        mask5x5_nchw = mx.transpose(mask5x5, (0, 3, 1, 2))
        mask7x7_nchw = mx.transpose(mask7x7, (0, 3, 1, 2))

        hns_nhwc = [hn]  # collect intermediate results in NHWC

        for i in range(self.pt):
            hn3x3_nchw = (1.0 - mask3x3_nchw) * self._apply_local_conv(hn3x3_nchw, weight3x3) + mask3x3_nchw * h0_nchw_val
            hn5x5_nchw = (1.0 - mask5x5_nchw) * self._apply_local_conv(hn5x5_nchw, weight5x5) + mask5x5_nchw * h0_nchw_val
            hn7x7_nchw = (1.0 - mask7x7_nchw) * self._apply_local_conv(hn7x7_nchw, weight7x7) + mask7x7_nchw * h0_nchw_val

            if i == self.pt // 2 - 1:
                # Intermediate fusion
                conf3x3_nchw = mx.transpose(conf3x3, (0, 3, 1, 2))
                conf5x5_nchw = mx.transpose(conf5x5, (0, 3, 1, 2))
                conf7x7_nchw = mx.transpose(conf7x7, (0, 3, 1, 2))
                mid = conf3x3_nchw * hn3x3_nchw + conf5x5_nchw * hn5x5_nchw + conf7x7_nchw * hn7x7_nchw
                hns_nhwc.append(mx.transpose(mid, (0, 2, 3, 1)))

        # Final fusion
        conf3x3_nchw = mx.transpose(conf3x3, (0, 3, 1, 2))
        conf5x5_nchw = mx.transpose(conf5x5, (0, 3, 1, 2))
        conf7x7_nchw = mx.transpose(conf7x7, (0, 3, 1, 2))
        final = conf3x3_nchw * hn3x3_nchw + conf5x5_nchw * hn5x5_nchw + conf7x7_nchw * hn7x7_nchw
        hns_nhwc.append(mx.transpose(final, (0, 2, 3, 1)))

        # Concatenate: list of (B, H, W, 1) -> (B, H, W, 3)
        hns = mx.concatenate(hns_nhwc, axis=-1)  # (B, H, W, 3)

        # Temporal confidence
        wt_in = mx.concatenate([fout, hns], axis=-1)  # (B, H, W, C+3)
        wt = mx.softmax(self.convct1(self.convct0(wt_in)), axis=-1)  # (B, H, W, 3)

        hn = mx.sum(wt * hns, axis=-1, keepdims=True)  # (B, H, W, 1)
        return hn


class Coef(nn.Module):
    """Coefficient prediction (3 outputs: Alpha, Beta, Omega)."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, kernel_size=1, bias=True)

    def __call__(self, x):
        feat = self.conv(x)
        XF = feat[..., 0:1]
        XB = feat[..., 1:2]
        XW = feat[..., 2:3]
        return XF, XB, XW


class Permute(nn.Module):
    """Depth-to-space / pixel shuffle upsampling with learned weights."""

    def __init__(self, in_ch: int, out_ch: int = 1, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.out_ch = out_ch
        self.conv0 = Conv2dBnRelu(in_ch, in_ch, kernel_size=1, padding=0, act="relu")
        self.conv1 = Conv2dBnRelu(in_ch, in_ch, kernel_size=1, padding=0, act="relu")
        self.conv2 = nn.Conv2d(in_ch, out_ch * stride * stride, kernel_size=1, bias=True)

    def __call__(self, x):
        """x: (B, H, W, C) in NHWC."""
        x = self.conv2(self.conv1(self.conv0(x)))
        # Rearrange: (B, H, W, c*h2*w2) -> (B, H*h2, W*w2, c)
        B, H, W, _ = x.shape
        s = self.stride
        c = self.out_ch
        x = x.reshape(B, H, W, c, s, s)
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))  # (B, H, s, W, s, c)
        x = x.reshape(B, H * s, W * s, c)
        return x


class WPool(nn.Module):
    """Weighted pooling of sparse depth."""

    def __init__(self, in_ch: int, level: int, drift: float = 1e6):
        super().__init__()
        self.level = level
        self.drift = drift
        self.permute = Permute(in_ch, out_ch=1, stride=2 ** level)

    def __call__(self, S, fout):
        """
        S: (B, H, W, 1) sparse depth in NHWC
        fout: (B, H, W, C) features in NHWC
        Returns: (B, Hp, Wp, 1) pooled sparse depth at reduced resolution
        """
        W = self.permute(fout)  # (B, H, W, 1) - learned weights
        size = int(2 ** self.level)
        M = (S > 1e-3).astype(mx.float32)

        # Convert to NCHW for pooling ops
        W_nchw = mx.transpose(W, (0, 3, 1, 2))  # (B, 1, H, W)
        M_nchw = mx.transpose(M, (0, 3, 1, 2))
        S_nchw = mx.transpose(S, (0, 3, 1, 2))

        # Max pool for numerical stability
        WM = (W_nchw + self.drift) * M_nchw
        # Use manual max pooling with reshape
        B, C, Hh, Ww = WM.shape
        Hp, Wp = Hh // size, Ww // size
        WM_blocks = WM.reshape(B, C, Hp, size, Wp, size)
        maxW = mx.max(WM_blocks, axis=(3, 5), keepdims=False)  # (B, C, Hp, Wp)
        # Upsample maxW back to (B, C, H, W) via nearest neighbor repeat
        maxW = mx.repeat(mx.repeat(maxW, size, axis=2), size, axis=3)  # (B, C, H, W)
        maxW = (maxW - self.drift) * M_nchw  # subtract drift back

        expW = mx.exp(W_nchw * M_nchw - maxW) * M_nchw

        # Average pool S*expW and expW
        SexpW = S_nchw * expW
        # Reshape and mean for avg pool
        SexpW_blocks = SexpW.reshape(B, C, Hp, size, Wp, size)
        avgS = mx.mean(SexpW_blocks, axis=(3, 5)) * (size * size)  # sum / (size*size) * (size*size) = sum
        # Actually F.avg_pool2d divides by kernel_size^2
        avgS = mx.sum(SexpW_blocks, axis=(3, 5)) / (size * size)

        expW_blocks = expW.reshape(B, C, Hp, size, Wp, size)
        avgexpW = mx.sum(expW_blocks, axis=(3, 5)) / (size * size)

        Sp_nchw = avgS / (avgexpW + 1e-6)

        # Back to NHWC
        return mx.transpose(Sp_nchw, (0, 2, 3, 1))


class Dist(nn.Module):
    """K-NN distance computation for sparse points."""

    def __init__(self, num: int = 4):
        super().__init__()
        self.num = num

    def __call__(self, S_nchw, xx, yy):
        """
        S_nchw: (B, 1, H, W) sparse depth in NCHW
        xx: (W,) or meshgrid x-coords
        yy: (H,) or meshgrid y-coords
        Returns:
            Ofnum: (B, 2, num, N)
            args: (B, num, N)
        """
        B, _, H, W = S_nchw.shape
        N = H * W
        S_flat = S_nchw.reshape(B, 1, N)
        Valid = (S_flat > 1e-3)

        xy = mx.stack([xx.reshape(-1).astype(mx.float32),
                       yy.reshape(-1).astype(mx.float32)], axis=0).reshape(1, 2, -1)
        idx = mx.arange(N).reshape(1, 1, N)

        Ofnum, args = bpdist_mlx(xy, idx, Valid, self.num, H, W)
        return Ofnum, args


class Prop(nn.Module):
    """Propagation module - propagates depth from sparse neighbors."""

    def __init__(self, Cfi: int, Cfp: int = 3, Cfo: int = 2):
        super().__init__()
        Ct = Cfo + Cfi + Cfi + Cfp
        self.convXF0 = Conv2dBnRelu(Ct, Cfi, kernel_size=1, padding=0, act="gelu")
        self.convXF1 = Conv2dBnRelu(Cfi, Cfi, kernel_size=1, padding=0, act="gelu")
        self.convXL0 = Conv2dBnRelu(Cfi, Cfi, kernel_size=1, padding=0, act="gelu")
        self.convXL1 = Conv2dBnRelu(Cfi, Cfi, kernel_size=1, padding=0, act="identity")
        self.coef = Coef(Cfi)

    def __call__(self, If_nhwc, Pf, Ofnum, args):
        """
        If_nhwc: (B, H, W, Cfi) image features in NHWC
        Pf: (B, Cfp, M) 3D point features (x,y,z) - in NCHW-like format
        Ofnum: (B, 2, num, N) offsets to neighbors
        args: (B, num, N) neighbor indices

        Returns: (B, H, W, 1) propagated depth in NHWC
        """
        num = args.shape[1]
        B, H, W, Cfi = If_nhwc.shape
        N = H * W
        Cfp = Pf.shape[1]

        # Convert features to NCHW flat: (B, Cfi, 1, N)
        If = mx.transpose(If_nhwc, (0, 3, 1, 2)).reshape(B, Cfi, 1, N)

        # Expand to (B, Cfi, num, N)
        Ifnum = mx.broadcast_to(If, (B, Cfi, num, N))

        # Gather neighbor features
        M = Pf.shape[2]
        If_expanded = mx.broadcast_to(If, (B, Cfi, num, N))
        Pf_expanded = mx.broadcast_to(Pf.reshape(B, Cfp, 1, M), (B, Cfp, num, M))

        # Use args to gather: args is (B, num, N) indices into the N dimension
        # We need to gather from If and Pf using these indices
        IPfnum_list = []
        Pfnum_list = []
        for b in range(B):
            for c in range(Cfi):
                gathered = If[b, c, 0][args[b]]  # (num, N) gathered from (N,)
                IPfnum_list.append(gathered)
            for c in range(Cfp):
                gathered = Pf[b, c][args[b]]  # (num, N) gathered from (M,) -- but args indexes into N
                Pfnum_list.append(gathered)

        IPfnum = mx.stack(IPfnum_list, axis=0).reshape(B, Cfi, num, N)
        Pfnum = mx.stack(Pfnum_list, axis=0).reshape(B, Cfp, num, N)

        # Concatenate: (B, Cfi+Cfi+Cfp+2, num, N)
        X = mx.concatenate([Ifnum, IPfnum, Pfnum, Ofnum], axis=1)

        # Process through conv (treat num*N as spatial dims)
        # Reshape to (B, num, N, C_total) for NHWC processing
        Ct = Cfi + Cfi + Cfp + 2
        X_nhwc = mx.transpose(X, (0, 2, 3, 1))  # (B, num, N, Ct)
        X_nhwc = X_nhwc.reshape(B * num, 1, N, Ct)  # treat as (B*num, 1, N, Ct) for 1x1 conv

        # Apply 1x1 convolutions
        # Since these are 1x1 convs, they're just linear transforms
        XF = self.convXF1(self.convXF0(X_nhwc))  # (B*num, 1, N, Cfi)
        XL = self.convXL1(self.convXL0(XF))  # (B*num, 1, N, Cfi)
        XF = nn.gelu(XF + XL)

        # Coefficient prediction
        Alpha, Beta, Omega = self.coef(XF)  # each (B*num, 1, N, 1)

        # Reshape back
        Alpha = Alpha.reshape(B, num, N, 1).transpose((0, 3, 1, 2))  # (B, 1, num, N)
        Beta = Beta.reshape(B, num, N, 1).transpose((0, 3, 1, 2))
        Omega = Omega.reshape(B, num, N, 1).transpose((0, 3, 1, 2))

        Omega = mx.softmax(Omega, axis=2)

        # Depth from last channel of Pfnum (z coordinate)
        depth_neighbors = Pfnum[:, -1:, :, :]  # (B, 1, num, N)

        dout = mx.sum(((Alpha + 1) * depth_neighbors + Beta) * Omega, axis=2, keepdims=True)
        # dout: (B, 1, 1, N) -> (B, 1, H, W) -> NHWC
        dout = dout.reshape(B, 1, H, W)
        return mx.transpose(dout, (0, 2, 3, 1))  # (B, H, W, 1)


class UBNet(nn.Module):
    """U-shaped fusion network for combining image features with 3D geometry."""

    def __init__(self, inplanes: int, dplanes: int = 1, blocknum: int = 2,
                 depth: int = 1, drop_path: float = 0.0):
        super().__init__()
        bc = inplanes // 2

        # Encoder
        self.encoder_layers = []
        # First layer: initial conv + residual blocks
        first_layer = {
            "init_conv": Conv2dBnRelu(inplanes + dplanes, bc * 2, kernel_size=3, padding=1),
            "blocks": [BasicBlock(bc * 2, bc * 2, drop_path=0) for _ in range(blocknum)]
        }
        self.encoder_layers.append(first_layer)

        in_ch = bc * 2
        for i in range(depth):
            out_ch = min(in_ch * 2, 256)
            ds = DownsampleBlock(in_ch, out_ch, stride=2) if in_ch != out_ch or True else None
            layer = {
                "blocks": [BasicBlock(in_ch if j == 0 else out_ch, out_ch,
                                      stride=2 if j == 0 else 1,
                                      downsample=ds if j == 0 else None,
                                      drop_path=0)
                           for j in range(blocknum)]
            }
            self.encoder_layers.append(layer)
            in_ch = min(in_ch * 2, 256)

        # Decoder
        self.decoder_layers = []
        in_ch = bc * 2
        for i in range(depth):
            out_ch = min(in_ch * 2, 256)
            self.decoder_layers.append(UpCC(out_ch, in_ch, in_ch))
            in_ch = min(in_ch * 2, 256)

    def __call__(self, x, d):
        """
        x: (B, H, W, C) features in NHWC
        d: (B, H, W, Cd) depth/point features in NHWC or None
        """
        if d is not None:
            x = mx.concatenate([x, d], axis=-1)

        feat = []
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                x = layer["init_conv"](x)
                for block in layer["blocks"]:
                    x = block(x)
            else:
                for block in layer["blocks"]:
                    x = block(x)
            feat.append(x)

        out = feat[-1]
        for idx in range(len(feat) - 2, -1, -1):
            out = self.decoder_layers[idx](out, feat[idx])

        return out


class UpCat(nn.Module):
    """Upsample + Concatenate for PMP inter-level connections."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upf = Conv2dTransposeBnRelu(in_ch + 1, out_ch)
        self.conv = Conv2dBnRelu(out_ch * 2, out_ch, kernel_size=3, padding=1)

    def __call__(self, y, x, d):
        """
        y: skip connection features (B, H, W, out_ch)
        x: features from previous level (B, H/2, W/2, in_ch)
        d: depth from previous level (B, H/2, W/2, 1)
        """
        up_in = mx.concatenate([x, d], axis=-1)
        fout = self.upf(up_in)
        fout = mx.concatenate([fout, y], axis=-1)
        fout = self.conv(fout)
        return fout


class PMP(nn.Module):
    """
    Pre + MF + Post module.
    The core depth completion block at each pyramid level.
    """

    def __init__(self, level: int, in_ch: int, out_ch: int,
                 drop_path: float = 0.1, up: bool = True, pool: bool = True):
        super().__init__()
        self.level = level
        self.has_up = up
        self.has_pool = pool

        if up:
            self.upcat = UpCat(in_ch, out_ch)
        if pool:
            self.wpool = WPool(out_ch, level=level)

        self.dist = Dist(num=4)
        self.prop = Prop(out_ch)
        self.fuse = UBNet(out_ch, dplanes=3, blocknum=2, depth=5 - level, drop_path=drop_path)
        self.conv = nn.Conv2d(out_ch, 1, kernel_size=3, padding=1, bias=True)
        self.cspn = CSPN(out_ch, pt=2 * (6 - level))

    def pinv(self, S_nchw, K, xx, yy):
        """Back-project depth to 3D points.
        S_nchw: (B, 1, H, W)
        K: (B, 3, 3) camera intrinsics
        Returns: (B, 3, M) where M = H*W
        """
        fx = K[:, 0:1, 0:1]
        fy = K[:, 1:2, 1:2]
        cx = K[:, 0:1, 2:3]
        cy = K[:, 1:2, 2:3]

        S = S_nchw.reshape(S_nchw.shape[0], 1, -1)
        xx_flat = xx.reshape(1, 1, -1).astype(mx.float32)
        yy_flat = yy.reshape(1, 1, -1).astype(mx.float32)

        Px = S * (xx_flat - cx) / fx
        Py = S * (yy_flat - cy) / fy
        Pz = S
        Pxyz = mx.concatenate([Px, Py, Pz], axis=1)
        return Pxyz

    def __call__(self, fout, dout, XI, S, K):
        """
        fout: (B, H/2, W/2, C) features from previous level or None
        dout: (B, H/2, W/2, 1) depth from previous level or None
        XI: (B, H, W, C) image features at this level
        S: (B, Hfull, Wfull, 1) full-res sparse depth
        K: (B, 3, 3) camera intrinsics

        Returns: (fout, dout) at this level's resolution
        """
        # Upsample from previous level
        if self.has_up:
            fout = self.upcat(XI, fout, dout)
        else:
            fout = XI

        # Pool sparse depth to this level's resolution
        if self.has_pool:
            Sp = self.wpool(S, fout)
        else:
            Sp = S

        # Scale intrinsics
        Kp = mx.array(K)  # copy
        scale = 2.0 ** self.level
        Kp_list = []
        for b in range(K.shape[0]):
            k = K[b]
            k_scaled = mx.array(k)
            # Scale fx, fy, cx, cy
            row0 = mx.concatenate([k[0:1, 0:1] / scale, k[0:1, 1:2], k[0:1, 2:3] / scale])
            row1 = mx.concatenate([k[1:2, 0:1], k[1:2, 1:2] / scale, k[1:2, 2:3] / scale])
            row2 = k[2:3]
            Kp_list.append(mx.stack([row0, row1, row2.squeeze(0)], axis=0))
        Kp = mx.stack(Kp_list, axis=0)

        # Get spatial dims at this level
        B = Sp.shape[0]
        Hh = Sp.shape[1]
        Ww = Sp.shape[2]

        ww = mx.arange(Ww)
        hh = mx.arange(Hh)
        xx, yy = mx.meshgrid(ww, hh, indexing='xy')

        # Convert Sp to NCHW for operations that need it
        Sp_nchw = mx.transpose(Sp, (0, 3, 1, 2))  # (B, 1, H, W)

        # ===== Pre: Propagation from sparse depth =====
        Pxyz = self.pinv(Sp_nchw, Kp, xx, yy)  # (B, 3, N)
        Ofnum, args = self.dist(Sp_nchw, xx, yy)  # (B,2,4,N), (B,4,N)
        dout = self.prop(fout, Pxyz, Ofnum, args)  # (B, H, W, 1) NHWC

        # ===== MF: Mean Field fusion =====
        dout_nchw = mx.transpose(dout, (0, 3, 1, 2))
        Pxyz = self.pinv(dout_nchw, Kp, xx, yy)  # (B, 3, N)
        Pxyz_nhwc = Pxyz.reshape(B, 3, Hh, Ww)
        Pxyz_nhwc = mx.transpose(Pxyz_nhwc, (0, 2, 3, 1))  # (B, H, W, 3)
        fout = self.fuse(fout, Pxyz_nhwc)
        res = self.conv(fout)  # (B, H, W, 1)
        dout = dout + res

        # ===== Post: CSPN refinement =====
        dout = self.cspn(fout, dout, Sp)

        return fout, dout


# ============================================================================
# Main DMD3C Model
# ============================================================================

class DMD3CModel(nn.Module):
    """
    DMD3C depth completion model ported to MLX.

    Input:
        image: (B, H, W, 3) normalized RGB image in NHWC
        sparse_depth: (B, H, W, 1) sparse depth map in NHWC
        K: (B, 3, 3) camera intrinsic matrix

    Output:
        List of 6 depth maps at different scales, all upsampled to (B, H, W, 1)
    """

    def __init__(self, bc: int = 16):
        super().__init__()

        # Image encoder
        self.conv_img_init = Conv2dBnRelu(3, bc * 2, kernel_size=3, padding=1)
        self.conv_img_blocks = [BasicBlock(bc * 2, bc * 2) for _ in range(2)]

        self.layer1_img = self._make_encoder_layer(bc * 2, bc * 4, 2, stride=2)
        self.layer2_img = self._make_encoder_layer(bc * 4, bc * 8, 2, stride=2)
        self.layer3_img = self._make_encoder_layer(bc * 8, bc * 16, 2, stride=2)
        self.layer4_img = self._make_encoder_layer(bc * 16, bc * 16, 2, stride=2)
        self.layer5_img = self._make_encoder_layer(bc * 16, bc * 16, 2, stride=2)

        # Prediction modules (pyramid levels)
        self.pred5 = PMP(level=5, in_ch=bc * 16, out_ch=bc * 16, up=False)
        self.pred4 = PMP(level=4, in_ch=bc * 16, out_ch=bc * 16)
        self.pred3 = PMP(level=3, in_ch=bc * 16, out_ch=bc * 16)
        self.pred2 = PMP(level=2, in_ch=bc * 16, out_ch=bc * 8)
        self.pred1 = PMP(level=1, in_ch=bc * 8, out_ch=bc * 4)
        self.pred0 = PMP(level=0, in_ch=bc * 4, out_ch=bc * 2, pool=False)

    def _make_encoder_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        ds = DownsampleBlock(in_ch, out_ch, stride) if in_ch != out_ch or stride != 1 else None
        layers.append(BasicBlock(in_ch, out_ch, stride=stride, downsample=ds))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return layers

    def _upsample(self, x, scale_factor):
        """Bilinear upsample in NHWC format."""
        B, H, W, C = x.shape
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)
        # MLX doesn't have a direct bilinear upsample, use repeat for nearest or implement bilinear
        # For now use a simple approach: transpose to NCHW, upsample, transpose back
        # MLX has nn.Upsample or we can implement manually
        x_nchw = mx.transpose(x, (0, 3, 1, 2))  # (B, C, H, W)

        # Bilinear interpolation
        # Use grid-based sampling
        h_coords = mx.linspace(0, H - 1, new_H)
        w_coords = mx.linspace(0, W - 1, new_W)
        grid_h, grid_w = mx.meshgrid(h_coords, w_coords, indexing='ij')

        # For each output pixel, compute bilinear interpolation
        h0 = mx.floor(grid_h).astype(mx.int32)
        w0 = mx.floor(grid_w).astype(mx.int32)
        h1 = mx.minimum(h0 + 1, H - 1)
        w1 = mx.minimum(w0 + 1, W - 1)

        ha = grid_h - h0.astype(mx.float32)
        wa = grid_w - w0.astype(mx.float32)

        # Gather corners
        result_list = []
        for b in range(B):
            channels = []
            for c in range(C):
                plane = x_nchw[b, c]  # (H, W)
                v00 = plane[h0, w0]
                v01 = plane[h0, w1]
                v10 = plane[h1, w0]
                v11 = plane[h1, w1]
                out = v00 * (1 - ha) * (1 - wa) + v01 * (1 - ha) * wa + \
                      v10 * ha * (1 - wa) + v11 * ha * wa
                channels.append(out)
            result_list.append(mx.stack(channels, axis=0))  # (C, new_H, new_W)

        result = mx.stack(result_list, axis=0)  # (B, C, new_H, new_W)
        return mx.transpose(result, (0, 2, 3, 1))  # (B, new_H, new_W, C)

    def __call__(self, image, sparse_depth, K):
        """
        image: (B, H, W, 3) normalized RGB
        sparse_depth: (B, H, W, 1) sparse depth
        K: (B, 3, 3) camera intrinsics
        Returns: list of 6 depth maps, each (B, H, W, 1)
        """
        output = []

        # Encoder
        XI0 = self.conv_img_init(image)
        for block in self.conv_img_blocks:
            XI0 = block(XI0)

        XI1 = XI0
        for layer in self.layer1_img:
            XI1 = layer(XI1)

        XI2 = XI1
        for layer in self.layer2_img:
            XI2 = layer(XI2)

        XI3 = XI2
        for layer in self.layer3_img:
            XI3 = layer(XI3)

        XI4 = XI3
        for layer in self.layer4_img:
            XI4 = layer(XI4)

        XI5 = XI4
        for layer in self.layer5_img:
            XI5 = layer(XI5)

        # Decoder (coarse to fine)
        fout, dout = self.pred5(fout=None, dout=None, XI=XI5, S=sparse_depth, K=K)
        output.append(self._upsample(dout, 2 ** 5))

        fout, dout = self.pred4(fout=fout, dout=dout, XI=XI4, S=sparse_depth, K=K)
        output.append(self._upsample(dout, 2 ** 4))

        fout, dout = self.pred3(fout=fout, dout=dout, XI=XI3, S=sparse_depth, K=K)
        output.append(self._upsample(dout, 2 ** 3))

        fout, dout = self.pred2(fout=fout, dout=dout, XI=XI2, S=sparse_depth, K=K)
        output.append(self._upsample(dout, 2 ** 2))

        fout, dout = self.pred1(fout=fout, dout=dout, XI=XI1, S=sparse_depth, K=K)
        output.append(self._upsample(dout, 2 ** 1))

        fout, dout = self.pred0(fout=fout, dout=dout, XI=XI0, S=sparse_depth, K=K)
        output.append(dout)

        return output

    def predict(self, image, sparse_depth, K):
        """Convenience method returning only the final (finest) depth map."""
        outputs = self(image, sparse_depth, K)
        return outputs[-1]
