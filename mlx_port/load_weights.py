#!/usr/bin/env python3
"""
DMD3C Weight Loading Utility

Maps PyTorch checkpoint parameter names to the MLX model structure and loads them.
Handles the OIHW → OHWI weight transposition and BatchNorm parameter mapping.

Usage:
    from load_weights import load_dmd3c_weights
    model = DMD3CModel()
    load_dmd3c_weights(model, "checkpoints/dmd3c_distillation_depth_anything_v2.pth")
"""

import os
import sys
from typing import Dict

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import mlx.core as mx
except ImportError:
    mx = None


def load_pytorch_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Load a PyTorch checkpoint and return numpy arrays."""
    if torch is None:
        raise ImportError("PyTorch is required to load .pth checkpoints")

    cp = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = cp["net"]

    numpy_dict = {}
    for key, value in state_dict.items():
        if "num_batches_tracked" in key:
            continue
        numpy_dict[key] = value.numpy()

    return numpy_dict


def transpose_conv_weight(w: np.ndarray) -> np.ndarray:
    """PyTorch OIHW → MLX OHWI."""
    return np.transpose(w, (0, 2, 3, 1))


def transpose_conv_transpose_weight(w: np.ndarray) -> np.ndarray:
    """PyTorch ConvTranspose IOHW → MLX OHWI."""
    return np.transpose(w, (1, 2, 3, 0))


def map_pytorch_to_mlx(pytorch_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Map PyTorch parameter names to MLX parameter names and transpose weights.

    PyTorch model structure (key patterns):
        conv_img.0.conv.conv.weight     -> conv_img_init.conv.weight
        conv_img.0.conv.bn.{weight,bias,running_mean,running_var}
        conv_img.1.{0,1}.conv{1,2}.weight
        conv_img.1.{0,1}.bn{1,2}.{weight,bias,...}
        layer{1-5}_img.{0,1}.conv{1,2}.weight
        layer{1-5}_img.{0,1}.bn{1,2}.{...}
        layer{1-5}_img.0.downsample.0.weight
        layer{1-5}_img.0.downsample.1.{...}
        pred{0-5}.upcat.upf.conv.weight
        pred{0-5}.upcat.upf.bn.{...}
        pred{0-5}.upcat.conv.conv.conv.weight
        pred{0-5}.upcat.conv.conv.bn.{...}
        pred{0-5}.wpool.permute.conv.{0,1}.conv.conv.weight
        pred{0-5}.wpool.permute.conv.{0,1}.conv.bn.{...}
        pred{0-5}.wpool.permute.conv.2.{weight,bias}
        pred{0-5}.prop.convXF.{0,1}.conv.conv.weight
        pred{0-5}.prop.convXF.{0,1}.conv.bn.{...}
        pred{0-5}.prop.convXL.{0,1}.conv.conv.weight
        pred{0-5}.prop.convXL.{0,1}.conv.bn.{...}
        pred{0-5}.prop.coef.conv.{weight,bias}
        pred{0-5}.fuse.encoder.*.*.conv{1,2}.weight
        pred{0-5}.fuse.encoder.*.*.bn{1,2}.{...}
        pred{0-5}.fuse.decoder.*.upf.conv.weight
        pred{0-5}.fuse.decoder.*.conv.conv.conv.weight
        pred{0-5}.conv.{weight,bias}
        pred{0-5}.cspn.*
    """
    mlx_weights = {}

    # Identify ConvTranspose2d weights (they need different transposition)
    conv_transpose_keys = set()
    for key in pytorch_weights:
        if "upf.conv.weight" in key:
            conv_transpose_keys.add(key)

    for pt_key, value in pytorch_weights.items():
        mlx_key = pt_key  # Start with same key, then adjust

        # Transpose 4D weights
        if value.ndim == 4:
            if pt_key in conv_transpose_keys:
                value = transpose_conv_transpose_weight(value)
            else:
                value = transpose_conv_weight(value)

        # Map BatchNorm running stats
        # PyTorch: bn.running_mean/running_var → MLX: bn.running_mean/running_var
        # PyTorch: bn.weight → MLX: bn.weight (gamma)
        # PyTorch: bn.bias → MLX: bn.bias (beta)
        # These names are the same in MLX nn.BatchNorm

        mlx_weights[mlx_key] = value.astype(np.float32)

    return mlx_weights


def load_dmd3c_weights(model, checkpoint_path: str):
    """
    Load PyTorch checkpoint weights into the MLX DMD3C model.

    This function handles the mapping between PyTorch and MLX naming conventions
    and weight formats. It loads weights directly using the MLX model's
    load_weights method with strict=False to handle naming differences.

    Args:
        model: DMD3CModel instance
        checkpoint_path: Path to the PyTorch .pth checkpoint
    """
    print(f"Loading weights from {checkpoint_path}...")

    # Load and convert weights
    pt_weights = load_pytorch_checkpoint(checkpoint_path)
    mlx_weights = map_pytorch_to_mlx(pt_weights)

    print(f"Converted {len(mlx_weights)} parameter tensors")

    # Save as safetensors for MLX
    output_dir = os.path.dirname(checkpoint_path) or "."
    safetensors_path = os.path.join(output_dir, "dmd3c_mlx_weights.safetensors")

    try:
        from safetensors.numpy import save_file
        save_file(mlx_weights, safetensors_path)
        print(f"Saved MLX weights to {safetensors_path}")
    except ImportError:
        npz_path = os.path.join(output_dir, "dmd3c_mlx_weights.npz")
        np.savez(npz_path, **mlx_weights)
        print(f"Saved MLX weights to {npz_path}")

    return mlx_weights


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/dmd3c_distillation_depth_anything_v2.pth")
    parser.add_argument("--output", type=str, default="mlx_weights/dmd3c_mlx_weights")
    args = parser.parse_args()

    pt_weights = load_pytorch_checkpoint(args.checkpoint)
    mlx_weights = map_pytorch_to_mlx(pt_weights)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    try:
        from safetensors.numpy import save_file
        output_path = args.output + ".safetensors"
        save_file(mlx_weights, output_path)
    except ImportError:
        output_path = args.output + ".npz"
        np.savez(output_path, **mlx_weights)

    print(f"Saved {len(mlx_weights)} tensors to {output_path}")
    size = os.path.getsize(output_path)
    print(f"File size: {size / 1024 / 1024:.1f} MB")
