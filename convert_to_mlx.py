#!/usr/bin/env python3
"""
DMD3C PyTorch → MLX Weight Conversion Script

Converts the pretrained DMD3C checkpoint (.pth) to MLX-compatible safetensors format.
The weight keys are remapped from PyTorch (NCHW) to MLX (NHWC) conventions where needed.

Usage:
    python convert_to_mlx.py [--checkpoint PATH] [--output PATH]

Defaults:
    --checkpoint checkpoints/dmd3c_distillation_depth_anything_v2.pth
    --output mlx_weights/dmd3c.safetensors
"""

import argparse
import os
import sys

import numpy as np

# We only need torch for loading the checkpoint
try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required to load the checkpoint. Install with:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

try:
    from safetensors.numpy import save_file
    USE_SAFETENSORS = True
except ImportError:
    USE_SAFETENSORS = False
    print("WARNING: safetensors not installed, will save as .npz instead")
    print("  pip install safetensors")


def convert_conv_weight(weight_np):
    """Convert a PyTorch conv weight from OIHW to MLX's OHWI format."""
    # PyTorch: (out_channels, in_channels, H, W)
    # MLX conv2d expects: (out_channels, H, W, in_channels)
    return np.transpose(weight_np, (0, 2, 3, 1))


def convert_conv_transpose_weight(weight_np):
    """Convert a PyTorch ConvTranspose2d weight from IOHW to MLX's OHWI format.
    
    PyTorch ConvTranspose2d weight: (in_channels, out_channels, H, W)
    MLX conv_transpose2d expects: (out_channels, H, W, in_channels)
    """
    return np.transpose(weight_np, (1, 2, 3, 0))


def main():
    parser = argparse.ArgumentParser(description="Convert DMD3C PyTorch checkpoint to MLX format")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/dmd3c_distillation_depth_anything_v2.pth",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="mlx_weights/dmd3c",
                        help="Output path (without extension)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Download with:")
        print("  wget https://github.com/Sharpiless/DMD3C/releases/download/"
              "pretrain-checkpoints/dmd3c_distillation_depth_anything_v2.pth")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = cp["net"]
    print(f"  Epoch: {cp.get('epoch', 'N/A')}")
    print(f"  Best metric EMA: {cp.get('best_metric_ema', 'N/A')}")
    print(f"  Number of parameter tensors: {len(state_dict)}")

    total_params = sum(v.numel() for v in state_dict.values())
    print(f"  Total parameters: {total_params:,}")

    # Convert all weights
    mlx_weights = {}
    conv_transpose_keys = set()
    skip_count = 0

    # First pass: identify ConvTranspose2d weights by their shape pattern
    # ConvTranspose2d weights in PyTorch are (in_ch, out_ch, kH, kW) - we identify them
    # by the parameter name containing 'upf.conv.weight' (from Basic2dTrans)
    for key in state_dict.keys():
        if "upf.conv.weight" in key:
            conv_transpose_keys.add(key)

    for key, value in state_dict.items():
        np_val = value.numpy()

        # Skip num_batches_tracked (not needed for inference)
        if "num_batches_tracked" in key:
            skip_count += 1
            continue

        # Handle conv weights: transpose from OIHW to OHWI
        if np_val.ndim == 4:
            if key in conv_transpose_keys:
                np_val = convert_conv_transpose_weight(np_val)
            else:
                np_val = convert_conv_weight(np_val)

        # Ensure float32
        if np_val.dtype != np.float32:
            np_val = np_val.astype(np.float32)

        mlx_weights[key] = np_val

    print(f"\nConverted {len(mlx_weights)} tensors (skipped {skip_count} tracking tensors)")

    # Save
    if USE_SAFETENSORS:
        output_path = args.output + ".safetensors"
        save_file(mlx_weights, output_path)
        print(f"Saved to {output_path}")
    else:
        output_path = args.output + ".npz"
        np.savez(output_path, **mlx_weights)
        print(f"Saved to {output_path}")

    # Print size
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")

    # Verify by spot-checking a few weights
    print("\nVerification (first 5 conv weights):")
    count = 0
    for key in mlx_weights:
        if mlx_weights[key].ndim == 4 and count < 5:
            orig = state_dict[key].numpy()
            converted = mlx_weights[key]
            if key in conv_transpose_keys:
                # ConvTranspose: IOHW -> OHWI
                reconstructed = np.transpose(converted, (3, 0, 1, 2))
            else:
                # Regular conv: OIHW -> OHWI
                reconstructed = np.transpose(converted, (0, 3, 1, 2))
            match = np.allclose(orig, reconstructed, atol=1e-7)
            print(f"  {key}: {orig.shape} -> {converted.shape} match={match}")
            count += 1

    print("\nDone! Use with mlx_port/dmd3c_mlx.py for inference.")


if __name__ == "__main__":
    main()
