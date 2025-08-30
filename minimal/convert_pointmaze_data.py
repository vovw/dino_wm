#!/usr/bin/env python3
"""
Convert existing PointMaze data to NPZ format for minimal implementation.
Usage: python convert_pointmaze_data.py --input_dir data/point_maze --output_dir converted_npz
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from einops import rearrange


def convert_pointmaze_to_npz(input_dir: str, output_dir: str):
    """Convert existing PointMaze .pth data to NPZ format."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load existing data
    print("Loading existing PointMaze data...")
    states = torch.load(input_path / "states.pth").float()
    actions = torch.load(input_path / "actions.pth").float()
    seq_lengths = torch.load(input_path / "seq_lengths.pth")
    obs_dir = input_path / "obses"

    print(f"Found {len(states)} trajectories")

    converted_count = 0
    for idx in tqdm(range(len(states)), desc="Converting trajectories"):
        try:
            # Load episode images
            episode_file = obs_dir / f"episode_{idx:03d}.pth"
            if not episode_file.exists():
                print(f"Warning: Missing episode file {episode_file}")
                continue

            episode_images = torch.load(episode_file)  # (T, H, W, C)

            # Get sequence length
            T = seq_lengths[idx]

            # Extract trajectory data
            traj_images = episode_images[:T]  # (T, H, W, C)
            traj_actions = actions[idx, :T-1]  # (T-1, A)

            # Convert to numpy and proper format
            images_np = traj_images.numpy().astype(np.uint8)  # Keep as uint8
            actions_np = traj_actions.numpy().astype(np.float32)

            # Scale actions to [-1, 1] range (existing data might be in different range)
            # Assuming original actions are roughly in [-3, 3] range based on env
            actions_np = np.clip(actions_np / 3.0, -1.0, 1.0)

            # Resize images from 224x224 to 64x64 to match minimal env
            if images_np.shape[1:3] == (224, 224):
                # Simple downsampling
                images_resized = []
                for t in range(T):
                    img = images_np[t]  # (224, 224, 3)
                    # Downsample by factor of ~3.5
                    img_resized = img[::3, ::3, :]  # (75, 75, 3)
                    # Center crop to 64x64
                    h_start = (75 - 64) // 2
                    w_start = (75 - 64) // 2
                    img_resized = img_resized[h_start:h_start+64, w_start:w_start+64, :]
                    images_resized.append(img_resized)
                images_np = np.stack(images_resized)  # (T, 64, 64, 3)

            # Save as NPZ
            out_file = output_path / f"traj_{idx:03d}.npz"
            np.savez(out_file, images=images_np, actions=actions_np)

            converted_count += 1

        except Exception as e:
            print(f"Error converting trajectory {idx}: {e}")
            continue

    print(f"Successfully converted {converted_count}/{len(states)} trajectories")
    print(f"Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert PointMaze data to NPZ format")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory with PointMaze data (states.pth, actions.pth, etc.)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for NPZ files")
    parser.add_argument("--max_trajs", type=int, default=None,
                       help="Maximum number of trajectories to convert")

    args = parser.parse_args()
    convert_pointmaze_to_npz(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
