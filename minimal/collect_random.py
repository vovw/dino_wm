#!/usr/bin/env python3
"""
Data collection script for minimal PointMaze environment.
Generates random trajectories and saves as NPZ files.
"""

import argparse
import os
from pathlib import Path

from envs.pointmaze import PointMazeEnv
from dataset_npz import collect_random_data


def main(args):
    """Collect random trajectories."""
    print(f"Collecting {args.n} trajectories of length {args.T}")
    print(f"Saving to {args.out}")

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Collect data
    collect_random_data(
        out_dir=str(out_dir),
        n_trajectories=args.n,
        T=args.T
    )

    print("Data collection completed!")
    print(f"Check {out_dir} for {args.n} NPZ files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect random PointMaze trajectories")
    parser.add_argument("--out", type=str, required=True, help="Output directory for NPZ files")
    parser.add_argument("--n", type=int, default=32, help="Number of trajectories")
    parser.add_argument("--T", type=int, default=40, help="Trajectory length")

    args = parser.parse_args()
    main(args)
