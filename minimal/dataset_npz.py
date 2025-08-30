import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import glob


class NPZDataset(Dataset):
    """
    Dataset for loading NPZ trajectory files.
    Each .npz contains:
    - images: uint8 shape (T, H, W, 3)
    - actions: float32 shape (T-1, A) where A=2 for PointMaze
    """

    def __init__(self, data_dir: str, transform=None, action_scale: float = 1.0):
        """
        Args:
            data_dir: Directory containing .npz files
            transform: Optional transform for images
            action_scale: Scale factor for actions (for compatibility with different ranges)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.action_scale = action_scale

        # Find all .npz files
        self.npz_files = sorted(glob.glob(str(self.data_dir / "*.npz")))
        if len(self.npz_files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        # Load metadata from first file to check shapes
        sample_data = np.load(self.npz_files[0])
        self.T = sample_data['images'].shape[0]
        self.H, self.W = sample_data['images'].shape[1:3]
        self.A = sample_data['actions'].shape[1]

        print(f"Loaded {len(self.npz_files)} trajectories")
        print(f"Trajectory length: {self.T}, Image size: {self.H}x{self.W}, Action dim: {self.A}")
        if action_scale != 1.0:
            print(f"Action scale: {action_scale}")

    def __len__(self) -> int:
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            images: (T, H, W, 3) uint8 -> (T, 3, H, W) float32 normalized to [0,1]
            actions: (T-1, A) float32
        """
        data = np.load(self.npz_files[idx])

        # Load images and convert to torch tensor
        images = data['images']  # (T, H, W, 3) uint8
        images = torch.from_numpy(images).float() / 255.0  # (T, H, W, 3) -> float32 [0,1]
        images = images.permute(0, 3, 1, 2)  # (T, 3, H, W)

        # Apply transform if provided
        if self.transform:
            images = self.transform(images)

        # Load actions
        actions = data['actions']  # (T-1, A) float32
        actions = torch.from_numpy(actions).float()

        # Apply action scaling if specified
        if self.action_scale != 1.0:
            actions = actions * self.action_scale

        return images, actions

    def get_trajectory_length(self) -> int:
        """Get the length of trajectories (T)."""
        return self.T

    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.A

    def get_image_size(self) -> Tuple[int, int]:
        """Get image dimensions (H, W)."""
        return self.H, self.W


def create_random_trajectory(env, seed: int, T: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random trajectory for testing.
    Returns:
        images: (T, H, W, 3) uint8
        actions: (T-1, A) float32
    """
    obs, _ = env.reset(seed)
    images = [obs['image']]
    actions = []

    for t in range(T - 1):
        # Random action
        action = np.random.uniform(-1.0, 1.0, size=2)
        actions.append(action)

        obs, _, _, _ = env.step(action)
        images.append(obs['image'])

    images = np.stack(images)  # (T, H, W, 3)
    actions = np.stack(actions)  # (T-1, 2)

    return images, actions


def collect_random_data(out_dir: str, n_trajectories: int = 32, T: int = 40):
    """
    Collect random trajectories and save as NPZ files.
    """
    from .envs.pointmaze import PointMazeEnv

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    env = PointMazeEnv()

    for i in range(n_trajectories):
        print(f"Collecting trajectory {i+1}/{n_trajectories}")
        images, actions = create_random_trajectory(env, seed=i, T=T)

        # Save as NPZ
        out_path = out_dir / f"traj_{i:03d}.npz"
        np.savez(out_path, images=images, actions=actions)

    print(f"Saved {n_trajectories} trajectories to {out_dir}")
