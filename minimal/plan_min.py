#!/usr/bin/env python3
"""
Minimal planning script using CEM in latent space.
Loads trained world model and plans action sequence to reach goal.
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from min_dinov2 import DinoV2Encoder
from transition import TransitionModel, FrameCausalTransformer
from cem import create_cem_planner
from envs.pointmaze import PointMazeEnv


def load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create encoder
    encoder = DinoV2Encoder()
    encoder = encoder.to(device)
    encoder.eval()

    # Create transformer
    embed_dim = checkpoint['embed_dim']
    grid_h, grid_w = checkpoint['grid']
    num_patches = grid_h * grid_w
    action_dim = 2  # For PointMaze

    transformer = FrameCausalTransformer(
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_patches=num_patches
    )

    # Load state dict
    transformer.load_state_dict(checkpoint['wm'])
    transformer = transformer.to(device)
    transformer.eval()

    # Create complete model
    model = TransitionModel(encoder, action_dim)
    model.transformer = transformer

    return model, encoder


def get_latent_from_image(encoder: DinoV2Encoder, image: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """Get latent representation from image."""
    # Convert to torch tensor
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    # Add batch dimension if needed
    if image.dim() == 3:  # (H, W, 3)
        image = image.unsqueeze(0)  # (1, H, W, 3)

    # Move to device
    image = image.to(device)

    # Encode
    with torch.no_grad():
        latent, _ = encoder(image)  # (1, N, D)

    return latent.squeeze(0)  # (N, D)


def create_goal_image(env: PointMazeEnv, goal_state: np.ndarray) -> np.ndarray:
    """Create goal image by setting env to goal state and rendering."""
    # Save current state
    current_state = env.get_state()

    # Set to goal state
    env.set_state(goal_state)

    # Render goal image
    goal_image = env.render_rgb()

    # Restore current state
    env.set_state(current_state)

    return goal_image


def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint from {args.ckpt}")
    model, encoder = load_checkpoint(args.ckpt, device)
    print("Model loaded successfully")

    # Create environment
    env = PointMazeEnv()
    print("Environment created")

    # Reset environment to get current image
    obs, state = env.reset(seed=args.seed)
    current_image = obs['image']
    print(f"Environment reset, current state: {state}")

    # Get goal state (either random or specified)
    if args.goal_state is not None:
        goal_state = np.array(args.goal_state)
    else:
        # Create random goal state
        _, goal_state = env.sample_random_init_goal_states(args.seed + 1000)
    print(f"Goal state: {goal_state}")

    # Create goal image
    goal_image = create_goal_image(env, goal_state)
    print("Goal image created")

    # Get latents
    current_latent = get_latent_from_image(encoder, current_image, device)
    goal_latent = get_latent_from_image(encoder, goal_image, device)
    print(f"Current latent shape: {current_latent.shape}")
    print(f"Goal latent shape: {goal_latent.shape}")

    # Add batch dimensions for planning
    current_latent = current_latent.unsqueeze(0)  # (1, N, D)
    goal_latent = goal_latent.unsqueeze(0)  # (1, N, D)

    # Create CEM planner
    planner = create_cem_planner(
        horizon=args.horizon,
        action_dim=2,  # PointMaze has 2D actions
        population_size=args.pop,
        device=device
    )
    print(f"CEM planner created with horizon={args.horizon}, population={args.pop}")

    # Plan action sequence
    print("Starting CEM planning...")
    optimal_actions = planner.plan(model, current_latent, goal_latent, args.iters)

    # Move to CPU and convert to numpy
    optimal_actions = optimal_actions.cpu().numpy()

    print("\nPlanning completed!")
    print(f"Optimal action sequence shape: {optimal_actions.shape}")
    print("Planned actions:")
    for t, action in enumerate(optimal_actions):
        print(".3f")

    # Print action bounds check
    action_min, action_max = optimal_actions.min(), optimal_actions.max()
    print(".3f")
    if action_min >= -1.0 and action_max <= 1.0:
        print("✓ Actions are within bounds [-1, 1]")
    else:
        print("⚠ Actions may be out of bounds [-1, 1]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan with minimal world model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--horizon", type=int, default=12, help="Planning horizon")
    parser.add_argument("--pop", type=int, default=256, help="CEM population size")
    parser.add_argument("--iters", type=int, default=5, help="CEM iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--goal_state", type=float, nargs=2, help="Goal state [x, y]")

    args = parser.parse_args()
    main(args)
