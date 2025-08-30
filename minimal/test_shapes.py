"""
Simple tests for tensor shapes and world model functionality.
Run with: python -m pytest test_shapes.py -v
"""

import torch
import numpy as np


def test_pointmaze_env():
    """Test PointMaze environment shapes and functionality."""
    from envs.pointmaze import PointMazeEnv

    env = PointMazeEnv()

    # Test reset
    obs, state = env.reset(seed=42)
    assert obs['image'].shape == (64, 64, 3), f"Image shape: {obs['image'].shape}"
    assert state.shape == (2,), f"State shape: {state.shape}"
    assert obs['state'].shape == (2,), f"Obs state shape: {obs['state'].shape}"

    # Test step
    action = np.array([0.1, -0.1])
    obs2, reward, done, info = env.step(action)
    assert obs2['image'].shape == (64, 64, 3)
    assert obs2['state'].shape == (2,)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert 'state' in info

    # Test rendering
    img = env.render_rgb()
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8

    print("✓ PointMaze environment tests passed")


def test_dinov2_encoder():
    """Test DINOv2 encoder with different input formats."""
    from min_dinov2 import DinoV2Encoder

    encoder = DinoV2Encoder()

    # Test BHWC input
    x_bhwc = torch.randn(2, 64, 64, 3)
    tokens, grid = encoder(x_bhwc)
    assert tokens.shape[0] == 2  # batch size
    assert tokens.shape[2] == encoder.embed_dim  # embedding dim
    assert grid[0] * grid[1] == tokens.shape[1]  # num patches

    # Test BCHW input
    x_bchw = torch.randn(2, 3, 64, 64)
    tokens2, grid2 = encoder(x_bchw)
    assert tokens2.shape == tokens.shape
    assert grid2 == grid

    # Test uint8 input
    x_uint8 = torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)
    tokens3, grid3 = encoder(x_uint8)
    assert tokens3.shape == tokens.shape
    assert grid3 == grid

    print("✓ DINOv2 encoder tests passed")


def test_world_model_shapes():
    """Test world model forward shapes."""
    from min_dinov2 import DinoV2Encoder
    from transition import TransitionModel

    # Create components
    encoder = DinoV2Encoder()
    model = TransitionModel(encoder, action_dim=2)

    # Test input shapes
    B, T = 2, 6
    images = torch.randn(B, T, 3, 64, 64)  # BHWC format
    actions = torch.randn(B, T-1, 2)

    # Forward pass
    preds, targets = model(images, actions)

    # Check shapes
    expected_shape = (B, T-1, encoder.num_patches, encoder.embed_dim)
    assert preds.shape == expected_shape, f"Preds shape: {preds.shape}, expected: {expected_shape}"
    assert targets.shape == expected_shape, f"Targets shape: {targets.shape}, expected: {expected_shape}"

    # Test loss computation
    loss = model.compute_loss(images, actions)
    assert loss.dim() == 0, f"Loss should be scalar, got shape: {loss.shape}"
    assert loss > 0, f"Loss should be positive, got: {loss.item()}"

    print("✓ World model shape tests passed")


def test_cem_planner():
    """Test CEM planner functionality."""
    from cem import create_cem_planner

    planner = create_cem_planner(horizon=5, action_dim=2, population_size=10)

    # Test action sampling
    actions = planner._sample_actions()
    assert actions.shape == (10, 5, 2), f"Actions shape: {actions.shape}"
    assert actions.min() >= -1.0, f"Min action: {actions.min()}"
    assert actions.max() <= 1.0, f"Max action: {actions.max()}"

    # Test distribution update
    elite_actions = torch.randn(5, 5, 2)
    planner._update_distribution(elite_actions)
    assert planner.mean.shape == (5, 2)
    assert planner.std.shape == (5, 2)

    print("✓ CEM planner tests passed")


def test_dataset_shapes():
    """Test dataset loading and shapes."""
    import tempfile
    import os
    from dataset_npz import create_random_trajectory
    from envs.pointmaze import PointMazeEnv

    # Create temporary NPZ file
    env = PointMazeEnv()
    images, actions = create_random_trajectory(env, seed=42, T=10)

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(f.name, images=images, actions=actions)

        # Load and check shapes
        data = np.load(f.name)
        assert data['images'].shape[0] == 10, f"Images T dim: {data['images'].shape[0]}"
        assert data['images'].shape[1:] == (64, 64, 3), f"Images spatial dims: {data['images'].shape[1:]}"
        assert data['actions'].shape == (9, 2), f"Actions shape: {data['actions'].shape}"

        os.unlink(f.name)

    print("✓ Dataset shape tests passed")


if __name__ == "__main__":
    # Run all tests
    test_pointmaze_env()
    test_dinov2_encoder()
    test_world_model_shapes()
    test_cem_planner()
    test_dataset_shapes()
    print("\n🎉 All tests passed!")
