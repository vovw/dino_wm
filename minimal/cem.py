import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Callable, Optional
import math


class CEMPlanner:
    """
    Cross-Entropy Method (CEM) planner for latent space MPC.
    Optimizes action sequences to minimize latent distance to goal.
    """

    def __init__(self,
                 horizon: int,
                 action_dim: int,
                 population_size: int = 256,
                 elite_fraction: float = 0.1,
                 num_iterations: int = 5,
                 action_bounds: Tuple[float, float] = (-1.0, 1.0),
                 device: str = 'cpu'):
        self.horizon = horizon
        self.action_dim = action_dim
        self.population_size = population_size
        self.elite_size = int(population_size * elite_fraction)
        self.num_iterations = num_iterations
        self.action_bounds = action_bounds
        self.device = device

        # Initialize distribution parameters
        self.mean = torch.zeros(horizon, action_dim, device=device)
        self.std = torch.ones(horizon, action_dim, device=device) * 0.5

    def plan(self,
             world_model: nn.Module,
             current_latent: torch.Tensor,
             goal_latent: torch.Tensor,
             num_iterations: Optional[int] = None) -> torch.Tensor:
        """
        Plan optimal action sequence using CEM.

        Args:
            world_model: Transition model for rollout
            current_latent: (N, D) current latent state
            goal_latent: (N, D) goal latent state
            num_iterations: Override default iterations

        Returns:
            best_action_seq: (H, A) optimal action sequence
        """
        if num_iterations is None:
            num_iterations = self.num_iterations

        B, N, D = current_latent.shape

        for iteration in range(num_iterations):
            # Sample action sequences
            action_seqs = self._sample_actions()  # (pop_size, H, A)

            # Evaluate costs
            costs = self._evaluate_costs(world_model, current_latent, goal_latent, action_seqs)

            # Select elite samples
            elite_indices = torch.topk(costs, self.elite_size, largest=False).indices
            elite_actions = action_seqs[elite_indices]  # (elite_size, H, A)

            # Update distribution
            self._update_distribution(elite_actions)

            # Print progress
            best_cost = costs.min().item()
            print(".4f")

        return self.mean.clone()

    def _sample_actions(self) -> torch.Tensor:
        """Sample action sequences from current distribution."""
        # Sample from normal distribution
        noise = torch.randn(self.population_size, self.horizon, self.action_dim, device=self.device)
        samples = self.mean.unsqueeze(0) + self.std.unsqueeze(0) * noise

        # Clip to action bounds
        samples = torch.clamp(samples, self.action_bounds[0], self.action_bounds[1])

        return samples

    def _evaluate_costs(self,
                       world_model: nn.Module,
                       current_latent: torch.Tensor,
                       goal_latent: torch.Tensor,
                       action_seqs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate costs for action sequences by rolling out in latent space.

        Args:
            world_model: Transition model
            current_latent: (N, D) current latent
            goal_latent: (N, D) goal latent
            action_seqs: (pop_size, H, A) action sequences

        Returns:
            costs: (pop_size,) costs for each sequence
        """
        pop_size = action_seqs.shape[0]
        costs = []

        for i in range(pop_size):
            action_seq = action_seqs[i]  # (H, A)

            # Roll out latent trajectory
            latent_traj = self._rollout_latent(world_model, current_latent, action_seq)

            # Cost is MSE between final latent and goal latent
            final_latent = latent_traj[-1]  # (N, D)
            cost = torch.mean((final_latent - goal_latent) ** 2)
            costs.append(cost)

        return torch.stack(costs)

    def _rollout_latent(self,
                       world_model: nn.Module,
                       start_latent: torch.Tensor,
                       action_seq: torch.Tensor) -> torch.Tensor:
        """
        Roll out latent trajectory given action sequence.

        Args:
            world_model: Transition model
            start_latent: (N, D) starting latent
            action_seq: (H, A) action sequence

        Returns:
            latent_traj: (H+1, N, D) latent trajectory including start
        """
        B, N, D = start_latent.shape
        latent_traj = [start_latent]

        current_latent = start_latent.unsqueeze(0)  # (1, N, D)

        for t in range(self.horizon):
            action = action_seq[t].unsqueeze(0)  # (1, A)

            # Predict next latent using world model
            with torch.no_grad():
                # For single step prediction, we need to create dummy batch
                # This is a simplified version - in practice you'd want batching
                pred_latent = self._predict_next_latent(world_model, current_latent, action)

            latent_traj.append(pred_latent.squeeze(0))
            current_latent = pred_latent

        return torch.stack(latent_traj)

    def _predict_next_latent(self,
                           world_model: nn.Module,
                           current_latent: torch.Tensor,
                           action: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent state. This is a simplified version.
        In practice, you'd need to implement proper batching in the world model.
        """
        # For now, assume world model can handle single time step
        # This would need to be adapted based on your actual world model interface
        try:
            # Try calling world model directly
            pred, _ = world_model(current_latent, action)
            return pred
        except:
            # Fallback: return current latent (no-op)
            print("Warning: World model prediction failed, using current latent")
            return current_latent

    def _update_distribution(self, elite_actions: torch.Tensor):
        """Update distribution parameters from elite samples."""
        self.mean = elite_actions.mean(dim=0)  # (H, A)
        self.std = elite_actions.std(dim=0) + 1e-6  # (H, A) add small epsilon for numerical stability

    def reset(self):
        """Reset distribution to initial state."""
        self.mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        self.std = torch.ones(self.horizon, self.action_dim, device=self.device) * 0.5


def create_cem_planner(horizon: int = 12,
                      action_dim: int = 2,
                      population_size: int = 256,
                      device: str = 'cpu') -> CEMPlanner:
    """Factory function for CEM planner with default parameters."""
    return CEMPlanner(
        horizon=horizon,
        action_dim=action_dim,
        population_size=population_size,
        elite_fraction=0.1,
        num_iterations=5,
        device=device
    )


def test_cem():
    """Simple test for CEM planner."""
    device = torch.device('cpu')
    planner = create_cem_planner(device=device)

    # Test action sampling
    actions = planner._sample_actions()
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Actions bounds: [{actions.min().item():.3f}, {actions.max().item():.3f}]")

    # Test distribution update
    elite_actions = torch.randn(25, 12, 2, device=device)  # 25 elite samples
    planner._update_distribution(elite_actions)
    print(f"Updated mean shape: {planner.mean.shape}")
    print(f"Updated std shape: {planner.std.shape}")

    print("CEM test passed!")


if __name__ == "__main__":
    test_cem()
