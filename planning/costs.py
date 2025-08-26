import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod


class BaseCost(ABC):
    """Base class for planning cost functions."""
    
    @abstractmethod
    def __call__(self, pred_latents: torch.Tensor, goal_latents: torch.Tensor) -> torch.Tensor:
        """
        Compute cost between predicted and goal latents.
        
        Args:
            pred_latents: (B, P, D) predicted final frame latents
            goal_latents: (B, P, D) goal frame latents
            
        Returns:
            costs: (B,) cost for each trajectory
        """
        pass


class MSECost(BaseCost):
    """Mean squared error cost."""
    
    def __call__(self, pred_latents: torch.Tensor, goal_latents: torch.Tensor) -> torch.Tensor:
        # MSE over patches and features, then mean
        mse = F.mse_loss(pred_latents, goal_latents, reduction='none')  # (B, P, D)
        return mse.mean(dim=(1, 2))  # (B,)


class CosineCost(BaseCost):
    """Cosine distance cost (1 - cosine_similarity)."""
    
    def __call__(self, pred_latents: torch.Tensor, goal_latents: torch.Tensor) -> torch.Tensor:
        # Flatten patches and features
        pred_flat = pred_latents.view(pred_latents.size(0), -1)  # (B, P*D)
        goal_flat = goal_latents.view(goal_latents.size(0), -1)  # (B, P*D)
        
        # Cosine similarity
        cosine_sim = F.cosine_similarity(pred_flat, goal_flat, dim=1)  # (B,)
        
        # Convert to distance (cost)
        return 1.0 - cosine_sim  # (B,)


class MahalanobisCost(BaseCost):
    """Mahalanobis distance cost with learned covariance."""
    
    def __init__(self, feature_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.eps = eps
        
        # Running covariance estimate
        self.register_buffer('running_cov', torch.eye(feature_dim))
        self.register_buffer('inv_cov', torch.eye(feature_dim))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.long))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register buffer - simplified version for standalone class."""
        setattr(self, name, tensor)
    
    def update_covariance(self, features: torch.Tensor):
        """
        Update running covariance estimate.
        
        Args:
            features: (B, P, D) or (N, D) features
        """
        if features.ndim > 2:
            features = features.view(-1, features.size(-1))  # (N, D)
        
        N, D = features.shape
        if N == 1:
            return  # Skip single samples
        
        # Compute sample covariance
        features_centered = features - features.mean(dim=0, keepdim=True)
        sample_cov = torch.mm(features_centered.T, features_centered) / (N - 1)
        
        # Update running covariance
        if self.num_updates == 0:
            self.running_cov = sample_cov
        else:
            self.running_cov = (1 - self.momentum) * self.running_cov + self.momentum * sample_cov
        
        # Update inverse covariance
        regularized_cov = self.running_cov + self.eps * torch.eye(
            D, device=self.running_cov.device
        )
        try:
            self.inv_cov = torch.inverse(regularized_cov)
        except RuntimeError:  # Singular matrix
            self.inv_cov = torch.pinverse(regularized_cov)
        
        self.num_updates += 1
    
    def __call__(self, pred_latents: torch.Tensor, goal_latents: torch.Tensor) -> torch.Tensor:
        # Flatten to (B, P*D)
        pred_flat = pred_latents.view(pred_latents.size(0), -1)
        goal_flat = goal_latents.view(goal_latents.size(0), -1)
        
        # Difference vector
        diff = pred_flat - goal_flat  # (B, P*D)
        
        # Mahalanobis distance: sqrt(diff^T @ inv_cov @ diff)
        mahal_dist_sq = torch.sum(diff * torch.mm(diff, self.inv_cov), dim=1)  # (B,)
        return torch.sqrt(torch.clamp(mahal_dist_sq, min=self.eps))


class PredictorEnergyCost(BaseCost):
    """Learned predictor energy for measuring alignment between predictions and goals."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # Concat pred and goal
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Output energy
        )
        
    def __call__(self, pred_latents: torch.Tensor, goal_latents: torch.Tensor) -> torch.Tensor:
        # Average pool over patches
        pred_pooled = pred_latents.mean(dim=1)  # (B, D)
        goal_pooled = goal_latents.mean(dim=1)  # (B, D)
        
        # Concatenate and compute energy
        combined = torch.cat([pred_pooled, goal_pooled], dim=-1)  # (B, 2*D)
        energy = self.energy_net(combined).squeeze(-1)  # (B,)
        
        # Return negative energy as cost (higher energy = lower cost)
        return -energy


class GoalHandler:
    """Handles different goal formats and preprocessing."""
    
    def __init__(self, goal_type: str = "single_image"):
        """
        Args:
            goal_type: "single_image" or "clip_average"
        """
        self.goal_type = goal_type
    
    def process_goal(self, goal_obs: Dict[str, torch.Tensor], encoder) -> torch.Tensor:
        """
        Process goal observation into goal latents.
        
        Args:
            goal_obs: Goal observation dict with "visual" key
            encoder: Encoder to extract features
            
        Returns:
            goal_latents: (B, P, D) goal latents
        """
        goal_visual = goal_obs["visual"]  # (B, T, 3, H, W) or (B, 3, H, W)
        
        if self.goal_type == "single_image":
            if goal_visual.ndim == 5:  # Take first frame if clip
                goal_visual = goal_visual[:, 0]  # (B, 3, H, W)
            goal_latents = encoder.encode_image(goal_visual)  # (B, P, D)
            
        elif self.goal_type == "clip_average":
            if goal_visual.ndim == 4:  # Expand single image to clip
                goal_visual = goal_visual.unsqueeze(1)  # (B, 1, 3, H, W)
            goal_clip_latents = encoder.encode_clip(goal_visual)  # (B, T, P, D)
            goal_latents = goal_clip_latents.mean(dim=1)  # (B, P, D) - average over time
            
        else:
            raise ValueError(f"Unknown goal_type: {self.goal_type}")
            
        return goal_latents


class CostFunction:
    """
    Unified cost function that handles different cost types and goal processing.
    """
    
    def __init__(
        self,
        cost_type: str = "mse",
        goal_type: str = "single_image",
        cost_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            cost_type: "mse", "cosine", "mahalanobis", "predictor_energy"
            goal_type: "single_image", "clip_average"
            cost_kwargs: Additional kwargs for cost function
        """
        self.cost_type = cost_type
        self.goal_handler = GoalHandler(goal_type)
        cost_kwargs = cost_kwargs or {}
        
        # Create cost function
        if cost_type == "mse":
            self.cost_fn = MSECost()
        elif cost_type == "cosine":
            self.cost_fn = CosineCost()
        elif cost_type == "mahalanobis":
            self.cost_fn = MahalanobisCost(**cost_kwargs)
        elif cost_type == "predictor_energy":
            self.cost_fn = PredictorEnergyCost(**cost_kwargs)
        else:
            raise ValueError(f"Unknown cost_type: {cost_type}")
    
    def update_statistics(self, features: torch.Tensor):
        """Update cost function statistics (e.g., covariance for Mahalanobis)."""
        if hasattr(self.cost_fn, 'update_covariance'):
            self.cost_fn.update_covariance(features)
    
    def __call__(
        self, 
        pred_latents: torch.Tensor, 
        goal_obs: Dict[str, torch.Tensor], 
        encoder,
        cache_key: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute planning cost.
        
        Args:
            pred_latents: (B, P, D) predicted final frame latents
            goal_obs: Goal observation dict
            encoder: Encoder for processing goal
            cache_key: Optional cache key for goal encoding
            
        Returns:
            costs: (B,) cost for each trajectory
        """
        # Process goal (with caching)
        if cache_key is not None:
            cache_key = f"goal_{cache_key}"
        
        goal_latents = self.goal_handler.process_goal(goal_obs, encoder)
        
        # Compute cost
        return self.cost_fn(pred_latents, goal_latents)


def create_cost_function(cost_config: Dict[str, Any]) -> CostFunction:
    """
    Factory function to create cost function from configuration.
    
    Args:
        cost_config: Configuration dict with keys:
            - type: cost type ("mse", "cosine", "mahalanobis", "predictor_energy")
            - goal_type: goal type ("single_image", "clip_average") 
            - kwargs: additional kwargs for cost function
            
    Returns:
        CostFunction instance
    """
    cost_type = cost_config.get("type", "mse")
    goal_type = cost_config.get("goal_type", "single_image")
    cost_kwargs = cost_config.get("kwargs", {})
    
    return CostFunction(
        cost_type=cost_type,
        goal_type=goal_type,
        cost_kwargs=cost_kwargs
    )