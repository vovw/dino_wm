import pytest
import torch
import torch.nn as nn
import numpy as np
from planning.costs import (
    MSECost, CosineCost, MahalanobisCost, PredictorEnergyCost,
    GoalHandler, CostFunction, create_cost_function
)


class TestBasicCosts:
    """Test basic cost functions."""
    
    @pytest.fixture
    def sample_latents(self):
        """Sample latents for testing."""
        B, P, D = 4, 64, 256
        pred_latents = torch.randn(B, P, D)
        goal_latents = torch.randn(B, P, D)
        return pred_latents, goal_latents
    
    def test_mse_cost(self, sample_latents):
        """Test MSE cost function."""
        pred_latents, goal_latents = sample_latents
        cost_fn = MSECost()
        
        costs = cost_fn(pred_latents, goal_latents)
        assert costs.shape == (pred_latents.shape[0],)
        assert torch.all(costs >= 0), "MSE costs should be non-negative"
        
        # Test that identical latents give zero cost
        zero_costs = cost_fn(pred_latents, pred_latents)
        assert torch.allclose(zero_costs, torch.zeros_like(zero_costs), atol=1e-6)
    
    def test_cosine_cost(self, sample_latents):
        """Test cosine distance cost function."""
        pred_latents, goal_latents = sample_latents
        cost_fn = CosineCost()
        
        costs = cost_fn(pred_latents, goal_latents)
        assert costs.shape == (pred_latents.shape[0],)
        assert torch.all(costs >= 0), "Cosine costs should be non-negative"
        assert torch.all(costs <= 2), "Cosine costs should be <= 2"
        
        # Test that identical latents give zero cost
        zero_costs = cost_fn(pred_latents, pred_latents)
        assert torch.allclose(zero_costs, torch.zeros_like(zero_costs), atol=1e-6)
    
    def test_mahalanobis_cost(self, sample_latents):
        """Test Mahalanobis distance cost function."""
        pred_latents, goal_latents = sample_latents
        B, P, D = pred_latents.shape
        
        cost_fn = MahalanobisCost(P * D)
        
        # Update covariance with some data
        cost_fn.update_covariance(pred_latents)
        cost_fn.update_covariance(goal_latents)
        
        costs = cost_fn(pred_latents, goal_latents)
        assert costs.shape == (B,)
        assert torch.all(costs >= 0), "Mahalanobis costs should be non-negative"
        assert torch.all(torch.isfinite(costs)), "Costs should be finite"
    
    def test_predictor_energy_cost(self, sample_latents):
        """Test predictor energy cost function."""
        pred_latents, goal_latents = sample_latents
        B, P, D = pred_latents.shape
        
        cost_fn = PredictorEnergyCost(D, hidden_dim=128)
        
        costs = cost_fn(pred_latents, goal_latents)
        assert costs.shape == (B,)
        assert torch.all(torch.isfinite(costs)), "Costs should be finite"


class TestGoalHandler:
    """Test goal handling and processing."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Mock encoder for testing."""
        class MockEncoder:
            def encode_image(self, image):
                B, C, H, W = image.shape
                P = (H // 16) ** 2  # Assume 16x16 patches
                D = 256
                return torch.randn(B, P, D)
            
            def encode_clip(self, clip):
                B, T, C, H, W = clip.shape
                P = (H // 16) ** 2
                D = 256
                return torch.randn(B, T, P, D)
        
        return MockEncoder()
    
    def test_single_image_goal(self, mock_encoder):
        """Test single image goal processing."""
        handler = GoalHandler("single_image")
        
        # Test with single image
        B, H, W = 2, 224, 224
        goal_obs = {"visual": torch.randn(B, 3, H, W)}
        
        goal_latents = handler.process_goal(goal_obs, mock_encoder)
        P = (H // 16) ** 2
        assert goal_latents.shape == (B, P, 256)
        
        # Test with clip (should take first frame)
        T = 4
        goal_obs_clip = {"visual": torch.randn(B, T, 3, H, W)}
        goal_latents_clip = handler.process_goal(goal_obs_clip, mock_encoder)
        assert goal_latents_clip.shape == (B, P, 256)
    
    def test_clip_average_goal(self, mock_encoder):
        """Test clip average goal processing."""
        handler = GoalHandler("clip_average")
        
        B, T, H, W = 2, 4, 224, 224
        P = (H // 16) ** 2
        
        # Test with clip
        goal_obs = {"visual": torch.randn(B, T, 3, H, W)}
        goal_latents = handler.process_goal(goal_obs, mock_encoder)
        assert goal_latents.shape == (B, P, 256)
        
        # Test with single image (should expand)
        goal_obs_single = {"visual": torch.randn(B, 3, H, W)}
        goal_latents_single = handler.process_goal(goal_obs_single, mock_encoder)
        assert goal_latents_single.shape == (B, P, 256)
    
    def test_invalid_goal_type(self, mock_encoder):
        """Test invalid goal type raises error."""
        with pytest.raises(ValueError):
            handler = GoalHandler("invalid_type")


class TestCostFunction:
    """Test unified cost function."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Mock encoder for testing."""
        class MockEncoder:
            def __init__(self):
                self.call_count = 0
                
            def encode_image(self, image):
                self.call_count += 1
                B, C, H, W = image.shape
                P = (H // 16) ** 2
                D = 256
                return torch.randn(B, P, D)
            
            def encode_clip(self, clip):
                self.call_count += 1
                B, T, C, H, W = clip.shape
                P = (H // 16) ** 2
                D = 256
                return torch.randn(B, T, P, D)
        
        return MockEncoder()
    
    def test_mse_cost_function(self, mock_encoder):
        """Test MSE cost function integration."""
        cost_fn = CostFunction(cost_type="mse", goal_type="single_image")
        
        B, P, D = 2, 64, 256
        pred_latents = torch.randn(B, P, D)
        goal_obs = {"visual": torch.randn(B, 3, 224, 224)}
        
        costs = cost_fn(pred_latents, goal_obs, mock_encoder)
        assert costs.shape == (B,)
        assert torch.all(costs >= 0)
    
    def test_cosine_cost_function(self, mock_encoder):
        """Test cosine cost function integration."""
        cost_fn = CostFunction(cost_type="cosine", goal_type="single_image")
        
        B, P, D = 2, 64, 256
        pred_latents = torch.randn(B, P, D)
        goal_obs = {"visual": torch.randn(B, 3, 224, 224)}
        
        costs = cost_fn(pred_latents, goal_obs, mock_encoder)
        assert costs.shape == (B,)
        assert torch.all(costs >= 0)
    
    def test_mahalanobis_cost_function(self, mock_encoder):
        """Test Mahalanobis cost function integration."""
        cost_kwargs = {"momentum": 0.1, "eps": 1e-4}
        cost_fn = CostFunction(
            cost_type="mahalanobis", 
            goal_type="single_image",
            cost_kwargs=cost_kwargs
        )
        
        B, P, D = 2, 64, 256
        
        # Update statistics first
        sample_features = torch.randn(10, P, D)
        cost_fn.update_statistics(sample_features)
        
        pred_latents = torch.randn(B, P, D)
        goal_obs = {"visual": torch.randn(B, 3, 224, 224)}
        
        costs = cost_fn(pred_latents, goal_obs, mock_encoder)
        assert costs.shape == (B,)
        assert torch.all(costs >= 0)
        assert torch.all(torch.isfinite(costs))
    
    def test_predictor_energy_cost_function(self, mock_encoder):
        """Test predictor energy cost function integration."""
        cost_kwargs = {"hidden_dim": 128}
        cost_fn = CostFunction(
            cost_type="predictor_energy",
            goal_type="single_image",
            cost_kwargs=cost_kwargs
        )
        
        B, P, D = 2, 64, 256
        pred_latents = torch.randn(B, P, D)
        goal_obs = {"visual": torch.randn(B, 3, 224, 224)}
        
        costs = cost_fn(pred_latents, goal_obs, mock_encoder)
        assert costs.shape == (B,)
        assert torch.all(torch.isfinite(costs))
    
    def test_caching(self, mock_encoder):
        """Test goal caching functionality.""" 
        cost_fn = CostFunction(cost_type="mse", goal_type="single_image")
        
        B, P, D = 2, 64, 256
        pred_latents = torch.randn(B, P, D)
        goal_obs = {"visual": torch.randn(B, 3, 224, 224)}
        
        # First call
        costs1 = cost_fn(pred_latents, goal_obs, mock_encoder, cache_key="test")
        initial_calls = mock_encoder.call_count
        
        # Second call with same cache key - should not call encoder again
        costs2 = cost_fn(pred_latents, goal_obs, mock_encoder, cache_key="test")
        assert mock_encoder.call_count == initial_calls, "Encoder should not be called again with caching"


class TestCostFunctionFactory:
    """Test cost function factory."""
    
    def test_create_mse_cost(self):
        """Test creating MSE cost function."""
        config = {
            "type": "mse",
            "goal_type": "single_image"
        }
        cost_fn = create_cost_function(config)
        assert isinstance(cost_fn, CostFunction)
        assert cost_fn.cost_type == "mse"
    
    def test_create_cosine_cost(self):
        """Test creating cosine cost function."""
        config = {
            "type": "cosine", 
            "goal_type": "clip_average"
        }
        cost_fn = create_cost_function(config)
        assert isinstance(cost_fn, CostFunction)
        assert cost_fn.cost_type == "cosine"
    
    def test_create_mahalanobis_cost(self):
        """Test creating Mahalanobis cost function."""
        config = {
            "type": "mahalanobis",
            "goal_type": "single_image", 
            "kwargs": {"momentum": 0.05, "eps": 1e-6}
        }
        cost_fn = create_cost_function(config)
        assert isinstance(cost_fn, CostFunction)
        assert cost_fn.cost_type == "mahalanobis"
    
    def test_create_predictor_energy_cost(self):
        """Test creating predictor energy cost function."""
        config = {
            "type": "predictor_energy",
            "goal_type": "single_image",
            "kwargs": {"hidden_dim": 256}
        }
        cost_fn = create_cost_function(config)
        assert isinstance(cost_fn, CostFunction)
        assert cost_fn.cost_type == "predictor_energy"
    
    def test_default_values(self):
        """Test factory with default values."""
        config = {}
        cost_fn = create_cost_function(config)
        assert cost_fn.cost_type == "mse"
        assert cost_fn.goal_handler.goal_type == "single_image"
    
    def test_invalid_cost_type(self):
        """Test invalid cost type raises error."""
        config = {"type": "invalid_cost"}
        with pytest.raises(ValueError):
            create_cost_function(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])