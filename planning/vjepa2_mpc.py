import torch
import numpy as np
from typing import Dict, Optional, Any
from planning.cem import CEMPlanner
from planning.costs import CostFunction
from models.dynamics.st_transformer import STTransformer
from models.encoders.vjepa2_encoder import VJEPA2Encoder


class VJEPA2CEMPlanner(CEMPlanner):
    """
    CEM planner adapted for V-JEPA-2 world model with spatio-temporal dynamics.
    """
    
    def __init__(
        self,
        horizon: int,
        topk: int,
        num_samples: int,
        var_scale: float,
        opt_steps: int,
        eval_every: int,
        wm,  # V-JEPA-2 world model
        action_dim: int,
        objective_fn: CostFunction,
        preprocessor,
        evaluator,
        wandb_run,
        log_filename: Optional[str] = None,
        history_length: int = 4,  # Number of history frames to maintain
        **kwargs
    ):
        # Call parent constructor but override some methods
        super().__init__(
            horizon=horizon,
            topk=topk, 
            num_samples=num_samples,
            var_scale=var_scale,
            opt_steps=opt_steps,
            eval_every=eval_every,
            wm=wm,
            action_dim=action_dim,
            objective_fn=objective_fn,
            preprocessor=preprocessor,
            evaluator=evaluator,
            wandb_run=wandb_run,
            log_filename=log_filename,
            **kwargs
        )
        
        self.history_length = history_length
        self.encoder_cache = {}  # Cache for encoder outputs
        self.goal_cache = {}    # Cache for goal encodings
        
    def _encode_with_cache(self, obs: Dict[str, torch.Tensor], cache_key: str) -> torch.Tensor:
        """
        Encode observations with caching.
        
        Args:
            obs: Observation dict with "visual" key
            cache_key: Cache key for storing/retrieving encoding
            
        Returns:
            features: Encoded features
        """
        if cache_key in self.encoder_cache:
            return self.encoder_cache[cache_key].to(obs["visual"].device)
        
        # Extract encoder from world model
        encoder = self.wm.encoder
        if hasattr(encoder, 'forward'):
            features = encoder.forward(obs["visual"], cache_key=cache_key)
        else:
            features = encoder.encode_clip(obs["visual"])
        
        self.encoder_cache[cache_key] = features.cpu()
        return features
    
    def _prepare_history(self, obs_0: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Prepare history features for rollout.
        
        Args:
            obs_0: Initial observations (B, T, 3, H, W)
            
        Returns:
            history_features: (B, H, P, D) history features
        """
        visual = obs_0["visual"]
        B, T, C, H, W = visual.shape
        
        # Ensure we have enough history
        if T < self.history_length:
            # Pad with repeated first frame if needed
            first_frame = visual[:, 0:1].expand(-1, self.history_length - T, -1, -1, -1)
            visual = torch.cat([first_frame, visual], dim=1)
            T = self.history_length
        elif T > self.history_length:
            # Take last history_length frames
            visual = visual[:, -self.history_length:]
            T = self.history_length
        
        # Encode history
        cache_key = "history_" + str(hash(visual.data.cpu().numpy().tobytes()))
        obs_hist = {"visual": visual}
        history_features = self._encode_with_cache(obs_hist, cache_key)  # (B, T, P, D)
        
        return history_features
    
    def rollout(self, obs_0: Dict[str, torch.Tensor], actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Rollout actions using V-JEPA-2 spatio-temporal transformer.
        
        Args:
            obs_0: Initial observations with "visual" key (B, T, 3, H, W)
            actions: Action sequences (B, horizon, action_dim)
            
        Returns:
            z_obs_pred: Dict with "visual" key containing predicted features
        """
        B, horizon, action_dim = actions.shape
        
        # Prepare history features
        history_features = self._prepare_history(obs_0)  # (B, H, P, D)
        
        # Extract dynamics model (STTransformer)
        dynamics = self.wm.predictor  # Assuming world model has predictor attribute
        
        # Perform rollout
        if hasattr(dynamics, 'rollout'):
            # Use built-in rollout method
            pred_features = dynamics.rollout(
                history_features,
                actions, 
                self.history_length
            )  # (B, horizon, P, D)
        else:
            # Manual rollout
            pred_features = []
            current_features = history_features
            
            for t in range(horizon):
                next_features = dynamics(
                    current_features,
                    actions[:, t:t+1]  # (B, 1, action_dim)
                )  # (B, P, D)
                
                pred_features.append(next_features.unsqueeze(1))  # (B, 1, P, D)
                
                # Update sliding window
                current_features = torch.cat([
                    current_features[:, 1:],  # Remove oldest
                    next_features.unsqueeze(1)  # Add newest
                ], dim=1)  # Still (B, H, P, D)
            
            pred_features = torch.cat(pred_features, dim=1)  # (B, horizon, P, D)
        
        # For MPC, we typically care about the final predicted state
        final_pred = pred_features[:, -1]  # (B, P, D) - final frame prediction
        
        # Package in expected format for objective function
        z_obs_pred = {
            "visual": final_pred,
            "proprio": torch.zeros(B, 1, device=final_pred.device)  # Dummy proprio
        }
        
        return z_obs_pred
    
    def plan(
        self, 
        obs_0: Dict[str, torch.Tensor], 
        obs_g: Dict[str, torch.Tensor],
        actions: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Plan actions using CEM with V-JEPA-2 rollouts.
        
        Args:
            obs_0: Initial observations
            obs_g: Goal observations  
            actions: Optional warm-start actions
            
        Returns:
            best_actions: (B, horizon, action_dim) planned actions
            info: Additional info dict
        """
        B = obs_0["visual"].shape[0]
        
        # Cache goal encoding
        goal_cache_key = "goal_" + str(hash(obs_g["visual"].data.cpu().numpy().tobytes()))
        if goal_cache_key not in self.goal_cache:
            if hasattr(self.wm.encoder, 'encode_image'):
                goal_visual = obs_g["visual"]
                if goal_visual.ndim == 5:  # Take first frame if clip
                    goal_visual = goal_visual[:, 0]
                goal_features = self.wm.encoder.encode_image(goal_visual)
            else:
                goal_features = self.wm.encoder(obs_g["visual"])
            self.goal_cache[goal_cache_key] = goal_features.cpu()
        
        goal_features = self.goal_cache[goal_cache_key].to(obs_0["visual"].device)
        
        # Update cost function statistics with current features
        if hasattr(self.objective_fn, 'update_statistics'):
            current_features = self._prepare_history(obs_0)
            self.objective_fn.update_statistics(current_features)
        
        # Use parent CEM planning (which will call our rollout method)
        return super().plan(obs_0, obs_g, actions)
    
    def clear_cache(self):
        """Clear all caches."""
        self.encoder_cache.clear()
        self.goal_cache.clear()
        
        # Also clear encoder cache if it exists
        if hasattr(self.wm.encoder, 'clear_cache'):
            self.wm.encoder.clear_cache()


class VJEPA2WorldModelWrapper:
    """
    Wrapper to make V-JEPA-2 components compatible with existing MPC interface.
    """
    
    def __init__(
        self,
        encoder: VJEPA2Encoder,
        dynamics: STTransformer,
        decoder=None,  # Optional decoder for visualization
    ):
        self.encoder = encoder
        self.predictor = dynamics  # Keep same name as original world model
        self.decoder = decoder
        
        # Compatibility attributes
        self.emb_dim = encoder.emb_dim
        self.num_hist = dynamics.max_seq_length // encoder.patches_per_frame  # Rough estimate
        
    def encode_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode observations (for compatibility)."""
        visual_features = self.encoder.encode_clip(obs["visual"])  # (B, T, P, D)
        
        return {
            "visual": visual_features,
            "proprio": obs.get("proprio", torch.zeros(
                visual_features.shape[0], visual_features.shape[1], 1, 
                device=visual_features.device
            ))
        }
    
    def predict(self, history_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict next frame features."""
        return self.predictor(history_features, actions)
    
    def decode_obs(self, features: Dict[str, torch.Tensor]):
        """Decode features back to observations (optional)."""
        if self.decoder is not None:
            return self.decoder(features)
        else:
            # Return dummy decoded observations
            B, T, P, D = features["visual"].shape
            H = W = int(np.sqrt(P))  # Assume square patches
            dummy_visual = torch.zeros(B, T, 3, H * 16, W * 16, device=features["visual"].device)
            return {"visual": dummy_visual}, 0.0


def create_vjepa2_planner(
    planner_config: Dict[str, Any],
    encoder: VJEPA2Encoder,
    dynamics: STTransformer,
    cost_fn: CostFunction,
    preprocessor,
    evaluator,
    wandb_run,
    decoder=None
) -> VJEPA2CEMPlanner:
    """
    Factory function to create V-JEPA-2 MPC planner.
    
    Args:
        planner_config: Configuration dict with CEM parameters
        encoder: V-JEPA-2 encoder
        dynamics: Spatio-temporal transformer
        cost_fn: Cost function
        preprocessor: Action preprocessor
        evaluator: Environment evaluator
        wandb_run: Wandb run for logging
        decoder: Optional decoder
        
    Returns:
        VJEPA2CEMPlanner instance
    """
    # Wrap components in world model interface
    wm = VJEPA2WorldModelWrapper(encoder, dynamics, decoder)
    
    # Extract planner parameters
    horizon = planner_config.get("horizon", 12)
    topk = planner_config.get("topk", 16)
    num_samples = planner_config.get("num_samples", 128)
    var_scale = planner_config.get("var_scale", 1.0)
    opt_steps = planner_config.get("opt_steps", 6)
    eval_every = planner_config.get("eval_every", 1)
    history_length = planner_config.get("history_length", 4)
    
    # Get action dimension from config or infer
    action_dim = planner_config.get("action_dim", dynamics.action_dim)
    
    return VJEPA2CEMPlanner(
        horizon=horizon,
        topk=topk,
        num_samples=num_samples,
        var_scale=var_scale,
        opt_steps=opt_steps,
        eval_every=eval_every,
        wm=wm,
        action_dim=action_dim,
        objective_fn=cost_fn,
        preprocessor=preprocessor,
        evaluator=evaluator,
        wandb_run=wandb_run,
        history_length=history_length
    )