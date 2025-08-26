#!/usr/bin/env python3
"""
Planning script using V-JEPA-2 world model.

Usage:
    python plan_vjepa2.py --config-name plan_vjepa2_point_maze
"""

import os
import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

# Import V-JEPA-2 components
from models.vjepa2_world_model import VJEPA2WorldModel
from models.encoders.vjepa2_encoder import VJEPA2Encoder
from models.dynamics.st_transformer import STTransformer
from planning.costs import create_cost_function
from planning.vjepa2_mpc import create_vjepa2_planner

# Import existing environment and evaluation utilities
from env.point_maze import PointMazeEnv
from datasets.point_maze_dset import PointMazeTrajDataset
from metrics.planning_metrics import PlanningEvaluator
from preprocessor import Preprocessor
import wandb


class VJEPA2Planner:
    """V-JEPA-2 planning system."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load world model
        self.world_model = self._load_world_model()
        
        # Set up environment
        self.env = self._setup_environment()
        
        # Set up preprocessor
        self.preprocessor = self._setup_preprocessor()
        
        # Set up evaluator
        self.evaluator = self._setup_evaluator()
        
        # Set up cost function
        self.cost_fn = self._setup_cost_function()
        
        # Set up planner
        self.planner = self._setup_planner()
        
        # Initialize wandb
        if cfg.get("wandb"):
            wandb.init(
                project=cfg.wandb.get("project", "vjepa2_planning"),
                entity=cfg.wandb.get("entity"),
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=cfg.wandb.get("tags", ["vjepa2", "planning"]),
                notes=cfg.wandb.get("notes", "V-JEPA-2 planning evaluation")
            )
    
    def _load_world_model(self) -> VJEPA2WorldModel:
        """Load trained world model."""
        model_path = self.cfg.get("model_path")
        if not model_path or not os.path.exists(model_path):
            print("Warning: No trained model found, using randomly initialized model")
            # Create model with default config
            encoder_config = {
                "variant": "vit-L",
                "resolution": 256,
                "tubelet": {"t": 2, "p": 16},
                "adapter": {"projection_dim": 256, "normalize": "l2", "frameize": True},
                "enable_cache": True
            }
            dynamics_config = {
                "feature_dim": 256,
                "num_layers": 6,
                "num_heads": 8,
                "action_dim": self.cfg.get("action_dim", 7),
                "action_conditioning": "film"
            }
            
            model = VJEPA2WorldModel(
                encoder_config=encoder_config,
                dynamics_config=dynamics_config,
                decoder_config=None,  # No decoder needed for planning
                train_encoder=False,
                train_dynamics=False,  # Set to eval mode for planning
                train_decoder=False
            )
        else:
            # Load from checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint.get("config", {})
            
            # Reconstruct model from saved config
            model = VJEPA2WorldModel(**model_config.get("model", {}))
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from: {model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _setup_environment(self):
        """Set up planning environment."""
        env_name = self.cfg.get("env_name", "point_maze")
        
        if env_name == "point_maze":
            return PointMazeEnv()
        else:
            raise NotImplementedError(f"Environment {env_name} not implemented")
    
    def _setup_preprocessor(self):
        """Set up action preprocessor.""" 
        return Preprocessor()  # Use default preprocessor
    
    def _setup_evaluator(self):
        """Set up planning evaluator."""
        return PlanningEvaluator(
            env=self.env,
            frameskip=self.cfg.get("frameskip", 1),
            # Add other evaluator config
        )
    
    def _setup_cost_function(self):
        """Set up planning cost function."""
        cost_config = self.cfg.get("cost", {})
        
        return create_cost_function({
            "type": cost_config.get("type", "cosine"),
            "goal_type": cost_config.get("goal_type", "single_image"),
            "kwargs": cost_config.get("kwargs", {})
        })
    
    def _setup_planner(self):
        """Set up MPC planner."""
        planner_config = self.cfg.get("planner", {})
        
        return create_vjepa2_planner(
            planner_config=planner_config,
            encoder=self.world_model.encoder,
            dynamics=self.world_model.dynamics,
            cost_fn=self.cost_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,
            wandb_run=wandb.run
        )
    
    def plan_single_trajectory(self, obs_0: Dict[str, torch.Tensor], obs_g: Dict[str, torch.Tensor]):
        """Plan a single trajectory."""
        print(f"Planning trajectory...")
        
        # Run planning
        actions, info = self.planner.plan(obs_0, obs_g)
        
        # Execute and evaluate
        success_rate, metrics = self.evaluator.evaluate_actions(actions, obs_0, obs_g)
        
        print(f"Planning completed. Success rate: {success_rate:.2%}")
        
        return actions, success_rate, metrics
    
    def evaluate_planning(self):
        """Evaluate planning performance on multiple trajectories."""
        num_eval = self.cfg.get("num_eval", 10)
        success_rates = []
        
        print(f"Evaluating planning on {num_eval} trajectories...")
        
        for i in range(num_eval):
            print(f"\nTrajectory {i+1}/{num_eval}")
            
            # Sample random start and goal
            obs_0, obs_g = self._sample_start_goal()
            
            # Plan trajectory
            try:
                actions, success_rate, metrics = self.plan_single_trajectory(obs_0, obs_g)
                success_rates.append(success_rate)
                
                # Log metrics
                if wandb.run:
                    wandb.log({
                        "trajectory": i,
                        "success_rate": success_rate,
                        **metrics
                    })
                    
            except Exception as e:
                print(f"Planning failed for trajectory {i}: {e}")
                success_rates.append(0.0)
        
        # Final statistics
        mean_success = np.mean(success_rates)
        std_success = np.std(success_rates)
        
        print(f"\nPlanning Evaluation Results:")
        print(f"Mean Success Rate: {mean_success:.2%} ± {std_success:.2%}")
        print(f"Success Rates: {success_rates}")
        
        if wandb.run:
            wandb.log({
                "final/mean_success_rate": mean_success,
                "final/std_success_rate": std_success,
                "final/min_success_rate": np.min(success_rates),
                "final/max_success_rate": np.max(success_rates)
            })
        
        return success_rates
    
    def _sample_start_goal(self):
        """Sample random start and goal states."""
        # This is a simplified version - replace with actual environment sampling
        B = 1  # Single trajectory
        H, W = 224, 224
        
        # Create dummy observations
        obs_0 = {
            "visual": torch.randn(B, 4, 3, H, W),  # 4 frames of history
            "proprio": torch.randn(B, 4, 4)
        }
        
        obs_g = {
            "visual": torch.randn(B, 3, H, W),  # Single goal image
            "proprio": torch.randn(B, 4)
        }
        
        return obs_0, obs_g
    
    def visualize_planning(self, save_path: str = "planning_visualization"):
        """Create planning visualization."""
        print(f"Creating planning visualization...")
        
        # Sample a trajectory
        obs_0, obs_g = self._sample_start_goal()
        
        # Plan with visualization
        actions, success_rate, metrics = self.plan_single_trajectory(obs_0, obs_g)
        
        # Create rollout visualization using world model
        with torch.no_grad():
            rollout_results = self.world_model.rollout(
                obs_0, actions, decode_images=bool(self.world_model.decoder)
            )
        
        # Save results
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        torch.save({
            "obs_0": obs_0,
            "obs_g": obs_g,
            "actions": actions,
            "rollout_results": rollout_results,
            "success_rate": success_rate,
            "metrics": metrics
        }, save_dir / "planning_results.pth")
        
        print(f"Visualization saved to: {save_dir}")


@hydra.main(version_base=None, config_path="conf", config_name="plan_vjepa2")
def main(cfg: DictConfig):
    """Main planning function."""
    print("V-JEPA-2 Planning Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seeds
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    
    # Create planner
    planner = VJEPA2Planner(cfg)
    
    # Run evaluation
    mode = cfg.get("mode", "evaluate")  # "evaluate", "single", "visualize"
    
    if mode == "evaluate":
        planner.evaluate_planning()
    elif mode == "single":
        obs_0, obs_g = planner._sample_start_goal()
        planner.plan_single_trajectory(obs_0, obs_g)
    elif mode == "visualize":
        planner.visualize_planning()
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()