#!/usr/bin/env python3
"""
Training script for V-JEPA-2 world model.

Usage:
    python train_vjepa2_world_model.py --config-name vjepa2_world_model
"""

import os
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import wandb
from typing import Dict, Any
import numpy as np
from pathlib import Path

# Import model components
from models.vjepa2_world_model import VJEPA2WorldModel
from datasets.clip_dataset import create_clip_dataloader

# Import existing dataset loaders
from datasets.point_maze_dset import PointMazeTrajDataset
from datasets.pusht_dset import PushTTrajDataset
from datasets.wall_dset import WallTrajDataset


class VJEPA2Trainer:
    """Trainer for V-JEPA-2 world model."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize wandb
        if cfg.get("wandb"):
            wandb.init(
                project=cfg.wandb.get("project", "vjepa2_world_model"),
                entity=cfg.wandb.get("entity"),
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=cfg.wandb.get("tags", []),
                notes=cfg.wandb.get("notes", "")
            )
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Build datasets and dataloaders
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Mixed precision training
        self.use_amp = cfg.get("amp", {}).get("enabled", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
    def _build_model(self) -> VJEPA2WorldModel:
        """Build V-JEPA-2 world model."""
        # Extract configurations
        encoder_config = OmegaConf.to_container(self.cfg.encoder, resolve=True)
        encoder_config.pop("_target_", None)
        
        dynamics_config = OmegaConf.to_container(self.cfg.predictor, resolve=True)
        dynamics_config.pop("_target_", None)
        
        decoder_config = None
        if self.cfg.get("decoder") and self.cfg.decoder is not None:
            decoder_config = OmegaConf.to_container(self.cfg.decoder, resolve=True)
            decoder_config.pop("_target_", None)
        
        # Model configuration
        model_config = self.cfg.get("model", {})
        
        model = VJEPA2WorldModel(
            encoder_config=encoder_config,
            dynamics_config=dynamics_config,
            decoder_config=decoder_config,
            history_length=model_config.get("history_length", 4),
            prediction_horizon=model_config.get("prediction_horizon", 1),
            train_encoder=False,  # Keep encoder frozen
            train_dynamics=True,
            train_decoder=True,
        )
        
        return model
        
    def _build_dataloaders(self) -> tuple:
        """Build training and validation dataloaders."""
        # This is a simplified version - in practice you'd load your specific datasets
        # For demonstration, create mock data
        
        class MockTrajDataset:
            def __init__(self, num_trajs=100, traj_length=50):
                self.num_trajs = num_trajs
                self.traj_length = traj_length
                self.proprio_dim = 4
                self.action_dim = 7
                self.state_dim = 4
                
            def __len__(self):
                return self.num_trajs
            
            def get_seq_length(self, idx):
                return self.traj_length
            
            def __getitem__(self, idx):
                T = self.traj_length
                obs = {
                    'visual': torch.randn(T, 3, 224, 224),
                    'proprio': torch.randn(T, self.proprio_dim)
                }
                actions = torch.randn(T, self.action_dim)
                states = torch.randn(T, self.state_dim)
                return obs, actions, states
        
        # Create mock datasets
        train_traj_dataset = MockTrajDataset(num_trajs=200, traj_length=50)
        val_traj_dataset = MockTrajDataset(num_trajs=50, traj_length=50)
        
        # Create clip dataloaders
        data_config = self.cfg.get("data", {})
        model_config = self.cfg.get("model", {})
        
        train_loader = create_clip_dataloader(
            train_traj_dataset,
            batch_size=self.cfg.batch_size,
            history_length=model_config.get("history_length", 4),
            frameskip=model_config.get("frameskip", 1),
            tubelet_t=self.cfg.encoder.get("tubelet", {}).get("t", 2),
            num_workers=data_config.get("num_workers", 4),
            shuffle=True,
            augment_temporal=data_config.get("augment_temporal", True),
            pad_mode=data_config.get("pad_mode", "repeat"),
            min_traj_length=data_config.get("min_traj_length", 8)
        )
        
        val_loader = create_clip_dataloader(
            val_traj_dataset,
            batch_size=self.cfg.batch_size,
            history_length=model_config.get("history_length", 4),
            frameskip=model_config.get("frameskip", 1),
            tubelet_t=self.cfg.encoder.get("tubelet", {}).get("t", 2),
            num_workers=data_config.get("num_workers", 4),
            shuffle=False,
            augment_temporal=False,  # No augmentation for validation
            pad_mode=data_config.get("pad_mode", "repeat"),
            min_traj_length=data_config.get("min_traj_length", 8)
        )
        
        return train_loader, val_loader
    
    def _build_optimizer(self):
        """Build optimizer."""
        optimizer_config = self.cfg.get("optimizer", {})
        name = optimizer_config.get("name", "adamw").lower()
        
        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.get("weight_decay", 1e-5),
                betas=optimizer_config.get("betas", [0.9, 0.95]),
                eps=optimizer_config.get("eps", 1e-8)
            )
        elif name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.get("weight_decay", 1e-5)
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler_config = self.cfg.get("scheduler", {})
        name = scheduler_config.get("name", "cosine").lower()
        
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get("T_max", self.cfg.num_epochs),
                eta_min=scheduler_config.get("eta_min", 1e-6)
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        elif name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {name}")
    
    def train_step(self, batch) -> Dict[str, float]:
        """Single training step."""
        history_obs, history_actions, next_frame = batch
        
        # Move to device
        for key in history_obs:
            history_obs[key] = history_obs[key].to(self.device)
        for key in next_frame:
            next_frame[key] = next_frame[key].to(self.device)
        history_actions = history_actions.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast():
                _, _, _, total_loss, loss_components = self.model(
                    history_obs, history_actions, next_frame
                )
            
            # Backward pass with gradient scaling
            self.scaler.scale(total_loss).backward()
            
            # Gradient clipping
            if self.cfg.get("gradient_clip_norm"):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.gradient_clip_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            _, _, _, total_loss, loss_components = self.model(
                history_obs, history_actions, next_frame
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.cfg.get("gradient_clip_norm"):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.gradient_clip_norm
                )
            
            self.optimizer.step()
        
        # Convert losses to float
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_components.items()}
    
    @torch.no_grad()
    def val_step(self, batch) -> Dict[str, float]:
        """Single validation step."""
        history_obs, history_actions, next_frame = batch
        
        # Move to device
        for key in history_obs:
            history_obs[key] = history_obs[key].to(self.device)
        for key in next_frame:
            next_frame[key] = next_frame[key].to(self.device)
        history_actions = history_actions.to(self.device)
        
        # Forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast():
                _, _, _, total_loss, loss_components = self.model(
                    history_obs, history_actions, next_frame
                )
        else:
            _, _, _, total_loss, loss_components = self.model(
                history_obs, history_actions, next_frame
            )
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_components.items()}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            losses = self.train_step(batch)
            
            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1
            self.step += 1
            
            # Logging
            if batch_idx % self.cfg.logging.get("log_every", 100) == 0:
                print(f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {losses['total_loss']:.4f}")
                
                if wandb.run:
                    wandb.log({f"train/{k}": v for k, v in losses.items()}, step=self.step)
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_losses = {}
        num_batches = 0
        
        for batch in self.val_loader:
            losses = self.val_step(batch)
            
            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "epoch": self.epoch,
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "config": OmegaConf.to_container(self.cfg, resolve=True)
        }
        
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")
    
    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.cfg.num_epochs} epochs...")
        
        for epoch in range(self.cfg.num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            print(f"Epoch {epoch}: Train Loss: {train_losses['total_loss']:.4f}, "
                  f"Val Loss: {val_losses['total_loss']:.4f}")
            
            if wandb.run:
                wandb.log({
                    "epoch": epoch,
                    **{f"train_epoch/{k}": v for k, v in train_losses.items()},
                    **{f"val_epoch/{k}": v for k, v in val_losses.items()},
                    "lr": self.optimizer.param_groups[0]["lr"]
                }, step=self.step)
            
            # Save checkpoint
            val_loss = val_losses["total_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pth")
            
            if (epoch + 1) % self.cfg.logging.get("save_every_epochs", 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
        
        print("Training completed!")


@hydra.main(version_base=None, config_path="conf", config_name="vjepa2_world_model")
def main(cfg: DictConfig):
    """Main training function."""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create trainer and start training
    trainer = VJEPA2Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()