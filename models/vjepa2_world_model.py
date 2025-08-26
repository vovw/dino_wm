import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from einops import rearrange

from models.encoders.vjepa2_encoder import VJEPA2Encoder
from models.dynamics.st_transformer import STTransformer
from models.decoder.vjepa2_decoder import create_vjepa2_decoder


class VJEPA2WorldModel(nn.Module):
    """
    Complete V-JEPA-2 world model integrating encoder, dynamics, and decoder.
    
    This model processes video clips through:
    1. V-JEPA-2 encoder: clips -> spatio-temporal features
    2. ST-Transformer: predict next frame features from history + actions
    3. Optional decoder: features -> reconstructed images
    """
    
    def __init__(
        self,
        encoder_config: Dict[str, Any],
        dynamics_config: Dict[str, Any],
        decoder_config: Optional[Dict[str, Any]] = None,
        history_length: int = 4,
        prediction_horizon: int = 1,
        train_encoder: bool = False,  # Keep encoder frozen by default
        train_dynamics: bool = True,
        train_decoder: bool = True,
    ):
        super().__init__()
        
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.train_encoder = train_encoder
        self.train_dynamics = train_dynamics
        self.train_decoder = train_decoder
        
        # Build encoder
        self.encoder = VJEPA2Encoder(**encoder_config)
        
        # Build dynamics model
        # Ensure action_dim is consistent
        dynamics_config = dynamics_config.copy()
        dynamics_config["feature_dim"] = self.encoder.emb_dim
        self.dynamics = STTransformer(**dynamics_config)
        
        # Build optional decoder
        if decoder_config is not None:
            decoder_config = decoder_config.copy()
            decoder_config["feature_dim"] = self.encoder.emb_dim
            decoder_config["patch_size"] = self.encoder.patch_size
            self.decoder = create_vjepa2_decoder(**decoder_config)
        else:
            self.decoder = None
        
        # Loss functions
        self.prediction_loss = nn.MSELoss()
        self.reconstruction_loss = nn.MSELoss() if self.decoder else None
        
        # Set training modes
        self._update_training_modes()
        
        print(f"VJEPA2WorldModel initialized:")
        print(f"  Encoder: {self.encoder.variant}, dim={self.encoder.emb_dim}")
        print(f"  Dynamics: {self.dynamics.num_layers} layers, {self.dynamics.num_heads} heads")
        print(f"  Decoder: {'Yes' if self.decoder else 'No'}")
        print(f"  History length: {history_length}")
    
    def _update_training_modes(self):
        """Update training modes for different components."""
        if not self.train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if not self.train_dynamics:
            self.dynamics.eval()
            for param in self.dynamics.parameters():
                param.requires_grad = False
                
        if self.decoder and not self.train_decoder:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False
    
    def train(self, mode: bool = True):
        """Override train to respect component-specific training flags."""
        super().train(mode)
        self._update_training_modes()
        return self
    
    def encode_history(self, history_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode history observations to features.
        
        Args:
            history_obs: Dict with "visual" key (B, H, 3, H, W)
            
        Returns:
            history_features: (B, H, P, D) encoded features
        """
        visual = history_obs["visual"]  # (B, H, 3, H, W)
        
        # Use encoder caching for efficiency during planning
        cache_key = f"history_{hash(visual.data.cpu().numpy().tobytes())}"
        features = self.encoder(visual, cache_key=cache_key)  # (B, H, P, D)
        
        return features
    
    def predict_next_features(
        self, 
        history_features: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next frame features from history and actions.
        
        Args:
            history_features: (B, H, P, D) history features
            actions: (B, H, action_dim) actions corresponding to history
            
        Returns:
            next_features: (B, P, D) predicted next frame features
        """
        return self.dynamics(history_features, actions)
    
    def decode_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Decode features back to images.
        
        Args:
            features: (B, P, D) or (B, T, P, D) features
            
        Returns:
            images: Decoded images
            decoder_loss: Decoder-specific loss (e.g., VQ loss)
        """
        if self.decoder is None:
            # Return dummy images
            if features.ndim == 3:  # (B, P, D)
                B, P, D = features.shape
                dummy_images = torch.zeros(B, 3, 224, 224, device=features.device)
            else:  # (B, T, P, D)
                B, T, P, D = features.shape
                dummy_images = torch.zeros(B, T, 3, 224, 224, device=features.device)
            return dummy_images, 0.0
        
        return self.decoder(features)
    
    def forward(
        self, 
        history_obs: Dict[str, torch.Tensor],
        history_actions: torch.Tensor,
        target_obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for training.
        
        Args:
            history_obs: History observations with "visual" key (B, H, 3, H, W)
            history_actions: History actions (B, H, action_dim)
            target_obs: Target observations for prediction (B, 1, 3, H, W)
            
        Returns:
            predicted_features: (B, P, D) predicted next frame features
            predicted_images: (B, 3, H, W) predicted next frame images (if decoder)
            reconstructed_images: (B, H, 3, H, W) reconstructed history images (if decoder)
            total_loss: Total training loss
            loss_components: Dict of individual loss components
        """
        # Encode history
        history_features = self.encode_history(history_obs)  # (B, H, P, D)
        
        # Predict next frame features
        predicted_features = self.predict_next_features(history_features, history_actions)  # (B, P, D)
        
        # Encode target for supervision
        target_visual = target_obs["visual"]  # (B, 1, 3, H, W)
        if target_visual.shape[1] == 1:
            target_visual = target_visual.squeeze(1)  # (B, 3, H, W)
        
        target_features = self.encoder.encode_image(target_visual)  # (B, P, D)
        
        # Compute prediction loss
        prediction_loss = self.prediction_loss(predicted_features, target_features.detach())
        
        loss_components = {
            "prediction_loss": prediction_loss,
        }
        total_loss = prediction_loss
        
        # Optional reconstruction
        predicted_images = None
        reconstructed_images = None
        
        if self.decoder is not None:
            # Decode predicted features
            predicted_images, pred_decoder_loss = self.decode_features(predicted_features)
            
            # Decode history for reconstruction loss
            reconstructed_images, recon_decoder_loss = self.decode_features(history_features)
            
            # Reconstruction loss on history
            target_history = history_obs["visual"]  # (B, H, 3, H, W)
            reconstruction_loss = self.reconstruction_loss(reconstructed_images, target_history)
            
            loss_components.update({
                "reconstruction_loss": reconstruction_loss,
                "pred_decoder_loss": pred_decoder_loss,
                "recon_decoder_loss": recon_decoder_loss,
            })
            
            # Add to total loss
            total_loss = total_loss + 0.1 * reconstruction_loss + 0.01 * (pred_decoder_loss + recon_decoder_loss)
        
        loss_components["total_loss"] = total_loss
        
        return predicted_features, predicted_images, reconstructed_images, total_loss, loss_components
    
    def rollout(
        self, 
        initial_obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        decode_images: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-step rollout for planning.
        
        Args:
            initial_obs: Initial observations with "visual" key (B, H, 3, H, W)
            actions: Action sequence (B, T, action_dim) for T future steps
            decode_images: Whether to decode predicted features to images
            
        Returns:
            rollout_results: Dict with predicted features and optionally images
        """
        B, T, action_dim = actions.shape
        device = actions.device
        
        # Encode initial history
        current_features = self.encode_history(initial_obs)  # (B, H, P, D)
        
        predicted_features_list = []
        predicted_images_list = []
        
        # Rollout loop
        for t in range(T):
            # Predict next frame
            next_features = self.dynamics(
                current_features, 
                actions[:, t:t+1].expand(-1, self.history_length, -1)  # Repeat action for all history frames
            )  # (B, P, D)
            
            predicted_features_list.append(next_features.unsqueeze(1))  # (B, 1, P, D)
            
            # Optional image decoding
            if decode_images:
                next_images, _ = self.decode_features(next_features)  # (B, 3, H, W)
                predicted_images_list.append(next_images.unsqueeze(1))  # (B, 1, 3, H, W)
            
            # Update sliding window of features
            current_features = torch.cat([
                current_features[:, 1:],  # Remove oldest frame
                next_features.unsqueeze(1)  # Add newest prediction
            ], dim=1)  # (B, H, P, D)
        
        # Concatenate results
        results = {
            "predicted_features": torch.cat(predicted_features_list, dim=1),  # (B, T, P, D)
        }
        
        if decode_images:
            results["predicted_images"] = torch.cat(predicted_images_list, dim=1)  # (B, T, 3, H, W)
        
        return results
    
    def clear_cache(self):
        """Clear encoder cache."""
        if hasattr(self.encoder, 'clear_cache'):
            self.encoder.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get encoder cache statistics."""
        if hasattr(self.encoder, 'get_cache_stats'):
            return self.encoder.get_cache_stats()
        return {"cache_enabled": False}


def create_vjepa2_world_model(
    encoder_config: Dict[str, Any],
    dynamics_config: Dict[str, Any], 
    decoder_config: Optional[Dict[str, Any]] = None,
    **model_kwargs
) -> VJEPA2WorldModel:
    """
    Factory function to create V-JEPA-2 world model.
    
    Args:
        encoder_config: V-JEPA-2 encoder configuration
        dynamics_config: ST-Transformer configuration
        decoder_config: Optional decoder configuration
        **model_kwargs: Additional model arguments
        
    Returns:
        VJEPA2WorldModel instance
    """
    return VJEPA2WorldModel(
        encoder_config=encoder_config,
        dynamics_config=dynamics_config,
        decoder_config=decoder_config,
        **model_kwargs
    )