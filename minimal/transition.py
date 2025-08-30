import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class FrameCausalTransformer(nn.Module):
    """
    Frame-causal Transformer world model for latent space prediction.
    Predicts next frame latents given current latents and actions.
    """

    def __init__(self, embed_dim: int, action_dim: int, num_patches: int,
                 num_layers: int = 6, num_heads: int = 8, ff_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.num_patches = num_patches
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Action embedding
        self.action_proj = nn.Linear(action_dim, embed_dim)

        # Position embeddings for patches within each frame
        self.patch_pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Frame position embeddings (for temporal sequence)
        self.frame_pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim))  # max 100 frames

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Parameter):
            if module.data.dim() > 1:
                torch.nn.init.xavier_uniform_(module.data)

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, N, D) latent representations
            actions: (B, T-1, A) actions

        Returns:
            preds: (B, T-1, N, D) predicted next latents
        """
        B, T, N, D = z.shape
        assert N == self.num_patches
        assert D == self.embed_dim

        # Flatten spatial and temporal dimensions for transformer
        z_flat = z.view(B, T * N, D)  # (B, T*N, D)

        # Add patch position embeddings
        patch_pos = self.patch_pos_embed.repeat(B, T, 1)  # (B, T*N, D)
        z_flat = z_flat + patch_pos

        # Add frame position embeddings
        frame_pos = self.frame_pos_embed[:, :T, :].repeat_interleave(N, dim=1)  # (B, T*N, D)
        z_flat = z_flat + frame_pos

        # Project actions
        action_emb = self.action_proj(actions)  # (B, T-1, D)

        # Create causal mask for frame-level attention
        # Each frame attends to previous frames but not future frames
        causal_mask = self._create_causal_mask(T, N, z_flat.device)

        # Apply transformer layers
        for layer in self.layers:
            # For decoder layer, we use the sequence as both input and target
            # but mask future tokens
            z_flat = layer(z_flat, z_flat, tgt_mask=causal_mask)

        # Reshape back to (B, T, N, D)
        z_out = z_flat.view(B, T, N, D)

        # Predict next frames by shifting
        preds = z_out[:, :-1]  # (B, T-1, N, D)

        # Apply output projection
        preds = self.out_proj(preds)

        return preds

    def _create_causal_mask(self, T: int, N: int, device) -> torch.Tensor:
        """Create causal mask for frame-level attention."""
        # Create mask where each position can attend to previous frames
        seq_len = T * N
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(device)
        return mask

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predictions and targets.
        """
        return F.mse_loss(preds, targets)


class TransitionModel(nn.Module):
    """
    Complete transition model combining encoder and transformer.
    """

    def __init__(self, encoder, action_dim: int, num_layers: int = 6):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.get_embed_dim()
        self.num_patches = encoder.get_num_patches()
        self.action_dim = action_dim

        self.transformer = FrameCausalTransformer(
            embed_dim=self.embed_dim,
            action_dim=action_dim,
            num_patches=self.num_patches,
            num_layers=num_layers
        )

    def forward(self, images: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, T, 3, H, W) or (B, T, H, W, 3)
            actions: (B, T-1, A)

        Returns:
            preds: (B, T-1, N, D) predicted latents
            targets: (B, T-1, N, D) target latents
        """
        B, T = images.shape[:2]

        # Encode all frames
        latents_list = []
        for t in range(T):
            frame_latents, _ = self.encoder(images[:, t])  # (B, N, D)
            latents_list.append(frame_latents)

        z = torch.stack(latents_list, dim=1)  # (B, T, N, D)

        # Get predictions from transformer
        preds = self.transformer(z, actions)  # (B, T-1, N, D)

        # Get targets (next frames)
        targets = z[:, 1:]  # (B, T-1, N, D)

        return preds, targets

    def compute_loss(self, images: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch."""
        preds, targets = self.forward(images, actions)
        return self.transformer.compute_loss(preds, targets)


def test_shapes():
    """Test tensor shapes."""
    from min_dinov2 import DinoV2Encoder

    # Create dummy components
    encoder = DinoV2Encoder()
    model = TransitionModel(encoder, action_dim=2)

    # Test input
    B, T = 2, 6
    images = torch.randn(B, T, 3, 64, 64)  # BHWC format would be (B, T, 64, 64, 3)
    actions = torch.randn(B, T-1, 2)

    preds, targets = model(images, actions)

    print(f"Images: {images.shape}")
    print(f"Actions: {actions.shape}")
    print(f"Preds: {preds.shape}")
    print(f"Targets: {targets.shape}")

    assert preds.shape == targets.shape == (B, T-1, model.num_patches, model.embed_dim)
    print("Shape test passed!")


if __name__ == "__main__":
    test_shapes()
