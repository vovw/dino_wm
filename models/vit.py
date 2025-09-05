import torch
import torch.nn as nn
import einops


class ViT(nn.Module):
    def __init__(self, num_patches, num_frames, dim, depth=6, heads=8, mlp_dim=1024):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim

        # Positional embeddings for patches and frames
        self.patch_pos_emb = nn.Parameter(torch.randn(1, num_patches, dim))
        self.frame_pos_emb = nn.Parameter(torch.randn(1, num_frames, 1, dim))

        # Transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=mlp_dim,
                    batch_first=True
                )
            )

    def forward(self, x):
        # x: (batch, num_frames * num_patches, dim)
        # For now, just apply transformer directly
        for layer in self.layers:
            x = layer(x)

        return x