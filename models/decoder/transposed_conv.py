import torch
import torch.nn as nn
import einops


class TransposedConvDecoder(nn.Module):
    def __init__(self, emb_dim, num_patches=196, patch_size=16, image_size=224):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size

        # Calculate grid size (assuming square patches)
        self.grid_size = int(num_patches ** 0.5)  # 256 -> 16

        # Linear layer to project to higher dim for conv
        self.proj = nn.Linear(emb_dim, 512)

        # Transposed convolution layers
        # For 16x16 input to 224x224 output
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64->128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 128->256
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # 256->256, reduce to 3 channels
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),  # Downsample to 224
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        # x: (batch, num_patches, emb_dim)
        batch_size = x.shape[0]

        # Project to higher dimension
        x = self.proj(x)  # (batch, num_patches, 512)

        # Reshape to spatial grid
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.grid_size, w=self.grid_size)

        # Decode
        x = self.decoder(x)  # (batch, 3, image_size, image_size)

        return x