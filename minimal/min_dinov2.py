import torch
import torch.nn as nn
from typing import Tuple, Union
import torchvision.transforms as T


class DinoV2Encoder(nn.Module):
    """
    Frozen DINOv2 encoder wrapper.
    Accepts BHWC or BCHW input, returns patch tokens and grid dimensions.
    """

    def __init__(self, model_name: str = "vit_base_patch14_dinov2", image_size: int = 224):
        super().__init__()
        self.image_size = image_size

        # Load DINOv2 model from timm
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Get model dimensions
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.num_patches = (image_size // self.patch_size) ** 2

        # ImageNet normalization
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Resize transform
        self.resize = T.Resize((image_size, image_size), antialias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: Input images, shape (B, H, W, 3) or (B, 3, H, W), uint8 or float

        Returns:
            tokens: (B, N, D) patch tokens (excluding CLS token)
            grid: (H_p, W_p) patch grid dimensions
        """
        # Handle different input formats
        if x.dim() == 4:
            if x.shape[-1] == 3:  # BHWC format
                x = x.permute(0, 3, 1, 2)  # BCHW
            # else assume already BCHW

        # Convert to float if needed and scale to [0,1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Ensure 3 channels
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]}")

        # Resize to target size
        x = self.resize(x)

        # Normalize
        x = self.normalize(x)

        # Forward pass
        with torch.no_grad():
            features = self.model.forward_features(x)  # (B, N+1, D) includes CLS token

        # Remove CLS token and return patch tokens
        tokens = features[:, 1:, :]  # (B, N, D)

        # Compute grid dimensions
        grid_size = int(self.num_patches ** 0.5)
        grid = (grid_size, grid_size)

        return tokens, grid

    def get_embed_dim(self) -> int:
        """Get embedding dimension."""
        return self.embed_dim

    def get_num_patches(self) -> int:
        """Get number of patches per image."""
        return self.num_patches

    def get_patch_grid(self) -> Tuple[int, int]:
        """Get patch grid dimensions (H_p, W_p)."""
        grid_size = int(self.num_patches ** 0.5)
        return grid_size, grid_size


def test_encoder():
    """Simple test function."""
    device = torch.device('cpu')
    encoder = DinoV2Encoder()

    # Test with BHWC input
    x_bhwc = torch.randn(2, 64, 64, 3)
    tokens, grid = encoder(x_bhwc)
    print(f"BHWC input: {x_bhwc.shape} -> tokens: {tokens.shape}, grid: {grid}")

    # Test with BCHW input
    x_bchw = torch.randn(2, 3, 64, 64)
    tokens, grid = encoder(x_bchw)
    print(f"BCHW input: {x_bchw.shape} -> tokens: {tokens.shape}, grid: {grid}")

    assert tokens.shape == (2, encoder.num_patches, encoder.embed_dim)
    print("Encoder test passed!")


if __name__ == "__main__":
    test_encoder()
