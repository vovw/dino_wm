import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange


class PatchDecoder(nn.Module):
    """
    Decoder that reconstructs images from patch tokens.
    Converts (B, P, D) patch features back to (B, 3, H, W) images.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        patch_size: int = 16,
        out_channels: int = 3,
        image_size: int = 224,
        hidden_dim: int = 512,
        num_layers: int = 3,
        activation: str = "gelu",
        normalize_output: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.normalize_output = normalize_output
        
        # Compute number of patches
        self.patches_per_side = image_size // patch_size
        self.num_patches = self.patches_per_side ** 2
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Decoder MLP layers
        layers = []
        current_dim = feature_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self.activation,
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim
        
        # Final layer to patch pixels
        patch_pixels = patch_size ** 2 * out_channels
        layers.append(nn.Linear(current_dim, patch_pixels))
        
        self.decoder_mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize decoder weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Decode patch features to images.
        
        Args:
            patch_features: (B, P, D) or (B, T, P, D) patch features
            
        Returns:
            images: (B, 3, H, W) or (B, T, 3, H, W) reconstructed images
            diff: Dummy VQ loss (0.0 for compatibility)
        """
        input_shape = patch_features.shape
        is_temporal = len(input_shape) == 4  # (B, T, P, D)
        
        if is_temporal:
            B, T, P, D = input_shape
            # Flatten temporal dimension for processing
            patch_features = rearrange(patch_features, "b t p d -> (b t) p d")
        else:
            B, P, D = input_shape
            T = 1
        
        # Check patch count
        assert P == self.num_patches, f"Expected {self.num_patches} patches, got {P}"
        
        # Decode patches to pixel values
        # (B*T, P, D) -> (B*T, P, patch_size^2 * 3)
        patch_pixels = self.decoder_mlp(patch_features)
        
        # Reshape to patch grid
        # (B*T, P, patch_size^2 * 3) -> (B*T, patches_per_side, patches_per_side, patch_size^2 * 3)
        patch_grid = rearrange(
            patch_pixels, 
            "(bt) (h w) (p1 p2 c) -> bt h w (p1 p2 c)",
            bt=B*T,
            h=self.patches_per_side,
            w=self.patches_per_side,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels
        )
        
        # Convert patch grid to image
        # (B*T, h, w, patch_size^2 * 3) -> (B*T, 3, H, W)
        images = rearrange(
            patch_grid,
            "bt h w (p1 p2 c) -> bt c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels
        )
        
        # Normalize output if requested
        if self.normalize_output:
            images = torch.tanh(images)  # Output in [-1, 1]
        
        # Restore temporal dimension if needed
        if is_temporal:
            images = rearrange(images, "(b t) c h w -> b t c h w", b=B, t=T)
        
        return images, 0.0  # Return dummy VQ loss for compatibility


class ConvDecoder(nn.Module):
    """
    Convolutional decoder for patch features.
    Uses transposed convolutions for upsampling.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        patch_size: int = 16,
        out_channels: int = 3,
        image_size: int = 224,
        hidden_dims: Tuple[int, ...] = (512, 256, 128, 64),
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        normalize_output: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.image_size = image_size
        self.normalize_output = normalize_output
        
        # Compute patch grid size
        self.patches_per_side = image_size // patch_size
        self.num_patches = self.patches_per_side ** 2
        
        # Initial feature map size after reshaping patches
        self.init_h = self.init_w = self.patches_per_side
        
        # Project to initial hidden dimension
        self.input_proj = nn.Linear(feature_dim, hidden_dims[0])
        
        # Transposed convolutional layers
        layers = []
        in_dim = hidden_dims[0]
        
        for i, out_dim in enumerate(hidden_dims[1:]):
            layers.extend([
                nn.ConvTranspose2d(
                    in_dim, out_dim, 
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=stride-1
                ),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = out_dim
        
        # Final layer to RGB
        layers.append(nn.ConvTranspose2d(
            in_dim, out_channels,
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding,
            output_padding=stride-1
        ))
        
        if normalize_output:
            layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize decoder weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Decode patch features to images using convolutions.
        
        Args:
            patch_features: (B, P, D) or (B, T, P, D) patch features
            
        Returns:
            images: (B, 3, H, W) or (B, T, 3, H, W) reconstructed images  
            diff: Dummy VQ loss (0.0 for compatibility)
        """
        input_shape = patch_features.shape
        is_temporal = len(input_shape) == 4  # (B, T, P, D)
        
        if is_temporal:
            B, T, P, D = input_shape
            # Flatten temporal dimension
            patch_features = rearrange(patch_features, "b t p d -> (b t) p d")
        else:
            B, P, D = input_shape
            T = 1
        
        # Check patch count
        assert P == self.num_patches, f"Expected {self.num_patches} patches, got {P}"
        
        # Project features
        features = self.input_proj(patch_features)  # (B*T, P, hidden_dim)
        
        # Reshape to feature maps
        # (B*T, P, hidden_dim) -> (B*T, hidden_dim, H, W)
        features = rearrange(
            features,
            "bt (h w) d -> bt d h w",
            h=self.init_h,
            w=self.init_w
        )
        
        # Decode through convolutional layers
        images = self.decoder(features)  # (B*T, 3, H, W)
        
        # Restore temporal dimension if needed
        if is_temporal:
            images = rearrange(images, "(b t) c h w -> b t c h w", b=B, t=T)
        
        return images, 0.0


def create_vjepa2_decoder(
    decoder_type: str = "patch",
    feature_dim: int = 256,
    patch_size: int = 16,
    image_size: int = 224,
    **kwargs
) -> nn.Module:
    """
    Factory function to create V-JEPA-2 compatible decoder.
    
    Args:
        decoder_type: "patch" or "conv"
        feature_dim: Feature dimension from encoder
        patch_size: Patch size used by encoder
        image_size: Output image size
        **kwargs: Additional decoder arguments
        
    Returns:
        Decoder module
    """
    if decoder_type == "patch":
        return PatchDecoder(
            feature_dim=feature_dim,
            patch_size=patch_size,
            image_size=image_size,
            **kwargs
        )
    elif decoder_type == "conv":
        return ConvDecoder(
            feature_dim=feature_dim,
            patch_size=patch_size,
            image_size=image_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")