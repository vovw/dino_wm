import pytest
import torch
import numpy as np
from models.encoders.vjepa2_encoder import VJEPA2Encoder, ProjectionHead, WhiteningNormalizer


class TestVJEPA2Encoder:
    """Test suite for VJEPA2Encoder."""
    
    @pytest.fixture
    def encoder_config(self):
        """Basic encoder configuration for testing."""
        return {
            "variant": "vit-L",
            "resolution": 256,
            "tubelet": {"t": 2, "p": 16},
            "adapter": {
                "projection_dim": 256,
                "normalize": "l2",
                "frameize": True
            },
            "enable_cache": True,
            "cache_size": 10
        }
    
    @pytest.fixture  
    def encoder(self, encoder_config):
        """Create encoder instance for testing."""
        try:
            return VJEPA2Encoder(**encoder_config)
        except ImportError:
            pytest.skip("V-JEPA-2 not available for testing")
    
    def test_encoder_initialization(self, encoder_config):
        """Test encoder initializes correctly."""
        try:
            encoder = VJEPA2Encoder(**encoder_config)
            assert encoder.variant == "vit-L"
            assert encoder.resolution == 256
            assert encoder.tubelet_t == 2
            assert encoder.tubelet_p == 16
            assert encoder.projection_dim == 256
            assert encoder.emb_dim == 256
        except ImportError:
            pytest.skip("V-JEPA-2 not available")
    
    def test_single_image_encoding(self, encoder):
        """Test encoding of single images."""
        B, C, H, W = 2, 3, 256, 256
        image = torch.randn(B, C, H, W)
        
        # Test via forward method
        features = encoder(image)
        expected_P = (H // encoder.tubelet_p) ** 2
        assert features.shape == (B, encoder.tubelet_t, expected_P, encoder.projection_dim)
        
        # Test via encode_image method
        features_img = encoder.encode_image(image)
        assert features_img.shape == (B, expected_P, encoder.projection_dim)
    
    def test_clip_encoding(self, encoder):
        """Test encoding of video clips."""
        B, T, C, H, W = 2, 4, 3, 256, 256
        clip = torch.randn(B, T, C, H, W)
        
        features = encoder.encode_clip(clip)
        expected_P = (H // encoder.tubelet_p) ** 2
        assert features.shape == (B, T, expected_P, encoder.projection_dim)
    
    def test_caching_functionality(self, encoder):
        """Test caching works correctly."""
        B, C, H, W = 1, 3, 256, 256
        image = torch.randn(B, C, H, W)
        
        # First forward pass
        cache_key = "test_key"
        features1 = encoder(image, cache_key=cache_key)
        
        # Second forward pass should use cache
        features2 = encoder(image, cache_key=cache_key)
        
        # Should be identical due to caching
        torch.testing.assert_close(features1, features2)
        
        # Check cache stats
        stats = encoder.get_cache_stats()
        assert stats["cache_size"] == 1
        assert stats["cache_enabled"] is True
    
    def test_deterministic_output(self, encoder):
        """Test that output is deterministic for same input."""
        torch.manual_seed(42)
        B, C, H, W = 1, 3, 256, 256
        image = torch.randn(B, C, H, W)
        
        encoder.eval()
        with torch.no_grad():
            features1 = encoder(image)
            features2 = encoder(image)
        
        torch.testing.assert_close(features1, features2)
    
    def test_different_normalizations(self, encoder_config):
        """Test different normalization types."""
        norm_types = ["none", "l2", "layernorm", "whiten"]
        
        for norm_type in norm_types:
            try:
                config = encoder_config.copy()
                config["adapter"]["normalize"] = norm_type
                encoder = VJEPA2Encoder(**config)
                
                B, C, H, W = 1, 3, 256, 256  
                image = torch.randn(B, C, H, W)
                features = encoder(image)
                
                # Check output shape is correct
                expected_P = (H // encoder.tubelet_p) ** 2
                assert features.shape == (B, encoder.tubelet_t, expected_P, encoder.projection_dim)
                
                # Check for any NaNs or infs
                assert torch.isfinite(features).all()
                
                # For L2 normalization, check that features are normalized
                if norm_type == "l2":
                    norms = torch.norm(features, dim=-1)
                    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6)
                    
            except ImportError:
                pytest.skip("V-JEPA-2 not available")


class TestProjectionHead:
    """Test suite for ProjectionHead."""
    
    def test_projection_head_basic(self):
        """Test basic projection head functionality."""
        input_dim, output_dim = 1024, 256
        head = ProjectionHead(input_dim, output_dim)
        
        B, T, P = 2, 4, 100
        features = torch.randn(B, T, P, input_dim)
        output = head(features)
        
        assert output.shape == (B, T, P, output_dim)
    
    def test_projection_head_with_layernorm(self):
        """Test projection head with layer normalization."""
        input_dim, output_dim = 1024, 256
        head = ProjectionHead(input_dim, output_dim, use_layernorm=True)
        
        B, T, P = 2, 4, 100
        features = torch.randn(B, T, P, input_dim)
        output = head(features)
        
        assert output.shape == (B, T, P, output_dim)
        # Check that layernorm approximately normalizes
        mean = output.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)


class TestWhiteningNormalizer:
    """Test suite for WhiteningNormalizer."""
    
    def test_whitening_normalizer_basic(self):
        """Test basic whitening normalizer functionality."""
        dim = 256
        normalizer = WhiteningNormalizer(dim)
        
        B, T, P = 2, 4, 100
        features = torch.randn(B, T, P, dim)
        
        # Training mode - should update statistics
        normalizer.train()
        output_train = normalizer(features)
        assert output_train.shape == features.shape
        
        # Eval mode - should use running statistics
        normalizer.eval()  
        output_eval = normalizer(features)
        assert output_eval.shape == features.shape
    
    def test_whitening_running_stats_update(self):
        """Test that running statistics are updated correctly."""
        dim = 10  # Small dim for easier testing
        normalizer = WhiteningNormalizer(dim, momentum=0.1)
        
        # Initial running mean should be zero
        assert torch.allclose(normalizer.running_mean, torch.zeros(dim))
        
        # Create data with known statistics
        features = torch.ones(1, 1, 1, dim) * 5.0  # All values = 5
        
        normalizer.train()
        _ = normalizer(features)
        
        # Running mean should have moved towards 5
        assert normalizer.running_mean.abs().sum() > 0
        assert torch.all(normalizer.running_mean > 0)  # Should be positive


if __name__ == "__main__":
    # Simple test runner
    pytest.main([__file__, "-v"])