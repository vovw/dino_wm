import pytest
import torch
import numpy as np
from models.dynamics.st_transformer import (
    STTransformer, STTransformerBlock, ActionFiLM, ActionCrossAttention, 
    ActionConcat, block_causal_mask
)


class TestBlockCausalMask:
    """Test block-causal attention mask."""
    
    def test_mask_shape(self):
        """Test mask has correct shape."""
        T, P = 3, 4
        device = torch.device('cpu')
        mask = block_causal_mask(T, P, device)
        
        expected_size = T * P
        assert mask.shape == (expected_size, expected_size)
        assert mask.dtype == torch.bool
        
    def test_mask_causality(self):
        """Test mask enforces causality between frames."""
        T, P = 3, 2
        device = torch.device('cpu')
        mask = block_causal_mask(T, P, device)
        
        # Within frame 0 (tokens 0-1): no masking
        assert not mask[0, 0]  # token 0 can attend to token 0
        assert not mask[0, 1]  # token 0 can attend to token 1
        assert not mask[1, 0]  # token 1 can attend to token 0
        assert not mask[1, 1]  # token 1 can attend to token 1
        
        # Frame 0 to frame 1: should be masked (future)
        assert mask[0, 2]  # token 0 cannot attend to token 2 (frame 1)
        assert mask[1, 3]  # token 1 cannot attend to token 3 (frame 1)
        
        # Frame 1 to frame 0: should not be masked (past)
        assert not mask[2, 0]  # token 2 can attend to token 0 (frame 0)
        assert not mask[3, 1]  # token 3 can attend to token 1 (frame 0)
        
    def test_mask_within_frame_full_attention(self):
        """Test full attention is allowed within frames."""
        T, P = 2, 3
        device = torch.device('cpu')
        mask = block_causal_mask(T, P, device)
        
        # All tokens within frame 0 should attend to each other
        for i in range(P):
            for j in range(P):
                assert not mask[i, j], f"Token {i} should attend to token {j} within frame"
        
        # All tokens within frame 1 should attend to each other  
        for i in range(P, 2*P):
            for j in range(P, 2*P):
                assert not mask[i, j], f"Token {i} should attend to token {j} within frame"


class TestActionConditioning:
    """Test different action conditioning methods."""
    
    def test_action_film(self):
        """Test FiLM action conditioning."""
        action_dim, feature_dim = 7, 256
        B, P = 2, 100
        
        film = ActionFiLM(action_dim, feature_dim)
        tokens = torch.randn(B, P, feature_dim)
        actions = torch.randn(B, action_dim)
        
        output = film(tokens, actions)
        assert output.shape == tokens.shape
        
        # Output should be different from input (unless actions are zeros)
        assert not torch.allclose(output, tokens, atol=1e-6)
    
    def test_action_cross_attention(self):
        """Test cross-attention action conditioning."""
        action_dim, feature_dim = 7, 256
        num_heads = 8
        B, P = 2, 100
        
        cross_attn = ActionCrossAttention(feature_dim, action_dim, num_heads)
        tokens = torch.randn(B, P, feature_dim)
        actions = torch.randn(B, action_dim)
        
        output = cross_attn(tokens, actions)
        assert output.shape == tokens.shape
    
    def test_action_concat(self):
        """Test concatenation-based action conditioning."""
        action_dim, feature_dim = 7, 256
        B, P = 2, 100
        
        concat_cond = ActionConcat(action_dim, feature_dim)
        tokens = torch.randn(B, P, feature_dim)
        actions = torch.randn(B, action_dim)
        
        output = concat_cond(tokens, actions)
        assert output.shape == tokens.shape


class TestSTTransformerBlock:
    """Test spatio-temporal transformer block."""
    
    @pytest.fixture
    def block_config(self):
        return {
            "feature_dim": 256,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "action_dim": 7,
            "action_conditioning": "film"
        }
    
    def test_block_forward_no_actions(self, block_config):
        """Test block forward pass without actions."""
        config = block_config.copy()
        config["action_dim"] = 0
        
        block = STTransformerBlock(**config)
        
        B, T, P = 2, 3, 100
        L = T * P
        tokens = torch.randn(B, L, config["feature_dim"])
        
        output = block(tokens)
        assert output.shape == tokens.shape
    
    def test_block_forward_with_actions(self, block_config):
        """Test block forward pass with actions."""
        block = STTransformerBlock(**block_config)
        
        B, T, P = 2, 3, 100
        L = T * P
        tokens = torch.randn(B, L, block_config["feature_dim"])
        actions = torch.randn(B, T, block_config["action_dim"])
        
        attn_mask = block_causal_mask(T, P, tokens.device)
        output = block(tokens, attn_mask=attn_mask, actions_per_frame=actions, T=T, P=P)
        assert output.shape == tokens.shape
    
    def test_different_action_conditioning_methods(self, block_config):
        """Test different action conditioning methods work."""
        methods = ["film", "cross_attn", "concat", "none"]
        
        for method in methods:
            config = block_config.copy()
            config["action_conditioning"] = method
            if method == "none":
                config["action_dim"] = 0
            
            block = STTransformerBlock(**config)
            
            B, T, P = 1, 2, 10
            L = T * P
            tokens = torch.randn(B, L, config["feature_dim"])
            actions = torch.randn(B, T, config["action_dim"]) if method != "none" else None
            
            attn_mask = block_causal_mask(T, P, tokens.device)
            output = block(tokens, attn_mask=attn_mask, actions_per_frame=actions, T=T, P=P)
            assert output.shape == tokens.shape


class TestSTTransformer:
    """Test full spatio-temporal transformer."""
    
    @pytest.fixture
    def transformer_config(self):
        return {
            "feature_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "action_dim": 7,
            "action_conditioning": "film",
            "max_seq_length": 1000
        }
    
    def test_transformer_forward_3d_input(self, transformer_config):
        """Test transformer with 3D input."""
        transformer = STTransformer(**transformer_config)
        
        B, T, P = 2, 4, 64
        L = T * P
        tokens = torch.randn(B, L, transformer_config["feature_dim"])
        actions = torch.randn(B, T, transformer_config["action_dim"])
        
        output = transformer(tokens, actions, T=T, P=P)
        assert output.shape == (B, P, transformer_config["feature_dim"])
    
    def test_transformer_forward_4d_input(self, transformer_config):
        """Test transformer with 4D input."""
        transformer = STTransformer(**transformer_config)
        
        B, T, P = 2, 4, 64
        tokens = torch.randn(B, T, P, transformer_config["feature_dim"])
        actions = torch.randn(B, T, transformer_config["action_dim"])
        
        output = transformer(tokens, actions)
        assert output.shape == (B, P, transformer_config["feature_dim"])
    
    def test_transformer_without_actions(self, transformer_config):
        """Test transformer without actions."""
        config = transformer_config.copy()
        config["action_dim"] = 0
        config["action_conditioning"] = "none"
        
        transformer = STTransformer(**config)
        
        B, T, P = 2, 4, 64
        tokens = torch.randn(B, T, P, config["feature_dim"])
        
        output = transformer(tokens)
        assert output.shape == (B, P, config["feature_dim"])
    
    def test_transformer_rollout(self, transformer_config):
        """Test multi-step rollout."""
        transformer = STTransformer(**transformer_config)
        
        B, H, P = 2, 3, 64  # 3 frames of history
        rollout_steps = 5
        
        initial_tokens = torch.randn(B, H, P, transformer_config["feature_dim"])
        actions = torch.randn(B, rollout_steps, transformer_config["action_dim"])
        
        predicted_tokens = transformer.rollout(initial_tokens, actions, history_length=H)
        assert predicted_tokens.shape == (B, rollout_steps, P, transformer_config["feature_dim"])
    
    def test_sequence_length_limit(self, transformer_config):
        """Test sequence length limit is enforced.""" 
        config = transformer_config.copy()
        config["max_seq_length"] = 100
        
        transformer = STTransformer(**config)
        
        B, T, P = 2, 20, 10  # T*P = 200 > max_seq_length
        tokens = torch.randn(B, T, P, config["feature_dim"])
        
        with pytest.raises(AssertionError):
            transformer(tokens)
    
    def test_deterministic_output(self, transformer_config):
        """Test deterministic output with same seed."""
        transformer = STTransformer(**transformer_config)
        transformer.eval()
        
        B, T, P = 1, 3, 16
        torch.manual_seed(42)
        tokens1 = torch.randn(B, T, P, transformer_config["feature_dim"])
        actions1 = torch.randn(B, T, transformer_config["action_dim"])
        
        torch.manual_seed(42)
        tokens2 = torch.randn(B, T, P, transformer_config["feature_dim"])
        actions2 = torch.randn(B, T, transformer_config["action_dim"])
        
        with torch.no_grad():
            output1 = transformer(tokens1, actions1)
            output2 = transformer(tokens2, actions2)
        
        torch.testing.assert_close(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])