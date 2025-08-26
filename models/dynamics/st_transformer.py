import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from einops import rearrange, repeat


def block_causal_mask(T: int, P: int, device: torch.device) -> torch.Tensor:
    """
    Create block-causal attention mask for spatio-temporal transformer.
    
    Args:
        T: Number of frames (time steps)
        P: Number of patches per frame
        device: Device to create mask on
        
    Returns:
        mask: (T*P, T*P) boolean mask where True means "masked out"
    """
    L = T * P
    mask = torch.zeros(L, L, dtype=torch.bool, device=device)
    
    for ti in range(T):
        for tj in range(T):
            if tj > ti:  # Mask future frames
                mask[ti*P:(ti+1)*P, tj*P:(tj+1)*P] = True
    
    return mask


class ActionFiLM(nn.Module):
    """FiLM conditioning for action integration per frame."""
    
    def __init__(self, action_dim: int, feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, 2 * feature_dim),
            nn.SiLU(),
            nn.Linear(2 * feature_dim, 2 * feature_dim)
        )
        
    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning to tokens.
        
        Args:
            tokens: (B, P, D) tokens for one frame
            actions: (B, action_dim) action for this frame
            
        Returns:
            conditioned_tokens: (B, P, D) FiLM-conditioned tokens
        """
        gamma_beta = self.mlp(actions)  # (B, 2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Each (B, D)
        
        # Apply FiLM: tokens * (1 + gamma) + beta
        return tokens * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class ActionCrossAttention(nn.Module):
    """Cross-attention for action conditioning."""
    
    def __init__(self, feature_dim: int, action_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(action_dim, feature_dim)
        self.v_proj = nn.Linear(action_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between tokens and actions.
        
        Args:
            tokens: (B, P, D) tokens for one frame
            actions: (B, action_dim) action for this frame
            
        Returns:
            attended_tokens: (B, P, D) cross-attention output
        """
        B, P, D = tokens.shape
        
        # Queries from tokens, keys/values from actions
        Q = self.q_proj(tokens)  # (B, P, D)
        K = self.k_proj(actions.unsqueeze(1))  # (B, 1, D) 
        V = self.v_proj(actions.unsqueeze(1))  # (B, 1, D)
        
        # Reshape for multi-head attention
        Q = Q.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, P, head_dim)
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, head_dim)
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, P, 1)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, V)  # (B, H, P, head_dim)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(B, P, D)  # (B, P, D)
        
        return self.out_proj(attended)


class ActionConcat(nn.Module):
    """Simple concatenation-based action conditioning."""
    
    def __init__(self, action_dim: int, feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply concatenation-based action conditioning.
        
        Args:
            tokens: (B, P, D) tokens for one frame
            actions: (B, action_dim) action for this frame
            
        Returns:
            conditioned_tokens: (B, P, D) action-conditioned tokens
        """
        action_emb = self.mlp(actions)  # (B, D)
        action_emb = action_emb.unsqueeze(1).expand(-1, tokens.size(1), -1)  # (B, P, D)
        return tokens + action_emb


class STTransformerBlock(nn.Module):
    """Spatio-temporal transformer block with action conditioning."""
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_dim: int = 0,
        action_conditioning: str = "film"  # "film", "cross_attn", "concat", "none"
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.action_conditioning = action_conditioning
        
        # Self-attention
        self.norm1 = nn.LayerNorm(feature_dim)
        self.self_attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(feature_dim)
        mlp_dim = int(feature_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, feature_dim),
            nn.Dropout(dropout)
        )
        
        # Action conditioning
        if action_dim > 0:
            if action_conditioning == "film":
                self.action_conditioner = ActionFiLM(action_dim, feature_dim)
            elif action_conditioning == "cross_attn":
                self.action_conditioner = ActionCrossAttention(feature_dim, action_dim, num_heads)
            elif action_conditioning == "concat":
                self.action_conditioner = ActionConcat(action_dim, feature_dim)
            else:
                self.action_conditioner = None
        else:
            self.action_conditioner = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        actions_per_frame: Optional[torch.Tensor] = None,
        T: int = None,
        P: int = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: (B, T*P, D) input tokens
            attn_mask: (T*P, T*P) attention mask
            actions_per_frame: (B, T, action_dim) actions per frame
            T: Number of frames
            P: Number of patches per frame
            
        Returns:
            output: (B, T*P, D) output tokens
        """
        # Self-attention with causal mask
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + x
        
        # Action conditioning (applied per frame)
        if self.action_conditioner is not None and actions_per_frame is not None:
            B, L, D = x.shape
            assert L == T * P, f"Expected {T * P} tokens, got {L}"
            
            # Reshape to (B, T, P, D)
            x_frames = x.view(B, T, P, D)
            
            # Apply action conditioning per frame
            for t in range(T):
                x_frames[:, t] = self.action_conditioner(x_frames[:, t], actions_per_frame[:, t])
            
            # Reshape back to (B, T*P, D)
            x = x_frames.view(B, L, D)
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class STTransformer(nn.Module):
    """
    Spatio-temporal transformer for video understanding with action conditioning.
    
    Processes sequences of frame tokens with block-causal attention, where:
    - Full spatial attention is allowed within each frame
    - Causal attention is enforced across frames (no future frame information)
    - Actions are conditioned per frame using configurable methods
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_dim: int = 0,
        action_conditioning: str = "film",  # "film", "cross_attn", "concat", "none"
        max_seq_length: int = 1000,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.action_conditioning = action_conditioning
        self.max_seq_length = max_seq_length
        
        # Positional encoding 
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, feature_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            STTransformerBlock(
                feature_dim=feature_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                action_dim=action_dim,
                action_conditioning=action_conditioning
            ) for _ in range(num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        # Prediction head for next frame
        self.pred_head = nn.Linear(feature_dim, feature_dim)
        
    def forward(
        self,
        tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        T: int = None,
        P: int = None
    ) -> torch.Tensor:
        """
        Forward pass through spatio-temporal transformer.
        
        Args:
            tokens: (B, T*P, D) or (B, T, P, D) input tokens
            actions: (B, T, action_dim) actions per frame (optional)
            T: Number of frames (inferred if tokens is 4D)
            P: Number of patches per frame (inferred if tokens is 4D)
            
        Returns:
            next_frame_tokens: (B, P, D) predicted tokens for next frame
        """
        # Handle input format
        if tokens.ndim == 4:  # (B, T, P, D)
            B, T, P, D = tokens.shape
            tokens = tokens.view(B, T * P, D)
        else:  # (B, T*P, D) 
            B, L, D = tokens.shape
            assert T is not None and P is not None, "T and P must be provided for 3D input"
            assert L == T * P, f"Expected {T * P} tokens, got {L}"
        
        seq_len = T * P
        assert seq_len <= self.max_seq_length, f"Sequence length {seq_len} exceeds maximum {self.max_seq_length}"
        
        # Add positional encoding
        tokens = tokens + self.pos_embedding[:, :seq_len, :]
        
        # Create block-causal attention mask
        attn_mask = block_causal_mask(T, P, tokens.device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            tokens = block(
                tokens, 
                attn_mask=attn_mask, 
                actions_per_frame=actions,
                T=T, 
                P=P
            )
        
        # Final normalization
        tokens = self.norm(tokens)
        
        # Predict next frame - take features from the last frame and predict next
        last_frame_tokens = tokens[:, -P:, :]  # (B, P, D) - tokens from last frame
        next_frame_tokens = self.pred_head(last_frame_tokens)  # (B, P, D)
        
        return next_frame_tokens
    
    def rollout(
        self,
        initial_tokens: torch.Tensor,
        actions: torch.Tensor,
        history_length: int
    ) -> torch.Tensor:
        """
        Perform multi-step rollout prediction.
        
        Args:
            initial_tokens: (B, H, P, D) initial frame tokens (H frames of history)
            actions: (B, T, action_dim) actions for T future steps
            history_length: Number of history frames to maintain
            
        Returns:
            predicted_tokens: (B, T, P, D) predicted tokens for T future frames
        """
        B, H, P, D = initial_tokens.shape
        T = actions.shape[1]
        
        # Initialize with history
        current_tokens = initial_tokens  # (B, H, P, D)
        predictions = []
        
        for t in range(T):
            # Get current action
            current_action = actions[:, t:t+1]  # (B, 1, action_dim)
            
            # Predict next frame
            next_tokens = self.forward(
                current_tokens, 
                current_action,
                T=current_tokens.shape[1], 
                P=P
            )  # (B, P, D)
            
            predictions.append(next_tokens.unsqueeze(1))  # (B, 1, P, D)
            
            # Update history: remove oldest, add newest
            current_tokens = torch.cat([
                current_tokens[:, 1:],  # Remove oldest frame
                next_tokens.unsqueeze(1)  # Add predicted frame
            ], dim=1)  # Still (B, H, P, D)
        
        return torch.cat(predictions, dim=1)  # (B, T, P, D)