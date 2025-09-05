import torch
import torch.nn as nn
import einops


class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size=224,
        num_hist=4,
        num_pred=4,
        encoder=None,
        proprio_encoder=None,
        action_encoder=None,
        decoder=None,
        predictor=None,
        proprio_dim=64,
        action_dim=64,
        concat_dim=0,
        num_action_repeat=1,
        num_proprio_repeat=1,
        train_encoder=True,
        train_predictor=True,
        train_decoder=True
    ):
        super().__init__()
        self.image_size = image_size
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.concat_dim = concat_dim
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat

        # Components
        assert encoder is not None, "Encoder must be provided"
        assert proprio_encoder is not None, "Proprio encoder must be provided"
        assert action_encoder is not None, "Action encoder must be provided"
        assert decoder is not None, "Decoder must be provided"
        assert predictor is not None, "Predictor must be provided"

        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder
        self.predictor = predictor

        # Training flags
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder

        # Freeze components if not training
        if not train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not train_predictor:
            for param in self.predictor.parameters():
                param.requires_grad = False
        if not train_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def encode_obs(self, obs):
        """Encode visual observations"""
        visual = obs['visual']  # (batch, T, 3, H, W)
        batch_size, T = visual.shape[0], visual.shape[1]

        # Encode each frame
        encoded_frames = []
        for t in range(T):
            frame = visual[:, t]  # (batch, 3, H, W)
            encoded = self.encoder(frame)  # (batch, num_patches, emb_dim)
            encoded_frames.append(encoded)

        return torch.stack(encoded_frames, dim=1)  # (batch, T, num_patches, emb_dim)

    def encode_proprio(self, proprio):
        """Encode proprioceptive states"""
        return self.proprio_encoder(proprio)  # (batch, T, emb_dim)

    def encode_actions(self, actions):
        """Encode actions"""
        return self.action_encoder(actions)  # (batch, T, emb_dim)

    def prepare_predictor_input(self, obs_emb, proprio_emb, action_emb):
        """Prepare input for predictor by concatenating modalities"""
        # obs_emb: (batch, num_hist, num_patches, emb_dim)
        # proprio_emb: (batch, num_hist, emb_dim)
        # action_emb: (batch, num_hist, emb_dim)

        # For now, just use visual embeddings (simplified)
        # Flatten for predictor
        combined = einops.rearrange(obs_emb, 'b t p d -> b (t p) d')

        return combined

    def forward(self, obs, actions):
        """
        Forward pass
        obs: dict with 'visual' (batch, T, 3, H, W) and 'proprio' (batch, T, proprio_dim)
        actions: (batch, T, action_dim)
        """
        # Encode inputs
        obs_emb = self.encode_obs(obs)  # (batch, num_hist, num_patches, emb_dim)
        proprio_emb = self.encode_proprio(obs['proprio'])  # (batch, num_hist, emb_dim)
        action_emb = self.encode_actions(actions)  # (batch, num_hist, emb_dim)

        # Prepare predictor input
        pred_input = self.prepare_predictor_input(obs_emb, proprio_emb, action_emb)

        # Predict future embeddings
        pred_emb = self.predictor(pred_input)  # (batch, num_hist * num_patches, emb_dim)

        # Reshape to (batch, num_hist, num_patches, emb_dim)
        pred_emb = einops.rearrange(pred_emb, 'b (t p) d -> b t p d',
                                   t=self.num_hist, p=256)  # Hardcoded for now

        # Decode to images
        pred_images = []
        for t in range(self.num_hist):
            frame_emb = pred_emb[:, t]  # (batch, num_patches, emb_dim)
            decoded = self.decoder(frame_emb)  # (batch, 3, H, W)
            pred_images.append(decoded)

        pred_images = torch.stack(pred_images, dim=1)  # (batch, num_hist, 3, H, W)

        # For now, return dummy loss (implement proper loss calculation)
        loss = torch.tensor(0.0, device=obs['visual'].device)

        return pred_emb, pred_images, obs['visual'], loss