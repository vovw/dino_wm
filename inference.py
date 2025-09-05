#!/usr/bin/env python3
"""
Inference script for DINOv2 World Model
Generates reconstructions from trained checkpoints
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, Normalize

# Import components
from models.dino import DinoV2Encoder
from models.proprio import ProprioEncoder
from models.dummy import DummyEncoder as ActionEncoder
from models.vit import ViT
from models.decoder.transposed_conv import TransposedConvDecoder
from models.visual_world_model import VWorldModel
from datasets.pusht_dset import PushTDataset


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    # Model components
    encoder = DinoV2Encoder(name='dinov2_vits14', feature_key='x_norm_patchtokens')
    proprio_encoder = ProprioEncoder(in_chans=2, emb_dim=64)
    action_encoder = ActionEncoder(in_chans=2, emb_dim=64)
    predictor = ViT(num_patches=256, num_frames=4, dim=384)
    decoder = TransposedConvDecoder(emb_dim=384, num_patches=256)

    # World model
    model = VWorldModel(
        image_size=224,
        num_hist=4,
        num_pred=4,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        decoder=decoder,
        predictor=predictor,
        proprio_dim=2,
        action_dim=2,
        train_encoder=True,
        train_predictor=True,
        train_decoder=True
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def run_inference(model, dataloader, device, output_dir, num_samples=5):
    """Run inference and save reconstruction plots"""
    to_pil = ToPILImage()
    denorm = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                      std=[1/0.229, 1/0.224, 1/0.225])

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (obs, actions) in enumerate(dataloader):
            if i >= num_samples:
                break

            # Move to device
            obs = {k: v.to(device) for k, v in obs.items()}
            actions = actions.to(device)

            # Forward pass
            z_pred, obs_pred, obs_recon, _ = model(obs, actions)

            # Compute losses
            mse_loss = nn.MSELoss()
            recon_loss = mse_loss(obs_pred, obs_recon)
            pred_loss = mse_loss(z_pred, model.encode_obs(obs))
            total_loss = recon_loss + pred_loss

            print(f"Sample {i}: Recon Loss: {recon_loss.item():.4f}, "
                  f"Pred Loss: {pred_loss.item():.4f}, Total: {total_loss.item():.4f}")

            # Get original and reconstructed images
            original = obs['visual'][0]  # First in batch
            reconstructed = obs_recon[0]

            # Plot comparison
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            for t in range(4):
                # Original
                orig_img = to_pil(denorm(original[t]))
                axes[0, t].imshow(orig_img)
                axes[0, t].set_title(f'Original t={t}')
                axes[0, t].axis('off')

                # Reconstructed
                recon_img = to_pil(denorm(reconstructed[t]))
                axes[1, t].imshow(recon_img)
                axes[1, t].set_title(f'Reconstructed t={t}')
                axes[1, t].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'inference_sample_{i}.png'))
            plt.close()

            print(f"Saved inference sample {i}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on trained DINOv2 World Model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to process')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint_path, device)

    # Load val dataset
    dataset = PushTDataset(
        data_dir=args.data_dir,
        mode='val',
        seq_len=4,
        image_size=224
    )

    # Use specified number of samples
    from torch.utils.data import Subset
    dataset = Subset(dataset, range(min(args.num_samples, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Run inference
    run_inference(model, dataloader, device, args.output_dir, args.num_samples)

    print(f"Inference completed. Check {args.output_dir} for saved images.")


if __name__ == "__main__":
    main()