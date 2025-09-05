import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import sys

# Add current directory to path
sys.path.append('.')

from models.dino import DinoV2Encoder
from models.proprio import ProprioEncoder
from models.dummy import DummyEncoder as ActionEncoder
from models.vit import ViT
from models.decoder.transposed_conv import TransposedConvDecoder
from models.visual_world_model import VWorldModel
from datasets.pusht_dset import PushTDataset
from preprocessor import Preprocessor


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model components
    encoder = DinoV2Encoder(name='dinov2_vits14', feature_key='x_norm_patchtokens')
    proprio_encoder = ProprioEncoder(in_chans=2, emb_dim=64)  # gripper x,y
    action_encoder = ActionEncoder(in_chans=2, emb_dim=64)  # dx, dy
    predictor = ViT(num_patches=256, num_frames=args.num_hist, dim=384)
    decoder = TransposedConvDecoder(emb_dim=384, num_patches=256)

    # World model
    model = VWorldModel(
        image_size=args.image_size,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
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

    # Dataset and dataloader
    dataset = PushTDataset(
        data_dir=args.data_dir,
        mode='train',
        seq_len=args.num_hist,
        image_size=args.image_size
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Preprocessor
    preprocessor = Preprocessor(image_size=args.image_size)

    # Optimizers
    enc_opt = torch.optim.Adam(model.encoder.parameters(), lr=args.enc_lr)
    pred_opt = torch.optim.AdamW([
        {'params': model.predictor.parameters()},
        {'params': model.action_encoder.parameters()},
        {'params': model.proprio_encoder.parameters()}
    ], lr=args.pred_lr * 0.1)  # Reduce predictor LR
    dec_opt = torch.optim.Adam(model.decoder.parameters(), lr=args.dec_lr * 2)  # Increase decoder LR

    # Loss functions
    mse_loss = nn.MSELoss()

    # Training loop
    import csv
    log_file = os.path.join(args.checkpoint_dir, 'training_log.csv')
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'batch', 'loss'])

        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (obs, actions) in enumerate(dataloader):
                # Move data to device
                obs = {k: v.to(device) for k, v in obs.items()}
                actions = actions.to(device)

                # Preprocess (simplified for now)
                # obs['visual'] = preprocessor.preprocess_images(obs['visual'])
                # obs['proprio'] = preprocessor.preprocess_proprio(obs['proprio'])
                # actions = preprocessor.preprocess_actions(actions)

                # Forward pass
                z_pred, obs_pred, obs_recon, _ = model(obs, actions)

                # Compute losses
                recon_loss = mse_loss(obs_pred, obs_recon)
                pred_loss = mse_loss(z_pred, model.encode_obs(obs))

                loss = recon_loss + pred_loss

                # Backward pass
                enc_opt.zero_grad()
                pred_opt.zero_grad()
                dec_opt.zero_grad()

                loss.backward()

                enc_opt.step()
                pred_opt.step()
                dec_opt.step()

                epoch_loss += loss.item()

                if batch_idx % args.log_interval == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    writer.writerow([epoch, batch_idx, loss.item()])

            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Epoch {epoch} completed. Average loss: {epoch_loss / len(dataloader):.4f}")
            writer.writerow([epoch, 'avg', epoch_loss / len(dataloader)])

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_hist', type=int, default=4)
    parser.add_argument('--num_pred', type=int, default=4)
    parser.add_argument('--proprio_dim', type=int, default=64)
    parser.add_argument('--action_dim', type=int, default=64)
    parser.add_argument('--enc_lr', type=float, default=1e-4)
    parser.add_argument('--pred_lr', type=float, default=1e-3)
    parser.add_argument('--dec_lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    main(args)