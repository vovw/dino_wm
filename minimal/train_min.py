#!/usr/bin/env python3
"""
Minimal training script for world model in latent space.
Trains next-step prediction using DINOv2 encoder + Transformer decoder.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

from dataset_npz import NPZDataset
from min_dinov2 import DinoV2Encoder
from transition import TransitionModel


def create_dataloader(data_dir: str, batch_size: int, seq_len: int, val_split: float = 0.1):
    """Create train/val dataloaders."""
    dataset = NPZDataset(data_dir)

    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders with custom collate function for sequence windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_sequence_windows(batch, seq_len),
        num_workers=0  # Use 0 for debugging
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_sequence_windows(batch, seq_len),
        num_workers=0
    )

    return train_loader, val_loader, dataset


def collate_sequence_windows(batch, seq_len: int):
    """Collate function to create sequence windows from trajectories."""
    all_images = []
    all_actions = []

    for images, actions in batch:
        T = images.shape[0]

        # Create windows of length seq_len
        for start in range(0, T - seq_len + 1, seq_len // 2):  # 50% overlap
            end = min(start + seq_len, T)
            if end - start >= seq_len:
                window_images = images[start:end]  # (seq_len, 3, H, W)
                window_actions = actions[start:end-1]  # (seq_len-1, A)
                all_images.append(window_images)
                all_actions.append(window_actions)

    if len(all_images) == 0:
        # Fallback: just take first seq_len frames
        images, actions = batch[0]
        all_images = [images[:seq_len]]
        all_actions = [actions[:seq_len-1]]

    # Stack into batch
    batch_images = torch.stack(all_images)  # (B, seq_len, 3, H, W)
    batch_actions = torch.stack(all_actions)  # (B, seq_len-1, A)

    return batch_images, batch_actions


def train_epoch(model: TransitionModel,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                use_amp: bool = False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    scaler = torch.amp.GradScaler() if use_amp else None

    for images, actions in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                loss = model.compute_loss(images, actions)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.compute_loss(images, actions)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model: TransitionModel,
                  dataloader: DataLoader,
                  device: torch.device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, actions in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            actions = actions.to(device)

            loss = model.compute_loss(images, actions)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main(args):
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create dataloaders
    print(f"Loading data from {args.data_dir}")
    train_loader, val_loader, dataset = create_dataloader(
        args.data_dir, args.batch_size, args.seq_len
    )

    # Create model
    print("Creating model...")
    encoder = DinoV2Encoder(image_size=args.image_size)
    model = TransitionModel(encoder, action_dim=dataset.get_action_dim())
    model = model.to(device)

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    # Create optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    best_val_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.max_steps):
        print(f"\nEpoch {epoch + 1}/{args.max_steps}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.use_amp)
        print(".4f")

        # Validate
        if val_loader is not None:
            val_loss = validate_epoch(model, val_loader, device)
            print(".4f")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, args.output_dir, epoch, val_loss)
                print(".4f")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, args.output_dir, epoch, train_loss, prefix="checkpoint")

    # Save final model
    save_checkpoint(model, args.output_dir, args.max_steps, train_loss, prefix="final")


def save_checkpoint(model: TransitionModel, output_dir: str, epoch: int, loss: float, prefix: str = "best"):
    """Save model checkpoint."""
    checkpoint = {
        "wm": model.transformer.state_dict(),
        "embed_dim": model.embed_dim,
        "grid": encoder.get_patch_grid(),
        "epoch": epoch,
        "loss": loss
    }

    checkpoint_path = Path(output_dir) / f"{prefix}_model.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train minimal world model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with NPZ files")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=6, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=200, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for encoder")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--save_freq", type=int, default=50, help="Save checkpoint every N epochs")

    args = parser.parse_args()
    main(args)
