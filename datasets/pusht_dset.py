import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import cv2
import torchvision.transforms as transforms


class PushTDataset(Dataset):
    def __init__(self, data_dir, mode='train', seq_len=8, image_size=224):
        self.data_dir = data_dir
        self.mode = mode
        self.seq_len = seq_len
        self.image_size = image_size

        # Data augmentation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load data
        self._load_data()

    def _load_data(self):
        """Load all data into memory"""
        data_dir = os.path.join(self.data_dir, self.mode)

        # Load states, actions, and sequence lengths
        self.states = torch.load(os.path.join(data_dir, 'states.pth')).float()
        self.actions = torch.load(os.path.join(data_dir, 'rel_actions.pth')).float()

        with open(os.path.join(data_dir, 'seq_lengths.pkl'), 'rb') as f:
            self.seq_lengths = pickle.load(f)

        # Convert to list for easier indexing
        self.seq_lengths = [int(length) for length in self.seq_lengths]

        # Video directory
        self.video_dir = os.path.join(data_dir, 'obses')

        print(f"Loaded {len(self.states)} episodes")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        """Load a single trajectory"""
        states = self.states[idx]  # (max_seq_len, state_dim)
        actions = self.actions[idx]  # (max_seq_len, action_dim)
        seq_len = self.seq_lengths[idx]

        # Load video frames
        video_path = os.path.join(self.video_dir, f'episode_{idx:03d}.mp4')
        frames = self._load_video_frames(video_path, seq_len)

        # Process frames
        processed_frames = []
        for frame in frames:
            processed_frames.append(self.transform(frame))

        images_tensor = torch.stack(processed_frames)  # (seq_len, 3, H, W)

        # Extract proprioceptive state (gripper position from states)
        # States format: [gripper_x, gripper_y, object_x, object_y, object_angle]
        proprio_tensor = states[:seq_len, :2]  # Use gripper position as proprio

        # Create sequence windows
        T = seq_len
        if T < self.seq_len:
            # Pad if too short
            pad_len = self.seq_len - T
            images_tensor = torch.cat([images_tensor, images_tensor[-1:].repeat(pad_len, 1, 1, 1)], dim=0)
            actions = torch.cat([actions[:T], actions[T-1:].repeat(pad_len, 1)], dim=0)
            proprio_tensor = torch.cat([proprio_tensor, proprio_tensor[-1:].repeat(pad_len, 1)], dim=0)
        elif T > self.seq_len:
            # Random crop if too long
            start_idx = np.random.randint(0, T - self.seq_len + 1)
            images_tensor = images_tensor[start_idx:start_idx + self.seq_len]
            actions = actions[start_idx:start_idx + self.seq_len]
            proprio_tensor = proprio_tensor[start_idx:start_idx + self.seq_len]
        else:
            actions = actions[:T]

        # Create observation dict
        obs = {
            'visual': images_tensor,  # (seq_len, 3, H, W)
            'proprio': proprio_tensor  # (seq_len, proprio_dim)
        }

        return obs, actions

    def _load_video_frames(self, video_path, seq_len):
        """Load frames from MP4 video"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        frame_count = 0
        while frame_count < seq_len:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1

        cap.release()

        # If we don't have enough frames, repeat the last frame
        while len(frames) < seq_len:
            frames.append(frames[-1])

        return frames