import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, List
from einops import rearrange, repeat
from datasets.traj_dset import TrajDataset


class ClipDataset(Dataset):
    """
    Dataset that yields video clips for V-JEPA-2 training.
    
    Transforms frame-based trajectory data into temporal clips with action conditioning.
    Each sample contains:
    - History clip: (clip_t-H+1:t, actions_t-H+1:t) 
    - Target frame: next_frame_t+1
    """
    
    def __init__(
        self,
        traj_dataset: TrajDataset,
        history_length: int = 4,
        frameskip: int = 1,
        tubelet_t: int = 2,
        min_traj_length: int = 8,
        augment_temporal: bool = False,
        pad_mode: str = "repeat"  # "repeat", "zero", "mirror"
    ):
        """
        Args:
            traj_dataset: Underlying trajectory dataset
            history_length: Number of frames in history clip (H)
            frameskip: Skip frames (temporal downsampling)
            tubelet_t: Temporal extent of V-JEPA-2 tubelets
            min_traj_length: Minimum trajectory length to include
            augment_temporal: Whether to randomly vary history length
            pad_mode: How to pad when insufficient history
        """
        self.traj_dataset = traj_dataset
        self.history_length = history_length
        self.frameskip = frameskip
        self.tubelet_t = tubelet_t
        self.min_traj_length = min_traj_length
        self.augment_temporal = augment_temporal
        self.pad_mode = pad_mode
        
        # Compute valid slices for clip extraction
        self._compute_valid_slices()
        
        # Dataset properties
        self.proprio_dim = getattr(traj_dataset, 'proprio_dim', 0)
        self.action_dim = getattr(traj_dataset, 'action_dim', 0)
        self.state_dim = getattr(traj_dataset, 'state_dim', 0)
        
        print(f"ClipDataset: {len(self.slices)} valid clips from {len(traj_dataset)} trajectories")
        print(f"History length: {history_length}, frameskip: {frameskip}")
        
    def _compute_valid_slices(self):
        """Compute valid (traj_idx, start_frame) pairs for clip extraction."""
        self.slices = []
        
        for traj_idx in range(len(self.traj_dataset)):
            traj_length = self.traj_dataset.get_seq_length(traj_idx)
            
            if traj_length < self.min_traj_length:
                continue
                
            # We need at least (history_length + 1) frames to get history + target
            min_frames_needed = self.history_length * self.frameskip + 1
            
            if traj_length < min_frames_needed:
                continue
                
            # All valid starting frames for this trajectory
            max_start = traj_length - min_frames_needed
            valid_starts = list(range(0, max_start + 1, self.frameskip))
            
            for start_frame in valid_starts:
                self.slices.append((traj_idx, start_frame))
        
        # Shuffle slices for better training
        np.random.shuffle(self.slices)
        
    def _pad_sequence(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Pad sequence to target length.
        
        Args:
            sequence: (T, ...) sequence tensor
            target_length: Desired sequence length
            
        Returns:
            padded_sequence: (target_length, ...) padded tensor
        """
        current_length = sequence.size(0)
        if current_length >= target_length:
            return sequence[:target_length]
        
        pad_length = target_length - current_length
        
        if self.pad_mode == "repeat":
            # Repeat first frame
            first_frame = sequence[0:1]  # (1, ...)
            padding = first_frame.expand(pad_length, *(-1,) * (sequence.ndim - 1))
            return torch.cat([padding, sequence], dim=0)
            
        elif self.pad_mode == "zero":
            # Zero padding  
            pad_shape = (pad_length,) + sequence.shape[1:]
            padding = torch.zeros(pad_shape, dtype=sequence.dtype, device=sequence.device)
            return torch.cat([padding, sequence], dim=0)
            
        elif self.pad_mode == "mirror":
            # Mirror padding
            if current_length == 1:
                padding = sequence[0:1].expand(pad_length, *(-1,) * (sequence.ndim - 1))
            else:
                # Create mirrored sequence
                mirrored = torch.flip(sequence, dims=[0])
                # Repeat as needed
                repeats = (pad_length - 1) // current_length + 1
                extended = mirrored.repeat(repeats, *([1] * (sequence.ndim - 1)))
                padding = extended[:pad_length]
            return torch.cat([padding, sequence], dim=0)
            
        else:
            raise ValueError(f"Unknown pad_mode: {self.pad_mode}")
    
    def _extract_clip(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor, 
                     start_frame: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Extract history clip and next frame from trajectory.
        
        Args:
            obs: Full trajectory observations
            actions: Full trajectory actions  
            start_frame: Starting frame index
            
        Returns:
            history_obs: History observations
            history_actions: History actions
            next_frame: Next frame observation
        """
        # Determine actual history length (for temporal augmentation)
        if self.augment_temporal and self.training:
            # Randomly vary history length between 2 and self.history_length
            actual_history = np.random.randint(2, self.history_length + 1)
        else:
            actual_history = self.history_length
            
        # Extract frames with frameskip
        end_frame = start_frame + actual_history * self.frameskip
        frame_indices = list(range(start_frame, end_frame, self.frameskip))
        
        # Extract history observations
        history_obs = {}
        for key, value in obs.items():
            if len(frame_indices) <= value.size(0):
                history_obs[key] = value[frame_indices]  # (H, ...)
            else:
                # Need padding
                available_frames = value[start_frame:]
                history_obs[key] = self._pad_sequence(available_frames, actual_history)
        
        # Extract history actions (actions that led to these frames)
        if len(frame_indices) <= actions.size(0):
            history_actions = actions[frame_indices]  # (H, action_dim)
        else:
            available_actions = actions[start_frame:]
            history_actions = self._pad_sequence(available_actions, actual_history)
        
        # Next frame (target for prediction)
        next_frame_idx = min(end_frame, obs['visual'].size(0) - 1)
        next_frame = {}
        for key, value in obs.items():
            next_frame[key] = value[next_frame_idx:next_frame_idx+1]  # (1, ...)
            
        # Ensure temporal dimension compatibility with tubelet_t
        if actual_history < self.tubelet_t:
            # Pad to tubelet_t if needed
            for key in history_obs:
                history_obs[key] = self._pad_sequence(history_obs[key], self.tubelet_t)
            history_actions = self._pad_sequence(history_actions, self.tubelet_t)
        
        return history_obs, history_actions, next_frame
    
    def __len__(self) -> int:
        return len(self.slices)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a training sample.
        
        Returns:
            history_obs: Dict with keys like 'visual' (H, 3, H, W), 'proprio' (H, D)
            history_actions: (H, action_dim) actions
            next_frame: Dict with same structure as history_obs but (1, ...)
        """
        traj_idx, start_frame = self.slices[idx]
        
        # Get full trajectory
        obs, actions, states = self.traj_dataset[traj_idx]
        
        # Extract clip
        history_obs, history_actions, next_frame = self._extract_clip(
            obs, actions, start_frame
        )
        
        return history_obs, history_actions, next_frame


class MultiTrajectoryClipDataset(Dataset):
    """
    Clip dataset that can handle multiple trajectory datasets.
    Useful for training on multiple environments or data sources.
    """
    
    def __init__(
        self,
        traj_datasets: List[TrajDataset],
        dataset_weights: Optional[List[float]] = None,
        **clip_kwargs
    ):
        """
        Args:
            traj_datasets: List of trajectory datasets
            dataset_weights: Optional weights for sampling from each dataset
            **clip_kwargs: Arguments passed to ClipDataset
        """
        self.clip_datasets = [
            ClipDataset(dataset, **clip_kwargs) 
            for dataset in traj_datasets
        ]
        
        # Compute dataset weights
        if dataset_weights is None:
            dataset_weights = [len(ds) for ds in self.clip_datasets]
        self.dataset_weights = np.array(dataset_weights, dtype=np.float32)
        self.dataset_weights /= self.dataset_weights.sum()
        
        # Compute cumulative dataset sizes for indexing
        self.dataset_sizes = [len(ds) for ds in self.clip_datasets]
        self.cumulative_sizes = np.cumsum([0] + self.dataset_sizes)
        self.total_size = self.cumulative_sizes[-1]
        
        print(f"MultiTrajectoryClipDataset: {len(self.clip_datasets)} datasets, "
              f"{self.total_size} total clips")
        
    def __len__(self) -> int:
        return self.total_size
    
    def __getitem__(self, idx: int):
        """Get item by global index."""
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        
        return self.clip_datasets[dataset_idx][local_idx]
    
    def sample_weighted(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Sample according to dataset weights."""
        dataset_idx = np.random.choice(len(self.clip_datasets), p=self.dataset_weights)
        local_idx = np.random.randint(len(self.clip_datasets[dataset_idx]))
        return self.clip_datasets[dataset_idx][local_idx]


def create_clip_dataloader(
    traj_datasets,
    batch_size: int = 32,
    history_length: int = 4,
    frameskip: int = 1,
    tubelet_t: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Factory function to create clip dataloader.
    
    Args:
        traj_datasets: Single TrajDataset or list of TrajDataset
        batch_size: Batch size
        history_length: Frames in history clip
        frameskip: Temporal downsampling
        tubelet_t: Temporal extent of tubelets
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        **dataset_kwargs: Additional ClipDataset arguments
        
    Returns:
        DataLoader for clip training
    """
    if isinstance(traj_datasets, list):
        dataset = MultiTrajectoryClipDataset(
            traj_datasets, 
            history_length=history_length,
            frameskip=frameskip,
            tubelet_t=tubelet_t,
            **dataset_kwargs
        )
    else:
        dataset = ClipDataset(
            traj_datasets,
            history_length=history_length,
            frameskip=frameskip, 
            tubelet_t=tubelet_t,
            **dataset_kwargs
        )
    
    def collate_fn(batch):
        """Custom collate function for clip data."""
        history_obs_batch = {}
        history_actions_batch = []
        next_frame_batch = {}
        
        # Collect all keys
        history_keys = set()
        next_frame_keys = set()
        for history_obs, _, next_frame in batch:
            history_keys.update(history_obs.keys())
            next_frame_keys.update(next_frame.keys())
        
        # Stack observations
        for key in history_keys:
            history_obs_batch[key] = torch.stack([
                sample[0][key] for sample in batch
            ], dim=0)
            
        for key in next_frame_keys:
            next_frame_batch[key] = torch.stack([
                sample[2][key] for sample in batch  
            ], dim=0)
        
        # Stack actions
        history_actions_batch = torch.stack([
            sample[1] for sample in batch
        ], dim=0)
        
        return history_obs_batch, history_actions_batch, next_frame_batch
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )