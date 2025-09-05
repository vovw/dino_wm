import torch
import torchvision.transforms as transforms
import numpy as np


class Preprocessor:
    def __init__(self, image_size=224, action_stats=None, proprio_stats=None):
        self.image_size = image_size

        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Action and proprio stats for normalization
        self.action_stats = action_stats or {'mean': 0.0, 'std': 1.0}
        self.proprio_stats = proprio_stats or {'mean': 0.0, 'std': 1.0}

    def preprocess_image(self, image):
        """Preprocess single image"""
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)
            return self.image_transform(image)
        elif isinstance(image, torch.Tensor):
            # If already a tensor, just normalize
            return image.float() / 255.0
        else:
            return self.image_transform(image)

    def preprocess_images(self, images):
        """Preprocess batch of images"""
        processed = []
        for img in images:
            processed.append(self.preprocess_image(img))
        return torch.stack(processed)

    def preprocess_actions(self, actions):
        """Normalize actions"""
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()

        mean = self.action_stats['mean']
        std = self.action_stats['std']

        return (actions - mean) / std

    def preprocess_proprio(self, proprio):
        """Normalize proprioceptive states"""
        if isinstance(proprio, np.ndarray):
            proprio = torch.from_numpy(proprio).float()

        mean = self.proprio_stats['mean']
        std = self.proprio_stats['std']

        return (proprio - mean) / std

    def preprocess_trajectory(self, trajectory):
        """Preprocess full trajectory"""
        images = trajectory['images']
        actions = trajectory['actions']
        proprio = trajectory['proprio']

        processed_images = self.preprocess_images(images)
        processed_actions = self.preprocess_actions(actions)
        processed_proprio = self.preprocess_proprio(proprio)

        return {
            'visual': processed_images,
            'proprio': processed_proprio
        }, processed_actions

    @staticmethod
    def compute_stats(data_list, key):
        """Compute mean and std for normalization"""
        all_data = []
        for data in data_list:
            all_data.append(data[key])

        all_data = np.concatenate(all_data, axis=0)
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)

        return {'mean': mean, 'std': std}