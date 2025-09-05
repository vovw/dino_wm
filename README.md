# DINOv2 World Model for Robotics

A PyTorch implementation of a visual world model for robotic manipulation using DINOv2 as the visual encoder. The model learns to predict and reconstruct visual sequences from pushing tasks in the Push-T dataset.

## Features

- **DINOv2 Encoder**: Uses pre-trained DINOv2 for robust visual feature extraction
- **Temporal Predictor**: ViT-based predictor for sequence modeling
- **Image Reconstruction**: Transposed convolution decoder for visual reconstruction
- **Multi-Modal**: Incorporates proprioceptive and action information
- **GPU Optimized**: CUDA acceleration with memory-efficient training

## Project Structure

```
dinowm/
├── main.py                 # Main training script
├── inference.py           # Inference and evaluation script
├── preprocessor.py        # Data preprocessing utilities
├── pyproject.toml         # Project dependencies
├── download_data.sh       # Data download script
├── models/
│   ├── dino.py           # DINOv2 encoder wrapper
│   ├── vit.py            # ViT predictor model
│   ├── proprio.py        # Proprioceptive encoder
│   ├── dummy.py          # Action encoder
│   ├── visual_world_model.py  # Main world model
│   └── decoder/
│       └── transposed_conv.py  # Image decoder
├── datasets/
│   └── pusht_dset.py     # Push-T dataset loader
└── checkpoints/          # Model checkpoints (gitignored)
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- uv package manager

### Setup
```bash
# Clone repository
git clone <repository-url>
cd dinowm

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

## Data Setup

### Download Push-T Dataset
```bash
# Download and extract dataset
./download_data.sh

# Expected structure:
# data/pusht_noise/
# ├── train/
# │   ├── states.pth
# │   ├── rel_actions.pth
# │   ├── seq_lengths.pkl
# │   └── obses/ (video files)
# └── val/
#     └── ... (same structure)
```

## Training

### Basic Training
```bash
# Train on full dataset
uv run python main.py \
  --data_dir ./data/pusht_noise/pusht_noise \
  --checkpoint_dir ./checkpoints \
  --num_epochs 500 \
  --batch_size 1 \
  --log_interval 50
```

### Training Parameters
- `--data_dir`: Path to dataset directory
- `--checkpoint_dir`: Where to save checkpoints
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size (keep small for memory)
- `--log_interval`: Logging frequency
- `--image_size`: Input image size (default: 224)
- `--num_hist`: History frames (default: 4)
- `--enc_lr`: Encoder learning rate (default: 1e-4)
- `--pred_lr`: Predictor learning rate (default: 1e-3)
- `--dec_lr`: Decoder learning rate (default: 1e-4)

### Monitoring Training
- Loss logs saved to `checkpoints/training_log.csv`
- Checkpoints saved every epoch as `model_{epoch}.pth`
- Use the plotting script for visualization

## Inference

### Run Inference on Validation Set
```bash
# Generate reconstructions
uv run python inference.py \
  --checkpoint_path ./checkpoints/model_111.pth \
  --data_dir ./data/pusht_noise/pusht_noise \
  --output_dir ./results
```

### Inference Parameters
- `--checkpoint_path`: Path to trained model
- `--data_dir`: Dataset directory
- `--output_dir`: Where to save results
- `--num_samples`: Number of samples to process (default: 5)

## File Descriptions

### Core Files
- **`main.py`**: Main training script with data loading, model setup, and training loop
- **`inference.py`**: Evaluation script for running inference and generating reconstructions
- **`preprocessor.py`**: Data preprocessing utilities (normalization, transforms)

### Models (`models/`)
- **`dino.py`**: Wrapper for DINOv2 encoder with feature extraction
- **`vit.py`**: Vision Transformer implementation for sequence prediction
- **`visual_world_model.py`**: Main world model combining all components
- **`proprio.py`**: Encoder for proprioceptive (robot state) information
- **`dummy.py`**: Simple encoder for action sequences
- **`decoder/transposed_conv.py`**: Image reconstruction decoder

### Data (`datasets/`)
- **`pusht_dset.py`**: Push-T dataset loader with video frame extraction

## Model Architecture

```
Input: [Batch, Time, 3, 224, 224] + Actions + Proprio
       ↓
DINOv2 Encoder → Visual Features [Batch, Time, 256, 384]
       ↓
Predictor (ViT) → Predicted Features [Batch, Time, 256, 384]
       ↓
Decoder → Reconstructed Images [Batch, Time, 3, 224, 224]
```

## Loss Function

```
Total Loss = Reconstruction Loss + Prediction Loss

Reconstruction Loss: MSE(predicted_images, original_images)
Prediction Loss: MSE(predicted_features, encoded_features)
```

## Troubleshooting

### Memory Issues
- Reduce batch_size to 1
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Monitor with `nvidia-smi`

### Data Issues
- Ensure dataset is properly downloaded
- Check file paths in data directory
- Verify video files are accessible

### Training Issues
- Check learning rates are appropriate
- Monitor loss convergence
- Ensure GPU memory is sufficient

## SSH Usage

This codebase is designed for headless SSH environments:

- No GUI dependencies
- All outputs saved to files
- Progress logged to console
- Memory-efficient for remote training

## Contributing

1. Follow the existing code style
2. Add type hints where possible
3. Update README for new features
4. Test on validation set before committing

## License

[Add license information]