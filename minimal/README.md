# Minimal DINOv2 World Model

A simplified implementation of DINOv2-based world models for latent space prediction and planning.

## Components

- **Environment**: Simple 2D PointMaze with wall obstacles (`envs/pointmaze.py`)
- **Dataset**: NPZ trajectory loader (`dataset_npz.py`)
- **Encoder**: Frozen DINOv2 feature extractor (`min_dinov2.py`)
- **World Model**: Frame-causal Transformer for next-frame prediction (`transition.py`)
- **Planner**: Cross-Entropy Method for action sequence optimization (`cem.py`)

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Collect toy data** (for testing):
   ```bash
   python -m minimal.collect_random --out toy_npz --n 32 --T 40
   ```

3. **Or convert existing PointMaze data** (recommended for real training):
   ```bash
   python -m minimal.convert_pointmaze_data --input_dir /path/to/point_maze_data --output_dir converted_npz
   ```

4. **Train world model**:
   ```bash
   python -m minimal.train_min --data_dir converted_npz --seq_len 6 --batch_size 4 --max_steps 200 --device cpu
   ```

5. **Run planning**:
   ```bash
   python -m minimal.plan_min --ckpt checkpoints/best_model.pt --horizon 12 --device cpu
   ```

## Architecture

- **Encoder**: DINOv2 ViT patches → frozen latent space
- **World Model**: Transformer with causal masking for temporal prediction
- **Planning**: CEM optimizes action sequences in latent space
- **Environment**: 2D continuous control with collision detection

## Key Features

- ✅ CPU-compatible for smoke testing
- ✅ Minimal dependencies (torch, timm, numpy, tqdm)
- ✅ Frozen encoder (no pixel reconstruction)
- ✅ Frame-causal attention masking
- ✅ Latent space planning with CEM
- ✅ Robust to input shapes/dtypes

## Compatibility with Existing PointMaze Data

The minimal implementation is designed to work with existing PointMaze data, but requires conversion:

### Data Format Differences
- **Original**: Images stored as separate `.pth` files per episode (224×224 MuJoCo renders)
- **Converted**: Single `.npz` files per trajectory (64×64 resized images)

### Conversion Process
The `convert_pointmaze_data.py` script handles:
- Loading existing `.pth` trajectory files
- Resizing 224×224 images to 64×64 (compatible with minimal env)
- Converting action ranges to [-1, 1]
- Packaging into NPZ format

### Limitations
- **Resolution**: Downsamples from 224×224 to 64×64 (loss of detail)
- **Rendering**: MuJoCo 3D renders vs simple 2D top-down (different visual features)
- **State**: Minimal env uses 2D position vs 4D (pos + vel) in original
- **Physics**: Simplified collision vs MuJoCo physics simulation

### Recommended Usage
1. Use converted real data for training (better visual features)
2. Use minimal environment for testing/planning (faster, no MuJoCo dependency)
3. For production, consider upgrading minimal env to match original physics
