# **DINO-WM**: World Models on Pre-trained Visual Features enable Zero-shot Planning
[[Paper]](https://arxiv.org/abs/2411.04983) [[Data]](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) [[Project Website]](https://dino-wm.github.io/) 

[Gaoyue Zhou](https://gaoyuezhou.github.io/), [Hengkai Pan](https://hengkaipan.github.io/), [Yann LeCun](https://yann.lecun.com/) and [Lerrel Pinto](https://www.lerrelpinto.com/), New York University, Meta AI

![teaser_figure](assets/intro.png)

## Research Experiment: Visual Foundation Models for World Modeling

This repository contains an experimental comparison of different visual foundation models for world modeling tasks, specifically comparing **DINOv2**, **DINOv3**, and **V-JEPA 2** architectures in the context of visual world model learning.

### Experiment Overview

**Research Question**: How do different pre-trained visual foundation models perform when adapted for world modeling tasks?

**Models Under Comparison**:
- **DINOv2**: Self-supervised vision transformer with knowledge distillation
- **DINOv3**: Improved DINO architecture with better training strategies  
- **V-JEPA 2**: Video Joint Embedding Predictive Architecture with mask-denoising in representation space

**Key Innovation**: V-JEPA 2 introduces a novel approach to video understanding through mask-denoising in representation space, potentially offering advantages for temporal modeling in world models.

### V-JEPA 2 Architecture Details

V-JEPA 2 represents a significant advancement in self-supervised video pretraining:

**Core Methodology**:
- **Mask-Denoising in Representation Space**: Predicts learned representations of masked video patches
- **Architecture**: Vision Transformer encoder + predictor with 3D-RoPE (Rotary Position Embedding)
- **Training Objective**: `minimize ||P_φ(Δ_y, E_θ(x)) - sg(E_θ̄(y))||₁`
- **Scaling**: Trained on 1M+ hours of video, models up to 1B+ parameters

**Key Advantages for World Modeling**:
- Temporal understanding through video pretraining
- Mask-denoising objective aligns with prediction tasks
- 3D-RoPE provides better spatiotemporal position encoding
- Progressive resolution training for efficiency

## Codebase Structure & Setup Guide

### What This Codebase Actually Is

This is a **research experiment codebase** for comparing visual foundation models (DINOv2, DINOv3, V-JEPA 2) in world modeling tasks. It's built on top of the original DINO-WM but heavily modified for experimental purposes.

### What's Useful vs. What's Useless

#### USEFUL - Core Components You Need

**Essential Files**:
- `train.py` - Main training script with Hydra integration
- `plan.py` - Planning and evaluation script
- `configs.py` - Python configuration classes (simplified from Hydra)
- `conf/train_local.yaml` - Local development configuration
- `models/` - All model implementations (encoders, decoders, predictors)
- `datasets/` - Dataset loaders for all environments
- `download_data.sh` - Automated dataset download script

**Key Model Files**:
- `models/dino.py` - DINOv2 encoder implementation
- `models/visual_world_model.py` - Main world model architecture
- `models/vqvae.py` - VQ-VAE decoder
- `models/vit.py` - Vision Transformer predictor
- `models/encoder/` - Various encoder implementations

**Environment Implementations**:
- `env/pusht/` - Robot manipulation environment
- `env/pointmaze/` - Navigation environment
- `env/wall/` - Navigation with obstacles
- `env/deformable_env/` - Physics simulation environment

#### USELESS - Can Be Ignored/Removed

**Legacy/Unused Files**:
- `train_pusht.py` - Standalone PushT trainer (use `train.py` instead)
- `train_granular.py` - Standalone granular trainer (use `train.py` instead)
- `plan_pusht.py` - Standalone PushT planner (use `plan.py` instead)
- `plan_granular.py` - Standalone granular planner (use `plan.py` instead)
- `distributed_fn/` - SLURM/distributed training (not needed for local experiments)
- `config_utils.py` - Hydra utilities (mostly unused now)

**Overcomplicated Dependencies**:
- `submitit` - SLURM job submission (unnecessary for local work)
- `pybullet` - Only needed for granular environment
- `pyglet` - Only needed for some visualizations
- `shapely` - Only needed for geometric computations

### Minimal Setup (Recommended)

#### 1. **Core Dependencies Only**
```bash
# Install only what you need
uv add torch torchvision numpy einops accelerate wandb tqdm pillow
uv add gym mujoco-py omegaconf hydra-core gdown
```

#### 2. **Environment-Specific Dependencies**
```bash
# For PushT only (recommended for experiments)
# No additional dependencies needed

# For PointMaze/Wall environments
uv add pymunk pygame

# For Deformable environment (optional, complex setup)
uv add pybullet pybind11
```

#### 3. **Clean Up Unnecessary Files**
```bash
# Remove standalone scripts (use main train.py instead)
rm train_pusht.py train_granular.py plan_pusht.py plan_granular.py

# Remove distributed training code
rm -rf distributed_fn/

# Remove unused config utilities
rm config_utils.py
```

### Simplified Project Structure

After cleanup, your codebase should look like:
```
dino_wm/
├── train.py                 # Main training script
├── plan.py                  # Main planning/evaluation script
├── configs.py               # Configuration classes
├── download_data.sh         # Dataset download script
├── conf/
│   ├── train_local.yaml     # Local training config
│   └── env/                 # Environment configs
├── models/                  # All model implementations
│   ├── dino.py             # DINOv2 encoder
│   ├── visual_world_model.py # Main world model
│   ├── vqvae.py            # VQ-VAE decoder
│   ├── vit.py              # ViT predictor
│   └── encoder/            # Additional encoders
├── datasets/               # Dataset loaders
├── env/                    # Environment implementations
├── planning/               # Planning algorithms
└── metrics/                # Evaluation metrics
```

### Quick Start

#### Installation

1. **Install uv** (fast Python package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup the repository**:
```bash
git clone https://github.com/gaoyuezhou/dino_wm.git
cd dino_wm
uv sync
```

3. **Activate the environment**:
```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Understanding the Codebase Components

#### **Core Training Pipeline** (`train.py`)
- **What it does**: Main training script with Hydra configuration system
- **Key features**: 
  - Supports all environments (PushT, PointMaze, Wall, Deformable)
  - Handles different encoders (DINOv2, DINOv3, custom V-JEPA 2)
  - Multi-stage training with progressive resolution
  - Wandb logging and checkpointing
- **Usage**: `python train.py --config-name train_local`

#### **Configuration System** (`configs.py` + `conf/`)
- **What it does**: Manages all training parameters and model configurations
- **Key files**:
  - `configs.py`: Python dataclasses for type-safe configuration
  - `conf/train_local.yaml`: Local development settings
  - `conf/env/`: Environment-specific configurations
- **Why it's useful**: Easy to modify experiments without changing code

#### **Model Architecture** (`models/`)
- **`visual_world_model.py`**: Main world model that combines encoder + decoder + predictor
- **`dino.py`**: DINOv2 encoder implementation (your baseline)
- **`vqvae.py`**: VQ-VAE decoder for image reconstruction
- **`vit.py`**: Vision Transformer predictor for future frame prediction
- **`encoder/`**: Additional encoder implementations (DINOv3, V-JEPA 2, etc.)

#### **Environment Implementations** (`env/`)
- **`pusht/`**: Robot manipulation with 2D physics (recommended for experiments)
- **`pointmaze/`**: 2D navigation tasks
- **`wall/`**: Navigation with obstacles
- **`deformable_env/`**: Complex physics simulation (avoid for initial experiments)

#### **Dataset Loaders** (`datasets/`)
- **`pusht_dset.py`**: PushT dataset with trajectory loading
- **`point_maze_dset.py`**: PointMaze dataset
- **`wall_dset.py`**: Wall environment dataset
- **`deformable_env_dset.py`**: Deformable physics dataset
- **`img_transforms.py`**: Image preprocessing and augmentation

#### **Planning & Evaluation** (`plan.py` + `planning/`)
- **What it does**: Evaluates trained models using planning algorithms
- **Planning methods**: Gradient Descent, Cross-Entropy Method (CEM)
- **Evaluation metrics**: Success rate, planning efficiency, reconstruction quality
- **Usage**: `python plan.py --env pusht --model-path ./outputs/...`

### Recommended Development Workflow

#### **1. Start Simple (PushT Only)**
```bash
# Download only PushT dataset
curl -L "https://osf.io/download/k2d8w/" -o data/pusht_dataset.zip
unzip data/pusht_dataset.zip -d data/pusht_noise/

# Set environment variable
export DATASET_DIR=$(pwd)/data

# Train with DINOv2 baseline
python train.py --config-name train_local --env pusht
```

#### **2. Add More Environments Gradually**
```bash
# Add PointMaze (requires pymunk, pygame)
uv add pymunk pygame
python train.py --config-name train_local --env point_maze

# Add Wall environment
python train.py --config-name train_local --env wall_single
```

#### **3. Experiment with Different Encoders**
```bash
# DINOv2 baseline
python train.py --config-name train_local --encoder dino

# DINOv3 comparison (when implemented)
python train.py --config-name train_local --encoder dinov3

# Custom V-JEPA 2 (when implemented)
python train.py --config-name train_local --encoder vjepa2
```

### Common Issues & Solutions

#### **"Module not found" errors**
- **Cause**: Missing dependencies or wrong Python environment
- **Solution**: `uv sync` and `source .venv/bin/activate`

#### **"CUDA out of memory" errors**
- **Cause**: Batch size too large for your GPU
- **Solution**: Reduce `batch_size` in `conf/train_local.yaml` (try 16 or 8)

#### **"Dataset not found" errors**
- **Cause**: `DATASET_DIR` not set or wrong path
- **Solution**: `export DATASET_DIR=$(pwd)/data` and run `./download_data.sh`

#### **"MuJoCo not found" errors**
- **Cause**: MuJoCo not installed or LD_LIBRARY_PATH not set
- **Solution**: Follow MuJoCo installation steps above

### Codebase Cleanup Script

Create a cleanup script to remove unnecessary files:

```bash
#!/bin/bash
# cleanup.sh - Remove unnecessary files for cleaner codebase

echo "Cleaning up DINO-WM codebase..."

# Remove standalone scripts (use main train.py instead)
rm -f train_pusht.py train_granular.py plan_pusht.py plan_granular.py

# Remove distributed training code
rm -rf distributed_fn/

# Remove unused config utilities
rm -f config_utils.py

# Remove development files
rm -rf __pycache__/
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

echo "Cleanup complete! Use train.py and plan.py for all operations."
```

### What You Actually Need to Focus On

**For Your Research Experiment**:
1. **`models/dino.py`** - Understand DINOv2 implementation
2. **`models/visual_world_model.py`** - Core world model architecture
3. **`train.py`** - Training pipeline and experiment management
4. **`conf/train_local.yaml`** - Your experiment configurations
5. **`plan.py`** - Evaluation and comparison metrics

**Ignore Everything Else**:
- Standalone scripts (`train_pusht.py`, etc.)
- Distributed training code (`distributed_fn/`)
- Complex environment setups (deformable)
- SLURM/job submission utilities

#### Dataset Setup

**Automated Download** (Recommended):
```bash
./download_data.sh
```

**Manual Setup**:
```bash
# Download datasets from [here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28)
export DATASET_DIR=/path/to/your/datasets
```

Expected structure:
```
data/
├── pusht_noise/     # Robot manipulation (PushT)
├── point_maze/      # Navigation tasks
├── wall_single/     # Navigation with obstacles
└── deformable/      # Rope and granular physics
```

#### MuJoCo Installation (Required)

```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz
```

Add to your shell configuration:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia  # For NVIDIA GPUs
```

### Running Experiments

#### Local Training Configuration

The repository includes optimized configurations for local development:

```bash
# Use the local training config
python train.py --config-name train_local

# Or specify individual components
python train.py \
  --env pusht \
  --encoder dino \
  --decoder vqvae \
  --predictor vit \
  --epochs 10 \
  --batch_size 32
```

#### Model Comparison Experiments

**DINOv2 Baseline**:
```bash
python train.py --config-name train_local --encoder dino
```

**DINOv3 Comparison**:
```bash
python train.py --config-name train_local --encoder dinov3
```

**V-JEPA 2 (Custom Implementation)**:
```bash
python train.py --config-name train_local --encoder vjepa2
```

#### Environment-Specific Training

**PushT (Robot Manipulation)**:
```bash
python train.py --env pusht --config-name train_local
```

**PointMaze (Navigation)**:
```bash
python train.py --env point_maze --config-name train_local
```

**Wall Environment**:
```bash
python train.py --env wall_single --config-name train_local
```

**Deformable (Physics Simulation)**:
```bash
python train.py --env deformable --config-name train_local
```

### Evaluation and Planning

#### Model Evaluation
```bash
# Evaluate trained model
python plan.py \
  --env pusht \
  --model-path ./outputs/2024-01-15/12-30-45 \
  --model-epoch latest \
  --output-dir ./plan_outputs
```

#### Planning Algorithms
- **Gradient Descent (GD)**: Direct optimization in action space
- **Cross-Entropy Method (CEM)**: Sampling-based optimization
- **Goal Sources**: Dataset goals vs. random states
- **Planning Horizons**: Configurable prediction lengths

### Configuration System

The repository uses Hydra for flexible configuration management:

**Key Configuration Files**:
- `conf/train_local.yaml`: Local development settings
- `conf/env/pusht.yaml`: PushT environment configuration
- `conf/encoder/`: Model-specific encoder configurations
- `conf/decoder/`: Decoder architecture settings
- `conf/predictor/`: Prediction head configurations

**Custom Configurations**:
```yaml
# Example: Custom V-JEPA 2 config
encoder:
  _target_: models.vjepa2_encoder.VJEPA2Encoder
  model_size: "large"  # small, base, large, giant
  use_3d_rope: true
  mask_ratio: 0.75
  temporal_window: 16
```

### Experimental Design

#### Evaluation Metrics
- **Reconstruction Quality**: MSE, SSIM, LPIPS
- **Prediction Accuracy**: Future frame prediction error
- **Planning Success Rate**: Goal-reaching performance
- **Computational Efficiency**: Training time, inference speed
- **Sample Efficiency**: Performance vs. data requirements

#### Ablation Studies
- **Encoder Architecture**: DINOv2 vs. DINOv3 vs. V-JEPA 2
- **Temporal Modeling**: Frame-by-frame vs. video-based approaches
- **Masking Strategies**: Random vs. structured vs. learned masking
- **Resolution Scaling**: Progressive vs. fixed resolution training

### Research Goals

1. **Architecture Comparison**: Systematic evaluation of visual foundation models
2. **Temporal Understanding**: How video pretraining affects world model performance
3. **Sample Efficiency**: Which models require less data for effective world modeling
4. **Generalization**: Cross-environment transfer capabilities
5. **Computational Trade-offs**: Performance vs. efficiency analysis

### Expected Findings

**Hypotheses**:
- V-JEPA 2 should excel at temporal modeling due to video pretraining
- DINOv3 may offer better sample efficiency than DINOv2
- Mask-denoising objective should improve prediction quality
- 3D-RoPE should provide better spatiotemporal understanding

**Research Questions**:
- Does video pretraining translate to better world model performance?
- How does mask-denoising compare to standard reconstruction objectives?
- What are the computational trade-offs between architectures?
- Which model generalizes best across different environments?

### Technical Implementation

#### Key Components
- **Visual Encoders**: DINOv2, DINOv3, V-JEPA 2 implementations
- **World Model**: VQ-VAE decoder + ViT predictor
- **Training Pipeline**: Multi-stage training with progressive resolution
- **Evaluation Suite**: Comprehensive metrics and planning algorithms

#### Optimization Strategies
- **Learning Rate Scheduling**: Warmup-constant-decay for stable training
- **Progressive Resolution**: Start low-res, scale up during training
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Accumulation**: Handle large batch sizes on limited hardware

### References

**DINO-WM Original Paper**:
```bibtex
@misc{zhou2024dinowmworldmodelspretrained,
  title={DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning}, 
  author={Gaoyue Zhou and Hengkai Pan and Yann LeCun and Lerrel Pinto},
  year={2024},
  eprint={2411.04983},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2411.04983}
}
```

**V-JEPA 2 Paper**:
```bibtex
@misc{vjepa2_2024,
  title={V-JEPA 2: Scaling Self-Supervised Video Pretraining},
  author={Meta AI Research},
  year={2024},
  url={https://arxiv.org/html/2506.09985v1}
}
```

### Contributing

This is a research experiment repository. Contributions are welcome for:
- Additional encoder implementations
- New evaluation metrics
- Environment configurations
- Performance optimizations

### License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This is a research experiment comparing different visual foundation models for world modeling. Results and findings will be documented in a research blog post upon completion of the experiments.