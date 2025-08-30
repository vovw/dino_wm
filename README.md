# **DINO-WM**: World Models on Pre-trained Visual Features enable Zero-shot Planning
[[Paper]](https://arxiv.org/abs/2411.04983) [[Data]](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) [[Project Website]](https://dino-wm.github.io/) 

[Gaoyue Zhou](https://gaoyuezhou.github.io/), [Hengkai Pan](https://hengkaipan.github.io/), [Yann LeCun](https://yann.lecun.com/) and [Lerrel Pinto](https://www.lerrelpinto.com/), New York University, Meta AI

![teaser_figure](assets/intro.png)

**This is a simplified version of DINO-WM with Hydra configuration system removed, keeping only the essential code for PushT and granular environments.**

## Quick Start

### Installation

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

### Install MuJoCo (Required)

Create the `.mujoco` directory and download MuJoCo210:

```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz
```

Add to your shell configuration (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# MuJoCo Path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# NVIDIA Library Path (if using NVIDIA GPUs)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Install PyFlex (Optional, for granular environments)

If you plan to use the granular environment, install PyFleX:

```bash
# Install pybind11
uv add "pybind11[global]"

# Pull Docker image and compile PyFleX
sudo docker pull xingyu/softgym

# Run the installation script
bash install_pyflex.sh
```

### Datasets

Download datasets from [here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28).

For the deformable dataset, combine parts and unzip:
```bash
zip -s- deformable.zip -O deformable_full.zip
unzip deformable_full.zip
```

Set the dataset directory:
```bash
export DATASET_DIR=/path/to/your/datasets
```

Expected structure:
```
data
├── deformable
│   └── granular
└── pusht_noise
```

## Usage

### Training

**For PushT environment:**
```bash
python train_pusht.py --output-dir ./outputs
```

**For granular environment:**
```bash
python train_granular.py --output-dir ./outputs
```

**Advanced training options:**
```bash
python train.py --env pusht --output-dir ./outputs
python train.py --env granular --output-dir ./outputs
```

### Planning

**For PushT environment:**
```bash
python plan_pusht.py --model-path /path/to/trained/model --output-dir ./plan_outputs
```

**For granular environment:**
```bash
python plan_granular.py --model-path /path/to/trained/model --output-dir ./plan_outputs
```

**Advanced planning options:**
```bash
python plan.py --env pusht --model-path /path/to/model --model-epoch latest --output-dir ./plan_outputs
python plan.py --env granular --model-path /path/to/model --model-epoch latest --output-dir ./plan_outputs
```

## Configuration

All configurations are handled via Python classes in `configs.py`. You can modify default settings by editing this file or creating your own configuration variants.

## Key Changes from Original

- ✅ **Removed Hydra configuration system**
- ✅ **Simplified to support only PushT and granular environments**
- ✅ **Direct Python configuration using dataclasses**
- ✅ **Command-line argument parsing instead of Hydra**
- ✅ **Replaced conda with uv for faster dependency management**
- ✅ **Simplified file structure**
- ✅ **Removed SLURM/distributed training complexity (accelerate still supported)**

## Citation

```
@misc{zhou2024dinowmworldmodelspretrained,
      title={DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning}, 
      author={Gaoyue Zhou and Hengkai Pan and Yann LeCun and Lerrel Pinto},
      year={2024},
      eprint={2411.04983},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.04983}, 
}
```
