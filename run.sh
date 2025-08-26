#!/bin/bash

# DINO World Model Planning Script
# This script sets up the environment and runs planning for the point_maze task

echo "Setting up DINO World Model environment..."

# Activate conda environment
echo "Activating conda environment: dino_wm"
source /home/sra-vjti/miniconda3/bin/activate dino_wm

# Set up MuJoCo library paths
echo "Setting up MuJoCo library paths..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sra-vjti/.mujoco/mujoco210/bin:/usr/lib/nvidia

# Set up dataset directory
echo "Setting up dataset directory..."
export DATASET_DIR=/home/sra-vjti/ksagar/dino_wm/data

# Print environment info
echo "Current working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "DATASET_DIR: $DATASET_DIR"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check if required files exist
echo "Checking required files..."
if [ -f "checkpoints/outputs/point_maze/hydra.yaml" ]; then
    echo "✓ Model checkpoint configuration found"
else
    echo "✗ Model checkpoint configuration not found at checkpoints/outputs/point_maze/hydra.yaml"
    exit 1
fi

if [ -f "checkpoints/outputs/point_maze/checkpoints/model_latest.pth" ]; then
    echo "✓ Model weights found"
else
    echo "✗ Model weights not found at checkpoints/outputs/point_maze/checkpoints/model_latest.pth"
    exit 1
fi

if [ -d "data/point_maze" ]; then
    echo "✓ Dataset directory found"
else
    echo "✗ Dataset directory not found at data/point_maze"
    exit 1
fi

# Run the planning script
echo ""
echo "Starting planning run for point_maze..."
echo "Command: python plan.py --config-name plan_point_maze.yaml model_name=point_maze ckpt_base_path=/home/sra-vjti/ksagar/dino_wm/checkpoints"
echo ""

python plan.py --config-name plan_point_maze.yaml \
    model_name=point_maze \
    ckpt_base_path=/home/sra-vjti/ksagar/dino_wm/checkpoints

# Check the exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Planning completed successfully!"
    echo "Results saved to plan_outputs/ directory"
else
    echo ""
    echo "✗ Planning failed with exit code: $EXIT_CODE"
    echo "Common issues:"
    echo "  - GPU memory error: Try reducing planning samples"
    echo "  - Environment setup: Check MuJoCo and conda environment"
    echo "  - File paths: Verify all required files exist"
fi

exit $EXIT_CODE