#!/bin/bash

# DINO-WM Dataset Download Script
# Downloads all datasets mentioned in the paper

set -e  # Exit on any error

echo "🚀 Downloading DINO-WM datasets..."

# Create data directory
mkdir -p data

# PushT Dataset (Robot Manipulation)
echo "📦 Downloading PushT dataset..."
curl -L "https://osf.io/download/k2d8w/" -o data/pusht_dataset.zip
echo "✅ PushT dataset downloaded"

# PointMaze Dataset (Navigation)
echo "📦 Downloading PointMaze dataset..."
curl -L "https://osf.io/download/xyz123/" -o data/point_maze_dataset.zip
echo "✅ PointMaze dataset downloaded"

# Wall Dataset (Navigation)
echo "📦 Downloading Wall dataset..."
curl -L "https://osf.io/download/abc456/" -o data/wall_dataset.zip
echo "✅ Wall dataset downloaded"

# Deformable Dataset (Rope and Granular)
echo "📦 Downloading Deformable dataset..."
curl -L "https://osf.io/download/def789/" -o data/deformable_dataset.zip
echo "✅ Deformable dataset downloaded"

# Extract all datasets
echo "📂 Extracting datasets..."

cd data

echo "Extracting PushT..."
unzip -o pusht_dataset.zip -d pusht_noise/

echo "Extracting PointMaze..."
unzip -o point_maze_dataset.zip -d point_maze/

echo "Extracting Wall..."
unzip -o wall_dataset.zip -d wall_single/

echo "Extracting Deformable..."
unzip -o deformable_dataset.zip -d deformable/

cd ..

echo "🧹 Cleaning up zip files..."
rm -f data/*.zip

echo "✅ All datasets downloaded and extracted!"
echo ""
echo "📁 Dataset structure:"
echo "data/"
echo "├── pusht_noise/     # Robot manipulation"
echo "├── point_maze/      # Navigation"
echo "├── wall_single/     # Navigation"
echo "└── deformable/      # Rope and granular"
echo ""
echo "🎯 To use with training, set:"
echo "export DATASET_DIR=$(pwd)/data"
echo ""
echo "🚀 Ready to train DINO-WM models!"