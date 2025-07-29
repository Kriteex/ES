# Semantic Segmentation with Super Compact NCA

This project explores semantic segmentation by integrating a super compact Neural Cellular Automata (NCA) into a traditional segmentation pipeline. The goal is to leverage the efficiency, compactness and unique update dynamics of an NCA to perform segmentation.

## Overview
- **Super Compact NCA:** A minimalistic cellular automata model is used to update cell states across an image grid, offering an innovative approach to semantic segmentation.
- **Flexible Data Handling:** A versatile dataset loader processes images and masks with optional augmentations and alpha-channel handling.
- **Live Visualization:** A PyQt5-based GUI visualizer provides real-time monitoring of the NCA’s state evolution, helping to analyze its dynamic behavior during segmentation.
- **Evaluation Metrics:** Implements pixel accuracy, precision, recall, and F1 score.

## Project Structure
- **config.py:** Central configuration for model parameters, training settings, and data management.
- **flexible_dataset.py:** A flexible dataset class for loading and processing images and masks with support for additional channels.
- **model.py:** Contains definitions for multiple segmentation architectures, including UNetTiny, UNetNormal, SegNet, and the super compact Cellular Automata model (CAModel).
- **train.py:** Provides the training pipeline for initializing models, executing training and evaluation loops, and logging performance metrics.
- **utils.py:** Utility functions for data loading, visualization, logging, and saving results.
- **nc.py:** A GUI visualizer built with PyQt5 to interactively observe and analyze the NCA's evolution during segmentation.

## Next Steps / MVP
- **Robust Evaluation:** Expand metrics and visualizations to capture the dynamic behavior and regeneration properties of the NCA during segmentation.
- **Scalability Testing:** Experiment with larger grid sizes and more complex datasets to evaluate performance under varied conditions.
- **Compare with other:** Find most relevant datasets, we can inspire here https://arxiv.org/abs/2008.04965 and here https://arxiv.org/abs/2302.03473.

## How to Run
1. pip install -r requirements.txt
2. python3 train.py
3. tensorboard --logdir==logs