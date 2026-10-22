# DAC-301: Deep Learning Assignment — Glasses Classification

**Author:** Ankur Dahiya  
**Course:** DAC-301 (Deep Learning)  
**Kaggle Public Leaderboard Score:** **0.99872** (Rank 1)  

---

## Overview
This repository contains the full from-scratch PyTorch implementation, exploratory data analysis, hyperparameter tuning experiments, LaTeX report, and extended training pipelines for the DAC-301 Glasses Classification competition.

### Core Architecture
- **Backbone:** 5-Stage VGG-style Deep Convolutional Neural Network (trained 100% from scratch with random initialization; no transfer learning).
- **Head:** Dynamically configurable Multi-Layer Perceptron (MLP) supporting arbitrary `num_hidden`, `sizes`, and `activation` functions.
- **Input Resolution:** $224 \times 224$ RGB to preserve fine spatial details of thin wireframes and rimless glasses.
- **Regularization & Optimization:** AdamW ($\text{wd}=10^{-4}$), Cosine Annealing with Warm Restarts, Label Smoothing ($\epsilon=0.05$), Dropout ($p=0.4$), and Test-Time Augmentation (TTA).

---

## Repository Structure
```
├── DAC_301_Assignment.ipynb         # Fully executed notebook with inline EDA and metrics
├── train_50ep.py                    # Extended 50-epoch script optimized for high-end GPUs
├── DAC301_Assignment_Report.pdf     # Publication-grade assignment report
├── DAC301_Report.tex                # Complete LaTeX report source
├── submission.csv                   # Winning Kaggle predictions (Score: 0.99872)
├── arch_diagram.png                 # Architecture schematic
├── eda_*.png                        # Exploratory data analysis plots
└── exp_*.png                        # Hyperparameter ablation curves
```

---

## Running the 50-Epoch Extended Training on Blackwell / High-VRAM GPUs

```bash
# Clone the repository
git clone https://github.com/ankur777jinn/DAC-301-Glasses-Classification.git
cd DAC-301-Glasses-Classification

# Run 50 epochs training with batch size 128 (adjust paths to where dataset is mounted)
python train_50ep.py \
    --train_csv /path/to/train.csv \
    --train_dir /path/to/train_images \
    --test_csv /path/to/test_dataset.csv \
    --test_dir /path/to/test_images \
    --sample_sub /path/to/sample_submission_dataset.csv \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --num_workers 8 \
    --output_csv extended_50ep_submission.csv
```
