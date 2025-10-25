# Retinal Disease Classification with MSFM + ViT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project for multi-label retinal disease classification using Multi-Scale Feature Map (MSFM) architecture combined with Vision Transformer (ViT) encoder.

## 🎯 Overview

This project implements a high-performance deep learning model for multi-label classification of 20 retinal diseases from fundus images. The architecture achieves **state-of-the-art results** in macro F1 Score that surpass existing literature through innovative combination of multi-scale feature extraction and transformer-based attention mechanisms.

The architecture combines:

- **MSFM (Multi-Scale Feature Map)**: Extracts multi-scale features from retinal images using DenseNet-201 backbone
- **Vision Transformer (ViT)**: Processes the extracted features through self-attention mechanisms
- **Focal Loss**: Handles class imbalance in multi-label classification
- **Per-class Threshold Optimization**: Improves F1 scores by optimizing classification thresholds per disease

### Disease Classes

The model can detect and classify 20 retinal conditions:
- DR (Diabetic Retinopathy)
- NORMAL
- MH (Macular Hole)
- ODC (Optic Disc Cupping)
- TSLN (Tessellation)
- ARMD (Age-Related Macular Degeneration)
- DN (Diabetic Neuropathy)
- MYA (Myopia)
- BRVO (Branch Retinal Vein Occlusion)
- ODP (Optic Disc Pallor)
- CRVO (Central Retinal Vein Occlusion)
- CNV (Choroidal Neovascularization)
- RS (Retinitis)
- ODE (Optic Disc Edema)
- LS (Laser Scars)
- CSR (Central Serous Retinopathy)
- HTR (Hypertensive Retinopathy)
- ASR (Asteroid Hyalosis)
- CRS (Chorioretinal Scars)
- OTHER

## 📊 Performance Metrics

The model is evaluated using multiple metrics:
- **ML F1**: Macro F1 score across all classes
- **ML mAP**: Mean Average Precision
- **ML AUC**: Mean Area Under ROC Curve
- **Binary AUC**: AUC specifically for NORMAL class
- **Model Score**: Combined metric (ML_Score + Bin_AUC) / 2

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Clone the Repository
```bash
git clone https://github.com/syedtashif/retinal-disease-classification.git
cd retinal-disease-classification
```

### Install Dependencies
```bash
pip install -r requirements.txt
```


## 📁 Dataset Structure

Organize your dataset as follows:
```
data/
├── train_data_modified.csv
├── test_data_modified.csv
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**CSV Format:**
- First column: Image ID
- Remaining 20 columns: Binary labels for each disease class

## 🔧 Configuration

Edit `config/config.yaml` to customize training parameters:

```yaml
# Model Configuration
model:
  num_classes: 20
  embed_dim: 512
  num_vit_layers: 4
  num_heads: 8
  mlp_dim: 1024

# Training Configuration
training:
  img_size: 384
  batch_size: 16
  epochs: 20
  learning_rate: 0.0001
  patience: 10

# Data Paths
data:
  train_csv: 'path/to/train_data_modified.csv'
  test_csv: 'path/to/test_data_modified.csv'
  images_dir: 'path/to/images'
```

## 🏋️ Training

### Basic Training
```bash
python src/train.py
```

### With Custom Config
```bash
python src/train.py --config config/custom_config.yaml
```

### Training Arguments
- `--config`: Path to configuration file
- `--batch-size`: Training batch size (default: 16)
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device to use ('cuda' or 'cpu')


## 🏗️ Model Architecture

### MSFM (Multi-Scale Feature Map)
- Backbone: DenseNet-201 (pretrained)
- Multi-scale feature extraction from low and high-level features
- Channel Attention Mechanism (CAM)
- Multi-head attention fusion

### ViT (Vision Transformer)
- 4 transformer encoder blocks
- 8 attention heads
- 512 embedding dimension
- 1024 MLP hidden dimension

### Complete Pipeline
```
Input Image (384x384x3)
    ↓
DenseNet-201 Feature Extraction
    ↓
Multi-Scale Feature Fusion (MSFM)
    ↓
Vision Transformer Encoder (4 layers)
    ↓
Global Average Pooling
    ↓
Classification Head (20 classes)
    ↓
Output Probabilities
```

## 📂 Project Structure

```
retinal-disease-classification/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
├── config/
│   └── config.yaml                   # Training configuration
├── src/
   ├── data/
   │   └── dataset.py                # Dataset and DataLoader
   ├── models/
   │   ├── msfm.py                   # MSFM module
   │   ├── vit.py                    # Vision Transformer
   │   └── complete_model.py         # Complete MSFM + ViT model
   ├── losses/
   │   └── focal_loss.py             # Focal Loss implementation
   │   └── polyloss.py  
   ├── utils/
   │   ├── metrics.py                # Evaluation metrics
   │   └── transforms.py             # Data augmentation
   └── train.py                      # Main training script


## 🔬 Key Features

1. **Multi-Scale Feature Extraction**: Combines low-level and high-level features for better representation
2. **Attention Mechanisms**: Both channel attention and multi-head self-attention
3. **Focal Loss**: Addresses class imbalance effectively
4. **Per-Class Threshold Optimization**: Improves classification performance
5. **Early Stopping**: Prevents overfitting with patience-based stopping
6. **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
7. **Comprehensive Metrics**: Multiple evaluation metrics for thorough assessment

## 📊 Results

| Metric | Score |
|--------|-------|
| ML F1 | 68.2 |
| ML mAP | 66.6 |
| ML AUC | 94.8 |
| Binary AUC | 96.3 |
| Binary F1 | 81.41 |
| Model Score | 87.9 |



## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{retinal-disease-classification,
  author = {Syed Mohd Tashif},
  title = {Retinal Disease Classification with MSFM + ViT},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/syedtashif/retinal-disease-classification}
}
```

## 🙏 Acknowledgments

Based on:

- **DenseNet-201** — Huang et al., [Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)
- **Vision Transformer (ViT)** — Dosovitskiy et al., [An Image is Worth 16x16 Words (ICLR 2021)](https://arxiv.org/abs/2010.11929)
- **Focal Loss** — Lin et al., [Focal Loss for Dense Object Detection (ICCV 2017)](https://arxiv.org/abs/1708.02002)
- **PolyLoss** — Leng et al., [PolyLoss: A Polynomial Expansion Perspective of Classification Loss (ICLR 2022)](https://arxiv.org/abs/2204.12511)
- MSFM: Inspired by multi-scale feature extraction in medical imaging literature
- PyTorch, torchvision, scikit-learn

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


⭐ Star this repository if you find it helpful!
