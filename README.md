# Transfer Learning for Image Classification 🧠

A deep learning project that benchmarks four pre-trained CNN architectures — **ResNet50**, **ResNet101**, **EfficientNetB0**, and **VGG16** — for classifying images across six natural scene categories using transfer learning.

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)

---

## 🗂 Dataset

Six-class natural scene classification dataset:

| Class | Label |
|-------|-------|
| 🏙 Buildings | `buildings` |
| 🌲 Forest | `forest` |
| 🏔 Glacier | `glacier` |
| ⛰ Mountain | `mountain` |
| 🌊 Sea | `sea` |
| 🛣 Street | `street` |

- **Training set:** 14,034 images
- **Validation split:** 20% of training data (stratified)
- **Test set:** 3,000 images
- **Image size:** 150 × 150 × 3

---

## 🏗 Model Architecture

Each model follows the same transfer learning structure:
```
Input (150×150×3)
    ↓
Data Augmentation (rotation, translation, zoom, flip, contrast)
    ↓
Pre-trained Backbone (frozen weights, ImageNet)
    ↓
Flatten
    ↓
Dense(512, activation='relu') + L2 Regularization
    ↓
BatchNormalization
    ↓
Dropout(0.2)
    ↓
Dense(6, activation='softmax') + L2 Regularization
```

**Optimizer:** Adam (lr=0.0001) | **Loss:** Categorical Cross-Entropy | **Early Stopping:** patience=10, starts from epoch 50

---

## 📊 Results

| Model | Test Loss | Test Accuracy | Precision | Recall | F1 Score | AUC |
|-------|-----------|---------------|-----------|--------|----------|-----|
| **EfficientNetB0** | **0.4397** | **0.9023** | **0.9020** | **0.9023** | **0.9020** | **0.9897** |
| ResNet50 | 0.5403 | 0.8803 | 0.8797 | 0.8803 | 0.8796 | 0.9842 |
| ResNet101 | 0.5677 | 0.8637 | 0.8632 | 0.8637 | 0.8615 | 0.9822 |
| VGG16 | 0.7023 | 0.8643 | 0.8635 | 0.8643 | 0.8637 | 0.9825 |

> ✅ **EfficientNetB0 outperforms all other models** with the lowest test loss and highest test accuracy, while also being the most parameter-efficient backbone (only 4.7M total params vs 43.7M for ResNet101).

---

## 🔬 Per-Class Performance (EfficientNetB0)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Buildings | 0.90 | 0.91 | 0.90 |
| Forest | 0.99 | 0.99 | 0.99 |
| Glacier | 0.84 | 0.83 | 0.83 |
| Mountain | 0.87 | 0.83 | 0.85 |
| Sea | 0.91 | 0.95 | 0.93 |
| Street | 0.92 | 0.91 | 0.91 |

---

## ⚙️ Training Details

| Setting | Value |
|---------|-------|
| Batch Size | 32 |
| Max Epochs | 100 |
| Early Stopping Patience | 10 (starts epoch 50) |
| Validation Split | 20% (stratified) |
| Optimizer | Adam (lr=0.0001) |
| Loss | Categorical Cross-Entropy |
| Regularization | L2 (λ=0.001) + Dropout (20%) |
| Backbone Weights | ImageNet (frozen) |

### Data Augmentation
- Random Rotation (±10%)
- Random Translation (±10%)
- Random Zoom (±10%)
- Random Vertical Flip
- Random Contrast (±10%)

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow scikit-learn opencv-python pandas numpy matplotlib
```

### Project Structure
```
project/
├── data/
│   ├── seg_train/          # Training images (per-class subdirectories)
│   └── seg_test/           # Test images (per-class subdirectories)
└── PM_DSML_PROJECT.ipynb   # Main notebook
```

### Run

Open and execute the notebook sequentially:
```bash
jupyter notebook PM_DSML_PROJECT.ipynb
```

---

## 📦 Model Parameter Counts

| Model | Total Params | Trainable Params | Non-Trainable |
|-------|-------------|------------------|---------------|
| ResNet50 | 24.6M (94 MB) | 1.05M (4 MB) | 23.6M (90 MB) |
| ResNet101 | 43.7M (166 MB) | 1.05M (4 MB) | 42.7M (162 MB) |
| EfficientNetB0 | 4.7M (18 MB) | 0.66M (2.5 MB) | 4.05M (15 MB) |
| VGG16 | 15M (57 MB) | 0.27M (1 MB) | 14.7M (56 MB) |

---

## 📚 References

- [Keras API Documentation](https://keras.io/api/)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [scikit-learn Classification Report](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.classification_report.html)
- [Transfer Learning for CNN](https://medium.com/@silvershine1st/transfer-learning-for-cnn-7eed1d8a5305)

---
