# Fashion MNIST Clothes Classification Neural Network

A fully connected neural network for classifying clothing items using the Fashion MNIST dataset, built with TensorFlow/Keras.

## Overview

This project implements an end-to-end deep learning pipeline that:
- Loads and preprocesses the Fashion MNIST dataset (70,000 grayscale images)
- Trains a Dense neural network to classify 10 types of clothing
- Evaluates performance with accuracy metrics, learning curves, and confusion matrix
- Predicts clothing type from new external images

## Dataset

**Fashion MNIST** (from Keras) - a drop-in replacement for classic MNIST:

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

- **Training**: 60,000 images (28x28 pixels, grayscale)
- **Test**: 10,000 images
- Auto-downloaded on first run

## Neural Network Architecture

```
Input (784) --> Dense(512, ReLU) --> Dropout(0.2) --> Dense(256, ReLU) --> Dropout(0.2) --> Dense(10, Softmax)
```

| Layer | Neurons | Activation | Purpose |
|-------|---------|------------|---------|
| Hidden 1 | 512 | ReLU | Learn low-level patterns (edges, textures) |
| Dropout 1 | - | - | Regularization (drop 20% of neurons) |
| Hidden 2 | 256 | ReLU | Combine into higher-level features |
| Dropout 2 | - | - | Additional overfitting prevention |
| Output | 10 | Softmax | Probability per clothing class |

**Total parameters**: 535,818

**Why Dense (not CNN)?** Dense layers are sufficient for small 28x28 centered images, achieving 85-90% accuracy while being simpler to understand and implement.

## Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Loss | Categorical Crossentropy | Standard for multi-class + one-hot labels |
| Optimizer | Adam (lr=0.001) | Adaptive learning rates, fast convergence |
| Batch Size | 128 | Balance between speed and generalization |
| Epochs | 20 | Sufficient for convergence on Fashion MNIST |
| Validation | 20% of training data | Monitor overfitting during training |

**Target accuracy**: 85%+ on test set

## Experiment Results

### Architecture Comparison

We compared 4 architectures using the same training setup (ReLU, Dropout 0.2, Adam, 20 epochs):

| Architecture | Layers | Parameters | Test Accuracy |
|-------------|--------|------------|---------------|
| **Baseline** | 512 → 256 → 10 | 535,818 | **88.27%** |
| Narrow | 128 → 64 → 10 | ~109K | Lower |
| Deep | 512 → 256 → 128 → 64 → 10 | ~553K | Lower |
| Wide | 1024 → 512 → 256 → 10 | ~1.3M | Lower |

### Loss Function Comparison

We tested 5 regularization strategies on the baseline architecture:

| Strategy | Loss Formula | Test Accuracy | Weight Norm |
|----------|-------------|---------------|-------------|
| No Regularization | `CE(y, ŷ)` | Baseline | Largest |
| L2 (λ=1e-4) | `CE + λ∑(w²)` | Improved | Smaller |
| L1 (λ=1e-5) | `CE + λ∑\|w\|` | Improved | Sparse |
| Elastic Net (L1+L2) | `CE + λ₁∑\|w\| + λ₂∑(w²)` | Improved | Smaller |
| **L2 + Label Smoothing** | `smooth_CE + λ∑(w²)` | **88.27%** | **601.48** |

### Conclusion

The combination of **L2 regularization + Label Smoothing** loss function with the **Baseline (512→256)** architecture achieved the best results at **88.27% test accuracy**. Adding weight regularization to the loss function forces the model to learn simpler, more generalizable patterns, while label smoothing prevents overconfident predictions.

## Project Structure

```
L-37-ClothesNeuralNetworkMIST/
├── fashion_mnist_classifier.ipynb      # Main notebook (Google Colab ready)
├── fashion_mnist_classifier.py         # Main standalone Python script
├── architecture_experiments.ipynb      # Architecture & loss function experiments notebook
├── architecture_experiments.py         # Architecture & loss function experiments script
├── cloth_examples/                     # Sample clothing images for testing
├── results/                            # Training results and screenshots
├── PRD.md                              # Product Requirements Document
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (recommended, CPU fallback available)

### Dependencies

```
tensorflow[and-cuda]
numpy
matplotlib
scikit-learn
pillow
seaborn
jupyter
```

## How to Run

### Option A: Google Colab (Recommended for GPU)

1. Upload `fashion_mnist_classifier.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime to GPU: **Runtime** > **Change runtime type** > **GPU**
3. Run all cells in order

### Option B: Local Python Script

```bash
# Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the script
python fashion_mnist_classifier.py
```

Close each plot window to advance to the next step.

### Option C: Local Jupyter Notebook

```bash
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook fashion_mnist_classifier.ipynb
```

## Features

1. **GPU Detection** - Auto-detects NVIDIA GPU, falls back to CPU with warning
2. **Data Preview** - Displays images 6-9 in a 2x2 grid with labels
3. **Preprocessing** - Normalization (0-1), flattening (784), one-hot encoding
4. **Training Visualization** - Loss and accuracy curves over epochs
5. **Confusion Matrix** - 10x10 heatmap with per-class accuracy and top misclassifications
6. **Unknown Image Prediction** - Load any clothing image, get prediction + confidence chart
7. **Model Saving** - Export trained model as `.keras` for reuse

## Usage: Predict Your Own Image

After training (or loading a saved model):

```python
# In the script/notebook
predict_clothing_image('path/to/your/clothing_image.jpg', model, class_names)

# Or load a saved model
import tensorflow as tf
model = tf.keras.models.load_model('fashion_mnist_model.keras')
predict_clothing_image('sneaker.jpg', model, class_names)
```

## GPU Setup (Local)

For local GPU training, ensure you have:
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version compatible with your TensorFlow)
- cuDNN library

TensorFlow 2.x with `pip install tensorflow[and-cuda]` handles most CUDA dependencies automatically.

## Course Info

- **Course**: AI Development Course
- **Lesson**: 37
- **Topic**: Neural Networks for Image Classification
- **Built with**: Claude Code (Claude Opus 4.6)
