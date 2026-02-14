# Product Requirements Document: Fashion MNIST Clothes Classification Neural Network

## Executive Summary

This project aims to build an intelligent clothes recognition and classification system using deep learning. The system will leverage the Fashion MNIST dataset to train a neural network capable of accurately identifying and categorizing clothing items from images. This tool serves as both an educational demonstration of neural network capabilities and a practical foundation for fashion image classification applications.

**Key Value Proposition:** A GPU-accelerated, fully-connected neural network that achieves high accuracy in classifying 10 different types of clothing items, with comprehensive visualization and prediction capabilities for both known and unknown images.

---

## 1. Product Overview

### 1.1 Vision Statement

Build a robust, interpretable, and performant neural network system that accurately classifies clothing items from grayscale images, providing clear insights into model performance through visualizations and enabling real-world prediction capabilities.

### 1.2 Problem Statement

**User Problem:** Understanding how neural networks process and classify visual data, specifically in the fashion/clothing domain, requires hands-on implementation with clear explanations and visualizations.

**Business Problem:** There is a need for an educational yet practical implementation that demonstrates:
- End-to-end neural network development
- Proper data preprocessing and model training
- Performance evaluation and interpretation
- Real-world prediction capabilities

### 1.3 Target Users

- **Primary:** Data science students and ML practitioners learning computer vision
- **Secondary:** Developers exploring TensorFlow/Keras for image classification
- **Tertiary:** Researchers prototyping fashion recognition systems

### 1.4 Success Criteria

- Model achieves >85% accuracy on test dataset
- All code is well-documented with explanations
- Visualizations clearly communicate model performance
- GPU acceleration successfully implemented
- System can classify new, unseen clothing images

---

## 2. Technical Stack

### 2.1 Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Programming language |
| TensorFlow | 2.x | Deep learning framework |
| Keras | (included in TF 2.x) | High-level neural network API |
| NumPy | Latest | Numerical computing |
| Matplotlib | Latest | Data visualization |
| scikit-learn | Latest | Metrics (confusion matrix) |

### 2.2 Hardware Requirements

- **GPU:** CUDA-compatible GPU (NVIDIA)
- **CUDA Toolkit:** Version compatible with TensorFlow
- **cuDNN:** Version compatible with TensorFlow
- **RAM:** Minimum 8GB recommended
- **Storage:** ~500MB for dataset and model

### 2.3 Dataset

**Fashion MNIST Dataset:**
- **Source:** keras.datasets.fashion_mnist
- **Training samples:** 60,000 grayscale images
- **Test samples:** 10,000 grayscale images
- **Image size:** 28x28 pixels
- **Classes:** 10 clothing categories
  - 0: T-shirt/top
  - 1: Trouser
  - 2: Pullover
  - 3: Dress
  - 4: Coat
  - 5: Sandal
  - 6: Shirt
  - 7: Sneaker
  - 8: Bag
  - 9: Ankle boot

---

## 3. Functional Requirements

### FR-1: Data Loading & Preprocessing

**Priority:** P0 (Critical)

**Description:** Load and prepare the Fashion MNIST dataset for neural network training.

**Acceptance Criteria:**
- Dataset successfully loaded from keras.datasets
- Data split into training and test sets (60,000 train / 10,000 test)
- Images normalized to [0, 1] range by dividing by 255.0
- Labels converted to one-hot encoded vectors (10 classes)
- Training data shape: (60000, 28, 28) → flattened to (60000, 784)
- Test data shape: (10000, 28, 28) → flattened to (10000, 784)
- Code includes comments explaining each preprocessing step

**Technical Details:**
```python
# Normalization formula: pixel_value / 255.0
# One-hot encoding: [3] → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# Flattening: (28, 28) → (784,) for dense layers
```

---

### FR-2: Neural Network Architecture

**Priority:** P0 (Critical)

**Description:** Design and implement a fully-connected (dense) neural network architecture optimized for Fashion MNIST classification.

**Acceptance Criteria:**
- Architecture uses only Dense (fully-connected) layers
- Input layer accepts flattened 784-dimensional vectors
- Hidden layers with appropriate activation functions
- Output layer with 10 neurons (one per class) and softmax activation
- Model compiled successfully
- Architecture documented with rationale

**Recommended Architecture:**
```
Input Layer: 784 neurons (28x28 flattened)
    ↓
Hidden Layer 1: 512 neurons, ReLU activation
    ↓
Dropout: 0.2 (optional, for regularization)
    ↓
Hidden Layer 2: 256 neurons, ReLU activation
    ↓
Dropout: 0.2 (optional)
    ↓
Output Layer: 10 neurons, Softmax activation
```

**Rationale for Architecture Choice:**
- **Dense layers:** Fully-connected architecture suitable for flattened image data
- **ReLU activation:** Addresses vanishing gradient, fast computation, introduces non-linearity
- **Softmax output:** Produces probability distribution over 10 classes
- **Dropout layers:** Prevent overfitting by randomly dropping neurons during training
- **Layer sizes:** Gradually decreasing (512→256→10) creates hierarchical feature learning

**Code Requirements:**
- Detailed comments explaining each layer
- Separate section with architecture explanation
- Model summary printed to console

---

### FR-3: Code Documentation

**Priority:** P0 (Critical)

**Description:** Comprehensive inline documentation explaining every significant code block.

**Acceptance Criteria:**
- Each major section has a comment header (e.g., "# === DATA LOADING ===")
- Every preprocessing step includes explanation comment
- Each layer in the model has a comment explaining its purpose
- Complex operations (normalization, one-hot encoding) have formula/logic comments
- Function docstrings follow standard Python conventions

**Example Standard:**
```python
# === DATA PREPROCESSING ===
# Normalize pixel values from [0, 255] to [0, 1]
# This helps the neural network converge faster by keeping values in similar ranges
X_train = X_train / 255.0
X_test = X_test / 255.0
```

---

### FR-4: Architecture Explanation Document

**Priority:** P1 (High)

**Description:** Detailed written explanation of the neural network architecture and design decisions.

**Acceptance Criteria:**
- Explanation provided as multiline comment or separate markdown section
- Covers why dense/fully-connected layers were chosen
- Explains activation function choices (ReLU, Softmax)
- Justifies number of layers and neurons per layer
- Discusses trade-offs considered

**Required Content:**
1. **Why Dense Layers:** Explain suitability for Fashion MNIST
2. **Activation Functions:** ReLU for hidden layers, Softmax for output
3. **Layer Depth:** Why 2-3 hidden layers are sufficient
4. **Neuron Count:** Reasoning for 512→256→10 progression
5. **Alternatives Considered:** Why not CNN, RNN, etc.

---

### FR-5: GPU Execution

**Priority:** P0 (Critical)

**Description:** Ensure the neural network trains and runs on GPU for accelerated performance.

**Acceptance Criteria:**
- Code checks for GPU availability using TensorFlow
- Prints GPU device information to console
- Model training executes on GPU (verified via TensorFlow logs)
- Graceful fallback to CPU if GPU unavailable (with warning message)
- Performance comparison documented (optional)

**Implementation:**
```python
import tensorflow as tf

# Check GPU availability
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Optional: Force GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Training on GPU: {gpus[0]}")
else:
    print("WARNING: No GPU found, training on CPU (slower)")
```

---

### FR-6: Dataset Preview Function

**Priority:** P1 (High)

**Description:** Visualize sample images from the dataset to understand the data.

**Acceptance Criteria:**
- Function displays images 6, 7, 8, 9 from the training dataset
- Images arranged in 2x2 grid using matplotlib
- Each image shows the corresponding class label
- Grayscale colormap used
- Function callable before training to inspect data

**Visualization Requirements:**
- Figure size: 8x8 inches or similar
- Subplot grid: 2 rows × 2 columns
- Each subplot displays one image with title showing class name
- Images use 'gray' colormap
- Axes turned off for cleaner appearance

**Example Output:**
```
[Image 6: Shirt]  [Image 7: Sneaker]
[Image 8: Bag]    [Image 9: Ankle boot]
```

---

### FR-7: Loss Function

**Priority:** P0 (Critical)

**Description:** Implement and explain the loss function used for training.

**Acceptance Criteria:**
- Loss function chosen: **Categorical Crossentropy**
- Explanation provided in comments or documentation
- Correctly configured in model.compile()
- Rationale clearly stated

**Loss Function Choice: Categorical Crossentropy**

**What it is:**
Categorical Crossentropy measures the difference between predicted probability distributions and true one-hot encoded labels. It penalizes incorrect predictions more heavily when the model is confident but wrong.

**Formula:**
```
Loss = -Σ(y_true * log(y_pred))
```

**Why this choice:**
1. **Multi-class classification:** Fashion MNIST has 10 mutually exclusive classes
2. **One-hot encoded labels:** Labels are in one-hot format [0,0,1,0,0,0,0,0,0,0]
3. **Softmax output:** Pairs perfectly with softmax activation to produce probabilities
4. **Standard practice:** Industry standard for multi-class classification
5. **Gradient behavior:** Provides clear gradients for backpropagation

**Alternative considered:**
- Sparse Categorical Crossentropy: If labels were integers (0-9) instead of one-hot

---

### FR-8: Optimization Function

**Priority:** P0 (Critical)

**Description:** Implement and explain the optimizer used for training.

**Acceptance Criteria:**
- Optimizer chosen: **Adam** (Adaptive Moment Estimation)
- Learning rate specified (default 0.001 or custom)
- Explanation provided in comments or documentation
- Correctly configured in model.compile()

**Optimizer Choice: Adam**

**What it is:**
Adam combines the benefits of two other optimizers (RMSprop and Momentum). It maintains adaptive learning rates for each parameter and uses momentum to accelerate convergence.

**Why this choice:**
1. **Adaptive learning rate:** Automatically adjusts learning rate per parameter
2. **Fast convergence:** Typically requires fewer epochs than SGD
3. **Robust to hyperparameters:** Works well with default settings (lr=0.001)
4. **Handles sparse gradients:** Good for varied gradient patterns in image data
5. **Industry standard:** Most widely used optimizer for deep learning
6. **Less tuning required:** Momentum and learning rate managed automatically

**Alternatives considered:**
- **SGD (Stochastic Gradient Descent):** Simpler but requires more tuning
- **RMSprop:** Good alternative but Adam generally performs better
- **AdaGrad:** Can have learning rate decay issues

**Configuration:**
```python
optimizer = 'adam'  # learning_rate=0.001 (default)
# or
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

---

### FR-9: Training Curve Visualization

**Priority:** P1 (High)

**Description:** Visualize the learning process by plotting loss over epochs.

**Acceptance Criteria:**
- Graph plots training loss vs. epochs
- Optional: Also plot validation loss for comparison
- X-axis: Epoch number (1 to N)
- Y-axis: Loss value
- Graph includes title, axis labels, and legend
- Saved to file and/or displayed
- Helps identify overfitting, underfitting, or convergence

**Visualization Requirements:**
- Line plot with matplotlib
- Title: "Model Loss Over Epochs" or similar
- X-label: "Epoch"
- Y-label: "Loss"
- Legend: "Training Loss", "Validation Loss" (if applicable)
- Grid enabled for readability

**Insights to derive:**
- Decreasing loss indicates learning
- Diverging train/validation loss indicates overfitting
- Plateauing loss indicates convergence
- Increasing loss indicates learning rate issues

---

### FR-10: Confusion Matrix

**Priority:** P1 (High)

**Description:** Generate and visualize a 10×10 confusion matrix showing prediction performance across all classes.

**Acceptance Criteria:**
- Matrix computed on test dataset predictions
- Size: 10×10 (one row/column per class)
- Displays as heatmap with numerical values
- Axis labels show class names (T-shirt, Trouser, etc.)
- Color scale indicates frequency
- Diagonal cells (correct predictions) easily identifiable
- Off-diagonal cells show misclassification patterns

**Implementation Requirements:**
- Use sklearn.metrics.confusion_matrix
- Visualize with matplotlib or seaborn
- Annotations show count in each cell
- Class names on both axes

**Insights to derive:**
- Diagonal values: Correctly classified samples per class
- Off-diagonal values: Common misclassifications
- Example: If cell [3,2] has high value, Dresses are often misclassified as Pullovers
- Identify which classes the model struggles with

**Example interpretation:**
```
Actual → Predicted
        T-sh Trou Pull Dres Coat Sand Shir Snak Bag  Boot
T-shirt [850   0   20   10    5    0   15    0   0    0 ]
Trouser [  0  920   0    0   10    5    0   15  30   20 ]
...
```

---

### FR-11: Unknown Image Prediction (User-Loaded External Images)

**Priority:** P1 (High)

**Description:** Enable the user to load their own clothing images and classify them using the trained model. The system must provide an interactive way for users to select images in both the Jupyter notebook and the standalone Python script.

**Acceptance Criteria:**
- **Interactive image loading** — user is prompted to provide an image path at runtime
- **Jupyter notebook:** Uses `ipywidgets.FileUpload` widget for drag-and-drop / browse file upload, with fallback to `input()` prompt if widget unavailable
- **Python script:** Uses `input()` prompt for the user to type/paste the image file path
- Supports common formats (PNG, JPG, JPEG, BMP, WEBP)
- Preprocesses image to match Fashion MNIST training format:
  - Convert to grayscale
  - Resize to 28×28 pixels
  - **Invert colors if needed** (Fashion MNIST uses light-on-dark; real photos are typically dark-on-light)
  - Normalize to [0, 1]
  - Flatten to (1, 784) shape
- Passes image through trained model
- Outputs predicted class and confidence score
- Displays side-by-side: original image, preprocessed 28×28 image, and confidence bar chart
- Shows top-3 predictions with confidence percentages
- Handles errors gracefully (file not found, wrong format, invalid image)
- **Loop mode:** After each prediction, asks the user if they want to classify another image

**Function Signature:**
```python
def predict_clothing_image(image_path, model, class_names):
    """
    Predict the class of a clothing image.

    Args:
        image_path (str): Path to the image file
        model: Trained Keras model
        class_names (list): List of class name strings

    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
```

**Preprocessing Steps:**
1. Load image using PIL
2. Convert to grayscale (if color)
3. Resize to 28×28 pixels
4. Convert to numpy array
5. **Auto-detect and invert colors** — if the mean pixel value > 127 (light background), invert so clothing appears light on dark background (matching Fashion MNIST format)
6. Normalize: pixel_values / 255.0
7. Flatten: (28, 28) → (784,)
8. Reshape to batch: (784,) → (1, 784)

**Color Inversion Rationale:**
Fashion MNIST images have a **black background with light-colored clothing**. Real-world photos typically have a **light/white background with dark clothing**. Without inversion, the model receives input that looks nothing like the training data, causing poor predictions. The function detects this automatically by checking the mean pixel value.

**Interactive Loading — Jupyter Notebook:**
```python
import ipywidgets as widgets
from IPython.display import display

uploader = widgets.FileUpload(accept='image/*', multiple=False, description='Upload Image')
display(uploader)
# On upload, read the file bytes, save to temp path, and call predict_clothing_image()
```

**Interactive Loading — Python Script:**
```python
while True:
    image_path = input("Enter the path to a clothing image (or 'quit' to exit): ").strip()
    if image_path.lower() in ('quit', 'q', 'exit'):
        break
    predict_clothing_image(image_path, model, class_names)
```

**Output Format:**
```
Predicted Class: Sneaker
Confidence: 94.3%

Top 3 predictions:
  1. Sneaker        : 94.32%
  2. Ankle boot     : 4.21%
  3. Sandal         : 1.02%
```

---

## 4. Non-Functional Requirements

### NFR-1: Performance

**Training Time:**
- Complete training in <5 minutes on GPU (for 10-20 epochs)
- Complete training in <15 minutes on CPU

**Inference Time:**
- Single image prediction: <100ms on GPU
- Batch prediction (10,000 test images): <5 seconds on GPU

**Model Size:**
- Trained model file: <50MB

### NFR-2: Accuracy

**Target Metrics:**
- Test accuracy: >85% (stretch goal: >90%)
- Per-class accuracy: >80% for at least 8 out of 10 classes
- No class with <70% accuracy

### NFR-3: Code Quality

**Maintainability:**
- Modular structure with clearly defined sections
- Consistent naming conventions (snake_case for variables/functions)
- No code duplication
- Each logical section separated by comments

**Documentation:**
- Every major code block has explanatory comments
- Function docstrings for all custom functions
- README with setup and usage instructions

### NFR-4: Reproducibility

**Deterministic Results:**
- Set random seeds for reproducibility
- Document TensorFlow/Keras versions
- Document hardware configuration used

**Seeds to set:**
```python
import numpy as np
import tensorflow as tf
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

### NFR-5: Error Handling

**Robustness:**
- Check GPU availability and notify user
- Handle missing image files gracefully
- Validate image format before prediction
- Provide informative error messages

### NFR-6: Visualization Quality

**Standards:**
- All plots have titles, axis labels, and legends
- Consistent color schemes
- Readable font sizes
- High-resolution output (300 DPI for saved images)
- Professional appearance

---

## 5. Architecture Decisions

### 5.1 Why Dense/Fully-Connected Network?

**Decision:** Use fully-connected (Dense) layers instead of Convolutional Neural Networks (CNNs).

**Rationale:**
1. **Simplicity:** Dense networks are easier to understand and implement
2. **Educational value:** Better for learning fundamental neural network concepts
3. **Fashion MNIST characteristics:** 28×28 images are small and centered
4. **Sufficient performance:** Dense networks can achieve 85-90% accuracy on Fashion MNIST
5. **Computational efficiency:** Fewer parameters than CNNs for this dataset size

**Trade-off accepted:**
- CNNs would achieve slightly higher accuracy (~92-95%)
- CNNs better preserve spatial relationships in images
- Dense networks sufficient for educational and demonstration purposes

### 5.2 Why ReLU Activation?

**Decision:** Use ReLU (Rectified Linear Unit) for hidden layers.

**Rationale:**
1. **Solves vanishing gradient:** Unlike sigmoid/tanh
2. **Computational efficiency:** Simple max(0, x) operation
3. **Sparsity:** Naturally produces sparse representations
4. **Industry standard:** Most widely used activation function
5. **Empirical success:** Proven effective across many tasks

**Formula:** f(x) = max(0, x)

**Alternative considered:** Tanh, Sigmoid (both suffer from vanishing gradients)

### 5.3 Why Softmax Output?

**Decision:** Use Softmax activation for output layer.

**Rationale:**
1. **Probability distribution:** Outputs sum to 1.0
2. **Multi-class classification:** Perfect for 10 mutually exclusive classes
3. **Interpretability:** Each output represents class probability
4. **Pairs with crossentropy:** Mathematical connection simplifies gradients

**Formula:** softmax(x_i) = exp(x_i) / Σ(exp(x_j))

### 5.4 Why Categorical Crossentropy Loss?

**Decision:** Use Categorical Crossentropy as loss function.

**Rationale:**
1. **Matches output format:** Works with softmax and one-hot labels
2. **Theoretical foundation:** Maximum likelihood estimation
3. **Standard practice:** Industry standard for multi-class problems
4. **Gradient properties:** Well-behaved gradients for optimization

### 5.5 Why Adam Optimizer?

**Decision:** Use Adam optimizer with default learning rate.

**Rationale:**
1. **Adaptive learning:** Automatically adjusts per-parameter learning rates
2. **Fast convergence:** Combines momentum and RMSprop benefits
3. **Minimal tuning:** Works well with default hyperparameters
4. **Proven track record:** Most popular optimizer in deep learning

---

## 6. Success Metrics

### 6.1 Model Performance Metrics

**Primary Metrics:**
- **Test Accuracy:** >85% (target: 88%)
- **Training Time:** <5 min on GPU (target: 3 min)
- **Per-Class F1-Score:** >0.80 average

**Secondary Metrics:**
- Loss convergence achieved within 15 epochs
- No signs of severe overfitting (train-val gap <10%)
- Confusion matrix shows balanced performance

### 6.2 Code Quality Metrics

**Documentation:**
- 100% of major code blocks documented
- All functions have docstrings
- Architecture explanation complete

**Functionality:**
- All 11 requirements implemented and working
- No runtime errors on standard execution
- GPU acceleration working (when available)

### 6.3 Deliverables Checklist

- [ ] Data loading and preprocessing code
- [ ] Neural network model definition
- [ ] Model training code with GPU support
- [ ] Dataset preview function (2×2 grid)
- [ ] Loss function with explanation
- [ ] Optimizer with explanation
- [ ] Training curve visualization
- [ ] Confusion matrix visualization
- [ ] Unknown image prediction function
- [ ] Comprehensive code comments
- [ ] Architecture explanation document
- [ ] README with setup instructions

---

## 7. Out of Scope

The following are explicitly **not** included in this project:

- Convolutional Neural Networks (CNNs)
- Transfer learning or pre-trained models
- Data augmentation techniques
- Hyperparameter tuning automation
- Model deployment (web app, API, mobile)
- Real-time video classification
- Multi-label classification
- Custom dataset creation
- Cloud deployment
- A/B testing different architectures

---

## 8. Timeline & Milestones

**Phase 1: Setup & Data (Day 1)**
- Environment setup (TensorFlow, GPU configuration)
- Data loading and preprocessing
- Dataset preview function
- **Deliverable:** Data pipeline working

**Phase 2: Model Development (Day 2)**
- Neural network architecture implementation
- Loss and optimizer configuration
- Training loop implementation
- GPU execution verification
- **Deliverable:** Model trains successfully

**Phase 3: Evaluation & Visualization (Day 3)**
- Training curve plotting
- Model evaluation on test set
- Confusion matrix generation
- Performance analysis
- **Deliverable:** Complete evaluation metrics

**Phase 4: Prediction & Documentation (Day 4)**
- Unknown image prediction function
- Code documentation completion
- Architecture explanation write-up
- README creation
- **Deliverable:** Complete, documented project

---

## 9. Dependencies & Risks

### 9.1 Technical Dependencies

**Critical:**
- Python 3.8+
- TensorFlow 2.x with GPU support
- CUDA Toolkit (for GPU)
- cuDNN libraries

**Standard:**
- NumPy, Matplotlib, scikit-learn
- PIL or OpenCV (for image loading)

### 9.2 Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GPU not available | Medium | Low | CPU fallback implemented |
| TensorFlow version issues | High | Medium | Document exact versions |
| Low model accuracy (<85%) | Medium | Low | Tune hyperparameters, add layers |
| Unknown image preprocessing errors | Low | Medium | Robust error handling |
| Training time too long | Low | Low | Reduce epochs, use GPU |

---

## 10. Appendix

### 10.1 Class Label Mapping

```python
class_names = [
    'T-shirt/top',  # 0
    'Trouser',      # 1
    'Pullover',     # 2
    'Dress',        # 3
    'Coat',         # 4
    'Sandal',       # 5
    'Shirt',        # 6
    'Sneaker',      # 7
    'Bag',          # 8
    'Ankle boot'    # 9
]
```

### 10.2 Expected Data Shapes

```
Training Data:
- X_train_raw: (60000, 28, 28)
- X_train_normalized: (60000, 28, 28)
- X_train_flattened: (60000, 784)
- y_train_raw: (60000,)
- y_train_onehot: (60000, 10)

Test Data:
- X_test_raw: (10000, 28, 28)
- X_test_flattened: (10000, 784)
- y_test_onehot: (10000, 10)
```

### 10.3 Recommended Model Configuration

```python
model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 10.4 Training Configuration

```python
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.2,
    verbose=1
)
```

---

## Document Control

**Version:** 1.0
**Created:** 2026-02-09
**Author:** Product Manager
**Status:** Approved
**Next Review:** Upon project completion

**Change Log:**
- v1.0 (2026-02-09): Initial PRD created

---

## Approval

This PRD has been reviewed and defines the complete scope for the Fashion MNIST Clothes Classification Neural Network project. Implementation should follow this specification to ensure all requirements are met.

**Questions or Clarifications:** Contact the product manager or refer to the reference documentation in this PRD.
