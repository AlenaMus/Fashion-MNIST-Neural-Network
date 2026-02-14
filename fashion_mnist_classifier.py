"""
=============================================================================
Fashion MNIST Clothes Classification Neural Network
=============================================================================
A fully connected neural network that classifies clothing items from the
Fashion MNIST dataset and allows users to predict on their own images.

Architecture: Input(784) -> Dense(512,ReLU) -> Dropout(0.2) -> Dense(256,ReLU) -> Dropout(0.2) -> Dense(10,Softmax)

Usage:
    python fashion_mnist_classifier.py
=============================================================================
"""

# === IMPORTS ===
import numpy as np
import random
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image

# === REPRODUCIBILITY ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === GPU CONFIGURATION ===
print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nGPU Available: {len(gpus)} device(s) detected")
    for gpu in gpus:
        print(f"  -> {gpu.name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("  Memory growth enabled for efficient GPU usage")
else:
    print("\nWARNING: No GPU detected! Training will run on CPU (slower).")
print("=" * 60)

# === CLASS NAMES ===
class_names = [
    'T-shirt/top',  # 0
    'Trouser',       # 1
    'Pullover',      # 2
    'Dress',         # 3
    'Coat',          # 4
    'Sandal',        # 5
    'Shirt',         # 6
    'Sneaker',       # 7
    'Bag',           # 8
    'Ankle boot'     # 9
]

# === LOAD DATASET ===
print("\nLoading Fashion MNIST dataset...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = fashion_mnist.load_data()
print(f"  Training: {X_train_raw.shape[0]} images | Test: {X_test_raw.shape[0]} images")


# === PREVIEW FUNCTION ===
def preview_dataset_images(images, labels, indices=[6, 7, 8, 9]):
    """Display a 2x2 grid of sample images from the dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Fashion MNIST - Sample Images', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx], cmap='gray')
        axes[i].set_title(f'Index: {idx} | {class_names[labels[idx]]}', fontsize=11)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


preview_dataset_images(X_train_raw, y_train_raw)

# === PREPROCESSING ===
print("\nPreprocessing data...")
# Step 1: Normalize pixel values [0,255] -> [0,1]
X_train = X_train_raw.astype('float32') / 255.0
X_test = X_test_raw.astype('float32') / 255.0

# Step 2: Flatten 28x28 -> 784
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Step 3: One-hot encode labels
y_train = to_categorical(y_train_raw, num_classes=10)
y_test = to_categorical(y_test_raw, num_classes=10)
print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape} | y_test: {y_test.shape}")

# === BUILD MODEL ===
print("\nBuilding neural network...")
model = Sequential([
    Dense(512, activation='relu', input_shape=(784,), name='hidden_layer_1'),
    Dropout(0.2, name='dropout_1'),
    Dense(256, activation='relu', name='hidden_layer_2'),
    Dropout(0.2, name='dropout_2'),
    Dense(10, activation='softmax', name='output_layer')
], name='fashion_mnist_classifier')

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# === TRAIN MODEL ===
print("\n" + "=" * 60)
print("TRAINING MODEL (20 epochs, batch_size=128)")
print("=" * 60)
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=20,
    validation_split=0.2,
    verbose=1
)

# === TRAINING CURVES ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Training History', fontsize=16, fontweight='bold')

axes[0].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss',
             color='red', linewidth=2, linestyle='--')
axes[0].set_title('Loss Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Training Accuracy',
             color='blue', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy',
             color='red', linewidth=2, linestyle='--')
axes[1].set_title('Accuracy Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === EVALUATE ON TEST SET ===
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
if test_accuracy >= 0.85:
    print("  Target accuracy (>85%) ACHIEVED!")
print("=" * 60)

# === CONFUSION MATRIX ===
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='gray', square=True)
plt.title('Confusion Matrix - Fashion MNIST Classification',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('True Label', fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print per-class accuracy
print("\nPer-Class Accuracy:")
for i, name in enumerate(class_names):
    class_acc = cm[i, i] / cm[i].sum()
    print(f"  {name:15s}: {class_acc:.2%}  ({cm[i, i]}/{cm[i].sum()})")

# === SAVE MODEL ===
model.save('fashion_mnist_model.keras')
print("\nModel saved to 'fashion_mnist_model.keras'")


# =============================================================================
# PREDICT ON USER-UPLOADED IMAGES
# =============================================================================

def predict_clothing_image(image_path, model, class_names):
    """
    Load an external clothing image and predict its class.

    Preprocesses the image to match Fashion MNIST format:
    grayscale, 28x28, light-on-dark colors, normalized, flattened.
    Auto-inverts colors for real-world photos (dark-on-light).

    Args:
        image_path (str): Path to the image file (PNG, JPG, JPEG, BMP, WEBP)
        model: Trained Keras model
        class_names (list): List of 10 class name strings

    Returns:
        tuple: (predicted_class_name, confidence_score) or (None, 0.0) on error
    """
    try:
        # Step 1: Load image
        img = Image.open(image_path)
        print(f"\nImage loaded: {image_path}")
        print(f"  Original size: {img.size}, Mode: {img.mode}")

        img_original = img.copy()

        # Step 2: Convert to grayscale
        img_gray = img.convert('L')

        # Step 3: Resize to 28x28
        img_resized = img_gray.resize((28, 28))

        # Step 4: Convert to numpy array
        img_array = np.array(img_resized)

        # Step 5: Auto-invert colors if needed
        # Fashion MNIST = black background, light clothing
        # Real photos = light background, dark clothing
        mean_pixel = img_array.mean()
        if mean_pixel > 127:
            img_array = 255 - img_array
            print(f"  Color inverted (mean pixel {mean_pixel:.0f} > 127"
                  f" -> light background detected)")
        else:
            print(f"  No inversion needed (mean pixel {mean_pixel:.0f}"
                  f" <= 127 -> dark background)")

        # Step 6: Normalize [0,255] -> [0,1]
        img_normalized = img_array.astype('float32') / 255.0

        # Step 7: Flatten and reshape for model
        img_flat = img_normalized.reshape(1, 784)

        # Step 8: Predict
        prediction = model.predict(img_flat, verbose=0)

        # Step 9: Extract results
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        predicted_name = class_names[predicted_class_idx]

        # Display results
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Left: original image
        if img_original.mode == 'L':
            axes[0].imshow(np.array(img_original), cmap='gray')
        else:
            axes[0].imshow(np.array(img_original))
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        # Center: preprocessed 28x28 (what the model sees)
        axes[1].imshow(img_array, cmap='gray')
        axes[1].set_title('Preprocessed 28x28\n(Model Input)', fontsize=12)
        axes[1].axis('off')

        # Right: confidence bar chart
        colors = ['green' if i == predicted_class_idx else 'steelblue'
                  for i in range(10)]
        axes[2].barh(class_names, prediction[0], color=colors)
        axes[2].set_title('Prediction Confidence', fontsize=12)
        axes[2].set_xlim([0, 1])
        axes[2].set_xlabel('Probability')

        fig.suptitle(f'Prediction: {predicted_name} ({confidence:.1%} confidence)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print(f"\n{'='*40}")
        print(f"  Predicted Class: {predicted_name}")
        print(f"  Confidence:      {confidence:.2%}")
        print(f"{'='*40}")

        # Top 3 predictions
        top3_idx = np.argsort(prediction[0])[::-1][:3]
        print("\nTop 3 predictions:")
        for rank, idx in enumerate(top3_idx, 1):
            print(f"  {rank}. {class_names[idx]:15s}: {prediction[0][idx]:.2%}")

        return predicted_name, float(confidence)

    except FileNotFoundError:
        print(f"ERROR: Image file not found at '{image_path}'")
        print("  Please check the file path and try again.")
        return None, 0.0
    except Exception as e:
        print(f"ERROR: Could not process image - {str(e)}")
        return None, 0.0


# =============================================================================
# INTERACTIVE IMAGE PREDICTION LOOP
# =============================================================================
print("\n" + "=" * 60)
print("PREDICT YOUR OWN CLOTHING IMAGES")
print("=" * 60)
print("Enter the full path to a clothing image to classify it.")
print("Supported formats: PNG, JPG, JPEG, BMP, WEBP")
print("Type 'quit' to exit.\n")

while True:
    image_path = input("Image path (or 'quit'): ").strip()

    # Remove surrounding quotes (from drag-and-drop or copy-paste)
    image_path = image_path.strip('"').strip("'")

    if image_path.lower() in ('quit', 'q', 'exit', ''):
        print("Goodbye!")
        break

    predict_clothing_image(image_path, model, class_names)
