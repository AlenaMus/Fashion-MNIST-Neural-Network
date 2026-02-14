"""
=============================================================================
Architecture Experiments & Custom Loss Functions
=============================================================================
This script compares different neural network architectures AND loss function
strategies on Fashion MNIST. It explores how weight regularization and other
loss improvements affect classification accuracy.

Experiments:
  Part A — Regularization Strategy Comparison (on baseline 512→256 architecture)
    1. No regularization (standard crossentropy)
    2. L2 regularization (weight decay)
    3. L1 regularization (sparsity)
    4. Elastic Net (L1 + L2)
    5. L2 + Label Smoothing

  Part B — Architecture Comparison (using the best loss from Part A)
    1. Baseline:  512 → 256 → 10
    2. Narrow:    128 → 64 → 10
    3. Deep:      512 → 256 → 128 → 64 → 10
    4. Wide:      1024 → 512 → 256 → 10

Usage:
    python architecture_experiments.py
=============================================================================
"""

# === IMPORTS ===
import numpy as np
import random
import os
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt
import seaborn as sns

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
    print("  Memory growth enabled")
else:
    print("\nWARNING: No GPU detected! Training will run on CPU (slower).")
print("=" * 60)

# === CLASS NAMES ===
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# === LOAD & PREPROCESS DATA ===
print("\nLoading Fashion MNIST dataset...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = fashion_mnist.load_data()
print(f"  Training: {X_train_raw.shape[0]} images | Test: {X_test_raw.shape[0]} images")

X_train = X_train_raw.astype('float32') / 255.0
X_test = X_test_raw.astype('float32') / 255.0
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
y_train = to_categorical(y_train_raw, num_classes=10)
y_test = to_categorical(y_test_raw, num_classes=10)
print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}  | y_test:  {y_test.shape}")


# =============================================================================
# CUSTOM LOSS FUNCTION WITH WEIGHT REGULARIZATION
# =============================================================================
#
# Standard loss:
#   Loss = CrossEntropy(y_true, y_pred)
#
# Custom loss with weight penalty:
#   Loss = CrossEntropy(y_true, y_pred) + lambda * Regularization(weights)
#
# The regularization term penalizes large weights, forcing the network to
# learn simpler patterns that generalize better to unseen data.
#
# In Keras, this is implemented via kernel_regularizer on Dense layers.
# The regularization loss is automatically added to the total loss during
# training (visible in model.losses).
# =============================================================================

def build_model(layer_sizes, reg_type=None, reg_lambda=1e-4):
    """
    Build a Sequential model with optional weight regularization.

    The regularization penalty is added to the loss function automatically:
      - L2:    Loss += lambda * sum(w_i^2)       -> keeps weights small
      - L1:    Loss += lambda * sum(|w_i|)        -> encourages zero weights (sparsity)
      - L1_L2: Loss += l1*sum(|w_i|) + l2*sum(w_i^2)  -> elastic net

    Args:
        layer_sizes: list of ints, e.g. [512, 256, 10]
        reg_type: None, 'l2', 'l1', or 'l1_l2'
        reg_lambda: regularization strength (float)

    Returns:
        Compiled Keras Sequential model
    """
    if reg_type == 'l2':
        kernel_reg = regularizers.l2(reg_lambda)
    elif reg_type == 'l1':
        kernel_reg = regularizers.l1(reg_lambda)
    elif reg_type == 'l1_l2':
        kernel_reg = regularizers.l1_l2(l1=reg_lambda, l2=reg_lambda)
    else:
        kernel_reg = None

    layers = []
    for i, units in enumerate(layer_sizes):
        is_output = (i == len(layer_sizes) - 1)
        if is_output:
            # Output layer: softmax, no regularization, no dropout
            layers.append(Dense(units, activation='softmax'))
        else:
            # Hidden layer: ReLU + regularization + dropout
            kwargs = {'activation': 'relu'}
            if i == 0:
                kwargs['input_shape'] = (784,)
            if kernel_reg is not None:
                kwargs['kernel_regularizer'] = kernel_reg
            layers.append(Dense(units, **kwargs))
            layers.append(Dropout(0.2))

    return Sequential(layers)


def train_and_evaluate(model, label_smoothing=0.0, epochs=20, batch_size=128):
    """
    Compile, train, and evaluate a model. Returns results dict.

    Args:
        model: Keras Sequential model (uncompiled)
        label_smoothing: float 0.0-1.0, smooths one-hot targets
        epochs: number of training epochs
        batch_size: training batch size

    Returns:
        dict with keys: test_accuracy, test_loss, params, train_time, history
    """
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing
    )
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )
    train_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    return {
        'model': model,
        'params': model.count_params(),
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'train_time': train_time,
        'history': history
    }


# =============================================================================
# PART A: REGULARIZATION STRATEGY COMPARISON
# =============================================================================
# Compare 5 loss function strategies on the baseline (512 -> 256 -> 10) arch.
#
# Criteria for improving the loss function:
#
# 1. L2 Regularization (Weight Decay)
#    Loss = CE + lambda * sum(w^2)
#    Keeps all weights small. Prevents any single weight from dominating.
#    Standard choice for preventing overfitting.
#
# 2. L1 Regularization (Sparsity)
#    Loss = CE + lambda * sum(|w|)
#    Drives unimportant weights to exactly zero, creating a sparser network.
#    Effectively performs automatic feature selection.
#
# 3. Elastic Net (L1 + L2)
#    Loss = CE + l1*sum(|w|) + l2*sum(w^2)
#    Combines L1 sparsity with L2 stability. Handles correlated features
#    better than L1 alone.
#
# 4. Label Smoothing
#    Instead of hard targets [0,0,1,0,...] use soft [0.01,0.01,0.91,0.01,...]
#    Prevents the model from becoming overconfident, improves calibration,
#    and acts as a form of regularization on the output distribution.
#
# 5. Combined: L2 + Label Smoothing
#    Best of both worlds: weight penalty + output calibration.
# =============================================================================

print("\n" + "=" * 70)
print("PART A: REGULARIZATION STRATEGY COMPARISON")
print("Architecture: 512 -> 256 -> 10 (Baseline)")
print("=" * 70)

baseline_arch = [512, 256, 10]

reg_configs = [
    {'name': 'No Regularization',       'reg_type': None,    'reg_lambda': 0,      'label_smoothing': 0.0},
    {'name': 'L2 (lambda=1e-4)',         'reg_type': 'l2',    'reg_lambda': 1e-4,   'label_smoothing': 0.0},
    {'name': 'L1 (lambda=1e-5)',         'reg_type': 'l1',    'reg_lambda': 1e-5,   'label_smoothing': 0.0},
    {'name': 'Elastic Net (L1+L2)',      'reg_type': 'l1_l2', 'reg_lambda': 1e-5,   'label_smoothing': 0.0},
    {'name': 'L2 + Label Smoothing',     'reg_type': 'l2',    'reg_lambda': 1e-4,   'label_smoothing': 0.1},
]

reg_results = []

for cfg in reg_configs:
    print("\n" + "-" * 60)
    print(f"Training: {cfg['name']}")
    print("-" * 60)

    # Reset seeds for fair comparison
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    m = build_model(baseline_arch, reg_type=cfg['reg_type'], reg_lambda=cfg['reg_lambda'])
    res = train_and_evaluate(m, label_smoothing=cfg['label_smoothing'])
    res['name'] = cfg['name']

    # Compute total weight magnitude for analysis
    total_weight_norm = sum(float(tf.reduce_sum(tf.square(w)).numpy())
                           for w in m.trainable_weights if 'kernel' in w.name)
    res['weight_norm'] = total_weight_norm

    reg_results.append(res)
    print(f"  Test Accuracy: {res['test_accuracy']:.4f} | "
          f"Weight Norm: {total_weight_norm:.2f} | "
          f"Time: {res['train_time']:.1f}s")

# --- Regularization Comparison Table ---
print("\n" + "=" * 90)
print("REGULARIZATION COMPARISON RESULTS")
print("=" * 90)
print(f"{'Strategy':<25s} {'Params':>10s} {'Test Acc':>10s} {'Test Loss':>10s} "
      f"{'W. Norm':>10s} {'Time (s)':>10s}")
print("-" * 90)
for r in reg_results:
    print(f"{r['name']:<25s} {r['params']:>10,d} {r['test_accuracy']:>10.4f} "
          f"{r['test_loss']:>10.4f} {r['weight_norm']:>10.2f} {r['train_time']:>10.1f}")
print("=" * 90)

best_reg = max(reg_results, key=lambda x: x['test_accuracy'])
print(f"\nBest regularization strategy: {best_reg['name']} "
      f"with {best_reg['test_accuracy']*100:.2f}% test accuracy")

# --- Regularization Charts ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Regularization Strategy Comparison (Baseline Architecture)',
             fontsize=16, fontweight='bold')

# Chart 1: Test Accuracy
names_r = [r['name'] for r in reg_results]
accs_r = [r['test_accuracy'] for r in reg_results]
colors_r = ['#2ecc71' if r is best_reg else '#3498db' for r in reg_results]
axes[0].barh(names_r, accs_r, color=colors_r, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Test Accuracy', fontsize=12)
axes[0].set_title('Test Accuracy', fontsize=13, fontweight='bold')
for i, acc in enumerate(accs_r):
    axes[0].text(acc + 0.001, i, f'{acc:.4f}', va='center', fontsize=10, fontweight='bold')
axes[0].set_xlim([min(accs_r) - 0.01, max(accs_r) + 0.015])

# Chart 2: Weight Norms
norms = [r['weight_norm'] for r in reg_results]
axes[1].barh(names_r, norms, color='#e74c3c', edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('Sum of Squared Weights', fontsize=12)
axes[1].set_title('Weight Norm (smaller = simpler model)', fontsize=13, fontweight='bold')
for i, n in enumerate(norms):
    axes[1].text(n + max(norms)*0.01, i, f'{n:.1f}', va='center', fontsize=10)

# Chart 3: Validation Accuracy Curves
style_map = [('blue', '-'), ('orange', '--'), ('green', '-.'), ('red', ':'), ('purple', '-')]
for i, r in enumerate(reg_results):
    c, ls = style_map[i]
    axes[2].plot(r['history'].history['val_accuracy'], label=r['name'],
                 color=c, linestyle=ls, linewidth=2)
axes[2].set_title('Validation Accuracy Over Epochs', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Validation Accuracy', fontsize=12)
axes[2].legend(fontsize=8, loc='lower right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# PART B: ARCHITECTURE COMPARISON (with best regularization)
# =============================================================================
# Now compare 4 architectures using the best loss function from Part A.
# =============================================================================

# Determine best regularization config
best_reg_cfg = reg_configs[reg_results.index(best_reg)]

print("\n" + "=" * 70)
print("PART B: ARCHITECTURE COMPARISON")
print(f"Using best loss: {best_reg['name']}")
print("=" * 70)

arch_configs = [
    {'name': 'Baseline (512-256)',       'layers': [512, 256, 10]},
    {'name': 'Narrow (128-64)',          'layers': [128, 64, 10]},
    {'name': 'Deep (512-256-128-64)',    'layers': [512, 256, 128, 64, 10]},
    {'name': 'Wide (1024-512-256)',      'layers': [1024, 512, 256, 10]},
]

arch_results = []

for arch in arch_configs:
    print("\n" + "-" * 60)
    print(f"Training: {arch['name']}")
    print("-" * 60)

    # Reset seeds for fair comparison
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    m = build_model(arch['layers'],
                    reg_type=best_reg_cfg['reg_type'],
                    reg_lambda=best_reg_cfg['reg_lambda'])
    res = train_and_evaluate(m, label_smoothing=best_reg_cfg['label_smoothing'])
    res['name'] = arch['name']
    arch_results.append(res)
    print(f"  Test Accuracy: {res['test_accuracy']:.4f} | "
          f"Params: {res['params']:,} | Time: {res['train_time']:.1f}s")

# --- Architecture Comparison Table ---
print("\n" + "=" * 80)
print("ARCHITECTURE COMPARISON RESULTS")
print(f"(All using: {best_reg['name']})")
print("=" * 80)
print(f"{'Model':<25s} {'Params':>10s} {'Test Acc':>10s} {'Test Loss':>10s} {'Time (s)':>10s}")
print("-" * 80)
for r in arch_results:
    print(f"{r['name']:<25s} {r['params']:>10,d} {r['test_accuracy']:>10.4f} "
          f"{r['test_loss']:>10.4f} {r['train_time']:>10.1f}")
print("=" * 80)

best_arch = max(arch_results, key=lambda x: x['test_accuracy'])
print(f"\nBest architecture: {best_arch['name']} "
      f"with {best_arch['test_accuracy']*100:.2f}% test accuracy")

# --- Architecture Charts ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(f'Architecture Comparison (Loss: {best_reg["name"]})',
             fontsize=16, fontweight='bold')

# Bar chart
names_a = [r['name'] for r in arch_results]
accs_a = [r['test_accuracy'] for r in arch_results]
colors_a = ['#2ecc71' if r is best_arch else '#3498db' for r in arch_results]
axes[0].bar(names_a, accs_a, color=colors_a, edgecolor='black', linewidth=0.5)
axes[0].set_title('Test Accuracy by Architecture', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Test Accuracy', fontsize=12)
axes[0].set_ylim([min(accs_a) - 0.02, max(accs_a) + 0.02])
axes[0].grid(axis='y', alpha=0.3)
for i, acc in enumerate(accs_a):
    axes[0].text(i, acc + 0.003, f'{acc:.4f}', ha='center', fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)

# Validation accuracy curves
style_map_a = [('blue', '-'), ('orange', '--'), ('green', '-.'), ('red', ':')]
for i, r in enumerate(arch_results):
    c, ls = style_map_a[i]
    axes[1].plot(r['history'].history['val_accuracy'], label=r['name'],
                 color=c, linestyle=ls, linewidth=2)
axes[1].set_title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Validation Accuracy', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)
print(f"\nBest regularization: {best_reg['name']}")
print(f"  -> Test accuracy: {best_reg['test_accuracy']*100:.2f}%")
print(f"  -> Weight norm:   {best_reg['weight_norm']:.2f}")
print(f"\nBest architecture:  {best_arch['name']}")
print(f"  -> Test accuracy: {best_arch['test_accuracy']*100:.2f}%")
print(f"  -> Parameters:    {best_arch['params']:,}")
print(f"  -> Training time: {best_arch['train_time']:.1f}s")
print(f"\nConclusion:")
print(f"  The combination of {best_reg['name']} loss function")
print(f"  with {best_arch['name']} architecture achieved the best results.")
print("=" * 70)
