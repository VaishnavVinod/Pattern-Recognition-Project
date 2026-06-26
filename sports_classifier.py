"""
======================================================================
  SPORTS ACTION RECOGNITION FRAMEWORK
  Pattern Recognition Course - Phase 2 Implementation
  Dataset: Sports Classification (Kaggle - gpiosenka)
  https://www.kaggle.com/datasets/gpiosenka/sports-classification
======================================================================
"""

# ─────────────────────────────────────────────────────────────────────
# 1. INSTALL / IMPORTS
# ─────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)

from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# ─────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
# ── Dataset root (Kaggle path) ──────────────────────────────────────
# On Kaggle, the gpiosenka dataset mounts at the path below.
# If running elsewhere, change DATA_ROOT accordingly.
DATA_ROOT = "/kaggle/input/datasets/gpiosenka/sports-classification"

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "valid")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

# Fail fast with a clear message if the path is wrong on a future run
assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isdir(VAL_DIR),   f"VAL_DIR not found: {VAL_DIR}"
assert os.path.isdir(TEST_DIR),  f"TEST_DIR not found: {TEST_DIR}"

IMG_SIZE   = (224, 224)   # Required by EfficientNetB0 / MobileNetV2
BATCH_SIZE = 32
EPOCHS_CNN = 30           # Custom CNN training epochs
EPOCHS_TL  = 20           # Transfer-learning fine-tune epochs
NUM_CLASSES = 100         # Dataset has 100 sport categories

OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# 3. DATASET LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────

# ── 3a. Data Augmentation for training ──────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # Normalise pixel values to [0, 1]
    rotation_range=20,          # Random rotation ±20°
    width_shift_range=0.15,     # Horizontal shift
    height_shift_range=0.15,    # Vertical shift
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,       # Flip left-right (valid for most sports)
    fill_mode="nearest"
)

# Validation & test: only rescale, no augmentation
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# ── 3b. Data Generators ─────────────────────────────────────────────
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED,
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED,
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED,
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())
NUM_CLASSES  = len(class_names)
print(f"\nTotal classes detected: {NUM_CLASSES}")
print(f"Training samples  : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Test samples      : {test_gen.samples}")

# ── 3c. Visualise Sample Images ─────────────────────────────────────
def plot_sample_images(generator, class_names, n=16, save_path=None):
    """Display a grid of sample training images."""
    images, labels = next(generator)
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle("Sample Training Images", fontsize=16, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[i])
            ax.set_title(class_names[np.argmax(labels[i])], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

plot_sample_images(train_gen, class_names,
                   save_path=os.path.join(OUTPUT_DIR, "sample_images.png"))

# ─────────────────────────────────────────────────────────────────────
# 4. CUSTOM CNN MODEL
# ─────────────────────────────────────────────────────────────────────
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=100):
    """
    5-block Custom CNN architecture:
      Block 1-2 : feature extraction with double conv layers
      Block 3-5 : deeper feature extraction
      Head      : GlobalAveragePooling → Dense → Dropout → Softmax
    """
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="Custom_CNN")

cnn_model = build_custom_cnn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
cnn_model.summary()

# ─────────────────────────────────────────────────────────────────────
# 5. TRAIN CUSTOM CNN
# ─────────────────────────────────────────────────────────────────────
cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_cnn = [
    EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_custom_cnn.keras"),
                    monitor="val_accuracy", save_best_only=True, verbose=1)
]

print("\n" + "="*60)
print("  TRAINING CUSTOM CNN")
print("="*60)
history_cnn = cnn_model.fit(
    train_gen,
    epochs=EPOCHS_CNN,
    validation_data=val_gen,
    callbacks=callbacks_cnn,
    verbose=1
)

# ─────────────────────────────────────────────────────────────────────
# 6. TRANSFER LEARNING – EfficientNetB0
# ─────────────────────────────────────────────────────────────────────
def build_efficientnet(num_classes=100):
    """EfficientNetB0 with frozen base + custom classification head."""
    base = EfficientNetB0(weights="imagenet", include_top=False,
                          input_shape=(*IMG_SIZE, 3))
    # Freeze base initially
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # EfficientNetB0 expects inputs in [0,255] when using its own preprocessing
    x = keras.applications.efficientnet.preprocess_input(inputs * 255)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="EfficientNetB0_TL")

eff_model = build_efficientnet(NUM_CLASSES)
eff_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_eff = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_efficientnet.keras"),
                    monitor="val_accuracy", save_best_only=True, verbose=1)
]

print("\n" + "="*60)
print("  TRAINING EfficientNetB0 (frozen base)")
print("="*60)
history_eff = eff_model.fit(
    train_gen,
    epochs=EPOCHS_TL,
    validation_data=val_gen,
    callbacks=callbacks_eff,
    verbose=1
)

# ── Fine-tuning: unfreeze top 30 layers ─────────────────────────────
# FIX: locate the EfficientNet base by type instead of a hard-coded
# layer index. The base is itself a nested Model, so we search for the
# first Model instance among the layers. This avoids an AttributeError
# that would otherwise crash AFTER the 20 frozen-base epochs complete.
print("\nFine-tuning EfficientNetB0 (unfreezing top 30 layers)…")
base_eff = None
for layer in eff_model.layers:
    if isinstance(layer, Model):       # the EfficientNetB0 base is a Model
        base_eff = layer
        break
assert base_eff is not None, "Could not locate the EfficientNet base model."
print(f"Located base model: {base_eff.name} with {len(base_eff.layers)} layers")

base_eff.trainable = True
for layer in base_eff.layers[:-30]:    # keep all but the top 30 frozen
    layer.trainable = False

eff_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),   # Very low LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_eff_ft = eff_model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=callbacks_eff,
    verbose=1
)

# ─────────────────────────────────────────────────────────────────────
# 7. TRANSFER LEARNING – MobileNetV2
# ─────────────────────────────────────────────────────────────────────
def build_mobilenet(num_classes=100):
    """MobileNetV2 with frozen base + custom head."""
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="MobileNetV2_TL")

mob_model = build_mobilenet(NUM_CLASSES)
mob_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_mob = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_mobilenet.keras"),
                    monitor="val_accuracy", save_best_only=True, verbose=1)
]

print("\n" + "="*60)
print("  TRAINING MobileNetV2 (frozen base)")
print("="*60)
history_mob = mob_model.fit(
    train_gen,
    epochs=EPOCHS_TL,
    validation_data=val_gen,
    callbacks=callbacks_mob,
    verbose=1
)

# ─────────────────────────────────────────────────────────────────────
# 8. EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────

def plot_training_curves(history, model_name, save_path=None):
    """Plot accuracy and loss curves side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=14, fontweight="bold")

    # Accuracy
    ax1.plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2)
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"],     label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def evaluate_model(model, test_gen, class_names, model_name, save_prefix):
    """
    Full evaluation pipeline:
      - Test accuracy & loss
      - Classification report
      - Confusion matrix (top-20 classes)
    Returns (test_loss, test_acc, y_true, y_pred, y_prob)
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {model_name}")
    print(f"{'='*60}")

    test_gen.reset()
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"  Test Loss    : {loss:.4f}")
    print(f"  Test Accuracy: {acc:.4f}  ({acc*100:.2f}%)")

    # Predictions
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0)
    print("\nClassification Report:\n", report)
    with open(f"{save_prefix}_classification_report.txt", "w") as f:
        f.write(f"Model: {model_name}\nTest Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Confusion matrix — show only top-20 most frequent classes for readability
    top20_idx   = np.argsort(np.bincount(y_true))[-20:]
    mask        = np.isin(y_true, top20_idx)
    y_true_top  = y_true[mask]
    y_pred_top  = y_pred[mask]
    top20_names = [class_names[i] for i in top20_idx]

    cm = confusion_matrix(y_true_top, y_pred_top, labels=top20_idx)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=top20_names, yticklabels=top20_names, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix (Top-20 Classes)", fontsize=14)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = f"{save_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {cm_path}")
    plt.show()

    return loss, acc, y_true, y_pred, y_prob


# ─────────────────────────────────────────────────────────────────────
# 9. PLOT CURVES & EVALUATE ALL MODELS
# ─────────────────────────────────────────────────────────────────────
plot_training_curves(history_cnn, "Custom CNN",
    save_path=os.path.join(OUTPUT_DIR, "cnn_training_curves.png"))

plot_training_curves(history_eff, "EfficientNetB0",
    save_path=os.path.join(OUTPUT_DIR, "efficientnet_training_curves.png"))

plot_training_curves(history_mob, "MobileNetV2",
    save_path=os.path.join(OUTPUT_DIR, "mobilenet_training_curves.png"))

loss_cnn, acc_cnn, yt_cnn, yp_cnn, prob_cnn = evaluate_model(
    cnn_model, test_gen, class_names, "Custom CNN",
    os.path.join(OUTPUT_DIR, "cnn"))

loss_eff, acc_eff, yt_eff, yp_eff, prob_eff = evaluate_model(
    eff_model, test_gen, class_names, "EfficientNetB0",
    os.path.join(OUTPUT_DIR, "efficientnet"))

loss_mob, acc_mob, yt_mob, yp_mob, prob_mob = evaluate_model(
    mob_model, test_gen, class_names, "MobileNetV2",
    os.path.join(OUTPUT_DIR, "mobilenet"))

# ─────────────────────────────────────────────────────────────────────
# 10. MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score, f1_score

def macro_metrics(y_true, y_pred):
    p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    f = f1_score       (y_true, y_pred, average="macro", zero_division=0)
    return p, r, f

p_cnn, r_cnn, f_cnn = macro_metrics(yt_cnn, yp_cnn)
p_eff, r_eff, f_eff = macro_metrics(yt_eff, yp_eff)
p_mob, r_mob, f_mob = macro_metrics(yt_mob, yp_mob)

comparison = pd.DataFrame({
    "Model"         : ["Custom CNN", "EfficientNetB0", "MobileNetV2"],
    "Test Loss"     : [f"{loss_cnn:.4f}", f"{loss_eff:.4f}", f"{loss_mob:.4f}"],
    "Test Accuracy" : [f"{acc_cnn*100:.2f}%", f"{acc_eff*100:.2f}%", f"{acc_mob*100:.2f}%"],
    "Macro Precision": [f"{p_cnn:.4f}", f"{p_eff:.4f}", f"{p_mob:.4f}"],
    "Macro Recall"  : [f"{r_cnn:.4f}", f"{r_eff:.4f}", f"{r_mob:.4f}"],
    "Macro F1"      : [f"{f_cnn:.4f}", f"{f_eff:.4f}", f"{f_mob:.4f}"],
    "Parameters"    : [
        f"{cnn_model.count_params():,}",
        f"{eff_model.count_params():,}",
        f"{mob_model.count_params():,}"
    ]
})

print("\n" + "="*60)
print("  MODEL COMPARISON")
print("="*60)
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

# Visual comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
models = ["Custom CNN", "EfficientNetB0", "MobileNetV2"]
accs   = [acc_cnn*100, acc_eff*100, acc_mob*100]
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars   = ax.bar(models, accs, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
ax.set_ylim(0, 105)
ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Test Accuracy (%)"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────
# 11. GRAD-CAM VISUALISATION  (best model)
# ─────────────────────────────────────────────────────────────────────
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Compute Grad-CAM heatmap for the given image and model."""
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads      = tape.gradient(class_channel, conv_outputs)
    pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out   = conv_outputs[0]
    heatmap    = conv_out @ pooled[..., tf.newaxis]
    heatmap    = tf.squeeze(heatmap)
    heatmap    = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def display_gradcam(img_path, model, last_conv_layer_name, class_names,
                    alpha=0.4, save_path=None):
    """Load an image, run Grad-CAM, and overlay heatmap."""
    img        = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array  = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    heatmap    = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap_r  = cv2.resize(heatmap, IMG_SIZE)
    heatmap_c  = np.uint8(255 * heatmap_r)
    heatmap_c  = cv2.applyColorMap(heatmap_c, cv2.COLORMAP_JET)
    heatmap_c  = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)

    orig       = np.uint8(img_array[0] * 255)
    superimposed = cv2.addWeighted(orig, 1 - alpha, heatmap_c, alpha, 0)

    preds      = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(preds[0])]
    conf       = np.max(preds[0]) * 100

    fig, axes  = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(orig);        axes[0].set_title("Original Image")
    axes[1].imshow(heatmap_r, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(superimposed); axes[2].set_title(f"Overlay\nPred: {pred_class} ({conf:.1f}%)")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ── Find the last conv layer in Custom CNN ───────────────────────────
last_conv_cnn = None
for layer in reversed(cnn_model.layers):
    if isinstance(layer, layers.Conv2D):
        last_conv_cnn = layer.name
        break

print(f"Last Conv layer in Custom CNN: {last_conv_cnn}")

# Run Grad-CAM on a few test images
test_gen.reset()
sample_imgs = []
for root, dirs, files in os.walk(TEST_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            sample_imgs.append(os.path.join(root, f))
    if len(sample_imgs) >= 4:
        break

for i, img_path in enumerate(sample_imgs[:4]):
    display_gradcam(
        img_path, cnn_model, last_conv_cnn, class_names,
        save_path=os.path.join(OUTPUT_DIR, f"gradcam_sample_{i+1}.png")
    )

# ─────────────────────────────────────────────────────────────────────
# 12. ROC CURVE (macro-average, best model)
# ─────────────────────────────────────────────────────────────────────

def plot_macro_roc(y_true, y_prob, class_names, model_name, save_path=None):
    """Plot macro-average ROC curve and report macro AUC."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Micro-style curve for the plotted line (flattened one-vs-rest)
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())

    # FIX: compute macro AUC directly from the binarised matrix.
    # Passing multi_class with an already-binarised 2D target can raise;
    # using the indicator matrix with average="macro" is the safe form.
    # Guard against any class that has no positive samples in the test set.
    try:
        auc = roc_auc_score(y_bin, y_prob, average="macro")
    except ValueError:
        # Fall back: average AUC only over classes that are present
        present = np.where(y_bin.sum(axis=0) > 0)[0]
        auc = roc_auc_score(y_bin[:, present], y_prob[:, present],
                            average="macro")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"Macro-avg ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — Macro-Average ROC Curve", fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    print(f"  Macro-Avg AUC: {auc:.4f}")

plot_macro_roc(yt_cnn, prob_cnn, class_names, "Custom CNN",
               save_path=os.path.join(OUTPUT_DIR, "roc_cnn.png"))
plot_macro_roc(yt_eff, prob_eff, class_names, "EfficientNetB0",
               save_path=os.path.join(OUTPUT_DIR, "roc_efficientnet.png"))
plot_macro_roc(yt_mob, prob_mob, class_names, "MobileNetV2",
               save_path=os.path.join(OUTPUT_DIR, "roc_mobilenet.png"))

# ─────────────────────────────────────────────────────────────────────
# 13. ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def error_analysis(y_true, y_pred, class_names, model_name, top_n=10, save_path=None):
    """Find and visualise most commonly confused class pairs."""
    errors = [(class_names[t], class_names[p])
              for t, p in zip(y_true, y_pred) if t != p]

    from collections import Counter
    top_errors = Counter(errors).most_common(top_n)

    pairs  = [f"{t} → {p}" for (t, p), _ in top_errors]
    counts = [c for _, c in top_errors]

    plt.figure(figsize=(12, 5))
    plt.barh(pairs[::-1], counts[::-1], color="tomato", edgecolor="white")
    plt.xlabel("Number of Misclassifications")
    plt.title(f"{model_name} — Top {top_n} Confusion Pairs", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

error_analysis(yt_cnn, yp_cnn, class_names, "Custom CNN",
               save_path=os.path.join(OUTPUT_DIR, "error_analysis_cnn.png"))
error_analysis(yt_eff, yp_eff, class_names, "EfficientNetB0",
               save_path=os.path.join(OUTPUT_DIR, "error_analysis_efficientnet.png"))

# ─────────────────────────────────────────────────────────────────────
# 14. SAVE FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)
print(comparison.to_string(index=False))

summary_path = os.path.join(OUTPUT_DIR, "final_summary.txt")
with open(summary_path, "w") as f:
    f.write("SPORTS ACTION RECOGNITION FRAMEWORK\n")
    f.write("Pattern Recognition Course – Phase 2\n")
    f.write("="*60 + "\n\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\nAll output figures saved to: " + OUTPUT_DIR)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Done! \u2713")
