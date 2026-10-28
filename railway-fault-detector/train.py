"""Train a binary railway-track fault classifier (MobileNetV2, two-phase fine-tuning).

Trained on Kaggle (GPU). The dataset is attached as a Kaggle input; change DATA_DIR
if you run this somewhere else. Expected sub-folders: train / valid / test.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
DATA_DIR = pathlib.Path("/kaggle/input/datasets/boddayamini/railway-track-fault-detection-resized-224-x-224/model")
OUT_DIR = pathlib.Path("model")
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
EPOCHS_HEAD = 8
EPOCHS_FINE = 8
# ----------------------------------------

OUT_DIR.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


def load(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / split, label_mode="binary", image_size=IMG_SIZE,
        batch_size=BATCH, shuffle=shuffle, seed=SEED)


train_ds, val_ds, test_ds = load("train", True), load("valid", False), load("test", False)
class_names = train_ds.class_names
print("Classes:", class_names)

# Which output index means "defective"?
DEFECT_IDX = [i for i, n in enumerate(class_names) if "non" not in n.lower()][0]

# Class weights (handles imbalance)
y_train = np.concatenate([y.numpy().ravel() for _, y in train_ds]).astype(int)
counts = np.bincount(y_train)
class_weight = {i: len(y_train) / (2 * c) for i, c in enumerate(counts)}
print("Train counts:", counts, "Class weights:", class_weight)

AUTOTUNE = tf.data.AUTOTUNE
train_ds, val_ds, test_ds = (d.prefetch(AUTOTUNE) for d in (train_ds, val_ds, test_ds))

# ---------------- MODEL ----------------
inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
x = layers.RandomRotation(0.05, name="aug_rot")(x)
x = layers.RandomContrast(0.2, name="aug_contrast")(x)
x = layers.RandomBrightness(0.2, value_range=(0, 255), name="aug_bright")(x)
x = layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(x)   # MobileNetV2 preprocessing

backbone = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                          input_shape=(*IMG_SIZE, 3), name="backbone")
backbone.trainable = False
x = backbone(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="drop")(x)
outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
model = keras.Model(inputs, outputs)

metrics = ["accuracy", keras.metrics.Recall(name="recall"),
           keras.metrics.Precision(name="precision"), keras.metrics.AUC(name="auc")]


def compile_model(lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=metrics)


callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
]

# Phase 1: train head only
compile_model(1e-3)
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
               class_weight=class_weight, callbacks=callbacks)

# Phase 2: fine-tune the last 30 backbone layers
backbone.trainable = True
for layer in backbone.layers[:-30]:
    layer.trainable = False
for layer in backbone.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
compile_model(1e-5)
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
               class_weight=class_weight, callbacks=callbacks)

# ---------------- EVALUATION ----------------
y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
p1 = model.predict(test_ds).ravel()                     # P(class index 1)
p_defect = p1 if DEFECT_IDX == 1 else 1 - p1
y_defect = (y_true == DEFECT_IDX).astype(int)
y_pred = (p_defect >= 0.5).astype(int)

report = classification_report(y_defect, y_pred, target_names=["Non-defective", "Defective"],
                               output_dict=True)
auc = roc_auc_score(y_defect, p_defect)
cm = confusion_matrix(y_defect, y_pred)
tn, fp, fn, tp = cm.ravel()
summary = {
    "test_images": int(len(y_true)),
    "accuracy": float(report["accuracy"]),
    "defective_recall": float(report["Defective"]["recall"]),
    "defective_precision": float(report["Defective"]["precision"]),
    "defective_f1": float(report["Defective"]["f1-score"]),
    "roc_auc": float(auc),
    "false_negative_rate": float(fn / max(fn + tp, 1)),
    "false_positive_rate": float(fp / max(fp + tn, 1)),
    "confusion_matrix": cm.tolist(),
}
print(json.dumps(summary, indent=2))

# ---------------- SAVE ----------------
model.save(OUT_DIR / "track_model.keras")
json.dump({"class_names": class_names, "defect_idx": DEFECT_IDX}, open(OUT_DIR / "class_names.json", "w"))
json.dump(summary, open(OUT_DIR / "metrics.json", "w"), indent=2)

ConfusionMatrixDisplay(cm, display_labels=["Non-defective", "Defective"]).plot(cmap="Blues")
plt.title("Test confusion matrix"); plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

fpr, tpr, _ = roc_curve(y_defect, p_defect)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); plt.plot([0, 1], [0, 1], "--", c="grey")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.title("ROC curve")
plt.savefig(OUT_DIR / "roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()

hist = {k: h1.history[k] + h2.history[k] for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist["accuracy"], label="train"); ax[1].plot(hist["val_accuracy"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
plt.savefig(OUT_DIR / "history.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved to", OUT_DIR.resolve())
