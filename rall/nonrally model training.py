import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Create Folder for Results
results_dir = "rally_model"
os.makedirs(results_dir, exist_ok=True)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using only GPU 0:", gpus[0])
    except RuntimeError as e:
        print("GPU Error:", e)
else:
    print(" No GPU detected. Running on CPU.")

# Dataset Paths & Parameters
data_dir = 'dataset'
img_size = (224, 224)
batch_size = 16

# Load Data with Proper Splitting (70% Train, 20% Validation, 10% Test)
train_ds = image_dataset_from_directory(
    data_dir, validation_split=0.3, subset="training", seed=123,
    image_size=img_size, batch_size=batch_size)

temp_ds = image_dataset_from_directory(
    data_dir, validation_split=0.3, subset="validation", seed=123,
    image_size=img_size, batch_size=batch_size)

# Further split temp dataset into validation (20%) and test (10%)
val_batches = int(0.66 * len(temp_ds))
val_ds = temp_ds.take(val_batches)  # First 2/3 for validation
test_ds = temp_ds.skip(val_batches)  # Remaining 1/3 for testing

# Data Augmentation (Only on Train Data)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2)
])

# Normalize Images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))  # Augment train data
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Load MobileNetV2 (Pretrained)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model layers

# Create a Robust Model with Regularization
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # L2 Regularization
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary Classification
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(os.path.join(results_dir, "best_model.h5"), save_best_only=True)
]

# Train Model (Initial Training)
history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# Unfreeze Base Model for Fine-Tuning
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])

# Fine-Tune Model
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

# Save Final Model
model.save(os.path.join(results_dir, "final_model.h5"))
print("🎉 Training Complete! Model Saved in 'rally_model'")

# **EVALUATION SECTION**
print(" Generating Evaluation Results...\n")

# Evaluate on Test Data
y_true, y_pred_probs = [], []
for images, labels in test_ds:
    y_true.extend(labels.numpy())
    y_pred_probs.extend(model.predict(images).flatten())

y_pred = [1 if p > 0.5 else 0 for p in y_pred_probs]

# *Plot Accuracy & Loss Graphs**
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")
plt.savefig(os.path.join(results_dir, "accuracy_plot.png"))

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.savefig(os.path.join(results_dir, "loss_plot.png"))

plt.show()

# Confusion Matrix**
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Rally", "Rally"], yticklabels=["Non-Rally", "Rally"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.show()

# Precision, Recall, F1-Score**
report = classification_report(y_true, y_pred, target_names=["Non-Rally", "Rally"])
print("\n📊 Classification Report:\n", report)

# Save report to a file
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# ROC Curve & AUC Score**
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
auc_score = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], "k--")  # Random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(results_dir, "roc_curve.png"))
plt.show()

print("\nAll Evaluation Results Saved in 'rally_model' Folder!")
