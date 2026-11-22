import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# File paths
TRAIN_PATH = Path("dataset/processed/train_sample.csv")
VAL_PATH = Path("dataset/processed/validation_sample.csv")

MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training data...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

# Separate features and labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_val = val_df.drop(columns=["label"])
y_val = val_df["label"]

# Check data
print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# Encode labels
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)

# Ensure validation labels are known
unknown_labels = set(y_val) - set(label_encoder.classes_)

if unknown_labels:
    raise ValueError(f"Unknown validation labels: {unknown_labels}")

y_val_encoded = label_encoder.transform(y_val)

# Convert features to numeric and check values
X_train = X_train.astype(np.float64)
X_val = X_val.astype(np.float64)

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Training data contains NaN or infinite values!")

if not np.isfinite(X_val.to_numpy()).all():
    raise ValueError("Validation data contains NaN or infinite values!")

# Feature scaling: fit only on training data
print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train baseline model
print("\nTraining SGD Classifier...")

model = SGDClassifier(
    loss="log_loss",
    max_iter=50,
    tol=1e-3,
    class_weight="balanced",
    random_state=42,
    average=True
)

model.fit(X_train_scaled, y_train_encoded)

print("\nTraining completed!")

# Validation predictions
print("\nEvaluating model...")

y_pred = model.predict(X_val_scaled)

accuracy = accuracy_score(y_val_encoded, y_pred)
macro_f1 = f1_score(
    y_val_encoded,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val_encoded,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# Save model files
joblib.dump(model, MODEL_DIR / "sgd_baseline.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

print("\nModel files saved in:", MODEL_DIR)
print("Baseline training pipeline completed!")