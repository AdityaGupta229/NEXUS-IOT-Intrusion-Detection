import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

TRAIN_PATH = Path("dataset/processed/train_sample.csv")
VAL_PATH = Path("dataset/processed/validation_sample.csv")
MODEL_DIR = Path("models/saved")

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_val = val_df.drop(columns=["label"])
y_val = val_df["label"]

# Encode labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

unknown = set(y_val) - set(encoder.classes_)
if unknown:
    raise ValueError(f"Unknown labels: {unknown}")

y_val_encoded = encoder.transform(y_val)

# Random Forest does not require feature scaling
X_train = X_train.astype(np.float32)
X_val = X_val[X_train.columns].astype(np.float32)

if not np.isfinite(X_train.to_numpy()).all():
    raise ValueError("Invalid training values!")

if not np.isfinite(X_val.to_numpy()).all():
    raise ValueError("Invalid validation values!")

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=18,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=2
)

model.fit(X_train, y_train_encoded)

print("Training completed!")

print("Evaluating model...")

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val_encoded, predictions)

macro_f1 = f1_score(
    y_val_encoded,
    predictions,
    average="macro",
    zero_division=0
)

print("\n========== RANDOM FOREST RESULTS ==========")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_val_encoded,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

joblib.dump(model, MODEL_DIR / "random_forest_baseline.joblib")
joblib.dump(encoder, MODEL_DIR / "rf_label_encoder.joblib")

print("\nRandom Forest model saved successfully!")