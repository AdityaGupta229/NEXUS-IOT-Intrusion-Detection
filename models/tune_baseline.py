import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# Paths
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

# Load previously fitted scaler and label encoder
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

# Ensure feature columns are in the same order
X_val = X_val[X_train.columns]

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

y_train_encoded = label_encoder.transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Alpha values to compare
alpha_values = [0.00001, 0.0001, 0.001]

results = []
best_f1 = -1
best_model = None
best_alpha = None

for alpha in alpha_values:

    print("\n" + "=" * 50)
    print("Testing alpha:", alpha)

    model = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        max_iter=50,
        tol=1e-3,
        class_weight="balanced",
        random_state=42,
        average=True
    )

    model.fit(X_train_scaled, y_train_encoded)

    predictions = model.predict(X_val_scaled)

    accuracy = accuracy_score(y_val_encoded, predictions)

    macro_f1 = f1_score(
        y_val_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    results.append({
        "alpha": alpha,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = model
        best_alpha = alpha

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    MODEL_DIR / "tuning_results.csv",
    index=False
)

# Save best model
joblib.dump(
    best_model,
    MODEL_DIR / "best_sgd_model.joblib"
)

print("\n" + "=" * 50)
print("TUNING COMPLETED")

print("\nBest Alpha:", best_alpha)
print("Best Macro F1:", round(best_f1, 4))

# Detailed report for best model
best_predictions = best_model.predict(X_val_scaled)

print("\nBest Model Classification Report:")

print(
    classification_report(
        y_val_encoded,
        best_predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

print("\nSaved:")
print("models/saved/best_sgd_model.joblib")
print("models/saved/tuning_results.csv")