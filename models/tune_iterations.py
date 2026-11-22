import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

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

scaler = joblib.load(MODEL_DIR / "scaler.joblib")
encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

X_val = X_val[X_train.columns]

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

y_train_encoded = encoder.transform(y_train)
y_val_encoded = encoder.transform(y_val)

ALPHA = 0.00001
ITERATIONS = [50, 100, 150]

best_f1 = -1
best_model = None
best_iterations = None
results = []

for iterations in ITERATIONS:

    print("\nTesting max_iter:", iterations)

    model = SGDClassifier(
        loss="log_loss",
        alpha=ALPHA,
        max_iter=iterations,
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
        "max_iter": iterations,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = model
        best_iterations = iterations

pd.DataFrame(results).to_csv(
    MODEL_DIR / "iteration_results.csv",
    index=False
)

joblib.dump(
    best_model,
    MODEL_DIR / "best_sgd_convergence_model.joblib"
)

print("\nIteration tuning completed!")
print("Best max_iter:", best_iterations)
print("Best Macro F1:", round(best_f1, 4))
print("Saved best convergence model.")