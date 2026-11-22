import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "saved"

TEST_PATH = (
    BASE_DIR / "dataset" / "raw" /
    "CICIoT2023" / "CICIOT23" / "test" / "test.csv"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 20000

# ---------------- LOAD MODEL ----------------

print("Loading Random Forest model...")

model = joblib.load(
    MODEL_DIR / "random_forest_baseline.joblib"
)

encoder = joblib.load(
    MODEL_DIR / "rf_label_encoder.joblib"
)

print("Model loaded successfully!")

# ---------------- EVALUATION ----------------

y_true_all = []
predictions_all = []

print("\nStarting Random Forest final test evaluation...")

for chunk_number, chunk in enumerate(
    pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    X = chunk.drop(columns=["label"])

    X = X[model.feature_names_in_].astype(np.float32)

    y = encoder.transform(chunk["label"])

    predictions = model.predict(X)

    y_true_all.extend(y)
    predictions_all.extend(predictions)

    print(
        f"Processed chunk {chunk_number} | "
        f"Records: {len(y_true_all)}"
    )

# ---------------- FINAL METRICS ----------------

y_true_all = np.array(y_true_all)
predictions_all = np.array(predictions_all)

accuracy = accuracy_score(
    y_true_all,
    predictions_all
)

macro_f1 = f1_score(
    y_true_all,
    predictions_all,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true_all,
    predictions_all,
    average="weighted",
    zero_division=0
)

print("\n========== RANDOM FOREST FINAL TEST RESULTS ==========")

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))

report = classification_report(
    y_true_all,
    predictions_all,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    RESULTS_DIR / "final_random_forest_test_report.csv"
)

print("\nClassification Report:")

print(
    classification_report(
        y_true_all,
        predictions_all,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

# ---------------- SAVE COMPARISON ----------------

comparison_path = RESULTS_DIR / "final_model_comparison.csv"

rf_row = pd.DataFrame([{
    "Model": "Random Forest",
    "Accuracy": accuracy,
    "Macro F1": macro_f1,
    "Weighted F1": weighted_f1
}])

if comparison_path.exists():
    comparison_df = pd.read_csv(comparison_path)

    comparison_df = comparison_df[
        comparison_df["Model"] != "Random Forest"
    ]

    comparison_df = pd.concat(
        [comparison_df, rf_row],
        ignore_index=True
    )
else:
    comparison_df = rf_row

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("\nRandom Forest test report saved successfully!")
print("Final model comparison updated!")
print("\nEvaluation completed!")