import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent.parent

SAVE_DIR = BASE_DIR / "models" / "saved" / "corrected_pipeline"

RESULTS_DIR = BASE_DIR / "results" / "corrected_pipeline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = (
    BASE_DIR / "dataset" / "raw" / "CICIoT2023"
    / "CICIOT23" / "test" / "test.csv"
)

CHUNK_SIZE = 20000
DEVICE = torch.device("cpu")

# ---------------- MODEL ARCHITECTURE ----------------

class IntrusionMLP(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ---------------- LOAD PREPROCESSING ----------------

print("Loading preprocessing files...")

scaler = joblib.load(
    SAVE_DIR / "centralized_torch_scaler.joblib"
)

encoder = joblib.load(
    SAVE_DIR / "centralized_torch_encoder.joblib"
)

NUM_FEATURES = len(scaler.feature_names_in_)
NUM_CLASSES = len(encoder.classes_)
# ---------------- LOAD MODELS ----------------

def load_model(filename):

    model = IntrusionMLP(
        NUM_FEATURES,
        NUM_CLASSES
    ).to(DEVICE)

    state = torch.load(
        SAVE_DIR / filename,
        map_location=DEVICE
    )

    model.load_state_dict(state)
    model.eval()

    return model


print("Loading Centralized model...")
centralized_model = load_model(
    "centralized_torch_model.pth"
)

print("Loading IID model...")
iid_model = load_model(
    "fedavg_iid_adam_5rounds_model.pth"
)

print("Loading Non-IID model...")
noniid_model = load_model(
    "fedavg_noniid_adam_5rounds_model.pth"
)

# ---------------- CHUNK-WISE EVALUATION ----------------

y_true_all = []

centralized_predictions_all = []
iid_predictions_all = []
noniid_predictions_all = []

print("\nStarting final test evaluation...")

for chunk_number, chunk in enumerate(
    pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    X = scaler.transform(
        chunk.drop(columns=["label"])
    ).astype(np.float32)

    y = encoder.transform(chunk["label"])

    X_tensor = torch.tensor(X, device=DEVICE)

    with torch.no_grad():

        centralized_pred = centralized_model(
            X_tensor
        ).argmax(dim=1).cpu().numpy()

        iid_pred = iid_model(
            X_tensor
        ).argmax(dim=1).cpu().numpy()

        noniid_pred = noniid_model(
            X_tensor
        ).argmax(dim=1).cpu().numpy()

    y_true_all.extend(y)
    centralized_predictions_all.extend(centralized_pred)
    iid_predictions_all.extend(iid_pred)
    noniid_predictions_all.extend(noniid_pred)

    print(
        f"Processed chunk {chunk_number} | "
        f"Records: {len(y_true_all)}"
    )

# ---------------- FINAL METRICS ----------------

y_true_all = np.array(y_true_all)

models_predictions = [
    (
        "Centralized PyTorch",
        centralized_predictions_all,
        "final_centralized_test_report.csv"
    ),
    (
        "IID FedAvg",
        iid_predictions_all,
        "final_iid_test_report.csv"
    ),
    (
        "Non-IID FedAvg",
        noniid_predictions_all,
        "final_noniid_test_report.csv"
    )
]

comparison_rows = []

for name, predictions, filename in models_predictions:

    predictions = np.array(predictions)

    print(f"\n========== {name} TEST RESULTS ==========")

    accuracy = accuracy_score(y_true_all, predictions)

    macro_f1 = f1_score(
        y_true_all,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true_all,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    report = classification_report(
        y_true_all,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(RESULTS_DIR / filename)

    comparison_rows.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })

    print("\nClassification Report:")
    print(classification_report(
        y_true_all,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=encoder.classes_,
        zero_division=0
    ))

comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    RESULTS_DIR / "final_model_comparison.csv",
    index=False
)

print("\nSaved all final test reports in:", RESULTS_DIR)
print("\nFinal test evaluation completed!")