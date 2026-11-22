import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent.parent

SAVE_DIR = BASE_DIR / "models" / "saved" / "corrected_pipeline"
RESULTS_DIR = BASE_DIR / "results" / "corrected_pipeline"
GRAPH_DIR = RESULTS_DIR / "graphs"

TEST_PATH = (
    BASE_DIR / "dataset" / "raw" / "CICIoT2023"
    / "CICIOT23" / "test" / "test.csv"
)

CHUNK_SIZE = 10000
DEVICE = torch.device("cpu")

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

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


print("Loading models...")

models = {
    "Centralized": load_model("centralized_torch_model.pth"),
    "IID_FedAvg": load_model("fedavg_iid_adam_5rounds_model.pth"),
    "NonIID_FedAvg": load_model("fedavg_noniid_adam_5rounds_model.pth")
}

# ---------------- INITIALIZE MATRICES ----------------

confusion_matrices = {
    name: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for name in models
}

print("\nStarting chunk-wise confusion matrix generation...")

# ---------------- PROCESS TEST DATA ----------------

for chunk_number, chunk in enumerate(
    pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    X = scaler.transform(
        chunk.drop(columns=["label"])
    ).astype(np.float32)

    y_true = encoder.transform(chunk["label"])

    X_tensor = torch.tensor(X, device=DEVICE)

    with torch.no_grad():

        for name, model in models.items():

            predictions = model(
                X_tensor
            ).argmax(dim=1).cpu().numpy()

            chunk_matrix = confusion_matrix(
                y_true,
                predictions,
                labels=np.arange(NUM_CLASSES)
            )

            confusion_matrices[name] += chunk_matrix

    print(
        f"Processed chunk {chunk_number} | "
        f"Records: {chunk_number * CHUNK_SIZE:,}"
    )

# ---------------- SAVE MATRICES AND GRAPHS ----------------

print("\nGenerating actual confusion matrices...")

for name, matrix in confusion_matrices.items():

    # Save numerical matrix
    matrix_df = pd.DataFrame(
        matrix,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    csv_name = f"{name.lower()}_confusion_matrix.csv"
    matrix_df.to_csv(RESULTS_DIR / csv_name)

    # Normalize each actual-class row for visualization
    row_totals = matrix.sum(axis=1, keepdims=True)

    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0
    )

    plt.figure(figsize=(18, 15))

    sns.heatmap(
        normalized_matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )

    plt.title(f"{name} - Normalized Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)

    plt.tight_layout()

    image_name = f"{name.lower()}_actual_confusion_matrix.png"

    plt.savefig(
        GRAPH_DIR / image_name,
        dpi=200
    )

    plt.close()

    print("Saved:", csv_name)
    print("Saved:", image_name)

print("\nAll actual confusion matrices generated successfully!")