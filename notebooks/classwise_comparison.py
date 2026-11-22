import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import classification_report

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")

# Load validation data
val_df = pd.read_csv(
    BASE_DIR / "dataset" / "processed" / "validation_sample.csv"
)

X_val_raw = val_df.drop(columns=["label"])
y_val_raw = val_df["label"]

# Load shared preprocessing objects
scaler = joblib.load(
    SAVE_DIR / "fedavg_noniid_adam_5rounds_scaler.joblib"
)

encoder = joblib.load(
    SAVE_DIR / "fedavg_noniid_adam_5rounds_encoder.joblib"
)

X_val = scaler.transform(X_val_raw).astype(np.float32)
y_val = encoder.transform(y_val_raw)

X_val_tensor = torch.tensor(X_val, device=DEVICE)

NUM_FEATURES = X_val.shape[1]
NUM_CLASSES = len(encoder.classes_)

# Same architecture as both trained models
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


def get_predictions(model_path):

    model = IntrusionMLP(
        NUM_FEATURES,
        NUM_CLASSES
    ).to(DEVICE)

    state = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        outputs = model(X_val_tensor)
        predictions = torch.argmax(
            outputs, dim=1
        ).cpu().numpy()

    return predictions


print("Loading IID model...")
iid_predictions = get_predictions(
    SAVE_DIR / "fedavg_torch_adam_5rounds_model.pth"
)

print("Loading Non-IID model...")
noniid_predictions = get_predictions(
    SAVE_DIR / "fedavg_noniid_adam_5rounds_model.pth"
)

# Classification reports
iid_report = classification_report(
    y_val,
    iid_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

noniid_report = classification_report(
    y_val,
    noniid_predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

# Build comparison table
comparison = pd.DataFrame({
    "Class": encoder.classes_,
    "IID_F1": [
        iid_report[c]["f1-score"] for c in encoder.classes_
    ],
    "NonIID_F1": [
        noniid_report[c]["f1-score"] for c in encoder.classes_
    ],
    "Support": [
        iid_report[c]["support"] for c in encoder.classes_
    ]
})

comparison["F1_Difference"] = (
    comparison["IID_F1"] - comparison["NonIID_F1"]
)

# Save CSV
comparison.to_csv(
    RESULTS_DIR / "classwise_iid_noniid_comparison.csv",
    index=False
)

print("\nCLASS-WISE COMPARISON")
print(comparison.to_string(index=False))

# Plot comparison
plot_data = comparison.sort_values("IID_F1")

positions = np.arange(len(plot_data))
height = 0.38

plt.figure(figsize=(12, 14))

plt.barh(
    positions - height / 2,
    plot_data["IID_F1"],
    height,
    label="IID"
)

plt.barh(
    positions + height / 2,
    plot_data["NonIID_F1"],
    height,
    label="Non-IID"
)

plt.yticks(positions, plot_data["Class"])
plt.xlabel("F1 Score")
plt.ylabel("Attack Class")
plt.title("Class-wise IID vs Non-IID Performance")
plt.xlim(0, 1)
plt.legend()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "classwise_iid_noniid_comparison.png",
    dpi=300
)

plt.show()

print("\nClass-wise comparison completed!")
print("CSV and graph saved in results folder.")