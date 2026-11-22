import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")

TRAIN_PATH = Path("dataset/processed/train_sample.csv")
VAL_PATH = Path("dataset/processed/validation_sample.csv")
SAVE_DIR = Path("models/saved/corrected_pipeline")

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_val = val_df.drop(columns=["label"])
y_val = val_df["label"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train).astype(np.int64)
y_val = encoder.transform(y_val).astype(np.int64)

# Class weights
counts = np.bincount(y_train)
weights = len(y_train) / (len(counts) * counts)

class_weights = torch.tensor(
    weights, dtype=torch.float32
)

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=0
)

X_val_tensor = torch.tensor(X_val)

class IntrusionMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(46, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(encoder.classes_))
        )

    def forward(self, x):
        return self.network(x)


model = IntrusionMLP().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

print("\nStarting centralized PyTorch training...")

for epoch in range(1, 6):

    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()

    with torch.no_grad():
        predictions = model(X_val_tensor).argmax(dim=1).numpy()

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    print(
        f"Epoch {epoch}: "
        f"Loss={total_loss / len(train_loader):.4f}, "
        f"Accuracy={accuracy:.4f}, "
        f"Macro F1={macro_f1:.4f}"
    )

SAVE_DIR.mkdir(parents=True, exist_ok=True)

torch.save(
    model.state_dict(),
    SAVE_DIR / "centralized_torch_model.pth"
)

print("\nCentralized PyTorch model saved!")
joblib.dump(
    scaler,
    SAVE_DIR / "centralized_torch_scaler.joblib"
)

joblib.dump(
    encoder,
    SAVE_DIR / "centralized_torch_encoder.joblib"
)

print("Model, scaler and encoder saved successfully!")