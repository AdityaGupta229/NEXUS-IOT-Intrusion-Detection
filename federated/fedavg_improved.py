import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------- CONFIGURATION ----------------

SEED = 42
NUM_CLIENTS = 4
ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 512
LEARNING_RATE = 0.001

CLIENT_DIR = Path("dataset/processed/noniid_clients")
TRAIN_PATH = Path("dataset/processed/train_sample.csv")
VAL_PATH = Path("dataset/processed/validation_sample.csv")
SAVE_DIR = Path("models/saved/corrected_pipeline")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

print("Device:", DEVICE)

# ---------------- LOAD TRAINING DATA ----------------

print("\nLoading training sample...")

train_df = pd.read_csv(TRAIN_PATH)

X_train_all = train_df.drop(columns=["label"])
y_train_all = train_df["label"]

scaler = StandardScaler()
scaler.fit(X_train_all)

encoder = LabelEncoder()
encoder.fit(y_train_all)

NUM_FEATURES = X_train_all.shape[1]
NUM_CLASSES = len(encoder.classes_)

print("Features:", NUM_FEATURES)
print("Classes:", NUM_CLASSES)

# Class weights: handle imbalance
encoded_train = encoder.transform(y_train_all)

class_counts = np.bincount(
    encoded_train,
    minlength=NUM_CLASSES
)

class_weights = len(encoded_train) / (
    NUM_CLASSES * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

# ---------------- LOAD CLIENT DATA ----------------

client_loaders = []
client_sizes = []

print("\nLoading 4 clients...")

for i in range(1, NUM_CLIENTS + 1):

    df = pd.read_csv(CLIENT_DIR / f"client_{i}.csv")

    X = scaler.transform(
        df.drop(columns=["label"])
    ).astype(np.float32)

    y = encoder.transform(
        df["label"]
    ).astype(np.int64)

    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    client_loaders.append(loader)
    client_sizes.append(len(dataset))

    print(f"Client {i}: {len(dataset)} records")

# ---------------- VALIDATION DATA ----------------

print("\nLoading validation data...")

val_df = pd.read_csv(VAL_PATH)

X_val = scaler.transform(
    val_df.drop(columns=["label"])
).astype(np.float32)

y_val = encoder.transform(
    val_df["label"]
).astype(np.int64)

X_val_tensor = torch.tensor(X_val, device=DEVICE)

# ---------------- NEURAL NETWORK ----------------

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


global_model = IntrusionMLP(
    NUM_FEATURES,
    NUM_CLASSES
).to(DEVICE)

print("\nGlobal model initialized!")

# ---------------- LOCAL TRAINING ----------------

def train_client(model, loader):

    model.train()

    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    for epoch in range(LOCAL_EPOCHS):

        total_loss = 0.0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"    Local epoch {epoch + 1}, "
            f"loss: {total_loss / len(loader):.4f}"
        )

    return model.state_dict()


# ---------------- FEDAVG AGGREGATION ----------------

def fedavg(client_states, client_sizes):

    total_samples = sum(client_sizes)

    global_state = copy.deepcopy(client_states[0])

    for key in global_state.keys():

        global_state[key] = sum(
            client_states[i][key] * (
                client_sizes[i] / total_samples
            )
            for i in range(NUM_CLIENTS)
        )

    return global_state


# ---------------- EVALUATION ----------------

def evaluate(model):

    model.eval()

    with torch.no_grad():

        outputs = model(X_val_tensor)
        predictions = torch.argmax(
            outputs, dim=1
        ).cpu().numpy()

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    return accuracy, macro_f1, predictions


# ---------------- FEDERATED TRAINING ----------------

history = []

for round_number in range(1, ROUNDS + 1):

    print(
        f"\n========== FEDERATED ROUND "
        f"{round_number} =========="
    )

    client_states = []

    for client_number in range(NUM_CLIENTS):

        local_model = copy.deepcopy(global_model)

        print(f"\nTraining Client {client_number + 1}...")

        local_state = train_client(
            local_model,
            client_loaders[client_number]
        )

        client_states.append(local_state)

        print(
            f"Client {client_number + 1} completed!"
        )

        del local_model

    # Aggregate client updates
    global_model.load_state_dict(
        fedavg(client_states, client_sizes)
    )

    accuracy, macro_f1, _ = evaluate(global_model)

    history.append({
        "round": round_number,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    print("\nServer aggregation completed!")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")

# ---------------- FINAL RESULTS ----------------

accuracy, macro_f1, predictions = evaluate(global_model)

print("\n========== FINAL FEDAVG RESULTS ==========")

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")

print(classification_report(
    y_val,
    predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=encoder.classes_,
    zero_division=0
))

# ---------------- SAVE RESULTS ----------------

torch.save(
    global_model.state_dict(),
    SAVE_DIR / "fedavg_noniid_adam_5rounds_model.pth"
)

joblib.dump(
    scaler,
    SAVE_DIR / "fedavg_noniid_adam_5rounds_scaler.joblib"
)

joblib.dump(
    encoder,
    SAVE_DIR / "fedavg_noniid_adam_5rounds_encoder.joblib"
)

pd.DataFrame(history).to_csv(
    SAVE_DIR / "fedavg_noniid_adam_5rounds_history.csv",
    index=False
)

print("\nModel, preprocessing and history saved!")