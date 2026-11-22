import copy
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configuration
NUM_CLIENTS = 4
ROUNDS = 2
LOCAL_EPOCHS = 1
RANDOM_STATE = 42

CLIENT_DIR = Path("dataset/processed/clients")
VALIDATION_PATH = Path("dataset/processed/validation_sample.csv")
SAVE_DIR = Path("models/saved")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading client datasets...")

client_data = []

for i in range(1, NUM_CLIENTS + 1):
    df = pd.read_csv(CLIENT_DIR / f"client_{i}.csv")

    X = df.drop(columns=["label"])
    y = df["label"]

    client_data.append((X, y))

    print(f"Client {i}: {len(df)} records")

# Prepare common feature scaling and labels
all_X = pd.concat([data[0] for data in client_data])
all_y = pd.concat([data[1] for data in client_data])

scaler = StandardScaler()
scaler.fit(all_X)

label_encoder = LabelEncoder()
label_encoder.fit(all_y)

classes = np.arange(len(label_encoder.classes_))

scaled_clients = []

for X, y in client_data:
    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)
    scaled_clients.append((X_scaled, y_encoded))

print("\nLoading validation data...")

val_df = pd.read_csv(VALIDATION_PATH)

X_val = scaler.transform(val_df.drop(columns=["label"]))
y_val = label_encoder.transform(val_df["label"])

# Initialize global neural network
global_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="sgd",
    learning_rate="constant",
    learning_rate_init=0.01,
    batch_size=512,
    momentum=0.0,
    max_iter=1,
    random_state=RANDOM_STATE
)

# Initialize model weights using a small training batch
X_init, y_init = scaled_clients[0]
global_model.partial_fit(
    X_init[:1000],
    y_init[:1000],
    classes=classes
)

print("\nGlobal model initialized!")

# Federated training
for round_number in range(1, ROUNDS + 1):

    print(f"\n========== FEDERATED ROUND {round_number} ==========")

    client_models = []
    client_sizes = []

    for client_number, (X_client, y_client) in enumerate(
        scaled_clients, start=1
    ):

        local_model = copy.deepcopy(global_model)

        for epoch in range(LOCAL_EPOCHS):
            local_model.partial_fit(X_client, y_client)

        client_models.append(local_model)
        client_sizes.append(len(X_client))

        print(f"Client {client_number} local training completed")

    # FedAvg weighted aggregation
    total_samples = sum(client_sizes)

    new_coefs = []
    new_intercepts = []

    for layer in range(len(global_model.coefs_)):

        avg_coef = sum(
            client_models[i].coefs_[layer] * client_sizes[i]
            for i in range(NUM_CLIENTS)
        ) / total_samples

        avg_intercept = sum(
            client_models[i].intercepts_[layer] * client_sizes[i]
            for i in range(NUM_CLIENTS)
        ) / total_samples

        new_coefs.append(avg_coef)
        new_intercepts.append(avg_intercept)

    global_model.coefs_ = new_coefs
    global_model.intercepts_ = new_intercepts

    # Validation
    predictions = global_model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")

# Final evaluation
predictions = global_model.predict(X_val)

print("\n========== FINAL FEDERATED RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_val, predictions), 4))
print(
    "Macro F1:",
    round(f1_score(y_val, predictions, average="macro", zero_division=0), 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
        labels=classes,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)
print("\n========== PREDICTED CLASS DISTRIBUTION ==========")

predicted_labels = label_encoder.inverse_transform(predictions)

print(
    pd.Series(predicted_labels)
    .value_counts()
    .head(15)
)

print("\n========== ACTUAL VALIDATION DISTRIBUTION ==========")

print(
    val_df["label"]
    .value_counts()
    .head(15)
)

print("\nScikit-learn version:")
import sklearn
print(sklearn.__version__)

# Save artifacts
joblib.dump(global_model, SAVE_DIR / "fedavg_model.joblib")
joblib.dump(scaler, SAVE_DIR / "fedavg_scaler.joblib")
joblib.dump(label_encoder, SAVE_DIR / "fedavg_label_encoder.joblib")

print("\nFederated model and preprocessing files saved!")