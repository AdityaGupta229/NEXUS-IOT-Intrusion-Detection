import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

TRAIN_PATH = Path("dataset/processed/train_sample.csv")
VAL_PATH = Path("dataset/processed/validation_sample.csv")
SAVE_DIR = Path("models/saved")

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_val = val_df.drop(columns=["label"])
y_val = val_df["label"]

print("Training records:", len(X_train))

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)

# Balance class contribution during training
weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# Centralized neural network
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    batch_size=512,
    max_iter=15,
    early_stopping=False,
    random_state=42,
    verbose=True
)

print("\nTraining centralized MLP...")

model.fit(
    X_train,
    y_train,
    sample_weight=weights
)

print("\nEvaluating model...")

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0
)

print("\n========== CENTRALIZED MLP RESULTS ==========")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_val,
    predictions,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

joblib.dump(model, SAVE_DIR / "centralized_mlp.joblib")
joblib.dump(scaler, SAVE_DIR / "centralized_mlp_scaler.joblib")
joblib.dump(encoder, SAVE_DIR / "centralized_mlp_encoder.joblib")

print("\nCentralized MLP saved successfully!")