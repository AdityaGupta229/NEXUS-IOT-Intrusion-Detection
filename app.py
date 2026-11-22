import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from flask import Flask, render_template, request, jsonify
from pathlib import Path

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "saved" / "corrected_pipeline"
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}
MAX_FILE_SIZE = 50 * 1024 * 1024

app = Flask(
    __name__,
    template_folder="dashboard/templates",
    static_folder="dashboard/static"
)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

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

scaler = joblib.load(
    MODEL_DIR / "centralized_torch_scaler.joblib"
)

encoder = joblib.load(
    MODEL_DIR / "centralized_torch_encoder.joblib"
)

NUM_FEATURES = len(scaler.feature_names_in_)
NUM_CLASSES = len(encoder.classes_)

EXPECTED_COLUMNS = list(scaler.feature_names_in_)


# ---------------- LOAD TRAINED MODELS ----------------

def load_model(filename):

    model = IntrusionMLP(
        NUM_FEATURES,
        NUM_CLASSES
    ).to(DEVICE)

    state = torch.load(
        MODEL_DIR / filename,
        map_location=DEVICE
    )

    model.load_state_dict(state)
    model.eval()

    return model


models = {
    "Centralized": load_model(
        "centralized_torch_model.pth"
    ),

    "IID FedAvg": load_model(
        "fedavg_iid_adam_5rounds_model.pth"
    ),

    "Non-IID FedAvg": load_model(
        "fedavg_noniid_adam_5rounds_model.pth"
    )
}


# ---------------- HELPER FUNCTIONS ----------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def is_normal_label(label):
    """
    Identify benign traffic labels.
    Handles spaces, underscores, hyphens and letter case.
    """

    normalized = re.sub(
        r"[^a-z0-9]",
        "",
        str(label).strip().casefold()
    )

    return normalized in {
        "benign",
        "benigntraffic"
    }


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("overview.html", active_page="overview")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html", active_page="analysis")


@app.route("/analytics")
def analytics():
    return render_template("analytics.html", active_page="analytics")


@app.route("/performance")
def performance():
    return render_template("performance.html", active_page="performance")


@app.route("/results")
def results():
    return render_template("results.html", active_page="results")


@app.route("/about")
def about():
    return render_template("about.html", active_page="about")


@app.route("/api/health")
def health():

    return jsonify({
        "status": "running",
        "models_loaded": list(models.keys()),
        "number_of_features": NUM_FEATURES,
        "number_of_classes": NUM_CLASSES
    })


# ---------------- PREDICTION API ----------------

@app.route("/api/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "Please upload a CSV file."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Only CSV files are supported."
        }), 400

    model_name = request.form.get(
        "model",
        "Centralized"
    )

    if model_name not in models:
        return jsonify({
            "error": "Invalid model selection."
        }), 400

    try:

        # Read CSV
        df = pd.read_csv(
            file,
            nrows=10001
        )

        if len(df) > 10000:
            return jsonify({
                "error": "Maximum 10,000 records allowed per upload."
            }), 400

        # Remove optional label column
        if "label" in df.columns:
            df = df.drop(columns=["label"])

        # Validate feature columns
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(EXPECTED_COLUMNS)

        if missing or extra:
            return jsonify({
                "error": "CSV columns do not match the training features.",
                "missing_columns": sorted(missing),
                "extra_columns": sorted(extra)
            }), 400

        df = df[EXPECTED_COLUMNS]

        if df.empty:
            return jsonify({
                "error": "CSV contains no records."
            }), 400

        # Validate numeric values
        if not all(
            pd.api.types.is_numeric_dtype(df[col])
            for col in df.columns
        ):
            return jsonify({
                "error": "All feature columns must be numeric."
            }), 400

        if not np.isfinite(
            df.to_numpy(dtype=np.float64)
        ).all():
            return jsonify({
                "error": "CSV contains missing or infinite values."
            }), 400

        # Preprocessing
        X = scaler.transform(df).astype(np.float32)

        X_tensor = torch.tensor(
            X,
            device=DEVICE
        )

        # Select model
        selected_model = models[model_name]

        # Model prediction
        with torch.no_grad():

            logits = selected_model(X_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = (
                probabilities.argmax(dim=1)
                .cpu()
                .numpy()
            )

            confidences = (
                probabilities.max(dim=1)
                .values
                .cpu()
                .numpy()
            )

        # Decode predictions
        labels = encoder.inverse_transform(predictions)

        # Debug output
        print("\n========== PREDICTION DEBUG ==========")

        results = []

        for i, (label, confidence) in enumerate(
            zip(labels, confidences)
        ):

            normalized_label = re.sub(
                r"[^a-z0-9]",
                "",
                str(label).strip().casefold()
            )

            is_normal = is_normal_label(label)

            status = "Normal" if is_normal else "Attack"

            print(
                f"Record {i + 1}: "
                f"Label={str(label)!r}, "
                f"Normalized={normalized_label!r}, "
                f"Status={status}"
            )

            results.append({

                "record": i + 1,

                "prediction": str(label),

                "confidence": round(
                    float(confidence) * 100,
                    2
                ),

                "status": status
            })

        print("======================================\n")

        # Calculate statistics
        normal_count = sum(
            is_normal_label(label)
            for label in labels
        )

        attack_count = len(labels) - normal_count

        counts = pd.Series(labels).value_counts()

        summary = {

            "total_records": len(labels),

            "normal_records": int(normal_count),

            "attack_records": int(attack_count),

            "attack_distribution": {
                str(label): int(count)
                for label, count in counts.items()
            }
        }

        # Return response
        return jsonify({

            "model": model_name,

            "summary": summary,

            "results": results

        })

    except Exception:

        app.logger.exception(
            "Prediction failed"
        )

        return jsonify({
            "error": "Prediction failed. Check CSV format and values."
        }), 400


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )