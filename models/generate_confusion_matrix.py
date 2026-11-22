import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results" / "corrected_pipeline"
GRAPH_DIR = RESULTS_DIR / "graphs"

# Load classification reports
files = {
    "Centralized": "final_centralized_test_report.csv",
    "IID FedAvg": "final_iid_test_report.csv",
    "Non-IID FedAvg": "final_noniid_test_report.csv"
}

for model_name, filename in files.items():

    report = pd.read_csv(RESULTS_DIR / filename, index_col=0)

    # Extract per-class recall and support
    class_report = report.iloc[:34]

    recalls = class_report["recall"].to_numpy()
    supports = class_report["support"].to_numpy()

    # Build a normalized diagonal performance matrix
    matrix = np.diag(recalls)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=class_report.index,
        yticklabels=class_report.index
    )

    plt.title(f"{model_name} - Per-Class Recall Matrix")
    plt.xlabel("Predicted Class (diagonal only)")
    plt.ylabel("Actual Class")

    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)

    plt.tight_layout()

    output_name = (
        model_name.lower().replace(" ", "_")
        + "_recall_matrix.png"
    )

    plt.savefig(GRAPH_DIR / output_name, dpi=250)
    plt.close()

    print("Saved:", output_name)

print("\nRecall matrices generated successfully!")