import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SAVED_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load IID and Non-IID training history
iid = pd.read_csv(
    SAVED_DIR / "fedavg_torch_adam_5rounds_history.csv"
)

noniid = pd.read_csv(
    SAVED_DIR / "fedavg_noniid_adam_5rounds_history.csv"
)

print("IID History:")
print(iid)

print("\nNon-IID History:")
print(noniid)

# Accuracy comparison
plt.figure(figsize=(8, 5))
plt.plot(iid["round"], iid["accuracy"], marker="o", label="IID")
plt.plot(noniid["round"], noniid["accuracy"], marker="o", label="Non-IID")
plt.xlabel("Federated Round")
plt.ylabel("Validation Accuracy")
plt.title("IID vs Non-IID Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "iid_vs_noniid_accuracy.png")
plt.show()

# Macro F1 comparison
plt.figure(figsize=(8, 5))
plt.plot(iid["round"], iid["macro_f1"], marker="o", label="IID")
plt.plot(noniid["round"], noniid["macro_f1"], marker="o", label="Non-IID")
plt.xlabel("Federated Round")
plt.ylabel("Validation Macro F1")
plt.title("IID vs Non-IID Macro F1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "iid_vs_noniid_macro_f1.png")
plt.show()

print("\nComparison graphs saved successfully!")