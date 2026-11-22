import pandas as pd
from pathlib import Path

path = Path(
    "models/saved/fedavg_torch_adam_5rounds_history.csv"
)

df = pd.read_csv(path)

print("\n========== FEDAVG TRAINING HISTORY ==========")
print(df.to_string(index=False))

best = df.loc[df["macro_f1"].idxmax()]

print("\n========== BEST ROUND ==========")
print("Round:", int(best["round"]))
print("Accuracy:", round(best["accuracy"], 4))
print("Macro F1:", round(best["macro_f1"], 4))

print("\n========== ROUND IMPROVEMENT ==========")

for i in range(1, len(df)):
    previous = df.iloc[i - 1]
    current = df.iloc[i]

    print(
        f"Round {int(current['round'])}: "
        f"Accuracy change = "
        f"{current['accuracy'] - previous['accuracy']:+.4f}, "
        f"Macro F1 change = "
        f"{current['macro_f1'] - previous['macro_f1']:+.4f}"
    )