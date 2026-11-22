import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
INPUT_PATH = Path("dataset/processed/train_sample.csv")
OUTPUT_DIR = Path("dataset/processed/noniid_clients")

NUM_CLIENTS = 4
ALPHA = 0.5
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_STATE)

print("Loading corrected training sample...")

df = pd.read_csv(INPUT_PATH)

print("Total records:", len(df))
print("Total classes:", df["label"].nunique())

client_parts = [[] for _ in range(NUM_CLIENTS)]

# Dirichlet-based Non-IID distribution
for label, group in df.groupby("label"):

    group = group.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    proportions = rng.dirichlet(
        np.repeat(ALPHA, NUM_CLIENTS)
    )

    split_points = (
        np.cumsum(proportions)[:-1] * len(group)
    ).astype(int)

    boundaries = [0] + split_points.tolist() + [len(group)]

    parts = [
        group.iloc[start:end].copy()
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    for i in range(NUM_CLIENTS):
        client_parts[i].append(parts[i])

# Save clients
total_saved = 0

for i, parts in enumerate(client_parts, start=1):

    client_df = pd.concat(parts, ignore_index=True)

    client_df = client_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    output_file = OUTPUT_DIR / f"client_{i}.csv"

    client_df.to_csv(output_file, index=False)

    total_saved += len(client_df)

    print(f"\nClient {i}")
    print("Records:", len(client_df))
    print("Classes:", client_df["label"].nunique())
    print("Saved:", output_file)

    print("\nClass distribution:")
    print(client_df["label"].value_counts().to_string())

print("\nTotal records distributed:", total_saved)
print("Original records:", len(df))

assert total_saved == len(df), "Record count mismatch!"

print("\nAll 4 Non-IID clients created successfully!")