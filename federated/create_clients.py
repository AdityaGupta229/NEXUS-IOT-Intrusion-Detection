import pandas as pd
from pathlib import Path

# Configuration
INPUT_PATH = Path("dataset/processed/train_sample.csv")
OUTPUT_DIR = Path("dataset/processed/clients")

NUM_CLIENTS = 4
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading corrected training sample...")

df = pd.read_csv(INPUT_PATH)

print("Total records:", len(df))
print("Total classes:", df["label"].nunique())

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# Prepare client datasets
client_parts = [[] for _ in range(NUM_CLIENTS)]

# Distribute every class across all clients
for label, group in df.groupby("label"):

    group = group.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    parts = [
        group.iloc[i::NUM_CLIENTS].copy()
        for i in range(NUM_CLIENTS)
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

print("\nTotal records distributed:", total_saved)
print("Original records:", len(df))

assert total_saved == len(df), "Record count mismatch!"

print("\nAll 4 IID clients created successfully!")