import pandas as pd
import numpy as np

from pathlib import Path

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "raw" / "CICIoT2023"

OUTPUT_PATH = (
    BASE_DIR / "dataset" / "processed" / "validation_sample.csv"
)

CHUNK_SIZE = 20000
MAX_PER_CLASS = 1000
RANDOM_STATE = 42

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------- FIND VALIDATION FILE ----------------

validation_files = list(
    DATASET_PATH.rglob("validation.csv")
)

if not validation_files:
    raise FileNotFoundError("validation.csv not found!")

validation_file = validation_files[0]

print("Reading:", validation_file)
print("Sampling method: Class-wise Random-Key Sampling")

# ---------------- INITIALIZE ----------------

rng = np.random.default_rng(RANDOM_STATE)

samples = {}
total_seen = {}

# ---------------- CHUNK-WISE SAMPLING ----------------

for chunk_no, chunk in enumerate(
    pd.read_csv(validation_file, chunksize=CHUNK_SIZE),
    start=1
):

    chunk = chunk.copy()

    chunk["_sample_key"] = rng.random(len(chunk))

    for label, group in chunk.groupby("label"):

        total_seen[label] = (
            total_seen.get(label, 0) + len(group)
        )

        if label not in samples:
            combined = group
        else:
            combined = pd.concat(
                [samples[label], group],
                ignore_index=True
            )

        samples[label] = combined.nsmallest(
            MAX_PER_CLASS,
            "_sample_key"
        ).copy()

    print(
        f"Processed chunk {chunk_no} | "
        f"Classes: {len(samples)}"
    )

# ---------------- COMBINE SAMPLES ----------------

final_sample = pd.concat(
    samples.values(),
    ignore_index=True
)

final_sample = final_sample.drop(
    columns=["_sample_key"]
)

final_sample = final_sample.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

final_sample.to_csv(
    OUTPUT_PATH,
    index=False
)

# ---------------- SAVE AUDIT ----------------

distribution = pd.DataFrame([
    {
        "Class": label,
        "Original_Records": total_seen[label],
        "Sampled_Records": len(samples[label])
    }
    for label in sorted(total_seen)
])

distribution.to_csv(
    OUTPUT_PATH.parent / "validation_sample_audit.csv",
    index=False
)

# ---------------- FINAL SUMMARY ----------------

print("\n========== VALIDATION SAMPLING COMPLETED ==========")

print("Total original records:", sum(total_seen.values()))
print("Total sampled records:", len(final_sample))
print("Number of classes:", final_sample["label"].nunique())

print("\nClass distribution:")
print(final_sample["label"].value_counts())

print("\nAudit saved at:")
print(OUTPUT_PATH.parent / "validation_sample_audit.csv")

print("\nSample saved at:")
print(OUTPUT_PATH)