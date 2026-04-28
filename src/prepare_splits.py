import json
import random
from pathlib import Path


def main():
    random.seed(42)

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "processed" / "dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    random.shuffle(samples)

    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_path = project_root / "data" / "processed" / "train.json"
    val_path = project_root / "data" / "processed" / "val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Saved train split to: {train_path}")
    print(f"Saved val split to: {val_path}")


if __name__ == "__main__":
    main()