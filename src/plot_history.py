from pathlib import Path
import json

import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parent.parent
    history_path = project_root / "outputs" / "history.json"
    save_path = project_root / "outputs" / "loss_curve.png"

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Loss curve saved to: {save_path}")


if __name__ == "__main__":
    main()