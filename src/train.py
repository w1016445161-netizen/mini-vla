import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import MiniVLADataset
from src.model import MiniVLA


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            instructions = batch["instruction"].to(device)
            targets = batch["action"].to(device)

            preds = model(images, instructions)
            loss = criterion(preds, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / total_count


def main():
    project_root = Path(__file__).resolve().parent.parent

    train_path = project_root / "data" / "processed" / "train.json"
    val_path = project_root / "data" / "processed" / "val.json"

    checkpoint_dir = project_root / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = project_root / "outputs" / "history.json"

    device = torch.device("cpu")
    print("Using device:", device)

    # 数据集
    train_dataset = MiniVLADataset(train_path)
    val_dataset = MiniVLADataset(val_path, vocab=train_dataset.vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型
    model = MiniVLA(vocab_size=len(train_dataset.vocab))
    model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练配置
    num_epochs = 50
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_train_loss = 0.0
        total_train_count = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            instructions = batch["instruction"].to(device)
            targets = batch["action"].to(device)

            optimizer.zero_grad()

##先清空旧梯度。

            preds = model(images, instructions)

#模型做一次预测。

            loss = criterion(preds, targets)

#计算预测错了多少。

            loss.backward()

#计算每个参数该怎么调整。

            optimizer.step()

#真正更新模型参数。

            batch_size = images.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_count += batch_size

        avg_train_loss = total_train_loss / total_train_count
        avg_val_loss = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": train_dataset.vocab,
                    "best_val_loss": best_val_loss,
                },
                best_path,
            )
            print(f"  -> Saved new best model to: {best_path}")

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nTraining finished.")
    print("Best val loss:", best_val_loss)
    print("History saved to:", history_path)


if __name__ == "__main__":
    main()