from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import MiniVLADataset
from src.model import MiniVLA


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "processed" / "dataset.json"

    dataset = MiniVLADataset(dataset_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    images = batch["image"]
    instructions = batch["instruction"]
    targets = batch["action"]

    model = MiniVLA(vocab_size=len(dataset.vocab))
    preds = model(images, instructions)

    print("输入检查:")
    print("images shape:", images.shape)
    print("instructions shape:", instructions.shape)
    print("targets shape:", targets.shape)

    print("\n模型输出检查:")
    print("preds shape:", preds.shape)
    print("preds:", preds)

    criterion = nn.MSELoss()
    loss = criterion(preds, targets)

    print("\nloss 检查:")
    print("mse loss:", loss.item())


if __name__ == "__main__":
    main()