import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import MiniVLADataset
from src.model import MiniVLA


def load_model_and_dataset(project_root, device):
    val_path = project_root / "data" / "processed" / "val.json"
    checkpoint_path = project_root / "outputs" / "checkpoints" / "best_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint["vocab"]

    val_dataset = MiniVLADataset(val_path, vocab=vocab)

    model = MiniVLA(vocab_size=len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, val_dataset


def sign_accuracy(preds, targets, dim, eps=0.05):
    """
    计算某一维方向符号是否正确。
    如果 target 接近 0，则不计入方向判断。
    """
    pred_dim = preds[:, dim]
    target_dim = targets[:, dim]

    valid_mask = target_dim.abs() > eps

    if valid_mask.sum().item() == 0:
        return None, 0

    pred_sign = torch.sign(pred_dim[valid_mask])
    target_sign = torch.sign(target_dim[valid_mask])

    correct = (pred_sign == target_sign).float().mean().item()
    count = valid_mask.sum().item()

    return correct, count


def both_xy_sign_accuracy(preds, targets, eps=0.05):
    """
    计算 x 和 y 方向同时正确的比例。
    如果 target 的 x 或 y 太接近 0，则不计入。
    """
    target_x = targets[:, 0]
    target_y = targets[:, 1]

    valid_mask = (target_x.abs() > eps) & (target_y.abs() > eps)

    if valid_mask.sum().item() == 0:
        return None, 0

    pred_sign = torch.sign(preds[valid_mask, :2])
    target_sign = torch.sign(targets[valid_mask, :2])

    correct = (pred_sign == target_sign).all(dim=1).float().mean().item()
    count = valid_mask.sum().item()

    return correct, count


def success_rate_from_l2(l2_errors, threshold):
    """
    根据 xy 平面的 L2 误差计算成功率。
    如果 l2_error <= threshold，则认为该样本动作预测成功。
    """
    success = (l2_errors <= threshold).float().mean().item()
    return success


def evaluate_metrics(model, dataloader, device):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            instructions = batch["instruction"].to(device)
            targets = batch["action"].to(device)

            preds = model(images, instructions)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    errors = preds - targets

    mse = torch.mean(errors ** 2).item()
    mae_per_dim = torch.mean(torch.abs(errors), dim=0)

    xy_errors = errors[:, :2]
    l2_xy = torch.sqrt(torch.sum(xy_errors ** 2, dim=1))
    mean_l2_xy = torch.mean(l2_xy).item()

    success_rate_03 = success_rate_from_l2(l2_xy, threshold=0.3)
    success_rate_05 = success_rate_from_l2(l2_xy, threshold=0.5)

    x_acc, x_count = sign_accuracy(preds, targets, dim=0)
    y_acc, y_count = sign_accuracy(preds, targets, dim=1)
    both_acc, both_count = both_xy_sign_accuracy(preds, targets)

    metrics = {
        "num_samples": len(targets),
        "mse_all_dims": mse,
        "mae_per_dim": mae_per_dim.tolist(),
        "mean_l2_xy": mean_l2_xy,
        "success_rate_0.3": success_rate_03,
        "success_rate_0.5": success_rate_05,
        "x_direction_accuracy": x_acc,
        "x_direction_count": x_count,
        "y_direction_accuracy": y_acc,
        "y_direction_count": y_count,
        "both_xy_direction_accuracy": both_acc,
        "both_xy_direction_count": both_count,
    }

    return metrics


def main():
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cpu")

    print("Using device:", device)

    model, val_dataset = load_model_and_dataset(project_root, device)

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    metrics = evaluate_metrics(model, val_loader, device)

    print("\nEvaluation Metrics")
    print("=" * 60)
    print(f"Num samples: {metrics['num_samples']}")
    print(f"MSE all dims: {metrics['mse_all_dims']:.6f}")
    print(f"MAE per dim: {metrics['mae_per_dim']}")
    print(f"Mean L2 error on xy: {metrics['mean_l2_xy']:.6f}")
    print(f"Success rate @0.3: {metrics['success_rate_0.3'] * 100:.2f}%")
    print(f"Success rate @0.5: {metrics['success_rate_0.5'] * 100:.2f}%")

    if metrics["x_direction_accuracy"] is not None:
        print(
            f"X direction accuracy: "
            f"{metrics['x_direction_accuracy'] * 100:.2f}% "
            f"({metrics['x_direction_count']} samples)"
        )

    if metrics["y_direction_accuracy"] is not None:
        print(
            f"Y direction accuracy: "
            f"{metrics['y_direction_accuracy'] * 100:.2f}% "
            f"({metrics['y_direction_count']} samples)"
        )

    if metrics["both_xy_direction_accuracy"] is not None:
        print(
            f"Both x/y direction accuracy: "
            f"{metrics['both_xy_direction_accuracy'] * 100:.2f}% "
            f"({metrics['both_xy_direction_count']} samples)"
        )

    save_path = project_root / "outputs" / "eval_metrics.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved metrics to:", save_path)


if __name__ == "__main__":
    main()