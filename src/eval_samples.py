import json
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from src.dataset import MiniVLADataset
from src.model import MiniVLA


NUM_SAMPLES_TO_SAVE = 20
RANDOM_SEED = 42


def denormalize_image(image_tensor):
    image = image_tensor.clone()
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    return image


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


def add_arrow(ax, start_x, start_y, dx, dy, color, linestyle="-", alpha=1.0):
    scale = 20

    arrow = FancyArrowPatch(
        (start_x, start_y),
        (start_x + dx * scale, start_y + dy * scale),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=2,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(arrow)


def draw_and_save_sample(image_tensor, agent_pos, target_action, pred_action, title, save_path):
    image = denormalize_image(image_tensor).permute(1, 2, 0).cpu().numpy()

    agent_x = float(agent_pos[0])
    agent_y = float(agent_pos[1])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    # Target action: blue solid arrow
    add_arrow(
        ax,
        agent_x,
        agent_y,
        float(target_action[0]),
        float(target_action[1]),
        color="blue",
        linestyle="-",
        alpha=1.0,
    )

    # Pred action: orange dashed arrow
    add_arrow(
        ax,
        agent_x,
        agent_y,
        float(pred_action[0]),
        float(pred_action[1]),
        color="orange",
        linestyle="--",
        alpha=0.8,
    )

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, linestyle="-", label="Target Action"),
        Line2D([0], [0], color="orange", lw=2, linestyle="--", label="Pred Action"),
    ]

    ax.set_title(title)
    ax.axis("off")
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cpu")

    output_dir = project_root / "outputs" / "visuals" / "eval_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Saving eval sample visualizations to:", output_dir)

    model, val_dataset = load_model_and_dataset(project_root, device)

    random.seed(RANDOM_SEED)
    sample_indices = random.sample(
        range(len(val_dataset)),
        k=min(NUM_SAMPLES_TO_SAVE, len(val_dataset)),
    )

    summary = []

    with torch.no_grad():
        for rank, idx in enumerate(sample_indices, start=1):
            sample = val_dataset[idx]

            image = sample["image"].unsqueeze(0).to(device)
            instruction = sample["instruction"].unsqueeze(0).to(device)

            target_action = sample["action"]
            pred_action = model(image, instruction).squeeze(0).cpu()

            xy_error = pred_action[:2] - target_action[:2]
            l2_error = torch.sqrt(torch.sum(xy_error ** 2)).item()

            sample_id = sample["id"]
            title = f'{sample["raw_instruction"]} | id={sample_id} | L2={l2_error:.3f}'

            save_path = output_dir / f"eval_sample_{rank:02d}_id_{sample_id}.png"

            draw_and_save_sample(
                image_tensor=sample["image"],
                agent_pos=sample["agent_pos"],
                target_action=target_action,
                pred_action=pred_action,
                title=title,
                save_path=save_path,
            )

            item = {
                "rank": rank,
                "sample_id": int(sample_id),
                "instruction": sample["raw_instruction"],
                "target_color": sample["target_color"],
                "target_action": target_action.tolist(),
                "pred_action": pred_action.tolist(),
                "xy_l2_error": l2_error,
                "image_path": str(save_path),
            }
            summary.append(item)

            print("=" * 60)
            print(f"[{rank:02d}] Sample ID:", sample_id)
            print("Instruction:", sample["raw_instruction"])
            print("Target Color:", sample["target_color"])
            print("Target Action:", target_action.tolist())
            print("Pred Action  :", pred_action.tolist())
            print(f"XY L2 Error  : {l2_error:.4f}")
            print("Saved figure :", save_path)

    summary_path = output_dir / "eval_samples_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nFinished saving eval sample visualizations.")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()