import json
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from src.dataset import MiniVLADataset
from src.model import MiniVLA


ACTION_SCALE = 20.0
NUM_SAMPLES_TO_SAVE = 5
RANDOM_SEED = 123


def clip_action(value):
    return max(-1.0, min(1.0, value))


def compute_action(agent_pos, target_pos):
    ax, ay = float(agent_pos[0]), float(agent_pos[1])
    tx, ty = float(target_pos[0]), float(target_pos[1])

    dx = clip_action((tx - ax) / ACTION_SCALE)
    dy = clip_action((ty - ay) / ACTION_SCALE)

    return torch.tensor([dx, dy, 0.0], dtype=torch.float32)


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


def predict_for_instruction(model, dataset, image_tensor, instruction_text, device):
    instruction_tensor = dataset.encode_instruction(instruction_text)

    image = image_tensor.unsqueeze(0).to(device)
    instruction = instruction_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_action = model(image, instruction).squeeze(0).cpu()

    return pred_action


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


def draw_and_save_instruction_comparison(image_tensor, agent_pos, results, title, save_path):
    image = denormalize_image(image_tensor).permute(1, 2, 0).cpu().numpy()

    agent_x = float(agent_pos[0])
    agent_y = float(agent_pos[1])

    color_map = {
        "red": "red",
        "green": "green",
        "blue": "blue",
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    legend_elements = []

    for color_name, item in results.items():
        target_action = item["target_action"]
        pred_action = item["pred_action"]

        draw_color = color_map[color_name]

        # Target action: solid line
        add_arrow(
            ax,
            agent_x,
            agent_y,
            float(target_action[0]),
            float(target_action[1]),
            color=draw_color,
            linestyle="-",
            alpha=1.0,
        )

        # Pred action: dashed line
        add_arrow(
            ax,
            agent_x,
            agent_y,
            float(pred_action[0]),
            float(pred_action[1]),
            color=draw_color,
            linestyle="--",
            alpha=0.6,
        )

        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=draw_color,
                lw=2,
                linestyle="-",
                label=f"{color_name} target",
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=draw_color,
                lw=2,
                linestyle="--",
                label=f"{color_name} pred",
            )
        )

    ax.set_title(title)
    ax.axis("off")
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cpu")

    output_dir = project_root / "outputs" / "visuals" / "compare_instructions"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Saving instruction comparison visualizations to:", output_dir)

    model, val_dataset = load_model_and_dataset(project_root, device)

    colors = ["red", "green", "blue"]

    random.seed(RANDOM_SEED)
    sample_indices = random.sample(
        range(len(val_dataset)),
        k=min(NUM_SAMPLES_TO_SAVE, len(val_dataset)),
    )

    summary = []

    for rank, idx in enumerate(sample_indices, start=1):
        sample = val_dataset[idx]
        image_tensor = sample["image"]
        agent_pos = sample["agent_pos"]
        blocks = sample["blocks"]

        sample_id = sample["id"]

        print("=" * 70)
        print(f"[{rank:02d}] Sample ID:", sample_id)
        print("Original instruction:", sample["raw_instruction"])
        print("Original target color:", sample["target_color"])
        print("Agent pos:", agent_pos.tolist())
        print("Blocks:")
        for color in colors:
            print(f"  {color}: {blocks[color].tolist()}")

        results = {}
        sample_record = {
            "rank": rank,
            "sample_id": int(sample_id),
            "original_instruction": sample["raw_instruction"],
            "original_target_color": sample["target_color"],
            "agent_pos": agent_pos.tolist(),
            "blocks": {color: blocks[color].tolist() for color in colors},
            "results": {},
        }

        for color in colors:
            instruction_text = f"go to {color} block"

            pred_action = predict_for_instruction(
                model=model,
                dataset=val_dataset,
                image_tensor=image_tensor,
                instruction_text=instruction_text,
                device=device,
            )

            target_action = compute_action(agent_pos, blocks[color])
            error_xy = pred_action[:2] - target_action[:2]
            l2_error = torch.sqrt(torch.sum(error_xy ** 2)).item()

            results[color] = {
                "instruction": instruction_text,
                "target_action": target_action,
                "pred_action": pred_action,
                "l2_error": l2_error,
            }

            sample_record["results"][color] = {
                "instruction": instruction_text,
                "target_action": target_action.tolist(),
                "pred_action": pred_action.tolist(),
                "xy_l2_error": l2_error,
            }

            print(f"\nInstruction: {instruction_text}")
            print("  Target action:", target_action.tolist())
            print("  Pred action  :", pred_action.tolist())
            print(f"  XY L2 error  : {l2_error:.4f}")

        save_path = output_dir / f"compare_{rank:02d}_id_{sample_id}.png"
        title = f"Same image, target vs pred | id={sample_id}"

        draw_and_save_instruction_comparison(
            image_tensor=image_tensor,
            agent_pos=agent_pos,
            results=results,
            title=title,
            save_path=save_path,
        )

        sample_record["image_path"] = str(save_path)
        summary.append(sample_record)

        print("Saved figure:", save_path)

    summary_path = output_dir / "compare_instructions_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nFinished saving instruction comparison visualizations.")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()