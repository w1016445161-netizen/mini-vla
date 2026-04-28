import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def load_samples(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sample(samples, sample_arg=None):
    if sample_arg is None:
        return random.choice(samples)

    try:
        sample_id = int(sample_arg)
    except ValueError:
        raise ValueError("样本编号必须是整数，例如: python src\\inspect_data.py 17")

    if sample_id < 0 or sample_id >= len(samples):
        raise IndexError(f"样本编号越界，当前数据集共有 {len(samples)} 条样本，可选范围是 0 到 {len(samples)-1}")

    return samples[sample_id]


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "processed" / "dataset.json"

    samples = load_samples(dataset_path)

    sample_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sample = get_sample(samples, sample_arg)

    image_path = project_root / sample["image"]
    image = Image.open(image_path)

    print("Sample info:")
    print(json.dumps(sample, indent=2, ensure_ascii=False))

    plt.figure(figsize=(5, 5))
    plt.imshow(image)

    # 画动作箭头
    agent_x, agent_y = sample["agent_pos"]
    dx, dy, _ = sample["action"]

    # 放大箭头，便于观察
    arrow_scale = 15

    plt.arrow(
        agent_x,
        agent_y,
        dx * arrow_scale,
        dy * arrow_scale,
        color="black",
        width=0.4,
        head_width=2.5,
        length_includes_head=True,
    )

    # 给目标方块加一个文字提示
    target_color = sample["target_color"]
    target_x, target_y = sample["block_positions"][target_color]
    plt.text(
        target_x,
        target_y - 6,
        f"target:{target_color}",
        color="black",
        fontsize=9,
        ha="center",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    plt.title(
        f'id={sample["id"]} | {sample["instruction"]}\n'
        f'action={sample["action"]}'
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()