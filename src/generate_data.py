import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_SIZE = 64
BLOCK_SIZE = 9
AGENT_RADIUS = 5
NUM_SAMPLES = 1000
ACTION_SCALE = 20.0

COLORS = {
    "red": (220, 50, 50),
    "green": (60, 180, 100),
    "blue": (70, 120, 220),
}

BACKGROUND_COLOR = (245, 245, 245)
AGENT_COLOR = (20, 20, 20)


def clip_action(value):
    return max(-1.0, min(1.0, value))


def compute_action(agent_pos, target_pos):
    """
    根据 agent 位置和目标块位置计算动作。
    这里的动作定义：
    dx = (target_x - agent_x) / ACTION_SCALE
    dy = (target_y - agent_y) / ACTION_SCALE

    因为图像坐标中 y 往下为正，所以 dy > 0 表示向下。
    """
    ax, ay = agent_pos
    tx, ty = target_pos

    dx = clip_action((tx - ax) / ACTION_SCALE)
    dy = clip_action((ty - ay) / ACTION_SCALE)

    # 第三维暂时保留为 0.0，后续可以扩展为 gripper / stop 等动作
    return [round(dx, 4), round(dy, 4), 0.0]


def random_position(margin=10):
    x = random.randint(margin, IMAGE_SIZE - margin)
    y = random.randint(margin, IMAGE_SIZE - margin)
    return [x, y]


def is_too_close(pos_a, pos_b, min_dist=14):
    ax, ay = pos_a
    bx, by = pos_b
    return (ax - bx) ** 2 + (ay - by) ** 2 < min_dist ** 2


def sample_scene_positions():
    """
    随机采样 agent 和三个颜色块的位置。
    简单保证它们不要太近，避免严重重叠。
    """
    positions = []

    while len(positions) < 4:
        pos = random_position()

        if all(not is_too_close(pos, old_pos) for old_pos in positions):
            positions.append(pos)

    agent_pos = positions[0]
    blocks = {
        "red": positions[1],
        "green": positions[2],
        "blue": positions[3],
    }

    return agent_pos, blocks


def draw_scene(agent_pos, blocks, save_path):
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    # 画颜色块
    half = BLOCK_SIZE // 2
    for color_name, pos in blocks.items():
        x, y = pos
        draw.rectangle(
            [x - half, y - half, x + half, y + half],
            fill=COLORS[color_name],
        )

    # 画 agent 黑点
    ax, ay = agent_pos
    r = AGENT_RADIUS
    draw.ellipse(
        [ax - r, ay - r, ax + r, ay + r],
        fill=AGENT_COLOR,
    )

    image.save(save_path)


def main():
    random.seed(42)

    project_root = Path(__file__).resolve().parent.parent
    image_dir = project_root / "data" / "raw" / "images"
    processed_dir = project_root / "data" / "processed"

    image_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    for i in range(NUM_SAMPLES):
        agent_pos, blocks = sample_scene_positions()

        target_color = random.choice(list(COLORS.keys()))
        instruction = f"go to {target_color} block"
        target_pos = blocks[target_color]
        action = compute_action(agent_pos, target_pos)

        image_rel_path = f"data/raw/images/sample_{i:04d}.png"
        image_abs_path = project_root / image_rel_path

        draw_scene(agent_pos, blocks, image_abs_path)

        sample = {
            "id": i,
            "image": image_rel_path,
            "instruction": instruction,
            "target_color": target_color,
            "action": action,
            "agent_pos": agent_pos,
            "blocks": blocks,
        }

        samples.append(sample)

    dataset_path = processed_dir / "dataset.json"

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"Generated {len(samples)} samples.")
    print("Saved dataset to:", dataset_path)
    print("Saved images to:", image_dir)


if __name__ == "__main__":
    main()