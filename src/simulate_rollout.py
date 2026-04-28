import json
import math
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from src.dataset import MiniVLADataset
from src.model import MiniVLA


IMAGE_SIZE = 64
BLOCK_SIZE = 9
AGENT_RADIUS = 5

ACTION_SCALE = 20.0
EXECUTION_SCALE = 6.0

MAX_STEPS = 20

# 严格成功半径
PRIMARY_SUCCESS_RADIUS = 6.0

# 多阈值评估半径
SUCCESS_RADII = [6.0, 8.0, 10.0]

# v0.7.3 新增：接近目标后停止，避免继续漂移
STOP_RADIUS = 6.0
NO_IMPROVE_PATIENCE = 5
MIN_IMPROVEMENT = 0.1

NUM_ROLLOUTS = 20
RANDOM_SEED = 2026

COLORS = {
    "red": (220, 50, 50),
    "green": (60, 180, 100),
    "blue": (70, 120, 220),
}

BACKGROUND_COLOR = (245, 245, 245)
AGENT_COLOR = (20, 20, 20)
PATH_COLOR = (255, 140, 0)
TARGET_RING_COLOR = (0, 0, 0)


def radius_key(radius):
    if float(radius).is_integer():
        return str(int(radius))
    return str(radius)


def clip(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def distance(pos_a, pos_b):
    ax, ay = pos_a
    bx, by = pos_b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def render_scene(
    agent_pos,
    blocks,
    target_color=None,
    path=None,
    draw_target_ring=False,
    draw_path=False,
):
    """
    渲染 2D 场景。

    模型输入时使用干净图像：
    - 只画 agent
    - 只画 red/green/blue blocks

    可视化保存时才画：
    - target ring
    - rollout path
    """
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    half = BLOCK_SIZE // 2

    for color_name, pos in blocks.items():
        x, y = float(pos[0]), float(pos[1])
        draw.rectangle(
            [x - half, y - half, x + half, y + half],
            fill=COLORS[color_name],
        )

    if draw_target_ring and target_color is not None:
        tx, ty = blocks[target_color]
        tx, ty = float(tx), float(ty)
        draw.rectangle(
            [tx - half - 2, ty - half - 2, tx + half + 2, ty + half + 2],
            outline=TARGET_RING_COLOR,
            width=2,
        )

    if draw_path and path is not None and len(path) >= 2:
        path_points = [(float(x), float(y)) for x, y in path]
        draw.line(path_points, fill=PATH_COLOR, width=2)

        for px, py in path_points:
            r = 2
            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=PATH_COLOR,
            )

    ax, ay = float(agent_pos[0]), float(agent_pos[1])
    r = AGENT_RADIUS
    draw.ellipse(
        [ax - r, ay - r, ax + r, ay + r],
        fill=AGENT_COLOR,
    )

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


def predict_action(model, dataset, scene_image, instruction_text, device):
    image_tensor = dataset.transform(scene_image).unsqueeze(0).to(device)
    instruction_tensor = dataset.encode_instruction(instruction_text).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_action = model(image_tensor, instruction_tensor).squeeze(0).cpu()

    return pred_action


def update_agent_pos(agent_pos, pred_action):
    """
    根据模型预测动作更新 agent 位置。

    ACTION_SCALE 是训练标签归一化尺度。
    EXECUTION_SCALE 是闭环执行时每一步实际移动像素尺度。
    """
    x, y = float(agent_pos[0]), float(agent_pos[1])

    dx = clip(float(pred_action[0]), -1.0, 1.0)
    dy = clip(float(pred_action[1]), -1.0, 1.0)

    new_x = x + dx * EXECUTION_SCALE
    new_y = y + dy * EXECUTION_SCALE

    new_x = clip(new_x, 0, IMAGE_SIZE - 1)
    new_y = clip(new_y, 0, IMAGE_SIZE - 1)

    return [new_x, new_y]


def success_dict_from_distance(dist):
    result = {}
    for radius in SUCCESS_RADII:
        key = radius_key(radius)
        result[f"success@{key}"] = dist <= radius
    return result


def simulate_one_rollout(model, dataset, sample, device, save_dir, rank):
    sample_id = sample["id"]
    instruction_text = sample["raw_instruction"]
    target_color = sample["target_color"]

    agent_pos = [float(sample["agent_pos"][0]), float(sample["agent_pos"][1])]

    blocks = {
        color: [float(pos[0]), float(pos[1])]
        for color, pos in sample["blocks"].items()
    }

    target_pos = blocks[target_color]

    initial_distance = distance(agent_pos, target_pos)

    best_distance = initial_distance
    best_step = 0
    best_agent_pos = agent_pos.copy()

    path = [agent_pos.copy()]
    step_records = []
    distance_history = []

    primary_success = False
    stopped_by_near_target = False
    stopped_by_no_improvement = False
    stopped_by_max_steps = False
    stop_reason = "unknown"

    final_distance = initial_distance

    no_improve_count = 0

    frames = []

    for step in range(MAX_STEPS + 1):
        current_distance = distance(agent_pos, target_pos)

        distance_history.append(
            {
                "step": step,
                "agent_pos": agent_pos.copy(),
                "distance_to_target": current_distance,
            }
        )

        if current_distance < best_distance:
            best_distance = current_distance
            best_step = step
            best_agent_pos = agent_pos.copy()

        visual_image = render_scene(
            agent_pos=agent_pos,
            blocks=blocks,
            target_color=target_color,
            path=path,
            draw_target_ring=True,
            draw_path=True,
        )
        frames.append(visual_image)

        # 严格成功：进入 primary radius
        if current_distance <= PRIMARY_SUCCESS_RADIUS:
            primary_success = True
            final_distance = current_distance
            stop_reason = "primary_success"
            break

        # v0.7.3 新增：接近目标即可停止，避免继续漂移
        if current_distance <= STOP_RADIUS:
            stopped_by_near_target = True
            final_distance = current_distance
            stop_reason = "near_target_stop"
            break

        if step == MAX_STEPS:
            stopped_by_max_steps = True
            final_distance = current_distance
            stop_reason = "max_steps"
            break

        # 模型输入必须是干净图像，不画路径和目标框
        model_input_image = render_scene(
            agent_pos=agent_pos,
            blocks=blocks,
            target_color=None,
            path=None,
            draw_target_ring=False,
            draw_path=False,
        )

        pred_action = predict_action(
            model=model,
            dataset=dataset,
            scene_image=model_input_image,
            instruction_text=instruction_text,
            device=device,
        )

        next_agent_pos = update_agent_pos(agent_pos, pred_action)
        next_distance = distance(next_agent_pos, target_pos)

        # 判断这一步是否相对历史最好距离有明显改进
        if next_distance < best_distance - MIN_IMPROVEMENT:
            no_improve_count = 0
        else:
            no_improve_count += 1

        step_records.append(
            {
                "step": step,
                "agent_pos": agent_pos.copy(),
                "pred_action": pred_action.tolist(),
                "next_agent_pos": next_agent_pos.copy(),
                "distance_to_target": current_distance,
                "next_distance_to_target": next_distance,
                "best_distance_before_step": best_distance,
                "no_improve_count_after_step": no_improve_count,
            }
        )

        agent_pos = next_agent_pos
        path.append(agent_pos.copy())

        # v0.7.3 新增：连续多步没有明显变好，则停止
        if no_improve_count >= NO_IMPROVE_PATIENCE:
            stopped_by_no_improvement = True
            final_distance = distance(agent_pos, target_pos)
            stop_reason = "no_improvement_stop"

            # 更新最后位置是否是 best
            if final_distance < best_distance:
                best_distance = final_distance
                best_step = step + 1
                best_agent_pos = agent_pos.copy()

            # 加一帧最终停止位置
            visual_image = render_scene(
                agent_pos=agent_pos,
                blocks=blocks,
                target_color=target_color,
                path=path,
                draw_target_ring=True,
                draw_path=True,
            )
            frames.append(visual_image)

            break

    # 循环结束后再确认 final_distance 和 best_distance
    final_distance = distance(agent_pos, target_pos)

    if final_distance < best_distance:
        best_distance = final_distance
        best_step = len(path) - 1
        best_agent_pos = agent_pos.copy()

    final_success_by_radius = success_dict_from_distance(final_distance)
    best_success_by_radius = success_dict_from_distance(best_distance)

    final_image = render_scene(
        agent_pos=agent_pos,
        blocks=blocks,
        target_color=target_color,
        path=path,
        draw_target_ring=True,
        draw_path=True,
    )

    status = "success" if primary_success else "fail"
    final_png_path = save_dir / f"rollout_{rank:02d}_id_{sample_id}_{status}.png"
    final_image.save(final_png_path)

    gif_path = save_dir / f"rollout_{rank:02d}_id_{sample_id}_{status}.gif"
    if len(frames) > 1:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0,
        )
    else:
        frames[0].save(gif_path)

    distance_reduction = initial_distance - final_distance
    improved = final_distance < initial_distance

    best_distance_reduction = initial_distance - best_distance

    record = {
        "rank": rank,
        "sample_id": int(sample_id),
        "instruction": instruction_text,
        "target_color": target_color,
        "initial_agent_pos": path[0],
        "final_agent_pos": agent_pos,
        "best_agent_pos": best_agent_pos,
        "target_pos": target_pos,
        "primary_success": primary_success,
        "primary_success_radius": PRIMARY_SUCCESS_RADIUS,
        "stop_radius": STOP_RADIUS,
        "stop_reason": stop_reason,
        "stopped_by_near_target": stopped_by_near_target,
        "stopped_by_no_improvement": stopped_by_no_improvement,
        "stopped_by_max_steps": stopped_by_max_steps,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "best_distance": best_distance,
        "best_step": best_step,
        "distance_reduction": distance_reduction,
        "best_distance_reduction": best_distance_reduction,
        "improved": improved,
        "num_steps_executed": len(step_records),
        "execution_scale": EXECUTION_SCALE,
        "no_improve_patience": NO_IMPROVE_PATIENCE,
        "min_improvement": MIN_IMPROVEMENT,
        "final_success_by_radius": final_success_by_radius,
        "best_success_by_radius": best_success_by_radius,
        "path": path,
        "distance_history": distance_history,
        "steps": step_records,
        "final_png_path": str(final_png_path),
        "gif_path": str(gif_path),
    }

    return record


def aggregate_success_rates(records, success_key):
    """
    success_key:
    - final_success_by_radius
    - best_success_by_radius
    """
    rates = {}
    for radius in SUCCESS_RADII:
        key = radius_key(radius)
        metric_name = f"success@{key}"
        count = sum(1 for r in records if r[success_key][metric_name])
        rate = count / len(records) if records else 0.0
        rates[metric_name] = {
            "count": count,
            "rate": rate,
        }
    return rates


def count_stop_reasons(records):
    stop_reasons = {}

    for record in records:
        reason = record["stop_reason"]
        if reason not in stop_reasons:
            stop_reasons[reason] = 0
        stop_reasons[reason] += 1

    return stop_reasons


def main():
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cpu")

    output_dir = project_root / "outputs" / "rollouts_v074"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Saving rollout results to:", output_dir)
    print("Execution scale:", EXECUTION_SCALE)
    print("Max steps:", MAX_STEPS)
    print("Primary success radius:", PRIMARY_SUCCESS_RADIUS)
    print("Stop radius:", STOP_RADIUS)
    print("No-improve patience:", NO_IMPROVE_PATIENCE)
    print("Min improvement:", MIN_IMPROVEMENT)
    print("Evaluation radii:", SUCCESS_RADII)

    model, val_dataset = load_model_and_dataset(project_root, device)

    random.seed(RANDOM_SEED)
    sample_indices = random.sample(
        range(len(val_dataset)),
        k=min(NUM_ROLLOUTS, len(val_dataset)),
    )

    records = []

    for rank, idx in enumerate(sample_indices, start=1):
        sample = val_dataset[idx]

        record = simulate_one_rollout(
            model=model,
            dataset=val_dataset,
            sample=sample,
            device=device,
            save_dir=output_dir,
            rank=rank,
        )

        records.append(record)

        print("=" * 70)
        print(f"[{rank:02d}] Sample ID:", record["sample_id"])
        print("Instruction:", record["instruction"])
        print("Target color:", record["target_color"])
        print("Initial agent pos:", record["initial_agent_pos"])
        print("Target pos:", record["target_pos"])
        print("Final agent pos:", record["final_agent_pos"])
        print("Best agent pos:", record["best_agent_pos"])
        print(f"Initial distance: {record['initial_distance']:.3f}")
        print(f"Final distance  : {record['final_distance']:.3f}")
        print(f"Best distance   : {record['best_distance']:.3f}")
        print("Best step:", record["best_step"])
        print(f"Distance reduction: {record['distance_reduction']:.3f}")
        print(f"Best distance reduction: {record['best_distance_reduction']:.3f}")
        print("Improved:", record["improved"])
        print("Primary success:", record["primary_success"])
        print("Stop reason:", record["stop_reason"])
        print("Final success by radius:", record["final_success_by_radius"])
        print("Best success by radius :", record["best_success_by_radius"])
        print("Steps executed:", record["num_steps_executed"])
        print("Saved PNG:", record["final_png_path"])
        print("Saved GIF:", record["gif_path"])

    num_primary_success = sum(1 for r in records if r["primary_success"])
    num_improved = sum(1 for r in records if r["improved"])

    primary_success_rate = num_primary_success / len(records) if records else 0.0
    improvement_rate = num_improved / len(records) if records else 0.0

    avg_initial_distance = (
        sum(r["initial_distance"] for r in records) / len(records)
        if records
        else 0.0
    )
    avg_final_distance = (
        sum(r["final_distance"] for r in records) / len(records)
        if records
        else 0.0
    )
    avg_best_distance = (
        sum(r["best_distance"] for r in records) / len(records)
        if records
        else 0.0
    )
    avg_distance_reduction = (
        sum(r["distance_reduction"] for r in records) / len(records)
        if records
        else 0.0
    )
    avg_best_distance_reduction = (
        sum(r["best_distance_reduction"] for r in records) / len(records)
        if records
        else 0.0
    )
    avg_steps_executed = (
        sum(r["num_steps_executed"] for r in records) / len(records)
        if records
        else 0.0
    )

    final_success_rates = aggregate_success_rates(records, "final_success_by_radius")
    best_success_rates = aggregate_success_rates(records, "best_success_by_radius")
    stop_reasons = count_stop_reasons(records)

    summary = {
        "num_rollouts": len(records),
        "num_primary_success": num_primary_success,
        "primary_success_rate": primary_success_rate,
        "num_improved": num_improved,
        "improvement_rate": improvement_rate,
        "primary_success_radius": PRIMARY_SUCCESS_RADIUS,
        "stop_radius": STOP_RADIUS,
        "success_radii": SUCCESS_RADII,
        "max_steps": MAX_STEPS,
        "execution_scale": EXECUTION_SCALE,
        "no_improve_patience": NO_IMPROVE_PATIENCE,
        "min_improvement": MIN_IMPROVEMENT,
        "avg_initial_distance": avg_initial_distance,
        "avg_final_distance": avg_final_distance,
        "avg_best_distance": avg_best_distance,
        "avg_distance_reduction": avg_distance_reduction,
        "avg_best_distance_reduction": avg_best_distance_reduction,
        "avg_steps_executed": avg_steps_executed,
        "final_success_rates": final_success_rates,
        "best_success_rates": best_success_rates,
        "stop_reasons": stop_reasons,
        "records": records,
    }

    summary_path = output_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nRollout Summary")
    print("=" * 70)
    print(f"Num rollouts: {len(records)}")
    print(f"Primary success count: {num_primary_success}")
    print(f"Primary success rate : {primary_success_rate * 100:.2f}%")
    print(f"Num improved: {num_improved}")
    print(f"Improvement rate: {improvement_rate * 100:.2f}%")
    print(f"Avg initial distance: {avg_initial_distance:.3f}")
    print(f"Avg final distance  : {avg_final_distance:.3f}")
    print(f"Avg best distance   : {avg_best_distance:.3f}")
    print(f"Avg distance reduction     : {avg_distance_reduction:.3f}")
    print(f"Avg best distance reduction: {avg_best_distance_reduction:.3f}")
    print(f"Avg steps executed: {avg_steps_executed:.2f}")

    print("\nFinal success rates:")
    for metric_name, item in final_success_rates.items():
        print(
            f"  {metric_name}: "
            f"{item['rate'] * 100:.2f}% "
            f"({item['count']}/{len(records)})"
        )

    print("\nBest success rates:")
    for metric_name, item in best_success_rates.items():
        print(
            f"  {metric_name}: "
            f"{item['rate'] * 100:.2f}% "
            f"({item['count']}/{len(records)})"
        )

    print("\nStop reasons:")
    for reason, count in stop_reasons.items():
        print(f"  {reason}: {count}")

    print("\nSummary saved to:", summary_path)


if __name__ == "__main__":
    main()