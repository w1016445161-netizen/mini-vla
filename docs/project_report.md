# MiniVLA v0.6 项目报告

## 1. 项目名称

**MiniVLA：一个 toy 级 Vision-Language-Action 动作预测系统**

当前版本：

```text
MiniVLA v0.6
```

版本说明：

```text
structured metadata + full validation metrics + success rate + visualization artifacts
```

---

## 2. 项目摘要

本项目基于 PyTorch 实现了一个 toy 级 Vision-Language-Action 系统。项目构建了一个简单的 2D 场景环境，其中包含一个黑色 agent 和 red、green、blue 三个颜色目标块。模型输入为场景图像和自然语言指令，例如 `go to red block`，输出为连续动作向量 `[dx, dy, 0.0]`，用于表示 agent 应该朝目标颜色块移动的方向。

项目完成了从数据生成、Dataset/DataLoader 构建、模型设计、训练验证、checkpoint 保存、loss 曲线分析、全验证集指标评估到样本级可视化的完整闭环。在 v0.6 版本中，项目进一步扩展了数据协议，在每条样本中显式保存 `agent_pos` 和 `blocks` 位置信息，使系统能够自动计算同一张图中不同语言指令对应的 ground-truth action。

实验结果显示，当前模型在验证集上达到 `0.0558` 的 MSE，`80.50%` 的 success rate@0.5，以及 `88.46%` 的 x/y 同时方向正确率。结果表明模型已经具备较强的方向判断能力，但在动作幅度预测和语言-视觉-动作精确对齐方面仍有提升空间。

---

## 3. 项目背景

Vision-Language-Action，简称 VLA，是机器人学习中的一种重要任务形式。它要求模型不仅能够理解图像和语言，还要将视觉与语言信息转化为可执行动作。

真实 VLA 系统通常涉及复杂机器人硬件、大规模数据集、真实环境交互和长时序控制。对于学习阶段而言，直接复现 OpenVLA、RT-1、RT-2 等大型系统难度较高。因此，本项目采用 toy setting，构建一个简化但完整的 VLA 工程闭环。

本项目关注的问题是：

```text
给定一张 2D 场景图像和一条语言指令，模型能否预测 agent 应该朝哪个目标块移动？
```

虽然该任务远小于真实机器人控制任务，但它保留了 VLA 系统的核心结构：

```text
视觉输入 + 语言指令 -> 动作输出
```

通过这个项目，可以理解 VLA 工程中最基础的几部分：

- 数据协议设计
- 多模态样本读取
- 图像和文本编码
- 特征融合
- 连续动作回归
- 模型训练与验证
- 行为级可视化评估
- 实验日志与项目文档整理

---

## 4. 项目目标

本项目的目标不是构建一个真实机器人控制系统，而是实现一个可解释、可训练、可评估、可展示的 mini VLA 工程。

具体目标包括：

1. 自动生成 toy 级 2D VLA 数据。
2. 构建支持图像、语言、动作标签的数据读取接口。
3. 实现一个最小版 Vision-Language-Action 模型。
4. 使用 PyTorch 完成训练、验证和 checkpoint 保存。
5. 记录训练历史并分析 loss 曲线。
6. 使用全验证集指标评估模型能力。
7. 引入 success rate 作为任务成功指标。
8. 固定保存样本级可视化结果。
9. 通过同图多指令实验检查模型是否利用语言条件。
10. 整理 README、experiment log、model card 和项目报告，为后续简历项目包装做准备。

---

## 5. 项目整体流程

项目整体流程如下：

```text
数据生成
  ↓
数据划分
  ↓
Dataset / DataLoader
  ↓
MiniVLA 模型
  ↓
训练与验证
  ↓
保存 best checkpoint
  ↓
记录 history.json
  ↓
绘制 loss 曲线
  ↓
全验证集指标评估
  ↓
样本级可视化
  ↓
同图多指令对照实验
  ↓
实验记录与项目文档
```

对应的核心命令如下：

```bash
python -m src.generate_data
python -m src.prepare_splits
python -m src.train
python -m src.plot_history
python -m src.eval_metrics
python -m src.eval_samples
python -m src.compare_instructions
```

---

## 6. 项目文件结构

当前项目结构如下：

```text
mini_vla/
├─ data/
│  ├─ raw/
│  │  └─ images/
│  └─ processed/
│     ├─ dataset.json
│     ├─ train.json
│     └─ val.json
│
├─ docs/
│  ├─ experiment_log.md
│  ├─ model_card.md
│  └─ project_report.md
│
├─ outputs/
│  ├─ checkpoints/
│  │  └─ best_model.pt
│  ├─ visuals/
│  │  ├─ eval_samples/
│  │  └─ compare_instructions/
│  ├─ eval_metrics.json
│  └─ history.json
│
├─ src/
│  ├─ generate_data.py
│  ├─ prepare_splits.py
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  ├─ plot_history.py
│  ├─ eval_metrics.py
│  ├─ eval_samples.py
│  └─ compare_instructions.py
│
└─ README.md
```

主要文件作用如下：

| 文件 | 作用 |
|---|---|
| `src/generate_data.py` | 自动生成 2D toy VLA 数据 |
| `src/prepare_splits.py` | 将完整数据集划分为训练集和验证集 |
| `src/dataset.py` | 构建 PyTorch Dataset，读取图像、指令、动作和位置元信息 |
| `src/model.py` | 定义 MiniVLA 模型结构 |
| `src/train.py` | 完成模型训练、验证和 checkpoint 保存 |
| `src/plot_history.py` | 根据 `history.json` 绘制 loss 曲线 |
| `src/eval_metrics.py` | 在整个验证集上统计定量指标 |
| `src/eval_samples.py` | 保存普通验证样本的动作预测可视化图 |
| `src/compare_instructions.py` | 保存同图多指令 target/pred 对照图 |
| `docs/experiment_log.md` | 记录实验过程、结果和观察 |
| `docs/model_card.md` | 描述模型用途、结构、指标和限制 |
| `docs/project_report.md` | 当前项目报告 |

---

## 7. 数据设计

### 7.1 场景设置

每个样本是一张 64x64 的 RGB 图像，图像中包含：

- 一个黑色 agent
- 一个红色方块
- 一个绿色方块
- 一个蓝色方块

agent 和三个目标块的位置随机生成，并通过简单距离约束避免严重重叠。

### 7.2 语言指令

当前语言指令采用固定模板：

```text
go to red block
go to green block
go to blue block
```

每条样本随机选择一个目标颜色，形成对应语言指令。

### 7.3 动作标签

动作标签为连续动作向量：

```text
[dx, dy, 0.0]
```

其中：

```text
dx = (target_x - agent_x) / ACTION_SCALE
dy = (target_y - agent_y) / ACTION_SCALE
```

当前使用：

```text
ACTION_SCALE = 20.0
```

动作值会被裁剪到：

```text
[-1.0, 1.0]
```

第三维当前固定为 `0.0`，作为后续扩展 gripper、stop 或其他控制维度的预留接口。

### 7.4 v0.6 数据协议

v0.6 每条样本格式如下：

```json
{
  "id": 0,
  "image": "data/raw/images/sample_0000.png",
  "instruction": "go to red block",
  "target_color": "red",
  "action": [dx, dy, 0.0],
  "agent_pos": [x, y],
  "blocks": {
    "red": [x, y],
    "green": [x, y],
    "blue": [x, y]
  }
}
```

新增 `agent_pos` 和 `blocks` 字段后，系统能够自动计算同一张图中不同语言指令对应的真实动作。这使得同图多指令对照实验更加严谨。

---

## 8. 数据集划分

当前数据规模为 1000 条样本。

| Split | Samples |
|---|---:|
| Full dataset | 1000 |
| Train set | 800 |
| Validation set | 200 |

数据生成命令：

```bash
python -m src.generate_data
```

数据划分命令：

```bash
python -m src.prepare_splits
```

---

## 9. Dataset / DataLoader 设计

`src/dataset.py` 定义了 `MiniVLADataset`，其作用是将原始 json、图片、语言指令和动作标签转换为模型可读取的格式。

每条样本返回一个字典，包括：

```text
image
instruction
action
raw_instruction
target_color
id
agent_pos
blocks
```

其中训练主要使用：

- `image`
- `instruction`
- `action`

评估和可视化还会使用：

- `raw_instruction`
- `target_color`
- `id`
- `agent_pos`
- `blocks`

这种设计使得同一份数据既能用于训练，也能用于后续更复杂的评估和可视化。

---

## 10. 模型设计

MiniVLA 模型定义在：

```text
src/model.py
```

模型输入：

```text
image: [B, 3, 64, 64]
instruction: [B, max_text_len]
```

模型输出：

```text
action: [B, 3]
```

模型主要由四部分组成：

| 模块 | 作用 |
|---|---|
| Image Encoder | 使用小型 CNN 提取图像空间特征 |
| Text Encoder | 使用 Embedding + mean pooling 提取语言指令特征 |
| Fusion Module | 拼接图像特征和文本特征，并通过 MLP 进行融合 |
| Action Head | 输出连续动作向量 |

整体流程为：

```text
image -> CNN -> image feature
instruction -> embedding + mean pooling -> text feature
image feature + text feature -> fusion MLP -> action head -> action
```

该结构体现了一个最小版 VLA 模型的基本思想：

> 先分别编码视觉和语言，再融合两种模态，最后预测动作。

---

## 11. 训练方法

训练脚本为：

```text
src/train.py
```

训练命令：

```bash
python -m src.train
```

训练配置如下：

| 项目 | 设置 |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Loss Function | MSELoss |
| Epochs | 50 |
| Batch Size | 32 |
| Device | CPU |
| Checkpoint | `outputs/checkpoints/best_model.pt` |
| History | `outputs/history.json` |

训练流程包括：

1. 加载训练集和验证集。
2. 创建 MiniVLA 模型。
3. 使用 MSELoss 比较预测动作与真实动作。
4. 使用 Adam 优化器更新模型参数。
5. 每轮训练后在验证集上计算 `val_loss`。
6. 保存验证集表现最好的 `best_model.pt`。
7. 保存 `history.json` 用于后续 loss 曲线分析。

---

## 12. 训练结果

当前 v0.6 最优验证集损失为：

```text
Best val loss = 0.055784
```

训练结果保存位置：

```text
outputs/checkpoints/best_model.pt
outputs/history.json
```

其中：

- `best_model.pt` 保存验证集表现最好的模型参数和词表。
- `history.json` 保存每一轮的训练损失和验证损失。

---

## 13. 评估方法

当前 v0.6 使用两类评估方法。

### 13.1 定量评估

定量评估脚本：

```bash
python -m src.eval_metrics
```

评估指标包括：

- MSE all dims
- MAE per dim
- Mean L2 error on xy
- Success rate @0.3
- Success rate @0.5
- X direction accuracy
- Y direction accuracy
- Both x/y direction accuracy

### 13.2 定性评估

普通样本可视化：

```bash
python -m src.eval_samples
```

输出目录：

```text
outputs/visuals/eval_samples/
```

同图多指令对照可视化：

```bash
python -m src.compare_instructions
```

输出目录：

```text
outputs/visuals/compare_instructions/
```

定性评估用于观察模型行为是否合理，例如：

- 是否朝正确目标移动
- 方向是否正确
- 幅度是否偏小或偏大
- 同一张图下不同语言指令是否会产生不同动作

---

## 14. 定量实验结果

当前 v0.6 在验证集上的指标如下：

| Metric | Value |
|---|---:|
| Num validation samples | 200 |
| Best val loss / MSE | 0.0558 |
| MAE dx | 0.2113 |
| MAE dy | 0.2219 |
| MAE third dim | 0.0078 |
| Mean L2 error on xy | 0.3403 |
| Success rate @0.3 | 55.50% |
| Success rate @0.5 | 80.50% |
| X direction accuracy | 95.21% |
| Y direction accuracy | 93.81% |
| Both x/y direction accuracy | 88.46% |

---

## 15. 结果分析

从当前结果可以看出：

1. 模型整体 MSE 为 0.0558，说明预测动作和真实动作之间的平均误差已经明显降低。
2. dx 和 dy 的 MAE 分别为 0.2113 和 0.2219，说明主要误差集中在平面移动动作上。
3. 第三维 MAE 为 0.0078，说明模型基本学会了第三维接近 0 的规律。
4. xy 平面平均 L2 误差为 0.3403，说明模型已经具备一定动作预测能力，但幅度预测仍不够精确。
5. x 方向正确率为 95.21%，y 方向正确率为 93.81%，说明模型在方向判断上表现较好。
6. x/y 同时方向正确率为 88.46%，说明大部分样本中模型能判断正确移动方向。
7. success rate@0.5 为 80.50%，说明在较宽松阈值下，模型能完成大部分单步动作预测任务。
8. success rate@0.3 为 55.50%，说明在更严格阈值下，模型仍有明显提升空间。
9. 总体来看，当前模型的主要问题不是完全不知道方向，而是动作幅度预测不够稳定。

---

## 16. 可视化实验结果

### 16.1 普通验证样本可视化

脚本：

```bash
python -m src.eval_samples
```

输出目录：

```text
outputs/visuals/eval_samples/
```

该脚本固定保存 20 张验证样本可视化图。

每张图包括：

- 2D 场景图像
- agent 位置
- target action 箭头
- pred action 箭头
- xy L2 error

这些图用于观察模型在普通验证样本上的动作预测能力。

### 16.2 同图多指令对照可视化

脚本：

```bash
python -m src.compare_instructions
```

输出目录：

```text
outputs/visuals/compare_instructions/
```

该脚本固定保存 5 张同图多指令对照图。

每张图对同一场景分别输入：

```text
go to red block
go to green block
go to blue block
```

并展示每个指令下的：

- target action
- pred action
- xy L2 error

该实验用于验证模型是否真正利用语言条件生成动作。

---

## 17. 定性观察

从可视化结果可以观察到：

1. 多数样本中，模型预测箭头与真实箭头方向较接近。
2. 模型经常能判断正确方向，但箭头长度有时偏短或偏长。
3. 在同图多指令实验中，模型输出会随着语言指令变化而变化。
4. 这说明文本条件对模型输出产生了影响。
5. 某些颜色目标或空间布局下仍存在较大 xy L2 error。
6. 当前模型的语言-视觉-动作对齐能力已经初步形成，但还不够稳定。

---

## 18. 项目亮点

当前项目的主要亮点包括：

1. 实现了从数据生成到训练、评估、可视化和文档整理的完整闭环。
2. 支持图像和语言双输入，并输出连续动作向量。
3. 数据协议中显式记录 agent 和目标块位置，便于自动生成多指令 ground truth。
4. 实现了全验证集指标评估，而不是只随机查看几个样本。
5. 引入 success rate，使评估更接近任务成功标准。
6. 实现同图多指令对照实验，用于验证模型是否利用语言条件。
7. 固定保存可视化图片和 summary json，便于复现实验和展示。
8. 整理了 README、experiment log、model card 和 project report，具备较好的项目可读性。

---

## 19. 当前限制

当前 v0.6 仍存在以下限制：

1. 场景是简单 2D toy environment，不是真实机器人场景。
2. 图像分辨率较低，仅为 64x64。
3. 语言模板较单一，只包含 `go to {color} block`。
4. 动作空间较简单，只是单步连续位移预测。
5. 第三维动作当前固定为 0，没有真实控制含义。
6. 模型结构较小，没有使用 Transformer 或预训练视觉模型。
7. 数据集规模只有 1000 条，场景多样性有限。
8. 当前没有真正的闭环仿真执行，只是预测单步动作。
9. success rate 是基于 xy L2 error 的近似成功定义。
10. 模型仍存在动作幅度预测不够精确的问题。

---

## 20. 后续改进方向

后续可以从以下方向继续改进：

1. 增加二维闭环仿真，让 agent 根据预测动作多步移动到目标位置。
2. 扩展语言模板，例如 `move to red block`、`reach the blue block` 等。
3. 增加数据规模，从 1000 条扩展到 3000 或 5000 条。
4. 增加更多场景变化，例如不同大小目标块、不同背景、干扰物体等。
5. 改进模型结构，例如加入更强 CNN、attention 或 Transformer。
6. 增加更严格的 success rate 定义，例如最终距离目标块是否小于阈值。
7. 对错误样本进行分类分析，找出模型失败的常见模式。
8. 将项目整理成 v1.0，形成可写入简历的完整项目。

---

## 21. 与简历项目的关系

当前项目已经具备简历项目的基本形态，因为它不仅包含模型训练，还包含：

- 数据生成
- 数据协议设计
- PyTorch Dataset
- 多模态模型
- 训练与验证
- checkpoint 管理
- 指标评估
- success rate
- 行为可视化
- 同图多指令对照实验
- 文档整理

可以在简历中概括为：

> 基于 PyTorch 实现 toy 级 Vision-Language-Action 系统，完成从 2D 场景数据生成、图文动作建模、训练验证、全验证集指标评估到同图多指令可视化对照的完整闭环。设计包含 agent 与多目标位置的结构化数据协议，实现连续动作回归，并使用 MSE、MAE、xy L2 error、方向准确率和 success rate 系统评估模型能力。

---

## 22. 总结

MiniVLA v0.6 已经完成了一个 toy VLA 项目的核心闭环。模型能够根据图像和语言指令预测 agent 的移动动作，并在验证集上达到较好的方向判断能力。

当前结果表明：

- 模型已经学会一定的语言条件动作预测能力。
- 方向判断较强，但动作幅度预测仍需提升。
- 结构化数据协议显著增强了评估能力。
- success rate 和可视化实验使项目更具展示价值。

总体而言，MiniVLA v0.6 已经从一个简单训练 demo 发展为一个具有数据、模型、训练、评估、可视化和文档体系的小型 VLA 工程项目。下一阶段的重点应放在二维闭环仿真、语言模板扩展和最终 v1.0 项目包装上。