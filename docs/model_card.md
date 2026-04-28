# MiniVLA Model Card

## 1. 模型名称

**MiniVLA v0.7.4**

完整版本说明：

```text
MiniVLA v0.7.4: structured metadata + full validation metrics + success rate + closed-loop rollout + tuned stop strategy
```

MiniVLA 是一个教学型 toy Vision-Language-Action 模型，用于学习和展示 VLA 工程中的基本流程，包括数据生成、Dataset 构建、模型训练、动作预测、指标评估、可视化分析和二维闭环 rollout 仿真。

---

## 2. 模型概述

MiniVLA v0.7.4 是一个基于 PyTorch 实现的小型多模态动作预测模型。

模型输入包括：

1. 一张 2D 场景图像。
2. 一条自然语言指令。

模型输出为：

```text
[dx, dy, 0.0]
```

其中：

- `dx` 表示 agent 在 x 方向上的移动动作。
- `dy` 表示 agent 在 y 方向上的移动动作。
- 第三维当前固定接近 `0.0`，作为后续扩展 gripper、stop 或其他动作维度的预留接口。

模型任务可以概括为：

```text
image + instruction -> continuous action
```

例如：

```text
Input image: 一张包含黑色 agent 和红、绿、蓝三个目标块的 2D 图像
Instruction: go to red block
Output action: agent 应该朝红色块移动的连续动作向量
```

从 v0.7 开始，MiniVLA 不再只进行单步动作预测评估，而是进一步加入二维 closed-loop rollout 仿真，使模型能够作为一个简单 policy 连续执行多步动作。

闭环执行流程为：

```text
current image + instruction
  -> predicted action
  -> update agent position
  -> render next image
  -> predict next action
```

---

## 3. 模型用途

MiniVLA v0.7.4 主要用于以下用途：

1. 学习 Vision-Language-Action 工程的基本结构。
2. 理解图像、语言和动作之间的数据流关系。
3. 构建一个可训练、可评估、可可视化的 toy VLA baseline。
4. 学习 PyTorch 中 Dataset、DataLoader、model、train、eval 的完整流程。
5. 学习如何设计 VLA 数据协议和实验指标。
6. 学习如何进行同图多指令对照实验。
7. 学习如何将单步动作预测扩展为二维闭环 rollout。
8. 作为个人简历项目中的一个小型多模态动作预测工程。

本模型不适用于真实机器人控制，也不应直接用于物理环境中的实际部署。

---

## 4. 不适用场景

MiniVLA v0.7.4 不适合以下用途：

1. 真实机器人控制。
2. 高精度连续控制任务。
3. 复杂场景下的长时序规划。
4. 多步骤真实任务执行。
5. 真实机械臂抓取或导航任务。
6. 开放词汇自然语言理解。
7. 复杂视觉场景中的目标识别和泛化。
8. 无人监督的实际物理环境部署。

当前模型只是一个可控 toy setting 下的教学型 VLA baseline。

---

## 5. 输入与输出

### 5.1 图像输入

图像输入为 64x64 的 RGB 场景图：

```text
image: [B, 3, 64, 64]
```

图像中包含：

- 一个黑色 agent
- 一个红色目标块
- 一个绿色目标块
- 一个蓝色目标块

图像会经过以下预处理：

```text
PIL image -> ToTensor -> Normalize(mean=0.5, std=0.5)
```

---

### 5.2 语言输入

语言输入为简单模板指令：

```text
go to red block
go to green block
go to blue block
```

文本会被转换为 token 编号，并补齐到固定长度：

```text
instruction: [B, max_text_len]
```

当前语言模板较简单，主要用于验证模型是否能够根据颜色指令选择不同目标。

---

### 5.3 动作输出

模型输出为连续动作向量：

```text
action: [B, 3]
```

动作格式为：

```text
[dx, dy, 0.0]
```

其中：

- `dx > 0` 表示向右移动。
- `dx < 0` 表示向左移动。
- `dy > 0` 表示向下移动。
- `dy < 0` 表示向上移动。
- 第三维当前固定为 `0.0`。

---

## 6. 模型结构

MiniVLA v0.7.4 由四个主要模块组成。

| Module | Description |
|---|---|
| Image Encoder | 小型 CNN，用于从 2D 场景图像中提取视觉特征 |
| Text Encoder | Embedding + mean pooling，用于将语言指令编码为文本特征 |
| Fusion Module | 将图像特征和文本特征拼接后，通过 MLP 进行融合 |
| Action Head | 将融合特征映射为连续动作向量 |

整体结构如下：

```text
image -> CNN image encoder -> image feature
instruction -> embedding + mean pooling -> text feature
image feature + text feature -> fusion MLP -> action head -> action
```

模型核心思想是：

```text
先分别理解图像和语言，再将二者融合，最后预测动作。
```

---

## 7. 训练数据

### 7.1 数据集规模

当前 v0.7.4 使用自动生成的 toy 数据集。

| Split | Samples |
|---|---:|
| Full dataset | 1000 |
| Train set | 800 |
| Validation set | 200 |

---

### 7.2 数据生成方式

数据由以下脚本自动生成：

```bash
python -m src.generate_data
```

每个样本包含：

- 图像路径
- 语言指令
- 目标颜色
- 动作标签
- agent 位置
- red、green、blue 三个目标块的位置

样本格式如下：

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

---

### 7.3 动作标签

动作标签根据 agent 和目标块之间的位置差计算得到：

```text
dx = (target_x - agent_x) / ACTION_SCALE
dy = (target_y - agent_y) / ACTION_SCALE
```

其中：

```text
ACTION_SCALE = 20.0
```

动作值会被裁剪到：

```text
[-1.0, 1.0]
```

第三维当前固定为：

```text
0.0
```

---

## 8. 训练设置

训练脚本：

```bash
python -m src.train
```

训练配置：

| Item | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Loss Function | MSELoss |
| Epochs | 50 |
| Batch Size | 32 |
| Device | CPU |
| Checkpoint | `outputs/checkpoints/best_model.pt` |
| History | `outputs/history.json` |

训练目标是最小化预测动作和真实动作之间的均方误差。

训练过程中会在每个 epoch 后计算验证集 loss，并保存验证集表现最好的模型参数。

---

## 9. 单步动作预测评估

### 9.1 评估脚本

单步动作预测评估脚本为：

```bash
python -m src.eval_metrics
```

该脚本在整个验证集上统计：

- MSE all dims
- MAE per dim
- Mean L2 error on xy
- Success rate @0.3
- Success rate @0.5
- X direction accuracy
- Y direction accuracy
- Both x/y direction accuracy

---

### 9.2 单步评估结果

当前 v0.7.4 沿用 v0.6 baseline 的单步模型指标：

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

### 9.3 单步指标解释

| Metric | Meaning |
|---|---|
| `MSE all dims` | 所有动作维度上的均方误差 |
| `MAE dx` | x 方向动作的平均绝对误差 |
| `MAE dy` | y 方向动作的平均绝对误差 |
| `MAE third dim` | 第三维动作的平均绝对误差 |
| `Mean L2 error on xy` | xy 平面预测动作和真实动作之间的平均距离 |
| `Success rate @0.3` | xy L2 error 小于等于 0.3 的样本比例 |
| `Success rate @0.5` | xy L2 error 小于等于 0.5 的样本比例 |
| `X direction accuracy` | x 方向正负号预测正确率 |
| `Y direction accuracy` | y 方向正负号预测正确率 |
| `Both x/y direction accuracy` | x 和 y 方向同时预测正确的比例 |

从单步结果看，模型已经具备较强的动作方向判断能力，但动作幅度预测仍然不够精确。

---

## 10. 定性可视化评估

### 10.1 普通验证样本可视化

运行脚本：

```bash
python -m src.eval_samples
```

输出目录：

```text
outputs/visuals/eval_samples/
```

该脚本会固定保存 20 张验证样本可视化图。

每张图包含：

- 当前场景
- agent 位置
- target action 箭头
- pred action 箭头
- xy L2 error

该可视化用于观察模型在普通验证样本上的动作预测行为。

---

### 10.2 同图多指令对照可视化

运行脚本：

```bash
python -m src.compare_instructions
```

输出目录：

```text
outputs/visuals/compare_instructions/
```

该脚本会固定保存 5 张同图多指令对照图。

每张图会对同一场景分别输入：

```text
go to red block
go to green block
go to blue block
```

并比较每条指令下的：

- target action
- pred action
- xy L2 error

该实验用于观察模型是否真的根据语言指令改变动作输出。

---

## 11. 闭环 rollout 评估

从 v0.7 开始，MiniVLA 加入二维 closed-loop rollout 仿真。

### 11.1 闭环执行流程

闭环执行流程如下：

```text
current image + instruction
  -> MiniVLA predicts action
  -> update agent position
  -> render next image
  -> MiniVLA predicts next action
```

该过程会持续执行，直到满足以下条件之一：

1. agent 进入严格成功半径。
2. 连续多步没有明显接近目标。
3. 达到最大执行步数。

---

### 11.2 闭环评估脚本

运行命令：

```bash
python -m src.simulate_rollout
```

输出目录：

```text
outputs/rollouts_v074/
```

输出内容包括：

- rollout 轨迹 PNG
- rollout 轨迹 GIF
- `rollout_summary.json`

每个 rollout 记录：

- 初始 agent 位置
- 目标块位置
- 每一步预测动作
- 每一步 agent 位置
- 最终距离
- 最优距离
- 是否成功
- 是否相比初始位置更接近目标
- 停止原因

---

### 11.3 v0.7.4 闭环配置

v0.7.4 的闭环配置如下：

| Item | Value |
|---|---:|
| Num rollouts | 20 |
| Max steps | 20 |
| Execution scale | 6.0 |
| Primary success radius | 6.0 |
| Stop radius | 6.0 |
| No-improve patience | 5 |
| Min improvement | 0.1 |

---

### 11.4 v0.7.4 闭环结果

| Metric | Value |
|---|---:|
| Primary success count | 3 / 20 |
| Primary success rate @6 | 15.00% |
| Improvement count | 19 / 20 |
| Improvement rate | 95.00% |
| Avg initial distance | 25.225 |
| Avg final distance | 10.113 |
| Avg best distance | 8.980 |
| Avg distance reduction | 15.112 |
| Avg best distance reduction | 16.245 |
| Avg steps executed | 11.00 |

Final success rates:

| Metric | Value |
|---|---:|
| Final success @6 | 15.00% |
| Final success @8 | 30.00% |
| Final success @10 | 50.00% |

Best success rates:

| Metric | Value |
|---|---:|
| Best success @6 | 15.00% |
| Best success @8 | 40.00% |
| Best success @10 | 60.00% |

Stop reasons:

| Stop reason | Count |
|---|---:|
| no_improvement_stop | 16 |
| primary_success | 3 |
| max_steps | 1 |

---

### 11.5 闭环结果解释

闭环 rollout 结果显示，MiniVLA v0.7.4 已经具备初步目标接近能力。

虽然严格成功率 `Primary success @6` 只有 15.00%，但 `Improvement rate` 达到 95.00%，说明绝大多数 rollout 都能让 agent 相比初始位置更接近目标。

平均初始距离为 25.225，平均最终距离为 10.113，说明模型作为 policy 执行时能够显著缩短 agent 与目标之间的距离。

不过，当前模型仍然缺乏稳定的末端精细控制能力。许多轨迹能够进入目标附近，但无法稳定进入 6 像素严格成功半径。较宽松的 `Best success @10 = 60.00%` 表明模型经常能将 agent 带到目标附近，但动作幅度控制仍不够精确。

---

## 12. 模型行为特点

从单步评估、可视化结果和 closed-loop rollout 可以观察到：

1. 模型在多数单步样本中能够预测正确动作方向。
2. 模型经常能判断目标方向，但动作幅度有时偏小或偏大。
3. 同图多指令实验中，模型输出会随着语言指令变化而变化。
4. 文本条件已经对动作输出产生影响。
5. 在 closed-loop rollout 中，模型能够让 agent 在大多数轨迹中接近目标。
6. 当前模型在末端精细控制上仍不稳定。
7. 当前模型更擅长“往目标方向走”，但不擅长“精准停在目标附近”。

---

## 13. 模型优势

MiniVLA v0.7.4 的主要优点：

1. 完成了从数据生成到训练、评估、可视化和 closed-loop rollout 的完整闭环。
2. 数据协议中保存了 agent 与目标块位置，便于自动生成多指令 ground truth。
3. 模型能够同时处理图像和语言输入。
4. 支持连续动作预测，而不是简单分类。
5. 实现了全验证集指标评估。
6. 实现了 success rate 指标。
7. 实现了同图多指令对照实验。
8. 固定保存可视化结果，便于复现实验和项目展示。
9. 实现二维 closed-loop rollout，使模型能够作为 policy 多步执行。
10. 记录了 closed-loop success rate、improvement rate、final distance 和 best distance。

---

## 14. 模型限制

当前模型仍有明显限制：

1. 场景是简单 2D toy environment。
2. 图像分辨率较低，仅为 64x64。
3. 语言模板较单一，仅包含 `go to {color} block`。
4. 动作是连续位移预测，但仍是 toy setting。
5. 第三维动作当前没有真实控制含义。
6. 模型结构较小，没有使用 Transformer 或预训练视觉 backbone。
7. 训练数据规模较小，仅 1000 条样本。
8. success rate 是基于 xy L2 error 或目标距离阈值的近似指标。
9. 当前 closed-loop success rate 仍然较低。
10. 模型没有显式 stop 动作，停止逻辑由规则控制。
11. 当前执行策略使用固定步长，没有根据距离动态调整。
12. 当前模型不适用于真实机器人环境。

---

## 15. 安全与伦理说明

MiniVLA v0.7.4 是一个教学型 toy 模型，不涉及真实机器人部署，也不应该用于实际物理控制。

如果未来扩展到真实机器人或更真实的仿真控制，需要额外考虑：

1. 动作安全约束。
2. 碰撞检测。
3. 执行失败恢复机制。
4. 环境边界限制。
5. 模型不确定性估计。
6. 人类监督和紧急停止机制。
7. 数据分布外样本的失败处理。

当前版本仅用于学习和实验展示。

---

## 16. 推荐后续改进方向

后续建议：

1. 继续逐行理解 `dataset.py`、`model.py`、`train.py`、`eval_samples.py`、`simulate_rollout.py`。
2. 扩展语言模板，例如加入 `move to red block`、`reach the blue block` 等表达。
3. 增加数据集规模和场景多样性。
4. 增加近距离样本，提高模型末端精细控制能力。
5. 改进执行策略，例如根据距离自适应调整步长。
6. 增加显式 stop 动作维度，而不是只依赖规则停止。
7. 改进模型结构，例如引入 attention 或更强 CNN encoder。
8. 尝试更复杂的二维环境或真实仿真环境。
9. 整理最终简历项目描述和面试讲解材料。

---

## 17. Model Card Summary

MiniVLA v0.7.4 是一个小型 toy VLA baseline，用于学习图像、语言和动作之间的基本建模流程。模型能够在简单 2D 场景中根据语言指令预测 agent 的移动方向，并在验证集上达到 88.46% 的 x/y 同时方向正确率和 80.50% 的单步 `success rate @0.5`。

在 closed-loop rollout 中，模型能够作为一个弱 policy 多步执行。虽然严格成功率 `Primary success @6` 只有 15.00%，但 `Improvement rate` 达到 95.00%，说明模型在大多数 rollout 中能够使 agent 更接近目标。当前模型仍存在动作幅度预测不够精确、语言模板单一、末端控制不稳定和缺少显式 stop 动作等限制。

整体上，MiniVLA v0.7.4 适合作为学习 VLA 工程流程和构建简历项目雏形的基础版本。