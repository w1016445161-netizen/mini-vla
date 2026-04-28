# MiniVLA Experiment Log

## v0.6 Baseline with Structured Scene Metadata

### 1. Experiment Goal

本次实验目标是将 mini_vla 从一个“能训练、能随机可视化”的 toy baseline，升级为具有较完整评估能力的 v0.6 baseline。

在早期版本中，模型虽然已经能够完成基本训练，并通过 `eval_samples.py` 随机观察预测动作，但评估仍然存在两个问题：

1. 只能查看少量随机样本，缺少全验证集层面的稳定指标。
2. 同一张图换不同语言指令时，无法自动计算每个指令对应的真实动作，因此语言条件评估不够严谨。

本次 v0.6 主要完成以下改进：

1. 在数据样本中新增 `agent_pos` 和 `blocks` 字段，记录 agent 与 red、green、blue 三个颜色块的位置。
2. 基于显式位置信息自动计算不同语言指令下的 ground-truth action。
3. 完善全验证集评估指标，包括 MSE、MAE、xy 平面 L2 误差、方向准确率和 success rate。
4. 使用同图不同指令对照实验检查模型是否真正利用语言条件生成动作。
5. 初步形成可复现实验记录，为后续 README、model card 和项目报告整理做准备。

---

### 2. Current Version

当前版本定义为：

**MiniVLA v0.6: structured metadata + full validation metrics + success rate**

该版本的核心特征是：

- 数据协议更加清晰。
- 评估方式更加系统。
- 模型不只通过 loss 判断效果，也通过方向准确率、success rate 和可视化对照实验判断行为能力。

---

### 3. Data Protocol

每条样本包含以下字段：

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

字段说明：

| Field | Meaning |
|---|---|
| `id` | 样本编号 |
| `image` | 当前 2D 场景图像路径 |
| `instruction` | 语言指令，例如 `go to red block` |
| `target_color` | 当前指令对应的目标颜色 |
| `action` | 从 agent 指向目标颜色块的连续动作 `[dx, dy, 0.0]` |
| `agent_pos` | agent 在图像中的坐标 |
| `blocks` | red、green、blue 三个颜色块的位置 |

动作计算方式：

```text
dx = (target_x - agent_x) / ACTION_SCALE
dy = (target_y - agent_y) / ACTION_SCALE
```

其中 `ACTION_SCALE = 20.0`，并将动作值裁剪到 `[-1.0, 1.0]` 范围内。

当前第三维动作为 `0.0`，主要作为后续扩展 gripper、stop 或其他控制维度的预留接口。

---

### 4. Dataset Setting

当前数据集规模：

| Split | Samples |
|---|---:|
| Full dataset | 1000 |
| Train split | 800 |
| Validation split | 200 |

数据生成与划分流程：

```bash
python -m src.generate_data
python -m src.prepare_splits
```

生成后的主要文件：

| File | Role |
|---|---|
| `data/processed/dataset.json` | 完整数据集 |
| `data/processed/train.json` | 训练集 |
| `data/processed/val.json` | 验证集 |
| `data/raw/images/` | 生成的 2D 场景图像 |

---

### 5. Model Setting

当前模型为一个最小版 Vision-Language-Action 网络，定义在：

```text
src/model.py
```

模型结构包括：

| Module | Role |
|---|---|
| Image encoder | 使用小型 CNN 提取图像空间特征 |
| Text encoder | 使用 Embedding + mean pooling 提取语言指令特征 |
| Fusion module | 拼接图像特征和文本特征，并通过 MLP 融合 |
| Action head | 输出连续动作向量 `[dx, dy, 0.0]` |

模型输入：

```text
image: [B, 3, 64, 64]
instruction: [B, max_text_len]
```

模型输出：

```text
action: [B, 3]
```

当前模型的核心任务是学习：

```text
image + language instruction -> continuous action
```

即根据场景图像和语言指令预测 agent 应该朝哪个目标块移动。

---

### 6. Training Setting

训练脚本：

```text
src/train.py
```

训练配置：

| Item | Value |
|---|---|
| Optimizer | Adam |
| Loss | MSELoss |
| Epochs | 50 |
| Batch size | 32 |
| Device | CPU |
| Checkpoint | `outputs/checkpoints/best_model.pt` |
| History | `outputs/history.json` |

训练命令：

```bash
python -m src.train
```

本次训练结果：

| Metric | Value |
|---|---:|
| Best val loss | 0.055784 |
| Best checkpoint | `outputs/checkpoints/best_model.pt` |

训练过程中，模型前期快速下降，中后期训练集 loss 继续降低，验证集 loss 在较低水平附近波动。最终使用验证集表现最好的 checkpoint，而不是最后一轮模型参数。

---

### 7. Evaluation Scripts

当前 v0.6 使用两个主要评估脚本。

#### 7.1 `eval_metrics.py`

文件位置：

```text
src/eval_metrics.py
```

作用：

对整个验证集进行定量评估，统计模型整体表现。

当前输出指标包括：

- MSE all dims
- MAE per dim
- Mean L2 error on xy
- Success rate @0.3
- Success rate @0.5
- X direction accuracy
- Y direction accuracy
- Both x/y direction accuracy

运行命令：

```bash
python -m src.eval_metrics
```

#### 7.2 `compare_instructions.py`

文件位置：

```text
src/compare_instructions.py
```

作用：

对同一张图输入不同语言指令，比较每个指令下的真实动作和预测动作。

测试指令包括：

```text
go to red block
go to green block
go to blue block
```

该脚本会输出：

- 当前样本的 agent 位置
- red、green、blue 三个块的位置
- 每个指令对应的 target action
- 每个指令对应的 pred action
- 每个指令下的 xy L2 error
- target/pred 可视化箭头

运行命令：

```bash
python -m src.compare_instructions
```

---

### 8. Evaluation Metrics

当前 v0.6 baseline 的验证集结果如下：

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

指标解释：

| Metric | Meaning |
|---|---|
| `MSE all dims` | 所有动作维度上的均方误差 |
| `MAE dx/dy` | x/y 动作维度的平均绝对误差 |
| `MAE third dim` | 第三维动作的平均绝对误差 |
| `Mean L2 error on xy` | xy 平面预测动作和真实动作的平均距离 |
| `Success rate @0.3` | xy L2 error 小于等于 0.3 的样本比例 |
| `Success rate @0.5` | xy L2 error 小于等于 0.5 的样本比例 |
| `X direction accuracy` | x 方向正负号预测正确率 |
| `Y direction accuracy` | y 方向正负号预测正确率 |
| `Both x/y direction accuracy` | x 和 y 方向同时预测正确的比例 |

---

### 9. Quantitative Findings

从全验证集指标可以得到以下结论：

1. 当前模型的整体 MSE 为 0.0558，相比早期 baseline 有明显提升。
2. x 和 y 两个平面动作维度的 MAE 分别为 0.2113 和 0.2219，说明主要误差仍集中在平面移动动作上。
3. 第三维动作 MAE 仅为 0.0078，说明当前第三维基本被模型学会。
4. xy 平面平均 L2 误差为 0.3403，说明模型的动作幅度预测仍有改进空间。
5. x 方向正确率为 95.21%，y 方向正确率为 93.81%，说明模型已经具备较强的方向判断能力。
6. x/y 同时方向正确率为 88.46%，说明模型在大部分验证样本上能够预测正确移动方向。
7. 严格阈值 `0.3` 下的成功率为 55.50%，宽松阈值 `0.5` 下的成功率为 80.50%。
8. success rate 与方向准确率之间的差异说明：模型很多时候方向判断正确，但动作幅度仍不够精确。

---

### 10. Qualitative Evaluation

通过 `compare_instructions.py` 对同一张图输入不同语言指令，可以观察模型是否利用语言条件生成不同动作。

#### Example 1: Sample ID 30

Agent position:

```text
[31.0, 18.0]
```

Blocks:

```text
red:   [50.0, 26.0]
green: [20.0, 38.0]
blue:  [45.0, 10.0]
```

Results:

| Instruction | Target Action | Pred Action | XY L2 Error |
|---|---|---|---:|
| go to red block | [0.95, 0.40, 0.0] | [0.7887, 0.2777, 0.0067] | 0.2024 |
| go to green block | [-0.55, 1.00, 0.0] | [-0.0725, 0.4853, 0.0049] | 0.7021 |
| go to blue block | [0.70, -0.40, 0.0] | [0.5467, -0.4494, 0.0160] | 0.1611 |

Observation:

模型对 red 和 blue 的预测较好，但对 green 的动作幅度明显不足。

#### Example 2: Sample ID 159

Agent position:

```text
[40.0, 25.0]
```

Blocks:

```text
red:   [53.0, 35.0]
green: [29.0, 39.0]
blue:  [14.0, 54.0]
```

Results:

| Instruction | Target Action | Pred Action | XY L2 Error |
|---|---|---|---:|
| go to red block | [0.65, 0.50, 0.0] | [0.4351, 0.8992, 0.0056] | 0.4534 |
| go to green block | [-0.55, 0.70, 0.0] | [-0.7200, 0.5799, -0.0166] | 0.2082 |
| go to blue block | [-1.00, 1.00, 0.0] | [-1.1436, 1.0940, -0.0164] | 0.1716 |

Observation:

模型对 green 和 blue 的预测较接近真实动作，对 red 的 y 方向幅度偏大。

#### Example 3: Sample ID 156

Agent position:

```text
[28.0, 15.0]
```

Blocks:

```text
red:   [50.0, 45.0]
green: [50.0, 29.0]
blue:  [15.0, 42.0]
```

Results:

| Instruction | Target Action | Pred Action | XY L2 Error |
|---|---|---|---:|
| go to red block | [1.00, 1.00, 0.0] | [0.9344, 0.7939, 0.0052] | 0.2163 |
| go to green block | [1.00, 0.70, 0.0] | [0.3067, 0.5975, 0.0079] | 0.7009 |
| go to blue block | [-0.65, 1.00, 0.0] | [-0.2163, 0.8544, -0.0128] | 0.4575 |

Observation:

模型对 red 的预测较好，但对 green 和 blue 的 x 方向幅度明显不足。

---

### 11. Main Findings

本次 v0.6 实验得到以下主要结论：

1. 显式保存 `agent_pos` 和 `blocks` 后，数据协议更加清晰，评估能力明显增强。
2. 重新训练后，模型验证集表现显著提升，best val loss 降至 0.0558。
3. xy 同时方向正确率达到 88.46%，说明模型已经具备较强的方向判断能力。
4. success rate @0.5 达到 80.50%，说明在较宽松阈值下，模型已经可以完成大部分单步动作预测任务。
5. success rate @0.3 为 55.50%，说明在更严格阈值下，模型仍有明显提升空间。
6. 第三维动作误差很小，说明当前第三维基本被模型学会。
7. 主要误差集中在 dx/dy 的幅度预测上，模型常常能够判断正确方向，但不一定能准确预测动作长度。
8. 同图不同指令实验显示，模型输出会随着语言指令变化，说明文本条件已经对动作预测产生影响。
9. 模型仍然存在部分目标颜色或空间布局下预测幅度不足的问题。
10. 当前项目已经从“能训练模型”进入“能系统评估模型能力”的阶段。

---

### 12. Limitations

当前 v0.6 仍有以下限制：

1. 场景仍是简单 2D toy environment，不是真实机器人环境。
2. 语言指令模板较单一，主要是 `go to {color} block`。
3. 动作空间较简单，只是从 agent 指向目标块的单步连续位移。
4. 当前模型仍是小型 CNN + Embedding + MLP，没有使用 Transformer 或更强视觉 backbone。
5. 当前还没有真正的闭环仿真执行，模型只做单步动作预测。
6. success rate 使用的是 xy L2 误差阈值，仍然是 toy setting 下的近似成功定义。
7. 当前第三维动作恒接近 0，还没有引入 gripper、stop 或其他真实控制含义。
8. 当前数据集规模只有 1000 条，泛化能力仍受数据规模和场景复杂度限制。

---

### 13. Files Involved in v0.6

| File | Role |
|---|---|
| `src/generate_data.py` | 生成 2D toy VLA 数据，并保存 agent 与 blocks 位置信息 |
| `src/prepare_splits.py` | 将完整数据集划分为 train/val |
| `src/dataset.py` | 读取图像、指令、动作和位置元信息，构建 PyTorch Dataset |
| `src/model.py` | 定义 MiniVLA 模型结构 |
| `src/train.py` | 训练模型、验证模型并保存 best checkpoint |
| `src/eval_metrics.py` | 统计全验证集定量指标 |
| `src/eval_samples.py` | 随机样本级行为可视化 |
| `src/compare_instructions.py` | 同图不同指令 target/pred 对照实验 |
| `outputs/checkpoints/best_model.pt` | 当前最佳模型 checkpoint |
| `outputs/history.json` | 训练和验证 loss 历史 |
| `outputs/eval_metrics.json` | 全验证集评估结果 |

---

### 14. Current Project Status

已完成：

- 数据生成
- Dataset/DataLoader
- MiniVLA 模型
- 训练脚本
- checkpoint 保存
- history.json
- loss 曲线分析
- 模型结构改进
- eval_metrics.py
- compare_instructions.py 初步实验
- structured metadata 数据协议
- success rate 指标
- experiment_log 初稿

待完成：

- 固定保存 10-20 张 `eval_samples` 可视化图
- 固定保存 3-5 张 `compare_instructions` 对照图
- README.md
- docs/model_card.md
- docs/project_report.md
- 继续逐行理解 dataset.py / model.py / train.py / eval_samples.py
- 二维机械臂仿真闭环
- 简历项目描述

---

### 15. Next Steps

后续计划如下：

1. 固定保存 10-20 张 `eval_samples` 可视化图到 `outputs/visuals/eval_samples/`。
2. 固定保存 3-5 张 `compare_instructions` 对照图到 `outputs/visuals/compare_instructions/`。
3. 整理 `README.md`，说明项目目标、安装方式、运行流程和实验结果。
4. 编写 `docs/model_card.md`，总结模型结构、输入输出、指标和限制。
5. 编写 `docs/project_report.md`，形成完整项目报告。
6. 在 v0.7 或 v0.8 中加入简单二维闭环仿真，让 agent 根据预测动作逐步移动到目标位置。
7. 最终整理为可写入简历的 mini VLA 项目。

---

### 16. Resume-Oriented Summary Draft

当前项目可以初步概括为：

基于 PyTorch 实现一个 toy 级 Vision-Language-Action 系统，完成从 2D 场景数据生成、图文动作建模、训练验证、checkpoint 保存、全验证集指标评估到同图多指令可视化对照的完整闭环。模型输入为场景图像和自然语言指令，输出连续动作向量。项目设计了包含 agent 与多目标位置的结构化数据协议，并实现 MSE、MAE、xy L2 error、方向准确率和 success rate 等指标，用于系统分析模型的动作预测能力和语言条件利用情况。

---
---

---

### 17. 可视化实验产物

在 v0.6 版本中，项目已经支持自动保存可视化结果，用于后续复现实验、撰写 README、整理项目报告和展示模型行为。

本次新增的可视化结果主要包括两类：

1. 普通验证样本的 target/pred 动作对比图。
2. 同一张图在不同语言指令下的 target/pred 对照图。

---

#### 17.1 验证样本动作预测可视化

运行脚本：

```bash
python -m src.eval_samples
```

输出目录：

```text
outputs/visuals/eval_samples/
```

该脚本会从验证集中固定抽取 20 个样本，并为每个样本保存一张动作预测可视化图。

每张图包含以下信息：

- 当前 2D 场景图像
- agent 位置
- 真实动作箭头，即 target action
- 模型预测动作箭头，即 pred action
- 当前样本的 xy 平面 L2 误差

本次生成的文件包括：

```text
eval_sample_01_id_828.png
eval_sample_02_id_610.png
eval_sample_03_id_445.png
...
eval_sample_20_id_405.png
eval_samples_summary.json
```

其中，`eval_samples_summary.json` 会记录每个样本的详细信息，包括：

- 样本编号
- 语言指令
- 目标颜色
- 真实动作
- 预测动作
- xy L2 误差
- 对应可视化图片路径

该可视化结果用于观察模型在普通验证样本上的动作预测能力。相比只查看 `val_loss`，样本级可视化可以更直观地发现模型是否存在方向错误、动作幅度不足或预测偏移等问题。

---

#### 17.2 同图多指令对照可视化

运行脚本：

```bash
python -m src.compare_instructions
```

输出目录：

```text
outputs/visuals/compare_instructions/
```

该脚本会固定抽取 5 张验证图像，并对同一张图分别输入三种语言指令：

```text
go to red block
go to green block
go to blue block
```

每张对照图包含以下信息：

- 同一场景下 red、green、blue 三个目标块的位置
- 不同语言指令对应的真实动作箭头
- 不同语言指令对应的模型预测动作箭头
- target action 与 pred action 的可视化对比

本次生成的文件包括：

```text
compare_01_id_768.png
compare_02_id_274.png
compare_03_id_541.png
compare_04_id_759.png
compare_05_id_166.png
compare_instructions_summary.json
```

其中，`compare_instructions_summary.json` 会记录每张图中的详细对照信息，包括：

- agent 位置
- red、green、blue 三个目标块的位置
- 每条语言指令
- 每条指令对应的真实动作
- 每条指令对应的预测动作
- 每条指令下的 xy L2 误差
- 对应可视化图片路径

该可视化结果用于检查模型是否真正利用了语言条件。若模型在同一张图下输入不同颜色指令后，预测动作能够朝不同目标块变化，说明文本条件已经对动作输出产生影响。

---

#### 17.3 当前观察结果

从本次保存的可视化结果看，模型在多数样本中能够预测出较合理的移动方向，尤其是在目标方向较明显的样本中，预测箭头和真实箭头较为接近。

在普通验证样本可视化中，部分样本的预测结果已经较准确，例如模型能够正确判断 agent 应该向左、向右、向上或向下移动。但仍有一些样本存在动作幅度不足的问题，即方向基本正确，但预测箭头长度偏短或偏长。

在同图多指令对照实验中，模型的预测动作通常会随着语言指令变化而变化，说明文本指令并不是完全无效的。模型已经初步具备根据不同颜色目标生成不同动作的能力。

不过，目前模型的语言-视觉-动作对齐还不够稳定。部分样本中，某些颜色目标对应的预测动作仍然存在较大的 xy L2 误差。这说明模型虽然已经学到了一定的语言条件控制能力，但在目标定位精度和动作幅度预测上仍有继续改进空间。

---

#### 17.4 工程意义

本次固定保存可视化结果后，mini_vla 项目从“临时运行并截图观察”升级为“自动生成可复现实验产物”。

这一改进具有以下意义：

1. 后续 README 可以直接引用固定图片展示模型效果。
2. 项目报告可以使用这些图片分析模型优点和不足。
3. 可视化结果和 summary json 可以对应起来，便于复现实验。
4. 模型评估不再只依赖 loss 和数值指标，而是同时包含行为层面的可视化证据。
5. 项目更接近一个可以展示、解释并写入简历的完整工程。

当前 v0.6 版本已经具备较完整的评估闭环：

- 全验证集数值指标
- success rate
- 普通样本动作预测可视化
- 同图多指令对照可视化
- 可复用 summary json 实验记录

这说明 mini_vla 已经从一个基础 toy demo，逐步发展为一个具备训练、评估、可视化和实验记录能力的小型 VLA 工程项目。

---

### 18. v0.7.1 Closed-Loop Rollout 初步实验

在 v0.7 中，项目从单步动作预测扩展到二维闭环 rollout。模型不再只对单张图像预测一次动作，而是作为 policy 多次执行：每一步根据当前场景图像和语言指令预测动作，agent 根据预测动作移动，环境重新渲染后继续下一步，直到到达目标附近或达到最大步数。

初版 v0.7 rollout 中，模型输入图像包含路径轨迹和目标框，导致输入分布与训练图像不一致；同时执行步长较大，容易导致 agent 偏离目标或跑到边界。修正后，v0.7.1 将模型输入图像和可视化图像分离，模型只接收干净场景图，可视化图才绘制路径和目标框。同时将闭环执行步长调整为 `EXECUTION_SCALE = 6.0`。

本次 v0.7.1 配置如下：

| Item | Value |
|---|---:|
| Num rollouts | 20 |
| Max steps | 20 |
| Success radius | 6.0 |
| Execution scale | 6.0 |

实验结果如下：

| Metric | Value |
|---|---:|
| Num success | 3 |
| Success rate | 15.00% |
| Num improved | 19 |
| Improvement rate | 95.00% |
| Avg initial distance | 25.225 |
| Avg final distance | 10.128 |
| Avg distance reduction | 15.096 |

结果表明，当前模型作为 closed-loop policy 时，虽然严格成功率只有 15.00%，但 95.00% 的 rollout 都使 agent 更接近目标，平均距离从 25.225 降低到 10.128。这说明模型已经具备明显的闭环接近能力。

当前主要问题是末端精细控制不足。许多失败样本并不是完全跑偏，而是停在目标附近，例如最终距离为 6 到 10 像素之间，略高于当前成功半径 6.0。这说明模型方向判断较好，但在接近目标时动作幅度控制仍不稳定。

该实验说明，单步动作预测指标较好并不等价于闭环执行成功率高。闭环 rollout 会放大单步预测误差，因此需要单独评估 closed-loop success rate、improvement rate 和 final distance。
---

### 19. v0.7.2 多阈值闭环评估

在 v0.7.2 中，项目继续完善二维闭环 rollout 评估。相比 v0.7.1 只统计单一成功半径，本版本增加了多阈值 success rate 和 best distance 统计，用于更细致地分析模型在闭环执行过程中的行为。

本次评估设置如下：

| Item | Value |
|---|---:|
| Num rollouts | 20 |
| Max steps | 20 |
| Execution scale | 6.0 |
| Primary success radius | 6.0 |
| Evaluation radii | 6.0 / 8.0 / 10.0 |

本次实验结果如下：

| Metric | Value |
|---|---:|
| Primary success count | 3 / 20 |
| Primary success rate @6 | 15.00% |
| Improvement count | 19 / 20 |
| Improvement rate | 95.00% |
| Avg initial distance | 25.225 |
| Avg final distance | 10.128 |
| Avg best distance | 8.971 |
| Avg distance reduction | 15.096 |
| Avg best distance reduction | 16.253 |

Final success rates:

| Metric | Value |
|---|---:|
| Final success @6 | 15.00% |
| Final success @8 | 30.00% |
| Final success @10 | 55.00% |

Best success rates:

| Metric | Value |
|---|---:|
| Best success @6 | 15.00% |
| Best success @8 | 40.00% |
| Best success @10 | 60.00% |

本次实验表明，当前 MiniVLA 模型已经具备初步 closed-loop 接近能力。虽然严格半径 `6.0` 下的最终成功率只有 15.00%，但 95.00% 的 rollout 都使 agent 相比初始位置更接近目标，平均距离从 25.225 降低到 10.128，平均最优距离达到 8.971。

多阈值评估进一步说明，模型失败样本中有相当一部分并不是完全跑偏，而是停在目标附近。在较宽松的 10 像素阈值下，final success rate 达到 55.00%，best success rate 达到 60.00%。这说明模型已经能够在多数情况下将 agent 移动到目标附近，但在最后几步的精细收敛上仍然不稳定。

`best_success_rate` 和 `final_success_rate` 的差异也提供了额外信息。例如在 8 像素阈值下，best success rate 为 40.00%，final success rate 为 30.00%，说明部分 rollout 曾经进入较近范围，但后续动作又使 agent 略微偏离目标。这表明当前 policy 缺少明确的停止机制和末端小步控制能力。

该实验说明，closed-loop 评估不能只看单一 final success rate，还需要同时分析 improvement rate、final distance、best distance 和多阈值 success rate。相比 v0.6 的单步动作预测，v0.7.2 已经开始评估模型作为 policy 在多步执行中的实际行为。

---

### 20. v0.7.3 Stop 策略实验

在 v0.7.3 中，项目在二维闭环 rollout 中加入了 stop 策略，目标是避免 agent 在已经接近目标后继续移动，或在连续多步没有明显接近目标时继续无意义执行。

本次加入了两种停止机制：

1. `near_target_stop`：当 agent 距离目标小于等于 `STOP_RADIUS = 8.0` 时停止。
2. `no_improvement_stop`：当连续若干步没有明显降低目标距离时停止。

本次配置如下：

| Item | Value |
|---|---:|
| Num rollouts | 20 |
| Max steps | 20 |
| Execution scale | 6.0 |
| Primary success radius | 6.0 |
| Stop radius | 8.0 |
| No-improve patience | 3 |
| Min improvement | 0.2 |

实验结果如下：

| Metric | Value |
|---|---:|
| Primary success count | 2 / 20 |
| Primary success rate @6 | 10.00% |
| Improvement count | 19 / 20 |
| Improvement rate | 95.00% |
| Avg initial distance | 25.225 |
| Avg final distance | 10.032 |
| Avg best distance | 9.239 |
| Avg distance reduction | 15.193 |
| Avg best distance reduction | 15.985 |
| Avg steps executed | 7.35 |

Final success rates:

| Metric | Value |
|---|---:|
| Final success @6 | 10.00% |
| Final success @8 | 35.00% |
| Final success @10 | 55.00% |

Best success rates:

| Metric | Value |
|---|---:|
| Best success @6 | 10.00% |
| Best success @8 | 35.00% |
| Best success @10 | 60.00% |

Stop reasons:

| Stop reason | Count |
|---|---:|
| no_improvement_stop | 13 |
| near_target_stop | 5 |
| primary_success | 2 |

从结果看，stop 策略显著减少了平均执行步数。v0.7.2 中 rollout 通常执行满 20 步，而 v0.7.3 的平均执行步数下降到 7.35 步，说明 stop 策略能够有效避免无意义的长时间执行。

但是，严格成功率并没有提升。primary success@6 从 v0.7.2 的 15.00% 降至 10.00%。这说明当前 stop 策略虽然提升了执行效率，但参数设置偏保守，可能在 agent 尚未进入严格成功半径之前提前停止。

其中，`near_target_stop` 使用的停止半径为 8.0，而严格成功半径为 6.0，因此部分 rollout 会在 6 到 8 像素之间停止，导致其满足 success@8，但不满足 success@6。此外，`no_improvement_stop` 触发了 13 次，说明该策略是当前主要停止原因，也可能过早终止了一些仍有继续接近目标机会的轨迹。

总体来看，v0.7.3 的主要收益是提升 closed-loop 执行效率，而不是直接提升严格成功率。该实验说明，闭环控制不仅需要动作预测模型，还需要合理的终止策略；同时，stop 策略的半径和 patience 参数会显著影响最终成功率。

---

### 21. v0.7.4 Stop 策略参数调优实验

在 v0.7.4 中，项目对 v0.7.3 的 stop 策略进行了参数调优。v0.7.3 虽然显著减少了平均执行步数，但严格成功率从 15.00% 降至 10.00%，说明 stop 策略存在一定过早停止问题。

本次调整如下：

| Parameter | v0.7.3 | v0.7.4 |
|---|---:|---:|
| STOP_RADIUS | 8.0 | 6.0 |
| NO_IMPROVE_PATIENCE | 3 | 5 |
| MIN_IMPROVEMENT | 0.2 | 0.1 |

调整目的：

1. 将 `STOP_RADIUS` 从 8.0 改为 6.0，使 near-target stop 与严格成功半径一致，避免 agent 在 6 到 8 像素范围内提前停止。
2. 将 `NO_IMPROVE_PATIENCE` 从 3 改为 5，给模型更多闭环修正机会。
3. 将 `MIN_IMPROVEMENT` 从 0.2 改为 0.1，使较小幅度的接近也被视为有效改善，减少误判为 no improvement 的情况。

本次 v0.7.4 实验结果如下：

| Metric | Value |
|---|---:|
| Num rollouts | 20 |
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

与 v0.7.3 相比，v0.7.4 将严格成功率从 10.00% 恢复到 15.00%，同时保持 95.00% 的 improvement rate。与 v0.7.2 相比，v0.7.4 在保持相同 primary success@6 和 best success@10 的情况下，将平均执行步数从 20 步降低到 11 步。

这说明 v0.7.4 是当前较平衡的闭环 rollout 策略版本：它保留了模型的闭环接近能力，同时减少了无意义的长时间执行。

不过，v0.7.4 中 `no_improvement_stop` 仍然触发了 16 次，说明当前模型在目标附近的精细控制能力仍然不足。stop 策略能够减少漂移和无效执行，但无法从根本上解决模型动作幅度预测不稳定的问题。后续如果要继续提升 closed-loop success rate，需要考虑改进执行策略、数据分布或训练目标。