# Final Resume Description

## 项目名称

MiniVLA：基于 PyTorch 的 Toy 级视觉-语言-动作系统

## 简历版本

基于 PyTorch 构建 toy 级 Vision-Language-Action 原型系统，完成从 2D 场景数据生成、Dataset/DataLoader、CNN+Embedding+MLP 多模态动作预测模型训练，到指标评估、可视化分析和 closed-loop rollout 仿真的完整工程闭环。

设计结构化数据协议，记录 agent_pos 与多颜色目标块位置信息，支持同图多语言指令下的 ground-truth action 自动计算；实现 target/pred 动作箭头可视化与同图多指令对照实验，验证模型是否根据语言指令改变动作输出。

构建 MSE、MAE、xy L2 error、方向准确率、success rate 与 closed-loop rollout 指标体系；验证集 success_rate@0.5 达 80.50%，x/y 同时方向正确率达 88.46%，rollout improvement rate 达 95.00%。

## 面试介绍版本

我做了一个 MiniVLA 项目，目标是从零实现一个最小版 Vision-Language-Action 系统。模型输入一张 2D 场景图像和一条语言指令，例如 go to red block，输出连续动作向量 dx、dy，用于控制 agent 朝目标块移动。

工程上我完成了数据生成、Dataset/DataLoader、模型训练、指标评估、可视化分析和 closed-loop rollout 仿真。模型结构是 CNN 图像编码器、Embedding 文本编码器和 MLP 融合动作头。评估上我不只看 loss，还统计了方向准确率、success rate 和 rollout improvement rate。

当前静态验证集 success_rate@0.5 达到 80.50%，x/y 同时方向准确率达到 88.46%；闭环 rollout 中 95% 的测试轨迹能够让 agent 更接近目标。但模型在近目标精细控制和动作幅度预测上仍有不足，后续计划通过近距离样本增强和 stop action 改进闭环成功率。