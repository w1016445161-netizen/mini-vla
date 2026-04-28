# MiniVLA v1.0

## Version

MiniVLA v1.0 Resume Version

## Based on

MiniVLA v0.7.4 closed-loop rollout baseline

## Version Description

This version packages the current MiniVLA project into a resume-ready engineering project.

The project includes:

- Synthetic 2D VLA data generation
- PyTorch Dataset / DataLoader pipeline
- CNN image encoder
- Text embedding encoder
- Multimodal fusion MLP
- Continuous action prediction
- Full validation metrics
- Success rate evaluation
- Same-image multi-instruction comparison
- Closed-loop 2D rollout simulation
- README, model card, project report, and experiment log

## Key Results

Static validation:

- Success rate @0.5: 80.50%
- Both x/y direction accuracy: 88.46%

Closed-loop rollout:

- Primary success rate: 15.00%
- Improvement rate: 95.00%
- Final success @10: 50.00%
- Best success @10: 60.00%
- Average final distance: 10.113
- Average steps executed: 11.00

## Limitations

MiniVLA is still a toy-level VLA prototype. It can predict basic target-directed actions and reduce distance in closed-loop rollout, but it still struggles with precise action magnitude control and stable convergence near the target.

## Next Version

MiniVLA v1.1 will focus on improving data distribution, especially near-target samples and small-step control.