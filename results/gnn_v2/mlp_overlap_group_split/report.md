# GNN 第二版无传播区域 MLP 基线

- 数据集摘要：`e9b1bc149a6a94dd744305e41817c3a2b9447cd5e3ba5a17c71fea38db2bf261`
- 随机种子：42, 43, 44, 45, 46
- 隐藏层：[64, 32]
- 训练框架：PyTorch `2.13.0+cu130`
- 训练设备：`NVIDIA GeForce RTX 4090 D`（CUDA `13.0`）
- 阶段门：`ready_for_nbfnet`
- 评测口径：同一 H→Y 时间对内按候选划分 train/validation/holdout；holdout 不是最终未来时间测试。

## Candidate holdout 五种子结果

| 指标 | MLP 均值 | 标准差 | 原始频率 |
| --- | ---: | ---: | ---: |
| Spearman | 0.8696 | 0.0197 | 0.8482 |
| NDCG@K | 0.9593 | 0.0143 | 0.9654 |
| Top-K 平均真实收益 | 136.841 | 2.764 | 137.622 |
| MAE | 12.552 | 0.874 | — |

## 结论边界

该 MLP 只读取历史起终点计数和静态道路特征的区域 mean/max 池化，不读取路径、标签窗口输入、
候选来源、shortcut、查询耗时或端点接入工作量。它是后续 NBFNet 必须公平超过的无传播对照。
