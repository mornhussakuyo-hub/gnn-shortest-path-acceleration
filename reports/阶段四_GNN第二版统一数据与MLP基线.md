# 阶段四：GNN 第二版统一数据接口与无传播 MLP 基线

## 结论

第二版统一数据接口和五种子无传播区域 MLP 已完成。全部训练由 PyTorch CUDA 在
NVIDIA GeForce RTX 3050 Laptop GPU 上执行，阶段状态为 `ready_for_nbfnet`。

五个种子的 candidate holdout Spearman 均为正，均值为 `0.8914 ± 0.0238`，明显高于
原始频率的 `0.7957`，证明当前历史特征和正式标签具有可学习关系。但 MLP 的 NDCG@K 和
Top-K 平均收益没有超过原始频率，因此结论只能是“学习链路成立”，不能宣称 MLP 已在所有
选区指标上优于简单基线。

## 数据身份与隔离

- 历史窗口 `H`：34,328 条 OD，只负责生成输入。
- 标签：1,200 个全图固定随机候选，每个候选使用 2,000 条固定 `Y` 窗口查询。
- 道路图：133,839 个节点、221,589 条有向边、13 类道路。
- 节点特征：32 维，包括历史起终点计数、入出度、入出边长和有向道路类型占比。
- 区域特征：对节点特征分别做 mean 和 max 池化，共 64 维。
- 坐标不进入模型特征，只用于把历史 OD 确定性聚成需求原型。
- 禁止输入历史路径、标签窗口 OD、shortcut、接入工作量、查询耗时和第一版 Proxy。
- 候选来源不进入模型；当前所有候选本身也只有 `fixed_random_bfs` 一种来源。

统一数据摘要为：

```text
ffcb11cc35a0e8d0a816b7dba98b05c31c1ac880eba9100de03dadebb205f74e
```

## 需求原型

接口包含 128 个需求原型，供 NBFNet 读取。原型只使用历史 OD 的起终点与静态节点坐标做
带权聚类，不读取历史最短路径。每个原型保留总出现权重、起点集合及权重、终点集合及权重。
MLP 不使用这些原型张量，只使用区域池化后的 64 维输入。

## 候选划分

当前 1,200 个候选均来自同一个全图固定随机池，因此用种子 42 对候选执行固定随机划分：

| 划分 | 候选数 | 用途 |
| --- | ---: | --- |
| train | 840 | 更新模型参数 |
| validation | 180 | 早停与模型选择 |
| holdout | 180 | 当前 `H→Y` 内的候选泛化复核 |

holdout 不参与模型选择，也不是最终未来时间测试。最终时间窗口在 NBFNet 结构、消融和预算
冻结前不得运行。

## MLP 结构与训练

- 输入：64 维区域池化特征。
- 隐藏层：64、32，ReLU。
- 主损失：标准化收益上的 Huber 损失。
- 排序损失：随机候选对的 pairwise logistic ranking loss，权重 0.20。
- 优化：Adam，学习率 0.003，权重衰减 0.0001。
- 早停：validation 组合损失，耐心值 60。
- 随机种子：42、43、44、45、46。
- 训练环境：PyTorch 2.13.0+cu130、CUDA 13.0、RTX 3050 Laptop GPU。

特征与标签标准化参数只从 train 拟合。模型种子 46 因 validation Spearman 最高被选中；
holdout 未参与选择。

## 五种子结果

| Candidate holdout 指标 | MLP 均值 | 标准差 | 原始频率 |
| --- | ---: | ---: | ---: |
| Spearman | 0.8914 | 0.0238 | 0.7957 |
| NDCG@K | 0.9556 | 0.0153 | 0.9661 |
| Top-K 平均真实收益 | 126.000 | 1.853 | 128.530 |
| MAE | 9.043 | 0.606 | — |

五个种子的 holdout Spearman 范围为 0.8479–0.9203；每次 Top-K 平均收益都高于该 holdout
全部候选均值 36.660。最佳 validation 模型是种子 46，其 holdout Spearman 为 0.9203，
Top-K 平均收益为 128.052。

## 结论边界与下一步

当前结果支持三点：标签可学、无传播 MLP 是强对照、NBFNet 可以进入正式训练。它不支持“MLP
全面优于原始频率”，也不证明跨时间或跨图泛化。下一步在完全相同的数据摘要、候选划分、
标签和评价代码上训练已实现的 OD 条件化双向 NBFNet，并同时报告 Spearman、NDCG@K、Top-K 收益
与未来时间评测。

复现命令：

```bash
.venv/bin/python scripts/build_demand_field_dataset.py
.venv-gnn/bin/python scripts/train_demand_field_model.py
.venv-gnn/bin/python scripts/evaluate_demand_field_model.py
```
