# 第一版 GNN 核心消融结果归档

本目录于 2026-07-14 从服务器
`/home/xixiangtang/AIC/results/gnn_ablation/` 下载，用于保存第一版核心 72 组验证集
消融实验的可审计证据。

仓库保留以下轻量结果：

- `ablation_manifest.json`：冻结的实验矩阵和公共参数。
- `ablation_runs.csv`：72 个运行组的唯一聚合主表。
- `reference/evaluation_summary.csv`：随机五种子和 OD 热点公共参考结果。
- `runs/*/training_summary.json` 或 `scoring_summary.json`：每组训练或解析评分摘要。
- `runs/*/evaluation_summary.csv`：每组验证集精确查询评测摘要。

服务器原目录大小约 623 MB。模型权重、逐节点分数、训练历史、区域明细和运行日志未
复制进 Git；这些文件体积较大，且聚合结论可以由本目录中的 manifest 与摘要文件核查。

完整性检查结果：

- 聚合数据行：72。
- 唯一运行 ID：72。
- 变体数：16。
- 评测切分：`validation`。
- 最低距离正确率：`1.000000`。
- 训练、评分和评测摘要文件：145。
- `ablation_runs.csv` SHA-256：
  `ab91c082e5494e0c8d3a5d77d2ab516ad2da3cce502281b1e3feba27e50d298f`。

服务器核验时存在 72 个运行目录，`failed_runs.txt` 不存在。完整 202 组扩展套件没有
运行，本目录只对应冻结后的 72 组核心验证集实验。
