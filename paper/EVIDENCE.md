# 初稿证据索引

本文件只负责把论文中的关键数字映射回仓库证据。若后续重跑实验，必须先更新机器可读结果和
正式报告，再更新论文；不得直接在论文中手改为更有利的数字。

## 数据与协议

| 论文内容 | 主要证据 |
| --- | --- |
| Porto 图、OD 与 H/Y/F 数量 | `results/gnn_v2/demand_field_dataset.json`、`results/gnn_v2/label_manifest.json`、`results/gnn_v2/future_window_z0/` |
| Chicago 图、OD 与 H/Y/F 数量 | `results/chicago/gnn_v2/demand_field_dataset.json`、`data/processed/chicago/chicago_query_manifest.json`、`results/chicago/gnn_v2/future_window_z0/` |
| 1,200 个候选、512 节点与 840/180/180 split | 两城 `candidate_manifest.json`、`demand_field_dataset.json` |
| Porto 521 个、Chicago 452 个 Jaccard 重叠组 | 两城候选/数据 manifest 与阶段七报告 |
| 每候选每窗口 2,000 条标签查询 | 两城当前和未来 `label_manifest.json` |
| 两城两窗口合计 960 万次标签查询 | `1,200×2,000×2×2`，由两城 Y/F `label_manifest.json` 共同核验 |

## 排序结果

| 论文表格 | 主要证据 |
| --- | --- |
| 表 2：空间 holdout | `results/gnn_v2/nbfnet_propagation/g4_frozen_evaluation/`、`results/chicago/gnn_v2/nbfnet_propagation/g4_frozen_evaluation/`、两城 MLP/Proxy 汇总 |
| 表 3：未来窗口 | 两城 `future_window_z0/` 与 `g4_frozen_evaluation/` |
| 表 4：度保持重连 | 两城 `z0_orthogonal_ablation/summary.json` |
| 表 7：BRIDGE-B 三种子 | 两城 `g5_cost_aware_exploration/s3_gain1_seeds43_44_short/` 及 full predictions |

统一人工可读口径见：

- `reports/最终研究成果与论文结论.md`；
- `reports/阶段六_最终对比与扩展验证.md`；
- `reports/阶段七_芝加哥跨城市冻结复现.md`。

## 在线与 C++ 结果

| 论文表格 | 主要证据 |
| --- | --- |
| 表 5：K=18 严格非重叠展开节点 | `results/gnn_v2/multi_region_online_g4/`、`results/chicago/gnn_v2/multi_region_online_g4_clean_rerun/` |
| 表 6：C++ 平均/P95/扫边 | `results/cpp_online_benchmark/porto/summary.{csv,json}`、`results/cpp_online_benchmark/chicago/summary.{csv,json}` |
| 228,000 个在线查询--索引配对 | `reports/最终研究成果与论文结论.md` 第六节及阶段七成本感知报告 |
| 560,000 个 C++ 正式计时配对 | `reports/阶段八_C++高性能在线评测.md` |
| BRIDGE-B 的 24,000 个三种子在线配对 | 两城 `g5_cost_aware_exploration/s{2,3}_*_online_k18/` |

## 方法实现

| 论文方法 | 代码入口 |
| --- | --- |
| CRP 式区域 shortcut | `src/compression_index.py` |
| 内部端点局部接入 | `src/indexed_query.py` |
| 单区域精确收益标签 | `src/region_labels.py` |
| Z0 固定双向多尺度传播 | `src/train_free_demand_field.py` |
| G4 传播层与区域读出 | `src/demand_field_nbfnet.py` |
| soft Spearman、残差门与 validation checkpoint | `scripts/train_demand_field_nbfnet.py` |
| BRIDGE-B Top-K/预算/冲突目标与快照门 | `scripts/train_demand_field_nbfnet.py` |
| hard-disjoint 选择 | `scripts/evaluate_non_overlapping_selection.py` |
| C++ 同实现查询 | `cpp/` |

## 不得由论文数字反推的结论

- Spearman 提高不等于 Top-K、展开节点或墙钟全面提高；
- 单区域标签不能相加为多区域在线收益；
- Porto 稳定未来窗口不等于强分布漂移鲁棒；
- 空间覆盖更广不等于已证明人口或偏远地区公平；
- C++ 的 Porto 平均加速不能外推为所有城市平均加速。
