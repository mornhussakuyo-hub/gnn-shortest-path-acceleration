# 正式制图清单

初稿正文目前使用可编译的方框占位。正式图全部重画，不直接截图日志或表格。

## 图 1：端到端方法总图

内容：

1. H/Y/F 时间切分；
2. 全图空间均匀候选与 Jaccard 重叠组隔离；
3. Z0 起点正向场、终点反向场与多尺度区域聚合；
4. G4 冻结 Z0 加神经残差；
5. hard-disjoint 选择；
6. CRP 式选择性物化、端点局部接入和精确查询。

需要特别标注：底层覆盖图为既有结构，本文贡献位于区域排序和部署协议。

## 图 2：两城排序结果

数据来源：

- `results/gnn_v2/nbfnet_propagation/g4_frozen_evaluation/`；
- `results/chicago/gnn_v2/nbfnet_propagation/g4_frozen_evaluation/`；
- 两城 `future_window_z0/`。

建议使用四个面板：holdout Spearman、future Spearman、holdout NDCG@18、future NDCG@18。
G4 显示三种子均值和标准差，Z0 显示确定性水平线。

## 图 3：机制消融

数据来源：

- `results/gnn_v2/z0_orthogonal_ablation/`；
- `results/chicago/gnn_v2/z0_orthogonal_ablation/`。

主图突出正确拓扑与度保持重连；附图展示单侧需求、无向图、边际保持 OD 重耦合、传播深度和
mean/max 池化。不得只展示有利消融。

## 图 4：系统收益与成本

数据来源：

- `results/gnn_v2/multi_region_online_g4/`；
- `results/chicago/gnn_v2/multi_region_online_g4_clean_rerun/`；
- `results/cpp_online_benchmark/{porto,chicago}/`。

建议使用展开节点、扫边、平均耗时、P95 四联图，并同时显示 Porto 与 Chicago。这样可以直观看到
Chicago 中“展开减少但扫边增加”的原因。

## 图 5：典型区域地图（可选）

在每城选择以下案例：

- Z0 与 oracle 均排在头部的区域；
- G4 修正 Z0 错排的区域；
- G4 新增错排的区域；
- 高收益但因重叠约束未被部署的区域。

底图必须保留 OpenStreetMap attribution；颜色只表达需求暴露、残差和精确收益，不推断人口公平
或偏远程度。
