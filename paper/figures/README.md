# 正式图表与复现说明

正式图全部从机器可读实验摘要生成，不直接截图日志或手工抄录表格。统一生成命令：

```bash
/home/MornHus/miniconda3/envs/mnist-v2/bin/python scripts/generate_paper_figures.py
```

每张图同时输出矢量 PDF 和高分辨率 PNG；`main_results.csv` 保存所有绘图数值、标准差、状态与
来源文件，`figure_manifest.json` 保存图表口径。

全文统一使用 `scripts/paper_figure_style.py`：Z0 为蓝色、BRIDGE 为橙色、BRIDGE-B 为绿色，
当前窗口为橙色、未来窗口为绿色；空间收益图中改善为蓝色、退化为红色、零值为近白色，道路底图
为浅灰色。所有图统一 DejaVu Sans 字体、标题层级、坐标轴线宽、网格线、图例和矢量字体嵌入。
空间主图采用正方形面板、等比例地图视口和 16×16 米制等边方格，不通过拉伸地图消除城市长宽比差异。道路作为浅灰底层，有效收益方格不透明着色，避免底图覆盖数据层。

## 图 1：端到端方法总图

内容：

1. H/Y/F 时间切分；
2. 全图空间均匀候选与 Jaccard 重叠组隔离；
3. Z0 起点正向场、终点反向场与多尺度区域聚合；
4. G4 冻结 Z0 加神经残差；
5. hard-disjoint 选择；
6. CRP 式选择性物化、端点局部接入和精确查询。

已生成：`method_pipeline.pdf`。底层覆盖图明确标为既有精确结构，本文贡献位于区域排序和部署协议。

## 图 2：两城排序结果

数据来源：

- `results/gnn_v2/nbfnet_propagation/g4_frozen_evaluation/`；
- `results/chicago/gnn_v2/nbfnet_propagation/g4_frozen_evaluation/`；
- 两城 `future_window_z0/`。

已生成：`ranking_results.pdf`。三个面板统一展示 Spearman、NDCG@5、NDCG@18；G4/BRIDGE
显示三种子均值和总体标准差，Z0 为确定性单次结果。

## 图 3：机制消融

数据来源：

- `results/gnn_v2/z0_orthogonal_ablation/`；
- `results/chicago/gnn_v2/z0_orthogonal_ablation/`。

已生成：`mechanism_ablation.pdf`。主图突出正确拓扑与度保持重连；其余面板完整展示单侧需求、
无向图、边际保持 OD 重耦合、传播深度和 mean/max 池化，没有只展示有利消融。

## 图 4：系统收益与成本

数据来源：

- `results/gnn_v2/multi_region_online_g4/`；
- `results/chicago/gnn_v2/multi_region_online_g4_clean_rerun/`；
- `results/cpp_online_benchmark/{porto,chicago}/`。

已生成：`system_results.pdf`。展开节点、扫边、平均耗时、P95 四联图同时显示 Porto 与 Chicago，
可直接观察 Chicago 中“展开减少但扫边增加”的原因。

## 图 5：BRIDGE-B 探索轨迹

已生成：`bridge_b_progression.pdf`。S0--S2 明确标注为 seed 42 方法开发结果；S3 汇总冻结协议下
seeds 42/43/44 的均值与总体标准差。Porto 三种子稳定优于 Z0，Chicago 则完整保留不利在线结果。

## 图 6--7：空间收益异质性

数据来源：

- `results/spatial_benefit_heterogeneity/{porto,chicago}/query_deltas.csv.gz`；
- 同目录 `grid_summary.csv`、`stratum_summary.csv` 与 `summary.json`。

已生成 `spatial_benefit_porto.pdf` 与 `spatial_benefit_chicago.pdf`，完整展示 BRIDGE、BRIDGE-B
相对 Z0 的 Y/F 改善与退化网格。`*_endpoints.pdf` 为起点/终点视角附图。地图只用于冻结后的
机制诊断，不把已解锁 Y/F 包装为新的时间外确认。

## 后续可选：典型区域地图

在每城选择以下案例：

- Z0 与 oracle 均排在头部的区域；
- G4 修正 Z0 错排的区域；
- G4 新增错排的区域；
- 高收益但因重叠约束未被部署的区域。

底图必须保留 OpenStreetMap attribution；颜色只表达需求暴露、残差和精确收益，不推断人口公平
或偏远程度。
