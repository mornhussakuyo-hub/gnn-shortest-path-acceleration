# 面向历史 OD 负载的无路径监督精确最短路压缩索引

本项目研究如何只使用历史起点、终点和道路图，在固定索引预算下选择值得压缩的
连通区域，从而减少未来最短路查询开销。项目不使用历史最短路径作为监督标签，
不让神经网络预测路径；压缩表由精确算法离线物化，在线结果必须与原图最短路完全
一致。

## 研究计划

```text
Porto OD 与 OSM 路网
        ↓
精确最短路基线与全量评测框架
        ↓
随机区域、OD 热点区域等传统压缩基线
        ↓
离线物化压缩图并验证真实在线收益
        ↓
参数与预算扫描，确定公平比较条件
        ↓
GNN 学习节点和候选区域的压缩价值
        ↓
同预算对比、消融实验和跨城市验证
```

GNN 位于第 4 步。它不替代 Dijkstra，而是根据道路拓扑和历史 OD 需求，为节点或
候选区域输出压缩价值分数。模型选出区域后，仍由现有精确预处理程序构建 shortcut
和物化压缩图，再由双向 Dijkstra 完成在线查询。

## 研究进度与成果

| 步骤 | 状态 | 已有成果 |
| --- | --- | --- |
| 1. 数据与评测框架 | 已完成 | 构建 Porto 道路图和 98,082 条可用 OD；实现 Dijkstra、双向 Dijkstra、逐查询明细和正确性评测。 |
| 2. 传统压缩与物化查询 | 已完成 | 实现随机、OD 热点区域；离线构建节点三态表、shortcut 和压缩图；全量配对实验正确率 100%，在线耗时下降。 |
| 3. 参数与预算扫描 | 已完成 | 完成 42 组全量控制变量实验，全部正确率 100%；推荐区域数 100、区域大小 512，并以约 3.8～4 万条 shortcut 作为第一版公平预算。 |
| 4. 可学习双向需求场 | 进行中，第一版与阶段 0 已完成 | 完成 GPU GraphSAGE 节点打分和核心 72 组验证集消融；第二版已冻结为“真实区域收益驱动的有向双向需求传播”，精确端点局部接入、有限 LRU 缓存和 98,082 条 OD 全量配对均已完成。 |
| 5. 最终对比实验 | 未开始 | 需要在相同预算下比较 GNN、随机区域和 OD 热点区域，并完成消融、稳定性和跨城市实验。 |

各阶段的目标、成果、证据、遗留问题和完成标准统一记录在
[`reports/`](reports/README.md) 中。
第二版的冻结研究主线与实施顺序见
[`reports/阶段四_GNN第二版实施计划.md`](reports/阶段四_GNN第二版实施计划.md)。
第二版到批量持续学习的版本边界见
[`reports/阶段四_持续学习闭环演进路线.md`](reports/阶段四_持续学习闭环演进路线.md)。

当前物化压缩图的全量配对结果：

| 方法 | 基线平均耗时 | 压缩平均耗时 | 在线耗时变化 | P95 变化 | 展开节点变化 | 正确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 随机区域 | 27.378 ms | 25.885 ms | **-5.45%** | -4.97% | -9.08% | 100% |
| OD 热点区域 | 28.122 ms | 27.235 ms | **-3.15%** | -2.07% | -5.57% | 100% |

这组结果证明离线物化区域压缩图能够降低平均在线查询开销。阶段三随后完成了公平预算
扫描，阶段四第一版 GNN 已在相近 shortcut 预算下优于随机五种子均值。核心 72 组
消融进一步表明：第一版主要收益来自端点风险与几何 Proxy，GraphSAGE 和固定无向扩散
尚未证明独立贡献。传统策略最终验证结果见
[`results/regions/porto_98082queries_r200_s512_paired_final_report.md`](results/regions/porto_98082queries_r200_s512_paired_final_report.md)。

## 数据准备

本仓库使用 Porto 出租车轨迹构造 OD 查询，并使用 OpenStreetMap Portugal 路网作为
最短路实验图。原始下载文件体积较大，因此不会提交到 Git 仓库。

在仓库根目录运行下面的命令即可一键准备数据：

```bash
python scripts/prepare_porto_data.py
```

Windows 上如果 `python` 不在 PATH 中，可以改用：

```powershell
py scripts\prepare_porto_data.py
```

脚本会自动创建 `.venv/`，安装 `requirements.txt` 中的依赖，下载 UCI Porto
taxi 数据集和 Geofabrik Portugal OSM PBF，然后生成：

- `data/processed/porto/波尔图起终点样本_10万.csv`
- `data/processed/porto/波尔图道路节点.csv`
- `data/processed/porto/波尔图道路边.csv`
- `data/processed/porto/波尔图可用起终点节点查询_200米.csv`
- `data/processed/porto/波尔图起终点吸附质量报告.md`

常用参数：

```bash
python scripts/prepare_porto_data.py --od-limit 10000
python scripts/prepare_porto_data.py --force
python scripts/prepare_porto_data.py --skip-road-plot
python scripts/prepare_porto_data.py --skip-dependency-install
```

- `--od-limit 10000`：只抽取 1 万条 OD 样本，适合快速测试。
- `--force`：重新下载并重新生成已有结果。
- `--skip-road-plot`：跳过道路底图热力图，减少运行时间。
- `--skip-dependency-install`：复用已有 `.venv`，不执行 pip，适合离线复查现有数据。

完整数据准备预计占用数 GB 磁盘空间。数据来源包括 UCI Taxi Service
Trajectory Prediction Challenge, ECML PKDD 2015 和 Geofabrik Portugal OSM
extract。

### 手动准备数据

如果一键脚本运行失败，按下面四步操作即可。所有命令都需要在仓库根目录执行。

#### 1. 下载数据

##### 最新数据

以下链接指向官方数据源，适合重新获取最新的原始数据：

1. [最新数据：UCI Porto taxi 数据集（ZIP）](https://archive.ics.uci.edu/static/public/339/taxi%2Bservice%2Btrajectory%2Bprediction%2Bchallenge%2Becml%2Bpkdd%2B2015.zip)
2. [最新数据：Geofabrik Portugal 路网（OSM PBF）](https://download.geofabrik.de/europe/portugal-latest.osm.pbf)

##### 实验版本数据

为了准确复现仓库中已有实验，本项目实际使用的数据保存在 Google Drive：

- [实验版本：`uci_porto_taxi.zip`（Google Drive）](https://drive.google.com/file/d/1aCHdkVaQ9IhW82zhA7c_267j0MDNgN2y/view?usp=drive_link)
- [实验版本：`portugal-latest.osm.pbf`（Google Drive）](https://drive.google.com/file/d/150c-FUbCGe_AArNQtVzxJEEBSKJtNpMb/view?usp=drive_link)

实验版本数据是生成当前 `processed` CSV 和已有结果时使用的固定版本。Google Drive
文件可访问时，优先使用该版本复现实验；只有需要基于最新路网重新生成数据时，才
使用上面的官方“最新数据”链接。

使用官方最新数据时，第一个文件是 ZIP 压缩包，下载后重命名为
`uci_porto_taxi.zip`。打开它，从中找到并取出 `train.csv.zip`，但不要继续解压
`train.csv.zip`。第二个文件应保持 PBF 格式，文件名必须是
`portugal-latest.osm.pbf`。

#### 2. 按固定名称放置文件

在项目中创建 `data/compressed/porto/`，最终目录必须是：

```text
data/compressed/porto/
├── uci_porto_taxi.zip          # UCI 下载得到的外层 ZIP
├── train.csv.zip               # 从 UCI 外层 ZIP 中取出的内层 ZIP
└── portugal-latest.osm.pbf     # Geofabrik 下载得到的 PBF
```

脚本会直接读取 `train.csv.zip` 内部的 `train.csv`，所以不要把它解压为 1.9 GB 左右的
CSV 文件。文件名和路径必须完全一致。

#### 3. 配置 Python 环境

建议使用 Python 3.11 或更新版本。

Windows（PowerShell）：

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux / macOS（Bash）：

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

如果 Debian / Ubuntu 提示缺少 `venv`，先运行
`sudo apt install python3-venv`，然后重新创建虚拟环境。

#### 4. 运行数据处理脚本

Windows（PowerShell）：

```powershell
.\.venv\Scripts\python.exe scripts\generate_porto_od_heatmap.py --limit 100000
.\.venv\Scripts\python.exe scripts\build_porto_road_graph_and_snap_od.py
```

Linux / macOS（Bash）：

```bash
.venv/bin/python scripts/generate_porto_od_heatmap.py --limit 100000
.venv/bin/python scripts/build_porto_road_graph_and_snap_od.py
```

第一条命令从出租车数据中提取 10 万条 OD，第二条命令从 PBF 中抽取 Porto 路网并
将 OD 吸附到道路节点。处理完成后，研究所需的主要文件是：

```text
data/processed/porto/波尔图道路节点.csv
data/processed/porto/波尔图道路边.csv
data/processed/porto/波尔图可用起终点节点查询_200米.csv
```

如果这三个 CSV 已经存在，可以跳过原始数据下载和处理，只配置 Python 环境后直接
运行实验。标准结果约为 133,839 个路网节点、221,589 条有向边和 98,082 条可用 OD；
由于 Geofabrik 的 `latest` 路网持续更新，不同下载日期的数量可能略有变化。

## 运行最短路 baseline

数据准备完成后，可以运行 Dijkstra 和双向 Dijkstra baseline：

```bash
.venv/bin/python scripts/run_baselines.py
```

默认读取 `data/processed/porto/` 下的道路节点、道路边和 200 米吸附阈值内的
可用 OD 查询。输出结果位于：

- `results/baselines/porto_allqueries_summary.csv`
- `results/baselines/porto_allqueries_details.csv`

当前 98,082 条 Porto 可用 OD 查询全量 baseline 结果：

| 方法 | 可达查询 | 平均耗时 ms | p95 耗时 ms | 平均展开节点 | 正确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dijkstra | 97009/98082 | 17.976 | 92.645 | 24831.37 | 1.000000 |
| 双向 Dijkstra | 97009/98082 | 8.339 | 37.341 | 10741.70 | 1.000000 |

## 运行物化压缩图实验

离线构建随机区域和 OD 热点区域的压缩图，并运行普通全量实验：

```bash
python scripts/run_region_experiments.py --region-count 200 --region-size 512
```

最终性能结论应使用配对验证。它会在同一工作进程中连续执行每条 OD 的基线与
压缩查询，并交替执行顺序，减少机器负载和缓存差异对计时的影响：

```bash
python scripts/verify_materialized_queries.py --region-count 200 --region-size 512
```

两个脚本默认使用全部 CPU 核心并运行全部 98,082 条可用 OD。区域生成、shortcut
计算和压缩图构建均属于离线预处理，不计入在线查询耗时。最终结果位于：

- `results/regions/porto_98082queries_r200_s512_paired_summary.csv`
- `results/regions/porto_98082queries_r200_s512_paired_details.csv`
- `results/regions/porto_98082queries_r200_s512_paired_final_report.md`

## 运行参数与预算扫描

正式扫描采用控制变量法：固定区域大小为 `512`，依次测试区域数量
`50、100、200、400`；再固定区域数量为 `200`，依次测试区域大小
`128、256、512、1024`。随机区域每组运行 5 个随机种子，OD 热点区域每组运行
一次，共 42 组。每组都使用全部 98,082 条 OD，并在同一进程内配对执行基线和
压缩查询。

先查看计划执行的配置，不会启动实验：

```bash
python scripts/run_parameter_scan.py --dry-run
```

确认后运行正式扫描：

```bash
python scripts/run_parameter_scan.py
```

Windows 上也可以运行：

```powershell
py scripts\run_parameter_scan.py
```

脚本默认使用全部 CPU 核心，预计需要数小时，具体时间取决于机器。每完成一组就会
立即把结果写入：

- `results/parameter_scan/porto_parameter_scan.csv`

如果运行中断，重新执行同一条命令即可继续，已经写入 CSV 的配置会被自动跳过。
只有确定要清空已有进度并从头运行时才使用：

```bash
python scripts/run_parameter_scan.py --restart
```

CSV 包含每组配置的实际区域数、shortcut 数、压缩图规模、回退率、预处理时间、
平均与 P95 在线耗时、展开节点变化、查询加速比例和正确率。42 组全量扫描已经完成，
全部配置正确率均为 100%。随机区域的推荐默认配置为区域数 `100`、区域大小 `512`，
平均在线耗时降低 13.21%，平均展开节点数降低 15.80%。完整分析见
[`reports/阶段三_参数与预算扫描.md`](reports/阶段三_参数与预算扫描.md)。

## 第一版 GNN 归档

第一版 GPU GraphSAGE、解析基线、评测脚本、72 组核心消融证据和两份结果报告已经冻结
到 [`archive/gnn_v1/`](archive/gnn_v1/README.md)。这些文件不再占用当前 `src/`、
`scripts/`、`tests/` 和 `results/` 的主线入口；需要复核旧实验时按归档说明创建独立
PyTorch/CUDA 环境。

第一版的正式结论保留为：历史 OD 能指导区域选择，但 GraphSAGE 相对 MLP 的独立贡献
没有成立，主要收益来自端点风险与几何 Proxy。因此当前主线不会继续扩大第一版代理目标
的消融矩阵。

## 第二版：可学习双向需求场

第二版不再把增加模型组件和扩大候选系统作为第一目标，而是集中验证最初的研究假设：
只使用历史 OD 起终点和有向道路图，能否根据后续时间窗口中的真实区域压缩收益，学习
起点需求的正向传播与终点需求的反向传播。

第二版已完成内部端点局部接入，消除了整图回退对区域价值的偏置；下一步固定候选区域，
用精确查询工作量差生成无路径监督的区域收益标签。损失从区域价值预测头反向传播到
双向需求场，学习边注意力、传播门控和不同传播深度的组合。核心对比必须包括原始频率、
第一版固定扩散、单向需求场、无向传播和无消息传递 MLP。

端点局部接入专项实现、全量正确性、缓存命中率和同进程缓存配对结果见
[`reports/阶段四_精确端点局部接入.md`](reports/阶段四_精确端点局部接入.md)。

完整设计、监督边界、阶段门和暂缓项见
[`reports/阶段四_GNN第二版实施计划.md`](reports/阶段四_GNN第二版实施计划.md)。

当前主线明确隔离查询层、需求传播层、成本与标签层、部署反馈层。端点局部接入属于精确
查询算法，不作为 GNN 的局部输入特征；持续学习按 OD 批次更新，不为单条查询重训模型
或重建索引。
