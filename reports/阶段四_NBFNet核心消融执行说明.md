# 阶段四：NBFNet 核心消融执行说明

## 目标

本轮消融只回答基础研究问题：NBFNet 的 Top-K 提升是否来自正确的有向道路拓扑、双向
OD 条件和可学习传播，而不是单点特征、普通平滑或模型容量。

原 `train_demand_field_nbfnet.py` 正式训练命令保持兼容；不传 `--variant` 时仍运行
`base`。大型矩阵由独立调度器添加变体参数，不改变已经完成的基础结果。

残差注意力和倍增传播不进入本轮。它们只在基础传播贡献成立后，以基础、倍增、跨层注意力
残差、二者组合四组独立比较。

## 当前状态

基础 NBFNet 正式五种子训练已经完成。整体 Spearman 与无传播 MLP 基本持平，但 NDCG@K
和 Top-K 收益超过 MLP 与原始频率。服务器当前正在运行固定种子 44 的单种子筛选；筛选
尚未完成，任何运行中的单项指标都只能用于调度与排错，不能提前形成消融结论。完成筛选、
汇总全部对照并决定正式五种子矩阵后，再更新本节结果。

## 已实现变体

| 变体 | 改动 | 主要问题 |
| --- | --- | --- |
| `origin_only` | 删除终点场 | 双向场是否必要 |
| `destination_only` | 删除起点场 | 双向场是否必要 |
| `shared_parameters` | 起终点编码器、token 和传播层共享 | 两个方向是否需要独立参数 |
| `undirected` | 每条边加入反向边 | 道路方向是否重要 |
| `degree_rewired` | 固定随机种子置换边终点，严格保持有向入度和出度序列 | 真实连接关系是否重要 |
| `shuffled_od` | 固定随机种子置换终点需求原型 | 正确 OD 条件是否重要 |
| `fixed_diffusion` | 用无参数的自状态/邻居均值各 0.5 扩散替代可学习消息 | 学习传播是否优于固定平滑 |
| `graphsage` | 用共享 GraphSAGE 均值层替代 NBFNet 消息层 | 专用双向传播是否优于通用 GNN |
| `no_edge_features` | 边长和道路类型全部置零 | 边属性是否有贡献 |
| `no_interactions` | 删除 `O×D` 与 `|O-D|` | 起终点交汇项是否有贡献 |
| `last_layer_only` | 只读取最后传播层 | 可学习多深度融合是否必要 |
| `no_ranking` | ranking loss 权重固定为零 | 排序损失是否有贡献 |

所有随机化变体默认使用 `20260730`，与模型随机种子分离。不同模型种子看到同一份随机
拓扑或 OD 置换，避免把控制图变化混入模型方差。

## 本机一键运行

先跑单种子筛选：

```powershell
.\scripts\run_nbfnet_ablation.ps1 -Mode screening
```

直接运行正式五种子矩阵：

```powershell
.\scripts\run_nbfnet_ablation.ps1 -Mode full
```

筛选模式固定种子 44；正式模式固定种子 42～46。两种模式都使用 32 维、6 层、最多
100 epoch 和耐心值 20。默认原型批次为 8；`undirected` 和 `graphsage` 的对称边集扩大
为两倍，自动使用批次 4 留出显存安全余量。原型分块不改变精确混合训练目标。也可以只
运行指定变体：

```powershell
.\scripts\run_nbfnet_ablation.ps1 `
  -Mode screening `
  -Variants "undirected,degree_rewired,origin_only,destination_only"
```

按 RTX 5060 Ti 基础模型每种子约 50 分钟估算，12 组单种子筛选约需 8～10 小时，完整
五种子矩阵约需 40～50 小时。实际时间会因固定扩散、GraphSAGE、早停和服务器 GPU 而变化。

## 服务器一键后台启动

仓库根目录的 `.server.env` 至少需要：

```text
SERVER_HOST=...
SERVER_PORT=22
SERVER_USER=...
SERVER_SSH_COMMAND=...
```

`SERVER_SSH_COMMAND` 存在时优先使用它；否则由 host、port 和 user 构造原生 SSH 命令。
该配置被 Git 忽略。脚本不会把 `SERVER_PASSWORD` 拼接到命令行，推荐使用 SSH key 或
ssh-agent；未配置免密时，SSH 会正常提示输入密码。

启动正式矩阵：

```powershell
.\scripts\server_nbfnet_ablation.ps1 -Action start -Mode full
```

查看状态和最近日志：

```powershell
.\scripts\server_nbfnet_ablation.ps1 -Action status
.\scripts\server_nbfnet_ablation.ps1 -Action tail -TailLines 100
```

服务器默认仓库路径为 `~/gnn-shortest-path-acceleration`，可用 `-RemoteRepo` 修改。启动
脚本先切换并快进更新 `main`，再用 `nohup` 后台运行，并记录 PID。

## 日志、续跑与结果

```text
results/gnn_v2/nbfnet_ablation/
├── screening/
│   ├── manifest.json
│   ├── ablation_summary.csv
│   ├── report.md
│   ├── logs/
│   └── runs/
└── full/
    ├── manifest.json
    ├── ablation_summary.csv
    ├── report.md
    ├── launcher.log
    ├── runner.pid
    ├── logs/
    │   └── <variant>__seed<seed>.log
    └── runs/
        └── <variant>/seed_<seed>/
```

- 每个子实验的 stdout 和 stderr 实时合并写入独立日志，同时显示在前台终端。
- `manifest.json` 在开始、成功或失败时原子更新，记录命令、时间、返回码和产物路径。
- 每个子实验结束后立即重写汇总 CSV 和 Markdown 报告。
- 重复执行同一命令会验证 summary、数据哈希、候选哈希和训练配置，自动跳过完整结果。
- 中断后重新运行即可续跑；失败项会重新执行，完成项不会重复计算。
- `.runner.lock` 防止同一输出目录被两个调度器同时写入；启动时会自动清理 PID 已失效的
  陈旧锁，仍在运行的锁不会被覆盖。
- 日志、PID 和锁文件不提交 Git；manifest、汇总和正式模型结果可追踪。

## 结果判定顺序

主指标是 holdout Top-K 平均真实收益和 NDCG@K，Spearman 为辅助指标。最低证据链为：

```text
base > undirected
base > degree_rewired
base > origin_only / destination_only
base > fixed_diffusion / graphsage
base > shuffled_od
```

基础模型不能稳定超过这些对照时，不进入倍增传播和跨层注意力残差实验。
