# AIC 仓库协作与研究交接

## 用户偏好与操作边界

- 使用中文交流，结论简短、直接，避免重复解释和无效 token 消耗。
- 长训练无需等待结束：启动后确认进程、GPU、日志均正常即可；用户允许等待较长时间，但不希望频繁轮询。
- 所有神经网络训练必须使用 CUDA，不要改用 CPU 训练。CPU 只可用于轻量、只读的数据检查或单元测试，且不要把 CPU 试跑写成正式实验结论。
- 本机 `/home/MornHus/Projects/AIC` 是唯一的代码、文档、提交和推送主机；服务器只提供训练算力，不在服务器编辑、整理、提交或推送代码。
- 服务器旧产物优先归档，不直接删除。服务器连接配置在被 Git 忽略的 `.server.env`，不要在日志、文档或回复中泄露其中的密码。
- 报告必须跟随当前产物及时更新；训练结果同步回本机后，应统一 README、报告、编号文档及其 PDF 口径。
- 不得为了迎合结论事后挑选或删除数据。可以依据预先说明的数据质量规则清洗噪声、异常尺寸或近重复样本，并如实记录。

## 服务器连接固定流程

- `.server.env` 中的 `SERVER_SSH_COMMAND` 仍会交互式询问密码；本机没有 `sshpass` 和可用的
  `ssh-askpass`，不要直接执行它并反复试错。
- 固定使用系统已有的 `/usr/bin/expect`，从环境变量读取密码；命令和日志都不得输出
  `SERVER_PASSWORD`。SSH 必须保留 `-F /dev/null`。
- 在仓库根目录先定义下面的函数，之后所有服务器只读检查、同步和启动都通过
  `server_ssh '<远端命令>'` 执行：

```bash
source .server.env
export SERVER_HOST SERVER_PORT SERVER_USER SERVER_PASSWORD

server_ssh() {
  REMOTE_COMMAND="$1" expect -c '
    set timeout 60
    log_user 0
    spawn ssh -F /dev/null -o StrictHostKeyChecking=no \
      -p $env(SERVER_PORT) $env(SERVER_USER)@$env(SERVER_HOST) \
      $env(REMOTE_COMMAND)
    expect "*assword:*"
    send -- "$env(SERVER_PASSWORD)\r"
    log_user 1
    expect eof
    lassign [wait] pid spawnid os_error exit_code
    exit $exit_code
  '
}

server_scp_from() {
  REMOTE_PATH="$1" LOCAL_PATH="$2" expect -c '
    set timeout -1
    log_user 0
    spawn scp -r -F /dev/null -o StrictHostKeyChecking=no \
      -P $env(SERVER_PORT) \
      "$env(SERVER_USER)@$env(SERVER_HOST):$env(REMOTE_PATH)" \
      $env(LOCAL_PATH)
    expect "*assword:*"
    send -- "$env(SERVER_PASSWORD)\r"
    log_user 1
    expect eof
    lassign [wait] pid spawnid os_error exit_code
    exit $exit_code
  '
}
```

- 已验证的连接检查：

```bash
server_ssh 'cd ~/gnn-shortest-path-acceleration && git status --short && git rev-parse --short HEAD && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader'
```

- 同步前先看服务器 `git status --short`。若训练产物以未跟踪目录阻挡 `git pull`，先移动到
  `~/aic-training-archive/`，不得删除；随后只允许服务器 fast-forward 拉取本机已推送的
  `main`。服务器不编辑、不提交、不推送。
- 后台训练统一用 `nohup env PYTHONUNBUFFERED=1 ... > launcher.log 2>&1 < /dev/null &`，保存
  PID 后只检查一次进程、`nvidia-smi` 和日志；三者正常即可停止轮询。
- 结果完成后通过
  `server_scp_from 'gnn-shortest-path-acceleration/<远端产物目录>' '<本机父目录>'`
  回传；回传后先校验摘要和文件完整性，再在本机更新报告、提交和推送。

## 第二版已经确定的方向

- 第二版唯一主模型确定为基于 NBFNet 的 OD 条件双向传播网络，目标是研究历史需求如何沿道路图传播并预测候选区域价值。
- 需求原型只由历史窗口 OD 和静态道路图构造，不使用历史最短路径；至少保存起点集合、终点集合和出现权重。
- 时间切分必须避免泄漏：较早历史窗口 H 构造输入，紧随其后的标签窗口 Y 构造真实区域收益，两者绝不重叠。
- 当前固定口径：H 为全部 98,082 条 OD 按时间排序后的前 35%，共 34,328 条；Y 为时间比例 `[0.35, 0.70)`，完整窗口约 34,329 条；正式标签从 Y 中以随机种子 42 抽取 2,000 条查询。
- 候选区域已从“高频端点附近选择”改为全图随机且尽量空间均匀，以避免候选本身围绕需求热点、削弱区域压缩研究意义。
- 当前正式标签已经完成：`region_training_labels.csv` 有 1,200 条区域记录，每条 `label_query_count=2000`；`label_manifest.json` 为 1,200/1,200、`status=complete`。任何“1,200×2,000 尚未完成”的说法均为旧口径。
- 可选 LRU 缓存、候选池是否应被端到端扩散替代等问题继续暂缓；纯传播验证已完成，后续只补同口径对照、重复种子和未来窗口。

## 数据切分与当前数据口径

- 候选切分已由随机切分改为 Jaccard 重叠连通组隔离，阈值为 `0.50`。
- 当前有 521 个重叠组，最大组 12；train / validation / holdout 为 `840 / 180 / 180`。
- 跨 split 不存在 `Jaccard >= 0.50` 的候选对。
- 当前数据摘要 SHA-256：`e9b1bc149a6a94dd744305e41817c3a2b9447cd5e3ba5a17c71fea38db2bf261`。
- 原随机 split 存在严重空间污染：最近重叠标签预测的 Spearman 可达约 `0.9936`，因此旧 split 上的高分不能作为泛化证据。
- Top-18 候选高度重叠，成员冗余系数约 1.59；分析必须同时查看固定 `K=5/10/18` 排名指标和唯一节点/成员冗余。

## 已实现的纯传播实验

- 已实现四个 32 层、约 418k 参数的传播模型：
  - `propagation_deep`
  - `propagation_residual`
  - `propagation_doubling`
  - `propagation_residual_doubling`
- 四种模型均禁止 64 维区域特征直通，禁止最终读出第 0 层，最大传播 32 跳，且不物化稠密多跳边。
- 倍增读出深度为 `1, 2, 4, 8, 16, 32`；残差版本用于缓解深层信号与梯度衰减。
- checkpoint 按 validation Spearman 选择，holdout 不参与模型选择。
- `shuffled_od` 已改为保持起点与终点两侧边际需求的半周期保测度耦合。
- 本轮最终测试为 `28 passed, 13 skipped, 78 subtests passed`；CUDA 相关项在本机跳过，不作为训练结论。编号“从零详解”文档及 PDF 本轮按用户要求不更新。

## 已有实验结论

- 基础五种子结果：Spearman `0.8873 ± 0.0039`，NDCG `0.9821 ± 0.0025`，Top-K 平均收益约 `129.051`。
- 但随机重连、无向图和去边特征在重新训练后几乎恢复基础结果，尚未证明正确道路拓扑的独立贡献。
- 旧模型可以依赖区域特征、第 0 层读出和残差路径绕开传播。
- 旧 6 层模型远小于候选尺度：候选半径中位数约 36 跳，直径约 66 跳。
- 空间隔离新 split 上的四组 32 层纯传播实验全部完成，holdout Spearman 为 `0.9513～0.9594`；模型不能读取 64 维区域特征或第 0 层，因此传播后的历史 OD 表示本身足以预测当前 H→Y 候选区域价值。
- 结构只按 validation 选择：`propagation_doubling` 为 `0.9400`，高于 deep `0.9334`、residual `0.9342`、residual_doubling `0.9393`，因此正式推荐 `propagation_doubling`。组合模型 holdout 最高的 `0.9594` 不用于反向选型。
- 所选倍增模型 holdout NDCG@5/10/18 为 `0.9720 / 0.9821 / 0.9882`，收益为 `171.922 / 158.022 / 142.332`；成员冗余为 `1.744 / 1.786 / 1.750`，仍需非重叠集合选择。
- 新 split 同口径 MLP 五种子已完成：holdout Spearman `0.8696 ± 0.0197`，NDCG@18 `0.9593 ± 0.0143`，Top-18 收益 `136.841 ± 2.764`。所选纯传播 seed 44 的对应值为 `0.9569 / 0.9882 / 142.332`，本轮观察上明显领先，但纯传播仍缺重复种子，不能声称正确道路拓扑必要、多种子稳定或未来时间泛化。
- 新 split midpoint Proxy 已在本机 CPU 完成：holdout Spearman `0.8918`，NDCG@5/10/18 `0.9645 / 0.9454 / 0.9830`，收益 `170.043 / 149.592 / 142.332`。它与 `propagation_doubling` 的 Top-18 候选集合完全相同；纯传播主要改善全局排序和 K=5/10，不能声称在 K=18 固定候选预算上超过 Proxy。
- 四组前 22 epoch 几乎不更新、第 23 epoch 同步翻转。每次前向本来就执行全部 32 层，这不是传播逐 epoch 到达中部；结合 FP16 `GradScaler`、更新前记录 train loss 和更新后计算 validation，最可能是前 22 次 optimizer step 因梯度溢出被跳过。日志未记录 scale，故这是高置信诊断而非直接观测事实。

## 服务器训练最终状态

- 服务器仓库：`~/gnn-shortest-path-acceleration`。
- SSH 必须加 `-F /dev/null`，系统 SSH 配置权限有问题；连接参数从本机 `.server.env` 读取。
- GPU 为 RTX 4090 D 24 GB。
- 实验 ID：`38bfa5d15809b383`。
- 输出目录：`results/gnn_v2/nbfnet_propagation/screening`。
- 启动参数：seed 44、hidden 32、layers 32、prototype batch 4、max epochs 300、patience 60。
- 旧筛选结果已归档到 `~/aic-training-archive/nbfnet_ablation_screening_20260731`。
- 实验已经 `complete`：四组完成、零失败，服务器重启后 GPU 空闲，不需要恢复 runner。
- 完整产物已同步回本机 `results/gnn_v2/nbfnet_propagation/screening/`，包含 manifest、汇总、日志、checkpoint、预测和训练历史。
- 本机与服务器的 `report.md` SHA-256 均为 `d0d617fa08e0c909f64a5fcf2dd0dcee6bdf95d1dceed67e8a517170ca1d66dd`；`manifest.json` 均为 `7539a5468bd4f6fed992d352083a18e30b098a1e2dbce11980ebde680a78383e`。
- 新 split MLP 输出目录为 `results/gnn_v2/mlp_overlap_group_split`；种子 `42～46` 已全部完成，使用 RTX 4090 D CUDA，无报错，完整产物已回传本机。
- 当前正在补跑正式结构 `propagation_doubling` 的重复种子 `42,43,45,46`，输出目录为 `results/gnn_v2/nbfnet_propagation/propagation_doubling_repeats`；配置为 hidden 32、layers 32、prototype batch 4、max epochs 300、patience 60。
- 本轮 runner 启动 PID 为 `8127`。启动后首次健康检查时进程存活，GPU 利用率 100%、显存约 10.9 GiB；seed 42 第 1 epoch 已完成，日志无 traceback、OOM 或 RuntimeError。PID 可能变化，接续时重新读取输出目录中的 `runner.pid`，不要停止 runner。

## 后续实验顺序

1. 先完成当前 `propagation_doubling` 的 seed `42,43,45,46`，与已有 seed 44 合并为五种子稳定性结果；当前 runner 不中途改配置。
2. 再用当前固定协议做调参前消融。现有互斥 `variant` 会让 `degree_rewired` 等条件退回基础架构，必须先拆成 `architecture=propagation_doubling` 与正交 `ablation` 两轴并补测试；随后 `degree_rewired`、`shuffled_od` 正式跑五种子，`undirected`、`no_edge_features`、`origin_only`、`destination_only` 先 seed 44，validation 差异达到 `0.02` 或改变 K=5/10 结论时扩展五种子。
3. 消融完成后才做平台诊断和调参。必须记录 GradScaler scale、step skipped、裁剪前梯度范数、实际学习率、首次有效 step 和首次正 Spearman；比较 FP16、较低 initial scale 与 BF16。
4. 初始化分别验证传播矩阵正交/近单位加正门 bias、小幅非零预测头、浅层倍增深度先验；三项先独立再考虑组合，仍禁止第 0 层读出。
5. 调参最小集合：固定 `1e-3`、固定 `3e-4`、转正后启用的 `ReduceLROnPlateau(0.3, patience=10, min_lr=1e-5)`，再比较 rank weight `0.20/0.50`。只用 seed 42/44 validation 筛选，冻结后跑五种子。
6. 若调参改变正式协议，必须用新协议重新跑 `degree_rewired` 和 `shuffled_od` 五种子，再做未来窗口、非重叠集合选择和精确在线配对评测。
7. 不再继续堆叠新的传播结构；编号“从零详解”文档本轮按用户要求暂不更新。完整规则见 `reports/阶段四_GNN第二版实施计划.md`。

## 仓库状态交接

- 交接前分支为 `main`。
- 本次计划修改前最新提交为 `695cb72 补齐空间隔离Proxy基线`。
- 主要诊断报告：`reports/阶段四_NBFNet传播诊断与深层纯传播实验.md`。
