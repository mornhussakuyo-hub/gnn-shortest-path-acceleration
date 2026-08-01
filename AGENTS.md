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

- 一号服务器使用 `.server.env`，二号服务器使用 `.server2.env`；两者字段名相同。切换服务器时
  重新 `source` 对应文件并导出变量，不要把两台服务器的连接参数写进文档或日志。
- 环境文件中的 `SERVER_SSH_COMMAND` 仍会交互式询问密码；本机没有 `sshpass` 和可用的
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

- 当前研究主线分为三部分：确定性 OD 图传播 Z0、神经网络传播与优化可解释性、可选的 Z0 神经
  排序残差。不能再把“必须训练成功一个 32 层黑盒 NBFNet”当作项目成败标准。
- Z0 只使用历史 OD 与静态道路图，不使用标签；神经模型固定采用
  `score = frozen_z0 + learned_residual`。只有学习残差在重复种子 validation 上稳定超过 Z0，
  才能写成监督学习贡献。
- NBFNet 仍使用 OD 条件双向传播：起点场沿正图、终点场沿反图传播，目标是研究历史需求如何沿
  道路图传播并预测候选区域价值；起点塔和终点塔参数不共享。
- 需求原型只由历史窗口 OD 和静态道路图构造，不使用历史最短路径；至少保存起点集合、终点集合和出现权重。
- 时间切分必须避免泄漏：较早历史窗口 H 构造输入，紧随其后的标签窗口 Y 构造真实区域收益，两者绝不重叠。
- 当前固定口径：H 为全部 98,082 条 OD 按时间排序后的前 35%，共 34,328 条；Y 为时间比例 `[0.35, 0.70)`，完整窗口约 34,329 条；正式标签从 Y 中以随机种子 42 抽取 2,000 条查询。
- 阶段六未来窗口协议已冻结：F 为时间比例 `[0.70, 1.00)`，完整窗口约 29,425 条；仍以随机
  种子 42 抽取 2,000 条正式查询。Z0 输入继续固定为最早 H，不读取 Y/F 更新需求场；候选池、
  单区域工作量定义和关闭端点缓存均不变。
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
- 四组前 22 epoch 几乎不更新、第 23 epoch 同步翻转。后续 B1/B2 已直接记录：默认 FP16 scale
  `65536` 的前 22 次 optimizer step 全部跳过；scale 从 `1` 开始时在降至
  `0.015625=1/64` 后于 epoch 7 首次成功。旧平台期已确定来自 FP16 backward 溢出和
  GradScaler 跳步，不是传播逐 epoch 到达中部，也不是学习率过低。

## Z0、rank-first 与梯度解剖结论

- Z0 是无参数、无标签的双向固定均值扩散，在深度 `1/2/4/8/16/32` 等权读出。validation /
  holdout Spearman 为 `0.9411 / 0.9356`，holdout NDCG@5/10/18 为
  `0.9307 / 0.9642 / 0.9745`。它已是主方法与所有神经残差的冻结基线。
- Z1 为随机 32 层传播网络只前向，五种子 holdout `|Spearman|=0.9376 ± 0.0285`；随机头主要
  选择同一潜在排序轴的正负号。Z2 只用 train 选择全局符号，不能保证 K=5/10 头部稳定。
- 完整 rank-first 模型 seed 42 从 `-0.9020` 开始，每轮 `inf→0`；Z0 残差模型首轮梯度
  `8.495e-3` 正常，第二轮开始 `inf→0`。两组机制证据充分后已正常停止，不再空转。
- CUDA FP32 深度扫描已完成。1/2/4/8/16/32 层 FP64 全局梯度范数约为
  `6.40e-2 / 5.75e-2 / 4.40e-1 / 8.38e2 / 8.28e9 / 2.34e23`。
- 32 层 418,183 个参数梯度元素全部有限，最大绝对值约 `1.14e23`；原始 FP32 全局范数在
  平方求和时溢出为 `inf`。因此 BF16/FP32 日志的 `inf` 不是参数元素先成为 Inf，而是深层反向
  指数放大后触发的 FP32 范数累计溢出；全局裁剪随后把所有梯度乘成近零。
- 32 层前向激活始终约 `2.3～3.4`，反向 hook 从第 32 层约 `1e-6` 放大到第 1 层约 `1e19`，
  最大参数梯度集中在最早的 LayerNorm bias。根因定位为无真实恒等路径的深层传播与归一化反向
  几何；尚不能在正交消融前把 LayerNorm 单独写成唯一根因。
- loss scale `1/64` 解缩放后 FP64 范数仍为 `2.34e23`，FP32 范数仍为 `inf`；GradScaler 只能
  缓解 FP16 中间溢出，不是结构性修复。
- 冻结主干后输出头 8 个 step 全部有效，validation Spearman `-0.9020→-0.8532`，证明
  pairwise loss 与输出头能够学习；阻断反转的是主干梯度污染。
- G3 恒定 `5e-3` 三种子虽有 Spearman 平均 `+0.004807`，但 NDCG@5 平均 `-0.003448`；
  plateau 调度版 Spearman 平均仅 `+0.002755`、NDCG@5 平均 `-0.005220`。调度版 seed 42
  也劣于恒定协议，因此 S3 不采用、S4 不通过，G 线按预注册停止，不运行 S5 或 P 预训练。
- 完整结果位于 `results/gnn_v2/nbfnet_propagation/gradient_anatomy/`，主要报告为
  `reports/阶段四_NBFNet传播诊断与深层纯传播实验.md`。

## BRIDGE 与成本感知扩展

- G4 是本项目自主组合设计的任务化神经模型：32 层 G3 稳定双向 OD 传播、多深度读出、
  `frozen_z0 + learned_residual` 和 soft Spearman 目标。论文暂命名为 **BRIDGE**
  (Bidirectional Residual Index Deployment Gain Estimator)；定稿前必须做同名检索。
- G4/BRIDGE 在 Porto 和 Chicago 三种子 holdout 与 future 全局 Spearman 均稳定高于 Z0，
  但 Top-K 和在线部署指标不全面超过 Z0；因此它是“全局排序神经增强”，不是 Z0 的全面替代。
- 阶段七新增 G5，论文暂称 **BRIDGE-B**：在 G4 上直接加入头部收益、shortcut 预算和
  区域冲突目标，每轮保存 validation 快照，并用真实 K=18 非重叠在线部署作最终门。
- G5-S0 两城 seed 42、12 epoch 全部有效；两城 validation/global Spearman 和 shortcut 成本均改善，
  但真实 K=18 在线展开节点均退化。这是目标错位，不是训练失败，不得直接扩大长训。
- S1 只修正 split 内 Top-K 比例：全候选部署仍为 `18/1200`，train 改为 `13/840`，
  validation 改为 `3/180`；学习率、损失权重、结构、seed 和 12 epoch 保持不变。

## 服务器训练最终状态

- 两台服务器仓库均为 `~/gnn-shortest-path-acceleration`，GPU 均为 RTX 4090 D 24 GB。
- 一号使用 `.server.env`，二号使用 `.server2.env`；SSH 必须加 `-F /dev/null`。
- 两机已确认快进到 `ee6b92b`，`tests.test_demand_field_nbfnet` 各 25 项全部通过。
- S1 两城 seed 42、12 epoch、冻结全候选和 K=18 Y/F 精确在线门已全部完成，服务器
  GPU 已回到约 15 MiB、0%。结果和完整原始日志已回传本机。
- 二号机访问 GitHub 时曾超时，本轮通过本机生成的增量 `git bundle` 快进；没有修改、删除或
  覆盖任何服务器训练产物。
- 服务器保留 S0 与旧 G4 未跟踪产物，不删除。本机的旧 launcher/PID 和新同步原始日志也
  不删除；提交时不得用宽泛 `git add`误纳入。

## 后续实验顺序

1. S1 通过了 validation 头部安全保持门：两城全局/holdout Spearman 改善，最佳 checkpoint 的
   Top-3 收益与成本均不劣于 Z0；但尚未通过全候选增量部署门，Top-18 与 S0 完全相同；
   Porto Y/F 展开节点相对 Z0 多 `10.346/14.495`，Chicago 多 `73.720/68.171`。
2. **下一步执行 S2**：仅将头部收益权重从 `0.25` 增大到 `1.0`，其余与 S1 完全一致；
   仍是两城 seed 42、12 epoch CUDA 短训，不进入长训。
3. S2 完成后仍必须冻结最佳 validation checkpoint、导出 1,200 全候选分数，再用 K=18
   hard-disjoint Y/F 精确在线门决策。
4. 若 S2 仍不改变 Top-18，停止单纯放大 gain weight，改为设计直接反映扫边/查询工作量的
   部署代理目标。
5. 任何不利结果均保留并写入阶段日志；不事后删除种子、epoch 或在线方法。

## 阶段七时间预算

- 32 层 G3/BRIDGE 单 epoch 实测约 60～70 秒；S1 两城 12 epoch 并行约 15～20 分钟。
- 冻结全候选导出约数分钟；两城 K=18、Y/F 在线复核的时间视现有缓存与服务器负载而定。
- 未通过真实部署门前不启动长训或三/五种子队列。

## 仓库状态交接

- 当前分支为 `main`；S1 split-scaled Top-K 代码、测试和阶段报告提交为
  `9624f11 按候选比例校准部署目标TopK`。
- G5 训练目标实现为 `f801d97`；冻结推理与诊断修正为 `630d916` / `0c7b8d3`。
- S0/S1 两城训练、冻结全候选和 K=18 在线产物已回传本机，位于两城
  `results/.../g5_cost_aware_exploration/s0_short_seed42*` 与 `s1_scaled_topk_short_seed42*`，尚未统一筛选提交。
- 当前详细执行记录：`reports/阶段七_成本感知神经部署优化.md`。
- 已完成的总结与纸面边界：`reports/最终研究成果与论文结论.md`、
  `reports/文献新颖性审查与论文定位.md`。阶段七若成功，再统一更新 README、总结、论文初稿和 PDF。
