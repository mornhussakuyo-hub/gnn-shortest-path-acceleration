# 正负样本构造方案

本文档整理当前项目从历史查询负载构造 GNN 训练样本的可行方案。

当前已经有的数据是 OD 查询对：

```text
data/processed/query_loads/small/queries_train.csv
data/processed/query_loads/small/queries_test.csv
```

也就是说，目前每条历史查询只有：

```text
origin, destination, query_type, source_hotspot, target_hotspot
```

还没有完整的最短路径节点序列。因此正式构造样本前，需要先完成：

```text
OD 查询 -> Dijkstra 最短路 -> 路径节点序列
```

后续所有正负样本都应该从历史最短路径中提取，而不是只从静态拓扑中提取。

## 总体目标

本项目的核心目标不是让 GNN 直接预测最短路，而是让 GNN 学习：

```text
哪些节点、边、区域在历史查询负载中具有相似功能或频繁协同出现
```

这些监督信号之后可以服务于：

- 节点表示学习
- 图聚类
- 图分区
- 图粗化
- 高频通道识别
- 面向历史负载的最短路预处理

## 方案一：节点/边经过频率标签

这是最简单的第一版方案。

### 做法

1. 对训练集中的 OD 查询运行 Dijkstra。
2. 恢复每条查询对应的最短路径。
3. 统计：

```text
node_pass_count
edge_pass_count
origin_count
destination_count
```

4. 根据经过频率构造标签。

### 正样本

可选规则：

```text
node_pass_count > 0 的节点
edge_pass_count > 0 的边
```

或者更严格：

```text
node_pass_count 排名前 10% 的节点
edge_pass_count 排名前 10% 的边
```

### 负样本

可选规则：

```text
node_pass_count = 0 或很低的节点
edge_pass_count = 0 或很低的边
```

### 输出示例

节点样本：

```csv
node,pass_count,origin_count,destination_count,label
100,35,2,0,1
200,0,0,0,0
```

边样本：

```csv
src,dst,weight,pass_count,label
1,2,803,35,1
10,11,120,0,0
```

### 优点

- 实现简单
- 容易验证
- 可以快速得到第一版训练数据

### 缺点

- 更像重要性预测
- 不直接学习节点之间的相似关系
- 对后续聚类帮助有限，但可以作为基础特征

## 方案二：历史路径窗口共现样本

这是当前最推荐的第一版相似度监督方案。

思路类似 NLP 中的 Word2Vec：把历史最短路径看成一句话，把路径上的节点看成单词。

例如一条历史路径是：

```text
[12, 45, 78, 93, 120, 151]
```

设置窗口大小 `k = 3`。如果两个节点在同一条路径中距离不超过 3，就认为它们有局部路径共现关系。

### 正样本

对每条路径使用滑动窗口。

如果：

```text
abs(position_i - position_j) <= k
```

则构造正样本：

```text
(node_i, node_j, label=1)
```

可以统计共现次数：

```text
cooccur_count(node_i, node_j) += 1
```

### 负样本

基础负样本：

```text
随机抽取没有路径窗口共现的节点对
```

更好的困难负样本：

```text
拓扑距离很近，但历史路径窗口中没有共现的节点对
```

例如两个节点在图上相距 1 到 2 跳，但从未在历史路径中形成窗口共现。这类样本能防止模型只学习静态拓扑邻近性。

### 输出示例

```csv
node_i,node_j,label,weight,source
100,105,1,23,path_window
200,315,0,0,random_negative
500,501,0,0,hard_negative
```

其中：

```text
weight = 共现次数
source = path_window / random_negative / hard_negative
```

### 优点

- 非常贴合历史路径行为
- 能学习节点之间的局部路径相似性
- 适合后续节点 embedding 和聚类
- 实现难度适中

### 缺点

- 需要先恢复历史路径
- 高频路径可能产生大量重复样本，需要计数或采样

## 方案三：边级路径样本

该方案直接把历史最短路经过的边作为监督对象。

### 正样本

```text
历史最短路径经过的边
```

### 负样本

```text
历史路径中没有经过的边
```

困难负样本：

```text
与高频边相邻或同区域，但历史中低频或未出现的边
```

### 输出示例

```csv
src,dst,weight,pass_count,label
1,2,803,35,1
10,11,120,0,0
```

### 优点

- 与最短路加速直接相关
- 容易识别高频通道
- 可以直接服务于边权重重估、边重要性预测、路径通道发现

### 缺点

- 如果最终任务是节点聚类，还需要把边级信号转化为节点表示
- 对 GNN 设计有一定要求，可能需要边特征或边预测任务

## 方案四：PPMI 全局共现样本

普通共现次数会受到超级高频节点影响。例如某些交通枢纽会出现在很多路径中，它们会和大量节点共现。如果直接用共现次数，容易导致聚类退化。

因此可以使用 PPMI：

```text
PPMI(i, j) = max(0, log(P(i,j) / (P(i)P(j))))
```

其中：

```text
P(i,j) = 节点 i 和 j 在历史路径中共同出现的概率
P(i) = 节点 i 出现的概率
P(j) = 节点 j 出现的概率
```

### 正样本

```text
PPMI(i, j) 高于阈值的节点对
```

### 负样本

```text
PPMI(i, j) = 0 或很低的节点对
```

### 推荐实现方式

不要计算所有节点对的 PPMI，因为规模过大。

推荐只对候选对计算：

```text
历史路径窗口共现过的节点对
```

也就是在方案二的基础上，把 `cooccur_count` 转成 PPMI 权重。

### 输出示例

```csv
node_i,node_j,label,weight,source
100,105,1,2.31,ppmi
200,315,0,0,ppmi_negative
```

### 优点

- 能惩罚超级高频枢纽
- 比简单共现更稳定
- 学术解释更强

### 缺点

- 实现复杂度更高
- 需要维护节点出现次数和节点对共现次数
- 不适合直接全量计算

## 方案五：基于 OD 服务集合的相似度样本

该方案更关注宏观交通流。

为每个节点维护一个服务集合：

```text
node -> {经过该节点的 OD 查询集合}
```

如果两个节点服务了高度重合的一批 OD 查询，就认为它们在历史负载中具有类似功能。

### 相似度

可以使用 Jaccard Similarity：

```text
Jaccard(A, B) = |S_A ∩ S_B| / |S_A ∪ S_B|
```

其中：

```text
S_A = 节点 A 服务过的 OD 集合
S_B = 节点 B 服务过的 OD 集合
```

### 正样本

```text
Jaccard(A, B) 高于阈值的节点对
```

### 负样本

```text
Jaccard(A, B) 很低或为 0 的节点对
```

### 优点

- 能捕捉宏观交通走廊
- 可以把物理上不相邻但服务同一批查询的节点关联起来
- 适合做 corridor abstraction

### 缺点

- 计算和存储成本较高
- 需要维护大量 OD 集合
- 第一版不建议优先实现

## 负样本策略

负样本非常关键。不能只做完全随机负采样。

### 随机负样本

从全图中随机抽取没有共现关系的节点对：

```text
(node_i, node_j, label=0)
```

优点是简单，缺点是太容易。

### 困难负样本

更推荐加入 hard negatives：

```text
拓扑距离近，但历史路径中不共现
```

例如：

- 1 跳邻居，但没有历史路径窗口共现
- 2 跳邻居，但没有历史路径窗口共现
- 同一地理区域附近，但没有共同服务 OD

困难负样本可以迫使模型学习历史负载特征，而不是退化成“只要拓扑近就相似”。

## 推荐路线

当前项目建议按以下顺序推进：

```text
1. 对 queries_train.csv 跑 Dijkstra，恢复路径节点序列
2. 保存历史路径文件 paths_train.csv
3. 基于路径窗口构造正样本
4. 构造随机负样本
5. 加入拓扑近但历史不共现的困难负样本
6. 统计 node_pass_count 和 edge_pass_count 作为节点/边特征
7. 在路径窗口共现基础上尝试 PPMI 加权
8. 最后再考虑 OD 服务集合与 Jaccard 相似度
```

## 第一版建议输出文件

建议生成以下文件：

```text
data/processed/training/small/
  paths_train.csv
  node_features.csv
  edge_features.csv
  node_pair_samples.csv
```

### paths_train.csv

```csv
query_id,origin,destination,distance,path
1,118822,93718,12345,"118822 118900 119100 93718"
```

### node_features.csv

```csv
node,lon,lat,origin_count,destination_count,pass_count
100,-73.9,40.8,3,1,28
```

### edge_features.csv

```csv
src,dst,weight,pass_count
1,2,803,35
```

### node_pair_samples.csv

```csv
node_i,node_j,label,weight,source
100,105,1,23,path_window
200,315,0,0,random_negative
500,501,0,0,hard_negative
```

## 当前最推荐方案

第一版优先实现：

```text
历史最短路径窗口共现正样本
+ 随机负样本
+ 困难负样本
```

这是当前最稳妥的方案，原因是：

- 实现难度可控
- 和历史查询负载强相关
- 能直接生成节点相似度监督信号
- 适合后续 GNN embedding 与图聚类
- 比简单频率标签更贴近最终研究目标
