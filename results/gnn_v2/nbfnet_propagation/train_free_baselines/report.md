# 零训练需求传播基线

- Z0 holdout Spearman：`0.9356`
- Z0 holdout NDCG@5/10/18：`0.9307 / 0.9642 / 0.9745`
- Z0 与 midpoint Proxy 的 holdout Spearman：`0.9120`
- Z2 五种子 holdout Spearman：`0.9376 ± 0.0285`

| Seed | Z1 Validation | Z1 Holdout | Train 定向 | Z2 Holdout |
| ---: | ---: | ---: | ---: | ---: |
| 42 | -0.9021 | -0.9284 | -1 | 0.9284 |
| 43 | 0.9458 | 0.9601 | +1 | 0.9601 |
| 44 | -0.9456 | -0.9572 | -1 | 0.9572 |
| 45 | -0.9340 | -0.9569 | -1 | 0.9569 |
| 46 | 0.7856 | 0.8855 | +1 | 0.8855 |

Z0 完全不使用可学习参数或标签；Z1 保留随机初始化网络但不训练；
Z2 只使用 train split 决定一个全局正负号，不使用 validation 或 holdout。
