# 零训练需求传播基线

- Z0 holdout Spearman：`0.9412`
- Z0 holdout NDCG@5/10/18：`0.9405 / 0.9297 / 0.9225`
- Z0 与 midpoint Proxy 的 holdout Spearman：`0.9612`
- Z2 五种子 holdout Spearman：`0.9242 ± 0.0656`

| Seed | Z1 Validation | Z1 Holdout | Train 定向 | Z2 Holdout |
| ---: | ---: | ---: | ---: | ---: |
| 42 | -0.9366 | -0.9586 | -1 | 0.9586 |
| 43 | 0.9352 | 0.9561 | +1 | 0.9561 |
| 44 | -0.9328 | -0.9529 | -1 | 0.9529 |
| 45 | -0.9382 | -0.9602 | -1 | 0.9602 |
| 46 | 0.8138 | 0.7931 | +1 | 0.7931 |

Z0 完全不使用可学习参数或标签；Z1 保留随机初始化网络但不训练；
Z2 只使用 train split 决定一个全局正负号，不使用 validation 或 holdout。
