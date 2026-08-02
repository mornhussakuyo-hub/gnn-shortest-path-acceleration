# Chicago 空间收益异质性

主量为神经方法相对 Z0 的逐查询展开节点差；正值表示神经方法更少展开。热点仅由历史 H 的端点需求定义，Y/F 不参与空间分层。

| 方法 | 窗口 | 分层 | 平均差 | 改善查询 | 退化查询 |
| --- | --- | --- | ---: | ---: | ---: |
| BRIDGE | current_y | head | -64.480 | 43.46% | 52.23% |
| BRIDGE | current_y | non_head | -10.564 | 10.83% | 11.67% |
| BRIDGE | future_f | head | -56.345 | 44.91% | 51.15% |
| BRIDGE | future_f | non_head | +13.457 | 17.02% | 6.38% |
| BRIDGE-B | current_y | head | -69.782 | 33.62% | 50.27% |
| BRIDGE-B | current_y | non_head | -5.867 | 20.00% | 8.33% |
| BRIDGE-B | future_f | head | -75.407 | 33.63% | 51.31% |
| BRIDGE-B | future_f | non_head | -3.596 | 18.09% | 11.70% |

## 预注册解释门

- **BRIDGE**：`no_stable_spatial_broadening`；非头部扩展门 `未通过`。
- **BRIDGE-B**：`no_stable_spatial_broadening`；非头部扩展门 `未通过`。

固定 16×16 网格、最少 5 条查询着色，共 202 个方法—窗口有效网格记录。
改善、持平和退化记录均保留；本分析是已解锁 Y/F 上的机制诊断，不是新的时间外确认。
