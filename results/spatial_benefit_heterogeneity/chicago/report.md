# Chicago 空间收益异质性

主量为神经方法相对 Z0 的逐查询展开节点差；正值表示神经方法更少展开。热点仅由历史 H 的端点需求定义，Y/F 不参与空间分层。

| 方法 | 窗口 | 分层 | 平均差 | 改善查询 | 退化查询 |
| --- | --- | --- | ---: | ---: | ---: |
| BRIDGE | current_y | head | -65.260 | 43.25% | 52.57% |
| BRIDGE | current_y | non_head | -4.424 | 16.67% | 10.61% |
| BRIDGE | future_f | head | -57.567 | 44.82% | 51.32% |
| BRIDGE | future_f | non_head | +25.809 | 22.22% | 9.26% |
| BRIDGE-B | current_y | head | -71.685 | 33.46% | 51.02% |
| BRIDGE-B | current_y | non_head | +15.260 | 23.48% | 1.52% |
| BRIDGE-B | future_f | head | -77.818 | 32.93% | 51.96% |
| BRIDGE-B | future_f | non_head | +29.330 | 32.41% | 5.56% |

## 预注册解释门

- **BRIDGE**：`no_stable_spatial_broadening`；非头部扩展门 `未通过`。
- **BRIDGE-B**：`no_stable_spatial_broadening`；非头部扩展门 `未通过`。

固定 16×16 米制等边方格、最少 5 条查询着色，共 128 个方法—窗口有效网格记录。
改善、持平和退化记录均保留；本分析是已解锁 Y/F 上的机制诊断，不是新的时间外确认。
