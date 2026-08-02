# Porto 空间收益异质性

主量为神经方法相对 Z0 的逐查询展开节点差；正值表示神经方法更少展开。热点仅由历史 H 的端点需求定义，Y/F 不参与空间分层。

| 方法 | 窗口 | 分层 | 平均差 | 改善查询 | 退化查询 |
| --- | --- | --- | ---: | ---: | ---: |
| BRIDGE | current_y | head | +13.882 | 48.67% | 46.80% |
| BRIDGE | current_y | non_head | +9.000 | 15.38% | 23.08% |
| BRIDGE | future_f | head | +3.713 | 45.99% | 50.38% |
| BRIDGE | future_f | non_head | +10.000 | 26.32% | 21.05% |
| BRIDGE-B | current_y | head | +75.706 | 51.08% | 24.46% |
| BRIDGE-B | current_y | non_head | +40.692 | 30.77% | 7.69% |
| BRIDGE-B | future_f | head | +76.761 | 51.44% | 26.10% |
| BRIDGE-B | future_f | non_head | +17.895 | 21.05% | 10.53% |

## 预注册解释门

- **BRIDGE**：`benefit_expansion`；非头部扩展门 `通过`。
- **BRIDGE-B**：`benefit_expansion`；非头部扩展门 `通过`。

固定 16×16 米制等边方格、最少 5 条查询着色，共 198 个方法—窗口有效网格记录。
改善、持平和退化记录均保留；本分析是已解锁 Y/F 上的机制诊断，不是新的时间外确认。
