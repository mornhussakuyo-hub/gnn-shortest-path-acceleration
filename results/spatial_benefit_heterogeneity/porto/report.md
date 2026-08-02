# Porto 空间收益异质性

主量为神经方法相对 Z0 的逐查询展开节点差；正值表示神经方法更少展开。热点仅由历史 H 的端点需求定义，Y/F 不参与空间分层。

| 方法 | 窗口 | 分层 | 平均差 | 改善查询 | 退化查询 |
| --- | --- | --- | ---: | ---: | ---: |
| BRIDGE | current_y | head | +13.992 | 48.61% | 46.85% |
| BRIDGE | current_y | non_head | -4.822 | 26.67% | 20.00% |
| BRIDGE | future_f | head | +3.714 | 45.92% | 50.30% |
| BRIDGE | future_f | non_head | +11.000 | 31.25% | 25.00% |
| BRIDGE-B | current_y | head | +75.836 | 51.08% | 24.48% |
| BRIDGE-B | current_y | non_head | +28.067 | 33.33% | 6.67% |
| BRIDGE-B | future_f | head | +76.645 | 51.36% | 26.06% |
| BRIDGE-B | future_f | non_head | +21.250 | 25.00% | 12.50% |

## 预注册解释门

- **BRIDGE**：`no_stable_spatial_broadening`；非头部扩展门 `未通过`。
- **BRIDGE-B**：`benefit_expansion`；非头部扩展门 `通过`。

固定 16×16 网格、最少 5 条查询着色，共 214 个方法—窗口有效网格记录。
改善、持平和退化记录均保留；本分析是已解锁 Y/F 上的机制诊断，不是新的时间外确认。
