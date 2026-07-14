# 第一版 GNN 归档

本目录冻结第一版节点种子价值 GNN 的代码、脚本、报告和轻量实验结果。第一版已经完成，
不再属于当前第二版实现入口；保留它是为了复核结论和复现实验，而不是继续在其代理目标上
迭代。

## 归档内容

- `src/`：第一版固定无向扩散数据构造和 GraphSAGE/MLP 模型。
- `scripts/`：训练、解析基线、精确评测和大型消融入口。
- `reports/`：第一版训练结果与 72 组核心消融报告。
- `results/gnn_v1/`：首次训练与测试结果。
- `results/gnn_ablation/`：从服务器取回的 72 组轻量证据。
- `requirements.txt`：第一版 GPU 训练的独立依赖。

## 运行边界

归档脚本仍从仓库根目录执行，但需要单独安装 PyTorch，并且训练脚本要求 CUDA：

```powershell
py -m venv .venv-gnn-v1
.\.venv-gnn-v1\Scripts\python.exe -m pip install -r archive\gnn_v1\requirements.txt
.\.venv-gnn-v1\Scripts\python.exe archive\gnn_v1\scripts\run_gnn_ablation.py --dry-run
```

当前主线、阶段门和下一步以 `reports/阶段四_GNN第二版实施计划.md` 为准。
