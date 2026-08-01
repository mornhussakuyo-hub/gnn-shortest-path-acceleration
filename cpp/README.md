# C++ 精确在线性能评测

该目录提供与 Python 压缩索引语义一致的 C++20 实现，用于判断展开节点减少能否转化为真实墙钟
收益。原图双向 Dijkstra 和压缩查询在同一个可执行程序、同一线程、同一计时器内运行，不进行
跨语言绝对耗时对比。

实现保持以下冻结条件：

- 选区固定为现有 `hard_disjoint` 的 K=18 结果；
- 区域内部节点从压缩图删除，边界间距离用区域内受限 Dijkstra 精确物化；
- 内部端点只在所属区域内生成正向或反向边界接入，不使用端点缓存；
- 每个查询先比较原图与压缩图距离，容差为 `1e-6`；
- shortcut 数和内部节点数必须精确重放 Python 冻结结果；平均展开节点允许每查询 `0.1` 以内的
  等长路径 tie 顺序差异（C++ 会主动清理陈旧堆项）；
- 单线程固定 CPU，先预热，再按查询编号与重复轮次交替执行原图/压缩查询顺序。

运行两城正式协议：

```bash
.venv/bin/python scripts/run_cpp_online_benchmark.py \
  --city all --cpu 10 --warmup 2 --repetitions 10
```

脚本会完成 Release 构建、小图 36 个 OD 对自测、两城二进制输入导出、正式计时、Python 结果重放
校验及报告生成。构建目录与二进制输入位于被 Git 忽略的 `build/cpp-online/`；正式摘要位于
`results/cpp_online_benchmark/{porto,chicago}/`。

也可以只构建和运行自测：

```bash
cmake -S cpp -B build/cpp-online -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-online -j2
build/cpp-online/aic_cpp_online_benchmark --self-test
```
