# 京东统一路径实验

当前路线实验统一使用 `EVRP-TW-SPD-HMA/data/jd_instances/jd200_1.txt`。

实验入口与指标说明见：

```text
experiment/统一实验说明.md
4天统一数据实验规划.md
```

快速运行：

```powershell
uv run python experiment/run.py --skip-build --time-limit 30
```

首次运行或需要重建求解器镜像时，去掉 `--skip-build`。

正式实验结果：

```text
results/正式实验结果解读.md
results/batch_summary.md
results/figures/
```

重新生成分析图：

```powershell
uv run python experiment/analyze_results.py
```
