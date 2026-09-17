# 统一数据实验

本实验只使用 `EVRP-TW-SPD-HMA/data/jd_instances/jd200_1.txt`。三个模型共享客户、仓库、充电站、取送货需求、车辆容量、距离矩阵和行驶时间矩阵。

| 模型 | 时间窗 | 电池与充电约束 |
|---|---|---|
| `vrp_spd` | 放宽 | 放宽 |
| `vrp_tw_spd` | 启用 | 放宽 |
| `evrp_tw_spd` | 启用 | 启用 |

运行完整实验：

```powershell
uv run python experiment/run_unified_experiment.py --time-limit 1800
```

快速检查：

```powershell
uv run python experiment/run_unified_experiment.py --time-limit 30
```

输出位于 `results`。`raw` 保存官方求解器输出，`logs` 保存运行日志，统一指标写入 `unified_experiment_metrics.csv` 和 `unified_experiment_metrics.json`。
