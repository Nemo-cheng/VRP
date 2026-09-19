# 订单运输VRP实验

项目只使用本地 `订单数据.xlsx`，研究运输任务的车辆分配、任务顺序、载货路径、空驶路径和发车时间联合优化。

核心设计见：

```text
VRP实验设计.md
```

当前工作分支：

```text
experiment/vrp-redesign
```

原始 Excel、运单号、车辆标识和网点名称不会提交到公开仓库。

运行数据可优化性审计：

```powershell
uv run experiment/audit_excel_optimization_scope.py
```

运行相关测试：

```powershell
uv run --with pytest --with pandas --with openpyxl pytest experiment/test_audit_excel_optimization_scope.py experiment/test_prepare_excel_transport_data.py -q
```

核心实验依次运行：

```powershell
uv run experiment/build_vrp_data_layer.py
uv run experiment/compare_basic_vrp.py
uv run experiment/optimize_type_compatible_vrp.py
uv run experiment/optimize_time_dependent_vrp.py
uv run experiment/build_task_path_options.py
uv run experiment/summarize_vrp_results.py
```

时间外验证依次运行：

```powershell
uv run experiment/build_vrp_holdout_data.py
uv run experiment/compare_basic_vrp.py --data-dir processed/company/vrp_holdout --result-dir results/company_transport/holdout
uv run experiment/optimize_type_compatible_vrp.py --data-dir processed/company/vrp_holdout --result-dir results/company_transport/holdout
uv run experiment/optimize_time_dependent_vrp.py --data-dir processed/company/vrp_holdout --result-dir results/company_transport/holdout
```

统一结果位于：

```text
results/company_transport/vrp_experiment_summary.json
results/company_transport/vrp_robustness_summary.csv
```
