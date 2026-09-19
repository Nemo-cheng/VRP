# 订单运输路径优化实验

项目当前只使用本地 `订单数据.xlsx`，研究网点级运输路径与发车时段优化。原始 Excel、含运单号和车辆标识的中间数据不会提交到公开仓库。

当前实验规划见：

```text
订单数据优化实验规划.md
```

运行数据可优化性审计：

```powershell
uv run experiment/audit_excel_optimization_scope.py
```

运行现有测试：

```powershell
uv run --with pytest --with pandas --with openpyxl pytest experiment/test_audit_excel_optimization_scope.py experiment/test_prepare_excel_transport_data.py -q
```
