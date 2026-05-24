# 最终论文编译与提交说明

## 1. 文件结构

```text
最终论文_整合稿/
├─ main.tex
├─ README_编译与提交说明.md
└─ figures/
   ├─ fig01_q1_power_balance.pdf
   ├─ fig02_q1_indicator_check.pdf
   ├─ fig03_q2_q3_cost_compare.pdf
   ├─ fig04_q2_q3_grid_indicator.pdf
   ├─ fig05_q4_storage_pareto.pdf
   ├─ fig06_q4_mode_cost_breakdown.pdf
   └─ fig07_q5_policy_evidence.pdf
```

## 2. 页码与格式约束

- 第 1 页：封面页；
- 第 2 页：题目、摘要、关键词；
- 正文：从第 3 页开始；
- 不设置目录；
- 正文目标控制在 25 页以内；
- 附录页数不限。

## 3. 编译方式

建议使用 XeLaTeX 编译：

```powershell
cd .\最终论文_整合稿
xelatex main.tex
xelatex main.tex
```

若使用 TeXstudio、Overleaf 或 VS Code LaTeX Workshop，请选择 `XeLaTeX` 引擎。

## 4. 当前写作状态

当前 `main.tex` 已完成：

- 封面页；
- 题目、摘要、关键词；
- 问题背景与重述；
- 问题分析；
- 模型假设；
- 符号说明；
- 数据说明及预处理；
- 统一模型框架；
- 问题一模型与核心结果初稿；
- 后续各问章节框架。

后续阶段建议按顺序继续补写：

1. 问题二离散开停机模型、算法和结果；
2. 问题三连续调节 MILP 与 Q2/Q3 对比；
3. 问题四离网储能模型、Pareto 权衡与公共电网支撑价值；
4. 问题五证据矩阵和政策建议；
5. 结论与摘要微调。

## 5. 支撑材料压缩包建议

支撑材料建议保留：

- `model_common/`
- `问题1_典型风光场景指标分析/solve_q1.py`
- `问题2_离散制氨调节优化/solve_q2.py`
- `问题3_连续制氨调节优化/solve_q3.py`
- `问题4_离网储能配置优化/solve_q4.py`
- `问题5_政策与电力系统影响分析/solve_q5.py`
- 各问 `outputs/*.csv`
- 各问 `outputs/*_result_summary.md`

不建议放入压缩包：

- `__pycache__/`
- `.pyc`
- 大量 PNG/PDF/SVG 全量图件，可按需要只保留论文使用图件；
- 临时 Word 文件。
