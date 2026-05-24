# 图件候选说明

本文件夹用于单独比较论文插图候选，不会自动改动 `main.tex`。

## 文件夹

- `bitmap_png/`：不可编辑位图，可直接用 `\includegraphics` 插入论文。
- `editable_svg/`：可编辑矢量图，可用 Inkscape、Illustrator、PowerPoint 等打开后改字、改颜色、移动元素。
- `image2_bitmap/`：使用 image2 生成的高美观度位图候选，非代码绘制，文字更少，更偏顶刊图示风格。
- `generate_candidates.py`：图件生成脚本，可重新生成同名 PNG/SVG。

## 候选图

1. `candidate_01_system_coupling`：系统背景与源-荷-储-网-氨耦合示意，适合放在“问题分析”后，替换当前图1。
2. `candidate_02_algorithm_pipeline`：数据读取、五问求解、校验和交付闭环，适合放在“求解算法与可复现流程”后，替换当前图2。
3. `candidate_03_flexibility_logic`：从刚性满负荷、离散开停、连续调节到离网储能的算法逻辑递进图，适合放在“问题分析”或“统一模型框架”附近。
4. `candidate_04_policy_evidence`：前四问结果支撑问题五政策建议的证据链，适合放在“问题五：绿电直连高渗透影响分析”中。

## image2 位图候选

1. `image2_01_graphical_abstract_system.png`：绿电直连电氢氨园区总图，适合替换当前图1。
2. `image2_02_algorithm_logic.png`：Q1--Q4柔性递进与证据输出，适合替换当前图2或作为统一模型框架图。
3. `image2_03_policy_evidence_chain.png`：模型证据到政策机制的图示链条，适合放入问题五。
4. `image2_04_policy_evidence_chain_cn.png`：`image2_03` 的中文版重绘，适合作为“问题背景与问题分析”中的背景图。
5. `image2_05_algorithm_flow_cn.png`：中文版算法与论文逻辑流程图，适合作为“求解算法与可复现流程”中的流程图。

## 论文插入示例

如果选择 PNG：

```tex
\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\textwidth]{figure_candidates/bitmap_png/candidate_01_system_coupling.png}
  \caption{绿电直连型电氢氨园区系统耦合示意}
\end{figure}
```

如果选择 SVG 矢量图，建议先用 Inkscape/Illustrator 修改并导出为 PDF，再在论文中插入 PDF。
