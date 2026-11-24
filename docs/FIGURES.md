# 3DGS 深度筛查与修正

已经训练好的 3DGS 模型可以渲染出深度。问题出在遮挡边缘：如果一个像素同时收到近处椅腿和远处墙面的贡献，平均深度就可能落在空中。这个项目先检查哪些像素的深度可疑，再用邻近视角判断一部分像素应该选前层还是后层。

**输入**是预训练 3DGS 模型、相机参数和配套 RGB。**输出**是深度风险图、修正点云和误差表。整个处理流程用 CPU，参考深度用来核对坐标和评估。

![中心深度与逐射线深度在三个物体上的配对误差变化](figures/fig4_ray_depth_metrics.png)

## 结果图

| 图 | 说明 |
| --- | --- |
| [图 1 射线分层与多视角核对](figures/fig1_method.pdf) | 深度候选怎么提出，四个邻近视角如何投票 |
| [图 2 深度图对照](figures/fig2_comparison.pdf) | 在完整 RGB 上框选局部，并排看参考深度、修正前后和误差变化 |
| [图 3 消融实验](figures/fig3_ablation.pdf) | 椅子和警示牌各 8 个视角，比较几何、RGB 和联合规则 |
| [图 4 逐射线深度定量结果](figures/fig4_ray_depth_metrics.pdf) | 24 个视角的配对 MAE 变化和大误差像素比例 |
| [图 5 逐射线深度局部对照](figures/fig5_ray_depth_views.pdf) | 椅子和电钻的改善实例，以及警示牌的退化实例 |

[中文图注和图的来源](docs/FIGURES.md) · [实验报告](docs/REFINEMENT_REPORT.md) · [实验协议](docs/REFINEMENT_PROTOCOL.md)

## 仓库目录

| 位置 | 放什么 |
| --- | --- |
| `gsdepth/core/` | 3DGS 投影、深度合成、跨视角核对 |
| `gsdepth/benchmarks/` | 合成实验、公开模型实验、误差统计 |
| `gsdepth/tools/` | 下载公开数据、复查示例像素、画图 |
| `tests/` | 深度分层、坐标变换和决策规则测试 |
| `figures/` | 论文图。`source_data/` 放绘图数据与示例的来源记录 |
| `evidence/` | 已运行的结果，包括逐视角 CSV、阈值和数据指纹 |
| `docs/` | 报告、实验协议与图注 |

下载的模型和重跑时生成的结果分别在 `data/` 和 `results/`，Git 会忽略这两个目录。

## 怎么跑

在仓库根目录运行。下面以 Windows PowerShell 为例，使用 Python 3.11 或更新版本。

~~~powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m unittest discover -s tests
~~~

只想重画论文图，直接用仓库自带的证据和绘图数据：

~~~powershell
.\.venv\Scripts\python -m gsdepth.tools.plot_paper
~~~

合成场景可以离线跑：

~~~powershell
.\.venv\Scripts\python -m gsdepth.benchmarks.experiment synthetic
.\.venv\Scripts\python -m gsdepth.tools.plot_results results/synthetic
~~~

下载两组公开数据，再跑椅子的单视角筛查和两个物体的多视角修正：

~~~powershell
.\.venv\Scripts\python -m gsdepth.tools.fetch_chair --object chair
.\.venv\Scripts\python -m gsdepth.tools.fetch_chair --object sign
.\.venv\Scripts\python -m gsdepth.benchmarks.experiment chair
.\.venv\Scripts\python -m gsdepth.benchmarks.run_refinement --object chair
.\.venv\Scripts\python -m gsdepth.benchmarks.run_refinement --object sign
~~~

## 这次实验看到了什么

几何单项在两个物体上总体误差最低。以 0.05 个场景单位为大误差阈值，对照加权中值，几何版合计修好 93 个点、改坏 18 个点，联合版修好 25 个点、改坏 2 个点。图 2 展示一次修对和一次误修，图 3 汇总两个物体的结果。完整分组和逐视角结果见[实验报告](docs/REFINEMENT_REPORT.md)。

后续实验把射线两侧的最低合成权重从 10% 降到 5%。在原有两个评估组上，几何版修好 102 个点、改坏 20 个点。用 `--layer-mass 0.05` 可以复跑这项对照。[后续实验和逐视角记录](docs/REFINEMENT_FOLLOWUP.md)

用 `--weak-layer-check` 可在一次运行中比较原版 10%、统一 5% 和弱候选三票规则。三票规则先保留原版已加的修正，再核对新增候选。椅子、警示牌和新加入的电钻结果见[三票对照](docs/REFINEMENT_FOLLOWUP.md#弱候选三票规则)。

`--ray-depth` 用每个像素射线上高斯响应最大的点计算深度。在椅子、警示牌和电钻上，逐射线深度配合 5% 几何规则将 MAE 分别降低 5.0%、2.8% 和 10.7%，对照为按高斯中心深度计算的 5% 几何版。[逐射线实验](docs/REFINEMENT_FOLLOWUP.md#逐射线高斯深度)

[公开 3DGS 数据与来源](https://huggingface.co/datasets/dxgl/multiview-datasets)
