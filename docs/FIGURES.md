# 论文配图与图注

三张图按双栏通栏宽度（约 182 mm）排版。PDF 和 SVG 保留可编辑文本，PNG 与 TIFF 为 600 dpi。图中使用英文短标签，中文图注如下。

## 图 1 射线分层与多视角核对

[PDF](../figures/fig1_method.pdf) · [PNG](../figures/fig1_method.png)

**图注**　固定 3DGS 模型的多视角深度修正。**a**，沿相机射线按深度排列高斯贡献。当前后两侧各有至少 10% 的累计合成权重且间隙达到阈值时，分别用两侧的加权中值提出候选。曲线和均值位置为示意。**b**，将候选深度对应的三维点及局部图像块重投影到邻近视角，分别核对深度一致性和光度代价是否下降。**c**，警示牌视角 120 中近层候选的实际投票记录。实心圆表示通过该行的检查。视角 107 和 128 同时提供两种支持，可靠视角中没有自由空间冲突，因此规则采纳了近层候选。图 2b 展示这次修改的误差。

[修正规则和阈值](REFINEMENT_PROTOCOL.md) · [示例像素和投票来源](../figures/source_data/case_provenance.json)

## 图 2 深度图的定性对照

[PDF](../figures/fig2_comparison.pdf) · [PNG](../figures/fig2_comparison.png)

**图注**　联合规则的一个修正实例和一个误修实例。每行从左到右依次为带定位框的完整 RGB 图、同位置的参考深度、加权中值深度、联合修正深度和绝对误差变化。黄色框对应后四列的 29 × 29 像素裁剪，白框中心是读数像素。每行前三张深度图共用该行下方的深度色标。右列显示修正后减去修正前的绝对误差，两行共用 ±1.2 个场景单位的发散色标。蓝色表示误差下降，红色表示上升，灰色为评估掩码以外的像素。**a**，椅子视角 12，标记点的绝对误差从 0.468 降至 0.017。**b**，警示牌视角 120，绝对误差从 0.047 升至 1.196。

两行均取自 128 × 128 的评估视角，局部按原像素放大。椅子选取视角 12 中跨过误差阈值且降幅最大的像素，警示牌沿用实验日志记录的误修。两处用于说明算法行为，全部保留视角的结果见图 3。[视角、坐标与精确读数](../figures/source_data/case_provenance.json) · [DX.GL 数据集](https://huggingface.co/datasets/dxgl/multiview-datasets)

## 图 3 消融实验

[PDF](../figures/fig3_ablation.pdf) · [PNG](../figures/fig3_ablation.png) · [绘图 CSV](../figures/source_data/fig3_paired_view_metrics.csv)

**图注**　三个邻近视角规则在保留视角上的对照。**a、b**，每个点表示一个视角中该规则的 MAE 减去加权中值的 MAE，单位为 0.001 个场景单位。菱形表示同一物体 8 个视角的等权平均，负值表示误差下降。**c**，合计两个物体的 16 个视角，以 0.05 个场景单位为阈值统计相对于加权中值从大误差变为合格的像素数（右），以及从合格变为大误差的像素数（左）。各方法在同一批有效像素上评估。视角来自两个物体，结果用于描述本实验的差异。

[椅子逐视角日志](../evidence/chair_heldout/metrics.csv) · [警示牌逐视角日志](../evidence/sign_heldout/metrics.csv)

## 排版参考

定性图采用完整视图定位、同位置放大和按方法对齐的列布局，参考 DNGaussian 图 4、5 的框选与放大方式，以及 2DGS 图 5 的逐场景对比方式。[DNGaussian，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_DNGaussian_Optimizing_Sparse-View_3D_Gaussian_Radiance_Fields_with_Global-Local_Depth_CVPR_2024_paper.pdf) · [2D Gaussian Splatting，SIGGRAPH 2024](https://www.cvlibs.net/publications/Huang2024SIGGRAPH.pdf)

## 重画与复核

用仓库中的证据和裁剪数组离线重画。

~~~powershell
python -m gsdepth.tools.plot_paper
~~~

如需重新生成图 1c 的投票记录和图 2 的像素数组，先按 README 下载两个对象，再运行以下命令。重放脚本核对源文件的 SHA-256，并对照已保存的逐视角实验记录检查示例重放的结果。

~~~powershell
python -m gsdepth.tools.prepare_paper_cases
python -m gsdepth.tools.plot_paper
~~~
