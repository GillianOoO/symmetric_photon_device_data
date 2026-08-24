# Symmetric photonic-state paper: figure data release

本仓库只整理论文 `main.tex` 和 `supplementary.tex` 中指定数据图/数据表的**直接输入、生成 plot 数据的代码、plot-ready 输出数据**，以及论文中使用的最终 PDF 作为核对基准。仓库不包含 MATLAB/Python 绘图代码，也不包含颜色、坐标轴、排版、拼图等绘图过程。

目标内容：主文 Fig. 3；SI Fig. 1、8、9、10、11；SI Table 1。`paper_sources/` 中保留了用于确认图号和图注的 LaTeX 源文件。

## 一键生成全部数据

需要 Python 3.9 或更高版本；脚本只使用 Python 标准库。

```bash
python3 run_all.py
```

该命令依次运行每个图目录下的纯数据脚本，并重新生成所有 CSV。不会生成或修改任何图片。

## 数据边界和通用格式

对于主文 Fig. 3、SI Fig. 8 和 SI Fig. 10，直接输入是每个方法在不同 shots 下的重复估计矩阵。每一行格式为：

```text
shots  reference  reported_RMSE  repeats  estimate_1 ... estimate_repeats
```

代码同时保留文件中报告的 RMSE，并由重复估计值重新计算

```text
RMSE = sqrt(mean((estimate_r - reference)^2))
```

输出中的 `rmse_rounding_delta` 用来检查原始文本有限小数位造成的差异。

本仓库把这些“直接进入数据图的重复估计/汇总表”定义为输入边界。更上游的探测器 `.bin` 文件、测量序列优化程序和绘图程序不属于本数据发布包：它们不直接作为当前图的数据列输入，并且用户要求只保留生成 plot 数据的代码。

## 图与代码对应关系

| 论文项目 | LaTeX 定位 | 数据代码 | 主要输入 | 生成输出 |
|---|---|---|---|---|
| Main Fig. 3 | `paper_sources/main.tex`, `fig:properties_numerics` | `main_fig3/code/build_plot_data.py` | `main_fig3/inputs/raw_estimates/` | `main_fig3/outputs/fig3_plot_data.csv` |
| SI Fig. 1 | `paper_sources/supplementary.tex`, `fig:symmetry_average` | `si_fig1/code/build_plot_data.py` | `si_fig1/inputs/summary.csv` | `si_fig1/outputs/si_fig1_plot_data.csv` |
| SI Fig. 8 | `paper_sources/supplementary.tex`, `fig:simulate_noiseless` | `si_fig8/code/build_plot_data.py` | `si_fig8/inputs/raw_estimates/` | `si_fig8/outputs/si_fig8_plot_data_as_published.csv` |
| SI Fig. 9 | `paper_sources/supplementary.tex`, `fig:simulates_large_qubits` | `si_fig9/code/build_plot_data.py` | `si_fig9/inputs/ogm_summary.csv`, `comparison_summary.csv` | `si_fig9/outputs/si_fig9_plot_data.csv` |
| SI Fig. 10 | `paper_sources/supplementary.tex`, `fig:exp_raw_target` | `si_fig10/code/build_plot_data.py` | `si_fig10/inputs/raw_estimates/` | `si_fig10/outputs/si_fig10_plot_data.csv` |
| SI Fig. 11 | `paper_sources/supplementary.tex`, `fig:variance` | `si_fig11/code/build_plot_data.py` | 两个 variance summary CSV | `si_fig11/outputs/si_fig11_plot_data.csv` |
| SI Table 1 | `paper_sources/supplementary.tex`, `tab:variance_maxT_experiment` | `si_table1/code/build_table_data.py` | 两个 variance summary CSV | `si_table1/outputs/si_table1.csv` |

共享解析和 RMSE 校验代码位于 `plot_data_utils.py`。主文 Fig. 3、SI Fig. 8、SI Fig. 10 的各自脚本明确调用该文件并定义各自的数据筛选规则。

## Main Fig. 3

论文位置：`paper_sources/main.tex` 第 303-309 行，最终图为 `fig-data.pdf`。

数据内容共有六个 panel：

- (a)-(b)：三比特随机 Hamiltonian，W/GHZ 实验态。
- (c)-(d)：四比特 spin Hamiltonian，W/GHZ 实验态。
- (e)-(f)：三比特非线性量 `Tr(rho^2 H)`，W/GHZ 实验态。

直接输入来自原工作区：

- `plots/fig_data/data_1021/data_rhoH/haozhaowu_outputs_W/`
- `plots/fig_data/data_1021/data_rhoH/haozhaowu_outputs_GHZ/`
- `plots/fig_data/data_1021/data_rho^2H/haozhaowu_outputs_W/`
- `plots/fig_data/data_1021/data_rho^2H/haozhaowu_outputs_GHZ/`

线性 panel 使用 SG、Derand、OGM、AP、Compact 五种方法；非线性 panel 使用 SG、Derand、OGM、Compact。`main_fig3/code/build_plot_data.py` 解析 28 个直接输入文件、校验重复估计 RMSE，并输出：

- `main_fig3/outputs/all_input_rows.csv`：包含输入文件中的所有 shots 行。
- `main_fig3/outputs/fig3_plot_data.csv`：与论文 Fig. 3 数据点一致，排除辅助的 1000-shot 行。
- `main_fig3/outputs/published_fig3.pdf`：论文最终图，仅用于与 CSV 核对，不附绘图代码。

## SI Fig. 1

论文位置：`paper_sources/supplementary.tex` 第 831-837 行。图比较 8-qubit spin Hamiltonian 在理想 GHZ 态上的三种 observable 版本：原始 `H`、compact `sym_H`、measurement-compatible `sym_ave`；四个 panel 分别是 SG、Derand、AP 和 OGM。

直接输入 `si_fig1/inputs/summary.csv` 来自：

```text
symmetry_averaging_workspace/outputs/h8_preview/summary.csv
```

`si_fig1/code/build_plot_data.py` 只选择 `target == H8`、四个目标方法和三个目标 variant，并按 panel、variant、shots 排序。输出为 `si_fig1/outputs/si_fig1_plot_data.csv`，共 72 个数据点。最终论文图保存在 `si_fig1/outputs/published_si_fig1.pdf`。

## SI Fig. 8

论文位置：`paper_sources/supplementary.tex` 第 1168-1174 行。六个 panel 与主文 Fig. 3 对应，但输入是理想 GHZ/W 态的 noiseless simulation 结果。

直接输入来自：

```text
postprocessing/test_outputs/fig3_noiseless/results/
```

`si_fig8/code/build_plot_data.py` 输出：

- `si_fig8/outputs/all_generated_data.csv`：保留模拟生成的全部数据点。
- `si_fig8/outputs/si_fig8_plot_data_as_published.csv`：严格复现现有 PDF 中实际使用的数据点。
- `si_fig8/outputs/published_si_fig8.pdf`：论文最终图。

重要核查结果：原绘图文件对每个数据系列无条件执行“删除第 5 行”。旧实验数据的第 5 行是 1000 shots，但当前 noiseless 输入没有 1000-shot 行，第 5 行实际是 2038 shots。因此现有 SI Fig. 8 PDF 的曲线从 572 直接连接到 7259，虽然 x 轴仍显示 2038。`all_generated_data.csv` 保留了 2038 数据；`si_fig8_plot_data_as_published.csv` 则保留论文 PDF 的现状。该差异不能当作数值模拟缺失。

## SI Fig. 9

论文位置：`paper_sources/supplementary.tex` 第 1185-1191 行。图展示 n = 8、12、14 的 GHZ 态上随机 spin-model Hamiltonian 的误差。

两个直接输入分别来自：

```text
postprocessing/test_outputs/spin_model_ghz_ogm_scaling/summary.csv
postprocessing/test_outputs/spin_model_ghz_sg_methods/summary.csv
```

`si_fig9/code/build_plot_data.py` 从第一个文件取 `variant == H` 作为 OGM、`variant == sym_H` 作为 Compact；从第二个文件取 ShadowGrouping、Derandomization、AdaptivePaulis，并统一方法名为 SG、Derand、AP。输出 `si_fig9/outputs/si_fig9_plot_data.csv`，共 90 个数据点。最终论文图为 `si_fig9/outputs/published_si_fig9.pdf`。

## SI Fig. 10

论文位置：`paper_sources/supplementary.tex` 第 1598-1623 行。六个 panel 使用与主文 Fig. 3 相同的重复实验估计，但 reference 改为实验 tomography 重构的 noisy target：线性 panel 使用 `Tr(rho_exp H)`，非线性 panel 使用 `Tr(rho_exp^2 H)`。

直接输入来自：

```text
postprocessing/test_outputs/experimental_outputs_actual/
```

本仓库保留其中 28 个 repeat-level estimator 文件和 `generated_output_summary.csv`。`si_fig10/code/build_plot_data.py` 重新校验 RMSE，并输出：

- `si_fig10/outputs/all_input_rows.csv`：所有直接输入行。
- `si_fig10/outputs/si_fig10_plot_data.csv`：与论文图一致，非线性 panel 排除辅助的 1000-shot 行。
- `si_fig10/outputs/published_si_fig10.pdf`：论文最终图。

原工作区没有找到生成 `exp-fig-data.pdf` 的最终绘图脚本；但 repeat-level 输入、noisy reference、RMSE 以及最终 PDF 均已保留，且已通过 PDF 数据点位置与 shots 列表交叉核对。这里新增的代码只重建其 plot-ready 数据，不伪造缺失的绘图来源。

## SI Fig. 11

论文位置：`paper_sources/supplementary.tex` 第 1733-1739 行。六个 panel 展示 noisy experimental states 下不同 estimator 的 `variance_T_shots`。

直接输入：

- `si_fig11/inputs/variance_noisy_summary.csv`：SG、Derand、OGM、AP。
- `si_fig11/inputs/variance_compact_summary.csv`：Compact。

它们来自原工作区 `postprocessing/test_outputs/experimental_variance_checks/`。`si_fig11/code/build_plot_data.py` 合并两个文件，去除 NaN variance 和辅助的 1000-shot 行，输出 `si_fig11/outputs/si_fig11_plot_data.csv`，共 188 个数据点。最终论文图为 `si_fig11/outputs/published_si_fig11.pdf`。

## SI Table 1

论文位置：`paper_sources/supplementary.tex` 第 1747 行开始，label 为 `tab:variance_maxT_experiment`。表格报告每个 case/method 最大可用 shots 对应的 single-shot variance：

```text
variance_single_shot = shots_max * variance_T_at_shots_max
```

`si_table1/code/build_table_data.py` 使用与 SI Fig. 11 相同的两个输入 CSV，为每个 case/method 选择最大 shots，验证上式后生成：

- `si_table1/outputs/si_table1.csv`：论文表格的宽表数据。
- `si_table1/outputs/si_table1_sources.csv`：每个单元格的 case、method、shots 和 variance 来源。

非线性 AP 在存档数据中不存在，因此对应单元格保持为空，与论文中的 `--` 一致。生成数值与原工作区 `variance_maxT_table.csv` 完全一致；论文 LaTeX 仅将这些值四舍五入到两位小数显示。

## 核查说明

- 六个最终 PDF 从论文目录复制，并与 `new_datas_afterJun18/symmetric_photon_paper/figs/` 中对应文件进行 SHA-256 比对，结果一致。
- 三类 repeat-level 输入均重新计算了 RMSE；最大差异仅来自旧文本保留 6 位小数时的舍入。
- SI Table 1 的全部可用单元格都验证了 `variance_T_shots * shots == variance_single_shot`。
- 仓库不包含大于 10 MB 的文件，也不需要 Git LFS。
- `CHECKSUMS.sha256` 记录所有发布文件的 SHA-256，可用于确认下载内容未改变。
