# Kwon Search 当前实验参数

当前实验参数以包内 experiment_parameters.json 为准。它记录原子初态、读出阱、量子化轴控制电压、探测光、频闪时序和收集效率。新实验中未见旧实验那样强的周期性振荡。

电压与磁场保留 2.6 V ≈ 3.77 G 这一个标定点，换算斜率及零场电压留空。红失谐 1–2 MHz 按“激光频率减跃迁频率”存为 [−2, −1] MHz。12–14% 为探测光开启占空比，73% 和 27% 分别为阱开与阱关占空比。

Python 可通过 scan_axis.EXPERIMENT 读取当前设置。下列脚本、旧标定假设和已有 CSV、PNG 均属于历史算例。

## 历史算例 · ⁸⁷Rb 荧光成像扫描复算

横轴为**量子化轴磁场控制电压**，单位 V。原始标签为 `scan volt front back SR detection trap 0.12+0.39us2 / 1.00`。包内没有实验逐发数据。reported_anchors_and_test.csv 录入提问者给出的峰谷近似，接近 0 或 1 的数值仅作标绘。左侧高平台只有区间描述，未补造平台数据点。

## 快速复算

解压到同一目录，在含 NumPy、SciPy、SymPy、Matplotlib 的 Python 3 环境运行以下命令。全谱采用小矩阵重复指数，设单线程可避免 BLAS 并行开销。中文字体推荐 Noto Sans CJK。

```sh
python scan_check.py .
python coherence.py --csv coherence.csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python coherence_scan.py --step 0.01 --csv coherence_scan.csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python depumping.py .
python plot_internal.py --coherence coherence_scan.csv --depumping depumping.csv --output internal_scan.png
```

扫描网格从 −8 V 至 +4 V，步长 0.01 V，共 1201 点。四张扫描表以 `quantization_voltage_V` 明示电压，并保留同值的旧列 `x`。模型纵场列 `b_assumed_G` 或 `b_G` 的单位为 G。coherence_scan.csv 对额外的事件计数列默认只在报告关注的各坐标和文献工作点求值，其他行的空白表示未计算；加 `--count-all` 可为每点求值。两个原子模型结果表另各含一行独立的 20 G 文献工作点，此行两个电压字段均留空。scan_grid.csv 只含扫描点。

## 输出表

- scan_grid.csv 给出试探纵场 b、阱开时总场、旋转圈数、单次离开伸展态概率、假定每周期完全复位时的累计概率、循环线失谐、3 ms 有光时的静止亮态散射总数、无冷却反冲能量尺度，以及有光时沿光束轴的线性摩擦系数除以原子质量（正值为阻尼）。reported_anchors_and_test.csv 在提问者给出的极值点列出同样的量和口述幸存率。
- coherence.csv 给出正式读出工作点下单探测窗 0 至 0.40 μs 的基态、激发态布居、相干矩阵元及平均发光数。effective_uniform 对应单频均匀光；equal_beams_uniform_phase 对应两束等强光（频差 500 kHz）对未知初相平均。
- coherence_scan.csv 为闭合 F=2→F′=3 的 12 态 Lindblad 计算。prob_emitted_at_least_one_pi_or_sigma_minus 为 7500 周期中至少发出一个指定非循环偏振光子的概率；fraction_outside_cycling_pair_preprobe_7500 为末次探测开启前瞬时处于 g₂/e₃ 以外的概率。两列的物理事件不同。ground_coherence_ratio_probe_end 是相邻基态相干幅度的光窗末初比；after_residual_decay 进一步等待激发态衰变，计算这项比较时关掉横向扰动。长期传播本身在开阱时保留扰动。
- depumping.csv 为含 F′=1,2,3 的光泵速率方程。p_F1_before_return_m2_given_m1 表示从 m=+1 开始，先落入 F=1 而后才可能回到 F=2,m=+2 的首达概率。p_F1_6ms_from_uniform_with_dephased_trap 是报告中等权初始布居和完全磁子态退相干假设下的最终 F=1 概率。其余列提供无矢量场、伸展初态及直接跃迁率对照。F=1 设为吸收态，仍在阱内的 F=1 会在普通第三帧被探测到。

除图上特别标示百分比、温度尺度 mK 及相干幅度比之外，CSV 概率均为 0 到 1 的无量纲数。非循环特定发光事件并不等于所有散射、所有磁子态改变、F=1 退泵或逃逸。

## 参数和来源

正式读出采用约 20 G、0.40 μs 阱窗、0.40 μs 探测窗、7500 周期、3 mG 等效场与 60° 夹角；相对于场移后的伸展线红失谐半线宽。全区间示例按 b = k(U − U₀) 计算纵场，U 为控制电压。模型用假定的 1.5 V 进动周期选取 k ≈ +2.38163 G/V，并假定 U₀ = 0 V。实际电压到磁场的斜率和零点均待独立标定。两者在 `scan_axis.py` 中定义，coherence_scan.py 沿用原计算保留五位小数的斜率。探测光频率假定相对于零场线红失谐 2 MHz。脚本中的原子常数取到的有效位数稍有不同，分别为 1.3996 和 1.3996246 MHz/G。几组模型针对不同观测量，不应把各列概率相乘得到幸存率。未模拟三维运动和相机阈值。

1. Kwon et al., Phys. Rev. Lett. 119, 180504 (2017). https://doi.org/10.1103/PhysRevLett.119.180504
2. 作者公开的正文与补充，特别是 §§VII–X、图 5–7 和式 (11)–(13)、(28)–(30)。https://qpal.physics.wisc.edu/images/papers_pubs/2017/2017%20parallel%20qubit%20measurement%20PRL%202017.pdf
3. arXiv v1（图号略有差异）。https://arxiv.org/pdf/1706.09497v1
4. Kwon 博士论文 (2019)，第 11 章。https://asset.library.wisc.edu/1711.dl/QK6SAN52JDK3B8Y/R/file-bede9.pdf
5. Daniel A. Steck, Rubidium 87 D Line Data. https://steck.us/alkalidata/
