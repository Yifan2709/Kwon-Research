# Kwon Research

⁸⁷Rb 态选择荧光成像研究代码，包含 Kwon Search 的多机制扫描、内部态动力学复算，以及当前实验参数。

## 代码来源

本仓库导入两份已更新实验参数的代码包，内容对应 2026-11-17 保存的版本。

| 目录 | 来源代码包 | 内容 |
| --- | --- | --- |
| [multimechanism_scan](multimechanism_scan) | Kwon_1G_多机制扫描.zip | 周期响应与加热假设、12 态控制计算、已有逐点结果和图 |
| [internal_dynamics](internal_dynamics) | Kwon_复算代码与逐点结果.zip | 相干演化、退泵计算、已有逐点结果和图 |

## 当前实验参数

两份目录中的 `experiment_parameters.json` 内容一致，记录初态 |F=2,mF=−2⟩、10 µK 温度、840 nm 光阱、1.1 µm 束腰、读出时 7 mK 阱深、双束 σ− 探测光、3 kHz 频差、1.416 MHz 频闪和荧光收集效率。

横轴为量子化轴磁场控制电压，单位 V。已知标定点为 2.6 V ≈ 3.77 G，斜率和零场电压字段留空。探测光占空比保留 12–14%，红失谐保留 1–2 MHz。全部数值见 [参数配置](multimechanism_scan/experiment_parameters.json)。

新实验中未见旧实验那样强的周期性振荡。当前 JSON 用于保存这组实验输入。已有扫描脚本采用历史算例的模型常数，CSV 和 PNG 对应历史计算及口述观测点。运行这些脚本会复现各自的历史算例。后续新实验仿真需要将参数配置接入相应模型，并明确时长、频率参考和场标定。

## 使用

使用 Python 3，安装 NumPy、SciPy、SymPy 和 Matplotlib。绘制中文图时可安装 Noto Sans CJK 字体。

```sh
python -m pip install numpy scipy sympy matplotlib
```

各目录中的说明文件给出历史算例的运行命令。当前参数可直接读取。

```python
import json
from pathlib import Path

parameters = json.loads(
    Path("multimechanism_scan/experiment_parameters.json").read_text(encoding="utf-8")
)
```
