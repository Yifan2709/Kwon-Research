# 内部动力学兼容入口

参数与模型已统一到根目录的 `config/` 和 `kwon/`，详见 [项目说明](../README.md)。本目录中的 Python 文件仅转发至共享实现，不包含第二份实验参数。

当前目录中既有 CSV/PNG 均为此前的归档结果，不代表新配置下的计算。新结果默认保存在运行目录下的 `outputs/`，或命令明确指定的位置。

从代码包根目录运行：

```bash
python internal_dynamics/coherence.py --csv outputs/coherence.csv
python internal_dynamics/coherence_scan.py --csv outputs/coherence_scan.csv
python internal_dynamics/depumping.py --csv outputs/depumping.csv
python internal_dynamics/scan_check.py outputs
python internal_dynamics/plot_internal.py
```

首先用 `python -m kwon.parameters --check` 检查缺失参数；未定值必须明确填写，不再采用旧算例默认值。
