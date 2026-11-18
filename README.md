# Kwon 态选择荧光读出模型

所有运行入口共用 `kwon/parameters.py`，只从 `config/` 下三个固定文件读取参数。原来两个目录中的重复模型已改为兼容入口；物理与数值设置不再写在脚本里。

| 文件 | 唯一负责的设置 |
| --- | --- |
| `config/atomic.json` | 原子结构、线宽、磁矩、质量及基本常数 |
| `config/experiment.json` | 初态、光阱、探测光、磁场标定、时序、扫描范围及明确的模型假设 |
| `config/numerics.json` | 求解器、误差容限、采样、绘图以及可选现象学模型设置 |

周期、各光窗时长、总饱和度、总曝光时间、反冲能和电压对应的磁场都由这些输入推导，不另存第二套数值。代码中的整数、π、单位换算因子和数学表达式不是实验参数。

## 修改与检查参数

在上述文件中修改数值后运行：

```bash
python -m pip install -r requirements.txt
python -m kwon.parameters --check
```

默认配置路径相对于代码包根目录确定，与启动命令所在目录无关。所有计算入口也支持 `--config-dir /absolute/path/to/config`，该目录必须完整包含同名的三个 JSON；不会叠加隐藏默认值，不使用环境变量覆盖物理参数。每次计算只读入一个 `Parameters` 对象并传给模型。循环跃迁的频率参考由偏振方向与超精细流形决定，不随制备态的 mF 改动。

重复 JSON 键、缺文件、缺必要键或非法值会报错。`null` 表示尚未提供，不会读取旧算例补齐。`--check` 列出未定值；具体模型只要求自己真正需要的输入。参数快照随每个新 CSV 保存为同名 `.parameters.json`，仅用于追溯，不是另一套运行配置。

当前实验输入已集中记录，但以下内容仍需补全：磁场标定斜率、单脉冲计算的工作电压、光阱矢量光移对应的磁场、确切失谐及其参考磁场、确切探测占空比和光窗起点、读出周期数、扫描起止及步长。已给出的范围不会擅自取中点。单个磁场标定点不足以推出整条电压—磁场曲线；工作电压也不会默认等于标定点。

光强计算以 `probe.reported_s0_per_beam` 为唯一数值入口，总饱和度由光束数推导。功率、光斑和报告光强保留为实验记录，当前模型不再另外从高斯光束峰值算一份饱和度。偏振杂质当前明确设为零，是理想偏振假设；需要研究杂质时必须在配置中修改。闭合相干模型会拒绝尚不支持的非纯圆偏振，不会忽略该设置。

## 运行

补全对应模型的输入后，从代码包根目录执行：

```bash
python -m kwon.coherence --csv outputs/coherence.csv
python -m kwon.coherence_scan --csv outputs/coherence_scan.csv
python -m kwon.depumping --csv outputs/depumping.csv
python -m kwon.scan_check outputs
python -m kwon.plot_internal
```

`coherence` 按配置的工作电压计算单个探测脉冲；`coherence_scan` 和 `depumping` 读取相同的电压扫描设置。相干单脉冲可按配置对双束的相对光学相位平均，周期扫描采用配置中明确声明的平均强度近似。

旧脚本路径仍然可用，例如：

```bash
python internal_dynamics/coherence_scan.py --csv outputs/coherence_scan.csv
python multimechanism_scan/check_kwon.py --csv outputs/kwon_control.csv
```

`multimechanism_scan/scan.py` 只是 `python -m kwon.scan` 的兼容入口。它的记忆系数、附加温度、包络位置和漂移等现象学参数默认全部未定，必须在 `numerics.json` 中明确填写后才能计算；不再沿用旧曲线拟合系数。

## 物理范围

相干模型包含完整的闭合基态/激发态 Zeeman 流形，当前对应 F=2↔F′=3；它报告非循环偏振散射事件和内部态相干，不包含 F=1，不能将其事件概率解释为 depump 或原子逃阱。默认时序依次为光阱、前暗间隔、探测光、后暗间隔，时长由频率、占空比和探测光窗位置共同决定。

Depumping 模型保留通往较低超精细基态的分支，但采用逐光束、逐跃迁稳态散射率及周期完全退相干近似，未包含多束饱和竞争、有限光脉冲光学瞬态或跨周期相干。它输出较低 F 的布居概率，不能直接作为相机幸存率。`scan_check` 的反冲统计是假定循环跃迁的稳态诊断，不是含冷却和运动的逃阱仿真。现象学幸存率模型只是条件示例，不是由当前装置参数推导的预测。

光阱波长、束腰、初始温度等已记录，但现有内部态模型不从这些量求运动、标量/张量光移或虚拟磁场；虚拟磁场必须作为独立输入。光束几何标志和理论收集效率也只作为记录；当前成像光子数估计使用实际收集效率。真实磁场当前按 z 轴量子化方向处理，虚拟磁场用 `[Bx, By, Bz]` 输入。未实现的物理不会由配置中的记录值自动生效。

## 验证与历史结果

```bash
python -m unittest discover -s tests -v
```

`tests/fixture.json` 中是明确标注的合成测试输入，只用于检查参数传递、守恒、缺值阻断和兼容入口，不属于实验参数，不被任何生产入口读取。测试不使用旧拟合曲线作为目标。

两个旧目录下既有 CSV/PNG 是此前的归档输出，未按本次参数结构重算。新入口只计算并保存新结果；绘图默认读取 `outputs/`，不会自动加载这些历史文件。
