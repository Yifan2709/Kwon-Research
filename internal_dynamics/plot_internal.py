"""Plot the checks of internal atomic dynamics imported from the CSV simulations.

Usage:
python plot_internal.py --coherence ../coherence_scan.csv --depumping ../leak/depumping.csv --output internal_scan.png

Also supports a flat directory containing this script, scan_check.py and both
CSVs. Rows with no x are published working-point checks and are excluded.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scan_check import setup_style
from scan_axis import VOLTAGE_LABEL

HERE = Path(__file__).resolve().parent


def resolve_input(basename, *candidates):
    for candidate in (HERE / basename, *candidates):
        if candidate.is_file():
            return candidate
    return HERE / basename


def trace(path, column):
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        # Original tables use x for the same control voltage.
        voltage_key = "quantization_voltage_V" if "quantization_voltage_V" in reader.fieldnames else "x"
        data = np.array(sorted((float(row[voltage_key]), float(row[column])) for row in reader
                               if row[voltage_key].strip()))
    return data[:, 0], data[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coherence", type=Path,
                        default=resolve_input("coherence_scan.csv", HERE.parent / "coherence_scan.csv"))
    parser.add_argument("--depumping", type=Path,
                        default=resolve_input("depumping.csv", HERE.parent / "leak" / "depumping.csv"))
    parser.add_argument("--output", type=Path, default=HERE / "internal_scan.png")
    args = parser.parse_args()
    setup_style()
    fig, (spin, raman) = plt.subplots(1, 2, figsize=(7.8, 3.01))
    fig.subplots_adjust(left=.094, right=.981, bottom=.22, top=.83, wspace=.37)
    sx, y = trace(args.coherence, "fraction_outside_cycling_pair_preprobe_7500")
    rx, p = trace(args.depumping, "p_F1_6ms_from_uniform_with_dephased_trap")
    spin.plot(sx, 100*y, color="#136665", lw=1.55)
    spin.set_yscale("log")
    spin.set_ylabel("循环跃迁以外的布居（%）", fontsize=10)
    spin.set_title("A    保留相干的周期演化", loc="left", fontsize=11,
                   fontweight="bold", color="#184147", pad=19)
    spin.text(0., 1.04, "第 7500 次探测光开启前", transform=spin.transAxes,
              fontsize=9, color="#48616a")
    raman.plot(rx, 100*p, color="#294e77", lw=2.)
    raman.set_ylim(bottom=0)
    raman.set_ylabel("退泵至 F = 1 的布居（%）", fontsize=10)
    raman.set_title("B    只传播布居的速率模型", loc="left", fontsize=11,
                    fontweight="bold", color="#244e69", pad=19)
    raman.text(0., 1.04, "初始 F = 2 各磁子态等布居，照相 6 ms 后", transform=raman.transAxes,
               fontsize=9, color="#48616a")
    for ax in (spin, raman):
        ax.set_xlim(-8.2, 4.2)
        ax.set_xticks(range(-8, 5, 2))
        ax.set_xlabel(VOLTAGE_LABEL)
        ax.tick_params(length=3., labelsize=9.5)
        ax.grid(axis="y", which="major", color="#dfe6e8", lw=.7)
        ax.set_axisbelow(True)
        for point in (-3., 2.2):
            ax.axvline(point, color="#b1a7a3", linestyle=(0, (2, 3)), lw=.7)
    ax = spin
    if np.min(y) < 0 or np.max(y) > 1 or np.min(p) < 0 or np.max(p) > 1:
        raise ValueError("Population outside physical range")
    fig.savefig(args.output, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
