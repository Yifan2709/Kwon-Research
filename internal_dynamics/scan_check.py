"""Reproduce a conditional benchmark of the reported scan, not a survival fit.

Run with Python 3, numpy and matplotlib: python scan_check.py [output_directory]
Kwon et al., PRL 119, 180504 (2017), https://doi.org/10.1103/PhysRevLett.119.180504,
main text and supplement X (Fig. 7, Eqs. 28-30), supply t_trap, N, B_fict,
inclination, and nominal saturation. Atomic constants: D. A. Steck, Rubidium
87 D Line Data, https://steck.us/alkalidata/ ; the rounded constants used
below are stated explicitly. No scan calibration has been established.
One putative period fixes the positive slope; offset is assumed zero.
The probe frequency assumes the quoted -2 MHz was referenced at zero field.
Scan coordinates are quantization-axis field control voltages (V). Modeled
fields use G; times use microseconds and frequencies use MHz = cycles/microsecond.
Initial stretched state is defined along the fixed lab probe axis on both
sides; at zero external field this is a model preparation, not a unique
magnetic quantization axis. A reset to this state before each trap pulse is
assumed when forming p_any_reset. Other tipping, repeated coherent drive,
thermal motion, Raman transitions to F=1, camera threshold and escape are
not included. Scattering and recoil columns are separate, stationary,
stretched two-level estimates; no transition from p_any_reset to loss is
assumed. Near-zero/~one qualitative extrema are conventionally plotted at
0/1 and do not carry that numeric precision. The plateau has no specified
individual points or precisely located endpoint, hence carries no CSV
sample. Individual reported extrema are plotted as separate points.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from scan_axis import (ASSUMED_SLOPE_G_PER_V, ASSUMED_ZERO_FIELD_V, VOLTAGE_LABEL,
                       assumed_field_G)

AXIS_ORIGINAL = "scan volt front back SR detection trap 0.12+0.39us2 / 1.00"
LARMOR = 0.6998  # MHz/G, g_F mu_B/h
TRAP_TIME = 0.4  # microseconds
CYCLES = 7500
B_FICT = 0.003  # G
ANGLE = np.pi / 3
SATURATION = 1.0
GAMMA_NU = 6.065  # MHz, natural linewidth in ordinary frequency
GAMMA_RATE = 2 * np.pi * GAMMA_NU  # decay rate, inverse microseconds
RECOIL_K_UK = 0.181  # E_r/k_B, microkelvin
PROBE_TIME = CYCLES * TRAP_TIME  # total 3000 us at 50% duty
CALIBRATION = ASSUMED_SLOPE_G_PER_V  # legacy name, in G/V
ANCHORS = [
    (-4., 0., "near zero"), (-3., 1., "near one"), (-2., 0., "near zero"),
    (-1.5, 1., "near one"), (-.8, 0., "near zero"), (0., .9, "about 0.9"),
    (.5, 0., "near zero"), (1.3, .95, "about 0.95"), (2.2, 0., "near zero"),
    (3., .7, "about 0.7"), (4., 0., "near zero")
]


def model(x):
    x = np.asarray(x, dtype=float)
    b = assumed_field_G(x)
    b_perp = B_FICT * np.sin(ANGLE)
    field = np.hypot(b + B_FICT * np.cos(ANGLE), b_perp)
    turns = LARMOR * TRAP_TIME * field
    # sinc is sin(pi*t)/(pi*t) including its t=0 limit.
    spin_half_transfer = (np.pi * LARMOR * TRAP_TIME * b_perp * np.sinc(turns))**2
    p_change = -np.expm1(4 * np.log1p(-spin_half_transfer))
    p_any = -np.expm1(CYCLES * np.log1p(-p_change))
    delta = -2.0 - 1.3996 * b  # MHz, signed lab sigma+, stationary atom
    y = delta / GAMMA_NU
    denominator = 1 + SATURATION + 4*y*y
    photons = 0.5 * GAMMA_RATE * SATURATION / denominator * PROBE_TIME
    recoil_mk = photons * 2 * RECOIL_K_UK / 1000  # no cooling or escape
    recoil_rad_s = 2 * np.pi * 3770
    # F ~ -alpha*v for balanced beams, on-window steady-state two-level force
    friction_per_mass = -8 * recoil_rad_s * SATURATION * y / denominator**2
    return dict(x=x, b_assumed_G=b, b_effective_trap_G=field,
                trap_precession_turns=turns, p_m_changes_one_trap_pulse=p_change,
                p_any_m_change_7500_independent_resets=p_any,
                cycling_detuning_MHz=delta, photons_3ms_on=photons,
                recoil_ledger_mK=recoil_mk, friction_on_per_mass_inv_s=friction_per_mass,
                quantization_voltage_V=x)


def write_tables(out):
    x = np.linspace(-8, 4, 1201)
    all_points = model(x)
    with (out / "scan_grid.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(all_points)
        writer.writerows(zip(*all_points.values()))
    values = model([r[0] for r in ANCHORS])
    with (out / "reported_anchors_and_test.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reported_approx_survival", "user_description", *values])
        for (_, survival, desc), row in zip(ANCHORS, zip(*values.values())):
            writer.writerow([survival, desc, *row])


def setup_style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for font in ("Noto Sans CJK JP", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        if font in names:
            plt.rcParams["font.family"] = [font, "DejaVu Sans"]
            break
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#98a4ab",
                         "ytick.color": "#3a4c57", "xtick.color": "#3a4c57",
                         "axes.labelcolor": "#233945", "axes.unicode_minus": False,
                         "savefig.facecolor": "white"})


def style_axes(ax):
    ax.set_xlim(-8.25, 4.24)
    ax.set_xticks(range(-8, 5))
    ax.tick_params(length=3.5)
    ax.grid(axis="y", color="#dfe6e8", linewidth=0.7)
    ax.set_axisbelow(True)


def draw_observed(out):
    fig, ax = plt.subplots(figsize=(8.2, 3.05))
    fig.subplots_adjust(left=.10, right=.97, top=.91, bottom=.23)
    style_axes(ax)
    ax.set_ylim(-.08, 1.12)
    ax.set_yticks([0, .25, .5, .75, 1])
    x, y = np.array([(a, b) for a, b, _ in ANCHORS]).T
    ax.scatter(x, y, color="#126164", edgecolors="white", linewidths=.6, s=35, zorder=3)
    # The arrow depicts the unquantified plateau only; its endpoint is not a measurement.
    ax.annotate("", xy=(-4.4, 1), xytext=(-8.0, 1),
                arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#126164"})
    ax.text(-6.35, .88, "口述高平台，边缘位置未定", color="#126164", fontsize=9, ha="center")
    ax.set_ylabel("口述幸存率（约）")
    ax.set_xlabel(VOLTAGE_LABEL)
    fig.savefig(out / "reported_scan.png", dpi=215)
    plt.close(fig)


def draw_test(out):
    x = np.linspace(-8, 4, 2401)
    m = model(x)
    fig, (spin, photon) = plt.subplots(2, 1, sharex=True, figsize=(8.2, 4.68),
                                       gridspec_kw={"height_ratios": [1.04, 1]})
    fig.subplots_adjust(left=.115, right=.97, top=.925, bottom=.13, hspace=.43)
    for ax in (spin, photon):
        style_axes(ax)
        for point in (-3., 2.2):
            ax.axvline(point, color="#b1a7a3", linestyle=(0, (2, 3)), linewidth=.85)
    spin.plot(x, m["p_any_m_change_7500_independent_resets"]*100, color="#136665", lw=2.)
    spin.set_title("A    7500 次独立复位中至少一次改变磁子态", loc="left",
                   fontsize=11, color="#183c47", pad=4)
    spin.set_yscale("symlog", linthresh=.1, linscale=.65)
    spin.set_yticks([0, .1, 1, 10])
    spin.set_yticklabels(["0", "0.1", "1", "10"])
    spin.set_ylim(-.006, 22)
    spin.set_ylabel("事件概率（%）")
    photon.set_title("B    理想伸展态散射与轴向阻尼方向", loc="left",
                     fontsize=11, color="#183c47", pad=4)
    photon.plot(x, m["photons_3ms_on"]/10000, color="#294e77", lw=2.)
    resonant_x = ASSUMED_ZERO_FIELD_V - 2 / (1.3996 * ASSUMED_SLOPE_G_PER_V)
    photon.axvspan(-8, resonant_x, color="#fbf1e8", zorder=-2, lw=0)
    photon.axvspan(resonant_x, 4, color="#eaf3f6", zorder=-2, lw=0)
    photon.text(-7.6, 2.6, "蓝失谐：反阻尼", color="#835137", fontsize=9)
    photon.text(1.25, 2.6, "红失谐：阻尼", color="#3f6481", fontsize=9)
    photon.set_ylabel("散射光子数（万）")
    photon.set_ylim(0, 3.13)
    photon.set_yticks([0, 1, 2, 3])
    photon.set_xlabel(VOLTAGE_LABEL + f"    模型假设  {ASSUMED_SLOPE_G_PER_V:+.2f} G/V，零场对应 {ASSUMED_ZERO_FIELD_V:g} V")
    fig.savefig(out / "conditional_scan.png", dpi=215)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    out = parser.parse_args().output_directory
    out.mkdir(parents=True, exist_ok=True)
    write_tables(out)
    setup_style()
    draw_observed(out)
    draw_test(out)


if __name__ == "__main__":
    main()
