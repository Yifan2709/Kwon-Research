"""Single-window spin and stationary recoil diagnostics with configured inputs."""
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .atomic import angular_momentum
from .output import setup_plot, write_rows
from .parameters import Parameters, ParameterError


def model(voltage, config=None):
    config = config or Parameters()
    field = float(config.field_g(voltage))
    fx, fy, fz = angular_momentum(config.ground_F)
    vector = np.asarray(config.trap_field_vector, dtype=float).copy()
    vector[-1] += field
    h = 2 * np.pi * config.mu * config.ground_g * sum(
        b * op for b, op in zip(vector, (fx, fy, fz)))
    unitary = expm(-1j * h * config.trap_us)
    initial_index = int(config.initial_m + config.ground_F)
    p_change = float(np.clip(1 - abs(unitary[initial_index, initial_index]) ** 2, 0, 1))
    with np.errstate(divide="ignore"):
        p_reset = -np.expm1(config.cycles * np.log1p(-p_change))
    delta = config.detuning_mhz(field)
    scaled_delta = delta / config.linewidth_MHz
    denominator = 1 + config.saturation + 4 * scaled_delta ** 2
    photons = config.gamma * config.saturation / (2 * denominator) * config.total_on_us
    recoil = 2 * config.recoil_temperature_mK * photons
    friction = -8 * config.recoil_angular_per_s * config.saturation * scaled_delta / denominator ** 2
    trap_depth = config.positive("experiment.trap.depth_over_kB_during_state_readout_mK")
    return dict(quantization_voltage_V=float(voltage), b_G=field,
                b_effective_trap_G=float(np.linalg.norm(vector)),
                trap_precession_turns=abs(config.mu * config.ground_g) * np.linalg.norm(vector) * config.trap_us,
                p_m_changes_one_trap_pulse=p_change,
                p_any_m_change_independent_resets=float(p_reset),
                cycling_detuning_MHz=float(delta), photons_on_window=float(photons),
                recoil_ledger_mK=float(recoil), recoil_to_trap_depth_ratio=float(recoil / trap_depth),
                mean_detected_photons=float(photons * config.fraction("experiment.fluorescence_collection.actual_efficiency_fraction")),
                friction_on_per_mass_inv_s=float(friction),
                readout_cycles=config.cycles, total_probe_on_us=config.total_on_us)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=Path("outputs"))
    parser.add_argument("--config-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = Parameters(args.config_dir)
        rows = [model(v, config) for v in config.scan_voltages()]
        path = write_rows(args.outdir / "scan_grid.csv", rows, config)
        plt = setup_plot(config)
        fig, axes = plt.subplots(2, 1, sharex=True, layout="constrained",
                                 figsize=config.get("numerics.plot.figure_size_inches"))
        x = [r["quantization_voltage_V"] for r in rows]
        axes[0].plot(x, [r["p_m_changes_one_trap_pulse"] for r in rows])
        axes[0].set_ylabel("Single-window spin change probability")
        axes[1].plot(x, [r["recoil_ledger_mK"] for r in rows])
        axes[1].set_ylabel("Uncooled recoil energy / kB (mK)")
        axes[1].set_xlabel("Quantization-field control voltage (V)")
        fig.savefig(args.outdir / "scan_diagnostics.png", dpi=config.number("numerics.plot.dpi"))
        plt.close(fig)
        print(path)
    except ParameterError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
