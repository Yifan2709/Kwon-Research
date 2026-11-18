"""Optional phenomenological diagnostics with explicitly configured model inputs.

The extra temperature and voltage envelopes require independent choices in
numerics.json. They are not inferred from the apparatus settings.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit, gammainc

from .output import setup_plot, write_rows
from .parameters import Parameters, ParameterError


def parameter(config, key):
    return config.number("numerics.phenomenological." + key)


def spin(config, voltage, period_multiplier=1):
    memory = parameter(config, "memory_amplitude")
    if not 0 <= memory < 1:
        raise ParameterError("numerics.phenomenological.memory_amplitude must lie in [0,1)")
    phase = 2 * np.pi * config.mu * config.ground_g * config.field_g(voltage) * config.period_us * period_multiplier
    return (1 - memory) ** 2 / ((1 - memory) ** 2 + 4 * memory * np.sin(phase / 2) ** 2)


def temperature(config, voltage, response, flat=False):
    base = parameter(config, "base_temperature_mK")
    extra = parameter(config, "extra_temperature_mK")
    if base <= 0 or extra < 0:
        raise ParameterError("Phenomenological base temperature must be positive and additional temperature nonnegative")
    if flat:
        return base + extra * response
    width = parameter(config, "activation_width_V")
    edge_width = parameter(config, "right_edge_width_V")
    edge_temperature = parameter(config, "right_edge_temperature_mK")
    if width <= 0 or edge_width <= 0 or edge_temperature < 0:
        raise ParameterError("Phenomenological envelope widths must be positive and heating nonnegative")
    gate = expit((voltage - parameter(config, "activation_voltage_V")) / width)
    edge = expit((voltage - parameter(config, "right_edge_voltage_V")) / edge_width)
    return base + edge_temperature * edge + extra * gate * response


def survival(config, temperature_mK):
    depth = config.positive("experiment.trap.depth_over_kB_during_state_readout_mK")
    # Three motional axes give the 3D classical harmonic-oscillator energy CDF.
    return gammainc(3, depth / temperature_mK)


def model(config, voltage, period_multiplier=1, flat=False):
    memory = parameter(config, "memory_amplitude")
    low = ((1 - memory) / (1 + memory)) ** 2
    if memory == 0:
        raise ParameterError("memory_amplitude must be positive for normalized spin heating")
    response = (spin(config, voltage, period_multiplier) - low) / (1 - low)
    return survival(config, temperature(config, voltage, response, flat))


def recoil(config, voltage):
    delta = config.detuning_mhz(config.field_g(voltage))
    photons = config.total_on_us * config.gamma * config.saturation / (2 * (1 + config.saturation + 4 * (delta / config.linewidth_MHz) ** 2))
    depth = config.positive("experiment.trap.depth_over_kB_during_state_readout_mK")
    diffusion = 2 * config.recoil_temperature_mK * photons / (6 * depth)
    count = config.integer("numerics.recoil.series_terms")
    tolerance = config.positive("numerics.validation.probability_atol")
    diffusion = np.asarray(diffusion, dtype=float)
    result = np.ones_like(diffusion)
    # Jacobi transformation avoids cancellation of the alternating series in
    # the weak-recoil limit. The analytic crossover is pi*D=1, not a fitted input.
    small = (diffusion > 0) & (np.pi * diffusion <= 1)
    large = np.pi * diffusion > 1
    d = diffusion[small]
    complement = sum(np.exp(-(k + 1 / 2) ** 2 / d) for k in range(count))
    coefficient = 2 / np.sqrt(np.pi * d)
    result[small] = 1 - coefficient * complement
    # Bound the remaining positive transformed-series terms by a geometric tail.
    tail = coefficient * np.exp(-(count + 1 / 2) ** 2 / d) / (-np.expm1(-(2 * count + 2) / d))
    d = diffusion[large]
    result[large] = sum(2 * (-1) ** (k + 1) * np.exp(-k * k * np.pi ** 2 * d)
                        for k in range(1, count + 1))
    alternating_error = 2 * np.exp(-(count + 1) ** 2 * np.pi ** 2 * d)
    if np.any(tail > tolerance) or np.any(alternating_error > tolerance):
        raise ParameterError("Increase numerics.recoil.series_terms: recoil survival series has not converged")
    return float(result) if result.ndim == 0 else result


def alias(config, voltage, speed_multiplier=1):
    speed = parameter(config, "scan_speed_V_per_s") * speed_multiplier
    period = parameter(config, "drift_period_s")
    power = parameter(config, "time_profile_power")
    if speed == 0 or period <= 0 or power <= 0:
        raise ParameterError("Drift period and profile power must be positive, scan speed nonzero")
    start = config.number("experiment.scan.start_voltage_V")
    phase = 2 * np.pi * (voltage - start) / (speed * period) + parameter(config, "drift_phase_rad")
    response = ((1 + np.cos(phase)) / 2) ** power
    return survival(config, temperature(config, voltage, response))


def calculate(config):
    voltage = config.scan_voltages()
    period_multiplier = parameter(config, "period_multiplier")
    speed_multiplier = parameter(config, "scan_speed_multiplier")
    if period_multiplier <= 0 or speed_multiplier <= 0:
        raise ParameterError("Period and scan-speed multipliers must be positive")
    columns = dict(quantization_voltage_V=voltage, b_G=config.field_g(voltage),
                   recoil=recoil(config, voltage), spin_response=spin(config, voltage),
                   spin_only_survival=model(config, voltage, flat=True),
                   combined=model(config, voltage), changed_period=model(config, voltage, period_multiplier),
                   alias=alias(config, voltage), changed_scan_speed=alias(config, voltage, speed_multiplier))
    return [dict(zip(columns, values)) for values in zip(*columns.values())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    try:
        config = Parameters(args.config_dir)
        rows = calculate(config)
        path = write_rows(args.output_dir / "scenarios.csv", rows, config)
        plt = setup_plot(config)
        fig, ax = plt.subplots(figsize=config.get("numerics.plot.figure_size_inches"), layout="constrained")
        x = [r["quantization_voltage_V"] for r in rows]
        for key in ("recoil", "combined", "changed_period", "alias", "changed_scan_speed"):
            ax.plot(x, [r[key] for r in rows], label=key)
        ax.set(xlabel="Quantization-field control voltage (V)", ylabel="Conditional model survival")
        ax.legend()
        fig.savefig(args.output_dir / "scenarios.png", dpi=config.number("numerics.plot.dpi"))
        plt.close(fig)
        print(path)
    except ParameterError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
