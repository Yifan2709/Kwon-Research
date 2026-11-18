"""Finite probe-pulse coherence using the complete closed optical manifold.

Parameters and sampling choices come from config/*.json. The initial state is
propagated through the first trap and dark interval before the probe is turned
on. The configured pulse beam model can resolve the measured optical beat and
average over an unknown relative optical phase. This model does not include
F=1 depumping, atomic motion, trap escape, or camera detection.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .coherence_scan import ClosedTransition
from .output import write_rows
from .parameters import Parameters, ParameterError


def sample_times(config):
    count = config.integer("numerics.integration.sample_count")
    if count < 2:
        raise ParameterError("numerics.integration.sample_count must include at least both pulse endpoints")
    return np.linspace(0, config.probe_us, count)


def _preparation(model, pure_stretched):
    cfg = model.config
    b = cfg.field_g(cfg.number("experiment.quantization_field.single_pulse_voltage_V"))
    delta = cfg.detuning_mhz(b)
    hbase = model.base_hamiltonian(b, delta)
    initial = model.initial_rho.copy()
    if pure_stretched:
        initial[:] = 0
        initial[model.g_cycle, model.g_cycle] = 1
    else:
        trap = model.interval_map(hbase + model.trap_hamiltonian(), cfg.trap_us)
        gap = model.interval_map(hbase, cfg.dark_before_probe_us, axial=True)
        initial = (gap @ trap @ initial.ravel(order="F")).reshape(initial.shape, order="F")
    times = sample_times(cfg)
    return hbase, initial, times


def _pulse(model, phase, pure_stretched, prepared=None):
    cfg = model.config
    hbase, initial, times = _preparation(model, pure_stretched) if prepared is None else prepared
    beat_mhz = None
    if phase is not None:
        if cfg.integer("experiment.probe.beam_count") != 2:
            raise ParameterError("equal_beams_uniform_phase requires exactly two beams")
        if not np.isfinite(phase):
            raise ParameterError("Relative optical phase must be finite")
        beat_mhz = cfg.number("experiment.probe.absolute_frequency_difference_kHz") / 1000
        if beat_mhz < 0:
            raise ParameterError("experiment.probe.absolute_frequency_difference_kHz must be nonnegative")
    if cfg.probe_us == 0:
        return np.repeat(initial[None, :, :], len(times), axis=0), np.zeros(len(times))

    def rhs(t, y):
        coefficient = model.rabi / 2
        if phase is not None:
            optical_phase = np.pi * beat_mhz * t
            coefficient *= (np.exp(1j * optical_phase) + np.exp(1j * (phase - optical_phase))) / np.sqrt(2)
        h = hbase + coefficient * model.absorption + np.conjugate(coefficient) * model.absorption.conj().T
        rho = y[:-1].reshape(initial.shape)
        derivative = -1j * (h @ rho - rho @ h) - (model.loss @ rho + rho @ model.loss) / 2
        derivative += sum(jump @ rho @ jump.conj().T for jump in model.jumps.values())
        return np.r_[derivative.ravel(), np.trace(model.loss @ rho)]

    solution = solve_ivp(
        rhs, (0, cfg.probe_us), np.r_[initial.ravel(), 0j], t_eval=times,
        method=cfg.require("numerics.integration.method"),
        rtol=cfg.positive("numerics.integration.rtol"),
        atol=cfg.positive("numerics.integration.atol"),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    rho = solution.y[:-1].T.reshape(-1, model.size, model.size)
    model.check_density(rho)
    return rho, solution.y[-1].real


def pulse(phase=None, pure_stretched=False, config=None):
    """Return density matrices and mean emission counts at configured times.

    phase=None selects the configured pulse beam model, including the configured
    phase quadrature when requested. A supplied phase selects two equal coherent
    beams separated by the configured frequency difference at that known phase.
    pure_stretched=True is a diagnostic preparation at light turn-on, bypassing
    the preceding trap interval; otherwise the configured initial state is used.
    """
    model = ClosedTransition(config)
    if phase is not None:
        return _pulse(model, phase, pure_stretched)
    return _configured_pulse(model, pure_stretched)


def _configured_pulse(model, pure_stretched):
    cfg = model.config
    name = cfg.require("experiment.probe.pulse_beam_model")
    if name == "effective_uniform":
        return _pulse(model, None, pure_stretched)
    elif name == "equal_beams_uniform_phase":
        count = cfg.integer("numerics.integration.phase_sample_count")
        prepared = _preparation(model, pure_stretched)
        results = [_pulse(model, phase, pure_stretched, prepared) for phase in 2 * np.pi * np.arange(count) / count]
        return tuple(np.mean(values, axis=0) for values in zip(*results))
    else:
        raise ParameterError(
            "experiment.probe.pulse_beam_model must be effective_uniform or equal_beams_uniform_phase"
        )


def records(config=None):
    cfg = Parameters() if config is None else config
    model = ClosedTransition(cfg)
    name = cfg.require("experiment.probe.pulse_beam_model")
    rho, photons = _configured_pulse(model, False)
    norm = abs(rho[0, model.g_adjacent, model.g_cycle])
    ratio_defined = bool(norm > model.matrix_atol)
    rows = []
    for t, density, mean in zip(sample_times(cfg), rho, photons):
        coherence = abs(density[model.g_adjacent, model.g_cycle])
        rows.append(dict(
            model=name, t_us=float(t), drive_q=cfg.q,
            cycling_ground_mF=model.cycling_m, adjacent_ground_mF=model.adjacent_m,
            cycling_excited_mF=model.cycling_m + cfg.q, adjacent_excited_mF=model.adjacent_m + cfg.q,
            ground_population=float(np.trace(density[:model.ground_size, :model.ground_size]).real),
            excited_population=float(np.trace(density[model.ground_size:, model.ground_size:]).real),
            ground_cycling_population=float(density[model.g_cycle, model.g_cycle].real),
            ground_adjacent_population=float(density[model.g_adjacent, model.g_adjacent].real),
            excited_cycling_population=float(density[model.e_cycle, model.e_cycle].real),
            excited_adjacent_population=float(density[model.e_adjacent, model.e_adjacent].real),
            abs_ground_adjacent_coherence=float(coherence),
            ground_coherence_ratio=float(coherence / norm) if ratio_defined else None,
            ground_coherence_ratio_defined=ratio_defined,
            abs_excited_adjacent_coherence=float(abs(density[model.e_adjacent, model.e_cycle])),
            mean_emissions=float(mean),
        ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = Parameters(args.config_dir)
        print(write_rows(args.csv, records(config), config))
    except ParameterError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
