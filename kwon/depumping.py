"""Configured population-rate diagnostic for state-selective readout.

Times are in microseconds, ordinary detunings in MHz, and fields in gauss.
The bright manifold is resolved in mF; the lower ground hyperfine manifold
is an absorbing sink. Probe scattering uses independent, line-specific
steady-state rates for each configured beam, polarization, and excited
hyperfine level, followed by spontaneous-emission branching. Beam frequency
offsets are retained, but optical interference and saturation competition
between different beams/lines are not. This approximation can overestimate
scattering when several beams or lines strongly saturate simultaneously.

Trap evolution includes all three components of the configured fictitious
field. Complete ground Zeeman dephasing discards coherences at the probe
boundary and between periods. Both dark gaps are included in the sequence;
their axial free precession cannot change already diagonal populations.
This model therefore cannot predict cross-period coherent Larmor resonances,
finite optical transients, motional heating, escape, or fluorescence-based
survival. A lower-manifold probability is a depumping probability, not an
atom-loss probability. No experimental or numerical defaults live here.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .atomic import angular_momentum, strength
from .output import write_rows
from .parameters import ParameterError, Parameters


def _manifolds(cfg):
    bright = cfg.ground_F
    lower = abs(cfg.number("atomic.nuclear_spin") - cfg.number("atomic.ground_J"))
    upper = cfg.number("atomic.nuclear_spin") + cfg.number("atomic.ground_J")
    if bright != upper or lower == bright:
        raise ParameterError(
            "The depumping model requires the initial F to be the upper ground "
            "hyperfine manifold, with a distinct lower manifold as its sink."
        )
    magnetic = np.arange(-bright, bright + 1)
    return bright, lower, magnetic


def _excited_offsets(cfg):
    raw = cfg.require("atomic.excited_hyperfine_offsets_MHz")
    if not isinstance(raw, dict) or not raw:
        raise ParameterError("atomic.excited_hyperfine_offsets_MHz must be a nonempty mapping.")
    result = {}
    for key in raw:
        try:
            f = float(key)
        except (TypeError, ValueError) as exc:
            raise ParameterError("Excited hyperfine offset keys must be angular momenta F'.") from exc
        if not np.isfinite(f) or f < 0 or not (2 * f).is_integer():
            raise ParameterError("Excited hyperfine offset keys must be nonnegative integer or half-integer F'.")
        result[f] = cfg.number(f"atomic.excited_hyperfine_offsets_MHz.{key}")
    if cfg.excited_F not in result:
        raise ParameterError("The configured cycling excited F' has no atomic hyperfine offset.")
    return result


def _polarizations(cfg):
    return tuple(cfg.polarization_fractions.items())


def _beam_offsets(cfg):
    count = cfg.number("experiment.probe.beam_count")
    if count == 1:
        return (0.0,)
    if count != 2:
        raise ParameterError("The depumping beam-frequency model supports one beam or a symmetric pair.")
    difference = cfg.number("experiment.probe.absolute_frequency_difference_kHz")
    if difference < 0:
        raise ParameterError("experiment.probe.absolute_frequency_difference_kHz must be nonnegative.")
    half_difference_MHz = difference / (2 * 1000)
    return (-half_difference_MHz, half_difference_MHz)


def channels(cfg):
    """Return absorption paths and normalized spontaneous-emission branches.

    Each tuple is (source_index, m, excited_F, excited_m, weighted_strength,
    emission_vector). Bright states run from -F to +F; the last entry is the
    aggregate lower-F sink. The strength includes polarization fraction but
    not intensity; intensity is applied once by :func:`probe_generator`.
    """
    bright, lower, magnetic = _manifolds(cfg)
    dark = len(magnetic)
    tolerance = cfg.number("numerics.validation.probability_atol")
    paths = []
    for state, m in enumerate(magnetic):
        for polarization, fraction in _polarizations(cfg):
            if fraction == 0:
                continue
            for excited in _excited_offsets(cfg):
                weighted_strength = fraction * strength(cfg, bright, m, polarization, excited)
                if weighted_strength == 0:
                    continue
                excited_m = m + polarization
                emission = np.zeros(dark + 1)
                for final_F in (lower, bright):
                    for emitted_q in (-1, 0, 1):
                        final_m = excited_m - emitted_q
                        if abs(final_m) <= final_F:
                            destination = dark if final_F == lower else int(final_m + bright)
                            emission[destination] += strength(cfg, final_F, final_m, emitted_q, excited)
                total = emission.sum()
                if not np.isfinite(total) or total <= 0:
                    raise ParameterError("An excited hyperfine state has no allowed spontaneous decay branch.")
                # Dipole strengths are cycling-normalized; normalize their
                # relative spontaneous weights independently of that convention.
                emission /= total
                if not np.isclose(emission.sum(), 1, atol=tolerance, rtol=0):
                    raise ArithmeticError("Spontaneous-emission branching failed normalization.")
                paths.append((state, m, excited, excited_m, weighted_strength, emission))
    return paths


def _laser_zero_field_detuning(cfg, b_G):
    offsets = _excited_offsets(cfg)
    cycling_shift = cfg.mu * b_G * (cfg.excited_g * cfg.cycling_excited_m - cfg.ground_g * cfg.cycling_ground_m)
    return cfg.detuning_mhz(b_G) + offsets[cfg.excited_F] + cycling_shift


def probe_generator(cfg, b_G):
    """Column-convention continuous-time population generator, in us^-1.

    Each beam contributes Gamma/2 * s/(1+s+(2*detuning/linewidth)^2),
    with s = configured per-beam s0 * polarization fraction * line strength.
    Spontaneous returns to the source cancel from the population generator;
    consequently it counts population transfers, not all scattered photons.
    """
    _, _, magnetic = _manifolds(cfg)
    qmat = np.zeros((len(magnetic) + 1, len(magnetic) + 1))
    detuning_zero = _laser_zero_field_detuning(cfg, b_G)
    offsets = _excited_offsets(cfg)
    beam_offsets = _beam_offsets(cfg)
    saturation = cfg.beam_saturation
    if saturation < 0:
        raise ParameterError("experiment.probe.reported_s0_per_beam must be nonnegative.")
    linewidth = cfg.number("atomic.linewidth_MHz")
    for state, m, excited, excited_m, weighted_strength, emission in channels(cfg):
        s = saturation * weighted_strength
        line_shift = offsets[excited] + cfg.mu * b_G * (cfg.excited_g * excited_m - cfg.ground_g * m)
        for beam_offset in beam_offsets:
            detuning = detuning_zero + beam_offset - line_shift
            rate = cfg.gamma / 2 * s / (1 + s + (2 * detuning / linewidth) ** 2)
            qmat[:, state] += rate * emission
            qmat[state, state] -= rate
    tolerance = cfg.number("numerics.validation.matrix_atol")
    if not np.allclose(qmat.sum(axis=0), 0, atol=tolerance, rtol=0):
        raise ArithmeticError("Probe population generator does not conserve probability.")
    return qmat


def _field_map(cfg, field_G, duration_us):
    """Unitary spin evolution followed by complete Zeeman dephasing."""
    _, _, magnetic = _manifolds(cfg)
    fx, fy, fz = angular_momentum(cfg.ground_F)
    bx, by, bz = field_G
    hamiltonian = 2 * np.pi * cfg.mu * cfg.ground_g * (bx * fx + by * fy + bz * fz)
    transition = np.eye(len(magnetic) + 1)
    transition[:-1, :-1] = np.abs(expm(-1j * duration_us * hamiltonian)) ** 2
    return transition


def trap_map(cfg, b_G):
    """Population map for the configured trap interval and full vector field."""
    field = np.array(cfg.trap_field_vector, dtype=float, copy=True)
    field[2] += b_G
    return _field_map(cfg, field, cfg.trap_us)


def period_map(cfg, b_G):
    """Trap -> dark gap -> probe -> dark gap, with complete dephasing."""
    axial_field = (0, 0, b_G)
    before = _field_map(cfg, axial_field, cfg.dark_before_probe_us)
    after = _field_map(cfg, axial_field, cfg.dark_after_probe_us)
    probe = expm(cfg.probe_us * probe_generator(cfg, b_G))
    return after @ probe @ before @ trap_map(cfg, b_G)


def _first_depumping_probability(qmat, initial_index, neighbor_index):
    """Hit the lower manifold before returning to the prepared mF state.

    Pruning states with no path to the dark sink makes zero-intensity and
    closed optical cycles well defined, without arbitrary numerical rates.
    """
    dark = len(qmat) - 1
    reachable = {dark}
    pending = [dark]
    while pending:
        destination = pending.pop()
        for source in range(dark):
            if source != initial_index and source not in reachable and qmat[destination, source] > 0:
                reachable.add(source)
                pending.append(source)
    if neighbor_index not in reachable:
        return 0.0
    transients = sorted(reachable - {dark})
    probabilities = np.linalg.solve(
        -qmat[np.ix_(transients, transients)].T, qmat[dark, transients]
    )
    return float(probabilities[transients.index(neighbor_index)])


def estimate(cfg, b_G):
    """Depumping diagnostics for the configured prepared state and readout."""
    bright, lower, magnetic = _manifolds(cfg)
    initial_index = int(cfg.initial_m + bright)
    neighbor_m = cfg.initial_m - cfg.q
    if neighbor_m not in magnetic:
        raise ParameterError("The first-return diagnostic requires initial_mF - probe_q inside the bright manifold.")
    neighbor_index = int(neighbor_m + bright)
    qmat = probe_generator(cfg, b_G)
    cycles = cfg.cycles
    no_trap = expm(cfg.probe_us * cycles * qmat)
    with_trap = np.linalg.matrix_power(period_map(cfg, b_G), cycles)
    tolerance = cfg.number("numerics.validation.probability_atol")
    for transition in (no_trap, with_trap):
        if not np.allclose(transition.sum(axis=0), 1, atol=tolerance, rtol=0):
            raise ArithmeticError("Readout map does not conserve probability at the configured tolerance.")
        if transition.min() < -tolerance or transition.max() > 1 + tolerance:
            raise ArithmeticError("Readout map contains an invalid probability.")
    return {
        "field_G": float(b_G),
        "initial_F": bright,
        "initial_mF": cfg.initial_m,
        "neighbor_mF": neighbor_m,
        "lower_F": lower,
        "readout_cycles": cycles,
        "readout_duration_us": cfg.period_us * cycles,
        "probe_on_duration_us": cfg.probe_us * cycles,
        "detuning_from_local_cycling_MHz": cfg.detuning_mhz(b_G),
        "laser_detuning_from_zero_field_MHz": _laser_zero_field_detuning(cfg, b_G),
        "rate_to_lower_F_from_initial_per_s": float(1e6 * qmat[-1, initial_index]),
        "rate_to_lower_F_from_neighbor_per_s": float(1e6 * qmat[-1, neighbor_index]),
        "p_lower_F_before_return_from_neighbor": _first_depumping_probability(qmat, initial_index, neighbor_index),
        "p_lower_F_from_initial_without_trap_mixing": float(no_trap[-1, initial_index]),
        "p_lower_F_from_initial_with_dephased_trap": float(with_trap[-1, initial_index]),
        "p_lower_F_from_uniform_without_trap_mixing": float(no_trap[-1, :-1].mean()),
        "p_lower_F_from_uniform_with_dephased_trap": float(with_trap[-1, :-1].mean()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path, nargs="?", default=Path("outputs"))
    parser.add_argument("--config-dir", type=Path, help="Directory containing the fixed parameter JSON files.")
    parser.add_argument("--csv", type=Path, help="Explicit output CSV path; otherwise outdir/depumping.csv.")
    args = parser.parse_args(argv)
    try:
        cfg = Parameters(args.config_dir)
        rows = [
            {"quantization_voltage_V": float(voltage), **estimate(cfg, cfg.field_g(voltage))}
            for voltage in cfg.scan_voltages()
        ]
        if not rows:
            raise ParameterError("The configured voltage scan contains no points.")
    except (ParameterError, ValueError) as exc:
        parser.error(str(exc))
    destination = args.csv if args.csv is not None else args.outdir / "depumping.csv"
    write_rows(destination, rows, cfg)
    print(destination)
    return destination


if __name__ == "__main__":
    main()
