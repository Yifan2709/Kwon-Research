"""Closed-transition spin and optical coherence through complete strobe periods.

The configured ground and maximal excited manifolds are retained in full. The
effective-uniform scan averages the two beams' intensity; it does not resolve
their optical beat. Noncycling photons are emissions with q different from the
driving polarization. They are not F=1 depumping, atom loss, or camera survival.
Time is measured in microseconds and Hamiltonians in radians/microsecond.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .atomic import angular_momentum, emission_matrix, validate
from .output import write_rows
from .parameters import Parameters, ParameterError


class ClosedTransition:
    """One file-backed closed-manifold model, shared with the finite pulse code."""

    def __init__(self, config=None):
        self.config = Parameters() if config is None else config
        cfg = self.config
        if cfg.excited_F != cfg.ground_F + 1:
            raise ParameterError("Coherence model requires the closed F to F'=F+1 transition")
        if any(fraction != 0 for q, fraction in cfg.polarization_fractions.items() if q != cfg.q):
            raise ParameterError(
                "The coherent model requires pure stated circular polarization. "
                "Nonzero pi/opposite fractions need their relative optical phases; "
                "they cannot be silently treated as one coherent drive."
            )
        validate(cfg)
        self.ground_m = np.arange(-cfg.ground_F, cfg.ground_F + 1)
        self.excited_m = np.arange(-cfg.excited_F, cfg.excited_F + 1)
        self.ground_size = len(self.ground_m)
        self.size = self.ground_size + len(self.excited_m)
        self.eye = np.eye(self.size)
        self.excited = np.diag(np.r_[np.zeros(self.ground_size), np.ones(len(self.excited_m))])
        self.ground_spin = tuple(self.embed(op, 0) for op in angular_momentum(cfg.ground_F))
        self.excited_spin = tuple(self.embed(op, self.ground_size) for op in angular_momentum(cfg.excited_F))
        self.emissions = {q: emission_matrix(cfg, q) for q in (-1, 0, 1)}
        self.jumps = {q: np.sqrt(cfg.gamma) * operator for q, operator in self.emissions.items()}
        self.loss = sum(jump.conj().T @ jump for jump in self.jumps.values())
        self.recycling = sum(np.kron(jump.conj(), jump) for jump in self.jumps.values())
        self.noncycling_recycling = sum(
            np.kron(jump.conj(), jump) for q, jump in self.jumps.items() if q != cfg.q
        )
        self.dissipator = self.recycling - (
            np.kron(self.eye, self.loss) + np.kron(self.loss.T, self.eye)
        ) / 2
        self.absorption = self.emissions[cfg.q].conj().T
        self.rabi = cfg.gamma * np.sqrt(cfg.saturation / 2)
        self.drive = self.rabi * (self.absorption + self.absorption.conj().T) / 2
        self.cycling_m = cfg.cycling_ground_m
        self.adjacent_m = self.cycling_m - cfg.q
        self.g_cycle = self.ground_index(self.cycling_m)
        self.g_adjacent = self.ground_index(self.adjacent_m)
        self.e_cycle = self.excited_index(self.cycling_m + cfg.q)
        self.e_adjacent = self.excited_index(self.adjacent_m + cfg.q)
        self.initial_rho = np.zeros((self.size, self.size), dtype=complex)
        self.initial_rho[self.ground_index(cfg.initial_m), self.ground_index(cfg.initial_m)] = 1
        labels = np.r_[self.ground_m, self.excited_m - cfg.q]
        coherence_order = (labels[:, None] - labels[None, :]).ravel(order="F")
        self.blocks = [np.flatnonzero(coherence_order == order) for order in np.unique(coherence_order)]
        self.matrix_atol = cfg.positive("numerics.validation.matrix_atol")
        self.probability_atol = cfg.positive("numerics.validation.probability_atol")

    def embed(self, operator, offset):
        result = np.zeros((self.size, self.size), dtype=complex)
        result[offset:offset + len(operator), offset:offset + len(operator)] = operator
        return result

    def ground_index(self, m):
        indices = np.flatnonzero(self.ground_m == m)
        if len(indices) != 1:
            raise ParameterError(f"Ground mF={m} does not exist in the configured manifold")
        return int(indices[0])

    def excited_index(self, m):
        indices = np.flatnonzero(self.excited_m == m)
        if len(indices) != 1:
            raise ParameterError(f"Excited mF={m} does not exist in the configured manifold")
        return self.ground_size + int(indices[0])

    def base_hamiltonian(self, b, delta):
        """Static axial field; delta is measured from the cycling line at b."""
        if not np.isfinite(b) or not np.isfinite(delta):
            raise ParameterError("Magnetic field and cycling-line detuning must be finite")
        cfg = self.config
        zg = 2 * np.pi * cfg.mu * b * cfg.ground_g
        ze = 2 * np.pi * cfg.mu * b * cfg.excited_g
        offset = -2 * np.pi * delta - ((self.cycling_m + cfg.q) * ze - self.cycling_m * zg)
        return zg * self.ground_spin[-1] + ze * self.excited_spin[-1] + offset * self.excited

    def trap_hamiltonian(self):
        """Ground-manifold vector shift; all three field components are used."""
        cfg = self.config
        return 2 * np.pi * cfg.mu * cfg.ground_g * sum(
            field * operator for field, operator in zip(cfg.trap_field_vector, self.ground_spin)
        )

    def generator(self, h, conditional=False):
        result = -1j * (np.kron(self.eye, h) - np.kron(h.T, self.eye)) + self.dissipator
        if conditional:
            # Retain the loss anticommutator and omit only counted recycling.
            result = result - self.noncycling_recycling
        return result

    def exp_blockdiag(self, operator):
        """Exact axial block decomposition for a pure single-q drive."""
        result = np.zeros_like(operator)
        remainder = operator.copy()
        for block in self.blocks:
            indices = np.ix_(block, block)
            result[indices] = expm(operator[indices])
            remainder[indices] = 0
        if np.max(np.abs(remainder)) > self.matrix_atol:
            raise RuntimeError("Attempted axial block propagation for a non-axial generator")
        return result

    def interval_map(self, h, duration, conditional=False, axial=False):
        if duration == 0:
            return np.eye(self.size**2, dtype=complex)
        operator = self.generator(h, conditional) * duration
        return self.exp_blockdiag(operator) if axial else expm(operator)

    def period_maps(self, b, delta, conditional=False):
        """Maps ordered as trap, dark-before-probe, probe, dark-after-probe.

        Spontaneous emission and excited-state Zeeman motion remain active in
        every interval, including the two gaps and the next trap window.
        """
        cfg = self.config
        if cfg.require("experiment.probe.scan_beam_model") != "effective_uniform":
            raise ParameterError(
                "experiment.probe.scan_beam_model must be effective_uniform for the "
                "periodic coherent scan; use the single-pulse model to resolve the optical beat"
            )
        hbase = self.base_hamiltonian(b, delta)
        return {
            "trap": self.interval_map(hbase + self.trap_hamiltonian(), cfg.trap_us, conditional),
            "dark_before_probe": self.interval_map(hbase, cfg.dark_before_probe_us, conditional, axial=True),
            "probe": self.interval_map(hbase + self.drive, cfg.probe_us, conditional, axial=True),
            "dark_after_probe": self.interval_map(hbase, cfg.dark_after_probe_us, conditional, axial=True),
        }

    def check_density(self, rho):
        if rho.shape[-2:] != (self.size, self.size):
            raise ValueError("Density matrix shape does not match the configured manifolds")
        if not np.all(np.isfinite(rho)):
            raise RuntimeError("Density matrix contains non-finite values")
        if np.max(np.abs(rho - np.swapaxes(rho.conj(), -1, -2))) > self.probability_atol:
            raise RuntimeError("Density matrix lost Hermiticity")
        if np.max(np.abs(np.trace(rho, axis1=-2, axis2=-1) - 1)) > self.probability_atol:
            raise RuntimeError("Density matrix trace is not one")
        if np.min(np.linalg.eigvalsh(rho)) < -self.probability_atol:
            raise RuntimeError("Density matrix is not positive within configured tolerance")

    def outside_pair(self, rho):
        return float((np.trace(rho) - rho[self.g_cycle, self.g_cycle] - rho[self.e_cycle, self.e_cycle]).real)


def _period(maps):
    return maps["dark_after_probe"] @ maps["probe"] @ maps["dark_before_probe"] @ maps["trap"]


def propagate(b, delta, start_rho=None, count_events=None, config=None):
    """Propagate configured periods and report internal-state observables.

    The coherence transfer coefficient is the response to a unit adjacent
    ground coherence, so it remains defined when the prepared coherence is zero.
    The residual-decay coefficient removes free ground Larmor phase and lets
    the remaining excited coherence decay with no new trap pulse.
    """
    model = ClosedTransition(config)
    cfg = model.config
    if count_events is None:
        count_events = cfg.require("numerics.scan.count_noncycling_events")
    if not isinstance(count_events, bool):
        raise ParameterError("numerics.scan.count_noncycling_events must be a boolean")
    initial = model.initial_rho if start_rho is None else np.asarray(start_rho, dtype=complex)
    model.check_density(initial)
    maps = model.period_maps(b, delta)
    vector = np.linalg.matrix_power(_period(maps), cfg.cycles - 1) @ initial.ravel(order="F")
    pre_vector = maps["dark_before_probe"] @ maps["trap"] @ vector
    post_vector = maps["probe"] @ pre_vector
    pre = pre_vector.reshape(initial.shape, order="F")
    post = post_vector.reshape(initial.shape, order="F")
    end = (maps["dark_after_probe"] @ post_vector).reshape(initial.shape, order="F")
    for density in (pre, post, end):
        model.check_density(density)

    before = np.zeros_like(initial)
    before[model.g_adjacent, model.g_cycle] = 1
    response = (maps["probe"] @ before.ravel(order="F")).reshape(initial.shape, order="F")
    rg = response[model.g_adjacent, model.g_cycle]
    re = response[model.e_adjacent, model.e_cycle]
    frequency_difference = 2 * np.pi * cfg.mu * b * (cfg.excited_g - cfg.ground_g) * (model.adjacent_m - model.cycling_m)
    dipole = model.emissions[cfg.q]
    transfer = dipole[model.g_adjacent, model.e_adjacent] * dipole[model.g_cycle, model.e_cycle].conjugate()
    after = rg + transfer * cfg.gamma * re / (cfg.gamma + 1j * frequency_difference)

    event_probability = None
    if count_events:
        no_count_maps = model.period_maps(b, delta, conditional=True)
        no_count = np.linalg.matrix_power(_period(no_count_maps), cfg.cycles) @ initial.ravel(order="F")
        p0 = float(np.trace(no_count.reshape(initial.shape, order="F")).real)
        if not -model.probability_atol <= p0 <= 1 + model.probability_atol:
            raise RuntimeError("No-count probability is outside [0,1] within configured tolerance")
        event_probability = float(np.clip(1 - p0, 0, 1))

    return dict(
        beam_model=cfg.require("experiment.probe.scan_beam_model"), b_G=float(b),
        delta_cycling_MHz=float(delta), readout_cycles=cfg.cycles,
        readout_duration_us=cfg.cycles * cfg.period_us,
        drive_q=cfg.q, initial_mF=cfg.initial_m,
        prob_emitted_at_least_one_noncycling_photon=event_probability,
        ground_coherence_ratio_probe_end=float(abs(rg)),
        ground_coherence_ratio_after_residual_decay=float(abs(after)),
        fraction_outside_cycling_pair_preprobe=model.outside_pair(pre),
        fraction_outside_cycling_pair_postprobe=model.outside_pair(post),
        fraction_outside_cycling_pair_period_end=model.outside_pair(end),
        abs_ground_adjacent_coherence_preprobe=float(abs(pre[model.g_adjacent, model.g_cycle])),
        abs_ground_adjacent_coherence_postprobe=float(abs(post[model.g_adjacent, model.g_cycle])),
        excited_population_period_end=float(np.trace(model.excited @ end).real),
    )


def records(config=None):
    cfg = Parameters() if config is None else config
    rows = []
    for voltage in cfg.scan_voltages():
        b = cfg.field_g(voltage)
        rows.append(dict(quantization_voltage_V=float(voltage),
                         **propagate(b, cfg.detuning_mhz(b), config=cfg)))
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
