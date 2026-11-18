"""Single file-backed source for experiment, atom and numerical settings.

Physical and numerical settings live in config/*.json. The properties below
only validate inputs or derive quantities and unit conversions from them.
"""

import argparse
import json
from pathlib import Path

import numpy as np


class ConfigurationError(ValueError):
    """An input setting is missing, ambiguous, or physically inconsistent."""


ParameterError = ConfigurationError


def unique_keys(pairs):
    """Reject duplicate JSON keys instead of silently accepting the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _finite_json(value, path):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{path}[{index}]")
    elif isinstance(value, (float, int)) and not np.isfinite(value):
        raise ConfigurationError(f"{path} must be finite")


class Parameters:
    """Read the same three JSON files once for one calculation configuration."""

    def __init__(self, config_dir=None):
        self.config_dir = (
            Path(config_dir).expanduser().resolve()
            if config_dir is not None
            else Path(__file__).resolve().parent.parent / "config"
        )
        self.data = {}
        for namespace in ("atomic", "experiment", "numerics"):
            source = self.config_dir / f"{namespace}.json"
            try:
                with source.open(encoding="utf-8") as stream:
                    value = json.load(stream, object_pairs_hook=unique_keys)
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigurationError(f"Cannot read {source}: {exc}") from exc
            if not isinstance(value, dict):
                raise ConfigurationError(f"{source} must contain a JSON object")
            _finite_json(value, namespace)
            self.data[namespace] = value
        self._validate_known_inputs()

    def get(self, dotted):
        value = self.data
        for name in dotted.split("."):
            if not isinstance(value, dict) or name not in value:
                raise ConfigurationError(f"Missing configuration key: {dotted}")
            value = value[name]
        return value

    def require(self, dotted):
        value = self.get(dotted)
        if value is None:
            raise ConfigurationError(
                f"Set {dotted} in {self.config_dir}; its current value is null"
            )
        return value

    def number(self, dotted):
        value = self.require(dotted)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"{dotted} must be a finite number")
        if not np.isfinite(value):
            raise ConfigurationError(f"{dotted} must be finite")
        return float(value)

    def positive(self, dotted):
        value = self.number(dotted)
        if value <= 0:
            raise ConfigurationError(f"{dotted} must be positive")
        return value

    def nonnegative(self, dotted):
        value = self.number(dotted)
        if value < 0:
            raise ConfigurationError(f"{dotted} must be nonnegative")
        return value

    def integer(self, dotted, *, positive=True):
        value = self.number(dotted)
        if not value.is_integer() or (positive and value <= 0):
            requirement = "positive integer" if positive else "integer"
            raise ConfigurationError(f"{dotted} must be a {requirement}")
        return int(value)

    def fraction(self, dotted):
        value = self.number(dotted)
        if not 0 <= value <= 1:
            raise ConfigurationError(f"{dotted} must lie between 0 and 1")
        return value

    def _validate_known_inputs(self):
        isotope = self.require("atomic.isotope")
        if not isinstance(isotope, str) or not isotope.strip():
            raise ConfigurationError("atomic.isotope must be a nonempty isotope identifier")
        if self.ground_F < 0 or not (2 * self.ground_F).is_integer():
            raise ConfigurationError("Initial F must be a nonnegative half-integer")
        if abs(self.initial_m) > self.ground_F:
            raise ConfigurationError("Initial mF must be within the configured F manifold")
        if not (self.initial_m + self.ground_F).is_integer():
            raise ConfigurationError("Initial mF must be an allowed magnetic sublevel")
        self.q
        self.polarization_fractions
        self.positive("atomic.linewidth_MHz")
        self.positive("numerics.validation.parameter_rtol")
        self.positive("numerics.validation.probability_atol")
        self.positive("numerics.validation.matrix_atol")
        self.positive("experiment.timing.strobe_frequency_MHz")
        self.fraction("experiment.timing.trap_on_fraction")
        self.integer("experiment.probe.beam_count")
        self.nonnegative("experiment.probe.reported_s0_per_beam")
        for path in (
            "atomic.mu_B_over_h_MHz_per_G",
            "atomic.D2_wavelength_nm",
            "atomic.mass_kg",
            "atomic.planck_J_s",
            "atomic.boltzmann_J_per_K",
            "experiment.trap.wavelength_nm",
            "experiment.trap.waist_um",
            "experiment.trap.depth_over_kB_during_state_readout_mK",
            "experiment.probe.diameter_1_over_e2_mm",
        ):
            if self.get(path) is not None:
                self.positive(path)
        for path in (
            "experiment.atom.initial_temperature_uK",
            "experiment.probe.power_per_beam_uW",
            "experiment.probe.reported_intensity_per_beam_mW_per_cm2",
            "experiment.probe.absolute_frequency_difference_kHz",
        ):
            if self.get(path) is not None:
                self.nonnegative(path)
        for path in ("atomic.ground_lande_g", "atomic.excited_lande_g"):
            if self.get(path) is not None:
                self.number(path)
        for path in ("atomic.nuclear_spin", "atomic.ground_J", "atomic.excited_J"):
            if self.get(path) is not None:
                value = self.nonnegative(path)
                if not (2 * value).is_integer():
                    raise ConfigurationError(f"{path} must be a nonnegative half-integer")
        for path in (
            "experiment.timing.probe_on_fraction",
            "experiment.timing.probe_start_fraction",
            "experiment.fluorescence_collection.actual_efficiency_fraction",
            "experiment.fluorescence_collection.theoretical_efficiency_fraction",
        ):
            if self.get(path) is not None:
                self.fraction(path)
        if all(self.get(path) is not None for path in (
            "experiment.timing.probe_on_fraction",
            "experiment.timing.probe_start_fraction",
        )):
            self._dark_windows()
        if self.get("experiment.timing.readout_cycles") is not None:
            self.cycles
        for path in (
            "experiment.timing.reported_probe_on_fraction_range",
            "experiment.probe.reported_detuning_range_MHz",
        ):
            values = self.get(path)
            if values is None:
                continue
            if not isinstance(values, list) or len(values) != 2:
                raise ConfigurationError(f"{path} must contain two ordered bounds")
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in values):
                raise ConfigurationError(f"{path} bounds must be numeric")
            if values[0] > values[1]:
                raise ConfigurationError(f"{path} bounds must be ordered")
            if path == "experiment.timing.reported_probe_on_fraction_range" and any(
                not 0 <= value <= 1 for value in values
            ):
                raise ConfigurationError(f"{path} bounds must lie between 0 and 1")

    @property
    def ground_F(self):
        return self.number("experiment.atom.initial_ground_state.F")

    @property
    def initial_m(self):
        return self.number("experiment.atom.initial_ground_state.mF")

    @property
    def cycling_ground_m(self):
        """Ground sublevel defining the laser detuning, independent of preparation."""
        return self.q * self.ground_F

    @property
    def cycling_excited_m(self):
        return self.cycling_ground_m + self.q

    @property
    def excited_F(self):
        levels = self.require("atomic.excited_hyperfine_offsets_MHz")
        if not isinstance(levels, dict) or not levels:
            raise ConfigurationError("atomic.excited_hyperfine_offsets_MHz must be a nonempty object")
        try:
            labels = [float(value) for value in levels]
        except ValueError as exc:
            raise ConfigurationError("Excited hyperfine labels must be numeric") from exc
        if any(not np.isfinite(value) or value < 0 or not (2 * value).is_integer() for value in labels):
            raise ConfigurationError("Excited hyperfine labels must be finite nonnegative half-integers")
        return max(labels)

    @property
    def q(self):
        polarization = self.require("experiment.probe.polarization")
        if polarization == "sigma_minus":
            return -1
        if polarization == "sigma_plus":
            return 1
        raise ConfigurationError("experiment.probe.polarization must be sigma_minus or sigma_plus")

    @property
    def polarization_fractions(self):
        opposite = self.fraction("experiment.probe.opposite_polarization_fraction")
        pi = self.fraction("experiment.probe.pi_polarization_fraction")
        if opposite + pi > 1:
            raise ConfigurationError("Polarization impurity fractions must sum to at most one")
        return {self.q: 1 - opposite - pi, -self.q: opposite, 0: pi}

    @property
    def linewidth_MHz(self):
        return self.positive("atomic.linewidth_MHz")

    @property
    def gamma(self):
        return 2 * np.pi * self.linewidth_MHz

    @property
    def mu(self):
        return self.number("atomic.mu_B_over_h_MHz_per_G")

    @property
    def ground_g(self):
        return self.number("atomic.ground_lande_g")

    @property
    def excited_g(self):
        return self.number("atomic.excited_lande_g")

    @property
    def beam_saturation(self):
        return self.number("experiment.probe.reported_s0_per_beam")

    @property
    def saturation(self):
        return self.integer("experiment.probe.beam_count") * self.beam_saturation

    @property
    def period_us(self):
        return 1 / self.positive("experiment.timing.strobe_frequency_MHz")

    @property
    def probe_us(self):
        return self.period_us * self.fraction("experiment.timing.probe_on_fraction")

    @property
    def trap_us(self):
        return self.period_us * self.fraction("experiment.timing.trap_on_fraction")

    def _dark_windows(self):
        start = self.fraction("experiment.timing.probe_start_fraction")
        duration = self.fraction("experiment.timing.probe_on_fraction")
        trap = self.fraction("experiment.timing.trap_on_fraction")
        tolerance = self.positive("numerics.validation.parameter_rtol")
        if start < trap - tolerance or start + duration > 1 + tolerance:
            raise ConfigurationError(
                "Probe must begin after the trap window and end within the same strobe period"
            )
        return max(0., start - trap) * self.period_us, max(0., 1 - start - duration) * self.period_us

    @property
    def dark_before_probe_us(self):
        return self._dark_windows()[0]

    @property
    def dark_after_probe_us(self):
        return self._dark_windows()[1]

    @property
    def cycles(self):
        return self.integer("experiment.timing.readout_cycles")

    @property
    def total_on_us(self):
        return self.cycles * self.probe_us

    @property
    def trap_field_vector(self):
        path = "experiment.trap.fictitious_field_vector_G"
        value = self.require(path)
        if not isinstance(value, list) or len(value) != 3:
            raise ConfigurationError(f"{path} must be [Bx, By, Bz] in G")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
            raise ConfigurationError(f"{path} components must be finite numbers")
        vector = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(vector)):
            raise ConfigurationError(f"{path} components must be finite numbers")
        return vector

    def field_g(self, voltage):
        voltage = np.asarray(voltage, dtype=float)
        if not np.all(np.isfinite(voltage)):
            raise ConfigurationError("Control voltage must be finite")
        reference_voltage = self.number("experiment.quantization_field.calibration_point.voltage_V")
        reference_field = self.number("experiment.quantization_field.calibration_point.field_G_approx")
        if np.all(voltage == reference_voltage):
            result = np.full_like(voltage, reference_field)
        else:
            slope = self.number("experiment.quantization_field.slope_G_per_V")
            if slope == 0:
                raise ConfigurationError("experiment.quantization_field.slope_G_per_V must be nonzero")
            result = reference_field + slope * (voltage - reference_voltage)
        return float(result) if result.ndim == 0 else result

    def detuning_mhz(self, b):
        field = np.asarray(b, dtype=float)
        if not np.all(np.isfinite(field)):
            raise ConfigurationError("Magnetic field must be finite")
        detuning = self.number("experiment.probe.detuning_from_cycling_transition_MHz")
        reference = self.number("experiment.probe.detuning_reference_field_G")
        if abs(self.cycling_excited_m) > self.excited_F:
            raise ConfigurationError("Configured polarization does not address a valid excited sublevel")
        shift = self.mu * (
            self.excited_g * self.cycling_excited_m - self.ground_g * self.cycling_ground_m
        )
        result = detuning + shift * (reference - field)
        return float(result) if result.ndim == 0 else result

    def scan_voltages(self):
        start = self.number("experiment.scan.start_voltage_V")
        stop = self.number("experiment.scan.stop_voltage_V")
        step = self.number("experiment.scan.step_voltage_V")
        if step == 0 or (stop - start) * step < 0:
            raise ConfigurationError("Scan step must be nonzero and point from start toward stop")
        count = (stop - start) / step
        tolerance = self.positive("numerics.validation.parameter_rtol")
        if not np.isclose(count, round(count), rtol=tolerance, atol=tolerance):
            raise ConfigurationError("Scan interval must be an integer multiple of step_voltage_V")
        return np.linspace(start, stop, int(round(count)) + 1)

    @property
    def recoil_temperature_mK(self):
        wavelength_m = self.positive("atomic.D2_wavelength_nm") * 1e-9
        h = self.positive("atomic.planck_J_s")
        mass = self.positive("atomic.mass_kg")
        kb = self.positive("atomic.boltzmann_J_per_K")
        return h**2 / (2 * mass * wavelength_m**2 * kb) * 1e3

    @property
    def recoil_angular_per_s(self):
        energy = self.recoil_temperature_mK * 1e-3 * self.positive("atomic.boltzmann_J_per_K")
        return 2 * np.pi * energy / self.positive("atomic.planck_J_s")

    def missing(self):
        result = []
        def collect(value, prefix):
            if isinstance(value, dict):
                for key, item in value.items():
                    collect(item, f"{prefix}.{key}" if prefix else key)
            elif value is None:
                result.append(prefix)
        collect(self.data, "")
        return result

    def summary(self):
        probe_fraction_range = self.get("experiment.timing.reported_probe_on_fraction_range")
        result = {
            "config_directory": str(self.config_dir),
            "initial_state": {"F": self.ground_F, "mF": self.initial_m},
            "polarization": self.require("experiment.probe.polarization"),
            "period_us": self.period_us,
            "trap_on_us": self.trap_us,
            "reported_probe_on_us_range": None if probe_fraction_range is None else [
                self.period_us * fraction
                for fraction in probe_fraction_range
            ],
            "total_saturation": self.saturation,
            "missing_inputs": self.missing(),
        }
        for name in ("probe_us", "dark_before_probe_us", "dark_after_probe_us", "cycles"):
            try:
                result[name] = getattr(self, name)
            except ConfigurationError as exc:
                result[name] = {"unavailable": str(exc)}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate known settings and list missing inputs")
    args = parser.parse_args()
    try:
        print(json.dumps(Parameters(args.config_dir).summary(), ensure_ascii=False, indent=2))
    except ConfigurationError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
