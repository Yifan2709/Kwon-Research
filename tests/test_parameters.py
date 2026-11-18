"""Regression checks for one file-backed parameter source and model propagation."""

import copy
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kwon.parameters import ParameterError, Parameters

FIXTURE = json.loads(Path(__file__).with_name("fixture.json").read_text())


def merge(target, changes):
    for key, value in changes.items():
        if isinstance(value, dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


class ConfigurationTests(unittest.TestCase):
    def make_config(self, changes=None, complete=True):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        data = {
            name: json.loads((ROOT / "config" / f"{name}.json").read_text())
            for name in ("atomic", "experiment", "numerics")
        }
        if complete:
            merge(data, FIXTURE["parameters"])
        if changes:
            merge(data, changes)
        for name, document in data.items():
            (path / f"{name}.json").write_text(json.dumps(document))
        return path

    def test_current_configuration_preserves_negative_stretched_state(self):
        cfg = Parameters()
        self.assertEqual(cfg.initial_m, -cfg.ground_F)
        self.assertEqual(cfg.q, -1)
        self.assertEqual(cfg.saturation, cfg.beam_saturation * cfg.number("experiment.probe.beam_count"))

    def test_missing_physical_inputs_do_not_have_fallback_values(self):
        cfg = Parameters()
        for name in ("probe_us", "cycles", "trap_field_vector"):
            with self.subTest(name=name), self.assertRaises(ParameterError):
                getattr(cfg, name)
        with self.assertRaises(ParameterError):
            cfg.scan_voltages()

    def test_duplicate_json_key_is_rejected(self):
        directory = self.make_config()
        file = directory / "atomic.json"
        text = file.read_text()
        key = "linewidth_MHz"
        duplicate = f'"{key}": {FIXTURE["mutations"][key]}, '
        file.write_text(text.replace("{", "{" + duplicate, 1))
        with self.assertRaises(ParameterError):
            Parameters(directory)

    def test_nonfinite_and_boolean_numbers_are_rejected(self):
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ParameterError):
                cfg = Parameters(self.make_config({"experiment": {"timing": {"strobe_frequency_MHz": value}}}))
                _ = cfg.period_us

    def test_zero_frequency_is_rejected(self):
        with self.assertRaises(ParameterError):
            cfg = Parameters(self.make_config({"experiment": {"timing": {"strobe_frequency_MHz": 0}}}))
            _ = cfg.period_us

    def test_invalid_state_and_polarization_fractions_are_rejected(self):
        changes = [
            {"experiment": {"atom": {"initial_ground_state": {"mF": FIXTURE["mutations"]["invalid_initial_mF"]}}}},
            {"experiment": {"probe": {
                "opposite_polarization_fraction": FIXTURE["mutations"]["inconsistent_polarization_fraction"],
                "pi_polarization_fraction": FIXTURE["mutations"]["inconsistent_polarization_fraction"],
            }}},
            {"experiment": {"timing": {"readout_cycles": 0}}},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ParameterError):
                Parameters(self.make_config(change))

    def test_missing_parameter_file_or_key_is_rejected(self):
        directory = self.make_config()
        (directory / "atomic.json").unlink()
        with self.assertRaises(ParameterError):
            Parameters(directory)
        cfg = Parameters(self.make_config())
        with self.assertRaises(ParameterError):
            cfg.require("experiment.probe.undefined_test_input")

    def test_timing_segments_sum_to_period(self):
        cfg = Parameters(self.make_config())
        total = cfg.trap_us + cfg.dark_before_probe_us + cfg.probe_us + cfg.dark_after_probe_us
        self.assertAlmostEqual(total, cfg.period_us)
        self.assertAlmostEqual(cfg.probe_us / cfg.period_us, cfg.number("experiment.timing.probe_on_fraction"))
        self.assertAlmostEqual(cfg.total_on_us, cfg.cycles * cfg.probe_us)

    def test_frequency_edit_changes_all_derived_windows(self):
        original = Parameters(self.make_config())
        frequency = FIXTURE["mutations"]["strobe_frequency_MHz"]
        changed = Parameters(self.make_config({"experiment": {"timing": {"strobe_frequency_MHz": frequency}}}))
        ratio = original.number("experiment.timing.strobe_frequency_MHz") / frequency
        for name in ("period_us", "trap_us", "probe_us", "dark_before_probe_us", "dark_after_probe_us"):
            with self.subTest(name=name):
                self.assertAlmostEqual(getattr(changed, name), getattr(original, name) * ratio)

    def test_overlapping_pulses_are_rejected(self):
        start = FIXTURE["mutations"]["overlapping_probe_start_fraction"]
        with self.assertRaises(ParameterError):
            cfg = Parameters(self.make_config({"experiment": {"timing": {"probe_start_fraction": start}}}))
            _ = cfg.dark_before_probe_us

    def test_anchor_point_is_available_without_inventing_a_slope(self):
        cfg = Parameters()
        anchor = cfg.get("experiment.quantization_field.calibration_point")
        self.assertAlmostEqual(cfg.field_g(anchor["voltage_V"]), anchor["field_G_approx"])
        offset = FIXTURE["parameters"]["experiment"]["scan"]["step_voltage_V"]
        with self.assertRaises(ParameterError):
            cfg.field_g(anchor["voltage_V"] + offset)

    def test_field_conversion_and_sigma_minus_detuning(self):
        cfg = Parameters(self.make_config())
        voltages = cfg.scan_voltages()
        anchor = cfg.get("experiment.quantization_field.calibration_point")
        expected = anchor["field_G_approx"] + cfg.number("experiment.quantization_field.slope_G_per_V") * (voltages - anchor["voltage_V"])
        np.testing.assert_allclose(cfg.field_g(voltages), expected)
        reference = cfg.number("experiment.probe.detuning_reference_field_G")
        detuning = cfg.number("experiment.probe.detuning_from_cycling_transition_MHz")
        self.assertAlmostEqual(cfg.detuning_mhz(reference), detuning)
        coefficient = cfg.mu * (cfg.excited_g * (cfg.initial_m + cfg.q) - cfg.ground_g * cfg.initial_m)
        np.testing.assert_allclose(cfg.detuning_mhz(expected), detuning + coefficient * (reference - expected))
        self.assertLess(coefficient, 0)

    def test_prepared_sublevel_does_not_change_physical_laser_reference(self):
        from kwon.depumping import probe_generator
        cfg = Parameters(self.make_config())
        changed = Parameters(self.make_config({"experiment": {"atom": {
            "initial_ground_state": {"mF": FIXTURE["mutations"]["initial_mF"]},
        }}}))
        self.assertNotEqual(cfg.initial_m, changed.initial_m)
        fields = cfg.field_g(cfg.scan_voltages())
        np.testing.assert_allclose(changed.detuning_mhz(fields), cfg.detuning_mhz(fields))
        for field in fields:
            with self.subTest(field=field):
                np.testing.assert_allclose(probe_generator(changed, field), probe_generator(cfg, field),
                                           atol=cfg.number("numerics.validation.matrix_atol"), rtol=0)

    def test_invalid_scan_step_is_rejected(self):
        step = FIXTURE["mutations"]["negative_step_voltage_V"]
        with self.assertRaises(ParameterError):
            cfg = Parameters(self.make_config({"experiment": {"scan": {"step_voltage_V": step}}}))
            cfg.scan_voltages()

    def test_missing_or_nonpositive_depth_blocks_survival(self):
        from kwon.scan import survival
        temperature = FIXTURE["parameters"]["numerics"]["phenomenological"]["base_temperature_mK"]
        for depth in (None, 0, FIXTURE["mutations"]["negative_trap_depth_mK"]):
            with self.subTest(depth=depth), self.assertRaises(ParameterError):
                cfg = Parameters(self.make_config({"experiment": {"trap": {
                    "depth_over_kB_during_state_readout_mK": depth,
                }}}))
                survival(cfg, temperature)

    def test_recoil_is_derived_from_atomic_constants(self):
        cfg = Parameters(self.make_config())
        h = cfg.number("atomic.planck_J_s")
        mass = cfg.number("atomic.mass_kg")
        wavelength = cfg.number("atomic.D2_wavelength_nm") * 1e-9
        boltzmann = cfg.number("atomic.boltzmann_J_per_K")
        energy = h**2 / (2 * mass * wavelength**2)
        self.assertAlmostEqual(cfg.recoil_temperature_mK, energy / boltzmann * 1e3)
        self.assertAlmostEqual(cfg.recoil_angular_per_s, energy * 2 * math.pi / h)

    def test_recoil_survival_approaches_one_for_weak_or_zero_light(self):
        from kwon.scan import recoil
        for intensity in (0, FIXTURE["mutations"]["weak_s0_per_beam"]):
            with self.subTest(intensity=intensity):
                cfg = Parameters(self.make_config({"experiment": {"probe": {
                    "reported_s0_per_beam": intensity,
                }}}))
                result = recoil(cfg, cfg.scan_voltages())
                np.testing.assert_allclose(result, 1, rtol=0,
                                           atol=cfg.number("numerics.validation.probability_atol"))

    def test_default_config_is_independent_of_working_directory(self):
        expected = Parameters().period_us
        previous = Path.cwd()
        directory = self.make_config()
        try:
            os.chdir(directory)
            self.assertEqual(Parameters().period_us, expected)
        finally:
            os.chdir(previous)

    def test_atomic_helpers_conserve_decay_probability(self):
        from kwon.atomic import emission_matrix, strength
        cfg = Parameters(self.make_config())
        matrices = [emission_matrix(cfg, q) for q in (-1, 0, 1)]
        total = sum(matrix.conj().T @ matrix for matrix in matrices)
        ground_size = int(2 * cfg.ground_F + 1)
        excited_size = int(2 * cfg.excited_F + 1)
        expected = np.diag([0] * ground_size + [1] * excited_size)
        np.testing.assert_allclose(total, expected, atol=cfg.number("numerics.validation.matrix_atol"))
        self.assertAlmostEqual(strength(cfg, cfg.ground_F, cfg.initial_m, cfg.q, cfg.excited_F), 1)

    def test_legacy_imports_do_not_run_calculations_or_write_outputs(self):
        modules = [
            ".".join(path.relative_to(ROOT).with_suffix("").parts)
            for directory in ("internal_dynamics", "multimechanism_scan")
            for path in sorted((ROOT / directory).glob("*.py"))
        ]
        script = "import importlib\nfor name in " + repr(modules) + ":\n    importlib.import_module(name)\n"
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=directory, env=environment,
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_legacy_pulse_entrypoints_share_model_and_follow_config(self):
        shared = importlib.import_module("kwon.coherence")
        wrappers = [importlib.import_module(name) for name in (
            "internal_dynamics.coherence", "multimechanism_scan.coherence",
        )]
        cfg = Parameters(self.make_config())
        changed = Parameters(self.make_config({"experiment": {"probe": {
            "reported_s0_per_beam": FIXTURE["mutations"]["reported_s0_per_beam"],
        }}}))
        for wrapper in wrappers:
            self.assertIs(wrapper.pulse, shared.pulse)
        rho, photons = wrappers[0].pulse(pure_stretched=True, config=cfg)
        _, changed_photons = wrappers[1].pulse(pure_stretched=True, config=changed)
        self.assertEqual(len(rho), cfg.number("numerics.integration.sample_count"))
        np.testing.assert_allclose(np.trace(rho, axis1=1, axis2=2), 1,
                                   atol=cfg.number("numerics.validation.probability_atol"))
        self.assertGreater(abs(photons[-1] - changed_photons[-1]),
                           cfg.number("numerics.validation.matrix_atol"))

    def test_depumping_maps_conserve_population_and_follow_intensity(self):
        from kwon.depumping import probe_generator, trap_map
        cfg = Parameters(self.make_config())
        changed = Parameters(self.make_config({"experiment": {"probe": {
            "reported_s0_per_beam": FIXTURE["mutations"]["reported_s0_per_beam"],
        }}}))
        field = cfg.number("experiment.quantization_field.calibration_point.field_G_approx")
        generator = probe_generator(cfg, field)
        altered = probe_generator(changed, field)
        tolerance = cfg.number("numerics.validation.matrix_atol")
        np.testing.assert_allclose(generator.sum(axis=0), 0, atol=tolerance)
        self.assertGreater(np.linalg.norm(generator - altered), tolerance)
        trap = trap_map(cfg, field)
        np.testing.assert_allclose(trap.sum(axis=0), 1, atol=tolerance)
        self.assertGreaterEqual(np.min(trap), -tolerance)
        sink = np.zeros(len(trap))
        sink[-1] = 1
        np.testing.assert_allclose(trap @ sink, sink, atol=tolerance)

    def test_atomic_linewidth_edit_reaches_shared_coherent_model(self):
        from kwon.coherence_scan import ClosedTransition
        cfg = Parameters(self.make_config())
        changed = Parameters(self.make_config({"atomic": {
            "linewidth_MHz": FIXTURE["mutations"]["linewidth_MHz"],
        }}))
        model, altered = ClosedTransition(cfg), ClosedTransition(changed)
        ratio = changed.linewidth_MHz / cfg.linewidth_MHz
        np.testing.assert_allclose(altered.loss, model.loss * ratio)
        np.testing.assert_allclose(altered.drive, model.drive * ratio)

    def test_single_pulse_requires_explicit_operating_voltage(self):
        from kwon.coherence import pulse
        cfg = Parameters(self.make_config({"experiment": {"quantization_field": {
            "single_pulse_voltage_V": None,
        }}}))
        with self.assertRaisesRegex(ParameterError, "single_pulse_voltage_V"):
            pulse(pure_stretched=True, config=cfg)

    def test_operating_voltage_edit_changes_single_pulse_result(self):
        from kwon.coherence import pulse
        cfg = Parameters(self.make_config())
        changed = Parameters(self.make_config({"experiment": {"quantization_field": {
            "single_pulse_voltage_V": FIXTURE["mutations"]["single_pulse_voltage_V"],
        }}}))
        _, photons = pulse(pure_stretched=True, config=cfg)
        _, altered = pulse(pure_stretched=True, config=changed)
        self.assertGreater(abs(photons[-1] - altered[-1]),
                           cfg.number("numerics.validation.matrix_atol"))

    def test_atomic_mu_and_field_slope_edits_reach_model_hamiltonian(self):
        from kwon.coherence_scan import ClosedTransition
        cfg = Parameters(self.make_config())
        voltage = cfg.scan_voltages()[0]
        field = cfg.field_g(voltage)
        model = ClosedTransition(cfg)
        original_h = model.base_hamiltonian(field, cfg.detuning_mhz(field))
        changes = [
            {"atomic": {"mu_B_over_h_MHz_per_G": FIXTURE["mutations"]["mu_B_over_h_MHz_per_G"]}},
            {"experiment": {"quantization_field": {"slope_G_per_V": FIXTURE["mutations"]["slope_G_per_V"]}}},
        ]
        for change in changes:
            with self.subTest(change=change):
                changed = Parameters(self.make_config(change))
                changed_field = changed.field_g(voltage)
                altered_h = ClosedTransition(changed).base_hamiltonian(
                    changed_field, changed.detuning_mhz(changed_field),
                )
                self.assertGreater(np.linalg.norm(altered_h - original_h),
                                   cfg.number("numerics.validation.matrix_atol"))

    def test_cycle_count_edit_reaches_legacy_scan_propagation(self):
        from kwon.coherence_scan import ClosedTransition, propagate
        cfg = Parameters(self.make_config())
        changed = Parameters(self.make_config({"experiment": {"timing": {
            "readout_cycles": FIXTURE["mutations"]["readout_cycles"],
        }}}))
        wrappers = [importlib.import_module(name) for name in (
            "internal_dynamics.coherence_scan", "multimechanism_scan.coherence_scan",
        )]
        for wrapper in wrappers:
            self.assertIs(wrapper.propagate, propagate)
        # An adjacent basis state makes noncycling emission appreciable over
        # the few synthetic periods, without changing any physical constant.
        model = ClosedTransition(cfg)
        initial = np.zeros_like(model.initial_rho)
        adjacent = model.ground_index(cfg.initial_m - cfg.q)
        initial[adjacent, adjacent] = 1
        field = cfg.number("experiment.quantization_field.calibration_point.field_G_approx")
        original = wrappers[0].propagate(field, cfg.detuning_mhz(field), start_rho=initial, config=cfg)
        altered = wrappers[1].propagate(field, changed.detuning_mhz(field), start_rho=initial, config=changed)
        self.assertEqual(original["readout_cycles"], cfg.cycles)
        self.assertEqual(altered["readout_cycles"], changed.cycles)
        self.assertAlmostEqual(altered["readout_duration_us"] / original["readout_duration_us"],
                               changed.cycles / cfg.cycles)
        key = "prob_emitted_at_least_one_noncycling_photon"
        self.assertGreater(altered[key], original[key])


if __name__ == "__main__":
    unittest.main()
