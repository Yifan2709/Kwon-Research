"""Run the 12-state control model at selected quantization-field control voltages."""
import csv
from pathlib import Path

from coherence_scan import propagate
from voltage_axis import SCENARIO_SLOPE_G_PER_V, SCENARIO_ZERO_V, model_field_g

out = Path(__file__).parent
voltages_v = [-8, -4, -3, -2, -1.5, -.8, 0, .5, 1.3, 2.2, 3, 4]
with (out / "kwon_control.csv").open("w", newline="") as f:
    for i, voltage_v in enumerate(voltages_v):
        b_g = model_field_g(voltage_v, SCENARIO_SLOPE_G_PER_V, SCENARIO_ZERO_V)
        result = propagate(b_g, -2 - 1.3996246 * b_g, count_events=True)
        row = dict(x=voltage_v, **result, scan_voltage_V=voltage_v)
        if i == 0:
            writer = csv.DictWriter(f, fieldnames=row)
            writer.writeheader()
        writer.writerow(row)
        if voltage_v in [-3, 2.2]:
            print(voltage_v, row["prob_emitted_at_least_one_pi_or_sigma_minus"])
