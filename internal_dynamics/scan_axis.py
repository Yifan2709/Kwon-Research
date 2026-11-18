"""Current experimental settings and the historical voltage-conversion helpers."""

import json
from pathlib import Path


with Path(__file__).with_name("experiment_parameters.json").open(encoding="utf-8") as source:
    EXPERIMENT = json.load(source)

# Numerical conversions below reproduce the historical benchmark only.
VOLTAGE_LABEL = "量子化轴磁场控制电压（V）"

# Example conversion inferred from a putative 1.5 V period, without an independent
# field calibration. The benchmark uses 0.6998 MHz/G and a 0.4 us trap pulse.
ASSUMED_SLOPE_G_PER_V = 1 / (0.6998 * 0.4 * 1.5)
ASSUMED_ZERO_FIELD_V = 0.0


def assumed_field_G(voltage_V, slope_G_per_V=ASSUMED_SLOPE_G_PER_V):
    return (voltage_V - ASSUMED_ZERO_FIELD_V) * slope_G_per_V
