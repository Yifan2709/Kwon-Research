"""Current experimental settings and the historical voltage-conversion helpers."""

import json
from pathlib import Path


with Path(__file__).with_name("experiment_parameters.json").open(encoding="utf-8") as source:
    EXPERIMENT = json.load(source)

# Numerical conversions below reproduce the historical benchmark only.
# Replace these pairs with independently measured values when available.
SCENARIO_SLOPE_G_PER_V = 1.0
SCENARIO_ZERO_V = 0.65
# Standalone coherence scan uses the original separate period-based hypothesis.
COHERENCE_SLOPE_G_PER_V = 2.38163
COHERENCE_ZERO_V = 0.0


def model_field_g(voltage_v, slope_g_per_v, zero_v):
    """Signed longitudinal field (G) at control voltage (V)."""
    return slope_g_per_v * (voltage_v - zero_v)
