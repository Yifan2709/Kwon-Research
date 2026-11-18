"""One calibrated voltage-to-field conversion for all model entry points."""
from .parameters import Parameters

VOLTAGE_LABEL = "Quantization-field control voltage (V)"


def model_field_g(voltage_v, config=None):
    """Read the measured anchor and slope from the shared configuration."""
    return (config if config is not None else Parameters()).field_g(voltage_v)


# Legacy name remains importable, but no longer supplies an assumed calibration.
assumed_field_G = model_field_g
