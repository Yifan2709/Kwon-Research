"""Angular-momentum algebra shared by all fluorescence calculations."""

import numpy as np
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_6j

from .parameters import ConfigurationError


def _angular(value):
    return Rational(str(value))


def angular_momentum(f):
    """Dimensionless Fx, Fy and Fz in increasing magnetic-sublevel order."""
    if f < 0 or not float(2 * f).is_integer():
        raise ConfigurationError("Angular momentum must be a nonnegative half-integer")
    m = np.arange(-f, f + 1)
    raising = np.diag(np.sqrt((f - m[:-1]) * (f + m[:-1] + 1)), k=-1)
    return (raising + raising.T) / 2, (raising - raising.T) / (2j), np.diag(m)


def strength(cfg, f, m, q, fp):
    """Squared D2 dipole matrix element relative to the stretched cycling line."""
    mp = m + q
    if abs(m) > f or abs(mp) > fp or abs(q) > 1:
        return 0.
    j = _angular(cfg.number("atomic.ground_J"))
    je = _angular(cfg.number("atomic.excited_J"))
    spin = _angular(cfg.number("atomic.nuclear_spin"))
    f, fp, m, mp, q = map(_angular, (f, fp, m, mp, q))
    cg = clebsch_gordan(f, 1, fp, m, q, mp)
    if cg == 0:
        return 0.
    six = wigner_6j(j, spin, f, fp, 1, je)
    return float((2 * je + 1) * (2 * f + 1) * six**2 * cg**2)


def emission_matrix(cfg, q):
    """Lowering operator on the closed ground-F to maximal excited-F manifold."""
    ground_f, excited_f = cfg.ground_F, cfg.excited_F
    ground_m = np.arange(-ground_f, ground_f + 1)
    excited_m = np.arange(-excited_f, excited_f + 1)
    size = len(ground_m) + len(excited_m)
    operator = np.zeros((size, size), dtype=complex)
    for index, m in enumerate(ground_m):
        mp = m + q
        if abs(mp) <= excited_f:
            excited_index = int(mp + excited_f)
            coefficient = clebsch_gordan(
                _angular(ground_f), 1, _angular(excited_f),
                _angular(m), _angular(q), _angular(mp)
            )
            operator[index, len(ground_m) + excited_index] = float(coefficient)
    return operator


def validate(cfg):
    """Verify normalization of the configured closed hyperfine transition."""
    spin = cfg.number("atomic.nuclear_spin")
    ground_j = cfg.number("atomic.ground_J")
    excited_j = cfg.number("atomic.excited_J")
    if cfg.ground_F != spin + ground_j or cfg.excited_F != spin + excited_j or cfg.excited_F != cfg.ground_F + 1:
        raise ConfigurationError("Closed cycling model requires maximal F and F'=F+1 manifolds")
    ground_dim = int(2 * cfg.ground_F + 1)
    excited_dim = int(2 * cfg.excited_F + 1)
    projector = np.diag(np.r_[np.zeros(ground_dim), np.ones(excited_dim)])
    operators = [emission_matrix(cfg, q) for q in (-1, 0, 1)]
    total = sum(operator.conj().T @ operator for operator in operators)
    tolerance = cfg.number("numerics.validation.matrix_atol")
    if not np.allclose(total, projector, atol=tolerance, rtol=0):
        raise ConfigurationError("Configured manifolds do not form a closed cycling transition")
    return True
