"""Conditional rate-equation diagnostic for 87Rb (Kwon et al., PRL 119, 180504).

Run: python depumping.py [output_directory]. Requires numpy, scipy, sympy.
Main text and supplement VIII–X, Eqs. 11–13; Steck, Rubidium 87 D Line Data.
Times in us, detunings in MHz, fields in G. Lab probe axis fixed, positive b
along this axis. All incident circular intensity is sigma+ except 1/1601
sigma-. All spontaneous photon polarizations and final sublevels are summed.
Branches to F=1 are absorbing. This is a population model of internal dynamics
ONLY. It assumes zero optical/field-axis angular misalignment, instantaneous
line-specific steady-state scattering in each probe pulse (Kwon Eq.12),
weak-field Zeeman levels, separate excited hyperfine levels, equal
transition prefactors within D2 multiplets and independent beam intensities.
It ignores saturation competition between lines, finite optical transients,
the 500-kHz relative beam offset, atomic motion/escape, other technical errors,
state-dependent light shifts while probing, cross-pulse Zeeman coherences,
and off-resonant excitation out of F=1. The incoherent sum of F' paths gives
the total in a final F multiplet when all m and polarizations are summed, by
CG orthogonality. Individual resolved F=2,m scattering branches here use
Kwon's sequential excitation and branching approximation. No observed curve
is fitted. Scan field calibration and reference frequency are hypothetical.
"""
import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_6j

from scan_axis import ASSUMED_SLOPE_G_PER_V, assumed_field_G

GAMMA_MHZ = 6.065
RATE_US = 2 * np.pi * GAMMA_MHZ
MU_B_MHZ_G = 1.3996
GROUND_GF = 0.5
EXCITED_GF = 2 / 3
FREQ_SHIFT = {1: -423.597, 2: -266.650, 3: 0.0}
IMPURITY = 1 / 1601
TRAP_FIELD_G = 0.003
ANGLE = np.pi / 3
HALF_US = 0.4
CYCLES = 7500
M = list(range(-2, 3))
DARK = 5
GRID_SLOPE = ASSUMED_SLOPE_G_PER_V  # legacy name, in G/V


@lru_cache(maxsize=None)
def strength(f, m, q, fp):
    """Squared dipole element relative to stretched D2 cycling line."""
    mp = m + q
    if abs(mp) > fp:
        return 0.
    j, je, spin = Rational(1, 2), Rational(3, 2), Rational(3, 2)
    coef = clebsch_gordan(f, 1, fp, m, q, mp)
    six = wigner_6j(j, spin, f, fp, 1, je)
    return float((2 * je + 1) * (2 * f + 1) * six**2 * coef**2)


@lru_cache(maxsize=None)
def channels(impurity):
    paths = []
    for state, m in enumerate(M):
        for q, fraction in [(1, 1 - impurity), (-1, impurity)]:
            for fp in (1, 2, 3):
                s = fraction * strength(2, m, q, fp)
                if s == 0:
                    continue
                mp = m + q
                emission = np.zeros(6)
                for f in (1, 2):
                    for qm in (-1, 0, 1):
                        mf = mp - qm
                        if abs(mf) <= f:
                            dest = DARK if f == 1 else mf + 2
                            emission[dest] += strength(f, mf, qm, fp)
                assert np.isclose(sum(emission), 1)
                paths.append((state, m, fp, mp, s, emission))
    return paths


def probe_generator(b, det_zero, impurity=IMPURITY):
    qmat = np.zeros((6, 6))
    for state, m, fp, mp, s, emission in channels(impurity):
        det = det_zero - FREQ_SHIFT[fp] - MU_B_MHZ_G*b*(EXCITED_GF*mp-GROUND_GF*m)
        rate = RATE_US / 2 * s / (1 + s + (2 * det / GAMMA_MHZ)**2)
        qmat[:, state] += rate * emission
        qmat[state, state] -= rate
    assert np.max(abs(qmat.sum(axis=0))) < 2e-12
    assert (qmat - np.diag(np.diag(qmat))).min() >= -1e-13
    return qmat


def trap_map(b):
    """Unitary trap evolution followed by complete Zeeman dephasing."""
    raising = np.zeros((5, 5))
    for j, m in enumerate(M[:-1]):
        raising[j + 1, j] = np.sqrt(6 - m*(m + 1))
    bx = TRAP_FIELD_G * np.sin(ANGLE)
    bz = b + TRAP_FIELD_G * np.cos(ANGLE)
    h_us = (2*np.pi*0.6998) * (bx*(raising+raising.T)/2 + bz*np.diag(M))
    transition = np.eye(6)
    transition[:5, :5] = abs(expm(-1j * HALF_US * h_us))**2
    return transition


def estimate(b, det_zero):
    qmat = probe_generator(b, det_zero)
    # First arrival to either absorbing boundary m=2 or F=1; integrated
    # occupation time of transients before absorption, starting with m=+1.
    transients = [0, 1, 2, 3]
    start = np.array([0., 0., 0., 1.])
    dwell = np.linalg.solve(-qmat[np.ix_(transients, transients)], start)
    first = qmat[DARK, transients] @ dwell
    probe = expm(HALF_US * qmat)
    no_trap = expm((HALF_US * CYCLES) * qmat)
    with_trap = np.linalg.matrix_power(probe @ trap_map(b), CYCLES)
    assert np.allclose(with_trap.sum(axis=0), 1, atol=1e-8)
    return dict(
        b_assumed_G=b, assumed_laser_detuning_from_zero_field_MHz=det_zero,
        r_to_F1_if_m2_held_per_s=1.e6*qmat[DARK, 4],
        r_to_F1_if_m1_held_per_s=1.e6*qmat[DARK, 3],
        p_F1_before_return_m2_given_m1=first,
        p_F1_6ms_from_m2_without_trap_mixing=no_trap[DARK, 4],
        p_F1_6ms_from_m2_with_dephased_trap=with_trap[DARK, 4],
        p_F1_6ms_from_uniform_without_trap_mixing=no_trap[DARK, :5].mean(),
        p_F1_6ms_from_uniform_with_dephased_trap=with_trap[DARK, :5].mean(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', type=Path, nargs='?', default=Path(__file__).resolve().parent)
    dest = parser.parse_args().outdir
    dest.mkdir(parents=True, exist_ok=True)
    assert np.isclose(strength(2, 2, 1, 3), 1)
    assert np.isclose(strength(2, 1, 1, 2), 1 / 6)
    assert np.allclose(probe_generator(20., 24.96, impurity=0.)[:, 4], 0.)
    values = [dict(case='scan', x=x, **estimate(assumed_field_G(x), -2.),
                   quantization_voltage_V=x) for x in np.linspace(-8., 4., 1201)]
    b = 20.
    det_zero = -GAMMA_MHZ / 2 + MU_B_MHZ_G * b
    values.append(dict(case='published_20G_probe_nominal', x='', **estimate(b, det_zero),
                       quantization_voltage_V=''))
    with (dest / 'depumping.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


if __name__ == '__main__':
    main()
