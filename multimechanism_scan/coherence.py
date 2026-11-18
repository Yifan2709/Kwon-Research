"""Near-stretched 87Rb D2 probe-pulse model; see function docstrings.

Run python coherence.py [--csv output.csv]. Requires NumPy and SciPy.
Time is in microseconds; Hamiltonians are in rad/us with hbar=1.
Outputs ground-state coherence normalized to its value at light turn-on.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

LINEWIDTH_MHZ = 6.065
GAMMA = 2 * np.pi * LINEWIDTH_MHZ
MU_MHZ_G = 1.3996246
GROUND_G, EXCITED_G = 0.5, 2 / 3
B_EXT_G, B_FICT_G, ANGLE = 20., .003, np.deg2rad(60)
PULSE_US = .4
SATURATION, LASER_BEAT_MHZ = 1., .500
# detuning from |g,2> -> |e,3>, not from a zero-field frequency
DETUNING_MHZ = -LINEWIDTH_MHZ / 2
TIMES = np.array([0, .05, .1, .2, .3, .4])
G1, G2, E2, E3 = range(4)


def element(row, col):
    op = np.zeros((4, 4), dtype=complex)
    op[row, col] = 1
    return op


def pulse(phase=None, pure_stretched=False):
    """Use effective uniform intensity (phase=None), or two equal coherent beams.

    States [g1,g2,e2,e3] span the optical dynamics for a starting state
    in g1,g2 with perfect sigma+ light. The full first trap interval would
    also yield tiny amplitudes of m<=0, omitted at second order in tilt.
    Each photon polarization has one jump; common sigma photons can transfer
    coherence between the two branches. Their frequency splitting is
    retained in the Hamiltonian (no excited-Zeeman secularization).
    """
    zg = GROUND_G * MU_MHZ_G * B_EXT_G
    ze = EXCITED_G * MU_MHZ_G * B_EXT_G
    detuning_g1 = DETUNING_MHZ + ze - zg
    h0 = 2*np.pi*np.diag([0., zg, -detuning_g1, zg-DETUNING_MHZ])
    drive = element(E3, G2) + np.sqrt(2/3)*element(E2, G1)
    sigma = np.sqrt(GAMMA)*drive.conj().T
    pi = np.sqrt(GAMMA/3)*element(G2, E2)
    loss = sigma.conj().T @ sigma + pi.conj().T @ pi
    omega = GAMMA*np.sqrt(SATURATION/2)
    b_perp = B_FICT_G*np.sin(ANGLE)
    b_total = np.hypot(B_EXT_G+B_FICT_G*np.cos(ANGLE), b_perp)
    mix = (b_perp/b_total*np.sin(np.pi*GROUND_G*MU_MHZ_G*b_total*PULSE_US))**2
    p1 = 0 if pure_stretched else 4*mix*(1-mix)**3
    p2 = 1 if pure_stretched else (1-mix)**4
    # The observable magnitude is insensitive to a common initial
    # adjacent-coherence phase. Normalize after truncating m<=0.
    state = np.sqrt([p1,p2,0.,0.])/np.sqrt(p1+p2)
    rho0 = np.outer(state, state)
    y0 = np.r_[rho0.ravel(),0j]
    def rhs(t, y):
        if phase is None:
            coef = omega/2
        else:
            phase1 = np.pi*LASER_BEAT_MHZ*t
            coef = omega/(2*np.sqrt(2))*(np.exp(1j*phase1) +
                                            np.exp(1j*(phase-phase1)))
        h = h0+coef*drive+np.conj(coef)*drive.conj().T
        rho = y[:-1].reshape(4,4)
        dr = -1j*(h@rho-rho@h) - (loss@rho+rho@loss)/2
        dr += sigma@rho@sigma.conj().T + pi@rho@pi.conj().T
        return np.r_[dr.ravel(), np.trace(loss@rho)]
    sol = solve_ivp(rhs, (0,PULSE_US), y0, t_eval=TIMES, method="DOP853",
                    rtol=2e-10, atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    rho = sol.y[:-1].T.reshape(-1,4,4)
    assert np.max(np.abs(np.trace(rho,axis1=1,axis2=2)-1)) < 1e-9
    assert np.min(np.linalg.eigvalsh(rho)) > -1e-10
    return rho, sol.y[-1].real


def records(nphase=16):
    pulses = [pulse(a) for a in 2*np.pi*np.arange(nphase)/nphase]
    models = {"effective_uniform": pulse(),
              "equal_beams_uniform_phase": tuple(np.mean(x,axis=0) for x in zip(*pulses))}
    rows = []
    for name, (rho, photons) in models.items():
        norm = abs(rho[0,G1,G2])
        for t, dm, mean in zip(TIMES, rho, photons):
            pg = np.real(dm[G1,G1]+dm[G2,G2])
            pe = np.real(dm[E2,E2]+dm[E3,E3])
            rows.append(dict(model=name,t_us=t,ground_population=pg,excited_population=pe,
                             ground_m1=float(dm[G1,G1].real),ground_m2=float(dm[G2,G2].real),
                             excited_m2=float(dm[E2,E2].real),excited_m3=float(dm[E3,E3].real),
                             abs_ground_rho12=abs(dm[G1,G2]),ground_coherence_ratio=abs(dm[G1,G2])/norm,
                             abs_excited_rho23=abs(dm[E2,E3]),mean_emissions=mean))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--phase-points", type=int, default=16)
    args = parser.parse_args()
    rows = records(args.phase_points)
    for row in rows:
        if row["t_us"] in [0, PULSE_US]:
            print(row)
    if args.csv:
        with args.csv.open("w",newline="") as fp:
            w = csv.DictWriter(fp,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
