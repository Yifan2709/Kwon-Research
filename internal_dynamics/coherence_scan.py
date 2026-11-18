"""Conditional test of residual-coherence effects for the Kwon fluorescence scan.

Install numpy scipy; run OPENBLAS_NUM_THREADS=1 python coherence_scan.py --csv coherence_scan.csv
Use --count-all to compute cumulative non-cycling photon fractions at all x;
the default computes these only at the landmarks and the published point.
No dark-state, recoil, escape or camera readout is included. The calculated
F=2->F'=3 spin/optical dynamics cannot be interpreted as retention.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from coherence import (MU_MHZ_G, B_FICT_G, ANGLE, GAMMA, LINEWIDTH_MHZ,
                       GROUND_G, EXCITED_G, PULSE_US, SATURATION)

from scan_axis import ASSUMED_SLOPE_G_PER_V, assumed_field_G

ALPHA = round(ASSUMED_SLOPE_G_PER_V, 5)  # original numerical precision, G/V
N = 7500
DM_SIZE = 12
LANDMARKS = [-8.,-6.,-4.,-3.,-2.,-1.5,-.8,0.,.5,1.3,2.2,3.,4.]


def angular_momentum(f):
    m = np.arange(-f, f+1)
    plus = np.diag(np.sqrt((f-m[:-1])*(f+m[:-1]+1)), k=-1)
    return (plus+plus.T)/2, np.diag(m.astype(float))


def embed(op, offset):
    out = np.zeros((DM_SIZE, DM_SIZE))
    out[offset:offset+len(op),offset:offset+len(op)] = op
    return out


gx, gz = [embed(x,0) for x in angular_momentum(2)]
ex, ez = [embed(x,5) for x in angular_momentum(3)]
excited = np.diag([0.]*5+[1.]*7)
eye = np.eye(DM_SIZE)
ground2 = np.diag([0.]*4+[1.]+[0.]*7).astype(complex)
e3_index, e2_index = 11,10
g2_index, g1_index = 4,3


def emission(q):
    # Standard Condon-Shortley CG <2,m;1,q|3,m+q> are positive for F'=F+1.
    d = np.zeros((DM_SIZE,DM_SIZE))
    for m in range(-2,3):
        if q==1:
            cg2 = (m+3)*(m+4)/30
        elif q==-1:
            cg2 = (3-m)*(4-m)/30
        else:
            cg2 = (3-m)*(3+m)/15
        d[m+2,5+(m+q+3)] = np.sqrt(cg2)
    return d


jumps = [np.sqrt(GAMMA)*emission(q) for q in (-1,0,1)]
W = sum(j.T@j for j in jumps)
np.testing.assert_allclose(W,GAMMA*excited,atol=1e-12)
dissipator = sum(np.kron(j.conj(),j) for j in jumps)
dissipator -= (np.kron(eye,W)+np.kron(W.T,eye))/2
drive = GAMMA*np.sqrt(SATURATION/2)*(emission(1)+emission(1).T)/2


def generator(h, conditional=False):
    out = -1j*(np.kron(eye,h)-np.kron(h.T,eye))+dissipator
    if conditional:
        # Unnormalized no-count map for spin-changing fluorescence.
        # Keep loss anti-commutator; omit recycling for emitted q=0,-1.
        # At pure sigma+ absorption, these photons signal non-cycling scattering.
        for j in jumps[:2]:
            out -= np.kron(j.conj(),j)
    return out


def exp_blockdiag(op):
    """Probe axial symmetry conserves the difference of effective m indices.

    An excitation by sigma+ adds one to m, so subtract one from all excited m
    to define those indices. Exponentiating independent blocks is exact.
    """
    labels = np.r_[np.arange(-2,3), np.arange(-3,4)-1]
    order = (labels[:,None]-labels[None,:]).ravel(order="F")
    out = np.zeros_like(op)
    for k in np.unique(order):
        sub = np.flatnonzero(order==k)
        idx = np.ix_(sub,sub)
        out[idx] = expm(op[idx])
    return out


def propagate(b, delta, start_rho=ground2, count_events=False):
    """Start stretched, before trap pulse, apply N complete trap+probe periods.

    Uniform single-frequency flat pulse, no motion. When trap is on, keep
    ordinary excited-state Zeeman splitting and spontaneous decay; only ground
    F=2 sees the vector light shift (excited trap shifts not known here).
    Static common optical frequency chosen throughout in a laser rotating frame.
    """
    zg, ze = (2*np.pi*MU_MHZ_G*b*g for g in (GROUND_G,EXCITED_G))
    # E(e3)-E(g2) in rotating frame is -2pi*delta.
    offset = -2*np.pi*delta-(3*ze-2*zg)
    hbase = zg*gz+ze*ez+offset*excited
    hfict = 2*np.pi*MU_MHZ_G*GROUND_G*B_FICT_G*(np.cos(ANGLE)*gz+np.sin(ANGLE)*gx)
    probe = exp_blockdiag(generator(hbase+drive)*PULSE_US)
    trap = expm(generator(hbase+hfict)*PULSE_US)
    # Single-pulse coherence transfer for initial ground coherence = 1. Linearity
    # makes its complex transfer coefficient independent of the actual tiny
    # amplitude introduced by the tilted field. Do not normalize by a varying
    # input at an accidental zero of the tilt.
    before = np.zeros_like(start_rho)
    before[g1_index,g2_index]=1
    resp = (probe @ before.ravel(order="F")).reshape((DM_SIZE,DM_SIZE),order="F")
    rg, re = resp[g1_index,g2_index],resp[e2_index,e3_index]
    # The square-wave field restarts with a fixed phase in the same rotating
    # frame. Optical coherences left after trap-on decay remain in period.
    # Switch probe off and let residual excited population decay (comparison
    # without adding a new transverse tilt). Product of two correct CG factors.
    after = rg + np.sqrt(2/3)*GAMMA*re/(GAMMA-1j*(ze-zg))
    period = probe@trap
    v = np.linalg.matrix_power(period,N-1)@start_rho.ravel(order="F")
    pre = (trap@v).reshape(start_rho.shape,order="F")
    post = (probe@pre.ravel(order="F")).reshape(start_rho.shape,order="F")
    assert abs(np.trace(post)-1)<1e-9
    assert np.min(np.linalg.eigvalsh(post)) > -1e-9
    assert np.min(np.linalg.eigvalsh(pre)) > -1e-9
    def others(rho):
        return (np.trace(rho)-rho[g2_index,g2_index]-rho[e3_index,e3_index]).real
    p_event = ""
    if count_events:
        trap_nc = expm(generator(hbase+hfict,conditional=True)*PULSE_US)
        probe_nc = exp_blockdiag(generator(hbase+drive,conditional=True)*PULSE_US)
        v_nc = np.linalg.matrix_power(probe_nc@trap_nc,N)@start_rho.ravel(order="F")
        p0 = np.trace(v_nc.reshape(start_rho.shape,order="F")).real
        assert -1e-9 <= p0 <= 1+1e-9
        p_event = max(0.,1.-p0)
    return dict(b_G=b,delta_stretched_MHz=delta,
                prob_emitted_at_least_one_pi_or_sigma_minus=p_event,
                ground_coherence_ratio_probe_end=abs(rg),
                ground_coherence_ratio_after_residual_decay=abs(after),
                fraction_outside_cycling_pair_preprobe_7500=others(pre),
                fraction_outside_cycling_pair_postprobe_7500=others(post),
                abs_ground_rho12_preprobe_7500=abs(pre[g1_index,g2_index]),
                abs_ground_rho12_postprobe_7500=abs(post[g1_index,g2_index]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",type=Path,required=True)
    parser.add_argument("--step",type=float,default=.01,help="Control voltage step in V")
    parser.add_argument("--count-all",action="store_true")
    args = parser.parse_args()
    rows = []
    for x in np.unique(np.round(np.r_[np.arange(-8.,4.+.001,args.step),LANDMARKS],7)):
        b = assumed_field_G(x, ALPHA)
        row = dict(case="illustrative_scan",x=x,**propagate(b,-2.-MU_MHZ_G*b,count_events=args.count_all or x in LANDMARKS),
                   quantization_voltage_V=x)
        rows.append(row)
        if x in LANDMARKS:
            print(x,row["ground_coherence_ratio_probe_end"],row["ground_coherence_ratio_after_residual_decay"],row["fraction_outside_cycling_pair_preprobe_7500"],flush=True)
    rows.append(dict(case="published_working_point",x="",**propagate(20.,-LINEWIDTH_MHZ/2,count_events=True),
                     quantization_voltage_V=""))
    with args.csv.open("w",newline="") as fp:
        w = csv.DictWriter(fp,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mx = max(rows[:-1],key=lambda r:r["fraction_outside_cycling_pair_preprobe_7500"])
    print("Largest sampled outside-cycle population:",mx)
    print("Published point:",rows[-1])
