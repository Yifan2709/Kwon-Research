"""Illustrative scenarios, not estimated apparatus parameters. Run python scan.py."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, expit
from voltage_axis import SCENARIO_SLOPE_G_PER_V, SCENARIO_ZERO_V, model_field_g
out=Path(__file__).resolve().parent
# All x coordinates, including envelopes, are control voltages in V.
x=np.linspace(-8,4,1201)
nu=.6998
# Fixed lab helicity. The trial zero voltage places a resonance at +0.65 V.
B=model_field_g(x, SCENARIO_SLOPE_G_PER_V, SCENARIO_ZERO_V)
def spin(T,r):
    phase=2*np.pi*nu*B*T
    # Normalized intensity of a train of identical short transverse kicks.
    return (1-r)**2/((1-r)**2+4*r*np.sin(phase/2)**2)
def model(T=.94,r=.55):
    L=spin(T,r)
    # Conditional 3D classical thermal energy CDF at escape depth.
    # Left-side protection and right-end heating are hypotheses, intentionally explicit.
    gate=expit((x+4.25)/.18)
    edge=expit((x-3.58)/.16)
    base=.075+2.7*edge
    temp=base+3.3*gate*(L-spin_min(r))/(1-spin_min(r))
    return gammainc(3,1/temp)
def spin_min(r):return ((1-r)/(1+r))**2
def recoil():
    delta=-2-1.3996*B
    nphot=3000*np.pi*6.065/(2+4*(delta/6.065)**2)
    a=.000362*nphot/(6*5.6)
    return np.clip(sum(2*(-1)**(k+1)*np.exp(-k*k*np.pi**2*a) for k in range(1,120)),0,1)
# Equal-time sequential scanning in presence of time-periodic unwanted heating.
# Same baseline as above; independent clock time, not atomic Larmor phase.
def alias(scan_rate_v_per_s):
    phase=2*np.pi*(x+8)/(scan_rate_v_per_s*15.2)+2*np.pi*(8+.65)/1.52*(-1)
    temp=.075+2.7*expit((x-3.58)/.16)+3.3*expit((x+4.25)/.18)*((1+np.cos(phase))/2)**2
    return gammainc(3,1/temp)
r=.55;L=spin(.94,r); cols=dict(x=x,recoil=recoil(),spin_response=L,spin_only_survival=gammainc(3,1/(.075+3.3*(L-spin_min(r))/(1-spin_min(r)))) ,combined=model(),changed_period=model(.94*1.1),alias=alias(.1),alias_double_speed=alias(.2))
cols.update(scan_voltage_V=x, b_assumed_G=B)
np.savetxt(out/'scenarios.csv',np.column_stack(list(cols.values())),header=','.join(cols),comments='',delimiter=',')
fig,axs=plt.subplots(2,1,figsize=(9,6.5),sharex=True,layout='constrained')
xs=[-4,-3,-2,-1.5,-.8,0,.5,1.3,2.2,3,4]; ys=[0,1,0,1,0,.9,0,.95,0,.7,0]
axs[0].scatter(xs,ys,label='Reported extrema (approx.)',color='black',s=24,zorder=4)
axs[0].plot(x,cols['recoil'],label='Recoil only',ls='--')
axs[0].plot(x,cols['combined'],label='Spin resonance + heating envelope')
axs[0].plot(x,cols['spin_only_survival'],label='Spin resonance, flat envelope',alpha=.55,lw=1)
axs[1].plot(x,cols['combined'],label='Pulse period 0.94 us')
axs[1].plot(x,cols['changed_period'],label='Pulse period +10%')
axs[1].plot(x,cols['alias_double_speed'],label='Time drift, double scan speed',ls=':')
for ax in axs:
 ax.legend(fontsize=8,ncol=2);ax.set(ylabel='Survival (illustrative)',ylim=(-.05,1.23));ax.grid(alpha=.2)
axs[-1].set_xlabel('Quantization-axis magnetic-field control voltage (V)');fig.savefig(out/'scenarios.png',dpi=170)
print([(a, round(cols['combined'][np.argmin(abs(x-a))],3)) for a in [-8,-5,-4,-3,-2,-1.5,-.8,0,.5,1.3,2.2,3,4]])
