import numpy as np
rng=np.random.default_rng(11)
def run(omega,K,steps=6000,dt=0.005):
    N=len(omega); theta=rng.uniform(0,2*np.pi,N); hist=[]
    for t in range(steps):
        z=np.mean(np.exp(1j*theta)); R=np.abs(z); psi=np.angle(z)
        dth=omega+K*R*np.sin(psi-theta); theta=theta+dt*dth
        if t>steps-400: hist.append(dth.copy())
    hist=np.array(hist); return hist.mean(0), hist.std(0), R

# DIAGNOSTIC 1: pure normal band, no quanta. Must synchronise.
on=rng.normal(1.0,0.15,300)
inst,insd,R=run(on,3.0)
print("PURE NORMAL BAND (no quanta):")
print(f"  R={R:.3f}  (R~1 => synchronised)")
print(f"  inst freq: mean={inst.mean():.3f} std={inst.std():.3f}")
print(f"  inst_std (per-osc time variation): min={insd.min():.4f} median={np.median(insd):.4f} max={insd.max():.4f}")
print(f"  spread of time-avg inst freq across oscillators: {inst.std():.4f}")
print()
# how many share ~same inst freq?
locked = np.abs(inst-np.median(inst))<0.1
print(f"  fraction within 0.1 of median inst freq: {locked.mean():.3f}")
print()
# DIAGNOSTIC 2: normal + quanta, look at the NORMAL subset's inst freq spread
oq=np.exp(rng.uniform(np.log(10),np.log(1e6),300))
om=np.concatenate([on,oq]); iq=np.concatenate([np.zeros(300,bool),np.ones(300,bool)])
inst2,insd2,R2=run(om,3.0)
print("NORMAL+QUANTA:")
print(f"  global R={R2:.3f}")
print(f"  normal subset: inst mean={inst2[~iq].mean():.3f} std={inst2[~iq].std():.4f}")
print(f"  quanta subset: inst mean={inst2[iq].mean():.1f} std={inst2[iq].std():.1f}")
print(f"  normal within 0.1 of normal-median: {(np.abs(inst2[~iq]-np.median(inst2[~iq]))<0.1).mean():.3f}")
print(f"  quanta within 0.1 of normal-median: {(np.abs(inst2[iq]-np.median(inst2[~iq]))<0.1).mean():.3f}")
