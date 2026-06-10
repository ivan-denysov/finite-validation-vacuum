"""
(1) корректно: энергия трубы через ЧАСТОТНЫЙ рассинхрон (без потолка фазы π).
Метрики:
 - mean |dtheta/dt - omega0| в трубе (частотное отклонение, не ограничено π)
 - интеграл рассинхрона по трубе (полный "заряд" следа)
 - длина и ширина для контроля
"""
import numpy as np

def build_amorphous(Nnodes=20000, deg=6, box=40.0, seed=42):
    rng=np.random.RandomState(seed)
    pos=rng.uniform(0,box,(Nnodes,3))
    cell=4.0; grid={}
    for i,p in enumerate(pos):
        key=tuple((p//cell).astype(int)); grid.setdefault(key,[]).append(i)
    nbrs=[]
    for i,p in enumerate(pos):
        key=(p//cell).astype(int); cand=[]
        for a in(-1,0,1):
            for b in(-1,0,1):
                for c in(-1,0,1):
                    cand+=grid.get((key[0]+a,key[1]+b,key[2]+c),[])
        cand=[j for j in cand if j!=i]
        if cand:
            d=np.sqrt(((pos[cand]-p)**2).sum(1)); order=np.argsort(d)[:deg]
            nbrs.append([cand[k] for k in order])
        else: nbrs.append([])
    return pos,nbrs

def run(pos,nbrs,drive_freq,K=0.6,steps=700,speed=0.04,omega0=1.0,seed=42):
    rng=np.random.RandomState(seed); Nn=len(pos)
    theta=rng.uniform(0,2*np.pi,Nn); dt=0.05
    maxd=max(len(n) for n in nbrs)
    nbr_arr=np.full((Nn,maxd),-1)
    for i,n in enumerate(nbrs): nbr_arr[i,:len(n)]=n
    valid=(nbr_arr>=0); cnt=valid.sum(1); cnt[cnt==0]=1
    def coup(th):
        tn=np.where(valid, th[nbr_arr], 0.0)
        return np.where(valid, np.sin(tn-th[:,None]),0.0).sum(1)/cnt
    # контроль: запоминаем частоту узлов в конце
    th=theta.copy(); 
    for t in range(steps):
        d_th=omega0+K*coup(th); th=th+dt*d_th
    ctrl_rate=omega0+K*coup(th)   # частота контроля (≈omega0)
    # возмущение
    th=theta.copy()
    for t in range(steps):
        bx=5+speed*t
        d=(pos[:,0]-bx)**2+(pos[:,1]-20)**2+(pos[:,2]-20)**2
        bi=np.argmin(d)
        om=np.full(Nn,omega0); om[bi]=drive_freq
        d_th=om+K*coup(th); th=th+dt*d_th
    pert_rate=np.full(Nn,omega0)+K*coup(th)
    freq_dev=np.abs(pert_rate-ctrl_rate)   # частотный рассинхрон, НЕ ограничен π
    return freq_dev

pos,nbrs=build_amorphous()
r=np.sqrt((pos[:,1]-20)**2+(pos[:,2]-20)**2)
tube_mask=(r<3.0)&(pos[:,0]>6)&(pos[:,0]<33)

print("=== Энергия трубы через частотный рассинхрон (без потолка) ===")
print(f"{'drive':>6} {'E_in':>6} {'mean_fdev':>10} {'total_charge':>13} {'peak_fdev':>10}")
res=[]
for drive in [1.5,2.0,3.0,5.0,8.0,12.0]:
    fd=run(pos,nbrs,drive)
    mean_fd=fd[tube_mask].mean()
    total=fd[tube_mask].sum()
    peak=fd.max()
    Ein=drive-1.0
    res.append((Ein,mean_fd,total,peak))
    print(f"{drive:>6} {Ein:>6.1f} {mean_fd:>10.4f} {total:>13.3f} {peak:>10.3f}")

# проверим линейность total_charge ~ E_in
import numpy as np
E=np.array([x[0] for x in res]); Q=np.array([x[2] for x in res])
A=np.vstack([E,np.ones_like(E)]).T
slope,intercept=np.linalg.lstsq(A,Q,rcond=None)[0]
ss_res=((Q-(slope*E+intercept))**2).sum(); ss_tot=((Q-Q.mean())**2).sum()
r2=1-ss_res/ss_tot
print()
print(f"линейная подгонка total_charge = {slope:.3f}*E + {intercept:.3f}, R^2={r2:.3f}")
print("R^2 близко к 1 => 'заряд трубы' линейно кодирует вложенную энергию")
