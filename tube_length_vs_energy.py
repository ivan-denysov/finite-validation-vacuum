"""
(1) Длина трубчатого следа vs ЭНЕРГИЯ возмутителя (аморфная 3D сеть).
Энергия = насколько возмутитель выбит из среднего ритма (drive_freq - omega0).
Гипотеза: больше энергия -> длиннее труба (дольше релаксирует)?
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
    # предкомпиляция соседей в массив для скорости
    return pos,nbrs

def run(pos,nbrs,drive_freq,K=0.6,steps=700,speed=0.04,omega0=1.0,seed=42):
    rng=np.random.RandomState(seed); Nn=len(pos)
    theta=rng.uniform(0,2*np.pi,Nn); dt=0.05
    # предсоберём индексы соседей в padded-массив
    maxd=max(len(n) for n in nbrs)
    nbr_arr=np.full((Nn,maxd),-1)
    for i,n in enumerate(nbrs): nbr_arr[i,:len(n)]=n
    valid=(nbr_arr>=0)
    def coup(th):
        tn=np.where(valid, th[nbr_arr], 0.0)
        s=np.where(valid, np.sin(tn-th[:,None]),0.0).sum(1)
        cnt=valid.sum(1); cnt[cnt==0]=1
        return s/cnt
    th=theta.copy()
    for t in range(steps): th=th+dt*(omega0+K*coup(th))
    ctrl=th.copy()
    th=theta.copy()
    for t in range(steps):
        bx=5+speed*t
        d=(pos[:,0]-bx)**2+(pos[:,1]-20)**2+(pos[:,2]-20)**2
        bi=np.argmin(d)
        om=np.full(Nn,omega0); om[bi]=drive_freq
        th=th+dt*(om+K*coup(th))
    diff=np.abs(np.angle(np.exp(1j*(th-ctrl))))
    return diff, 5+speed*(steps-1)

pos,nbrs=build_amorphous()
print("=== Длина трубы vs энергия (drive_freq) ===")
print(f"{'drive':>6} {'energy':>7} {'tube_len':>9} {'n_nodes':>8} {'max_dev':>8}")
thr=0.05
for drive in [1.5, 2.0, 3.0, 5.0, 8.0]:
    diff,bx=run(pos,nbrs,drive)
    # труба: узлы возле оси (r<3) с возмущением>порог
    r=np.sqrt((pos[:,1]-20)**2+(pos[:,2]-20)**2)
    tube=(r<3.0)&(diff>thr)
    if tube.sum()>0:
        xs=pos[tube,0]
        length=xs.max()-xs.min()
    else:
        length=0.0
    energy=drive-1.0
    print(f"{drive:>6} {energy:>7.1f} {length:>9.2f} {int(tube.sum()):>8} {diff.max():>8.3f}")
print()
print("Если tube_len растёт с energy => длина трубы кодирует энергию/массу")
