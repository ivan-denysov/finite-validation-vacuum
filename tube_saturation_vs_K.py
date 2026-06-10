"""
Проверка: порог насыщения трубы зависит от связи K или это устойчивая граница?
Для нескольких K строим кривую total_charge(E) и находим уровень плато.
"""
import numpy as np

def build_amorphous(Nnodes=15000, deg=6, box=40.0, seed=42):
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

def run(pos,nbr_arr,valid,cnt,drive_freq,K,steps=600,speed=0.045,omega0=1.0,seed=42):
    rng=np.random.RandomState(seed); Nn=len(pos)
    theta=rng.uniform(0,2*np.pi,Nn); dt=0.05
    def coup(th):
        tn=np.where(valid, th[nbr_arr], 0.0)
        return np.where(valid, np.sin(tn-th[:,None]),0.0).sum(1)/cnt
    th=theta.copy()
    for t in range(steps): th=th+dt*(omega0+K*coup(th))
    ctrl=omega0+K*coup(th)
    th=theta.copy()
    for t in range(steps):
        bx=5+speed*t
        d=(pos[:,0]-bx)**2+(pos[:,1]-20)**2+(pos[:,2]-20)**2
        bi=np.argmin(d)
        om=np.full(Nn,omega0); om[bi]=drive_freq
        th=th+dt*(om+K*coup(th))
    pert=omega0+K*coup(th)
    return np.abs(pert-ctrl)

pos,nbrs=build_amorphous()
Nn=len(pos); maxd=max(len(n) for n in nbrs)
nbr_arr=np.full((Nn,maxd),-1)
for i,n in enumerate(nbrs): nbr_arr[i,:len(n)]=n
valid=(nbr_arr>=0); cnt=valid.sum(1); cnt[cnt==0]=1
r=np.sqrt((pos[:,1]-20)**2+(pos[:,2]-20)**2)
tube=(r<3.0)&(pos[:,0]>6)&(pos[:,0]<32)

print("=== total_charge(E) для разных K — где плато? ===")
Es=[0.5,1.0,2.0,4.0,8.0]
print(f"{'K':>5} | " + " ".join(f"E={e:<4}" for e in Es) + " | plateau")
for K in [0.3,0.6,1.0,1.5]:
    row=[]
    for E in Es:
        fd=run(pos,nbr_arr,valid,cnt,1.0+E,K)
        row.append(fd[tube].sum())
    plateau=np.mean(row[2:])  # уровень при больших E
    print(f"{K:>5} | " + " ".join(f"{v:>6.2f}" for v in row) + f" | {plateau:>6.2f}")
print()
print("Если plateau РАСТЁТ с K => порог = свойство связи (не фундаментален).")
print("Если plateau ~постоянен => устойчивый квант возмущения.")
