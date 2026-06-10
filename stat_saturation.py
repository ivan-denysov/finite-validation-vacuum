"""
СТАТИСТИКА 1: сатурация трубы — 8 seeds x скан энергии, среднее±std.
Вопросы: (а) устойчив ли линейный старт + плато; (б) значение плато (среднее, разброс).
"""
import numpy as np

def build(Nnodes, seed, deg=6, box=40.0):
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
    Nn=len(pos); maxd=max(len(n) for n in nbrs)
    na=np.full((Nn,maxd),-1)
    for i,n in enumerate(nbrs): na[i,:len(n)]=n
    valid=(na>=0); cnt=valid.sum(1); cnt[cnt==0]=1
    return pos,na,valid,cnt

def run(pos,na,valid,cnt,drive,K=0.6,steps=500,speed=0.05,omega0=1.0,seed=0):
    rng=np.random.RandomState(seed+1000); Nn=len(pos)
    theta=rng.uniform(0,2*np.pi,Nn); dt=0.05
    def coup(th):
        tn=np.where(valid, th[na],0.0)
        return np.where(valid,np.sin(tn-th[:,None]),0.0).sum(1)/cnt
    th=theta.copy()
    for t in range(steps): th=th+dt*(omega0+K*coup(th))
    ctrl=omega0+K*coup(th)
    th=theta.copy()
    for t in range(steps):
        bx=5+speed*t
        d=(pos[:,0]-bx)**2+(pos[:,1]-20)**2+(pos[:,2]-20)**2
        bi=np.argmin(d)
        om=np.full(Nn,omega0); om[bi]=drive
        th=th+dt*(om+K*coup(th))
    fd=np.abs((omega0+K*coup(th))-ctrl)
    r=np.sqrt((pos[:,1]-20)**2+(pos[:,2]-20)**2)
    tube=(r<3.0)&(pos[:,0]>6)&(pos[:,0]<5+speed*steps)
    return fd[tube].sum()

Es=[0.5,1.0,2.0,4.0,8.0]
seeds=list(range(8))
N=12000
print(f"=== Сатурация: {len(seeds)} seeds, N={N}, скан E ===")
results=np.zeros((len(seeds),len(Es)))
for si,seed in enumerate(seeds):
    pos,na,valid,cnt=build(N,seed)
    for ei,E in enumerate(Es):
        results[si,ei]=run(pos,na,valid,cnt,1.0+E,seed=seed)
    print(f"seed {seed}: " + " ".join(f"{results[si,ei]:6.2f}" for ei in range(len(Es))))
print()
mean=results.mean(0); std=results.std(0)
print(f"{'E':>5} {'mean':>7} {'std':>6} {'CV%':>5}")
for ei,E in enumerate(Es):
    print(f"{E:>5} {mean[ei]:>7.2f} {std[ei]:>6.2f} {100*std[ei]/mean[ei]:>5.0f}")
print()
# линейность на старте (E=0.5 -> 1.0): отношение зарядов ~2?
ratio=mean[1]/mean[0]
print(f"линейный старт: Q(1.0)/Q(0.5) = {ratio:.2f} (если ~2 => линейно)")
# плато: среднее по E>=2
plateau=mean[2:].mean(); plateau_std=results[:,2:].std()
print(f"ПЛАТО (E>=2): {plateau:.2f} ± {plateau_std:.2f}")
print(f"рост плато от E=2 к E=8: {mean[-1]/mean[2]:.2f} (если ~1 => настоящая сатурация)")
