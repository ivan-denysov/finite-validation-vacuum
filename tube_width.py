"""
(2) Ширина трубы: фундаментальна или = параметрам сети (плотность, радиус связи)?
Меняем плотность узлов и среднюю степень (deg), меряем ширину следа.
Если ширина МАСШТАБИРУЕТСЯ с шагом сети (расстоянием между узлами) -> она задаётся
дискретностью сети (~ несколько шагов), не фундаментальна сама по себе, НО постоянна
в единицах шага. Если меняется произвольно -> артефакт.
"""
import numpy as np

def build(Nnodes, deg, box=40.0, seed=42):
    rng=np.random.RandomState(seed)
    pos=rng.uniform(0,box,(Nnodes,3))
    cell=box/ (Nnodes**(1/3)) * 2.0
    grid={}
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
    # средний шаг сети = среднее расстояние до ближайшего соседа
    step=np.mean([np.sqrt(((pos[n[0]]-pos[i])**2).sum()) for i,n in enumerate(nbrs) if n])
    return pos,nbrs,step

def run_width(Nnodes,deg,K=0.6,drive=3.0,steps=600,speed=0.045,omega0=1.0,seed=42):
    pos,nbrs,step=build(Nnodes,deg,seed=seed)
    Nn=len(pos); maxd=max(len(n) for n in nbrs)
    na=np.full((Nn,maxd),-1)
    for i,n in enumerate(nbrs): na[i,:len(n)]=n
    valid=(na>=0); cnt=valid.sum(1); cnt[cnt==0]=1
    rng=np.random.RandomState(seed); theta=rng.uniform(0,2*np.pi,Nn); dt=0.05
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
    # ширина: среди возмущённых узлов вдоль пути — радиальное распределение
    r=np.sqrt((pos[:,1]-20)**2+(pos[:,2]-20)**2)
    onpath=(pos[:,0]>8)&(pos[:,0]<30)&(fd>0.03)
    if onpath.sum()<3: return step,0.0,0.0
    width_abs=r[onpath].mean()          # средний радиус возмущённых (абс. ширина)
    width_steps=width_abs/step          # ширина в шагах сети
    return step,width_abs,width_steps

print("=== Ширина трубы vs плотность сети и степень ===")
print(f"{'Nnodes':>7} {'deg':>4} {'step':>6} {'width_abs':>10} {'width/step':>11}")
for Nnodes,deg in [(8000,6),(15000,6),(25000,6),(15000,4),(15000,8)]:
    step,wa,ws=run_width(Nnodes,deg)
    print(f"{Nnodes:>7} {deg:>4} {step:>6.2f} {wa:>10.2f} {ws:>11.2f}")
print()
print("Если width/step ~ПОСТОЯННО при разных плотностях => ширина = фикс. число ШАГОВ сети")
print("(труба шириной в ~N шагов независимо от масштаба) — это уже свойство, не артефакт.")
