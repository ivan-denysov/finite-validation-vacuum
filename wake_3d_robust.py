"""
Закрепление формы следа:
(A) устойчивость конуса к ПОРОГУ (thr scan)
(B) АМОРФНАЯ сеть (случайные связи) vs регулярная решётка — артефакт осей или нет
"""
import numpy as np

def run_regular(N=40, K=0.6, steps=700, speed=0.04, drive_freq=3.0, seed=42):
    rng=np.random.RandomState(seed)
    theta0=rng.uniform(0,2*np.pi,(N,N,N)); omega0=1.0; dt=0.05
    cy,cz=N//2,N//2
    def coup(th):
        c =np.sin(np.roll(th,1,0)-th)+np.sin(np.roll(th,-1,0)-th)
        c+=np.sin(np.roll(th,1,1)-th)+np.sin(np.roll(th,-1,1)-th)
        c+=np.sin(np.roll(th,1,2)-th)+np.sin(np.roll(th,-1,2)-th)
        return c
    th=theta0.copy()
    for t in range(steps): th=th+dt*(omega0+(K/6.0)*coup(th))
    ctrl=th.copy()
    th=theta0.copy()
    for t in range(steps):
        dx=max(1,min(N-2,int(5+speed*t)))
        om=np.full((N,N,N),omega0); om[dx,cy,cz]=drive_freq
        th=th+dt*(om+(K/6.0)*coup(th))
    diff=np.abs(np.angle(np.exp(1j*(th-ctrl))))
    fx=max(1,min(N-2,int(5+speed*(steps-1))))
    return diff,cy,cz,fx

def radius_at(diff,x,cy,cz,thr):
    pl=diff[x,:,:]; m=pl>thr
    if m.sum()==0: return 0,0.0
    ys,zs=np.where(m); return int(m.sum()),float(np.sqrt((ys-cy)**2+(zs-cz)**2).mean())

print("=== (A) Устойчивость к порогу (регулярная 3D) ===")
diff,cy,cz,fx=run_regular()
print(f"тело на x={fx}. Сравниваем радиус у начала пути (x=10) vs у тела (x={fx-3})")
print(f"{'thr':>6} {'r@start':>8} {'r@body':>8} {'форма':>22}")
for thr in [0.02,0.05,0.10,0.20,0.40]:
    n1,r1=radius_at(diff,10,cy,cz,thr)
    n2,r2=radius_at(diff,fx-3,cy,cz,thr)
    form = 'конус назад (расшир.)' if r1>r2+0.5 else 'постоянная' if abs(r1-r2)<=0.5 else 'сужается назад'
    print(f"{thr:>6} {r1:>8.2f} {r2:>8.2f} {form:>22}")
print("(r@start > r@body => след расширяется ОТ тела назад = кильватер)")
print()

# (B) Аморфная сеть: случайный граф со средней степенью ~6 (как у решётки)
print("=== (B) Аморфная сеть (случайные связи, ~6 соседей) ===")
def run_amorphous(Nnodes=20000, deg=6, K=0.6, steps=700, speed_idx=0.04, drive_freq=3.0, seed=42):
    rng=np.random.RandomState(seed)
    # узлы в 3D кубе случайно
    pos=rng.uniform(0,40,(Nnodes,3))
    # связи: каждый узел к deg ближайшим (kNN) — аморфно, но локально
    from numpy import argsort
    # строим kNN грубо (для скорости — по случайной подвыборке расстояний нельзя, делаем точно но память ок)
    # чтобы не O(N^2) на 20000 — берём сетку-хеш по ячейкам
    # упрощаем: соседи = deg случайных узлов в радиусе R
    R=3.0
    nbrs=[[] for _ in range(Nnodes)]
    # пространственный хеш
    cell=4.0; grid={}
    for i,p in enumerate(pos):
        key=tuple((p//cell).astype(int)); grid.setdefault(key,[]).append(i)
    for i,p in enumerate(pos):
        key=(p//cell).astype(int); cand=[]
        for dxk in(-1,0,1):
            for dyk in(-1,0,1):
                for dzk in(-1,0,1):
                    cand+=grid.get((key[0]+dxk,key[1]+dyk,key[2]+dzk),[])
        cand=[j for j in cand if j!=i]
        if not cand: continue
        d=np.sqrt(((pos[cand]-p)**2).sum(1))
        order=np.argsort(d)[:deg]
        nbrs[i]=[cand[k] for k in order]
    theta=rng.uniform(0,2*np.pi,Nnodes); omega0=1.0; dt=0.05
    # тело едет вдоль x по линии y=z=20
    axis=np.array([20.0,20.0]); 
    def coup(th):
        out=np.zeros(Nnodes)
        for i in range(Nnodes):
            ns=nbrs[i]
            if ns: out[i]=np.sin(th[ns]-th[i]).sum()/len(ns)
        return out
    # контроль
    th=theta.copy()
    for t in range(steps): th=th+dt*(omega0+K*coup(th))
    ctrl=th.copy()
    th=theta.copy()
    for t in range(steps):
        bx=5+speed_idx*t
        # ближайший узел к телу
        d=(pos[:,0]-bx)**2+(pos[:,1]-20)**2+(pos[:,2]-20)**2
        bi=np.argmin(d)
        om=np.full(Nnodes,omega0); om[bi]=drive_freq
        th=th+dt*(om+K*coup(th))
    diff=np.abs(np.angle(np.exp(1j*(th-ctrl))))
    return pos,diff,5+speed_idx*(steps-1)

pos,diffA,bx=run_amorphous()
print(f"тело на x≈{bx:.1f}. Радиус следа по слоям x (порог 0.05):")
def amorph_radius(pos,diff,xlo,xhi,thr=0.05):
    m=(pos[:,0]>=xlo)&(pos[:,0]<xhi)&(diff>thr)
    if m.sum()==0: return 0,0.0
    r=np.sqrt((pos[m,1]-20)**2+(pos[m,2]-20)**2).mean()
    return int(m.sum()),float(r)
for xlo in range(6,int(bx),4):
    n,r=amorph_radius(pos,diffA,xlo,xlo+4)
    print(f"  x={xlo:>4}-{xlo+4}: {n:>4} узлов, радиус {r:.2f}")
n1,r1=amorph_radius(pos,diffA,8,12)
n2,r2=amorph_radius(pos,diffA,int(bx)-4,int(bx))
print(f"start r={r1:.2f} vs body r={r2:.2f} => {'конус назад' if r1>r2+0.5 else 'постоянная' if abs(r1-r2)<=0.5 else 'сужается назад'}")
