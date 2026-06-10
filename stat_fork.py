"""
СТАТИСТИКА 3: развилка (асимметричные щели) — 5 seeds с реалистичными вариациями.
Устойчивы ли: (а) разрыв max_такта vs центроид; (б) посадка argmax энергии на argmax такта.
"""
import numpy as np

Nx,Ny=300,200; dx=0.5; c=1.0; dt=0.2*dx/c
steps=1500
barrier_x=120; screen_x=260
slit_y1,slit_y2=80,120
w1,w2=2,6
damp_width=15

damp=np.zeros((Ny,Nx))
for i in range(damp_width):
    w=(1-i/damp_width)*0.5
    damp[i,:]=np.maximum(damp[i,:],w); damp[Ny-1-i,:]=np.maximum(damp[Ny-1-i,:],w)
    damp[:,Nx-1-i]=np.maximum(damp[:,Nx-1-i],w); damp[:,i]=np.maximum(damp[:,i],w)

wall=np.zeros((Ny,Nx))
for y in range(Ny):
    if not(abs(y-slit_y1)<=w1 or abs(y-slit_y2)<=w2):
        wall[y,barrier_x]=1; wall[y,barrier_x+1]=1

def lap_cut(th):
    tot=np.zeros_like(th)
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        nb=np.roll(th,sh,ax); wb=np.roll(wall,sh,ax)
        link=1.0-np.maximum(wall,wb)
        tot+=link*(nb-th)
    return tot/dx**2

def run(seed):
    rng=np.random.RandomState(seed)
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    dy=rng.randint(-2,3); Aj=0.8*(1+rng.uniform(-0.1,0.1))
    x0=40*dx; y0=(100+dy)*dx; s=6.0; k=1.2
    gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
    theta=Aj*gauss*np.cos(k*(X-x0))+0.003*rng.randn(Ny,Nx)
    theta_t=-c*np.gradient(theta,dx,axis=1)+0.003*rng.randn(Ny,Nx)
    envT=np.zeros(Ny); E_acc=np.zeros(Ny)
    for t in range(steps):
        a=c**2*lap_cut(theta)-np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
        col=theta[:,screen_x]
        gx=np.gradient(theta[:,screen_x-1:screen_x+2],dx,axis=1)[:,1]
        gy=np.gradient(col,dx)
        E=theta_t[:,screen_x]**2/2+c**2*(gx**2+gy**2)/2+(1-np.cos(col))
        E_acc+=E*dt
        envT=np.maximum(envT,np.abs(theta_t[:,screen_x]))
    inner=slice(damp_width+3,Ny-damp_width-3)
    ys=np.arange(Ny)
    yt=ys[inner][np.argmax(envT[inner])]
    ye=ys[inner][np.argmax(E_acc[inner])]
    wgt=envT[inner]**2
    yc=(ys[inner]*wgt).sum()/wgt.sum()
    return yt,ye,yc

print("=== Развилка (асимметрия 2vs6): 5 seeds, реалистичные вариации ===")
print(f"{'seed':>5} {'y_такт':>7} {'y_энергия':>10} {'y_центроид':>11} {'разрыв':>7}")
yts=[];yes=[];ycs=[];gaps=[]
for seed in range(5):
    yt,ye,yc=run(seed)
    gap=abs(yt-yc)
    yts.append(yt);yes.append(ye);ycs.append(yc);gaps.append(gap)
    print(f"{seed:>5} {yt:>7} {ye:>10} {yc:>11.1f} {gap:>7.1f}")
print()
print(f"такт: {np.mean(yts):.1f}±{np.std(yts):.1f}; энергия: {np.mean(yes):.1f}±{np.std(yes):.1f}; центроид: {np.mean(ycs):.1f}±{np.std(ycs):.1f}")
print(f"разрыв такт-центроид: {np.mean(gaps):.1f}±{np.std(gaps):.1f}")
print(f"|энергия - такт| по seeds: {[abs(a-b) for a,b in zip(yes,yts)]}")
print("Устойчиво, если: разрыв стабильно >5 И |энергия-такт| стабильно мал")
