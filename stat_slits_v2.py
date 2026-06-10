"""
СТАТИСТИКА 2 (испр.): (а) 5 seeds с РЕАЛИСТИЧНЫМИ вариациями (шум 0.003, джиттер пуска ±2, A±10%);
(б) скан уровня шума -> порог смерти узора.
"""
import numpy as np

Nx,Ny=300,200; dx=0.5; c=1.0; dt=0.2*dx/c
steps=1500
barrier_x=120; screen_x=260
slit_y1,slit_y2=80,120; halfw=4
damp_width=15

damp=np.zeros((Ny,Nx))
for i in range(damp_width):
    w=(1-i/damp_width)*0.5
    damp[i,:]=np.maximum(damp[i,:],w); damp[Ny-1-i,:]=np.maximum(damp[Ny-1-i,:],w)
    damp[:,Nx-1-i]=np.maximum(damp[:,Nx-1-i],w); damp[:,i]=np.maximum(damp[:,i],w)

wall=np.zeros((Ny,Nx))
for y in range(Ny):
    if not(abs(y-slit_y1)<=halfw or abs(y-slit_y2)<=halfw):
        wall[y,barrier_x]=1; wall[y,barrier_x+1]=1

def lap_cut(th):
    tot=np.zeros_like(th)
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        nb=np.roll(th,sh,ax); wb=np.roll(wall,sh,ax)
        link=1.0-np.maximum(wall,wb)
        tot+=link*(nb-th)
    return tot/dx**2

def run(seed, noise, jitter=True):
    rng=np.random.RandomState(seed)
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    dy = rng.randint(-2,3) if jitter else 0
    Aj = 0.8*(1+ (rng.uniform(-0.1,0.1) if jitter else 0))
    x0=40*dx; y0=(100+dy)*dx; s=6.0; k=1.2
    gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
    theta=Aj*gauss*np.cos(k*(X-x0)) + noise*rng.randn(Ny,Nx)
    theta_t=-c*np.gradient(theta,dx,axis=1) + noise*rng.randn(Ny,Nx)
    env=np.zeros(Ny)
    for t in range(steps):
        a=c**2*lap_cut(theta)-np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
        env=np.maximum(env,np.abs(theta_t[:,screen_x]))
    return env

print("=== (а) 5 seeds, реалистичные вариации (шум 0.003, джиттер ±2, A±10%) ===")
print(f"{'seed':>5} {'y_пик':>6} {'контраст':>9}")
ymaxes=[]; contrasts=[]
for seed in range(5):
    env=run(seed,0.003)
    inner=slice(damp_width+3,Ny-damp_width-3)
    ys=np.arange(Ny)
    ymax=ys[inner][np.argmax(env[inner])]
    ctr=env[95:106].max()/max((env[78:83].mean()+env[118:123].mean())/2,1e-9)
    ymaxes.append(ymax); contrasts.append(ctr)
    print(f"{seed:>5} {ymax:>6} {ctr:>9.1f}")
print(f"пик: {np.mean(ymaxes):.1f}±{np.std(ymaxes):.1f}; контраст: {np.mean(contrasts):.1f}±{np.std(contrasts):.1f}")
print()
print("=== (б) порог смерти узора: скан шума (seed=0, без джиттера) ===")
print(f"{'noise':>7} {'контраст':>9}")
for noise in [0.0,0.002,0.005,0.01,0.02]:
    env=run(0,noise,jitter=False)
    ctr=env[95:106].max()/max((env[78:83].mean()+env[118:123].mean())/2,1e-9)
    print(f"{noise:>7} {ctr:>9.1f}")
print("контраст>2 = узор жив; ~1 = мёртв")
