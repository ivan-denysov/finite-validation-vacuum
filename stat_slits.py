"""
СТАТИСТИКА 2: двухщелевой — устойчивость узора к шуму нач. условий поля.
5 seeds: к пакету добавлен слабый случайный фоновый шум поля.
Держатся ли: 3 пика, провалы напротив щелей, максимум в центре.
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

def run(seed, noise=0.02):
    rng=np.random.RandomState(seed)
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=100*dx; s=6.0; k=1.2; A=0.8
    gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
    theta=A*gauss*np.cos(k*(X-x0)) + noise*rng.randn(Ny,Nx)
    theta_t=-c*np.gradient(theta,dx,axis=1) + noise*rng.randn(Ny,Nx)
    env=np.zeros(Ny)
    for t in range(steps):
        a=c**2*lap_cut(theta)-np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
        env=np.maximum(env,np.abs(theta_t[:,screen_x]))
    return env

print("=== Двухщелевой: 5 seeds с шумом поля 0.02 ===")
print(f"{'seed':>5} {'y_центр_пик':>11} {'y80(провал)':>11} {'y120(провал)':>12} {'центр/провал':>12}")
stats=[]
for seed in range(5):
    env=run(seed)
    inner=slice(damp_width+3,Ny-damp_width-3)
    ys=np.arange(Ny)
    ymax=ys[inner][np.argmax(env[inner])]
    contrast=env[100]/max((env[80]+env[120])/2,1e-9)
    stats.append((ymax,env[80],env[120],env[100],contrast))
    print(f"{seed:>5} {ymax:>11} {env[80]:>11.4f} {env[120]:>12.4f} {contrast:>12.1f}")
ymaxes=[s[0] for s in stats]; contrasts=[s[4] for s in stats]
print()
print(f"центральный пик: y = {np.mean(ymaxes):.1f} ± {np.std(ymaxes):.1f}")
print(f"контраст центр/провалы: {np.mean(contrasts):.1f} ± {np.std(contrasts):.1f}")
print("Узор устойчив, если пик стабильно ~100 и контраст >>1 на всех seeds")
