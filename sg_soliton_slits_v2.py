"""
(3-SG.5) Исправленный барьер: РЕЗ СВЯЗЕЙ в лапласиане (через стену передачи нет).
Проверка стены: энергия за барьером при сплошной стене должна быть ~0.
Потом: пакет по центру, две щели vs одна.
"""
import numpy as np

Nx,Ny=300,200; dx=0.5; c=1.0; dt=0.2*dx/c
steps=1400
barrier_x=120; screen_x=260
slit_y1,slit_y2=80,120; halfw=4
damp_width=15

damp=np.zeros((Ny,Nx))
for i in range(damp_width):
    w=(1-i/damp_width)*0.5
    damp[i,:]=np.maximum(damp[i,:],w); damp[Ny-1-i,:]=np.maximum(damp[Ny-1-i,:],w)
    damp[:,Nx-1-i]=np.maximum(damp[:,Nx-1-i],w)

def make_wallmask(mode):
    # 1 в зоне стены
    m=np.zeros((Ny,Nx))
    for y in range(Ny):
        in1=abs(y-slit_y1)<=halfw
        in2=abs(y-slit_y2)<=halfw and (mode=='two')
        blocked = not(in1 or (in2))
        if mode=='solid': blocked=True
        if blocked:
            m[y,barrier_x]=1; m[y,barrier_x+1]=1
    return m

def lap_cut(th, wall):
    # связь рвётся, если ЛЮБОЙ конец в стене
    tot=np.zeros_like(th)
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        nb=np.roll(th,sh,ax); wb=np.roll(wall,sh,ax)
        link=1.0-np.maximum(wall,wb)
        tot+=link*(nb-th)
    return tot/dx**2

def run(mode, y_launch=100):
    wall=make_wallmask(mode)
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=y_launch*dx; v=0.5; g=1/np.sqrt(1-v**2)
    envy=np.exp(-((Y-y0)**2)/(2*(8*dx)**2))
    theta=4*np.arctan(np.exp((X-x0)*g))*envy
    theta_t=(-v*g*2/np.cosh((X-x0)*g))*envy
    env=np.zeros(Ny)
    for t in range(steps):
        a=c**2*lap_cut(theta,wall)-np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
        theta[:,0]=theta[:,1]
        env=np.maximum(env,np.abs(theta_t[:,screen_x]))
    E_right=(theta_t[:,barrier_x+5:]**2).sum()
    return env,E_right

# 0) проверка стены: сплошная
_,E_solid=run('solid')
print(f"ПРОВЕРКА СТЕНЫ (сплошная): энергия за барьером = {E_solid:.4f} (должна быть ~0)")
env2,E2=run('two'); env1,E1=run('one')
def report(env,label):
    sm=np.convolve(env,np.ones(5)/5,mode='same')
    pk=[(y,round(sm[y],3)) for y in range(damp_width+3,Ny-damp_width-3)
        if sm[y]>sm[y-1] and sm[y]>sm[y+1] and sm[y]>0.35*sm.max()]
    print(f"{label}: max={env.max():.3f}; пики {pk}")
    print(f"   y80={env[80]:.3f} y120={env[120]:.3f} y100={env[100]:.3f}")
print()
report(env2,"ДВЕ щели")
report(env1,"ОДНА щель (y=80)")
print(f"энергия за барьером: solid={E_solid:.2f}, two={E2:.2f}, one={E1:.2f}")
