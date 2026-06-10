"""
(3-SG.6) 'Частица' = ЛОКАЛИЗОВАННЫЙ бегущий гауссов пакет (нетопологический).
theta(x,y,0)=A*exp(-((x-x0)^2+(y-y0)^2)/2s^2)*cos(k(x-x0)); theta_t = +c*d theta/dx (бежит вправо).
Стена = рез связей. Шаг 0: сплошная стена -> за ней ~0 (валидация).
Затем: две щели vs одна, пуск по центру.
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
    damp[:,Nx-1-i]=np.maximum(damp[:,Nx-1-i],w)
    damp[:,i]=np.maximum(damp[:,i],w)

def make_wall(mode):
    m=np.zeros((Ny,Nx))
    for y in range(Ny):
        in1=abs(y-slit_y1)<=halfw
        in2=abs(y-slit_y2)<=halfw and (mode=='two')
        blocked=not(in1 or in2)
        if mode=='solid': blocked=True
        if mode=='none': blocked=False
        if blocked: m[y,barrier_x]=1; m[y,barrier_x+1]=1
    return m

def lap_cut(th,wall):
    tot=np.zeros_like(th)
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        nb=np.roll(th,sh,ax); wb=np.roll(wall,sh,ax)
        link=1.0-np.maximum(wall,wb)
        tot+=link*(nb-th)
    return tot/dx**2

def run(mode,y_launch=100,A=0.8,s=6.0,k=1.2):
    wall=make_wall(mode)
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=y_launch*dx
    R2=((X-x0)**2+(Y-y0)**2)
    gauss=np.exp(-R2/(2*(s*dx*2)**2))
    theta=A*gauss*np.cos(k*(X-x0))
    # бежит вправо: theta_t = -c * d/dx theta (волна f(x-ct))
    dthdx=np.gradient(theta,dx,axis=1)
    theta_t=-c*dthdx
    env=np.zeros(Ny)
    for t in range(steps):
        a=c**2*lap_cut(theta,wall)-np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
        env=np.maximum(env,np.abs(theta_t[:,screen_x]))
    E_right=(theta_t[:,barrier_x+5:]**2).sum()+ (np.gradient(theta[:,barrier_x+5:],dx,axis=1)**2).sum()
    return env,E_right

_,E_solid=run('solid')
_,E_none=run('none')
print(f"ВАЛИДАЦИЯ: сплошная стена E_right={E_solid:.3f}; без стены E_right={E_none:.3f}")
print(f"(нужно: solid << none; иначе стена не работает)")
print()
env2,E2=run('two'); env1,E1=run('one')
def report(env,label):
    sm=np.convolve(env,np.ones(5)/5,mode='same')
    pk=[(y,round(sm[y],3)) for y in range(damp_width+3,Ny-damp_width-3)
        if sm[y]>sm[y-1] and sm[y]>sm[y+1] and sm[y]>0.35*sm.max()]
    print(f"{label}: max={env.max():.4f}; пики {pk}")
    print(f"   y80={env[80]:.4f} y120={env[120]:.4f} y100={env[100]:.4f}")
report(env2,"ДВЕ щели")
report(env1,"ОДНА щель (y=80)")
print(f"E_right: two={E2:.3f}, one={E1:.3f}, solid={E_solid:.3f}")
