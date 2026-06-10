"""
(3-SG.3) Контроль: ОДНА щель vs ДВЕ щели, с ПОГЛОЩАЮЩИМИ краями (демпфер у границ),
чтобы убрать отражения. Если узор двух щелей != узора одной => интерференция реальна.
"""
import numpy as np

Nx,Ny=300,200; dx=0.5; c=1.0; dt=0.2*dx/c
steps=1100
barrier_x=120; screen_x=260
slit_y1,slit_y2=80,120; halfw=4
damp_width=15  # поглощающий слой у границ

def make_barrier(two_slits):
    b=np.zeros((Ny,Nx),bool)
    for y in range(Ny):
        in1=abs(y-slit_y1)<=halfw
        in2=abs(y-slit_y2)<=halfw and two_slits
        if not (in1 or in2):
            b[y,barrier_x]=True; b[y,barrier_x+1]=True
    return b

# демпфирующая маска: gamma растёт к границам по y и к правому краю
damp=np.zeros((Ny,Nx))
for i in range(damp_width):
    w=(1-i/damp_width)*0.5
    damp[i,:]=np.maximum(damp[i,:],w); damp[Ny-1-i,:]=np.maximum(damp[Ny-1-i,:],w)
    damp[:,Nx-1-i]=np.maximum(damp[:,Nx-1-i],w)

def lap(th):
    return (np.roll(th,1,1)-2*th+np.roll(th,-1,1))/dx**2 + (np.roll(th,1,0)-2*th+np.roll(th,-1,0))/dx**2

def run(two_slits):
    barrier=make_barrier(two_slits)
    theta=np.zeros((Ny,Nx)); theta_t=np.zeros((Ny,Nx))
    v=0.5; g=1/np.sqrt(1-v**2); x0=40
    X=np.arange(Nx)*dx
    theta[:]= (4*np.arctan(np.exp((X-x0*dx)*g)))[None,:]
    theta_t[:]=(-v*g*2/np.cosh((X-x0*dx)*g))[None,:]
    env=np.zeros(Ny)
    for t in range(steps):
        a=c**2*lap(theta)-np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a
        theta=theta+dt*theta_t
        theta[barrier]=0.0; theta_t[barrier]=0.0
        theta[:,0]=theta[:,1]
        env=np.maximum(env,np.abs(theta_t[:,screen_x]))
    return env

env2=run(True); env1=run(False)
def peaks_of(env):
    sm=np.convolve(env,np.ones(5)/5,mode='same')
    return [(y,round(sm[y],3)) for y in range(damp_width+3,Ny-damp_width-3)
            if sm[y]>sm[y-1] and sm[y]>sm[y+1] and sm[y]>0.35*sm.max()]

p2=peaks_of(env2); p1=peaks_of(env1)
print("=== ДВЕ щели (поглощающие края) ===")
print(f"max={env2.max():.3f}; пики: {p2}")
print(f"под щелями: y80={env2[80]:.3f} y120={env2[120]:.3f}; центр y100={env2[100]:.3f}")
print()
print("=== ОДНА щель (та же, y=80) ===")
print(f"max={env1.max():.3f}; пики: {p1}")
print(f"под щелью y80={env1[80]:.3f}; y120={env1[120]:.3f}; центр y100={env1[100]:.3f}")
print()
print(f"число пиков: две щели={len(p2)}, одна={len(p1)}")
print("Интерференция реальна, если: у двух щелей БОЛЬШЕ пиков/иная структура, чем у одной,")
print("и структура двух щелей симметрична относительно центра между щелями (y=100).")
