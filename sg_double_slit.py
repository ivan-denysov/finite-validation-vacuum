"""
(3-SG.2) 2D sine-Gordon: фронт -> барьер с 2 щелями -> экран. Есть ли узор?
theta_tt = c^2 (theta_xx+theta_yy) - sin(theta), барьер = узлы где theta заморожена (=0).
"""
import numpy as np

Nx,Ny=300,200; dx=0.5; c=1.0; dt=0.2*dx/c
steps=1100
barrier_x=120; screen_x=260
slit_y1,slit_y2=80,120; halfw=4

barrier=np.zeros((Ny,Nx),bool)
for y in range(Ny):
    if not (abs(y-slit_y1)<=halfw or abs(y-slit_y2)<=halfw):
        barrier[y,barrier_x]=True
        barrier[y,barrier_x+1]=True

theta=np.zeros((Ny,Nx)); theta_t=np.zeros((Ny,Nx))
# плоский kink-фронт слева, бежит вправо
v=0.5; g=1/np.sqrt(1-v**2); x0=40
X=np.arange(Nx)*dx
prof=4*np.arctan(np.exp((X-x0*dx)*g))
theta[:]=prof[None,:]
theta_t[:]=(-v*g*2/np.cosh((X-x0*dx)*g))[None,:]

def lap(th):
    return (np.roll(th,1,1)-2*th+np.roll(th,-1,1))/dx**2 + (np.roll(th,1,0)-2*th+np.roll(th,-1,0))/dx**2

env_max=np.zeros(Ny)  # огибающая |theta_t| на экране (энергия прихода)
for t in range(steps):
    a=c**2*lap(theta)-np.sin(theta)
    theta_t=theta_t+dt*a
    theta=theta+dt*theta_t
    theta[barrier]=0.0; theta_t[barrier]=0.0
    theta[:,0]=theta[:,1]; theta[:,-1]=theta[:,-2]
    theta[0,:]=theta[1,:]; theta[-1,:]=theta[-2,:]
    env_max=np.maximum(env_max, np.abs(theta_t[:,screen_x]))

print("=== экран: огибающая |theta_t| по y ===")
print(f"max={env_max.max():.3f}, mean={env_max.mean():.3f}")
sm=np.convolve(env_max,np.ones(5)/5,mode='same')
peaks=[(y,round(sm[y],3)) for y in range(3,Ny-3)
       if sm[y]>sm[y-1] and sm[y]>sm[y+1] and sm[y]>0.35*sm.max()]
print(f"пики (y, амп), порог 35% max: {peaks}")
print(f"число пиков: {len(peaks)}")
print(f"под щелью1 y={slit_y1}: {env_max[slit_y1]:.3f}; под щелью2 y={slit_y2}: {env_max[slit_y2]:.3f}; центр y={(slit_y1+slit_y2)//2}: {env_max[(slit_y1+slit_y2)//2]:.3f}")
print()
print("2 пика строго под щелями = частицеподобно/тень; >2 пиков с максимумом ПО ЦЕНТРУ = интерференция")
