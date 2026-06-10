"""
(4-энергия) Что приводит к точке сборки ФОТОНА: проверка 4-го кандидата —
ПЛОТНОСТЬ ЭНЕРГИИ поля E = theta_t^2/2 + c^2|grad theta|^2/2 + (1-cos theta).
Асимметричные щели (пш 2 vs 6). Сравниваем с прежними: такт y=99, |θ| y=134, центроид y=114.6.
Накапливаем энергию на экране по времени (как фотопластинка копит вспышки).
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

A=0.8
X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
x0=40*dx; y0=100*dx; s=6.0; k=1.2
gauss=np.exp(-(((X-x0)**2+(Y-y0)**2))/(2*(s*dx*2)**2))
theta=A*gauss*np.cos(k*(X-x0))
theta_t=-c*np.gradient(theta,dx,axis=1)

E_accum=np.zeros(Ny)      # накопленная энергия на экране (фотопластинка)
E_peak =np.zeros(Ny)      # пиковая мгновенная энергия
for t in range(steps):
    a=c**2*lap_cut(theta)-np.sin(theta)-damp*theta_t
    theta_t=theta_t+dt*a; theta=theta+dt*theta_t
    col=theta[:,screen_x]
    gy=np.gradient(theta[:,screen_x-1:screen_x+2],dx,axis=1)[:,1]  # d/dx на экране
    gyy=np.gradient(col,dx)                                        # d/dy на экране
    Edens=theta_t[:,screen_x]**2/2 + c**2*(gy**2+gyy**2)/2 + (1-np.cos(col))
    E_accum+=Edens*dt
    E_peak=np.maximum(E_peak,Edens)

inner=slice(damp_width+3,Ny-damp_width-3)
ys=np.arange(Ny)
ya=ys[inner][np.argmax(E_accum[inner])]
yp=ys[inner][np.argmax(E_peak[inner])]
wgt=E_accum[inner]
yc=(ys[inner]*wgt).sum()/wgt.sum()
print("=== 4-й кандидат: ПЛОТНОСТЬ ЭНЕРГИИ (асимметричные щели) ===")
print(f"argmax накопленной энергии (фотопластинка): y={ya}")
print(f"argmax пиковой энергии: y={yp}")
print(f"центроид энергии: y={yc:.1f}")
print()
print("Прежние кандидаты: такт(|θ_t|max)=99, |θ|max=134, центроид(θ_t²)=114.6")
print()
sm=np.convolve(E_accum,np.ones(5)/5,mode='same')
pk=[(y,round(sm[y],3)) for y in range(damp_width+3,Ny-damp_width-3)
    if sm[y]>sm[y-1] and sm[y]>sm[y+1] and sm[y]>0.3*sm[inner].max()]
print(f"профиль накопл. энергии — пики: {pk}")
print(f"E_accum: y80={E_accum[80]:.4f} y99={E_accum[99]:.4f} y114={E_accum[114]:.4f} y120={E_accum[120]:.4f} y134={E_accum[134]:.4f}")
