"""
(4-финал) РЕШАЮЩИЙ тест: АСИММЕТРИЧНАЯ постановка (щели разной ширины).
Если max|theta_t| и центроид совпадают и тут => совпадение фундаментально.
Если разошлись => развилка реальна, надо выбирать.
"""
import numpy as np

Nx,Ny=300,200; dx=0.5; c=1.0; dt=0.2*dx/c
steps=1500
barrier_x=120; screen_x=260
slit_y1,slit_y2=80,120
w1,w2=2,6   # АСИММЕТРИЯ: щель1 узкая (полуширина 2), щель2 широкая (6)
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
envA=np.zeros(Ny); envT=np.zeros(Ny)
for t in range(steps):
    a=c**2*lap_cut(theta)-np.sin(theta)-damp*theta_t
    theta_t=theta_t+dt*a; theta=theta+dt*theta_t
    envA=np.maximum(envA,np.abs(theta[:,screen_x]))
    envT=np.maximum(envT,np.abs(theta_t[:,screen_x]))

inner=slice(damp_width+3,Ny-damp_width-3)
ys=np.arange(Ny)
ya=ys[inner][np.argmax(envA[inner])]
yt=ys[inner][np.argmax(envT[inner])]
wgt=envT[inner]**2
yc=(ys[inner]*wgt).sum()/wgt.sum()
print("=== АСИММЕТРИЧНЫЕ щели (узкая y80 пш2, широкая y120 пш6) ===")
print(f"argmax|θ| = {ya}")
print(f"argmax|θ_t| (макс. такта) = {yt}")
print(f"центроид (баланс) = {yc:.1f}")
print(f"профиль θ_t у щелей: y80={envT[80]:.4f}  y100={envT[100]:.4f}  y120={envT[120]:.4f}")
sm=np.convolve(envT,np.ones(5)/5,mode='same')
pk=[(y,round(sm[y],3)) for y in range(damp_width+3,Ny-damp_width-3)
    if sm[y]>sm[y-1] and sm[y]>sm[y+1] and sm[y]>0.35*sm.max()]
print(f"пики θ_t: {pk}")
print()
d=abs(yt-yc)
print(f"|max_такта - центроид| = {d:.1f} узлов")
print("=> СОВПАДАЮТ (≤3) — фундаментально; РАЗОШЛИСЬ (>3) — развилка реальна, выбирать.")
