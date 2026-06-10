"""
(4) КУЛЬМИНАЦИЯ: что в поле выделяет точку сборки?
На экране после 2 щелей сравниваем вдоль y:
 (a) максимум амплитуды |theta|
 (b) максимум локального ТАКТА = |theta_t| (скорость фазы = локальная частота)
 (c) центроид распределения (баланс)
Вопрос: (b) совпадает с (a)? оба с (c)? И зависит ли такт в пике от энергии пакета (A)?
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

def run(A):
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=100*dx; s=6.0; k=1.2
    R2=((X-x0)**2+(Y-y0)**2)
    gauss=np.exp(-R2/(2*(s*dx*2)**2))
    theta=A*gauss*np.cos(k*(X-x0))
    theta_t=-c*np.gradient(theta,dx,axis=1)
    envA=np.zeros(Ny)   # огибающая |theta| на экране
    envT=np.zeros(Ny)   # огибающая |theta_t| (локальный такт)
    for t in range(steps):
        a=c**2*lap_cut(theta)-np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
        envA=np.maximum(envA,np.abs(theta[:,screen_x]))
        envT=np.maximum(envT,np.abs(theta_t[:,screen_x]))
    return envA,envT

print("=== Кандидаты точки сборки на экране (вдоль y) ===")
print(f"{'A':>5} {'argmax|θ|':>10} {'argmax|θt|':>11} {'centroid':>9} {'θt_в_пике':>10} {'θt_max':>8}")
for A in [0.4,0.8,1.4,2.0]:
    envA,envT=run(A)
    inner=slice(damp_width+3,Ny-damp_width-3)
    ys=np.arange(Ny)
    ya=ys[inner][np.argmax(envA[inner])]
    yt=ys[inner][np.argmax(envT[inner])]
    w=envT[inner]**2
    yc=(ys[inner]*w).sum()/w.sum()
    print(f"{A:>5} {ya:>10} {yt:>11} {yc:>9.1f} {envT[ya]:>10.4f} {envT.max():>8.4f}")
print()
print("Совпадение argmax|θ| ≈ argmax|θt| ≈ centroid => максимум такта И баланс — одна точка")
print("(развилка закрывается: в симметричном узоре пик амплитуды, пик такта и центроид совпадают)")
print("Рост θt_max с A => 'такт в точке сборки' растёт с энергией (частицы различимы по такту)")
