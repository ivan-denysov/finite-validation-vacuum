"""
(3-SG.4) САМ локализованный импульс ("частица") на 2 щели.
2D sine-Gordon. Начальное условие: локализованный по x И по y пакет (гауссова огибающая),
летит вправо на барьер с 2 щелями. Поглощающие края.
Вопросы: (а) проходит ли через щели; (б) одна точка на экране или распределение;
(в) отличается ли распределение от случая 1 щели.
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

def make_barrier(two):
    b=np.zeros((Ny,Nx),bool)
    for y in range(Ny):
        in1=abs(y-slit_y1)<=halfw
        in2=abs(y-slit_y2)<=halfw and two
        if not(in1 or in2):
            b[y,barrier_x]=True; b[y,barrier_x+1]=True
    return b

def lap(th):
    return (np.roll(th,1,1)-2*th+np.roll(th,-1,1))/dx**2+(np.roll(th,1,0)-2*th+np.roll(th,-1,0))/dx**2

def run(two, y_launch):
    barrier=make_barrier(two)
    X=(np.arange(Nx)*dx)[None,:]; Y=(np.arange(Ny)*dx)[:,None]
    x0=40*dx; y0=y_launch*dx; v=0.5; g=1/np.sqrt(1-v**2)
    # локализованный пакет: kink-профиль по x, гауссова огибающая по y
    envy=np.exp(-((Y-y0)**2)/(2*(8*dx)**2))
    theta=4*np.arctan(np.exp((X-x0)*g))*envy
    theta_t=(-v*g*2/np.cosh((X-x0)*g))*envy
    env=np.zeros(Ny); passed=0.0
    for t in range(steps):
        a=c**2*lap(theta)-np.sin(theta)-damp*theta_t
        theta_t=theta_t+dt*a; theta=theta+dt*theta_t
        theta[barrier]=0.0; theta_t[barrier]=0.0
        theta[:,0]=theta[:,1]
        env=np.maximum(env,np.abs(theta_t[:,screen_x]))
    # сколько энергии прошло за барьер (грубая мера)
    E_right=(theta_t[:,barrier_x+5:]**2).sum()
    return env,E_right

print("=== 'Частица' (локализованный пакет) на щели ===")
# запуск ПО ЦЕНТРУ между щелями (y=100): пакет видит ОБЕ щели
env2,E2=run(True ,100)
env1,E1=run(False,100)
def report(env,label):
    sm=np.convolve(env,np.ones(5)/5,mode='same')
    pk=[(y,round(sm[y],3)) for y in range(damp_width+3,Ny-damp_width-3)
        if sm[y]>sm[y-1] and sm[y]>sm[y+1] and sm[y]>0.35*sm.max()]
    print(f"{label}: max={env.max():.3f}; пики {pk}")
    print(f"   y80={env[80]:.3f} y120={env[120]:.3f} y100={env[100]:.3f}")
report(env2,"ДВЕ щели, пуск по центру (видит обе)")
report(env1,"ОДНА щель, пуск по центру")
print(f"энергия за барьером: две={E2:.2f}, одна={E1:.2f}")
print()
print("Ключ: при двух щелях у пакета, видящего ОБЕ, структура на экране")
print("симметричная/полосатая (волна) или один пик (прошёл как частица в одну)?")
