"""
(3.1) Самораспространение: с ФАЗОВОЙ ИНЕРЦИЕЙ (2-й порядок) бежит ли возмущение
само, оторвавшись от источника, и доходит ли до экрана?
Уравнение: m*theta_ddot + gamma*theta_dot = omega + K*coupling
m -> инерция (память), gamma -> затухание. При m>0 фаза несёт импульс => волна бежит.
Тест БЕЗ щелей: импульс в источнике на короткое время, потом источник убираем,
смотрим — волна доходит до экрана сама?
"""
import numpy as np

N=120; K=0.8; omega0=1.0; dt=0.04; steps=1400
src_x, src_y = 10, N//2
screen_x=N-8

def coupling(th):
    c=np.zeros_like(th)
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        c+=np.sin(np.roll(th,sh,ax)-th)
    return c

def run(m, gamma, pulse_until=120, drive=4.0):
    rng=np.random.RandomState(42)
    theta=rng.uniform(0,2*np.pi,(N,N))
    vel=np.zeros((N,N))            # theta_dot
    # контроль без импульса
    th=theta.copy(); v=vel.copy()
    for t in range(steps):
        acc=(omega0 + (K/4.0)*coupling(th) - gamma*v)/m
        v=v+dt*acc; th=th+dt*v
    ctrl=th.copy()
    # с импульсом (источник толкает короткое время, потом отпускаем)
    th=theta.copy(); v=vel.copy()
    front=[]
    for t in range(steps):
        om=np.full((N,N),omega0)
        if t<pulse_until:
            om[src_y,src_x]=drive
        acc=(om + (K/4.0)*coupling(th) - gamma*v)/m
        v=v+dt*acc; th=th+dt*v
        if t%200==0 or t==steps-1:
            diff=np.abs(np.angle(np.exp(1j*(th-ctrl))))
            # фронт = самый дальний x по центральной строке с возмущением>0.05
            row=diff[src_y,:]
            xs=np.where(row>0.05)[0]
            front.append((t, xs.max() if len(xs) else 0, round(diff[src_y,screen_x],3)))
    diff=np.abs(np.angle(np.exp(1j*(th-ctrl))))
    return diff, front

print("=== Бежит ли волна сама и доходит до экрана (x=%d)? ===" % screen_x)
for m,gamma in [(1.0,1.0),(3.0,0.5),(5.0,0.2),(8.0,0.1)]:
    diff,front=run(m,gamma)
    reached=diff[src_y,screen_x]
    maxfront=max(f[1] for f in front)
    print(f"m={m}, gamma={gamma}: фронт дошёл до x={maxfront}, на экране={reached:.3f}")
    print(f"   динамика фронта (t,x_front,экран): {front}")
print()
print("Нужно: фронт доходит до screen_x И возмущение на экране >0 ПОСЛЕ снятия источника")
print("=> волна самораспространяется (условие для двухщелевого)")
