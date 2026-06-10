"""
(3) Двухщелевой: нить-возмущение -> барьер с двумя щелями -> экран.
Вариант A: щели = проходы в стене СРЕЗАННЫХ связей (барьер рвёт связи, кроме двух окон).
Вариант B: щели = две зоны пониженной задержки (барьер = замедление, окна = норм. такт).
Сетка регулярная 2D для наглядности узора на экране (потом проверим на аморфной).
Смотрим: распределение возмущения вдоль ЭКРАНА (линия x=screen) — 1 полоса / 2 полосы / узор.
"""
import numpy as np

N=120
K=0.6; omega0=1.0; dt=0.05; steps=900; drive=3.0
barrier_x=N//2
screen_x=N-8
slit_y1, slit_y2 = N//2-12, N//2+12
slit_halfwidth=3
src_y=N//2

def coupling_masked(th, mask_break):
    # связь с 4 соседями, но связи через mask_break (стена) обнулены
    c=np.zeros_like(th)
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        nb=np.roll(th,sh,ax)
        link=np.ones_like(th)
        if mask_break is not None:
            br=np.roll(mask_break,sh,ax)
            link=1.0-np.maximum(mask_break,br)  # связь рвётся если любой конец в стене
        c+=link*np.sin(nb-th)
    return c

def run(variant):
    rng=np.random.RandomState(42)
    theta0=rng.uniform(0,2*np.pi,(N,N))
    omega=np.full((N,N),omega0)
    mask_break=np.zeros((N,N))   # 1 = стена (рвёт связи) для A
    slow=np.ones((N,N))          # множитель такта для B
    for y in range(N):
        is_slit = (abs(y-slit_y1)<=slit_halfwidth) or (abs(y-slit_y2)<=slit_halfwidth)
        if not is_slit:
            mask_break[y,barrier_x]=1.0       # A: стена кроме щелей
            slow[y,barrier_x]=0.15            # B: сильное замедление кроме щелей
    def ctrl_and_pert(use_break,use_slow):
        # контроль
        th=theta0.copy()
        for t in range(steps):
            mb=mask_break if use_break else None
            sm=slow if use_slow else 1.0
            th=th+dt*(omega*sm+(K/4.0)*coupling_masked(th,mb))
        ctrl=th.copy()
        # возмущение: источник слева, ведёт нить к барьеру
        th=theta0.copy()
        for t in range(steps):
            mb=mask_break if use_break else None
            sm=slow if use_slow else 1.0
            om=omega.copy()
            sx=min(barrier_x-2, 4+int(0.05*t))
            om[src_y, sx]=drive
            th=th+dt*(om*sm+(K/4.0)*coupling_masked(th,mb))
        return np.abs(np.angle(np.exp(1j*(th-ctrl))))
    if variant=='A':
        return ctrl_and_pert(True,False)
    else:
        return ctrl_and_pert(False,True)

for variant in ['A','B']:
    diff=run(variant)
    screen=diff[:,screen_x]
    # сглаживание
    sm=np.convolve(screen,np.ones(3)/3,mode='same')
    peaks=[]
    for y in range(2,N-2):
        if sm[y]>sm[y-1] and sm[y]>sm[y+1] and sm[y]>sm.max()*0.3:
            peaks.append((y,round(sm[y],3)))
    print(f"=== Вариант {variant} ===")
    print(f"возмущение на экране: max={screen.max():.3f}, среднее={screen.mean():.4f}")
    print(f"пики на экране (y,амплитуда): {peaks}")
    print(f"число пиков выше 0.3*max: {len(peaks)}")
    # за щелями: ровно под щелями (геом. проекция) или между (интерференция к центру)?
    print(f"амплитуда под щелью 1 (y={slit_y1}): {screen[slit_y1]:.3f}")
    print(f"амплитуда под щелью 2 (y={slit_y2}): {screen[slit_y2]:.3f}")
    print(f"амплитуда по центру (y={src_y}): {screen[src_y]:.3f}")
    print()
