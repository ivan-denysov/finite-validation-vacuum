"""
Тест v2: шлейф рождается из КОНЕЧНОГО ТЕМПА РЕСИНХРОНИЗАЦИИ.
Аналогия воды: среда не успевает сомкнуться за телом, если тело быстрее отклика.

Ключевое изменение против v1:
- метрика = возмущение ОТНОСИТЕЛЬНО КОНТРОЛЯ (та же сеть без возмутителя),
  убирает шум фона.
- сканируем СКОРОСТЬ возмутителя относительно темпа ресинхронизации.
- ожидание: медленный возмутитель -> сеть смыкается -> следа нет;
  быстрый -> шлейф остаётся (как кильватер).
"""
import numpy as np

def run(driver_speed, K=0.6, N=80, steps=1600, drive_freq=3.0, seed=42):
    rng = np.random.RandomState(seed)
    theta0 = rng.uniform(0, 2*np.pi, (N,N))
    omega0 = 1.0
    dt = 0.05
    y0 = N//2

    def coupling(th):
        c  = np.sin(np.roll(th,1,0)-th); c += np.sin(np.roll(th,-1,0)-th)
        c += np.sin(np.roll(th,1,1)-th); c += np.sin(np.roll(th,-1,1)-th)
        return c

    # КОНТРОЛЬ: без возмутителя
    th = theta0.copy()
    for t in range(steps):
        th = th + dt*(omega0 + (K/4.0)*coupling(th))
    control = th.copy()

    # С ВОЗМУТИТЕЛЕМ
    th = theta0.copy()
    path_x = []
    for t in range(steps):
        dx = int(5 + driver_speed * t) % (N-1)
        dx = max(1, min(N-2, int(5 + driver_speed*t)))
        if 5 + driver_speed*t >= N-2:
            dx = N-2
        om = np.full((N,N), omega0)
        om[y0, dx] = drive_freq
        th = th + dt*(om + (K/4.0)*coupling(th))
        path_x.append(dx)
    perturbed = th.copy()

    # чистое возмущение = разница с контролем (по фазе)
    diff = np.abs(np.angle(np.exp(1j*(perturbed-control))))
    final_x = path_x[-1]
    # шлейф ЗА телом (слева) vs фон вдали по y
    behind = diff[:, 10:max(11,final_x-5)]
    trail_strength = behind[y0-3:y0+4,:].mean()      # у линии пути
    bg = diff[5:12,:].mean()                          # вдали от пути
    return trail_strength, bg, diff, y0, final_x

print("=== Скан скорости возмутителя (шлейф появляется выше порога?) ===")
print(f"{'speed':>6} {'trail':>8} {'bg':>8} {'trail/bg':>9}")
for sp in [0.005, 0.01, 0.02, 0.04, 0.08]:
    trail, bg, *_ = run(sp)
    print(f"{sp:>6} {trail:>8.4f} {bg:>8.4f} {trail/max(bg,1e-9):>9.2f}")
print()
print("trail/bg > 1 => есть шлейф вдоль пути; растёт со скоростью => эффект кильватера")
