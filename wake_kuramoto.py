"""
Тест: оставляет ли движущийся возмутитель ШЛЕЙФ в решётке Курамото,
и какой он формы (воронка/клин или гаснет на месте).

Модель:
- 2D решётка фазовых осцилляторов (Курамото, связь с соседями).
- Среднее поле тянет к общему ритму omega0.
- Возмутитель = узел с ПОВЫШЕННОЙ собственной частотой (быстрее поля = "энергия"),
  движется по решётке слева направо.
- Снимаем шлейф = |отклонение фазы узла от среднего поля| после прохода.
"""
import numpy as np

np.random.seed(42)
N = 80                      # решётка N x N
K = 1.2                     # сила связи соседей
omega0 = 1.0                # средний ритм поля
dt = 0.05
steps = 1600

# фазы
theta = np.random.uniform(0, 2*np.pi, (N,N))
omega = np.full((N,N), omega0)   # все узлы на среднем ритме

# возмутитель: повышенная частота, движется по средней строке
drive_freq = 3.0            # быстрее поля = "энергия"
y0 = N//2
def driver_x(t):            # позиция возмутителя по x во времени
    return int(5 + (N-10) * (t/steps))

def laplacian_coupling(th):
    # связь Курамото с 4 соседями: sum sin(theta_neighbor - theta)
    c = np.zeros_like(th)
    c += np.sin(np.roll(th,1,0)-th)
    c += np.sin(np.roll(th,-1,0)-th)
    c += np.sin(np.roll(th,1,1)-th)
    c += np.sin(np.roll(th,-1,1)-th)
    return c

wake_accum = np.zeros((N,N))   # накопленный шлейф
for t in range(steps):
    dx = driver_x(t)
    om = omega.copy()
    om[y0, dx] = drive_freq    # возмутитель едет
    theta = theta + dt*(om + (K/4.0)*laplacian_coupling(theta))
    # среднее поле (глобальная фаза)
    mean_phase = np.angle(np.mean(np.exp(1j*theta)))
    # локальное отклонение от среднего поля
    dev = np.abs(np.angle(np.exp(1j*(theta-mean_phase))))
    if t > steps//2:           # копим во второй половине (устаканилось)
        wake_accum += dev

wake = wake_accum/(steps - steps//2)

# Анализ формы: профиль отклонения ВДОЛЬ пути за возмутителем и ПОПЕРЁК
final_x = driver_x(steps-1)
print("=== Есть ли шлейф? ===")
print(f"среднее отклонение по решётке: {wake.mean():.4f}")
print(f"отклонение в зоне пути (строка y0): {wake[y0,:].mean():.4f}")
print(f"отклонение вдали от пути (строка 5): {wake[5,:].mean():.4f}")
ratio = wake[y0,:].mean()/max(wake[5,:].mean(),1e-9)
print(f"отношение путь/вдали: {ratio:.2f}  (>1 => шлейф есть вдоль пути)")
print()
# поперечный профиль ЗА возмутителем (слева от него) vs ПЕРЕД
behind = wake[:, 10:final_x-5].mean(axis=1)   # за телом
ahead  = wake[:, final_x+3:final_x+10].mean(axis=1) if final_x+10<N else wake[:, -8:].mean(axis=1)
print("=== Форма: за телом vs перед телом (поперечный профиль по y) ===")
print(f"за телом, пик отклонения на y={np.argmax(behind)} (центр пути y0={y0})")
print(f"за телом, ширина (узлов выше полумакс): {np.sum(behind>behind.max()/2)}")
print(f"перед телом, ширина: {np.sum(ahead>max(ahead.max(),1e-9)/2)}")
print()
print("=== Расходится ли шлейф (воронка)? ===")
# ширина шлейфа в начале пути vs в конце (за телом)
seg1 = wake[:, 12:22].mean(axis=1)   # ранний участок пути
seg2 = wake[:, final_x-25:final_x-15].mean(axis=1)  # поздний участок (ближе к телу)
w1 = np.sum(seg1>seg1.max()/2); w2 = np.sum(seg2>seg2.max()/2)
print(f"ширина шлейфа в начале пути: {w1} узлов")
print(f"ширина шлейфа ближе к телу:  {w2} узлов")
print(f"=> {'РАСХОДИТСЯ (воронка)' if w2>w1+1 else 'НЕ расходится, ~постоянная ширина' if abs(w2-w1)<=1 else 'СУЖАЕТСЯ'}")
