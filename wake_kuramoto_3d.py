"""
3D тест: оставляет ли движущийся возмутитель КОНУС-след (3D кильватер)
в решётке Курамото. Метрика = разница с контролем (та же сеть без возмутителя).
"""
import numpy as np

N = 40                    # 40^3 = 64000 узлов
K = 0.6
omega0 = 1.0
dt = 0.05
steps = 700
drive_freq = 3.0
speed = 0.04              # узлов за шаг по оси x
seed = 42

rng = np.random.RandomState(seed)
theta0 = rng.uniform(0, 2*np.pi, (N,N,N))
cy, cz = N//2, N//2       # возмутитель едет вдоль x по центральной оси

def coupling(th):
    c  = np.sin(np.roll(th,1,0)-th); c += np.sin(np.roll(th,-1,0)-th)
    c += np.sin(np.roll(th,1,1)-th); c += np.sin(np.roll(th,-1,1)-th)
    c += np.sin(np.roll(th,1,2)-th); c += np.sin(np.roll(th,-1,2)-th)
    return c

# контроль
th = theta0.copy()
for t in range(steps):
    th = th + dt*(omega0 + (K/6.0)*coupling(th))
control = th.copy()

# с возмутителем
th = theta0.copy()
for t in range(steps):
    dx = max(1, min(N-2, int(5 + speed*t)))
    om = np.full((N,N,N), omega0)
    om[dx, cy, cz] = drive_freq
    th = th + dt*(om + (K/6.0)*coupling(th))
perturbed = th.copy()
final_x = max(1, min(N-2, int(5 + speed*(steps-1))))

diff = np.abs(np.angle(np.exp(1j*(perturbed-control))))  # чистое возмущение

print(f"финальная позиция тела: x={final_x}")
print(f"среднее возмущение по решётке: {diff.mean():.5f}")
print(f"возмущение на оси пути (line cy,cz): {diff[:,cy,cz].mean():.5f}")
print(f"возмущение вдали от оси (угол): {diff[:,2,2].mean():.5f}")
print()

# КОНУС: радиус возмущённой зоны на разных x ЗА телом
print("=== Радиус следа по сечениям x (за телом) — расходится ли конус? ===")
print(f"{'x':>4} {'radius(узлов>порог)':>20} {'max_dev':>9}")
thr = 0.05
for x in range(8, final_x, max(1,(final_x-8)//8)):
    plane = diff[x,:,:]                    # сечение поперёк пути
    # радиус: среднее расстояние возмущённых узлов от центра (cy,cz)
    mask = plane > thr
    if mask.sum()>0:
        ys,zs = np.where(mask)
        r = np.sqrt((ys-cy)**2+(zs-cz)**2).mean()
    else:
        r = 0.0
    print(f"{x:>4} {mask.sum():>12}        r={r:>5.2f}  {plane.max():>7.3f}")
print()
# сравнение радиуса у начала пути vs у тела
def radius_at(x):
    plane=diff[x,:,:]; m=plane>thr
    if m.sum()==0: return 0,0
    ys,zs=np.where(m); return m.sum(), np.sqrt((ys-cy)**2+(zs-cz)**2).mean()
n1,r1 = radius_at(10)
n2,r2 = radius_at(final_x-3)
print(f"у начала пути (x=10):   {n1} узлов, радиус {r1:.2f}")
print(f"ближе к телу (x={final_x-3}): {n2} узлов, радиус {r2:.2f}")
print(f"=> {'КОНУС расходится к телу' if r2>r1+0.5 else 'НЕ расходится' if abs(r2-r1)<=0.5 else 'сужается к телу'}")
