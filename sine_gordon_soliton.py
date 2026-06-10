"""
(3-SG.1) sine-Gordon: бежит ли солитон устойчиво и сохраняет форму?
Уравнение: theta_tt - c^2 theta_xx + sin(theta) = 0
Солитон (kink): theta(x,t) = 4*arctan(exp( (x - v t)/sqrt(1-v^2) ))
- частицеподобное устойчивое решение, имеет "массу", движется со скоростью v.
Тест 1D: запускаем kink со скоростью v, проверяем что он бежит, не разваливается,
скорость сохраняется.
"""
import numpy as np

L=400; dx=0.5; Nx=int(L/dx); c=1.0; dt=0.2*dx/c  # CFL
steps=2000
x=np.arange(Nx)*dx

def kink(x, x0, v):
    g=1.0/np.sqrt(1-v**2)
    return 4*np.arctan(np.exp((x-x0)*g))

def kink_dot(x,x0,v):
    g=1.0/np.sqrt(1-v**2)
    arg=(x-x0)*g
    # d/dt theta = -v*g * sech(arg)*... ; точная производная kink по t
    return -v*g*2/np.cosh(arg)

v=0.4
theta=kink(x, 60, v)
theta_t=kink_dot(x,60,v)

def laplacian(th):
    return (np.roll(th,1)-2*th+np.roll(th,-1))/dx**2

# центр кинка = где theta пересекает 2pi (середина перепада 0->2pi)
def center(th):
    target=np.pi
    idx=np.argmin(np.abs(th-target))
    return x[idx]

positions=[]
theta_tt=c**2*laplacian(theta)-np.sin(theta)
for t in range(steps):
    theta_tt=c**2*laplacian(theta)-np.sin(theta)
    theta_t=theta_t+dt*theta_tt
    theta=theta+dt*theta_t
    # фиксируем границы (kink уходит — держим края)
    theta[0]=theta[1]; theta[-1]=theta[-2]
    if t%400==0 or t==steps-1:
        positions.append((t, round(center(theta),1)))

print("=== sine-Gordon kink: бежит ли устойчиво? ===")
print(f"начальная скорость v={v}, ожидаемое смещение за прогон ~ {v*c*steps*dt:.1f}")
print(f"позиция центра кинка (t, x): {positions}")
# скорость из смещения
if len(positions)>=2:
    (t1,x1),(t2,x2)=positions[0],positions[-1]
    measured_v=(x2-x1)/((t2-t1)*dt)/c
    print(f"измеренная скорость: {measured_v:.3f} c  (задавали {v})")
# форма сохранилась? сравним профиль с теоретическим кинком в новой позиции
final_center=center(theta)
theory=kink(x, final_center, v)
shape_err=np.mean(np.abs(np.angle(np.exp(1j*(theta-theory)))))
print(f"ошибка формы vs теоретический кинк: {shape_err:.4f}  (мало => форма сохранилась)")
print()
print("Если скорость ~сохранилась И форма цела => солитон бежит устойчиво (база для щелей)")
