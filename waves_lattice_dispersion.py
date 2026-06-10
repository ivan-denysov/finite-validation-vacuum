import numpy as np

# Правильная решёточная модель распространения волны.
# Среда = решётка узлов с шагом a. Волна задаётся на узлах.
# Берём РАЗНЫЕ длины волн и смотрим дисперсию: меняется ли скорость с lambda
# на дискретной решётке (это известный, реальный эффект решёточной дисперсии).
#
# Стандартный результат теории решёток: на дискретной решётке скорость волны
# зависит от lambda (числовая дисперсия) — омега(k) = (2/dt)*arcsin(c*dt/a * sin(k*a/2)).
# Это ТОЧНО та "дисперсия на масштабе сетки", про которую мы говорили.
# Проверим её прямо.

a = 1.0          # lattice step
c = 1.0          # target wave speed
dt = 0.5         # time step (Courant-safe: c*dt/a < 1)

print("="*64)
print("Lattice dispersion: does wave speed depend on lambda?")
print("="*64)
print(f"lattice step a={a}, target c={c}")
print(f"{'lambda/a':>9} | {'k=2pi/lambda':>12} | {'v_phase':>9} | {'dev from c':>10}")
print("-"*54)

results=[]
for lam in [100, 50, 20, 10, 5, 4, 3, 2.5, 2.1]:
    k = 2*np.pi/lam
    # dispersion relation on the lattice (leapfrog / d'Alembert discretisation):
    # sin(omega*dt/2) = (c*dt/a) * sin(k*a/2)
    arg = (c*dt/a)*np.sin(k*a/2)
    if abs(arg) <= 1:
        omega = (2/dt)*np.arcsin(arg)
        v = omega/k                      # phase velocity
        dev = (v-c)/c*100
        results.append((lam,v,dev))
        print(f"{lam/a:>9.1f} | {k:>12.4f} | {v:>9.5f} | {dev:>9.2f}%")
    else:
        print(f"{lam/a:>9.1f} | {k:>12.4f} |   (evanescent: lambda below cutoff)")

print()
print("="*64)
print("READING")
print("="*64)
print("""
This is the EXACT effect we were after, done correctly:

- For long waves (lambda >> a): v ~ c, deviation ~0%. Classical optics holds.
- As lambda approaches the lattice step a: v DROPS below c, deviation grows.
- Near lambda ~ 2a: strong deviation, and below 2a the wave can't propagate
  (evanescent) -- a natural high-frequency cutoff.

So a discrete network DOES predict that different wavelengths travel at slightly
different speeds, with the effect appearing only as lambda nears the lattice
step. This is real, standard lattice physics (numerical/lattice dispersion) --
not something I'm inventing.

WHAT THIS MEANS FOR THE MODEL (honest):
- If the vacuum is a discrete network with step a, light of different lambda
  should show dispersion: v(lambda) deviates from c near the lattice scale.
- This is a GENUINE, falsifiable prediction that differs from classical optics
  (where c is exactly the same for all lambda).
- The catch: experiments (gamma-ray bursts over cosmological distances) so far
  see NO such dispersion down to ~Planck scale -> this bounds a to be extremely
  small (at or below Planck length). So the prediction exists and is testable;
  current data pushes the lattice step very small but does not kill the picture.
""")
print("Deviation summary (how fast the dym goes):")
for lam,v,dev in results:
    bar = "#"*int(abs(dev)*2)
    print(f"  lambda/a={lam/a:5.1f}: v={v:.4f} dev={dev:+6.2f}% {bar}")
