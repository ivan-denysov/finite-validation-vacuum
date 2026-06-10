"""
УРОВЕНЬ 3: рискованная оценка физической Lambda из закона остатка eps = sigma^2/(2K^2).
Логика:
- Планковская плотность rho_P = E_P / l_P^3 ~ 4.6e113 Дж/м^3 — естественная шкала сети
  (узел на ячейку l_P^3, энергия кванта связи ~ E_P).
- Наша энергия вакуума = ОСТАТОК: rho_vac = eps * rho_P = (sigma^2/2K^2) * rho_P.
- Наблюдаемое: rho_obs ~ 6e-10 Дж/м^3  => eps_needed = rho_obs/rho_P.
- ВОПРОС: какой разброс sigma/K нужен? Реалистичен ли он, или абсурден?
"""
import numpy as np

hbar=1.054571817e-34; c=2.99792458e8; G=6.67430e-11
E_P=np.sqrt(hbar*c**5/G)        # Дж
l_P=np.sqrt(hbar*G/c**3)        # м
t_P=l_P/c
rho_P=E_P/l_P**3                # Дж/м^3

rho_obs=6e-10                   # Дж/м^3 (набл. тёмная энергия)

eps_needed=rho_obs/rho_P
sigma_over_K=np.sqrt(2*eps_needed)   # из eps=sigma^2/(2K^2) => sigma/K=sqrt(2 eps)

print(f"E_P = {E_P:.3e} Дж;  l_P = {l_P:.3e} м;  rho_P = {rho_P:.3e} Дж/м^3")
print(f"rho_obs (тёмная энергия) = {rho_obs:.1e} Дж/м^3")
print()
print(f"требуемый остаток: eps = rho_obs/rho_P = {eps_needed:.3e}")
print(f"=> sigma/K = sqrt(2 eps) = {sigma_over_K:.3e}")
print()
print(f"в логарифмах: eps ~ 10^{np.log10(eps_needed):.1f};  sigma/K ~ 10^{np.log10(sigma_over_K):.1f}")
print()
# интерпретация: насколько однородна сеть?
print("ИНТЕРПРЕТАЦИЯ:")
print(f"разброс собственных темпов узлов относительно связи: sigma/K ~ {sigma_over_K:.1e}")
print(f"т.е. узлы вакуума одинаковы по темпу с точностью ~ {sigma_over_K:.0e} (одна часть на 10^{-np.log10(sigma_over_K):.0f})")
print()
# обратная проверка: что если sigma/K взять "разумным" (например 0.1)?
for soK in [0.1, 1e-3, 1e-6]:
    rho=0.5*soK**2*rho_P
    print(f"если sigma/K={soK:g}: rho_vac = {rho:.2e} Дж/м^3 (наблюдаемое 6e-10) — расхождение 10^{np.log10(rho/rho_obs):.0f}")
