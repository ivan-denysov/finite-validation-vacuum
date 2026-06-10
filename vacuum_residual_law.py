"""
УРОВЕНЬ 1: закон остатка синхрона (энергия вакуума в модели).
Аналитика: eps = 1 - R = sigma^2/(2 K^2) при sigma << K (узкая полоса, R~1).
Проверка: sigma-скан и K-скан, сравнение с формулой, R^2 подгонки.
Среднее поле Курамото, только нормальные узлы (кванты вне кластера не влияют — отдельно подтверждено).
"""
import numpy as np

def run_R(sigma, K, N=2000, steps=8000, dt=0.005, seed=7):
    rng=np.random.default_rng(seed)
    omega=rng.normal(1.0, sigma, N)
    theta=rng.uniform(0,2*np.pi,N)
    Racc=0.0; cnt=0
    for t in range(steps):
        z=np.mean(np.exp(1j*theta)); R=np.abs(z); psi=np.angle(z)
        theta=theta+dt*(omega+K*R*np.sin(psi-theta))
        if t>steps-2000: Racc+=R; cnt+=1
    return Racc/cnt

print("=== sigma-скан (K=1.0): eps=1-R vs формула sigma^2/2K^2 ===")
print(f"{'sigma':>7} {'eps_num':>10} {'eps_theory':>11} {'ratio':>7}")
K=1.0
sig_data=[]
for sigma in [0.05,0.10,0.15,0.20,0.30]:
    R=run_R(sigma,K)
    eps=1-R; th=sigma**2/(2*K**2)
    sig_data.append((sigma,eps,th))
    print(f"{sigma:>7} {eps:>10.5f} {th:>11.5f} {eps/th:>7.2f}")

print()
print("=== K-скан (sigma=0.15): eps vs sigma^2/2K^2 ===")
print(f"{'K':>5} {'eps_num':>10} {'eps_theory':>11} {'ratio':>7}")
sigma=0.15
K_data=[]
for K in [0.6,0.8,1.0,1.5,2.0]:
    R=run_R(sigma,K)
    eps=1-R; th=sigma**2/(2*K**2)
    K_data.append((K,eps,th))
    print(f"{K:>5} {eps:>10.5f} {th:>11.5f} {eps/th:>7.2f}")

# глобальная проверка закона: log-log регрессия eps ~ sigma^a / K^b
import numpy as np
pts=[(s,K0:=1.0,e) for s,e,_ in sig_data]+[(0.15,k,e) for k,e,_ in K_data]
S=np.array([p[0] for p in pts]); Kv=np.array([p[1] for p in pts]); E=np.array([p[2] for p in pts])
A=np.vstack([np.log(S),np.log(Kv),np.ones_like(S)]).T
coef,res,_,_=np.linalg.lstsq(A,np.log(E),rcond=None)
a,b,c0=coef
pred=A@coef
r2=1-((np.log(E)-pred)**2).sum()/((np.log(E)-np.log(E).mean())**2).sum()
print()
print(f"глобальная подгонка: eps ~ sigma^{a:.2f} / K^{-b:.2f} * {np.exp(c0):.3f}")
print(f"теория: sigma^2 / K^2 * 0.5  |  R^2 лог-подгонки = {r2:.4f}")
