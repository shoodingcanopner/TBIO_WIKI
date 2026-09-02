import numpy as np
from rank1 import rank1_update

rng = np.random.default_rng(1)
worst = 0.0
for trial in range(30):
    n = rng.integers(3, 120)
    d = np.sort(rng.standard_normal(n) * rng.uniform(0.1, 100))
    # inject degeneracies sometimes
    if trial % 3 == 0 and n > 5:
        d[2:5] = d[2]
    z = rng.standard_normal(n)
    if trial % 4 == 0:
        z[rng.integers(0, n)] = 0.0  # exact deflation
    rho = rng.choice([-1, 1]) * rng.uniform(1e-3, 10)
    lam, V = rank1_update(d, z, rho)
    M = np.diag(d) + rho * np.outer(z, z)
    ref = np.linalg.eigvalsh(M)
    sc = max(np.abs(M).max(), 1)
    e1 = np.max(np.abs(lam - ref)) / sc
    e2 = np.max(np.abs(V.T @ V - np.eye(n)))
    e3 = np.max(np.abs(M @ V - V * lam[None, :])) / sc
    worst = max(worst, e1, e2, e3)
    assert e1 < 1e-10 and e2 < 1e-10 and e3 < 1e-10, (trial, n, rho, e1, e2, e3)
print("unit OK, worst =", worst)
