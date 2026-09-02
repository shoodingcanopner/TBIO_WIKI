"""Regression suite mirroring HANDOFF.md section 3 (28 cases)."""
import numpy as np
from looped_polymer import looped_polymer_eig, build_K

TOL = 1e-8

def check(name, n, k, is_ring, loops):
    lam, U = looped_polymer_eig(n, k, is_ring, loops, return_eigenvectors=True)
    K = build_K(n, k, is_ring, loops)
    ref = np.linalg.eigvalsh(K)
    sc = max(np.abs(K).max(), 1.0)
    e_val = np.max(np.abs(np.sort(lam) - ref)) / sc
    e_orth = np.max(np.abs(U.T @ U - np.eye(n)))
    e_res = np.max(np.abs(K @ U - U * lam[None, :])) / sc
    ok = e_val < TOL and e_orth < TOL and e_res < TOL
    print(f"{'OK ' if ok else 'FAIL'} {name:44s} val={e_val:.1e} orth={e_orth:.1e} res={e_res:.1e}")
    return ok

rng = np.random.default_rng(7)
results = []

for ring in (False, True):
    for n in (64, 65):
        results.append(check(f"base ring={ring} n={n}", n, 1.0, ring, []))

for ring in (False, True):
    results.append(check(f"single interior ring={ring}", 100, 1.0, ring, [[20, 70]]))
    results.append(check(f"single adjacent ring={ring}", 100, 1.0, ring, [[40, 41]]))
    results.append(check(f"single antipodal ring={ring}", 100, 1.0, ring, [[0, 50]]))

for ring in (False, True):
    for L in (5, 25, 60):
        n = 120
        loops = [list(rng.choice(n, 2, replace=False)) for _ in range(L)]
        results.append(check(f"L={L} random ring={ring}", n, 1.0, ring, loops))

results.append(check("repeated pair", 80, 1.0, False, [[10, 60], [10, 60]]))
results.append(check("shared index", 80, 1.0, True, [[10, 60], [10, 30], [10, 75]]))

for m in (2, 4, 8):
    n = 60
    loops = [[i, (i + m) % n] for i in range(n)]
    results.append(check(f"constant offset m={m} ring", n, 1.0, True, loops))

for k in (1e-4, 1.0, 1e4):
    results.append(check(f"k={k:g}", 90, k, False, [[3, 77], [40, 50]]))

print("\nanalytic circulant cross-check:")
all_analytic = True
for m in (2, 4, 8, 16, 32):
    n = 128
    k = 1.0
    loops = [[i, (i + m) % n] for i in range(n)]
    lam, _ = looped_polymer_eig(n, k, True, loops, return_eigenvectors=False)
    p = np.arange(n)
    ana = np.sort(-4.0 * k * (np.sin(p * np.pi / n) ** 2 + np.sin(p * m * np.pi / n) ** 2))
    err = np.max(np.abs(np.sort(lam) - ana)) / (4 * k)
    ok = err < 1e-12
    all_analytic &= ok
    print(f"{'OK ' if ok else 'FAIL'} m={m:2d}  err={err:.2e}")

npass = sum(results)
print(f"\n{npass}/{len(results)} matrix cases pass, analytic={'OK' if all_analytic else 'FAIL'}")
assert npass == len(results) and all_analytic
