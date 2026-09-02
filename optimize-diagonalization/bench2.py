import time, numpy as np, sys
from looped_polymer import looped_polymer_eig, build_K
_=looped_polymer_eig(50,1.0,False,[[0,25]],return_eigenvectors=False)
rng=np.random.default_rng(0)
print(f"{'n':>6} {'seq(s)':>9} {'eigvalsh(s)':>11} {'ratio':>8} {'val err':>9}")
for n in [200,400,800,1600,3200,6400]:
    loops=[list(rng.choice(n,2,replace=False)) for _ in range(20)]
    t0=time.perf_counter(); lam,_=looped_polymer_eig(n,1.0,False,loops,return_eigenvectors=False); t1=time.perf_counter()
    K=build_K(n,1.0,False,loops)
    t2=time.perf_counter(); ref=np.linalg.eigvalsh(K); t3=time.perf_counter()
    err=np.max(np.abs(np.sort(lam)-ref))/max(np.abs(K).max(),1)
    print(f"{n:>6} {t1-t0:>9.3f} {t3-t2:>11.3f} {(t1-t0)/(t3-t2):>7.2f}x {err:>9.1e}"); sys.stdout.flush()
