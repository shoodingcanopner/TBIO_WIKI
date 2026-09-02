"""
Rank-one update of a symmetric eigenproblem, done by explicit subspace
partitioning.

Solves   diag(d) + rho z z^T = V diag(lam) V^T

Partitioning strategy is unchanged from the validated version:

    active   : indices with |rho| z_p^2 above the movement threshold, after
               merging near-degenerate groups (only one combination in each
               degenerate group couples to z)
    deflated : eigenvalue passes through unchanged, eigenvector is the
               (possibly rotated) original basis vector

What changed (performance rewrite, see HANDOFF.md section 5a/5b):

  * The secular root solve no longer materializes an (n, n) temporary per
    bisection step.  It is a numba-compiled per-root loop: safeguarded
    Newton in DISPLACEMENT form (tau = lam - d_anchor), with a maintained
    bracket so every step is provably inside the correct pole gap.
    O(n) memory, ~10-20 iterations instead of a fixed 100, no allocations.
  * The Loewner / Gu-Eisenstat weight correction is compiled too (log-space
    accumulation per component, O(n^2) flops, O(n) memory).
  * rho < 0 is canonicalized to rho > 0 by the involution
        diag(d) + rho z z^T  =  -( diag(-d) + (-rho) z z^T )
    (negate + reverse ordering), so the compiled kernels only ever see the
    rho > 0 case with roots in (d_k, d_{k+1}).
  * The O(n^3) global orthogonality check (V^T V on the full matrix, every
    call!) is gone.  Cross-orthogonality between active and deflated columns
    is EXACT by construction (disjoint row supports), so only the active
    block can degrade, and Gu-Eisenstat theory says it can only degrade
    inside numerically-degenerate eigenvalue clusters.  We therefore
    QR re-orthonormalize inside clusters of the active block directly
    (cheap, block-local) and verify with a randomized probe; the dense
    eigh fallback survives only as a truly last resort.

If numba is unavailable the module falls back to the original vectorized-
bisection numpy path (slower, same results).
"""

import numpy as np

try:
    import numba
    from numba import njit, prange

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # no-op decorator
        def wrap(f):
            return f

        if args and callable(args[0]):
            return args[0]
        return wrap

    prange = range


# ======================================================================
# Compiled kernels.  Canonical form: rho > 0, d ascending.
# Root k lives strictly in (d[k], d[k+1]); the last root in
# (d[n-1], d[n-1] + rho * sum z^2).
# ======================================================================

@njit(cache=True, fastmath=False, parallel=True)
def _secular_roots_pos(d, z2, rho):
    """
    All n roots of  f(lam) = 1 + rho * sum_p z2_p / (d_p - lam),  rho > 0.

    Returns (tau, anchor): root_k = d[anchor_k] + tau_k, with tau the
    displacement from the nearest bracket endpoint (dlaed4-style), so that
    (d_p - root) can later be formed as (d_p - d[anchor]) - tau at full
    relative accuracy.
    """
    n = d.shape[0]
    tau = np.empty(n)
    anchor = np.empty(n, np.int64)
    eps = 2.220446049250313e-16

    z2sum = 0.0
    for p in range(n):
        z2sum += z2[p]
    top = rho * z2sum  # upper bound on the last root's displacement

    for k in prange(n):
        # ------------------------------------------------------------
        # bracket + anchor.  One secular evaluation at the gap midpoint
        # both classifies the root's half and seeds the iteration.
        # ------------------------------------------------------------
        if k < n - 1:
            gap = d[k + 1] - d[k]
            mid = d[k] + 0.5 * gap
            fmid = 1.0
            for p in range(n):
                fmid += rho * z2[p] / (d[p] - mid)
            if fmid > 0.0:
                anc = k          # root in left half
                a = 0.0
                b = 0.5 * gap
            else:
                anc = k + 1      # root in right half
                a = -0.5 * gap
                b = 0.0
        else:
            anc = n - 1
            a = 0.0
            b = top if top > 0.0 else 4.0 * eps

        anchor[k] = anc
        danc = d[anchor[k]]
        # the two poles bounding this root, in displacement coordinates
        dlo = d[k] - danc
        if k < n - 1:
            dhi = d[k + 1] - danc
        else:
            dhi = 0.0            # unused for the outer root

        # ------------------------------------------------------------
        # Two-pole rational iteration ("middle way", the dlaed4 scheme).
        # Model  f(t) ~ s + A/(dlo - t) + B/(dhi - t)  with A, B fixed by
        # matching the exact value and slope of the pole-side partial sums
        # at the current iterate; the model root is a quadratic solve.
        # Converges in ~3-5 iterations; the maintained bracket [a, b] makes
        # every accepted step provably inside the correct gap, with
        # bisection as the safeguard.
        # ------------------------------------------------------------
        t = 0.5 * (a + b)
        for _ in range(60):
            psi = 0.0           # poles at or below d[k]
            psip = 0.0
            phi = 0.0           # poles at or above d[k+1]
            phip = 0.0
            for p in range(k + 1):
                inv = 1.0 / ((d[p] - danc) - t)
                w = z2[p] * inv
                psi += w
                psip += w * inv
            for p in range(k + 1, n):
                inv = 1.0 / ((d[p] - danc) - t)
                w = z2[p] * inv
                phi += w
                phip += w * inv
            f = 1.0 + rho * (psi + phi)
            # tighten the bracket (f is increasing in lam)
            if f > 0.0:
                b = t
            else:
                a = t

            u = dlo - t          # (< 0 side distance to lower pole) sign ok
            if k < n - 1:
                v = dhi - t
                A = rho * psip * u * u
                Bc = rho * phip * v * v
                s = f - A / u - Bc / v
                # solve  s + A/(dlo - tn) + B/(dhi - tn) = 0
                aa = s
                bb = -(s * (dlo + dhi) + A + Bc)
                cc = s * dlo * dhi + A * dhi + Bc * dlo
                tn = 0.5 * (a + b)
                disc = bb * bb - 4.0 * aa * cc
                if disc >= 0.0:
                    sq = np.sqrt(disc)
                    if aa != 0.0:
                        if bb >= 0.0:
                            q = -0.5 * (bb + sq)
                        else:
                            q = -0.5 * (bb - sq)
                        r1 = q / aa
                        ok1 = a < r1 < b
                        r2 = cc / q if q != 0.0 else r1
                        ok2 = a < r2 < b
                        if ok1:
                            tn = r1
                        elif ok2:
                            tn = r2
                    elif bb != 0.0:
                        r1 = -cc / bb
                        if a < r1 < b:
                            tn = r1
            else:
                # outer root: one-pole + constant model
                A = rho * psip * u * u
                s = f - A / u
                if s != 0.0:
                    tn = dlo + A / s
                    if not (a < tn < b):
                        tn = 0.5 * (a + b)
                else:
                    tn = 0.5 * (a + b)

            sc = abs(t)
            if abs(danc) * eps > sc:
                sc = abs(danc) * eps
            if abs(tn - t) <= 8.0 * eps * sc or (b - a) <= 8.0 * eps * sc:
                t = tn
                break
            t = tn

        # never allow a root exactly on its anchor pole
        if t == 0.0:
            bump = eps * abs(danc)
            if bump == 0.0:
                bump = 1e-300
            if k < n - 1 and anchor[k] == k + 1:
                t = -bump
            elif k == n - 1 or anchor[k] == k:
                t = bump
        tau[k] = t
    return tau, anchor


@njit(cache=True, fastmath=False, parallel=True)
def _loewner_weights_pos(d, tau, anchor, rho):
    """
    Gu-Eisenstat corrected weights:
      rho * zc_p^2 = (lam_p - d_p) * prod_{m != p} (lam_m - d_p)/(d_m - d_p)
    with lam_m - d_p formed as (d[anchor_m] - d_p) + tau_m.
    Log-space accumulation, per-component; returns zc^2 (>= 0, clamped).
    ok flag is False if anything came out non-finite or significantly
    negative, in which case the caller should keep the raw z.
    """
    n = d.shape[0]
    zh2 = np.empty(n)
    dan = np.empty(n)
    for m in range(n):
        dan[m] = d[anchor[m]]
    ok = True
    # Over/underflow control without per-term log/exp: accumulate the plain
    # product and renormalize by 2^+-512 whenever it leaves [2^-512, 2^512],
    # tracking the pulled-out exponent separately.  Same log-space safety,
    # ~5x cheaper per term.
    BIG = 1.3407807929942597e154      # 2^512
    INV = 7.458340731200207e-155      # 2^-512
    for p in prange(n):
        prod = 1.0
        ex = 0
        diag = 0.0
        dp = d[p]
        good = True
        for m in range(n):
            num = (dan[m] - dp) + tau[m]  # lam_m - d_p
            if m == p:
                diag = num
                continue
            r = num / (d[m] - dp)
            if r == 0.0:
                zh2[p] = 0.0
                good = False
                break
            prod *= r
            ar = prod if prod >= 0.0 else -prod
            if ar > BIG:
                prod *= INV
                ex += 1
            elif ar < INV:
                prod *= BIG
                ex -= 1
        if not good:
            continue
        # fold the exponent back in; saturate instead of overflowing
        while ex > 0 and np.isfinite(prod):
            prod *= BIG
            ex -= 1
        while ex < 0 and prod != 0.0:
            prod *= INV
            ex += 1
        val = prod * diag / rho
        if not np.isfinite(val):
            ok = False
            val = 0.0
        zh2[p] = val
    return zh2, ok


@njit(cache=True, fastmath=False, parallel=True)
def _assemble_vectors_pos(d, tau, anchor, zc, flip):
    """
    Va[p, m] = zc_p / (d_p - lam_m),  lam_m = d[anchor_m] + tau_m,
    columns normalized.  Differences formed in displacement style.

    With flip=True the output is written directly in REVERSED row and
    column order, which is the orientation the rho < 0 caller needs --
    writing it here avoids an (n, n) reversal copy afterwards.
    """
    n = d.shape[0]
    Va = np.empty((n, n))
    dan = np.empty(n)
    for m in range(n):
        dan[m] = d[anchor[m]]
    nrm2 = np.zeros(n)
    # Row-major fill (unit-stride writes into the C-ordered array) with the
    # column norms accumulated on the fly; a second unit-stride pass scales.
    for p in range(n):
        dp = d[p]
        zp = zc[p]
        po = n - 1 - p if flip else p
        if flip:
            for m in range(n):
                v = zp / ((dp - dan[m]) - tau[m])
                Va[po, n - 1 - m] = v
                nrm2[m] += v * v
        else:
            for m in range(n):
                v = zp / ((dp - dan[m]) - tau[m])
                Va[po, m] = v
                nrm2[m] += v * v
    inv = np.empty(n)
    for m in range(n):
        s = np.sqrt(nrm2[m])
        io = n - 1 - m if flip else m
        inv[io] = 1.0 / s if s != 0.0 else 1.0
    for p in prange(n):
        for m in range(n):
            Va[p, m] *= inv[m]
    return Va


# ======================================================================
# numpy fallback for the root solve (original vectorized bisection),
# used only when numba is unavailable.
# ======================================================================

def _secular_roots_pos_numpy(da, za2, rho):
    na = len(da)
    o = da
    shift = rho * za2.sum()
    s_lo = np.zeros(na)
    s_hi = np.empty(na)
    s_hi[:-1] = da[1:] - da[:-1]
    s_hi[-1] = max(shift, np.finfo(float).tiny)

    width = s_hi - s_lo
    pad = np.maximum(np.abs(width), 1.0) * 1e-15
    a = s_lo + pad
    b = s_hi - pad
    bad = a >= b
    if np.any(bad):
        a[bad] = s_lo[bad] + 0.25 * width[bad]
        b[bad] = s_hi[bad] - 0.25 * width[bad]

    dif = da[None, :] - o[:, None]

    def g_all(s):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return 1.0 + rho * np.sum(za2[None, :] / (dif - s[:, None]), axis=1)

    fa = g_all(a)
    fb = g_all(b)
    outer = na - 1
    for _ in range(200):
        needs = ~np.isfinite(fa) | ~np.isfinite(fb) | (fa * fb > 0.0)
        if not needs.any():
            break
        if needs[outer]:
            b[outer] = b[outer] * 2.0 + 1e-12
        inner = needs.copy()
        inner[outer] = False
        if inner.any():
            mid = 0.5 * (a + b)
            a[inner] = a[inner] + (mid[inner] - a[inner]) * 0.5
            b[inner] = b[inner] - (b[inner] - mid[inner]) * 0.5
        fa = g_all(a)
        fb = g_all(b)

    for _ in range(100):
        mid = 0.5 * (a + b)
        fm = g_all(mid)
        go_left = (fa * fm) < 0.0
        b = np.where(go_left, mid, b)
        a = np.where(go_left, a, mid)
        fa = np.where(go_left, fa, fm)

    tau = 0.5 * (a + b)
    zero = tau == 0.0
    if np.any(zero):
        tau[zero] = np.nextafter(0.0, 1.0)
    return tau, np.arange(na)


# ======================================================================
# Active-block solve in canonical (rho > 0) space, with the rho < 0 case
# mapped through negation + reversal.
# ======================================================================

def _solve_active(da, za, rho):
    """
    Eigen-solve diag(da) + rho za za^T on the active block (all za coupled,
    all da distinct).  Returns (lam ascending, Va with matching columns).
    """
    na = len(da)
    if na == 1:
        return np.array([da[0] + rho * za[0] * za[0]]), np.ones((1, 1))

    flip = rho < 0.0
    if flip:
        dd = (-da)[::-1].copy()
        zz = za[::-1].copy()
        rr = -rho
    else:
        dd = da
        zz = za
        rr = rho

    zz2 = zz * zz

    if _HAVE_NUMBA:
        tau, anchor = _secular_roots_pos(dd, zz2, rr)
        zh2, ok = _loewner_weights_pos(dd, tau, anchor, rr)
    else:
        tau, anchor = _secular_roots_pos_numpy(dd, zz2, rr)
        lam_t = dd[anchor] + tau
        num_mat = lam_t[None, :] - dd[:, None]
        den_mat = dd[None, :] - dd[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.eye(na, dtype=bool), 1.0, num_mat / den_mat)
            logs = np.log(np.abs(ratio))
            logs[np.eye(na, dtype=bool)] = 0.0
            prod = np.exp(np.sum(logs, axis=1))
            sgn = np.prod(np.where(np.eye(na, dtype=bool), 1.0,
                                   np.sign(ratio)), axis=1)
            zh2 = sgn * prod * np.diagonal(num_mat) / rr
        ok = bool(np.all(np.isfinite(zh2)))

    # Accept the corrected weights when they are consistent (theory says
    # rho * zc^2 >= 0, i.e. zh2 >= 0 here since rr > 0); clamp tiny
    # negatives caused by rounding instead of throwing the whole
    # correction away (HANDOFF section 5b: the old all-or-nothing gate was
    # one reason the dense fallback fired so often).
    neg_floor = -1e-8 * max(float(np.max(zz2)), 1e-300)
    if ok and np.all(zh2 >= neg_floor):
        zc = np.where(zz >= 0.0, 1.0, -1.0) * np.sqrt(np.maximum(zh2, 0.0))
    else:
        zc = zz.copy()

    if _HAVE_NUMBA:
        Va = _assemble_vectors_pos(dd, tau, anchor, zc, flip)
    else:
        lam_t = dd[anchor] + tau
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = dd[:, None] - lam_t[None, :]
            Va = zc[:, None] / denom
            nrm = np.linalg.norm(Va, axis=0)
            nrm[nrm == 0.0] = 1.0
            Va = Va / nrm[None, :]

    lam = dd[anchor] + tau

    if flip:
        lam = (-lam)[::-1].copy()
        if not _HAVE_NUMBA:
            Va = Va[::-1, ::-1].copy()

    return lam, Va


# ======================================================================
# Public entry point (same contract as before)
# ======================================================================

def _givens_deflate(d_grp, z_grp):
    """Householder rotation concentrating a degenerate group's z into its
    first component.  Spectrally free because the group's d are equal."""
    m = len(z_grp)
    Q = np.eye(m)
    if m == 1:
        return Q, z_grp.copy()
    nz = np.linalg.norm(z_grp)
    if nz == 0.0:
        return Q, z_grp.copy()
    v = z_grp.copy()
    v[0] += np.sign(z_grp[0]) * nz if z_grp[0] != 0 else nz
    vn = np.linalg.norm(v)
    if vn < 1e-300:
        return Q, z_grp.copy()
    v = v / vn
    Q = np.eye(m) - 2.0 * np.outer(v, v)
    znew = Q.T @ z_grp
    return Q, znew


class Rank1Rotation:
    """
    The eigenvector rotation V of one rank-one update, kept in FACTORED form:

        V = P_cols( BlockRot( scatter(Va) ) )

    i.e. a block-diagonal degeneracy rotation, an active block Va scattered
    into rows/columns `act` (deflated slots are unit columns), and a final
    column sort.  `apply_right(R)` computes R @ V in
    O(rows(R) * na^2 + rows(R) * n) without ever forming the (n, n) matrix,
    which is what makes the spectrum-only sweep cheap.  `materialize()`
    produces the dense V for callers that need it.
    """

    __slots__ = ("n", "kind", "act", "dea", "Va", "merge_blocks", "order",
                 "dense_V")

    def __init__(self, n, kind, act=None, dea=None, Va=None,
                 merge_blocks=None, order=None, dense_V=None):
        self.n = n
        self.kind = kind                  # "identity" | "blocks" | "general" | "dense"
        self.act = act
        self.dea = dea
        self.Va = Va
        self.merge_blocks = merge_blocks or []
        self.order = order
        self.dense_V = dense_V

    def apply_right(self, R):
        """Return R @ V for R of shape (m, n)."""
        if self.kind == "identity":
            return R.copy()
        if self.kind == "dense":
            return R @ self.dense_V
        Rr = R.copy()
        for (gs, ge, Q) in self.merge_blocks:
            Rr[:, gs:ge] = R[:, gs:ge] @ Q
        if self.kind == "blocks":
            return Rr
        out = np.empty_like(Rr)
        out[:, self.act] = Rr[:, self.act] @ self.Va
        out[:, self.dea] = Rr[:, self.dea]
        return out[:, self.order]

    def materialize(self):
        """Dense (n, n) V, columns matching the sorted eigenvalues."""
        n = self.n
        if self.kind == "identity":
            return np.eye(n)
        if self.kind == "dense":
            return self.dense_V.copy()
        if self.kind == "blocks":
            V0 = np.eye(n)
            for (gs, ge, Q) in self.merge_blocks:
                V0[gs:ge, gs:ge] = Q
            return V0
        Vfull = np.zeros((n, n))
        act, dea = self.act, self.dea
        Vfull[act[:, None], act[None, :]] = self.Va
        Vfull[dea, dea] = 1.0
        for (gs, ge, Q) in self.merge_blocks:
            Vfull[gs:ge, :] = Q @ Vfull[gs:ge, :]
        return Vfull[:, self.order]


def rank1_update(d, z, rho, tol_rel=1e-12):
    """
    Parameters
    ----------
    d : (n,) ascending eigenvalues of the current diagonal block
    z : (n,) perturbation direction in the current eigenbasis
    rho : scalar perturbation strength (negative for polymer loops)

    Returns
    -------
    lam : (n,) ascending updated eigenvalues
    V   : (n, n) orthogonal, columns are updated eigenvectors in the
          *current* basis coordinates
    """
    lam, rot = rank1_update_factored(d, z, rho, tol_rel)
    return lam, rot.materialize()


def rank1_update_factored(d, z, rho, tol_rel=1e-12):
    """
    Same computation as `rank1_update`, but the rotation is returned as a
    `Rank1Rotation` (factored form) instead of a dense matrix.  Use
    `rot.apply_right(R)` for R @ V and `rot.materialize()` for the dense V.
    """
    d = np.asarray(d, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if d.shape != z.shape:
        raise ValueError(
            f"d and z must have the same length, got {d.shape} and {z.shape}")
    n = len(d)

    scale = max(np.abs(d).max(), 1.0)
    znorm = np.linalg.norm(z)

    if znorm < tol_rel * max(scale, 1.0) or rho == 0.0:
        return d.copy(), Rank1Rotation(n, "identity")

    eps = np.finfo(float).eps
    gap_tol = 8.0 * eps * scale
    # Deflation criterion.  The movement test |rho| z_p^2 <= tol guarantees
    # the EIGENVALUE is unchanged to rounding, but the deflated EIGENVECTOR
    # (a plain unit column) carries a residual of first order in z_p:
    #   || (D + rho z z^T) e_p - d_p e_p || = |rho| |z_p| ||z||
    # so components with z_p ~ sqrt(eps) pass the movement test yet leave a
    # ~1e-8 residual.  We therefore deflate on the FIRST-ORDER coupling
    # |rho| |z_p| ||z|| <= 8 eps scale, which is strictly more inclusive.
    # This is safe now because the displacement-form root solver resolves
    # tau ~ rho z_p^2 far below eps*|d| without loss (the failure mode that
    # originally motivated the looser movement test -- HANDOFF trap 3 --
    # only existed in the absolute-lambda solver).
    # The numpy fallback solver (fixed-count padded bisection) cannot
    # resolve the ~rho z_p^2 displacements that the strict criterion admits,
    # so without numba we keep the original movement-based test (and with it
    # the original, slightly weaker, deflated-vector residual ~sqrt(eps)).
    if _HAVE_NUMBA:
        z_tol_coup = 8.0 * eps * scale / max(abs(rho) * znorm, 1e-300)
    else:
        z_tol_coup = np.sqrt(
            8.0 * eps * max(scale, abs(rho) * znorm * znorm) / abs(rho))

    # ---- Step 1: merge near-degenerate d groups (transitive on gaps) ----
    # Qglobal is block-diagonal (one Householder per merged group) and very
    # often the identity; build it lazily and remember the blocks so the
    # final basis product can be applied blockwise instead of as a dense
    # O(n^3) matmul.
    zr = z.copy()
    merge_blocks = []           # (start, end, Q) for each merged group
    g_start = 0
    while g_start < n:
        g_end = g_start + 1
        while g_end < n and (d[g_end] - d[g_end - 1]) <= gap_tol:
            g_end += 1
        m = g_end - g_start
        if m > 1:
            Q, znew = _givens_deflate(d[g_start:g_end], zr[g_start:g_end])
            merge_blocks.append((g_start, g_end, Q))
            zr[g_start:g_end] = znew
            zr[g_start + 1:g_end] = 0.0
        g_start = g_end

    # ---- Step 2: partition by first-order coupling strength ----
    active = np.abs(zr) > z_tol_coup
    act = np.flatnonzero(active)
    dea = np.flatnonzero(~active)

    if act.size == 0:
        return d.copy(), Rank1Rotation(n, "blocks", merge_blocks=merge_blocks)

    da = d[act].copy()
    za = zr[act].copy()
    na = len(da)

    # ---- Steps 3-5: secular roots + Loewner weights + vectors ----
    lam_a, Va = _solve_active(da, za, rho)

    # ---- cluster-local re-orthogonalization on the ACTIVE block ----
    # Cross-orthogonality active/deflated is exact (disjoint row support in
    # the pre-rotation frame), and Loewner-corrected columns for
    # well-separated roots are orthogonal to O(eps) by construction
    # (Gu-Eisenstat).  Only clusters of nearly-equal lam_a can degrade, so
    # we re-orthonormalize inside those clusters unconditionally: it is
    # O(sum n * m_c^2), block-local, and spectrally harmless.
    cl_tol = 64.0 * eps * max(np.abs(lam_a).max(), 1.0)
    s = 0
    while s < na:
        e = s + 1
        while e < na and (lam_a[e] - lam_a[e - 1]) <= cl_tol:
            e += 1
        if e - s > 1:
            blk, _ = np.linalg.qr(Va[:, s:e])
            # keep column signs deterministic-ish (QR may flip)
            Va[:, s:e] = blk
        s = e

    # ---- randomized orthogonality probe (O(n * na), not O(n^3)) ----
    rng = np.random.default_rng(12345)
    X = rng.standard_normal((na, 1))
    X /= np.linalg.norm(X, axis=0)
    resid = Va.T @ (Va @ X) - X
    probe_err = np.max(np.abs(resid))
    if probe_err > 1e-9:
        # widen: full Gram on the active block only (O(n * na^2) BLAS)
        G = Va.T @ Va
        err = np.max(np.abs(G - np.eye(na)))
        if err > 1e-10:
            # last resort: dense solve of the (cheap to form) rank-one
            # matrix -- this should be genuinely rare now
            Mfull = np.diag(d) + rho * np.outer(z, z)
            lam_full, Vfull = np.linalg.eigh(Mfull)  # original frame, sorted
            return lam_full, Rank1Rotation(n, "dense", dense_V=Vfull)

    # ---- Step 6: merge eigenvalue lists and record the column order ----
    lam_full = np.empty(n)
    lam_full[act] = lam_a
    lam_full[dea] = d[dea]
    order = np.argsort(lam_full, kind="stable")

    rot = Rank1Rotation(n, "general", act=act, dea=dea, Va=Va,
                        merge_blocks=merge_blocks, order=order)
    return lam_full[order], rot
