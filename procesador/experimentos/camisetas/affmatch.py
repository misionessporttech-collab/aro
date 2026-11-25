import numpy as np
from scipy.optimize import linear_sum_assignment
def affine_icp(src, dst, it=8, trim=0.7, init_gate=200):
    """Alinea puntos proyectados (src) con pies detectados (dst) con una afín robusta; devuelve pares y error."""
    if len(src) < 3 or len(dst) < 3: return [], 99.0, None
    best = None
    # arranques: identidad y corrimientos según pares cercanos
    starts = [np.zeros(2)]
    for i in range(len(src)):
        d = np.linalg.norm(dst - src[i], axis=1); j = np.argmin(d)
        if d[j] < init_gate: starts.append(dst[j] - src[i])
    for sh in starts[:12]:
        A = np.eye(2); b = sh.copy()
        for _ in range(it):
            P = src @ A.T + b
            Dm = np.linalg.norm(P[:, None] - dst[None], axis=2)
            r, c = linear_sum_assignment(Dm)
            d = Dm[r, c]; keep = d <= np.quantile(d, trim) + 1e-6
            r, c = r[keep], c[keep]
            if len(r) < 3: break
            X = np.c_[src[r], np.ones(len(r))]
            sol, *_ = np.linalg.lstsq(X, dst[c], rcond=None)
            A2, b2 = sol[:2].T, sol[2]
            # limitar a algo cercano a similaridad
            s = np.sqrt(abs(np.linalg.det(A2)))
            if not (0.6 < s < 1.6): break
            A, b = A2, b2
        P = src @ A.T + b
        Dm = np.linalg.norm(P[:, None] - dst[None], axis=2); r, c = linear_sum_assignment(Dm)
        d = Dm[r, c]; good = d < 35
        score = (good.sum(), -np.median(d[good]) if good.any() else -99)
        if best is None or score > best[0]: best = (score, [(int(i), int(j)) for i, j, g in zip(r, c, good) if g], float(np.median(d[good])) if good.any() else 99.0, (A, b))
    return best[1], best[2], best[3]
