import sys, json, numpy as np, cv2
from scipy.optimize import least_squares, linear_sum_assignment
sys.path.insert(0, '/home/claude/aro-repo/procesador')
from calib import rot
PC = np.array(json.load(open('/home/claude/full/trabajo/calib.json'))['params'])
C = PC[:3]; W, H = 1920, 1080; CX, CY = W / 2, H / 2
S = 8  # downscale for search

def proj(q, XY):
    yaw, pitch, roll, f = q
    XY = np.atleast_2d(XY)
    Pw = np.c_[XY, np.zeros(len(XY))] - C
    base = np.c_[Pw[:, 0], -Pw[:, 2], -Pw[:, 1]]
    d = base @ rot(yaw, pitch, roll).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.c_[CX + f * d[:, 0] / d[:, 2], CY + f * d[:, 1] / d[:, 2]], d[:, 2] > 0.1

def wood_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); Hh, Ss, Vv = cv2.split(hsv)
    m = (((Hh >= 8) & (Hh <= 40) & (Vv > 110) & (Ss > 40)) | ((Hh > 20) & (Hh < 35) & (Ss > 120))).astype(np.uint8) * 255
    k = 15
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return m
    big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); m = (lab == big).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m); cv2.drawContours(out, cnts, -1, 255, -1); return out

# dense court polygon (subdivided so projection stays valid)
t = np.linspace(0, 1, 200)
RECT = np.r_[np.c_[t * 28, 0 * t], np.c_[28 + 0 * t, t * 15], np.c_[28 - t * 28, 15 + 0 * t], np.c_[0 * t, 15 - t * 15]]

def render(q, scale=S):
    uv, ok = proj(q, RECT)
    m = np.zeros((H // scale, W // scale), np.uint8)
    if ok.mean() < 0.999:  # parts behind camera: clip by keeping valid ones
        uv = uv[ok]
        if len(uv) < 10: return m
    p = np.clip(uv / scale, -5000, 5000).astype(np.int32)
    cv2.fillPoly(m, [p.reshape(-1, 1, 2)], 255); return m

def iou(a, b):
    A, B = a > 0, b > 0
    u = (A | B).sum(); return (A & B).sum() / u if u else 0

def search(mask_small, prior=None):
    best = (-1, None)
    yaws = np.linspace(PC[3] - 1.0, PC[3] + 1.0, 41) if prior is None else np.linspace(prior[0] - 0.15, prior[0] + 0.15, 11)
    pitches = np.linspace(PC[4] - 0.35, PC[4] + 0.15, 9) if prior is None else np.linspace(prior[1] - 0.06, prior[1] + 0.06, 5)
    fs = (700, 950, 1250, 1600, 2000, 2600) if prior is None else (prior[3] * 0.85, prior[3], prior[3] * 1.18)
    roll = PC[5] if prior is None else prior[2]
    for y in yaws:
        for p in pitches:
            for f in fs:
                q = np.array([y, p, roll, f]); s = iou(render(q), mask_small)
                if s > best[0]: best = (s, q)
    return best

def refine(q, mask, XY=None, feet=None):
    edge = cv2.Canny(mask, 50, 150)
    edge[:6] = 0; edge[-6:] = 0; edge[:, :6] = 0; edge[:, -6:] = 0
    dist = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)
    def res(p):
        uv, ok = proj(p, RECT)
        inside = ok & (uv[:, 0] > 8) & (uv[:, 0] < W - 8) & (uv[:, 1] > 8) & (uv[:, 1] < H - 8)
        x = np.clip(uv[:, 0], 0, W - 1).astype(np.float32); y = np.clip(uv[:, 1], 0, H - 1).astype(np.float32)
        d = cv2.remap(dist, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        return np.where(inside, d, 0.0)
    sol = least_squares(res, q, loss='soft_l1', f_scale=6, x_scale=[0.01, 0.01, 0.01, 50], max_nfev=300)
    r = res(sol.x); uv, ok = proj(sol.x, RECT)
    inside = ok & (uv[:, 0] > 8) & (uv[:, 0] < W - 8) & (uv[:, 1] > 8) & (uv[:, 1] < H - 8)
    return sol.x, float(np.median(r[inside])) if inside.sum() > 20 else 99.0, int(inside.sum())

def paint_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); Hh, Ss, Vv = cv2.split(hsv)
    m = ((Hh > 22) & (Hh < 36) & (Ss > 140) & (Vv > 120)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] > 4000: keep[lab == k] = 255
    return keep

u = np.linspace(0, 1, 60)
def keybox(x0, x1):
    return np.r_[np.c_[x0 + (x1 - x0) * u, 5.05 + 0 * u], np.c_[x1 + 0 * u, 5.05 + 4.9 * u], np.c_[x1 - (x1 - x0) * u, 9.95 + 0 * u], np.c_[x0 + 0 * u, 9.95 - 4.9 * u]]
KEYS = np.r_[keybox(0, 5.8), keybox(28, 22.2)]

def refine2(q, wmask, pmask):
    def dmap(m):
        e = cv2.Canny(m, 50, 150); e[:6] = 0; e[-6:] = 0; e[:, :6] = 0; e[:, -6:] = 0
        return cv2.distanceTransform(255 - e, cv2.DIST_L2, 5)
    dw, dp = dmap(wmask), dmap(pmask)
    def samp(dist, pts, p):
        uv, ok = proj(p, pts)
        inside = ok & (uv[:, 0] > 8) & (uv[:, 0] < W - 8) & (uv[:, 1] > 8) & (uv[:, 1] < H - 8)
        x = np.clip(np.nan_to_num(uv[:, 0]), 0, W - 1).astype(np.float32); y = np.clip(np.nan_to_num(uv[:, 1]), 0, H - 1).astype(np.float32)
        d = cv2.remap(dist, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        return np.where(inside, d, 0.0), inside
    def res(p): return np.r_[samp(dw, RECT, p)[0], 2.0 * samp(dp, KEYS, p)[0]]
    sol = least_squares(res, q, loss='soft_l1', f_scale=6, x_scale=[0.01, 0.01, 0.01, 50], max_nfev=400)
    rw, iw = samp(dw, RECT, sol.x); rp, ip = samp(dp, KEYS, sol.x)
    medw = float(np.median(rw[iw])) if iw.sum() > 20 else 99.0
    medp = float(np.median(rp[ip])) if ip.sum() > 20 else 99.0
    return sol.x, medw, medp, int(iw.sum()), int(ip.sum())

ANCH = np.float32([[7, 3], [21, 3], [21, 12], [7, 12]])
def H_from(p8): return cv2.getPerspectiveTransform(ANCH, np.float32(p8).reshape(4, 2))
def projH(Hm, XY):
    XY = np.atleast_2d(XY).astype(np.float64)
    q = np.c_[XY, np.ones(len(XY))] @ Hm.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return q[:, :2] / q[:, 2:3], q[:, 2] > 1e-6

def refineH(p8, wmask, pmask, w_keys=2.0):
    def dmap(m):
        e = cv2.Canny(m, 50, 150); e[:6] = 0; e[-6:] = 0; e[:, :6] = 0; e[:, -6:] = 0
        return cv2.distanceTransform(255 - e, cv2.DIST_L2, 5)
    dw, dp = dmap(wmask), dmap(pmask)
    def samp(dist, pts, p):
        uv, ok = projH(H_from(p), pts)
        inside = ok & (uv[:, 0] > 8) & (uv[:, 0] < W - 8) & (uv[:, 1] > 8) & (uv[:, 1] < H - 8)
        x = np.clip(np.nan_to_num(uv[:, 0]), 0, W - 1).astype(np.float32); y = np.clip(np.nan_to_num(uv[:, 1]), 0, H - 1).astype(np.float32)
        d = cv2.remap(dist, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        return np.where(inside, d, 0.0), inside
    def res(p): return np.r_[samp(dw, RECT, p)[0], w_keys * samp(dp, KEYS, p)[0]]
    sol = least_squares(res, p8, loss='soft_l1', f_scale=6, max_nfev=600)
    rw, iw = samp(dw, RECT, sol.x); rp, ip = samp(dp, KEYS, sol.x)
    medw = float(np.median(rw[iw])) if iw.sum() > 20 else 99.0
    medp = float(np.median(rp[ip])) if ip.sum() > 20 else 99.0
    return sol.x, medw, medp, int(iw.sum()), int(ip.sum())

from calib import forward as cyl_forward
def projC(p7, XY):
    p = np.r_[C, p7]
    return cyl_forward(p, XY)

def refineC(p7, wmask, pmask, w_keys=2.0, extra=None):
    def dmap(m):
        e = cv2.Canny(m, 50, 150); e[:6] = 0; e[-6:] = 0; e[:, :6] = 0; e[:, -6:] = 0
        return cv2.distanceTransform(255 - e, cv2.DIST_L2, 5)
    dw, dp = dmap(wmask), dmap(pmask)
    def samp(dist, pts, p):
        uv = projC(p, pts)
        inside = (uv[:, 0] > 8) & (uv[:, 0] < W - 8) & (uv[:, 1] > 8) & (uv[:, 1] < H - 8)
        x = np.clip(np.nan_to_num(uv[:, 0]), 0, W - 1).astype(np.float32); y = np.clip(np.nan_to_num(uv[:, 1]), 0, H - 1).astype(np.float32)
        d = cv2.remap(dist, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        return np.where(inside, d, 0.0), inside
    def res(p): return np.r_[samp(dw, RECT, p)[0], w_keys * samp(dp, KEYS, p)[0]]
    sol = least_squares(res, p7, loss='soft_l1', f_scale=6, x_scale='jac', max_nfev=800)
    rw, iw = samp(dw, RECT, sol.x); rp, ip = samp(dp, KEYS, sol.x)
    medw = float(np.median(rw[iw])) if iw.sum() > 20 else 99.0
    medp = float(np.median(rp[ip])) if ip.sum() > 20 else 99.0
    return sol.x, medw, medp, int(iw.sum()), int(ip.sum())
