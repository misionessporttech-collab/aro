"""Calibración de la cámara panorámica de Veo: cancha (metros) <-> imagen (píxeles).

Modelo de cámara cilíndrica (como la unión panorámica de Veo):
params = [Cx, Cy, Cz, yaw, pitch, roll, f, f2, u0, v0]
Se ajusta al borde del piso de madera (máscara de color) y, si se dan, a puntos marcados a mano.
"""
import json, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

CW, CH = 28.0, 15.0

def rot(yaw, pitch, roll):
    cy_, sy = np.cos(yaw), np.sin(yaw); cp, sp = np.cos(pitch), np.sin(pitch); cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy_, -sy, 0], [sy, cy_, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rr @ Rx @ Rz

def forward(params, XY):
    Cx, Cy, Cz, yaw, pitch, roll, f, f2, u0, v0 = params[:10]
    XY = np.atleast_2d(XY)
    P = np.c_[XY, np.zeros(len(XY))] - [Cx, Cy, Cz]
    base = np.c_[P[:, 0], -P[:, 2], -P[:, 1]]
    d = base @ rot(yaw, pitch, roll).T
    th = np.arctan2(d[:, 0], d[:, 2])
    rho = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
    return np.c_[u0 + f * th, v0 + f2 * d[:, 1] / rho]

def court_mask(img):
    """Piso de madera + pintura amarilla, sin el azul de afuera."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    wood = ((H >= 8) & (H <= 40) & (V > 110)).astype(np.uint8) * 255
    k = max(5, int(img.shape[1] / 130))
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((k // 2 + 1, k // 2 + 1), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(wood)
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (lab == big).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(mask); cv2.drawContours(fill, cnts, -1, 255, -1)
    return fill

def _edges():
    t = np.linspace(0, 1, 120)
    return np.r_[np.c_[t * CW, 0 * t], np.c_[t * CW, 0 * t + CH], np.c_[0 * t, t * CH], np.c_[0 * t + CW, t * CH]]

def fit(mask, p0, pts=None, w_pts=2.0):
    edge = cv2.Canny(mask, 50, 150)
    dist = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)
    Hh, Ww = mask.shape
    E = _edges()
    kp = np.array(list(pts.keys()), float) if pts else None
    kv = np.array(list(pts.values()), float) if pts else None
    def res(p):
        r = []
        if pts: r.append(((forward(p, kp) - kv) * w_pts).ravel())
        pe = forward(p, E)
        x = np.clip(pe[:, 0], 0, Ww - 1).astype(np.float32); y = np.clip(pe[:, 1], 0, Hh - 1).astype(np.float32)
        d = cv2.remap(dist, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        out = ((pe[:, 0] < 0) | (pe[:, 0] >= Ww) | (pe[:, 1] < 0) | (pe[:, 1] >= Hh)) * 50.0
        r.append(d + out)
        return np.concatenate(r)
    sol = least_squares(res, p0, loss="soft_l1", f_scale=8.0, x_scale="jac", max_nfev=4000)
    r = res(sol.x)
    rb = r[-len(E):]
    return sol.x, float(np.median(rb)), float(np.percentile(rb, 90))

def draw(img, p, path, scale=0.4):
    vis = img.copy()
    t = np.linspace(0, 1, 100)
    th = max(2, int(img.shape[1] / 800))
    def poly(pts, col):
        pi = forward(p, np.array(pts, float)).astype(int)
        cv2.polylines(vis, [pi.reshape(-1, 1, 2)], False, col, th)
    for seg in [np.c_[t * 28, 0 * t], np.c_[t * 28, 0 * t + 15], np.c_[0 * t, t * 15], np.c_[0 * t + 28, t * 15], np.c_[0 * t + 14, t * 15]]:
        poly(seg, (0, 0, 255))
    a = np.linspace(0, 2 * np.pi, 120)
    poly(np.c_[14 + 1.8 * np.cos(a), 7.5 + 1.8 * np.sin(a)], (0, 255, 0))
    for X in (lambda v: v, lambda v: 28 - v):
        poly([[X(0), 5.05], [X(5.8), 5.05], [X(5.8), 9.95], [X(0), 9.95]], (255, 0, 255))
        th_ = np.linspace(-1.35, 1.35, 100)
        poly(np.c_[[X(v) for v in 1.575 + 6.75 * np.cos(th_)], 7.5 + 6.75 * np.sin(th_)], (255, 255, 0))
    cv2.imwrite(path, cv2.resize(vis, None, fx=scale, fy=scale))

def refit_auto(frame, base, vis_path=None):
    """Reajusta una calibración previa (por ejemplo la de Mitre) a una grabación nueva."""
    p0 = np.array(base["params"], float)
    bw, bh = base.get("size", [3332, 2160])
    h, w = frame.shape[:2]
    sx, sy = w / bw, h / bh
    p0[6] *= sx; p0[8] *= sx; p0[7] *= sy; p0[9] *= sy
    mask = court_mask(frame)
    p, med, p90 = fit(mask, p0)
    print(f"calibración automática: borde a {med:.1f} px (mediana), {p90:.1f} px (p90)")
    if med > 6:
        print("ATENCIÓN: la calibración no encaja bien; puede ser otra cancha u otro encuadre. Revisá calib_vis.jpg.")
    if vis_path: draw(frame, p, vis_path)
    return p

class Mapper:
    """Imagen -> cancha por inversión numérica."""
    def __init__(self, params):
        self.p = np.array(params)
        gx, gy = np.meshgrid(np.arange(-3, 31.01, 0.25), np.arange(-3, 18.01, 0.25))
        self.grid = np.c_[gx.ravel(), gy.ravel()]
        self.img = forward(self.p, self.grid)
        self.tree = cKDTree(self.img)

    def to_court(self, uv):
        """Inversa exacta del modelo cilíndrico: rayo desde la cámara hasta el piso (z = 0)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        Cx, Cy, Cz, yaw, pitch, roll, f, f2, u0, v0 = self.p[:10]
        th = (uv[:, 0] - u0) / f; t = (uv[:, 1] - v0) / f2
        d = np.c_[np.sin(th), t, np.cos(th)]
        base = d @ rot(yaw, pitch, roll)
        P = np.c_[base[:, 0], -base[:, 2], -base[:, 1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -Cz / P[:, 2]
        out = np.c_[Cx + s * P[:, 0], Cy + s * P[:, 1]]
        out[~np.isfinite(out).all(1) | (s <= 0)] = np.nan
        return out

    def to_image(self, XY):
        return forward(self.p, np.atleast_2d(np.asarray(XY, float)))
