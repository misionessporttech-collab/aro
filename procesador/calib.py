"""Calibración de la cámara panorámica: cancha (metros) <-> imagen (píxeles).
Modelo: homografía + distorsión radial (k1, k2) con centro libre.
Se ajusta con: esquinas y puntas de la línea central (marcadas a mano) +
el borde completo del piso (máscara), con pérdida robusta.
"""
import cv2, numpy as np, json
from scipy.optimize import least_squares

CW, CH = 28.0, 15.0
PTS = {  # (X, Y) en metros -> (u, v) en píxeles de la imagen recortada
    (0, 0): (837, 527), (28, 0): (2478, 483), (0, 15): (100, 1519), (28, 15): (3200, 1184),
    (14, 15): (2000, 2045),
}
S = 2000.0

def rot(yaw, pitch, roll):
    cy_, sy = np.cos(yaw), np.sin(yaw); cp, sp = np.cos(pitch), np.sin(pitch); cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy_, -sy, 0], [sy, cy_, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rr @ Rx @ Rz

def forward(params, XY):
    """Cámara cilíndrica: params = [Cx, Cy, Cz, yaw, pitch, roll, f, f2, u0, v0]"""
    Cx, Cy, Cz, yaw, pitch, roll, f, f2, u0, v0 = params[:10]
    XY = np.atleast_2d(XY)
    P = np.c_[XY, np.zeros(len(XY))] - [Cx, Cy, Cz]
    # marco cámara: z adelante (hacia -Y del mundo), x a la derecha (+X), y abajo (-Z)
    base = np.c_[P[:, 0], -P[:, 2], -P[:, 1]]
    d = base @ rot(yaw, pitch, roll).T
    th = np.arctan2(d[:, 0], d[:, 2])
    rho = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
    return np.c_[u0 + f * th, v0 + f2 * d[:, 1] / rho]

def fit(mask_path='court_mask.png'):
    mask = cv2.imread(mask_path, 0)
    edge = cv2.Canny(mask, 50, 150)
    dist = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)
    Hh, Ww = mask.shape
    best = None
    for Cy in (18, 21, 24):
      for Cz in (4, 7, 10):
        for pitch in (-0.3, -0.5, -0.7):
          p0 = np.r_[14.5, Cy, Cz, 0.0, pitch, 0.0, 1350, 1350, Ww / 2, Hh / 2]
          s0 = least_squares(lambda p: (forward(p, np.array(list(PTS.keys()), float)) - np.array(list(PTS.values()), float)).ravel(), p0, max_nfev=3000)
          if best is None or s0.cost < best.cost: best = s0
    p0 = best.x
    print('init cost', round(best.cost, 1), np.round(p0, 2))
    t = np.linspace(0, 1, 120)
    edges = np.r_[np.c_[t * CW, 0 * t], np.c_[t * CW, 0 * t + CH], np.c_[0 * t, t * CH], np.c_[0 * t + CW, t * CH]]
    kp = np.array(list(PTS.keys()), float); kv = np.array(list(PTS.values()), float)

    def res(p):
        pr = forward(p, kp)
        r1 = ((pr - kv) * 2.0).ravel()
        pe = forward(p, edges)
        x = np.clip(pe[:, 0], 0, Ww - 1); y = np.clip(pe[:, 1], 0, Hh - 1)
        r2 = cv2.remap(dist, x.astype(np.float32).reshape(-1, 1), y.astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        out = ((pe[:, 0] < 0) | (pe[:, 0] >= Ww) | (pe[:, 1] < 0) | (pe[:, 1] >= Hh)) * 50.0
        return np.r_[r1, r2 + out]

    sol = least_squares(res, p0, loss='soft_l1', f_scale=8.0, x_scale='jac', max_nfev=4000)
    r = res(sol.x)
    print('error esquinas (px):', np.round(np.abs(r[:10]).reshape(-1, 2).sum(1) / 2, 1))
    print('borde: mediana px', round(float(np.median(r[10:])), 1), 'p90', round(float(np.percentile(r[10:], 90)), 1))
    json.dump({'params': sol.x.tolist()}, open('calib.json', 'w'))
    return sol.x

class Mapper:
    """Imagen -> cancha por inversión numérica, con tabla densa para inicializar."""
    def __init__(self, params):
        self.p = np.array(params)
        gx, gy = np.meshgrid(np.arange(-3, 31.01, 0.25), np.arange(-3, 18.01, 0.25))
        self.grid = np.c_[gx.ravel(), gy.ravel()]
        self.img = forward(self.p, self.grid)
        from scipy.spatial import cKDTree
        self.tree = cKDTree(self.img)

    def to_court(self, uv):
        uv = np.atleast_2d(np.asarray(uv, float))
        _, idx = self.tree.query(uv)
        out = []
        for q, i in zip(uv, idx):
            x0 = self.grid[i]
            s = least_squares(lambda xy: forward(self.p, xy[None])[0] - q, x0, method='lm')
            out.append(s.x)
        return np.array(out)

    def to_image(self, XY):
        return forward(self.p, np.atleast_2d(np.asarray(XY, float)))

if __name__ == '__main__':
    p = fit()
    img = cv2.imread('frames/pano_crop.png')
    def poly(pts, col, th=4):
        pi = forward(p, np.array(pts, float)).astype(int)
        cv2.polylines(img, [pi.reshape(-1, 1, 2)], False, col, th)
    t = np.linspace(0, 1, 100)
    for seg in [np.c_[t * 28, 0 * t], np.c_[t * 28, 0 * t + 15], np.c_[0 * t, t * 15], np.c_[0 * t + 28, t * 15], np.c_[0 * t + 14, t * 15]]:
        poly(seg, (0, 0, 255))
    a = np.linspace(0, 2 * np.pi, 120)
    poly(np.c_[14 + 1.8 * np.cos(a), 7.5 + 1.8 * np.sin(a)], (0, 255, 0))
    for sx in (0, 1):
        X = (lambda v: v) if sx == 0 else (lambda v: 28 - v)
        poly([[X(0), 5.05], [X(5.8), 5.05], [X(5.8), 9.95], [X(0), 9.95]], (255, 0, 255))
        poly(np.c_[[X(5.8 + 1.8 * np.cos(v)) for v in a], 7.5 + 1.8 * np.sin(a)], (255, 0, 255), 3)
        aa = np.linspace(-np.arccos(6.6 / 6.75) - np.pi / 2 + np.pi, np.arccos(6.6 / 6.75) + np.pi / 2 - np.pi, 2)
        th = np.linspace(-1.35, 1.35, 100)
        arc = np.c_[1.575 + 6.75 * np.cos(th), 7.5 + 6.75 * np.sin(th)]
        arc[:, 0] = [X(v) for v in arc[:, 0]]
        poly(arc, (255, 255, 0), 3)
    cv2.imwrite('frames/calib_vis.jpg', cv2.resize(img, None, fx=0.4, fy=0.4))
