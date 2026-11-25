"""Calibración del archivo panorámico original de Veo (dos lentes, una arriba de la otra, 2048x1024 cada una).

Cada lente se modela como una cámara (posición, orientación) con proyección "ojo de pez" y distorsión radial:
  lente = [Cx, Cy, Cz, yaw, pitch, roll, f, f2, u0, v0, k1, k2]
La lente de arriba ve la mitad izquierda de la cancha (aro en x=0) y la de abajo la derecha (aro en x=28).
Para pasar de píxel a cancha se usa un polinomio inverso ajustado sobre la zona de juego.
"""
import json, numpy as np
from calib import rot

LENTES = ("arriba", "abajo")
LW, LH = 2048, 1024


def proyectar(modelo, q, XY):
    """cancha (m) -> píxel de la lente."""
    Cx, Cy, Cz, yaw, pitch, roll, f, f2, u0, v0, k1, k2 = q
    XY = np.atleast_2d(np.asarray(XY, float))
    P = np.c_[XY, np.zeros(len(XY))] - [Cx, Cy, Cz]
    base = np.c_[P[:, 0], -P[:, 2], -P[:, 1]]
    d = base @ rot(yaw, pitch, roll).T
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    if modelo == "rect":
        x, y = dx / dz, dy / dz
    elif modelo == "cyl":
        rho = np.hypot(dx, dz); x, y = np.arctan2(dx, dz), dy / rho
    elif modelo == "equirect":
        rho = np.hypot(dx, dz); x, y = np.arctan2(dx, dz), np.arctan2(dy, rho)
    else:  # fisheye
        n = np.linalg.norm(d, axis=1); ang = np.arccos(np.clip(dz / n, -1, 1)); r = np.hypot(dx, dy) + 1e-12
        x, y = ang * dx / r, ang * dy / r
    rr = x * x + y * y; s = 1 + k1 * rr + k2 * rr * rr
    return np.c_[u0 + f * x * s, v0 + f2 * y * s]


def _poly(uv, deg):
    u = (uv[:, 0] - LW / 2) / (LW / 2); v = (uv[:, 1] - LH / 2) / (LH / 2)
    return np.stack([u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)], 1)


def ajustar_inversa(modelo, q, xr, deg=6):
    """Polinomio píxel -> cancha ajustado con puntos de la cancha (más un margen) que caen dentro de la imagen."""
    gx, gy = np.meshgrid(np.linspace(xr[0], xr[1], 120), np.linspace(-2.5, 17.5, 80))
    XY = np.c_[gx.ravel(), gy.ravel()]
    uv = proyectar(modelo, q, XY)
    ok = np.isfinite(uv).all(1) & (uv[:, 0] > -40) & (uv[:, 0] < LW + 40) & (uv[:, 1] > -40) & (uv[:, 1] < LH + 40)
    c, *_ = np.linalg.lstsq(_poly(uv[ok], deg), XY[ok], rcond=None)
    err = np.linalg.norm(_poly(uv[ok], deg) @ c - XY[ok], axis=1)
    return {"deg": deg, "coef": c.tolist(), "err_med_m": float(np.median(err)), "err_max_m": float(err.max())}


class Lentes:
    """Dos tipos de calibración:
    - "modelo": cámara física por lente (ver proyectar) + polinomio inverso.
    - "warp": polinomio lente -> píxel de una panorámica ya calibrada (modelo cilíndrico de calib.py) -> cancha.
      Se obtiene emparejando jugadores entre el archivo original y una grabación de la panorámica."""
    def __init__(self, path):
        j = json.load(open(path)); self.tipo = j.get("tipo", "modelo")
        if self.tipo == "warp":
            from calib import Mapper
            self.deg = j["deg"]; self.c = {k: np.array(j[k]) for k in LENTES}; self.pano = Mapper(j["pano_calib"])
        else:
            self.modelo = j["modelo"]; self.q = {k: np.array(j[k]) for k in LENTES}
            self.inv = {k: (j["inv"][k]["deg"], np.array(j["inv"][k]["coef"])) for k in LENTES}

    def a_cancha(self, lente, uv):
        uv = np.atleast_2d(np.asarray(uv, float))
        if self.tipo == "warp":
            return self.pano.to_court(_poly(uv, self.deg) @ self.c[lente])
        deg, c = self.inv[lente]
        return _poly(uv, deg) @ c

    def a_imagen(self, lente, XY):
        return proyectar(self.modelo, self.q[lente], XY)


def guardar(path, modelo, q_arriba, q_abajo, extra=None):
    inv = {"arriba": ajustar_inversa(modelo, q_arriba, (-3, 17)), "abajo": ajustar_inversa(modelo, q_abajo, (11, 31))}
    j = {"modelo": modelo, "arriba": list(map(float, q_arriba)), "abajo": list(map(float, q_abajo)), "inv": inv}
    if extra: j.update(extra)
    json.dump(j, open(path, "w"), indent=1)
    return inv
