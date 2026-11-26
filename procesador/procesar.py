"""Aro · procesador de seguimiento físico.

Toma las detecciones de la panorámica (detect.py + features.py), la calibración de la cancha
(calib.json) y el respaldo del partido exportado desde Aro (.json), y genera fisico.json:
recorridos de cada jugador, identidad por camiseta, métricas físicas y posiciones para el minimapa.

Uso:
  python procesar.py --partido respaldo.json [--dets dets.json --feats feats.json --calib calib.json --out fisico.json]
"""
import argparse, json, os, pickle, time, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from calib import Mapper

CW, CH = 28.0, 15.0
FPS = 10.0            # cuadros procesados por segundo (detect.py usa 1 de cada 3 a 30 fps)
POS_HZ = 2            # muestras por segundo guardadas para minimapa / mapa de calor
HI_SPEED = 5.0        # m/s (18 km/h) = alta intensidad
SPRINT = 5.5          # m/s (~20 km/h) sostenido 1 s = sprint
VMAX_CAP = 9.5        # m/s, tope físico


# ---------------------------------------------------------------- utilidades del partido (Aro)
def period_idx(game, t):
    return sum(1 for q in game.get("qEnd", []) if t > q)

def attack_right(game, t):
    p = period_idx(game, t); first = game.get("dir", "R") == "R"
    return first if p < 2 else ((not first) if game.get("flip", True) else first)

def aro_events(game):
    """Jugadas del equipo analizado con ubicación, en metros de la panorámica (misma orientación que el video)."""
    out = []
    for e in game["events"]:
        if e.get("team") == "opp" or e.get("x") is None or e["type"] == "ft":
            continue
        xa, ya = e["x"], e["y"]
        if not attack_right(game, e["t"]):
            xa, ya = 100 - xa, 100 - ya
        out.append((float(e["t"]), str(e["j"]) if e.get("j") else None, xa * CW / 100, ya * CH / 100))
    return out


# ---------------------------------------------------------------- detecciones
def _read_json(path):
    try:
        import orjson
        with open(path, "rb") as f: return orjson.loads(f.read())
    except ImportError:
        return json.load(open(path))

def floor_masks(lentes_path):
    """Polígono del piso de juego de cada lente (calibración de la cancha): deja afuera tribunas y bancos."""
    try: piso = json.load(open(lentes_path, encoding="utf-8")).get("piso")
    except Exception: return None
    if not piso: return None
    import cv2
    out = {}
    for k, n in enumerate(("arriba", "abajo")):
        M = np.zeros((1024, 2048), np.uint8); cv2.fillPoly(M, [np.array(piso[n], np.int32)], 1); out[k] = M > 0
    return out


def load(dets_path, feats_path, calib_path, masks=None):
    """Carga detecciones y colores por partes para no agotar la memoria con partidos completos."""
    import gc
    dets = _read_json(dets_path)
    cnt = np.array([len(d["b"]) for d in dets], np.int64)
    B = np.array([b for d in dets for b in d["b"]], np.float32).reshape(-1, 5)
    has_xy = bool(dets) and "xy" in dets[0]
    XYd = np.array([p for d in dets for p in d["xy"]], np.float64).reshape(-1, 2) if has_xy else None
    LEN = np.array([l for d in dets for l in d["l"]], np.int8) if has_xy else None
    sub = np.concatenate([np.arange(c) for c in cnt]) if len(cnt) else np.zeros(0, int)
    nD = len(dets); del dets; gc.collect()
    fe = _read_json(feats_path)
    nF = min(nD, len(fe))
    ok_f = np.array([len(fe[i]) == cnt[i] for i in range(nF)])
    F = np.array([c for i in range(nF) if ok_f[i] for c in fe[i]], np.float32)
    del fe; gc.collect()
    fidx = np.repeat(np.arange(nD), cnt)
    keep = (fidx < nF) & np.repeat(np.r_[ok_f, np.zeros(nD - nF, bool)], cnt)
    fidx, B, sub = fidx[keep], B[keep], sub[keep]
    x1, y1, x2, y2, cf = B.T
    R = np.c_[fidx, cf, (x1 + x2) / 2, y2, x1, y1, x2, y2, F].astype(np.float64)
    if has_xy:
        XY = XYd[keep]; lens = LEN[keep]
    else:
        M = Mapper(json.load(open(calib_path))["params"])
        XY = M.to_court(R[:, 2:4]); lens = np.zeros(len(R), np.int8)
    ok = (XY[:, 0] > -1.0) & (XY[:, 0] < CW + 1.0) & (XY[:, 1] > -0.35) & (XY[:, 1] < CH + 0.8) & (R[:, 15] >= 15)
    if masks is not None and has_xy:   # pies dentro del piso de juego de su lente
        fx = np.clip(R[:, 2].astype(int), 0, 2047); fy = np.clip(R[:, 3].astype(int), 0, 1023)
        inside = np.zeros(len(R), bool)
        for k, M in masks.items():
            m = lens == k; inside[m] = M[fy[m], fx[m]]
        ok &= inside
    orig = np.c_[fidx, sub][ok]
    return R[ok], XY[ok], nF, orig, lens[ok]

def classify(R, lens=None):
    """Equipo por color de camiseta. Con el original de Veo, cada lente tiene su propia luz: se agrupa por lente."""
    if lens is not None and len(np.unique(lens)) > 1:
        out = np.empty(len(R), dtype=object)
        for k in np.unique(lens):
            m = lens == k; out[m] = classify(R[m])
        return out.astype(str)
    F = R[:, 8:15]  # gold, white, red, dark, blue, medB, medL
    feat = np.c_[F[:, 0] * 2, F[:, 1] * 2, F[:, 2] * 3, F[:, 3] * 2, F[:, 4], F[:, 5] / 25, F[:, 6] / 120]
    km = KMeans(3, n_init=10, random_state=0).fit(feat)
    c = km.cluster_centers_
    ref = int(np.argmax(c[:, 3] * 2 - c[:, 6]))
    rest = [k for k in range(3) if k != ref]
    us = max(rest, key=lambda k: c[k, 0] + c[k, 5] - c[k, 2] - c[k, 1])
    lab = {k: ("ref" if k == ref else "us" if k == us else "opp") for k in range(3)}
    return np.array([lab[k] for k in km.labels_])


# ---------------------------------------------------------------- seguimiento por lugares
class Track:
    def __init__(s, i, p, f, team):
        s.id, s.team = i, team
        s.p, s.v, s.last = p.copy(), np.zeros(2), f
        s.pts, s.det = {f: p.copy()}, {}

def near_edge(p):
    return p[0] < 1.2 or p[0] > CW - 1.2 or p[1] < 1.2 or p[1] > CH - 1.2

def track(R, XY, team, nF):
    NS = {"us": 5, "opp": 5, "ref": 3}
    fr = R[:, 0].astype(np.int64)
    order = np.argsort(fr, kind="stable"); cuts = np.searchsorted(fr[order], np.arange(nF + 1))
    by_f = [order[cuts[f]:cuts[f + 1]] for f in range(nF)]
    slots = {k: [] for k in NS}; done = []; nid = 0
    for f in range(nF):
        ids = by_f[f]
        for tk, nmax in NS.items():
            di = ids[team[ids] == tk]; P = XY[di]
            keep = []
            for t in slots[tk]:
                lost = f - t.last
                limit = 18 if near_edge(t.p) else 30     # cerca del borde (posible cambio) -> se suelta antes
                (keep if lost <= limit else done).append(t)
            slots[tk] = keep
            used = set()
            if slots[tk] and len(di):
                pred = np.array([t.p + t.v * min(f - t.last, 5) for t in slots[tk]])
                C = np.linalg.norm(pred[:, None] - P[None], axis=2)
                gate = np.array([min(1.2 + 0.45 * (f - t.last - 1), 6.0) for t in slots[tk]])[:, None]
                C[C > gate] = 1e6
                for a, b in zip(*linear_sum_assignment(C)):
                    if C[a, b] >= 1e6: continue
                    t = slots[tk][a]; dt = f - t.last
                    t.v = 0.7 * t.v + 0.3 * (P[b] - t.p) / dt
                    t.p = P[b].copy(); t.last = f; t.pts[f] = P[b].copy(); t.det[f] = int(di[b]); used.add(b)
            free = sorted((b for b in range(len(di)) if b not in used), key=lambda b: -R[di[b], 1])
            for b in free:
                if len(slots[tk]) >= nmax: break
                if slots[tk] and np.min(np.linalg.norm(np.array([t.p for t in slots[tk]]) - P[b], axis=1)) < 0.7: continue
                t = Track(nid, P[b], f, tk); t.det[f] = int(di[b]); nid += 1; slots[tk].append(t)
    for k in slots: done += slots[k]
    done = split_on_contact(done)
    return [t for t in done if len(t.pts) >= 10], by_f


def split_on_contact(tracks, dmin=0.9, nmin=2):
    """Corta las trayectorias en cada cruce cercano entre compañeros: después del cruce
    la identidad es dudosa, así que cada tramo se identifica por separado."""
    out = []; nid = max([t.id for t in tracks] + [0]) + 1
    by_team = {}
    for t in tracks: by_team.setdefault(t.team, []).append(t)
    for team, ts in by_team.items():
        cuts = {t.id: set() for t in ts}
        span = [(min(t.pts), max(t.pts)) for t in ts]
        arr = [(np.array(sorted(t.pts)), np.array([t.pts[k] for k in sorted(t.pts)])) for t in ts]
        order = sorted(range(len(ts)), key=lambda i: span[i][0])
        for oi, i in enumerate(order):
            for j in order[oi + 1:]:
                if span[j][0] > span[i][1]: break
                a, b = ts[i], ts[j]
                (fa, pa), (fb, pb) = arr[i], arr[j]
                common, ia, ib = np.intersect1d(fa, fb, assume_unique=True, return_indices=True)
                if not len(common): continue
                dd = np.linalg.norm(pa[ia] - pb[ib], axis=1)
                close = [int(f) for f in common[dd < dmin]]
                run = []
                for f in close + [10**9]:
                    if run and f - run[-1] > 3:
                        if len(run) >= nmin:
                            c = run[len(run) // 2]
                            cuts[a.id].add(c); cuts[b.id].add(c)
                        run = []
                    run.append(f)
        for t in ts:
            if not cuts[t.id] or team == "ref":
                out.append(t); continue
            bounds = sorted(cuts[t.id])
            ks = sorted(t.pts)
            pieces = [[]]
            for k in ks:
                if bounds and k >= bounds[0]:
                    while bounds and k >= bounds[0]: bounds.pop(0)
                    pieces.append([])
                pieces[-1].append(k)
            for pc in pieces:
                if not pc: continue
                n = Track(nid, t.pts[pc[0]], pc[0], team); nid += 1
                n.pts = {k: t.pts[k] for k in pc}; n.det = {k: t.det[k] for k in pc if k in t.det}; n.last = pc[-1]
                out.append(n)
    return out


# ---------------------------------------------------------------- sincronización (por tramos)
def sync(evs, XY, team, by_f, nF, video_len, hint=None):
    pano_len = nF / FPS
    ev_t = np.array([e[0] for e in evs]); ev_xy = np.array([[e[2], e[3]] for e in evs])
    us_idx = [i[team[i] == "us"] for i in by_f]
    near = {}
    def nearest(f, xy_i):
        key = (f, xy_i)
        if key not in near:
            near[key] = float(np.min(np.linalg.norm(XY[us_idx[f]] - ev_xy[xy_i], axis=1))) if len(us_idx[f]) else None
        return near[key]

    def score(off, lo=0, hi=None, window=True):
        hi = pano_len if hi is None else hi
        sel = np.where((ev_t - off >= lo) & (ev_t - off < hi))[0]
        d = []
        for i in sel:
            f = int(round((ev_t[i] - off) * FPS))
            frames = range(f - 20, f + 6, 2) if window else (f - 7,)
            best_d = None
            for ff in frames:
                if 0 <= ff < nF:
                    dd = nearest(ff, i)
                    if dd is not None: best_d = dd if best_d is None else min(best_d, dd)
            if best_d is not None: d.append(best_d)
        d = np.array(d)
        return (int((d < 1.0).sum()), len(d), float(np.median(d)) if len(d) else 99.0)

    key = lambda o, **k: (score(o, **k)[0], -score(o, **k)[2])
    if hint is not None:
        cands = np.arange(hint - 180, hint + 180, 1.0)
    else:
        cands = np.arange(-pano_len, video_len, 4.0)
    coarse = sorted(cands, key=lambda o: key(o, window=False), reverse=True)[:5]
    best = max((o2 for o in coarse for o2 in np.arange(o - 4, o + 4, 0.5)), key=lambda o: key(o))
    best = max(np.arange(best - 0.6, best + 0.6, 0.1), key=lambda o: key(o))
    knots = []
    W = 300.0
    for lo in np.arange(0, pano_len, W):
        hi = min(lo + W, pano_len)
        loc = max(np.arange(best - 20, best + 20, 0.5), key=lambda o: (score(o, lo, hi)[0], -score(o, lo, hi)[2]))
        s = score(loc, lo, hi)
        if s[0] >= 8:
            knots.append(((lo + hi) / 2, float(loc), s[0], s[1]))
    if not knots:
        knots = [(pano_len / 2, float(best), *score(best)[:2])]
    g = score(best)
    if g[0] < 15:
        print(f"ATENCIÓN: sincronización con pocas coincidencias ({g[0]} de {g[1]} jugadas). Si el resultado no tiene sentido, indicá --inicio-aprox con el segundo del video del partido en que arranca la grabación panorámica.")
    return float(best), knots, {"coincidencias": g[0], "jugadas": g[1], "mediana_m": round(g[2], 2)}

def offset_at(knots, t):
    xs = [k[0] for k in knots]; ys = [k[1] for k in knots]
    return float(np.interp(t, xs, ys))


# ---------------------------------------------------------------- identidad
def read_votes(tracks, reads, orig, roster, min_conf=0.8):
    """Votos de número por recorrido a partir de las lecturas de camiseta (veo_raw.py)."""
    if not reads or orig is None: return {}
    row = {(int(f), int(i)): k for k, (f, i) in enumerate(orig)}
    owner = {}
    for t in tracks:
        for f, di in t.det.items(): owner[int(di)] = t
    V = {}
    for f, i, num, c in reads:
        if c < min_conf or (roster and num not in roster): continue
        k = row.get((int(f), int(i)))
        t = owner.get(k) if k is not None else None
        if t is None or t.team != "us": continue
        V.setdefault(t.id, {}); V[t.id][num] = V[t.id].get(num, 0.0) + c
    return V

def identify(tracks, evs, knots, nF, votes=None, use_events=True):
    us = [t for t in tracks if t.team == "us"]
    votes = votes or {}
    ev = []
    for t, j, X, Y in evs:
        if not j: continue
        # tiempo de la jugada en la panorámica
        pt = t - offset_at(knots, t)
        f = int(round(pt * FPS))
        if 0 <= f < nF: ev.append((f, j, X, Y))
    if not use_events: ev = []
    jerseys = sorted({e[1] for e in ev} | {j for v in votes.values() for j in v}, key=lambda s: (len(s), s))
    S = np.zeros((len(us), len(jerseys)))
    for a, t in enumerate(us):
        for j, w in votes.get(t.id, {}).items():
            S[a, jerseys.index(j)] += 2.0 * w      # una lectura confiable pesa más que una jugada cercana
    for f, j, X, Y in ev:
        for a, t in enumerate(us):
            ks = [k for k in range(f - 20, f + 6) if k in t.pts]
            if not ks: continue
            d = min(np.linalg.norm(t.pts[k] - [X, Y]) for k in ks)
            S[a, jerseys.index(j)] += np.exp(-d * d / (2 * 1.2 ** 2))
    ident, conf = {}, {}
    spans = {}
    def free(j, lo, hi):
        return all(hi < l2 or lo > h2 for l2, h2 in spans.get(j, []))
    # 1) asignaciones firmes
    pairs = sorted(((S[a, b], a, b) for a in range(len(us)) for b in range(len(jerseys))), reverse=True)
    for s, a, b in pairs:
        if s < 0.8: break
        t = us[a]
        if t.id in ident: continue
        lo, hi = min(t.pts), max(t.pts); j = jerseys[b]
        if not free(j, lo, hi): continue
        ident[t.id] = j; tot = S[a].sum()
        conf[t.id] = round(float(min(1.0, s / tot if tot else 0)), 2)
        spans.setdefault(j, []).append((lo, hi))
    # 2) el resto: mejor puntaje entre los números libres del quinteto activo en ese tramo
    for a in sorted(range(len(us)), key=lambda a: -len(us[a].pts)):
        t = us[a]
        if t.id in ident or len(t.pts) < 30 or not ev: continue
        lo, hi = min(t.pts), max(t.pts)
        active = {j for f, j, X, Y in ev if lo - 600 <= f <= hi + 600}
        opts = [(S[a, jerseys.index(j)], j) for j in active if free(j, lo, hi)]
        if not opts: continue
        s, j = max(opts)
        if s >= 0.15 or len(opts) == 1:
            ident[t.id] = j; conf[t.id] = 0.3 if s < 0.8 else 0.6
            spans.setdefault(j, []).append((lo, hi))
    return ident, conf


def agreement(tracks, evs, knots, ident):
    """Cuántas jugadas etiquetadas (con número) caen sobre un recorrido con ese mismo número."""
    us = [t for t in tracks if t.team == "us"]
    span = [(min(t.pts), max(t.pts)) for t in us]
    ok = bad = unk = 0
    for t_, j, X, Y in evs:
        if not j: continue
        f = int(round((t_ - offset_at(knots, t_)) * FPS)); best = None
        for k, t in enumerate(us):
            if f + 6 < span[k][0] or f - 20 > span[k][1]: continue
            ks = [q for q in range(f - 20, f + 6) if q in t.pts]
            if not ks: continue
            d = min(np.linalg.norm(t.pts[q] - [X, Y]) for q in ks)
            if d < 1.0 and (best is None or d < best[0]): best = (d, t)
        if best is None: continue
        g = ident.get(best[1].id)
        if g is None: unk += 1
        elif g == j: ok += 1
        else: bad += 1
    return {"acierto": round(ok / max(1, ok + bad), 2), "bien": ok, "mal": bad, "sin_numero": unk}

def ident_quality(tracks, evs, knots, nF, seeds=4, frac=0.3):
    """Prueba interna: esconde parte de las jugadas, identifica con el resto y mide cuántas
    de las escondidas caen sobre un recorrido con el número correcto."""
    import random
    us = [t for t in tracks if t.team == "us"]
    span = [(min(t.pts), max(t.pts)) for t in us]
    ev_j = [e for e in evs if e[1]]
    ok = bad = unk = 0
    for sd in range(seeds):
        rnd = random.Random(sd)
        hold = set(rnd.sample(range(len(ev_j)), int(len(ev_j) * frac)))
        ident, _ = identify(tracks, [e for i, e in enumerate(ev_j) if i not in hold], knots, nF)
        for i in hold:
            t_, j, X, Y = ev_j[i]
            f = int(round((t_ - offset_at(knots, t_)) * FPS)); best = None
            for k, t in enumerate(us):
                if f + 6 < span[k][0] or f - 20 > span[k][1]: continue
                ks = [q for q in range(f - 20, f + 6) if q in t.pts]
                if not ks: continue
                d = min(np.linalg.norm(t.pts[q] - [X, Y]) for q in ks)
                if d < 1.0 and (best is None or d < best[0]): best = (d, t)
            if best is None: continue
            g = ident.get(best[1].id)
            if g is None: unk += 1
            elif g == j: ok += 1
            else: bad += 1
    return {"acierto": round(ok / max(1, ok + bad), 2), "pruebas": ok + bad + unk}


# ---------------------------------------------------------------- métricas por recorrido
HOOP_XY = {"izquierdo": (1.575, CH / 2), "derecho": (CW - 1.575, CH / 2)}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def shots_from(path, knots, game, tracks=None, boxes=None, orig=None, lens=None, ident=None, lentes=None):
    """Tiros detectados en la panorámica original (tiros.py), en tiempo del video del partido, con equipo según el aro
    y, si se pudo seguir la pelota hasta las manos, el tirador (recorrido -> camiseta) y la distancia al aro."""
    j = json.load(open(path, encoding="utf-8")); out = []
    aros = j.get("aros") or []
    if lentes and os.path.exists(lentes):   # umbrales actuales de la cancha (pueden ser más nuevos que la detección)
        aros = json.load(open(lentes, encoding="utf-8")).get("aros") or aros
    umb = {x["nombre"]: x["umbral_red"] for x in aros if "umbral_red" in x}
    owner, rows_f = {}, {}
    if tracks is not None and orig is not None:
        for t in tracks:
            for f, di in t.det.items(): owner[int(di)] = t
        for k, f in enumerate(orig[:, 0]): rows_f.setdefault(int(f), []).append(k)
    ident = ident or {}
    for s in j["tiros"]:
        tv = s["t"] + offset_at(knots, s["t"])
        team = "us" if (s["aro"] == "derecho") == attack_right(game, tv) else "opp"
        made = s["red"] >= umb[s["aro"]] if s["aro"] in umb and "red" in s else bool(s["convertido"])
        o = {"t": round(tv, 1), "aro": s["aro"], "team": team, "made": bool(made), "rebote": bool(s.get("rebote"))}
        tb = s.get("tirador_box")
        if tb and owner:
            f0 = int(round(s.get("tirador_t", s["suelta_t"]) * FPS)); L = 0 if s.get("lente") == "arriba" else 1; best = None
            for df in (0, -1, 1, -2, 2):
                for k in rows_f.get(f0 + df, []):
                    if lens is not None and int(lens[k]) != L: continue
                    v = _iou(tb, boxes[k]) - 0.05 * abs(df)
                    if v > 0.3 and (best is None or v > best[0]): best = (v, owner.get(k), f0 + df)
            if best and best[1] is not None:
                t = best[1]
                fs = sorted(t.pts, key=lambda f: abs(f - best[2]))[0]; X, Y = t.pts[fs]
                hx, hy = HOOP_XY[s["aro"]]; d = float(np.hypot(X - hx, Y - hy))
                o.update({"j": ident.get(t.id) if t.team == "us" else None, "equipo_tirador": t.team,
                          "x": round(float(X), 2), "y": round(float(Y), 2), "dist": round(d, 2)})
        if "dist" not in o and s.get("tirador_xy"):
            X, Y = s["tirador_xy"]; hx, hy = HOOP_XY[s["aro"]]
            o.update({"x": X, "y": Y, "dist": round(float(np.hypot(X - hx, Y - hy)), 2)})
        # libre: jugadores quietos y uno parado en la línea de libres (si no se encontró al tirador, ese es el tirador)
        st = ft_setup(tracks, s.get("suelta_t", s["t"] - 1.0), s["aro"]) if tracks is not None else None
        if st and ("dist" not in o or abs(o["dist"] - 4.225) < 1.2):
            t, X, Y = st
            o.update({"tipo": "libre", "x": round(X, 2), "y": round(Y, 2), "dist": 4.23})
            if t.team == "us" and not o.get("j"): o["j"] = ident.get(t.id)
        elif "dist" in o:
            o["tipo"] = "triple" if o["dist"] > 6.6 else "doble"
        out.append(o)
    return warmup_filter(out, game)


def warmup_filter(shots, game):
    """Fuera del partido (antes del inicio, después del final) no cuentan; en los entretiempos hay entrada en calor:
    varias pelotas a la vez (tiros en los dos aros casi juntos) o muchos tiros seguidos."""
    t_ini = (game.get("start") or 0) - 5; t_fin = (game.get("qEnd") or [1e9])[-1] + 5 if len(game.get("qEnd") or []) >= 4 else 1e9
    shots = [s for s in shots if t_ini <= s["t"] <= t_fin]
    T = np.array([s["t"] for s in shots]); A = np.array([s["aro"] for s in shots])
    for i, s in enumerate(shots):
        otro = np.any((np.abs(T - s["t"]) < 6) & (A != s["aro"]))
        muchos = np.sum(np.abs(T - s["t"]) <= 30) >= 7
        if otro or muchos: s["calentamiento"] = True
    return shots


def ft_setup(tracks, t_rel, aro, win=2.0):
    """¿Formación de tiro libre al momento del lanzamiento? Devuelve (recorrido, x, y) del que está en la línea, o None."""
    fx = 5.8 if aro == "izquierdo" else CW - 5.8
    f1 = int(round(t_rel * FPS)); f0 = f1 - int(win * FPS)
    speeds, at_line = [], []
    for t in tracks:
        if t.team == "ref": continue
        a = next((t.pts[f0 + d] for d in (0, 1, -1, 2, -2) if f0 + d in t.pts), None)
        b = next((t.pts[f1 + d] for d in (0, -1, 1, -2, 2) if f1 + d in t.pts), None)
        if a is None or b is None: continue
        speeds.append(np.hypot(*(b - a)) / win)
        if np.hypot(b[0] - fx, b[1] - CH / 2) < 0.9: at_line.append((np.hypot(b[0] - fx, b[1] - CH / 2), t, b))
    if len(speeds) < 4 or not at_line or np.median(speeds) > 0.6: return None
    _, t, b = min(at_line, key=lambda z: z[0])
    return t, float(b[0]), float(b[1])


def shots_check(shots, game, win=(-1.5, 4.0)):
    """Control contra las jugadas etiquetadas a mano: tiros (fg/ft) propios y puntos del rival."""
    shots = [s for s in shots if not s.get("calentamiento")]
    ours = [e for e in game["events"] if e["type"] in ("fg", "ft") and e.get("team") != "opp"]
    opp = [e for e in game["events"] if e["type"] == "opp"]
    used = set(); r = {"propios": len(ours), "encontrados": 0, "resultado_ok": 0, "rival_puntos": len(opp), "rival_encontrados": 0}
    for e in ours:
        c = [i for i, s in enumerate(shots) if i not in used and s["team"] == "us" and win[0] <= e["t"] - s["t"] <= win[1]]
        if c:
            i = min(c, key=lambda i: abs(e["t"] - shots[i]["t"] - 1)); used.add(i); r["encontrados"] += 1
            r["resultado_ok"] += shots[i]["made"] == bool(e.get("d", {}).get("made"))
            if shots[i].get("j"): r["tirador_con_numero"] = r.get("tirador_con_numero", 0) + 1; r["tirador_ok"] = r.get("tirador_ok", 0) + (shots[i]["j"] == str(e.get("j")))
            if shots[i].get("tipo"):
                real = "libre" if e["type"] == "ft" else ("triple" if e.get("d", {}).get("pts") == 3 else "doble")
                r["tipo_con_dato"] = r.get("tipo_con_dato", 0) + 1; r["tipo_ok"] = r.get("tipo_ok", 0) + (shots[i]["tipo"] == real)
    for e in opp:
        c = [i for i, s in enumerate(shots) if i not in used and s["team"] == "opp" and s["made"] and -3 <= e["t"] - s["t"] <= 8]
        if c: used.add(min(c, key=lambda i: abs(e["t"] - shots[i]["t"]))); r["rival_encontrados"] += 1
    r["propios_sin_etiqueta"] = sum(1 for i, s in enumerate(shots) if s["team"] == "us" and i not in used)
    return r


def seg_metrics(t, knots, game):
    ks = np.array(sorted(t.pts)); P = np.array([t.pts[k] for k in ks])
    full = np.arange(ks[0], ks[-1] + 1)
    Pi = np.c_[np.interp(full, ks, P[:, 0]), np.interp(full, ks, P[:, 1])]
    if len(full) >= 15:
        Pi = savgol_filter(Pi, 15, 2, axis=0)
    step = np.linalg.norm(np.diff(Pi, axis=0), axis=1)
    sp = np.clip(step * FPS, 0, VMAX_CAP)
    t_pano = full[1:] / FPS
    t_vid = t_pano + np.interp(t_pano, [k[0] for k in knots], [k[1] for k in knots])
    per = np.searchsorted(np.array(game.get("qEnd", []), float), t_vid, side="left")
    nper = max(4, len(game.get("qEnd", [])) + 1)
    byP = [round(float((sp[per == i] / FPS).sum()), 1) for i in range(nper)]
    vmax, spr = 0.0, 0
    if len(sp) >= 10:
        sm = np.convolve(sp, np.ones(10) / 10, mode="valid")
        vmax = float(sm.max())
        above = sm > SPRINT
        spr = int(((~above[:-1]) & above[1:]).sum())
    # posiciones a POS_HZ, en decímetros
    stepk = int(FPS / POS_HZ)
    sub = Pi[::stepk]
    t0 = float(full[0] / FPS + offset_at(knots, full[0] / FPS))
    return {
        "dist": round(float((sp / FPS).sum()), 1), "dur": round(len(sp) / FPS, 1), "vmax": round(vmax, 2),
        "hi": round(float((sp[sp > HI_SPEED] / FPS).sum()), 1), "spr": spr, "byP": byP,
        "t0": round(t0, 2), "hz": POS_HZ,
        "x": [int(round(v * 10)) for v in sub[:, 0]], "y": [int(round(v * 10)) for v in sub[:, 1]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partido", required=True)
    ap.add_argument("--dets", default="dets.json"); ap.add_argument("--feats", default="feats.json")
    ap.add_argument("--calib", default="calib.json"); ap.add_argument("--out", default="fisico.json")
    ap.add_argument("--video-len", type=float, default=0)
    ap.add_argument("--lecturas", default="", help="lecturas de camiseta (veo_raw.py)")
    ap.add_argument("--inicio-aprox", type=float, default=None, help="segundo aproximado del video del partido en que arranca la panorámica")
    ap.add_argument("--tiros", default="", help="tiros detectados en la panorámica original (tiros.py)")
    ap.add_argument("--lentes", default="", help="calibración de lentes de la cancha (piso de juego de cada lente)")
    a = ap.parse_args()
    game = json.load(open(a.partido))
    t0 = time.time()
    cache = os.path.join(os.path.dirname(os.path.abspath(a.dets)), "seguimiento.pkl")
    evs = aro_events(game)
    vlen = a.video_len or (max(e["t"] for e in game["events"]) + 60)
    deps = [a.dets, a.feats] + [p for p in (a.calib, a.lentes) if p and os.path.exists(p)]
    if os.path.exists(cache) and os.path.getmtime(cache) > max(os.path.getmtime(x) for x in deps) and a.inicio_aprox is None:
        C = pickle.load(open(cache, "rb"))
        boxes, tracks, nF, T0, knots, q = C["boxes"], C["tracks"], C["nF"], C["T0"], C["knots"], C["q"]
        orig, lens = C.get("orig"), C.get("lens")
        print("uso el seguimiento ya calculado (seguimiento.pkl); para rehacerlo, borralo")
    else:
        masks = floor_masks(a.lentes) if a.lentes else None
        R, XY, nF, orig, lens = load(a.dets, a.feats, a.calib, masks)
        print(f"detecciones cargadas ({time.time() - t0:.0f} s)" + (" · solo dentro del piso de juego" if masks else ""))
        team = classify(R, lens if masks else None)
        print("detecciones por cuadro:", {k: round(float((team == k).sum()) / nF, 1) for k in ("us", "opp", "ref")})
        tracks, by_f = track(R, XY, team, nF)
        print("recorridos:", {k: sum(t.team == k for t in tracks) for k in ("us", "opp", "ref")}, f"({time.time() - t0:.0f} s)")
        T0, knots, q = sync(evs, XY, team, by_f, nF, vlen, a.inicio_aprox)
        boxes = R[:, 4:8].astype(np.float32)
        pickle.dump({"boxes": boxes, "tracks": tracks, "nF": nF, "T0": T0, "knots": knots, "q": q, "orig": orig, "lens": lens}, open(cache, "wb"))
    print("sincronización: la panorámica arranca en el segundo", round(T0, 1), "del video del partido ·", q, "· tramos:", len(knots), f"({time.time() - t0:.0f} s)")
    roster = {str(r.get("j")) for r in game.get("roster", []) if r.get("j") is not None}
    reads = json.load(open(a.lecturas)) if a.lecturas and os.path.exists(a.lecturas) else []
    votes = read_votes(tracks, reads, orig, roster)
    usT = [t for t in tracks if t.team == "us"]
    tot_us = max(1, sum(len(t.pts) for t in usT))
    if votes:
        # control independiente: identidad solo con camisetas, comparada con las jugadas etiquetadas
        id_cam, _ = identify(tracks, evs, knots, nF, votes, use_events=False)
        ag = agreement(tracks, evs, knots, id_cam)
        cov_cam = sum(len(t.pts) for t in usT if t.id in id_cam) / tot_us
        print(f"camisetas: {len(reads)} lecturas; solo con camisetas se identifica el {cov_cam*100:.0f}% del tiempo y coincide con "
              f"{ag['acierto']*100:.0f}% de las jugadas etiquetadas ({ag['bien']} bien, {ag['mal']} mal, {ag['sin_numero']} sin número)")
    ident, conf = identify(tracks, evs, knots, nF, votes)
    print("recorridos de", game["us"]["name"], "identificados:", len(ident), "de", len(usT))
    cov = sum(len(t.pts) for t in usT if t.id in ident) / tot_us
    if votes:
        iq = {"metodo": "camisetas", "acierto": ag["acierto"], "pruebas": ag["bien"] + ag["mal"], "cobertura": round(cov, 2), "cobertura_camisetas": round(cov_cam, 2)}
    else:
        iq = ident_quality(tracks, evs, knots, nF); iq["cobertura"] = round(cov, 2)
    print(f"identidad: {iq['cobertura']*100:.0f}% del tiempo con número asignado; control: acierta {iq['acierto']*100:.0f}% ({iq['pruebas']} jugadas)")
    if iq["acierto"] < 0.8:
        print("ATENCIÓN: la identificación por jugador es poco confiable en este partido; los totales del equipo sí son válidos.")
    segs = []
    for t in tracks:
        if t.team == "ref": continue
        m = seg_metrics(t, knots, game)
        m.update({"id": t.id, "team": t.team, "j": ident.get(t.id), "conf": conf.get(t.id, 0)})
        segs.append(m)
    # etiquetas por cuadro para el video con nombres (render.py)
    names = {r["j"]: r.get("n", "") for r in game.get("roster", [])}
    TEAMC = {"us": 0, "opp": 1, "ref": 2}
    frames = [[] for _ in range(nF)]
    for t in tracks:
        j = ident.get(t.id)
        lbl = (f"#{j} {names.get(j, '')}".strip() if j else ("?" if t.team == "us" else ""))
        for f, di in t.det.items():
            x1, y1, x2, y2 = boxes[di]
            frames[f].append([int(x1), int(y1), int(x2), int(y2), TEAMC[t.team], lbl, round(float(t.pts[f][0]), 2), round(float(t.pts[f][1]), 2)])
    json.dump({"fps": FPS, "colores": [game["us"]["color"], game["opp"]["color"]], "frames": frames},
              open(os.path.join(os.path.dirname(os.path.abspath(a.dets)), "etiquetas.json"), "w"), separators=(",", ":"), ensure_ascii=False)
    shots = shots_from(a.tiros, knots, game, tracks, boxes, orig, lens, ident, a.lentes) if a.tiros and os.path.exists(a.tiros) else []
    if shots:
        sc = shots_check(shots, game)
        print(f"tiros: {len(shots)} detectados ({sum(s['made'] for s in shots)} convertidos). Contra lo etiquetado: "
              f"{sc['encontrados']} de {sc['propios']} tiros propios encontrados, resultado correcto en {sc['resultado_ok']}; "
              f"{sc['rival_encontrados']} de {sc['rival_puntos']} anotaciones del rival; {sc['propios_sin_etiqueta']} tiros propios sin etiqueta; "
              f"tirador con número en {sc.get('tirador_con_numero', 0)} (bien {sc.get('tirador_ok', 0)}); tipo de tiro bien en {sc.get('tipo_ok', 0)} de {sc.get('tipo_con_dato', 0)}")
    out = {"version": 1, "fuente": "original" if votes else "panoramica", "T0": round(T0, 2), "sync": [[round(k[0], 1), round(k[1], 2)] for k in knots],
           "calidad": q, "identidad": iq, "umbral_alta": HI_SPEED, "umbral_sprint": SPRINT, "segs": segs,
           "cubre": [round(T0, 1), round(T0 + nF / FPS, 1)]}
    if shots: out["tiros"] = shots; out["tiros_control"] = sc
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    # resumen
    agg = {}
    for s in segs:
        if s["team"] != "us": continue
        k = s["j"] or "?"
        d = agg.setdefault(k, {"dist": 0, "dur": 0, "vmax": 0, "hi": 0, "spr": 0})
        d["dist"] += s["dist"]; d["dur"] += s["dur"]; d["vmax"] = max(d["vmax"], s["vmax"]); d["hi"] += s["hi"]; d["spr"] += s["spr"]
    print(f"\n{'Camiseta':<10}{'Min':>6}{'Dist m':>9}{'m/min':>8}{'Vmax km/h':>11}{'Alta int m':>12}{'Sprints':>9}")
    for k, d in sorted(agg.items(), key=lambda kv: -kv[1]["dist"]):
        mins = d["dur"] / 60
        print(f"{'#'+k:<10}{mins:6.1f}{d['dist']:9.0f}{d['dist']/max(mins,1e-6):8.0f}{d['vmax']*3.6:11.1f}{d['hi']:12.0f}{d['spr']:9d}")
    print("\narchivo:", a.out, round(os.path.getsize(a.out) / 1024), "KB")

if __name__ == "__main__":
    main()
