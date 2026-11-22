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

def load(dets_path, feats_path, calib_path):
    """Carga detecciones y colores por partes para no agotar la memoria con partidos completos."""
    import gc
    dets = _read_json(dets_path)
    cnt = np.array([len(d["b"]) for d in dets], np.int64)
    B = np.array([b for d in dets for b in d["b"]], np.float32).reshape(-1, 5)
    nD = len(dets); del dets; gc.collect()
    fe = _read_json(feats_path)
    nF = min(nD, len(fe))
    ok_f = np.array([len(fe[i]) == cnt[i] for i in range(nF)])
    F = np.array([c for i in range(nF) if ok_f[i] for c in fe[i]], np.float32)
    del fe; gc.collect()
    fidx = np.repeat(np.arange(nD), cnt)
    keep = (fidx < nF) & np.repeat(np.r_[ok_f, np.zeros(nD - nF, bool)], cnt)
    fidx, B = fidx[keep], B[keep]
    x1, y1, x2, y2, cf = B.T
    R = np.c_[fidx, cf, (x1 + x2) / 2, y2, x1, y1, x2, y2, F].astype(np.float64)
    M = Mapper(json.load(open(calib_path))["params"])
    XY = M.to_court(R[:, 2:4])
    ok = (XY[:, 0] > -1.0) & (XY[:, 0] < CW + 1.0) & (XY[:, 1] > -0.35) & (XY[:, 1] < CH + 0.8) & (R[:, 15] >= 15)
    return R[ok], XY[ok], nF

def classify(R):
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
def identify(tracks, evs, knots, nF):
    us = [t for t in tracks if t.team == "us"]
    ev = []
    for t, j, X, Y in evs:
        if not j: continue
        # tiempo de la jugada en la panorámica
        pt = t - offset_at(knots, t)
        f = int(round(pt * FPS))
        if 0 <= f < nF: ev.append((f, j, X, Y))
    jerseys = sorted({e[1] for e in ev}, key=lambda s: (len(s), s))
    S = np.zeros((len(us), len(jerseys)))
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
        if t.id in ident or len(t.pts) < 30: continue
        lo, hi = min(t.pts), max(t.pts)
        active = {j for f, j, X, Y in ev if lo - 600 <= f <= hi + 600}
        opts = [(S[a, jerseys.index(j)], j) for j in active if free(j, lo, hi)]
        if not opts: continue
        s, j = max(opts)
        if s >= 0.15 or len(opts) == 1:
            ident[t.id] = j; conf[t.id] = 0.3 if s < 0.8 else 0.6
            spans.setdefault(j, []).append((lo, hi))
    return ident, conf


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
    ap.add_argument("--inicio-aprox", type=float, default=None, help="segundo aproximado del video del partido en que arranca la panorámica")
    a = ap.parse_args()
    game = json.load(open(a.partido))
    t0 = time.time()
    cache = os.path.join(os.path.dirname(os.path.abspath(a.dets)), "seguimiento.pkl")
    evs = aro_events(game)
    vlen = a.video_len or (max(e["t"] for e in game["events"]) + 60)
    if os.path.exists(cache) and os.path.getmtime(cache) > max(os.path.getmtime(a.dets), os.path.getmtime(a.feats), os.path.getmtime(a.calib)) \
            and a.inicio_aprox is None:
        C = pickle.load(open(cache, "rb"))
        boxes, tracks, nF, T0, knots, q = C["boxes"], C["tracks"], C["nF"], C["T0"], C["knots"], C["q"]
        print("uso el seguimiento ya calculado (seguimiento.pkl); para rehacerlo, borralo")
    else:
        R, XY, nF = load(a.dets, a.feats, a.calib)
        print(f"detecciones cargadas ({time.time() - t0:.0f} s)")
        team = classify(R)
        print("detecciones por cuadro:", {k: round(float((team == k).sum()) / nF, 1) for k in ("us", "opp", "ref")})
        tracks, by_f = track(R, XY, team, nF)
        print("recorridos:", {k: sum(t.team == k for t in tracks) for k in ("us", "opp", "ref")}, f"({time.time() - t0:.0f} s)")
        T0, knots, q = sync(evs, XY, team, by_f, nF, vlen, a.inicio_aprox)
        boxes = R[:, 4:8].astype(np.float32)
        pickle.dump({"boxes": boxes, "tracks": tracks, "nF": nF, "T0": T0, "knots": knots, "q": q}, open(cache, "wb"))
    print("sincronización: la panorámica arranca en el segundo", round(T0, 1), "del video del partido ·", q, "· tramos:", len(knots), f"({time.time() - t0:.0f} s)")
    ident, conf = identify(tracks, evs, knots, nF)
    print("recorridos de", game["us"]["name"], "identificados:", len(ident), "de", sum(t.team == "us" for t in tracks))
    usT = [t for t in tracks if t.team == "us"]
    cov = sum(len(t.pts) for t in usT if t.id in ident) / max(1, sum(len(t.pts) for t in usT))
    iq = ident_quality(tracks, evs, knots, nF); iq["cobertura"] = round(cov, 2)
    print(f"identidad: {iq['cobertura']*100:.0f}% del tiempo con número asignado; prueba interna: acierta {iq['acierto']*100:.0f}% ({iq['pruebas']} jugadas de control)")
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
    out = {"version": 1, "fuente": "panoramica", "T0": round(T0, 2), "sync": [[round(k[0], 1), round(k[1], 2)] for k in knots],
           "calidad": q, "identidad": iq, "umbral_alta": HI_SPEED, "umbral_sprint": SPRINT, "segs": segs,
           "cubre": [round(T0, 1), round(T0 + nF / FPS, 1)]}
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
