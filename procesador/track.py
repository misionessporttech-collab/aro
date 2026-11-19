"""Seguimiento, equipos, sincronización con jugadas etiquetadas, identidad y métricas físicas."""
import json, os, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from calib import Mapper

FPS = 10.0
dets = json.load(open("dets.json"))
M = Mapper(json.load(open("calib.json"))["params"])

# ---------- detecciones a metros ----------
FE = json.load(open("feats.json"))
dets = dets[:len(FE)]
rows = []
for fi, fr in enumerate(dets):
    for b, c in zip(fr["b"], FE[fi]):
        x1, y1, x2, y2, cf = b
        rows.append([fi, cf, (x1 + x2) / 2, y2, x1, y1, x2, y2, *c])
R = np.array(rows)
XY = M.to_court(R[:, 2:4])
inside = (XY[:, 0] > -1.0) & (XY[:, 0] < 29.0) & (XY[:, 1] > -0.35) & (XY[:, 1] < 15.8) & (R[:, 15] >= 15)
R, XY = R[inside], XY[inside]
F = R[:, 8:15]  # gold, white, red, dark, blue, medB, medL (solo píxeles del jugador)
feat = np.c_[F[:, 0] * 2, F[:, 1] * 2, F[:, 2] * 3, F[:, 3] * 2, F[:, 4], F[:, 5] / 25, F[:, 6] / 120]
km = KMeans(3, n_init=10, random_state=0).fit(feat)
cent = km.cluster_centers_
ref_c = int(np.argmax(cent[:, 3] * 2 - cent[:, 6]))
rest = [k for k in range(3) if k != ref_c]
mitre_c = max(rest, key=lambda k: cent[k, 0] + cent[k, 5] - cent[k, 2] - cent[k, 1])
lab = {k: ("ref" if k == ref_c else "mitre" if k == mitre_c else "suardi") for k in range(3)}
team = np.array([lab[k] for k in km.labels_])
print("centroides [gold, white, red, dark, blue, B, L]:", np.round(cent, 2).tolist(), lab)
print("detecciones por equipo:", {k: int((team == k).sum()) for k in ("mitre", "suardi", "ref")})

# ---------- seguimiento por "lugares": 5 por equipo, 3 árbitros ----------
class T:
    def __init__(s, i, p, fi, tm):
        s.id = i; s.p = p.copy(); s.v = np.zeros(2); s.last = fi; s.pts = {fi: p.copy()}; s.teams = [tm]; s.det = {}; s.team = tm
nF = len(dets)
idx_by_f = [np.where(R[:, 0] == f)[0] for f in range(nF)]
NSLOT = {"mitre": 5, "suardi": 5, "ref": 3}
done, nid = [], 0
slots = {k: [] for k in NSLOT}
for f in range(nF):
    ids = idx_by_f[f]
    for tmk, nmax in NSLOT.items():
        di = ids[team[ids] == tmk]
        P = XY[di]
        # soltar lugares perdidos hace más de 3 s
        keep = []
        for t in slots[tmk]:
            (keep if f - t.last <= 30 else done).append(t)
        slots[tmk] = keep
        used = set()
        if slots[tmk] and len(di):
            pred = np.array([t.p + t.v * min(f - t.last, 5) for t in slots[tmk]])
            Cm = np.linalg.norm(pred[:, None] - P[None], axis=2)
            gate = np.array([min(1.2 + 0.45 * (f - t.last - 1), 6.0) for t in slots[tmk]])[:, None]
            Cm[Cm > gate] = 1e6
            ri, ci = linear_sum_assignment(Cm)
            for a, b in zip(ri, ci):
                if Cm[a, b] >= 1e6: continue
                t = slots[tmk][a]; dt = f - t.last
                t.v = 0.7 * t.v + 0.3 * (P[b] - t.p) / dt
                t.p = P[b].copy(); t.last = f; t.pts[f] = P[b].copy(); t.det[f] = di[b]
                used.add(b)
        # abrir lugares nuevos solo si faltan jugadores de ese equipo
        free = [b for b in range(len(di)) if b not in used]
        free.sort(key=lambda b: -R[di[b], 1])
        for b in free:
            if len(slots[tmk]) >= nmax: break
            if slots[tmk] and np.min(np.linalg.norm(np.array([t.p for t in slots[tmk]]) - P[b], axis=1)) < 0.7: continue
            t = T(nid, P[b], f, tmk); t.det[f] = di[b]; nid += 1; slots[tmk].append(t)
for k in slots: done += slots[k]
tr = [t for t in done if len(t.pts) >= 10]
print("trayectorias:", len(tr), {k: sum(t.team == k for t in tr) for k in NSLOT},
      "· cobertura Mitre (jugador-segundos):", round(sum(len(t.pts) for t in tr if t.team == "mitre") / FPS / 5, 1), "de", round(nF / FPS, 1))

# ---------- jugadas etiquetadas (Kore) ----------
raw = open("/home/claude/kore/chunks.txt").read().replace("\n", "")
evs = []
for e in raw.split(";"):
    st, en, c, j, x, y = e.split(",")
    if x == "" or c[0] == "O": continue
    st = float(st); en = float(en) if en else st
    evs.append((st, en, c, j, float(x) * 0.28, (100 - float(y)) * 0.15))

def match_cost(T0, flipx=False, flipy=False):
    d = []
    for st, en, c, j, X, Y in evs:
        if flipx: X = 28 - X
        if flipy: Y = 15 - Y
        tm = (st + en) / 2
        f = int(round((tm - T0) * FPS))
        if f < 0 or f >= nF: continue
        ids = idx_by_f[f]; ids = ids[team[ids] == "mitre"]
        if len(ids) == 0: continue
        d.append(np.min(np.linalg.norm(XY[ids] - [X, Y], axis=1)))
    d = np.array(d)
    return (-(d < 1.0).sum() + 0.01 * (np.median(d) if len(d) else 9), len(d), float(np.median(d)) if len(d) else 99)

res = []
for fx in (False, True):
    for fy in (False, True):
        for T0 in np.arange(600, 900, 0.5):
            c = match_cost(T0, fx, fy); res.append((c[0], T0, fx, fy))
res.sort()
print("mejores candidatos de sincronización:", [(round(-r[0]), r[1], r[2], r[3]) for r in res[:5]])
_, T0, FX, FY = res[0]
fine = min(((match_cost(t, FX, FY)[0], t) for t in np.arange(T0 - 1, T0 + 1, 0.1)))
T0 = fine[1]
mc = match_cost(T0, FX, FY)
print("sincronización: el clip arranca en", round(T0, 1), "s del original · inversión X/Y:", FX, FY, "· jugadas en el clip:", mc[1], "· a <1 m:", round(-mc[0]), "· mediana", round(mc[2], 2), "m")
evs = [(st, en, c, j, (28 - X) if FX else X, (15 - Y) if FY else Y) for st, en, c, j, X, Y in evs]

# ---------- identidad: asignación conjunta trayectorias <-> camisetas ----------
mitre_tr = [t for t in tr if t.team == "mitre"]
evw = []
for st, en, c, j, X, Y in evs:
    if not j: continue
    f = int(round(((st + en) / 2 - T0) * FPS))
    if 0 <= f < nF: evw.append((f, j, X, Y))
lineup = sorted({j for _, j, _, _ in evw}, key=int)
score = np.zeros((len(mitre_tr), len(lineup)))
for f, j, X, Y in evw:
    for a, t in enumerate(mitre_tr):
        ks = [k for k in t.pts if abs(k - f) <= 3]
        if not ks: continue
        k = min(ks, key=lambda k: abs(k - f))
        d = np.linalg.norm(t.pts[k] - [X, Y])
        score[a, lineup.index(j)] += np.exp(-d * d / (2 * 1.2 ** 2))
if os.path.exists("ocr_votes.json"):
    ov = json.load(open("ocr_votes.json"))
    for a, t in enumerate(mitre_tr):
        for j, w in ov.get(str(t.id), {}).items():
            if j in lineup: score[a, lineup.index(j)] += 1.5 * w
print("quinteto según las jugadas:", lineup)
print("puntajes (fila = trayectoria, columna = camiseta):")
for a, t in enumerate(mitre_tr):
    print(f"  tray {t.id:3d} [{min(t.pts)/FPS:5.1f}-{max(t.pts)/FPS:5.1f}s] ", " ".join(f"#{j}:{score[a, b]:.2f}" for b, j in enumerate(lineup)))
order = sorted(range(len(mitre_tr)), key=lambda a: -len(mitre_tr[a].pts))
main = order[:len(lineup)]
ri, ci = linear_sum_assignment(-score[main])
ident = {}
for a, b in zip(ri, ci):
    ident[mitre_tr[main[a]].id] = lineup[b]
for a in order[len(lineup):]:
    t = mitre_tr[a]; a0, a1 = min(t.pts), max(t.pts)
    busy = {ident[u.id] for u in mitre_tr if u.id in ident and not (max(u.pts) < a0 or min(u.pts) > a1)}
    opts = [(score[a, b], j) for b, j in enumerate(lineup) if j not in busy]
    if opts:
        ident[t.id] = max(opts)[1]
print("identidades:", {k: v for k, v in ident.items()})

# ---------- métricas ----------
def series(t):
    ks = sorted(t.pts); P = np.array([t.pts[k] for k in ks]); ks = np.array(ks)
    full = np.arange(ks[0], ks[-1] + 1)
    Pi = np.c_[np.interp(full, ks, P[:, 0]), np.interp(full, ks, P[:, 1])]
    if len(full) >= 15:
        Pi = savgol_filter(Pi, 15, 2, axis=0)
    return full, Pi

players = {}
for t in tr:
    if t.team == "ref": continue
    key = ("mitre", ident[t.id]) if t.id in ident else (t.team, None)
    full, Pi = series(t)
    step = np.linalg.norm(np.diff(Pi, axis=0), axis=1)
    sp = step * FPS
    sp = np.clip(sp, 0, 9.5)
    k = key if key[1] else (t.team, "?")
    d = players.setdefault(k, {"dist": 0.0, "time": 0.0, "vmax": 0.0, "hi": 0.0, "sprints": 0, "n": 0})
    d["dist"] += float((sp / FPS).sum()); d["time"] += len(sp) / FPS; d["n"] += 1
    if len(sp) >= 10:
        sm = np.convolve(sp, np.ones(10) / 10, mode="valid")
        d["vmax"] = max(d["vmax"], float(sm.max()))
        d["hi"] += float((sp[sp > 5.0] / FPS).sum())
        above = sm > 5.5
        d["sprints"] += int(((~above[:-1]) & above[1:]).sum())

json.dump({"T0": T0, "players": {f"{a}|{b}": v for (a, b), v in players.items()},
           "tracks": [{"id": t.id, "team": t.team, "j": ident.get(t.id), "pts": {int(k): [round(float(v[0]), 2), round(float(v[1]), 2)] for k, v in t.pts.items()},
                       "det": {int(k): int(v) for k, v in t.det.items()}} for t in tr]}, open("tracks.json", "w"))
np.save("R.npy", R); np.save("XY.npy", XY); np.save("team.npy", team)
print()
print(f"{'Jugador':<18}{'Min':>6}{'Dist m':>9}{'m/min':>8}{'Vmax km/h':>11}{'Alta int. m':>12}{'Sprints':>9}")
for (tm, j), v in sorted(players.items(), key=lambda kv: (kv[0][0] != "mitre", kv[0][1] == "?", -kv[1]["dist"])):
    name = f"Mitre #{j}" if tm == "mitre" and j != "?" else ("Mitre (sin id)" if tm == "mitre" else "Suardi (sin id)")
    mins = v["time"] / 60
    print(f"{name:<18}{mins:6.1f}{v['dist']:9.0f}{(v['dist']/max(mins,1e-6)):8.0f}{v['vmax']*3.6:11.1f}{v['hi']:12.0f}{v['sprints']:9d}")
