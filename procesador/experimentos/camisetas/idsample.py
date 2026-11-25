import json, numpy as np, sys, pickle
from collections import Counter, defaultdict
sys.path.insert(0, '/home/claude/aro-repo/procesador')
from calib import Mapper
import procesar as P
sys.modules['__main__'].Track = P.Track
from fitcam2 import C
fitfile = sys.argv[1] if len(sys.argv) > 1 else 'fitC.json'
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
F = json.load(open(fitfile)); R = json.load(open('reads_parseq.json'))
Cache = pickle.load(open('/home/claude/full/trabajo/seguimiento.pkl', 'rb')); knots = Cache['knots']
us = [t for t in Cache['tracks'] if t.team == 'us']
game = json.load(open('/home/claude/respaldo_mitre.json'))
ROSTER = {p['n'] if isinstance(p, dict) and 'n' in p else str(p.get('num', '')) for p in game['roster']} if isinstance(game['roster'][0], dict) else set(map(str, game['roster']))
def pfr(t):
    tm = 720 + t; return int(round((tm - P.offset_at(knots, tm)) * 10))
mapper_cache = {}
votes = defaultdict(list); used = 0
for r in R:
    if r['team'] != 'us' or r['conf'] < THR: continue
    f = F[r['k']]
    if not f['ok']: continue
    M = mapper_cache.setdefault(r['k'], Mapper(np.r_[C, f['p']]))
    x1, y1, x2, y2 = r['box']; xy = M.to_court(np.array([[(x1 + x2) / 2, y2]]))[0]
    pf = pfr(r['t'])
    cands = [(np.linalg.norm(t.pts[pf] - xy), t.id) for t in us if pf in t.pts]
    if not cands: continue
    cands.sort()
    if cands[0][0] > 1.5 or (len(cands) > 1 and cands[1][0] < cands[0][0] + 0.7): continue
    votes[cands[0][1]].append(r['num']); used += 1
ident = {}
for tid, v in votes.items():
    num, n = Counter(v).most_common(1)[0]
    if n >= 2 and n / len(v) >= 0.6: ident[tid] = num
lo, hi = pfr(0), pfr(300)
tot = sum(sum(1 for f in t.pts if lo <= f <= hi) for t in us)
cov = sum(sum(1 for f in t.pts if lo <= f <= hi) for t in us if t.id in ident)
print(f'{fitfile} umbral {THR}: lecturas usadas {used}, recorridos con número {len(ident)}, tiempo de Mitre con número {cov / max(tot,1):.0%}')
# validación contra Kore en la ventana
ok = bad = unk = 0
old_ok = old_bad = 0
evs = [e for e in P.aro_events(game) if e[1] and 720 <= e[0] <= 1020]
ident_old, _ = P.identify(Cache['tracks'], [e for e in P.aro_events(game) if not (720 <= e[0] <= 1020)], knots, Cache['nF'])
for t_, j, X, Y in evs:
    f = int(round((t_ - P.offset_at(knots, t_)) * 10)); best = None
    for t in us:
        ks = [q for q in range(f - 20, f + 6) if q in t.pts]
        if not ks: continue
        d = min(np.linalg.norm(t.pts[q] - [X, Y]) for q in ks)
        if d < 1.0 and (best is None or d < best[0]): best = (d, t.id)
    if best is None: continue
    g = ident.get(best[1])
    if g is None: unk += 1
    elif g == j: ok += 1
    else: bad += 1
    go = ident_old.get(best[1])
    if go is not None: old_ok += go == j; old_bad += go != j
print(f'  contra jugadas de Kore en la ventana: bien {ok}, mal {bad}, sin número {unk}  | método anterior (jugadas de otros minutos): bien {old_ok}, mal {old_bad}')
