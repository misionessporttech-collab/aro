import json, numpy as np, sys
sys.path.insert(0, '/home/claude/aro-repo/procesador')
from calib import Mapper
import procesar as P
from fitcam2 import C
F = json.load(open('fitC.json')); R = json.load(open('reads.json')); D = json.load(open('follow_dets.json'))
PP = json.load(open('pano_pos.json')); knots = [tuple(k) for k in PP['knots']]; pos = {int(k): v for k, v in PP['pos'].items()}
def pfr(t):
    tm = 720 + t; return int(round((tm - P.offset_at(knots, tm)) * 10))
# 1) calidad de la proyección: jugadores detectados en el seguimiento vs posiciones de la panorámica
errs = []
for k, f in enumerate(F):
    if not f['ok']: continue
    M = Mapper(np.r_[C, f['p']])
    feet = np.array([[(d[0] + d[2]) / 2, d[3]] for d in D[k]['d'] if d[3] - d[1] > 60])
    if not len(feet): continue
    XY = M.to_court(feet)
    pl = pos.get(pfr(f['t']), [])
    if not pl: continue
    PXY = np.array([[a[0], a[1]] for a in pl])
    for xy in XY:
        if not np.isfinite(xy).all() or not (-0.5 < xy[0] < 28.5 and -0.5 < xy[1] < 15.5): continue
        errs.append(np.min(np.linalg.norm(PXY - xy, axis=1)))
errs = np.array(errs); print('jugadores en cancha proyectados', len(errs), 'distancia al más cercano de la panorámica: mediana', np.median(errs).round(2), 'p75', np.percentile(errs, 75).round(2))
# 2) votos por recorrido
votes = {}; used = 0
for r in R:
    f = F[r['k']]
    if not f['ok']: continue
    M = Mapper(np.r_[C, f['p']]); x1, y1, x2, y2 = r['box']
    xy = M.to_court(np.array([[(x1 + x2) / 2, y2]]))[0]
    pl = [a for a in pos.get(pfr(r['t']), []) if a[2] == 'us']
    if not pl: continue
    d = np.array([np.hypot(a[0] - xy[0], a[1] - xy[1]) for a in pl]); o = np.argsort(d)
    if d[o[0]] > 1.2 or (len(o) > 1 and d[o[1]] < d[o[0]] + 0.6): continue
    votes.setdefault(pl[o[0]][3], []).append(r['j']); used += 1
print('lecturas asociadas', used, 'de', len(R), '| recorridos con votos', len(votes))
from collections import Counter
pur = [Counter(v).most_common(1)[0][1] / len(v) for v in votes.values() if len(v) >= 2]
print('pureza de votos (recorridos con 2+):', np.round(np.mean(pur), 2) if pur else None, len(pur))
json.dump({str(k): v for k, v in votes.items()}, open('votes.json', 'w'))
