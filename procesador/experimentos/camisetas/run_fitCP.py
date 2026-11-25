import cv2, json, numpy as np, time, sys
from fitcam2 import *
sys.path.insert(0, '/home/claude/aro-repo/procesador'); import procesar as P
F0 = json.load(open('fitC.json')); D = json.load(open('follow_dets.json'))
PP = json.load(open('pano_pos.json')); knots = [tuple(k) for k in PP['knots']]; pos = {int(k): v for k, v in PP['pos'].items()}
def pfr(t):
    tm = 720 + t; return int(round((tm - P.offset_at(knots, tm)) * 10))
cap = cv2.VideoCapture('/mnt/user-data/uploads/Downloads/muestra_seguimiento.mp4')
# punto de partida: el ajuste previo si estaba bien; si no, el vecino bueno más cercano
good = [k for k, f in enumerate(F0) if f['ok']]
res = []; i = 0; k = 0; t0 = time.time()
while True:
    ok = cap.grab()
    if not ok: break
    if i % 15 == 0:
        _, fr = cap.retrieve(); wm = wood_mask(fr); pmk = paint_mask(fr)
        src = k if F0[k]['ok'] else min(good, key=lambda g: abs(g - k))
        p0 = np.array(F0[src]['p'])
        if res and res[-1]['ok'] and not F0[k]['ok']: p0 = np.array(res[-1]['p'])
        pl = pos.get(pfr(F0[k]['t']), [])
        XYp = np.array([[a[0], a[1]] for a in pl]) if pl else np.zeros((0, 2))
        det = [d for d in D[k]['d'] if d[3] - d[1] > 60]
        feet = np.array([[(d[0] + d[2]) / 2, d[3]] for d in det]) if det else np.zeros((0, 2))
        p, mw, npl, pe = refineCP(p0, wm, pmk, XYp, feet)
        good_k = (mw < 8 and (npl < 3 or pe < 25)) or (npl >= 6 and pe < 12)
        res.append({'t': F0[k]['t'], 'p': list(map(float, p)), 'mw': round(mw, 2), 'npl': npl, 'pe': round(pe, 1), 'ok': bool(good_k)})
        k += 1
        if k % 100 == 0: print(k, sum(r['ok'] for r in res), round(time.time() - t0), flush=True)
    i += 1
json.dump(res, open('fitCP.json', 'w')); print('ok', sum(r['ok'] for r in res), 'de', len(res))
