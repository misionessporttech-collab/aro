import cv2, json, numpy as np, time
from fitcam2 import *
cap = cv2.VideoCapture('/mnt/user-data/uploads/Downloads/muestra_seguimiento.mp4')
base = PC[3:].copy()
def init_search(wm, pm):
    best = None
    for zoom in (2.5, 3.5, 4.5, 6):
        for du in np.linspace(-400, 400, 9):
            p = base.copy(); p[3] *= zoom; p[4] *= zoom
            uvc = cyl_forward(np.r_[C, p], np.array([[14, 7.5]]))[0]
            p[5] += W / 2 - uvc[0] + du; p[6] += H / 2 - uvc[1]
            out = refineC(p, wm, pm); sc = out[1] + out[2] + (0 if out[3] > 100 else 50)
            if best is None or sc < best[0]: best = (sc, out)
    return best[1]
res = []; prior = None; i = 0; t0 = time.time()
while True:
    ok = cap.grab()
    if not ok: break
    if i % 15 == 0:
        _, fr = cap.retrieve(); wm = wood_mask(fr); pm = paint_mask(fr)
        out = refineC(prior, wm, pm) if prior is not None else None
        if out is None or out[1] > 8 or out[3] < 100: out = init_search(wm, pm)
        p, mw, mp, nw, npp = out
        good = mw < 8 and nw > 100
        prior = p if good else None
        res.append({'t': round(i / 29.97, 3), 'p': list(map(float, p)), 'mw': round(mw, 2), 'mp': round(mp, 2), 'nw': nw, 'ok': bool(good)})
        if len(res) % 50 == 0: print(len(res), sum(r['ok'] for r in res), round(time.time() - t0), flush=True)
    i += 1
json.dump(res, open('fitC.json', 'w')); print('ok', sum(r['ok'] for r in res), 'de', len(res))
