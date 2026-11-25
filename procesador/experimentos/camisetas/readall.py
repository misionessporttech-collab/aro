# lee números con el modelo ajustado en todos los jugadores (ambos equipos) de la muestra
import cv2, json, re, sys, numpy as np, torch, time
sys.path.insert(0, '/home/claude/models/parseq')
from strhub.models.utils import create_model
torch.set_num_threads(1)
pm = create_model('parseq', False); pm.model.load_state_dict(torch.load('/home/claude/models/parseq_aro_neg.pt', map_location='cpu')); pm.eval()
def torso(c, a=0.18, b=0.80, l=0.08, r=0.92):
    h, w = c.shape[:2]; return c[max(0, int(a * h)):int(b * h), max(0, int(l * w)):int(r * w)]
def prep(v):
    h, w = v.shape[:2]; s = 32 / max(h, 1); v = cv2.resize(v, (max(1, int(w * s)), 32), interpolation=cv2.INTER_CUBIC)
    if v.shape[1] > 128: v = cv2.resize(v, (128, 32))
    pad = np.full((32, 128, 3), 127, np.uint8); x0 = (128 - v.shape[1]) // 2; pad[:, x0:x0 + v.shape[1]] = v
    x = torch.from_numpy(cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    return (x - 0.5) / 0.5
D = json.load(open('follow_dets.json'))
cap = cv2.VideoCapture('/mnt/user-data/uploads/Downloads/muestra_seguimiento.mp4')
out = []; i = 0; k = 0; t0 = time.time()
while True:
    ok = cap.grab()
    if not ok: break
    if i % 15 == 0:
        dets = [(j, d) for j, d in enumerate(D[k]['d']) if d[3] - d[1] > 70 and (d[5] > 0.12 or (d[6] > 0.2 and d[7] > 0.02))]
        if dets:
            _, fr = cap.retrieve(); batch = []; idx = []
            for j, d in dets:
                x1, y1, x2, y2 = map(int, d[:4]); h = y2 - y1
                c = fr[max(0, y1 - 2):y1 + int(.62 * h), max(0, x1 - 6):x2 + 6]
                for v in (torso(c), torso(c, .12, .7), torso(c, .25, .9)):
                    batch.append(prep(v)); idx.append(j)
            with torch.no_grad():
                preds, conf = pm.tokenizer.decode(pm(torch.stack(batch)).softmax(-1))
            best = {}
            for j, p, cf in zip(idx, preds, conf):
                s = float(cf.prod()) if len(cf) else 0
                if re.fullmatch(r'\d{1,2}', p) and s > best.get(j, (None, 0))[1]: best[j] = (p, s)
            for j, (p, s) in best.items():
                d = D[k]['d'][j]
                out.append({'k': k, 't': D[k]['t'], 'j': j, 'box': d[:4], 'team': 'us' if d[5] > 0.12 else 'opp', 'num': p, 'conf': round(s, 3)})
        k += 1
        if k % 100 == 0: print(k, len(out), round(time.time() - t0), flush=True)
    i += 1
json.dump(out, open('reads_parseq.json', 'w')); print('lecturas', len(out))
