import cv2, json, re, sys, numpy as np, torch, time
sys.path.insert(0, '/home/claude/models/parseq')
ROSTER = {"1","2","3","4","5","7","10","14","31","32","35"}
L = {int(k): v for k, v in json.load(open('crops/labels.json')).items()}
imgs = {n: cv2.imread(f'crops/{n:03d}.png') for n in L}

def torso_variants(c):
    h, w = c.shape[:2]; out = []
    for (a, b) in [(0.18, 0.80), (0.25, 0.95), (0.12, 0.65)]:
        for (l, r) in [(0.08, 0.92), (0.18, 0.82)]:
            out.append(c[int(a * h):int(b * h), int(l * w):int(r * w)])
    return out

def score(pred_fn, name):
    t0 = time.time(); ok = bad = none = fp = 0; conf_list = []
    for n, lab in L.items():
        p, c = pred_fn(imgs[n])
        conf_list.append((c, p, lab))
        if lab in ('x', 'o'):
            if p is not None: fp += 1
        else:
            if p is None: none += 1
            elif p == lab: ok += 1
            else: bad += 1
    print(f'{name}: legibles 76 -> bien {ok}, mal {bad}, sin lectura {none} | falsas en ilegibles/rivales {fp} de 164 | {time.time()-t0:.0f}s')
    return conf_list

from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()
def rapid_det(c):
    big = cv2.resize(c, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    r, _ = ocr(big); best = (None, 0)
    for _, t, s in (r or []):
        d = re.sub(r'\D', '', t)
        if d in ROSTER and float(s) > best[1]: best = (d, float(s))
    return best
def rapid_rec(c):
    best = (None, 0)
    for v in torso_variants(c)[::2]:
        if v.size == 0: continue
        big = cv2.resize(v, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        r, _ = ocr(big)
        for _, t, s in (r or []):
            d = re.sub(r'\D', '', t)
            if d in ROSTER and float(s) > best[1]: best = (d, float(s))
    return best

from strhub.models.utils import create_model
pm = create_model('parseq', False); pm.model.load_state_dict(torch.load('/home/claude/models/parseq.pt', map_location='cpu')); pm.eval()
def prep(v):
    h, w = v.shape[:2]; s = 32 / h; v = cv2.resize(v, (max(1, int(w * s)), 32), interpolation=cv2.INTER_CUBIC)
    if v.shape[1] > 128: v = cv2.resize(v, (128, 32))
    pad = np.full((32, 128, 3), 127, np.uint8); x0 = (128 - v.shape[1]) // 2; pad[:, x0:x0 + v.shape[1]] = v
    x = torch.from_numpy(cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    return (x - 0.5) / 0.5
def parseq_read(c):
    vs = [v for v in torso_variants(c) if v.size]
    with torch.no_grad():
        logits = pm(torch.stack([prep(v) for v in vs]))
        probs = logits.softmax(-1)
        preds, conf = pm.tokenizer.decode(probs)
    best = (None, 0)
    for p, cf in zip(preds, conf):
        d = re.sub(r'\D', '', p)
        s = float(cf.prod()) if len(cf) else 0
        if d in ROSTER and d == p.strip() and s > best[1]: best = (d, s)
    return best

R = {}
R['rapid_rec'] = score(rapid_rec, 'OCR genérico sobre 3 recortes del torso')
R['parseq'] = score(parseq_read, 'PARSeq (6 recortes)')
json.dump({k: v for k, v in R.items()}, open('evalread.json', 'w'))
