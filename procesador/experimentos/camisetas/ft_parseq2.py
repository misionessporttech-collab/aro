import cv2, json, re, sys, random, numpy as np, torch, time
sys.path.insert(0, '/home/claude/models/parseq')
from strhub.models.utils import create_model
torch.set_num_threads(2); random.seed(0); torch.manual_seed(0)
def load(d):
    L = json.load(open(f'{d}/labels.json')); M = {m['n']: m for m in json.load(open(f'{d}/meta.json'))}
    return [(cv2.imread(f'{d}/{int(n):03d}.png'), v, M[int(n)]['k']) for n, v in L.items() if v != 'o']
data = load('crops') + load('crops2')
train = [d for d in data if d[2] < 480]; test = [d for d in data if d[2] >= 480]
pos_tr = [d for d in train if d[1] != 'x']
neg_tr = [(c, '', k) for c, l, k in train if l == 'x']
print('train pos', len(pos_tr), 'test', len(test), 'test pos', sum(d[1] != 'x' for d in test))
def torso(c, jitter=False):
    h, w = c.shape[:2]
    a, b, l, r = 0.18, 0.80, 0.08, 0.92
    if jitter:
        a += random.uniform(-.08, .08); b += random.uniform(-.1, .1); l += random.uniform(-.06, .08); r += random.uniform(-.08, .06)
    return c[max(0, int(a * h)):int(b * h), max(0, int(l * w)):int(r * w)]
def prep(v, aug=False):
    if aug:
        if random.random() < .5:
            ang = random.uniform(-10, 10); M = cv2.getRotationMatrix2D((v.shape[1] / 2, v.shape[0] / 2), ang, 1); v = cv2.warpAffine(v, M, (v.shape[1], v.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        if random.random() < .5: v = cv2.GaussianBlur(v, (3, 3), random.uniform(.3, 1.2))
        v = np.clip(v.astype(np.float32) * random.uniform(.7, 1.3) + random.uniform(-25, 25), 0, 255).astype(np.uint8)
    h, w = v.shape[:2]; s = 32 / max(h, 1); v = cv2.resize(v, (max(1, int(w * s)), 32), interpolation=cv2.INTER_CUBIC)
    if v.shape[1] > 128: v = cv2.resize(v, (128, 32))
    pad = np.full((32, 128, 3), 127, np.uint8); x0 = (128 - v.shape[1]) // 2 if not aug else random.randint(0, 128 - v.shape[1]); pad[:, x0:x0 + v.shape[1]] = v
    x = torch.from_numpy(cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    return (x - 0.5) / 0.5
pm = create_model('parseq', False); pm.model.load_state_dict(torch.load('/home/claude/models/parseq.pt', map_location='cpu'))
pm.log = lambda *a, **k: None
def predict(c):
    vs = [torso(c)] + [torso(c, True) for _ in range(3)]
    with torch.no_grad():
        probs = pm(torch.stack([prep(v) for v in vs])).softmax(-1)
        preds, conf = pm.tokenizer.decode(probs)
    best = (None, 0)
    for p, cf in zip(preds, conf):
        s = float(cf.prod()) if len(cf) else 0
        if re.fullmatch(r'\d{1,2}', p) and s > best[1]: best = (p, s)
    return best
def evaluate(tag):
    pm.eval(); res = [(predict(c), lab) for c, lab, k in test]
    for thr in (0.5, 0.7, 0.85, 0.95):
        ok = sum(1 for (p, s), l in res if l != 'x' and p == l and s >= thr)
        bad = sum(1 for (p, s), l in res if l != 'x' and p is not None and p != l and s >= thr)
        fp = sum(1 for (p, s), l in res if l == 'x' and p is not None and s >= thr)
        npos = sum(1 for _, l in res if l != 'x'); nneg = len(res) - npos
        print(f'{tag} umbral {thr}: legibles {npos} -> bien {ok}, mal {bad} | ilegibles {nneg} -> lecturas falsas {fp}', flush=True)
    return res
opt = torch.optim.AdamW(pm.parameters(), lr=5e-5, weight_decay=0.01)
t0 = time.time()
for ep in range(18):
    pm.train(); ep_data = pos_tr + random.sample(neg_tr, min(len(neg_tr), len(pos_tr))); random.shuffle(ep_data); tot = 0
    for i in range(0, len(ep_data), 24):
        b = ep_data[i:i + 24]
        imgs = torch.stack([prep(torso(c, True), True) for c, l, k in b]); labs = [l for c, l, k in b]
        loss = pm.training_step((imgs, labs), 0); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    print('época', ep, 'pérdida', round(tot, 3), round(time.time() - t0), 's', flush=True)
torch.save(pm.model.state_dict(), '/home/claude/models/parseq_aro_neg.pt')
evaluate('después')
