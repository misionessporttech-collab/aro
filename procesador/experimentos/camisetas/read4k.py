import json, re, sys, cv2, numpy as np, torch, time, subprocess
sys.path.insert(0, '/home/claude/models/parseq')
from strhub.models.utils import create_model
torch.set_num_threads(2)
pm = create_model('parseq', False); pm.model.load_state_dict(torch.load('/home/claude/models/parseq_aro_neg.pt', map_location='cpu')); pm.eval()
T = json.load(open('/home/claude/track/tracks.json')); R = np.load('/home/claude/track/R.npy')
SRC = "/mnt/user-data/uploads/Downloads/Clip 2.mp4"; X0 = 355; W = 3332; H = 2160
want = {}
for t in T['tracks']:
    if t['team'] != 'mitre': continue
    ks = sorted(int(k) for k in t['det'])
    for k in ks[::4]: want.setdefault(k, []).append((t['id'], t['det'][str(k)]))
def prep(v):
    h, w = v.shape[:2]; s = 32 / max(h, 1); v = cv2.resize(v, (max(1, int(w * s)), 32), interpolation=cv2.INTER_CUBIC)
    if v.shape[1] > 128: v = cv2.resize(v, (128, 32))
    pad = np.full((32, 128, 3), 127, np.uint8); x0 = (128 - v.shape[1]) // 2; pad[:, x0:x0 + v.shape[1]] = v
    x = torch.from_numpy(cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    return (x - 0.5) / 0.5
cmd = ["ffmpeg", "-loglevel", "error", "-i", SRC, "-vf", f"select='not(mod(n\\,3))',crop={W}:{H}:{X0}:0", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
out = []; heights = []; saved = 0; t0 = time.time()
for f in range(max(want) + 1):
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    if f not in want: continue
    img = np.frombuffer(b, np.uint8).reshape(H, W, 3)
    batch = []; meta = []
    for tid, di in want[f]:
        x1, y1, x2, y2 = R[di, 4:8].astype(int); hh = y2 - y1
        crop = img[max(0, y1):y1 + int(.62 * hh), max(0, x1 - 4):x2 + 4]
        if crop.size == 0: continue
        crop = cv2.resize(crop, (crop.shape[1], max(1, int(crop.shape[0] * 0.55))))  # deshacer el estiramiento
        heights.append(crop.shape[0] / 0.62)
        if saved < 96 and f % 12 == 0: cv2.imwrite(f'c_{saved:03d}_{tid}.png', crop); saved += 1
        h, w = crop.shape[:2]
        for a, bb in ((.18, .8), (.12, .7), (.25, .9)):
            batch.append(prep(crop[int(a * h):int(bb * h), int(.08 * w):int(.92 * w)])); meta.append(tid)
    if not batch: continue
    with torch.no_grad():
        preds, conf = pm.tokenizer.decode(pm(torch.stack(batch)).softmax(-1))
    best = {}
    for tid, pr, cf in zip(meta, preds, conf):
        s = float(cf.prod()) if len(cf) else 0
        if re.fullmatch(r'\d{1,2}', pr) and s > best.get(tid, ('', 0))[1]: best[tid] = (pr, s)
    for tid, (pr, s) in best.items(): out.append({'f': f, 'tid': tid, 'num': pr, 'conf': round(s, 3)})
json.dump(out, open('reads4k.json', 'w'))
print('lecturas', len(out), 'altura jugador px (natural) p10/50/90', np.percentile(heights, [10, 50, 90]).round(), round(time.time() - t0), 's')
from collections import Counter, defaultdict
v = defaultdict(list)
for r in out:
    if r['conf'] >= 0.85: v[r['tid']].append(r['num'])
for tid, l in sorted(v.items()): print(tid, Counter(l).most_common(3), len(l))
