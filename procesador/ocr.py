"""Lee números de camiseta de los jugadores de Mitre en resolución completa y genera votos por tracklet."""
import json, subprocess, numpy as np, cv2, re
from rapidocr_onnxruntime import RapidOCR

SRC = "/mnt/user-data/uploads/Downloads/Clip 2.mp4"
X0, W, H = 355, 3332, 2160
ROSTER = {"1", "2", "3", "4", "5", "7", "10", "14", "31", "32", "35"}
data = json.load(open("tracks.json"))
R = np.load("R.npy")
ocr = RapidOCR()
want = {}  # frame -> [(track id, det idx)]
for t in data["tracks"]:
    if t["team"] != "mitre": continue
    ks = sorted(int(k) for k in t["det"])
    for k in ks[::3]:
        want.setdefault(k, []).append((t["id"], t["det"][str(k)]))
cmd = ["ffmpeg", "-loglevel", "error", "-i", SRC, "-vf", f"select='not(mod(n\\,3))',crop={W}:{H}:{X0}:0", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
votes = {}
reads = 0
for f in range(max(want) + 1 if want else 0):
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    if f not in want: continue
    img = np.frombuffer(b, np.uint8).reshape(H, W, 3)
    for tid, di in want[f]:
        x1, y1, x2, y2 = R[di, 4:8].astype(int)
        hh = y2 - y1
        crop = img[max(y1 + int(0.1 * hh), 0):y1 + int(0.55 * hh), max(x1 - 5, 0):x2 + 5]
        if crop.size == 0 or crop.shape[0] < 30: continue
        crop = cv2.resize(crop, None, fx=2.5, fy=2.5 * 0.6, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(crop)
        for box, txt, conf in (res or []):
            digits = re.sub(r"[^0-9]", "", txt)
            if digits in ROSTER and float(conf) > 0.5:
                votes.setdefault(str(tid), {}).setdefault(digits, 0)
                votes[str(tid)][digits] += float(conf)
                reads += 1
json.dump(votes, open("ocr_votes.json", "w"))
print("lecturas válidas:", reads, "tracklets con lectura:", len(votes))
