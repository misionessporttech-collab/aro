"""Rasgos de color por detección usando solo los píxeles del jugador (fondo = mediana de la cámara fija)."""
import json, subprocess, numpy as np, cv2

SRC = "/mnt/user-data/uploads/Downloads/Clip 2.mp4"
X0, W, H = 355, 3332, 2160
SC = 0.5                      # trabajo a mitad de resolución (1666x1080): suficiente para color
w, h = int(W * SC), int(H * SC)
dets = json.load(open("dets.json"))
STEP = 3

def frames(select):
    cmd = ["ffmpeg", "-loglevel", "error", "-i", SRC, "-vf", f"select='{select}',crop={W}:{H}:{X0}:0,scale={w}:{h}", "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    while True:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3: break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)

# fondo: mediana de 90 cuadros repartidos
bg_stack = [f for f in frames("not(mod(n\\,39))")]
bg = np.median(np.stack(bg_stack[:90]), axis=0).astype(np.uint8)
cv2.imwrite("bg.jpg", bg)
bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.int16)

out = []
for i, fr in enumerate(frames(f"not(mod(n\\,{STEP}))")):
    if i >= len(dets): break
    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.int16)
    diff = np.abs(lab - bg_lab).sum(2)
    feats = []
    for (x1, y1, x2, y2, cf) in dets[i]["b"]:
        X1, Y1, X2, Y2 = int(x1 * SC), int(y1 * SC), int(x2 * SC), int(y2 * SC)
        hh = Y2 - Y1
        ty1, ty2 = Y1 + int(0.12 * hh), Y1 + int(0.62 * hh)     # torso + short
        sub = lab[max(ty1, 0):max(ty2, ty1 + 1), max(X1, 0):max(X2, X1 + 1)].reshape(-1, 3)
        m = diff[max(ty1, 0):max(ty2, ty1 + 1), max(X1, 0):max(X2, X1 + 1)].reshape(-1) > 45
        px = sub[m]
        if len(px) < 15:
            feats.append([0, 0, 0, 0, 0, 0, 0, int(m.sum())]); continue
        L, A, B = px[:, 0].astype(float), px[:, 1] - 128.0, px[:, 2] - 128.0
        gold = ((B > 20) & (A < 16) & (L > 110)).mean()
        white = ((L > 185) & (np.abs(A) < 12) & (B < 16)).mean()
        red = ((A > 18) & (L > 50)).mean()
        dark = (L < 75).mean()
        blue = ((B < -12)).mean()
        feats.append([round(gold, 3), round(white, 3), round(red, 3), round(dark, 3), round(blue, 3), round(float(np.median(B)), 1), round(float(np.median(L)), 1), int(m.sum())])
    out.append(feats)
    if i % 100 == 0: print(i, flush=True)
json.dump(out, open("feats.json", "w"))
print("listo", len(out))
