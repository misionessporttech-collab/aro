"""Detección de personas en la panorámica a 10 fps. Guarda cajas en coordenadas de la imagen recortada."""
import subprocess, numpy as np, cv2, json, time, sys
from ultralytics import YOLO

SRC = "/mnt/user-data/uploads/Downloads/Clip 2.mp4"
X0, X1 = 355, 3687          # recorte de bandas negras
W, H = X1 - X0, 2160
VY = 0.55                   # compresión vertical (la panorámica viene estirada)
STEP = 3                    # 30 fps -> 10 fps

model = YOLO("yolo11s.pt")
cmd = ["ffmpeg", "-loglevel", "error", "-i", SRC, "-vf", f"select='not(mod(n\\,{STEP}))',crop={W}:{H}:{X0}:0,scale={W}:{int(H*VY)}",
       "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
hs = int(H * VY)
fsz = W * hs * 3
out = []
i = 0
t0 = time.time()
while True:
    buf = p.stdout.read(fsz)
    if len(buf) < fsz:
        break
    fr = np.frombuffer(buf, np.uint8).reshape(hs, W, 3)
    r = model.predict(fr, imgsz=2560, classes=[0], conf=0.15, verbose=False)[0]
    bx = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    bx[:, [1, 3]] /= VY
    # color del torso (para separar equipos): tomo la franja central superior de cada caja en la imagen comprimida
    feats = []
    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(float)
    for (x1, y1, x2, y2) in r.boxes.xyxy.cpu().numpy():
        h = y2 - y1; w = x2 - x1
        ty1, ty2 = int(y1 + 0.15 * h), int(y1 + 0.48 * h)
        tx1, tx2 = int(x1 + 0.22 * w), int(x2 - 0.22 * w)
        c = lab[max(ty1, 0):max(ty2, ty1 + 1), max(tx1, 0):max(tx2, tx1 + 1)].reshape(-1, 3)
        if len(c) == 0:
            feats.append([0] * 7); continue
        L, A, B = c[:, 0], c[:, 1] - 128, c[:, 2] - 128
        gold = ((B > 22) & (A > -8) & (A < 22) & (L > 90)).mean()
        white = ((L > 175) & (np.abs(A) < 14) & (np.abs(B) < 18)).mean()
        red = ((A > 22) & (L > 60)).mean()
        dark = (L < 80).mean()
        feats.append([gold, white, red, dark, float(np.median(L)), float(np.median(A)), float(np.median(B))])
    out.append({"f": i * STEP, "b": np.c_[bx, cf].round(1).tolist(), "c": np.round(feats, 3).tolist()})
    i += 1
    if i % 25 == 0:
        el = time.time() - t0
        print(f"{i} cuadros · {el/i:.2f} s/cuadro", flush=True)
        json.dump(out, open("dets.json", "w"))
json.dump(out, open("dets.json", "w"))
print("listo", i, "cuadros en", round(time.time() - t0), "s")
