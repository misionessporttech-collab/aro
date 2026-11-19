"""Video de muestra: panorámica con recuadro + nombre de cada jugador y minimapa de la cancha."""
import json, subprocess, numpy as np, cv2, sys

SRC = "/mnt/user-data/uploads/Downloads/Clip 2.mp4"
X0, W, H = 355, 3332, 2160
VY = 0.55
OW = 1600
sx = OW / W; sy = sx * VY / 1.0
OH = int(H * VY * sx) // 2 * 2
data = json.load(open("tracks.json"))
R = np.load("R.npy")
names = {"1": "Amprino R", "2": "Clara V", "3": "Montes", "4": "Tantos L", "5": "Perez D", "7": "Corsi B", "10": "Lopez M",
         "14": "Nuñez L", "31": "Tonarelli", "32": "Dade M", "35": "Barber R"}
COL = {"mitre": (40, 200, 255), "suardi": (60, 60, 230), "ref": (160, 160, 160)}

# cajas por cuadro de detección (10 fps)
boxes = {}
for t in data["tracks"]:
    lbl = (f"#{t['j']} {names.get(t['j'], '')}".strip() if t["j"] else ("Mitre" if t["team"] == "mitre" else "Suardi" if t["team"] == "suardi" else "Árbitro"))
    for f, di in t["det"].items():
        x1, y1, x2, y2 = R[di, 4:8]
        boxes.setdefault(int(f), []).append((t["id"], t["team"], lbl, np.array([x1, y1, x2, y2]), t["pts"].get(f)))

def draw_minimap(img, items):
    mw, mh = 300, 161; ox, oy = 14, OH - mh - 14
    s = mw / 28.0
    cv2.rectangle(img, (ox - 6, oy - 6), (ox + mw + 6, oy + mh + 6), (25, 25, 25), -1)
    cv2.rectangle(img, (ox, oy), (ox + mw, oy + int(15 * s)), (70, 130, 190), -1)
    L = lambda x, y: (int(ox + x * s), int(oy + y * s))
    c = (235, 235, 235)
    cv2.rectangle(img, L(0, 0), L(28, 15), c, 1); cv2.line(img, L(14, 0), L(14, 15), c, 1)
    cv2.circle(img, L(14, 7.5), int(1.8 * s), c, 1)
    for side in (0, 1):
        x0 = 0 if side == 0 else 28 - 5.8
        cv2.rectangle(img, L(x0, 5.05), L(x0 + 5.8, 9.95), c, 1)
    for tid, tm, lbl, b, p in items:
        if p is None: continue
        cv2.circle(img, L(*p), 6, COL[tm], -1); cv2.circle(img, L(*p), 6, (20, 20, 20), 1)
        if lbl.startswith("#"):
            cv2.putText(img, lbl.split()[0][1:], (L(*p)[0] + 7, L(*p)[1] + 4), 0, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

cmd = ["ffmpeg", "-loglevel", "error", "-i", SRC, "-vf", f"crop={W}:{H}:{X0}:0,scale={OW}:{OH}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
outp = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", "30", "-i", "-",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "muestra_seguimiento.mp4"], stdin=subprocess.PIPE)
n = 0
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
while n < limit:
    buf = p.stdout.read(OW * OH * 3)
    if len(buf) < OW * OH * 3: break
    img = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3).copy()
    fi = n / 3.0; a = int(np.floor(fi)); w = fi - a
    A = {b[0]: b for b in boxes.get(a, [])}; B = {b[0]: b for b in boxes.get(a + 1, [])}
    items = []
    for tid, it in A.items():
        bb = it[3]; pp = it[4]
        if tid in B and w > 0:
            bb = bb * (1 - w) + B[tid][3] * w
            if pp is not None and B[tid][4] is not None: pp = list(np.array(pp) * (1 - w) + np.array(B[tid][4]) * w)
        items.append((tid, it[1], it[2], bb, pp))
    for tid, tm, lbl, bb, pp in items:
        x1, y1, x2, y2 = bb
        X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sx * VY), int(x2 * sx), int(y2 * sx * VY)
        col = COL[tm]
        cv2.rectangle(img, (X1, Y1), (X2, Y2), col, 2)
        if tm != "ref":
            (tw, th), _ = cv2.getTextSize(lbl, 0, 0.42, 1)
            cv2.rectangle(img, (X1, Y1 - th - 6), (X1 + tw + 6, Y1), col, -1)
            cv2.putText(img, lbl, (X1 + 3, Y1 - 4), 0, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    draw_minimap(img, items)
    outp.stdin.write(img.tobytes()); n += 1
outp.stdin.close(); outp.wait()
print("cuadros", n)
