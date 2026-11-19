"""Video panorámico con el nombre sobre cada jugador y minimapa de la cancha.

Uso:
  python render.py --video panoramica.mp4 --salida carpeta/ [--desde 0 --hasta 0 --ancho 1920]
Usa carpeta/crop.json y carpeta/etiquetas.json (generados por pipeline.py y procesar.py).
"""
import argparse, json, os, subprocess, numpy as np, cv2

def hex2bgr(h):
    h = h.lstrip("#"); r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--salida", required=True)
    ap.add_argument("--desde", type=float, default=0); ap.add_argument("--hasta", type=float, default=0)
    ap.add_argument("--ancho", type=int, default=1920); ap.add_argument("--vy", type=float, default=0.55)
    a = ap.parse_args()
    crop = json.load(open(os.path.join(a.salida, "crop.json")))
    E = json.load(open(os.path.join(a.salida, "etiquetas.json")))
    fps_det = E["fps"]; frames = E["frames"]
    cols = [hex2bgr(c) for c in E["colores"]]
    cols = [c if sum(c) < 700 else (27, 100, 232) for c in cols] + [(160, 160, 160)]
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    OW = a.ancho // 2 * 2; s = OW / cw; OH = int(ch * a.vy * s) // 2 * 2
    out_fps = 30
    dur = (a.hasta or len(frames) / fps_det) - a.desde
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(a.desde), "-i", a.video, "-t", str(dur),
           "-vf", f"fps={out_fps},crop={cw}:{ch}:{crop[0]}:{crop[1]},scale={OW}:{OH}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    outp = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(out_fps), "-i", "-",
                             "-i", a.video, "-ss", str(a.desde), "-map", "0:v", "-map", "1:a?", "-shortest",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                             os.path.join(a.salida, "panoramica_con_etiquetas.mp4")], stdin=subprocess.PIPE)
    mw = int(OW * 0.19); ms = mw / 28.0; mh = int(15 * ms)
    n = 0
    while True:
        b = p.stdout.read(OW * OH * 3)
        if len(b) < OW * OH * 3: break
        img = np.frombuffer(b, np.uint8).reshape(OH, OW, 3).copy()
        fi = (a.desde + n / out_fps) * fps_det
        k = int(np.floor(fi)); w = fi - k
        A = frames[k] if 0 <= k < len(frames) else []
        B = frames[k + 1] if 0 <= k + 1 < len(frames) else []
        Bd = {(r[4], r[5]): r for r in B if r[5] and r[5] != "?"}
        ox, oy = 14, OH - mh - 14
        cv2.rectangle(img, (ox - 5, oy - 5), (ox + mw + 5, oy + mh + 5), (25, 25, 25), -1)
        cv2.rectangle(img, (ox, oy), (ox + mw, oy + mh), (70, 130, 190), -1)
        L = lambda x, y: (int(ox + x * ms), int(oy + y * ms))
        cv2.rectangle(img, L(0, 0), L(28, 15), (235, 235, 235), 1); cv2.line(img, L(14, 0), L(14, 15), (235, 235, 235), 1)
        cv2.circle(img, L(14, 7.5), int(1.8 * ms), (235, 235, 235), 1)
        for x0 in (0, 22.2): cv2.rectangle(img, L(x0, 5.05), L(x0 + 5.8, 9.95), (235, 235, 235), 1)
        for r in A:
            x1, y1, x2, y2, tc, lbl, X, Y = r
            if (tc, lbl) in Bd and w > 0:
                r2 = Bd[(tc, lbl)]
                x1, y1, x2, y2 = [v * (1 - w) + v2 * w for v, v2 in zip((x1, y1, x2, y2), r2[:4])]
                X, Y = X * (1 - w) + r2[6] * w, Y * (1 - w) + r2[7] * w
            col = cols[tc]
            X1, Y1, X2, Y2 = int(x1 * s), int(y1 * s * a.vy), int(x2 * s), int(y2 * s * a.vy)
            cv2.rectangle(img, (X1, Y1), (X2, Y2), col, 2)
            if tc == 0 and lbl:
                fs = 0.42 * OW / 1600
                (tw, th), _ = cv2.getTextSize(lbl, 0, fs, 1)
                cv2.rectangle(img, (X1, Y1 - th - 6), (X1 + tw + 6, Y1), col, -1)
                ink = (20, 20, 20) if sum(col) > 380 else (255, 255, 255)
                cv2.putText(img, lbl, (X1 + 3, Y1 - 4), 0, fs, ink, 1, cv2.LINE_AA)
            cv2.circle(img, L(X, Y), 5, col, -1); cv2.circle(img, L(X, Y), 5, (20, 20, 20), 1)
            if tc == 0 and lbl.startswith("#"):
                cv2.putText(img, lbl.split()[0][1:], (L(X, Y)[0] + 6, L(X, Y)[1] + 4), 0, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        outp.stdin.write(img.tobytes()); n += 1
    outp.stdin.close(); outp.wait()
    print("video listo:", os.path.join(a.salida, "panoramica_con_etiquetas.mp4"), n, "cuadros")

if __name__ == "__main__":
    main()
