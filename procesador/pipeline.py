"""Aro · paso pesado del procesador: recorte automático, calibración, detección y color por jugador.

Recorre la grabación panorámica a 10 cuadros por segundo, detecta personas (YOLO) y mide el color
de cada una restando el fondo (la cámara está fija). Guarda el avance cada 200 cuadros y, si se corta,
retoma donde quedó.

Uso:
  python pipeline.py --video panoramica.mp4 --out carpeta/ [--calib-base calib_mitre.json] [--modelo yolo11s.pt]
Salida en carpeta/: dets.json, feats.json, calib.json, crop.json, bg.jpg
"""
import argparse, json, os, subprocess, time, numpy as np, cv2

def probe(video):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate:format=duration", "-of", "json", video])
    j = json.loads(out); s = j["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return int(s["width"]), int(s["height"]), num / den, float(j["format"]["duration"])

def grab(video, t, W, H, crop=None, scale=None):
    vf = []
    if crop: vf.append(f"crop={crop[2]-crop[0]}:{crop[3]-crop[1]}:{crop[0]}:{crop[1]}")
    if scale: vf.append(f"scale={scale[0]}:{scale[1]}")
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", video, "-frames:v", "1"] + (["-vf", ",".join(vf)] if vf else []) + \
          ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    b = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
    w = scale if scale else ((crop[2] - crop[0], crop[3] - crop[1]) if crop else (W, H))
    return np.frombuffer(b, np.uint8).reshape(w[1], w[0], 3) if len(b) == w[0] * w[1] * 3 else None

def auto_crop(video, W, H, dur):
    """Detecta las bandas negras (laterales y superior/inferior) con varios cuadros."""
    acc = None
    for t in np.linspace(dur * 0.1, dur * 0.9, 6):
        f = grab(video, t, W, H)
        if f is None: continue
        m = f.astype(np.float32).mean(2)
        acc = m if acc is None else np.maximum(acc, m)
    cols = np.where(acc.mean(0) > 12)[0]; rows = np.where(acc.mean(1) > 12)[0]
    return [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--calib-base", default=os.path.join(os.path.dirname(__file__), "calib_mitre.json"))
    ap.add_argument("--modelo", default="yolo11s.pt"); ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--vy", type=float, default=0.55, help="compresión vertical (la panorámica de Veo viene estirada)")
    ap.add_argument("--imgsz", type=int, default=2560); ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--max-seg", type=float, default=0, help="procesar solo los primeros N segundos (prueba)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W, H, vfps, dur = probe(a.video)
    if a.max_seg: dur = min(dur, a.max_seg)
    print(f"video {W}x{H} a {vfps:.2f} fps, {dur/60:.1f} min")

    # 1) recorte y calibración
    cp = os.path.join(a.out, "crop.json")
    crop = json.load(open(cp)) if os.path.exists(cp) else auto_crop(a.video, W, H, dur)
    json.dump(crop, open(cp, "w"))
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    print("zona útil:", crop, f"({cw}x{ch})")
    calp = os.path.join(a.out, "calib.json")
    if not os.path.exists(calp):
        import calib
        frame = grab(a.video, dur * 0.3, W, H, crop)
        cv2.imwrite(os.path.join(a.out, "cuadro_calibracion.png"), frame)
        base = json.load(open(a.calib_base))
        p = calib.refit_auto(frame, base, os.path.join(a.out, "calib_vis.jpg"))
        json.dump({"params": list(map(float, p)), "size": [cw, ch]}, open(calp, "w"))
        print("calibración guardada (revisá calib_vis.jpg: las líneas rojas tienen que caer sobre el borde de la cancha)")

    # 2) fondo (mediana de cuadros repartidos)
    SC = 0.5
    bw, bh = int(cw * SC) // 2 * 2, int(ch * SC) // 2 * 2
    bgp = os.path.join(a.out, "bg.png")
    if not os.path.exists(bgp):
        st = [grab(a.video, t, W, H, crop, (bw, bh)) for t in np.linspace(5, dur - 5, 60)]
        bg = np.median(np.stack([s for s in st if s is not None]), axis=0).astype(np.uint8)
        cv2.imwrite(bgp, bg)
    bg_lab = cv2.cvtColor(cv2.imread(bgp), cv2.COLOR_BGR2LAB).astype(np.int16)

    # 3) detección + color, con reanudación
    from ultralytics import YOLO
    model = YOLO(a.modelo)
    dp, fp = os.path.join(a.out, "dets.json"), os.path.join(a.out, "feats.json")
    dets = json.load(open(dp)) if os.path.exists(dp) else []
    feats = json.load(open(fp)) if os.path.exists(fp) else []
    n = min(len(dets), len(feats)); dets, feats = dets[:n], feats[:n]
    start_t = n / a.fps
    total = int(dur * a.fps)
    if n >= total:
        print("detección ya completa"); return
    hs = int(ch * a.vy)
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{start_t:.3f}", "-i", a.video, "-t", f"{dur - start_t:.3f}",
           "-vf", f"fps={a.fps},crop={cw}:{ch}:{crop[0]}:{crop[1]}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    fsz = cw * ch * 3
    t0 = time.time(); done0 = n
    batch = []
    def flush(batch):
        imgs = [cv2.resize(fr, (cw, hs)) for fr in batch]
        res = model.predict(imgs, imgsz=a.imgsz, classes=[0], conf=0.15, verbose=False)
        for fr, r in zip(batch, res):
            bx = r.boxes.xyxy.cpu().numpy().copy(); cf = r.boxes.conf.cpu().numpy()
            bx[:, [1, 3]] /= a.vy
            small = cv2.resize(fr, (bw, bh))
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)
            diff = np.abs(lab - bg_lab).sum(2)
            fe = []
            for (x1, y1, x2, y2) in bx:
                X1, Y1, X2, Y2 = int(x1 * SC), int(y1 * SC), int(x2 * SC), int(y2 * SC)
                hh = Y2 - Y1; ty1, ty2 = Y1 + int(0.12 * hh), Y1 + int(0.62 * hh)
                sub = lab[max(ty1, 0):max(ty2, ty1 + 1), max(X1, 0):max(X2, X1 + 1)].reshape(-1, 3)
                msk = diff[max(ty1, 0):max(ty2, ty1 + 1), max(X1, 0):max(X2, X1 + 1)].reshape(-1) > 45
                px = sub[msk]
                if len(px) < 15:
                    fe.append([0, 0, 0, 0, 0, 0, 0, int(msk.sum())]); continue
                L, A, B = px[:, 0].astype(float), px[:, 1] - 128.0, px[:, 2] - 128.0
                fe.append([round(((B > 20) & (A < 16) & (L > 110)).mean(), 3), round(((L > 185) & (np.abs(A) < 12) & (B < 16)).mean(), 3),
                           round(((A > 18) & (L > 50)).mean(), 3), round((L < 75).mean(), 3), round((B < -12).mean(), 3),
                           round(float(np.median(B)), 1), round(float(np.median(L)), 1), int(msk.sum())])
            dets.append({"f": len(dets), "b": np.c_[bx, cf].round(1).tolist()}); feats.append(fe)
    while len(dets) < total:
        b = p.stdout.read(fsz)
        if len(b) < fsz: break
        batch.append(np.frombuffer(b, np.uint8).reshape(ch, cw, 3))
        if len(batch) == a.lote:
            flush(batch); batch = []
            if len(dets) % 200 < a.lote:
                json.dump(dets, open(dp, "w")); json.dump(feats, open(fp, "w"))
                el = time.time() - t0; hechos = len(dets) - done0
                print(f"{len(dets)}/{total} cuadros · {hechos/el:.1f} cuadros/s · faltan ~{(total-len(dets))/max(hechos/el,1e-6)/60:.0f} min", flush=True)
    if batch: flush(batch)
    json.dump(dets, open(dp, "w")); json.dump(feats, open(fp, "w"))
    print("detección completa:", len(dets), "cuadros")

if __name__ == "__main__":
    main()
