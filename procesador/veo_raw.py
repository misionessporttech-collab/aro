"""Aro · paso pesado para el archivo panorámico ORIGINAL de Veo (2048x2048: dos lentes, una arriba de la otra).

A 10 cuadros por segundo: detecta personas en cada lente, mide el color de la camiseta (restando el fondo),
las ubica en la cancha con la calibración de las lentes y, 2 veces por segundo, lee el número de camiseta.
Guarda el avance cada 200 cuadros y, si se corta, retoma donde quedó.

Uso:
  python veo_raw.py --video panoramica_original.mp4 --out trabajo/ --lentes calib_lentes_mitre.json
         [--modelo yolo11s.pt] [--lector lector_camisetas.pt --parseq-dir carpeta_parseq]
Salida en trabajo/: dets.json (cajas, posición en cancha y lente), feats.json (color), lecturas.json (números)
"""
import argparse, json, os, re, subprocess, sys, time, numpy as np, cv2
from lentes import Lentes, LENTES, LW, LH

FPS = 10.0


def probe(video):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate:format=duration", "-of", "json", video])
    j = json.loads(out); s = j["streams"][0]; num, den = map(int, s["r_frame_rate"].split("/"))
    return int(s["width"]), int(s["height"]), num / den, float(j["format"]["duration"])


def grab(video, t, size):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", "-vf", f"scale={size}:{size}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    b = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
    return np.frombuffer(b, np.uint8).reshape(size, size, 3) if len(b) == size * size * 3 else None


class Lector:
    """Lector de números de camiseta (PARSeq ajustado con camisetas de Aro)."""
    def __init__(self, pesos, parseq_dir, device):
        import torch
        sys.path.insert(0, parseq_dir)
        from strhub.models.utils import create_model
        self.torch = torch; self.dev = device
        self.m = create_model("parseq", False)
        sd = torch.load(pesos, map_location="cpu"); sd = {k: v.float() for k, v in sd.items()}
        self.m.model.load_state_dict(sd); self.m.eval().to(device)

    @staticmethod
    def _prep(v):
        h, w = v.shape[:2]; s = 32 / max(h, 1)
        v = cv2.resize(v, (max(1, int(w * s)), 32), interpolation=cv2.INTER_CUBIC)
        if v.shape[1] > 128: v = cv2.resize(v, (128, 32))
        pad = np.full((32, 128, 3), 127, np.uint8); x0 = (128 - v.shape[1]) // 2; pad[:, x0:x0 + v.shape[1]] = v
        return cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)

    def leer(self, recortes):
        """recortes: lista de imágenes del torso. Devuelve [(numero|None, confianza)]."""
        if not recortes: return []
        torch = self.torch; batch, idx = [], []
        for i, c in enumerate(recortes):
            h, w = c.shape[:2]
            for a, b in ((.18, .8), (.12, .7), (.25, .9)):
                v = c[int(a * h):int(b * h), int(.08 * w):int(.92 * w)]
                if v.size: batch.append(self._prep(v)); idx.append(i)
        out = [(None, 0.0)] * len(recortes)
        for s in range(0, len(batch), 512):
            x = torch.from_numpy(np.stack(batch[s:s + 512])).permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.5).to(self.dev)
            with torch.no_grad():
                preds, conf = self.m.tokenizer.decode(self.m(x).softmax(-1))
            for i, p, cf in zip(idx[s:s + 512], preds, conf):
                sc = float(cf.prod()) if len(cf) else 0.0
                if re.fullmatch(r"\d{1,2}", p) and sc > out[i][1]: out[i] = (p, sc)
        return out


def color_feats(lab, diff, bx, SC):
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
    return fe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lentes", required=True); ap.add_argument("--modelo", default="yolo11s.pt")
    ap.add_argument("--lector", default=""); ap.add_argument("--parseq-dir", default="")
    ap.add_argument("--imgsz", type=int, default=1408); ap.add_argument("--lote", type=int, default=4)
    ap.add_argument("--max-seg", type=float, default=0); ap.add_argument("--desde", type=float, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W, H, vfps, dur = probe(a.video)
    if (W, H) != (LW, 2 * LH): raise SystemExit(f"Esperaba el archivo panorámico original de Veo (2048x2048) y este es {W}x{H}.")
    if a.max_seg: dur = min(dur, a.desde + a.max_seg)
    print(f"video {W}x{H} a {vfps:.2f} fps, {dur/60:.1f} min", flush=True)
    LEN = Lentes(a.lentes)

    # fondo de cada lente (mediana de cuadros repartidos) para medir solo los píxeles del jugador
    SC = 0.5
    bgp = os.path.join(a.out, "fondo.png")
    if not os.path.exists(bgp):
        st = [grab(a.video, t, int(W * SC)) for t in np.linspace(a.desde + 5, dur - 5, 50)]
        cv2.imwrite(bgp, np.median(np.stack([s for s in st if s is not None]), axis=0).astype(np.uint8))
    bg = cv2.cvtColor(cv2.imread(bgp), cv2.COLOR_BGR2LAB).astype(np.int16)
    bgL = {"arriba": bg[:int(LH * SC)], "abajo": bg[int(LH * SC):]}

    import torch
    from ultralytics import YOLO
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(a.modelo)
    lector = Lector(a.lector, a.parseq_dir, dev) if a.lector and os.path.exists(a.lector) else None
    if lector is None: print("ATENCIÓN: sin lector de camisetas; solo detección y posiciones.", flush=True)

    dp, fp, rp = (os.path.join(a.out, n) for n in ("dets.json", "feats.json", "lecturas.json"))
    dets = json.load(open(dp)) if os.path.exists(dp) else []
    feats = json.load(open(fp)) if os.path.exists(fp) else []
    reads = json.load(open(rp)) if os.path.exists(rp) else []
    n = min(len(dets), len(feats)); dets, feats = dets[:n], feats[:n]
    reads = [r for r in reads if r[0] < n]
    total = int((dur - a.desde) * FPS)
    if n >= total:
        print("detección ya completa", flush=True); return
    start_t = a.desde + n / FPS
    base = ["ffmpeg", "-loglevel", "error"]
    tail = ["-ss", f"{start_t:.3f}", "-i", a.video, "-t", f"{dur - start_t:.3f}", "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(base + ["-hwaccel", "cuda"] + tail, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
    fsz = W * H * 3
    first = p.stdout.read(fsz)
    if len(first) < fsz:   # sin decodificación por placa de video: usar el procesador
        p.kill(); p = subprocess.Popen(base + tail, stdout=subprocess.PIPE, bufsize=10 ** 8); first = p.stdout.read(fsz)
    t0 = time.time(); done0 = n; batch = []
    OCR_CADA = 5   # cada 5 cuadros procesados = 2 por segundo

    def flush(batch):
        imgs = []
        for fr in batch: imgs += [fr[:LH], fr[LH:]]
        res = model.predict(imgs, imgsz=a.imgsz, classes=[0], conf=0.2, verbose=False, half=(dev == "cuda"))
        for k, fr in enumerate(batch):
            fidx = len(dets)
            small = cv2.resize(fr, (int(W * SC), int(H * SC)))
            lab_all = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)
            B, XYs, Ls, F = [], [], [], []
            for li, lente in enumerate(LENTES):
                r = res[2 * k + li]
                bx = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
                if not len(bx): continue
                foot = np.c_[(bx[:, 0] + bx[:, 2]) / 2, bx[:, 3]]
                XY = LEN.a_cancha(lente, foot)
                lab = lab_all[:int(LH * SC)] if li == 0 else lab_all[int(LH * SC):]
                diff = np.abs(lab - bgL[lente]).sum(2)
                fe = color_feats(lab, diff, bx, SC)
                for b, c, xy, f_ in zip(bx, cf, XY, fe):
                    if not (-2.5 < xy[0] < 30.5 and -2.5 < xy[1] < 17.5): continue
                    B.append([*map(float, b), float(c)]); XYs.append(xy); Ls.append(li); F.append(f_)
            # una misma persona vista por las dos lentes (zona del medio): queda la caja más grande
            keep = list(range(len(B)))
            for i in range(len(B)):
                for j in range(i + 1, len(B)):
                    if Ls[i] != Ls[j] and i in keep and j in keep and np.hypot(*(XYs[i] - XYs[j])) < 0.7:
                        hi, hj = B[i][3] - B[i][1], B[j][3] - B[j][1]
                        keep.remove(j if hi >= hj else i)
            B = [B[i] for i in keep]; XYs = [XYs[i] for i in keep]; Ls = [Ls[i] for i in keep]; F = [F[i] for i in keep]
            dets.append({"f": fidx, "b": np.round(B, 1).tolist(), "xy": np.round(XYs, 3).tolist(), "l": Ls}); feats.append(F)
            if lector is not None and fidx % OCR_CADA == 0 and B:
                crops, ids = [], []
                for i, (b, li) in enumerate(zip(B, Ls)):
                    x1, y1, x2, y2 = map(int, b[:4]); h = y2 - y1
                    if h < 45 or b[4] < 0.35: continue
                    img = fr[:LH] if li == 0 else fr[LH:]
                    c = img[max(0, y1 - 2):y1 + int(.62 * h), max(0, x1 - 4):x2 + 4]
                    if c.size: crops.append(c); ids.append(i)
                for i, (num, sc) in zip(ids, lector.leer(crops)):
                    if num is not None and sc >= 0.5: reads.append([fidx, i, num, round(sc, 3)])

    frames = [first]
    while len(dets) < total:
        if frames:
            batch.append(np.frombuffer(frames.pop(), np.uint8).reshape(H, W, 3))
        else:
            b = p.stdout.read(fsz)
            if len(b) < fsz: break
            batch.append(np.frombuffer(b, np.uint8).reshape(H, W, 3))
        if len(batch) == a.lote:
            flush(batch); batch = []
            if len(dets) % 200 < a.lote:
                json.dump(dets, open(dp, "w")); json.dump(feats, open(fp, "w")); json.dump(reads, open(rp, "w"))
                el = time.time() - t0; hechos = len(dets) - done0
                print(f"{len(dets)}/{total} cuadros · {hechos/el:.1f} cuadros/s · {len(reads)} números leídos · faltan ~{(total-len(dets))/max(hechos/el,1e-6)/60:.0f} min", flush=True)
    if batch: flush(batch)
    json.dump(dets, open(dp, "w")); json.dump(feats, open(fp, "w")); json.dump(reads, open(rp, "w"))
    open(os.path.join(a.out, "deteccion_completa.txt"), "w").write(str(len(dets)))
    print(f"detección completa: {len(dets)} cuadros, {len(reads)} números leídos", flush=True)


if __name__ == "__main__":
    main()
