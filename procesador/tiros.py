"""Aro · detección de tiros al aro en el archivo panorámico ORIGINAL de Veo (2048x2048, dos lentes).

Mira solo un recorte alrededor de cada aro, a 30 cuadros por segundo:
  1. separa lo que se mueve del fondo (la pelota, la red),
  2. sigue la pelota cuando llega a la altura del aro,
  3. decide si entró: no rebota hacia arriba y la red se sacude como en un tiro convertido,
  4. sigue la pelota hacia atrás en la imagen completa hasta las manos del tirador y lo ubica en la cancha
     (de ahí salen la posición del tiro, si fue de 2, de 3 o libre, y quién tiró).

La posición de cada aro en la imagen viene de la calibración de lentes de la cancha ("aros").

Uso:
  python tiros.py --video panoramica_original.mp4 --lentes calib_lentes_mitre.json --out trabajo/tiros.json
"""
import argparse, json, os, queue, subprocess, threading, time
import numpy as np, cv2

S = 300                      # lado del recorte normalizado (el aro queda en (150,200) y mide ~30 px)
RIM = (150.0, 200.0)
NET = (slice(206, 248), slice(128, 172))


def probe(video):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate:format=duration", "-of", "json", video])
    j = json.loads(out); s = j["streams"][0]; num, den = map(int, s["r_frame_rate"].split("/"))
    return int(s["width"]), int(s["height"]), num / den, float(j["format"]["duration"])


def recortes(aros, LH=1024):
    """Filtro de ffmpeg que arma, lado a lado, el recorte de cada aro escalado a SxS."""
    parts, labels = [f"[0:v]split={len(aros)}" + "".join(f"[s{i}]" for i in range(len(aros)))], []
    for i, a in enumerate(aros):
        lado = 10 * a["ancho"]
        x0 = a["x"] - lado / 2; y0 = a["y"] - lado * 2 / 3 + (LH if a["lente"] == "abajo" else 0)
        parts.append(f"[s{i}]crop={int(round(lado))}:{int(round(lado))}:{int(round(x0))}:{int(round(y0))},scale={S}:{S}[r{i}]")
        labels.append(f"[r{i}]")
    return ";".join(parts) + ";" + "".join(labels) + f"hstack={len(aros)}"


class Aro:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=20, detectShadows=False)
        r = cfg.get("reloj")   # tablero del reloj de posesión (cambia solo, no es pelota)
        self.clock = (slice(r[1], r[3]), slice(r[0], r[2])) if r else None
        self.K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def cuadro(self, roi):
        m = self.sub.apply(roi, learningRate=0.01)
        if self.clock: m[self.clock] = 0
        red = int((cv2.morphologyEx(m[NET], cv2.MORPH_OPEN, self.K3) > 0).sum())
        ctrl = int((m[0:60, :] > 0).sum() + (m[150:250, 0:60] > 0).sum())
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.K5)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        cs = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < 40 or a > 700 or a / (w * h) < 0.4 or not (0.4 < w / h < 2.5): continue
            cx, cy = cen[i]
            if 50 <= cx < 250 and 70 <= cy < 265: cs.append((float(cx), float(cy)))
        return cs, red, ctrl


def seguir(F, k, f0, p0, paso, maxlen=45, maxgap=5, dist=32):
    """Encadena posiciones de pelota desde (f0,p0) hacia adelante (+1) o atrás (-1)."""
    pts = [(f0, p0)]; v = np.zeros(2); f = f0; gap = 0
    while len(pts) < maxlen:
        f += paso
        if f < 0 or f >= len(F): break
        pred = np.array(pts[-1][1]) + v * abs(f - pts[-1][0])
        cs = F[f][k][0]
        if cs:
            d = [np.hypot(*(np.array(c) - pred)) for c in cs]
            j = int(np.argmin(d))
            if d[j] < dist + 6 * gap:
                nv = (np.array(cs[j]) - np.array(pts[-1][1])) / abs(f - pts[-1][0])
                v = nv if len(pts) == 1 else 0.5 * v + 0.5 * nv
                pts.append((f, cs[j])); gap = 0; continue
        gap += 1
        if gap > maxgap: break
    return pts


def detectar(F, fps, nombres, umbrales=None):
    """Convertido = la red se sacude fuerte (umbral propio de cada aro; el rebote no decide: la red también tira la pelota para arriba)."""
    umbrales = umbrales or [260] * len(nombres)
    ev = []
    for k, nombre in enumerate(nombres):
        last = -999
        for f in range(len(F)):
            if f - last < 45: continue
            hit = [c for c in F[f][k][0] if abs(c[0] - RIM[0]) <= 30 and 180 <= c[1] <= 200]
            if not hit: continue
            c = min(hit, key=lambda c: abs(c[0] - RIM[0]))
            back = seguir(F, k, f, c, -1, maxlen=25)
            if len(back) < 5: continue
            xs = [p[1][0] for p in back]; ys = [p[1][1] for p in back]
            if max(np.ptp(xs), np.ptp(ys)) < 25: continue          # no viene en vuelo
            ctrl = max(F[g][k][2] for g in range(max(0, f - 10), min(len(F), f + 10)))
            if ctrl > 2500: continue                                 # temblor de cámara o cambio de luz
            fwd = seguir(F, k, f, c, +1, maxlen=40)
            rebote = any(p[1] < c[1] - 8 and g - f <= 30 for g, p in fwd[1:])
            red = max(F[g][k][1] for g in range(f, min(len(F), f + 30)))
            ev.append({"t": round(f / fps, 2), "aro": nombre, "convertido": red >= umbrales[k],
                       "rebote": rebote, "red": red, "desde": [round(xs[-1]), round(ys[-1])], "llegada": [round(c[0]), round(c[1])],
                       "vuelo": round((back[0][0] - back[-1][0]) / fps, 2), "_k": k,
                       "_atras": [[g, round(p[0], 1), round(p[1], 1)] for g, p in back]})
            last = f
    return sorted(ev, key=lambda e: e["t"])


# ------------------------------------------------------------ tirador: la pelota hacia atrás hasta sus manos
LW, LH = 2048, 1024
HOOPS = {"izquierdo": (1.575, 7.5), "derecho": (28 - 1.575, 7.5)}


def roi_a_lente(aro, x, y):
    lado = 10 * aro["ancho"]; s = lado / S
    return aro["x"] - lado / 2 + x * s, aro["y"] - lado * 2 / 3 + y * s


def cuadros_lente(video, lente, t0, dur):
    y0 = 0 if lente == "arriba" else LH
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{max(0, t0):.3f}", "-i", video, "-t", f"{dur:.3f}",
           "-vf", f"crop={LW}:{LH}:0:{y0},format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    b = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
    n = len(b) // (LW * LH)
    return np.frombuffer(b[:n * LW * LH], np.uint8).reshape(n, LH, LW)


def hacia_atras(G, pts, maxback=75, R=40):
    """pts: [(i, x, y)] posiciones conocidas (en la lente) de la pelota llegando al aro. Devuelve el camino completo."""
    path = sorted((int(i), float(x), float(y)) for i, x, y in pts if 0 <= i < len(G))
    if len(path) < 3: return path
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    miss = 0; f = path[0][0]
    while f > 3 and len(path) < maxback:
        f -= 1
        q = np.array(path[:8], float)
        ax = np.polyfit(q[:, 0], q[:, 1], 1); ay = np.polyfit(q[:, 0], q[:, 2], 2 if len(q) >= 4 else 1)
        px, py = np.polyval(ax, f), np.polyval(ay, f)
        x0 = int(max(0, min(LW - 2 * R, px - R))); y0 = int(max(0, min(LH - 2 * R, py - R)))
        win = lambda k: G[k][y0:y0 + 2 * R, x0:x0 + 2 * R].astype(np.int16)
        a = win(f); d = np.minimum(np.abs(a - win(f - 3)), np.abs(a - win(min(len(G) - 1, f + 3))))
        m = cv2.morphologyEx(((d > 18) * 255).astype(np.uint8), cv2.MORPH_OPEN, K)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        best = None
        for i in range(1, n):
            if st[i][4] < 12: continue
            cx, cy = cen[i][0] + x0, cen[i][1] + y0; dd = np.hypot(cx - px, cy - py)
            if dd < R * 0.9 and (best is None or dd < best[0]): best = (dd, cx, cy, st[i][4])
        if best is None or best[3] > 900:
            miss += 1
            if miss >= 4: break
            continue
        miss = 0; path.insert(0, (f, float(best[1]), float(best[2])))
    return path


def tiradores(video, aros, ev, fps, desde, lentes_path):
    from ultralytics import YOLO
    from lentes import Lentes
    LEN = Lentes(lentes_path); yolo = YOLO("yolo11s.pt")
    for e in ev:
        aro = aros[e["_k"]]
        t_arr = e["t"]; t0 = t_arr - 2.6
        G = cuadros_lente(video, aro["lente"], desde + t0, 2.8)
        if len(G) < 20: continue
        f_arr0 = int(round(t_arr * fps)) - int(round(t0 * fps))   # índice local del cuadro de llegada
        g_arr = e["_atras"][0][0]
        pts = [(g - g_arr + f_arr0, *roi_a_lente(aro, x, y)) for g, x, y in e["_atras"]]
        path = hacia_atras(G, pts)
        if len(path) < 6: continue
        # vuelo libre: desde la llegada hacia atrás, mientras los puntos sigan una parábola; antes de eso, la pelota en las manos
        P = np.array(path, float); r = len(P) - 3
        while r > 0:
            q = P[r - 1:]
            ax = np.polyfit(q[:, 0], q[:, 1], 1); ay = np.polyfit(q[:, 0], q[:, 2], 2)
            res = np.hypot(np.polyval(ax, q[:, 0]) - q[:, 1], np.polyval(ay, q[:, 0]) - q[:, 2]).max()
            if res > 7: break
            r -= 1
        manos = P[:r]
        if len(manos) >= 3:   # punto más bajo con la pelota en las manos: a la altura de la cintura/pecho del tirador
            k = int(np.argmax(manos[:, 2])); i0, sx, sy = int(manos[k, 0]), manos[k, 1], manos[k, 2]; rel = 0.5
        else:                  # solo se vio el vuelo: el primer punto está cerca de las manos, arriba de la cabeza
            i0, sx, sy = int(P[0, 0]), P[0, 1], P[0, 2]; rel = -0.05
        frame = cv2.cvtColor(G[i0], cv2.COLOR_GRAY2BGR)
        bx = yolo.predict(frame, imgsz=1408, classes=[0], conf=0.25, verbose=False)[0].boxes.xyxy.cpu().numpy()
        best = None
        for x1, y1, x2, y2 in bx:
            w, h = x2 - x1, y2 - y1
            if not (x1 - 0.5 * w <= sx <= x2 + 0.5 * w): continue
            # la pelota en las manos puede estar a la altura de la cintura, arriba de la cabeza o picando junto a los pies
            sc = min(abs(sy - (y1 + r_ * h)) / h + (0 if r_ == rel else 0.06) for r_ in (rel, -0.05, 0.5, 1.0)) + 0.5 * abs(sx - (x1 + x2) / 2) / w
            if sc < 0.6 and (best is None or sc < best[0]): best = (sc, x1, y1, x2, y2)
        e["suelta_t"] = round(t0 + int(P[r, 0] if r < len(P) else P[0, 0]) / fps + desde, 2)
        e["tirador_t"] = round(t0 + i0 / fps + desde, 2)
        e["suelta_px"] = [round(sx), round(sy)]
        e["lente"] = aro["lente"]
        if best:
            _, x1, y1, x2, y2 = best
            e["tirador_box"] = [round(float(v), 1) for v in (x1, y1, x2, y2)]
            X, Y = LEN.a_cancha(aro["lente"], np.array([[(x1 + x2) / 2, y2]]))[0]
            hx, hy = HOOPS[aro["nombre"]]
            d = float(np.hypot(X - hx, Y - hy))
            e["tirador_xy"] = [round(float(X), 2), round(float(Y), 2)]
            e["distancia"] = round(d, 2)
            ft = abs(d - 4.225) < 0.7 and abs(Y - 7.5) < 0.8
            e["tipo"] = "libre" if ft else ("triple" if d > 6.6 else "doble")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--lentes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--desde", type=float, default=0); ap.add_argument("--max-seg", type=float, default=0)
    a = ap.parse_args()
    cfg = json.load(open(a.lentes, encoding="utf-8"))
    aros = cfg.get("aros")
    if not aros: raise SystemExit("La calibración de lentes no tiene la posición de los aros: no puedo detectar tiros.")
    W, H, fps, dur = probe(a.video)
    if (W, H) != (2048, 2048): raise SystemExit(f"Esperaba el archivo original de Veo (2048x2048) y este es {W}x{H}.")
    if a.max_seg: dur = min(dur, a.desde + a.max_seg)
    total = int((dur - a.desde) * fps)
    print(f"tiros: {len(aros)} aros, {total} cuadros a {fps:.2f} fps", flush=True)
    vf = recortes(aros)
    tail = ["-ss", f"{a.desde:.3f}", "-i", a.video, "-t", f"{dur - a.desde:.3f}", "-filter_complex", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    fsz = S * len(aros) * S * 3
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-hwaccel", "cuda"] + tail, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    def leer():
        buf = bytearray(fsz); mv = memoryview(buf); k = 0
        while k < fsz:
            r = p.stdout.readinto(mv[k:])
            if not r: return None
            k += r
        return buf

    first = leer()
    if first is None:
        p.kill(); p = subprocess.Popen(["ffmpeg", "-loglevel", "error"] + tail, stdout=subprocess.PIPE, bufsize=0); first = leer()
    cola = queue.Queue(maxsize=64)

    def lector():
        fr = first
        while fr is not None:
            cola.put(fr); fr = leer()
        cola.put(None)
    threading.Thread(target=lector, daemon=True).start()

    A = [Aro(x) for x in aros]
    F = []; t0 = time.time()
    while True:
        fr = cola.get()
        if fr is None: break
        img = np.frombuffer(fr, np.uint8).reshape(S, S * len(aros), 3)
        F.append([A[k].cuadro(img[:, S * k:S * (k + 1)]) for k in range(len(aros))])
        if len(F) % 9000 == 0:
            el = time.time() - t0
            print(f"tiros: {len(F)}/{total} cuadros · {len(F)/el:.0f} cuadros/s · faltan ~{(total-len(F))/(len(F)/el)/60:.0f} min", flush=True)
    p.kill()
    ev = detectar(F, fps, [x["nombre"] for x in aros], [x.get("umbral_red", 260) for x in aros])
    print(f"tiros: {len(ev)} intentos; ahora busco al tirador de cada uno", flush=True)
    t1 = time.time()
    try:
        tiradores(a.video, aros, ev, fps, a.desde, a.lentes)
    except Exception as ex:
        print("tiros: no pude ubicar a los tiradores:", ex, flush=True)
    print(f"tiros: tiradores ubicados en {sum('tirador_xy' in e for e in ev)} de {len(ev)} ({(time.time()-t1)/60:.0f} min)", flush=True)
    for e in ev:
        e["t"] = round(e["t"] + a.desde, 2); e.pop("_k", None); e.pop("_atras", None)
    json.dump({"version": 1, "fps": fps, "aros": aros, "tiros": ev}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    conv = sum(e["convertido"] for e in ev)
    print(f"tiros detectados: {len(ev)} ({conv} convertidos)", flush=True)


if __name__ == "__main__":
    main()
