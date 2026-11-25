"""Procesa en la PC (con la placa de video) cada carpeta de partido dentro de Aro\\partidos.

Cada carpeta necesita:
  - la grabación panorámica (.mp4 / .mkv / .mov; si hay varias, la que tenga "panor" en el nombre o la más grande)
  - el respaldo del partido exportado desde Aro (.json)
Deja en la misma carpeta: fisico.json (para importar en Aro) y la subcarpeta "trabajo" con lo intermedio.
"""
import ctypes, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.dirname(HERE)
PY = sys.executable
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi")
OUT_NAMES = {"fisico.json", "etiquetas.json"}


def ensure_deps():
    """Instala librerías nuevas que haya sumado una actualización del programa."""
    need = []
    for mod, pkg in (("orjson", "orjson"),):
        try: __import__(mod)
        except ImportError: need.append(pkg)
    if not need: return
    uv = os.path.join(os.path.expanduser("~"), "Aro", "herramientas", "uv", "uv.exe")
    if os.path.exists(uv):
        print("Instalando complementos nuevos:", ", ".join(need))
        subprocess.run([uv, "pip", "install", "--python", PY, *need])


def keep_awake(on):
    """Evita que Windows suspenda la PC mientras procesa."""
    try:
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def gpu_batch():
    try:
        import torch
        if not torch.cuda.is_available():
            print("ATENCIÓN: no se detecta la placa de video; va a funcionar pero muy lento.")
            return 2
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        name = torch.cuda.get_device_name(0)
        lote = 8 if mem >= 10 else 4 if mem >= 5 else 2
        print(f"Placa de video: {name} ({mem:.0f} GB) - lote {lote}")
        return lote
    except Exception as e:
        print("No pude revisar la placa de video:", e)
        return 2


def pick_inputs(folder):
    vids = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXT)]
    if not vids:
        return None, None, "falta la grabación panorámica"
    pan = [v for v in vids if "panor" in v.lower()]
    video = (pan or sorted(vids, key=lambda v: -os.path.getsize(os.path.join(folder, v))))[0]
    js = [f for f in os.listdir(folder) if f.lower().endswith(".json") and f.lower() not in OUT_NAMES]
    if not js:
        return None, None, "falta el respaldo del partido (.json exportado desde Aro)"
    resp = sorted(js, key=lambda f: -os.path.getmtime(os.path.join(folder, f)))[0]
    return os.path.join(folder, video), os.path.join(folder, resp), None


def run(cmd, log):
    print("  >", " ".join(os.path.basename(c) if os.sep in c else c for c in cmd[:3]), "...")
    with open(log, "a", encoding="utf-8") as lf:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        p = subprocess.Popen(cmd, cwd=PROC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print("   ", line.rstrip()); lf.write(line)
        return p.wait()


def make_videos(games):
    """Genera la panorámica con nombres para cada partido ya procesado que todavía no la tenga."""
    for g in games:
        name = os.path.basename(g); work = os.path.join(g, "trabajo")
        out = os.path.join(g, "panoramica_con_nombres.mp4")
        video, resp, err = pick_inputs(g)
        print(f"\n=== {name} ===")
        if err or not os.path.exists(os.path.join(work, "etiquetas.json")):
            print("  Salteo: primero procesá el partido con procesar.bat."); continue
        if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(os.path.join(work, "etiquetas.json")):
            print("  Ya está hecho:", out); continue
        log = os.path.join(g, "proceso.log"); t0 = time.time()
        print("  Armando el video con los nombres (usa la placa de video para comprimir)...")
        ok = run([PY, "render.py", "--video", video, "--salida", work, "--archivo", out, "--codec", "h264_nvenc"], log) == 0
        if not ok:
            print("  La placa no pudo comprimir el video; pruebo con el procesador (más lento)...")
            ok = run([PY, "render.py", "--video", video, "--salida", work, "--archivo", out], log) == 0
        if ok:
            print(f"  LISTO en {(time.time() - t0) / 60:.0f} min: {out}")
            print("  Subilo a YouTube (oculto) y pegá el link en Aro: Etiquetar > Ajustes > Link de YouTube de la panorámica con nombres.")
        else:
            print("  ERROR al generar el video: mirá proceso.log en la carpeta del partido.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else os.path.join(os.path.expanduser("~"), "Aro", "partidos")
    ensure_deps()
    # deja al día los accesos directos (.bat) de la carpeta Aro
    import shutil
    for b in ("procesar.bat", "generar_video.bat"):
        try: shutil.copyfile(os.path.join(HERE, b), os.path.join(os.path.dirname(os.path.abspath(root)), b))
        except Exception: pass
    games = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if not games:
        print(f"No hay carpetas de partidos en {root}. Creá una y poné adentro la panorámica y el respaldo .json.")
        return
    keep_awake(True)
    if "--video" in sys.argv:
        try: make_videos(games)
        finally: keep_awake(False)
        return
    lote = gpu_batch()
    try:
        for g in games:
            name = os.path.basename(g)
            out = os.path.join(g, "fisico.json")
            video, resp, err = pick_inputs(g)
            print(f"\n=== {name} ===")
            if err:
                print("  Salteo:", err); continue
            ver = os.path.getmtime(os.path.join(PROC, "procesar.py"))
            if os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(video), os.path.getmtime(resp), ver):
                print("  Ya procesado (fisico.json está al día). Para rehacerlo, borrá fisico.json."); continue
            work = os.path.join(g, "trabajo"); os.makedirs(work, exist_ok=True)
            log = os.path.join(g, "proceso.log")
            t0 = time.time()
            print("  Video:", os.path.basename(video), "| Respaldo:", os.path.basename(resp))
            done = os.path.join(work, "deteccion_completa.txt")
            if os.path.exists(done) and os.path.getmtime(done) > os.path.getmtime(video):
                print("  Paso 1/2: la detección de jugadores ya está hecha, la reutilizo.")
            else:
                print("  Paso 1/2: detectar jugadores (la parte larga). Si se corta, volvé a abrir procesar.bat y sigue desde donde quedó.")
                if run([PY, "pipeline.py", "--video", video, "--out", work, "--modelo", "yolo11s.pt", "--lote", str(lote)], log):
                    print("  ERROR en la detección: mirá proceso.log en la carpeta del partido."); continue
            print("  Paso 2/2: seguimiento, identidad y métricas.")
            if run([PY, "procesar.py", "--partido", resp, "--dets", os.path.join(work, "dets.json"), "--feats", os.path.join(work, "feats.json"),
                    "--calib", os.path.join(work, "calib.json"), "--out", out], log):
                print("  ERROR en el análisis: mirá proceso.log en la carpeta del partido."); continue
            print(f"  LISTO en {(time.time() - t0) / 60:.0f} min: {out}")
            print("  Importalo en Aro: Análisis > Rendimiento físico > Importar análisis físico (.json).")
            print("  Revisá también trabajo\\calib_vis.jpg: las líneas rojas tienen que caer sobre el borde de la cancha.")
    finally:
        keep_awake(False)


if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    os.system("chcp 65001 >nul") if os.name == "nt" else None
    main()
