"""Procesa en la PC (con la placa de video) cada carpeta de partido dentro de Aro\\partidos.

Cada carpeta necesita:
  - el archivo panorámico original de Veo (2048x2048, dos lentes) o, si no está, una grabación de pantalla de la panorámica
  - el respaldo del partido exportado desde Aro (.json)
Deja en la misma carpeta: fisico.json (para importar en Aro) y la subcarpeta "trabajo" con lo intermedio.
"""
import ctypes, glob, json, os, shutil, subprocess, sys, time, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.dirname(HERE)
PY = sys.executable
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi")
OUT_NAMES = {"fisico.json", "etiquetas.json", "calib_lentes.json"}
ARO = os.path.join(os.path.expanduser("~"), "Aro")
PARSEQ_COMMIT = "1902db0"
PARSEQ_DIR = os.path.join(ARO, "herramientas", "parseq")
LECTOR = os.path.join(ARO, "modelos", "lector_camisetas_v1.pt")


def video_size(path):
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                       "-of", "csv=p=0", path], text=True).strip()
        w, h = out.split(",")[:2]; return int(w), int(h)
    except Exception:
        return 0, 0


def ensure_lector():
    """El modelo del lector puede llegar a Descargas o a la carpeta de partidos: lo ubica en Aro\\modelos."""
    if os.path.exists(LECTOR): return True
    for d in (os.path.join(os.path.expanduser("~"), "Downloads"), os.path.join(ARO, "partidos")):
        f = os.path.join(d, os.path.basename(LECTOR))
        if os.path.exists(f):
            os.makedirs(os.path.dirname(LECTOR), exist_ok=True)
            try: shutil.move(f, LECTOR); print("  Lector de números instalado en Aro\\modelos."); return True
            except Exception as e: print("  No pude mover el lector de números:", e)
    return False


def ensure_parseq():
    """Código del lector de números (PARSeq, licencia Apache 2.0)."""
    if os.path.exists(os.path.join(PARSEQ_DIR, "strhub")): return True
    try:
        print("Descargando el lector de números (una sola vez)...")
        tmpd = os.environ.get("TEMP", ARO)
        z = os.path.join(tmpd, "parseq.zip")
        urllib.request.urlretrieve(f"https://github.com/baudm/parseq/archive/{PARSEQ_COMMIT}.zip", z)
        tmp = os.path.join(tmpd, "parseq_x")
        shutil.rmtree(tmp, ignore_errors=True); zipfile.ZipFile(z).extractall(tmp)
        src = os.path.join(tmp, os.listdir(tmp)[0]); shutil.rmtree(PARSEQ_DIR, ignore_errors=True); shutil.move(src, PARSEQ_DIR)
        return True
    except Exception as e:
        print("  No pude descargar el lector de números:", e); return False


def adopt_original(folder):
    """Si el archivo original de Veo quedó en Descargas, lo mueve a la carpeta del partido."""
    if any(f.lower().endswith(VIDEO_EXT) and "original" in f.lower() for f in os.listdir(folder)): return
    dl = os.path.join(os.path.expanduser("~"), "Downloads")
    name = os.path.basename(folder).lower().split("-")[0]
    for f in glob.glob(os.path.join(dl, "*panoramica_original*.mp4")):
        b = os.path.basename(f).lower()
        if name in b and "muestra" not in b:
            print(f"  Muevo {os.path.basename(f)} de Descargas a la carpeta del partido.")
            try: os.replace(f, os.path.join(folder, os.path.basename(f)))
            except Exception as e: print("   no pude moverlo:", e)
            return


def ensure_deps():
    """Instala librerías nuevas que haya sumado una actualización del programa."""
    need = []
    for mod, pkg in (("orjson", "orjson"), ("timm", "timm"), ("pytorch_lightning", "pytorch-lightning"), ("nltk", "nltk"), ("yaml", "pyyaml")):
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
    vids = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXT) and "con_nombres" not in f.lower()]
    if not vids:
        return None, None, "falta la grabación panorámica"
    orig = [v for v in vids if video_size(os.path.join(folder, v)) == (2048, 2048)]
    pan = [v for v in vids if "panor" in v.lower()]
    video = (orig or pan or sorted(vids, key=lambda v: -os.path.getsize(os.path.join(folder, v))))[0]
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
            adopt_original(g)
            video, resp, err = pick_inputs(g)
            print(f"\n=== {name} ===")
            if err:
                print("  Salteo:", err); continue
            original = video_size(video) == (2048, 2048)
            ver = os.path.getmtime(os.path.join(PROC, "procesar.py"))
            try: fuente = json.load(open(out, encoding="utf-8")).get("fuente", "") if os.path.exists(out) else ""
            except Exception: fuente = ""
            if original and fuente != "original":
                print("  Hay un archivo original de Veo nuevo: rehago el análisis con él.")
            elif os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(video), os.path.getmtime(resp), ver):
                print("  Ya procesado (fisico.json está al día). Para rehacerlo, borrá fisico.json."); continue
            work = os.path.join(g, "trabajo_original" if original else "trabajo"); os.makedirs(work, exist_ok=True)
            log = os.path.join(g, "proceso.log")
            t0 = time.time()
            print("  Video:", os.path.basename(video), "(original de Veo, dos lentes)" if original else "(grabación de la panorámica)", "| Respaldo:", os.path.basename(resp))
            done = os.path.join(work, "deteccion_completa.txt")
            lentes, lector = "", []
            if original:
                lentes = os.path.join(g, "calib_lentes.json")
                if not os.path.exists(lentes):
                    lentes = os.path.join(PROC, "calib_lentes_mitre.json")
                    print("  Uso la calibración de lentes de la cancha de Mitre (si es otra cancha, poné calib_lentes.json en la carpeta).")
                lector = ["--lector", LECTOR, "--parseq-dir", PARSEQ_DIR] if ensure_lector() and ensure_parseq() else []
                if not lector: print("  ATENCIÓN: falta el lector de números (Aro\\modelos\\lector_camisetas_v1.pt); sigo sin leer camisetas.")
            if os.path.exists(done) and os.path.getmtime(done) > os.path.getmtime(video):
                print("  Paso 1/2: la detección de jugadores ya está hecha, la reutilizo.")
            else:
                print("  Paso 1/2: detectar jugadores" + (" y leer números" if original else "") + " (la parte larga). Si se corta, volvé a abrir procesar.bat y sigue desde donde quedó.")
                if original:
                    cmd = [PY, "veo_raw.py", "--video", video, "--out", work, "--lentes", lentes, "--modelo", "yolo11s.pt", "--lote", str(lote)] + lector
                else:
                    cmd = [PY, "pipeline.py", "--video", video, "--out", work, "--modelo", "yolo11s.pt", "--lote", str(lote)]
                if run(cmd, log):
                    print("  ERROR en la detección: mirá proceso.log en la carpeta del partido."); continue
            print("  Paso 2/2: seguimiento, identidad y métricas.")
            cmd = [PY, "procesar.py", "--partido", resp, "--dets", os.path.join(work, "dets.json"), "--feats", os.path.join(work, "feats.json"),
                   "--calib", os.path.join(work, "calib.json"), "--out", out]
            if original: cmd += ["--lecturas", os.path.join(work, "lecturas.json"), "--inicio-aprox", "0"]
            if run(cmd, log):
                print("  ERROR en el análisis: mirá proceso.log en la carpeta del partido."); continue
            print(f"  LISTO en {(time.time() - t0) / 60:.0f} min: {out}")
            print("  Importalo en Aro: Análisis > Rendimiento físico > Importar análisis físico (.json).")
            if not original: print("  Revisá también trabajo\\calib_vis.jpg: las líneas rojas tienen que caer sobre el borde de la cancha.")
    finally:
        keep_awake(False)


if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    os.system("chcp 65001 >nul") if os.name == "nt" else None
    main()
