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
        p = subprocess.Popen(cmd, cwd=PROC, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print("   ", line.rstrip()); lf.write(line)
        return p.wait()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Aro", "partidos")
    games = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if not games:
        print(f"No hay carpetas de partidos en {root}. Creá una y poné adentro la panorámica y el respaldo .json.")
        return
    lote = gpu_batch()
    keep_awake(True)
    try:
        for g in games:
            name = os.path.basename(g)
            out = os.path.join(g, "fisico.json")
            video, resp, err = pick_inputs(g)
            print(f"\n=== {name} ===")
            if err:
                print("  Salteo:", err); continue
            if os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(video), os.path.getmtime(resp)):
                print("  Ya procesado (fisico.json está al día). Para rehacerlo, borrá fisico.json."); continue
            work = os.path.join(g, "trabajo"); os.makedirs(work, exist_ok=True)
            log = os.path.join(g, "proceso.log")
            t0 = time.time()
            print("  Video:", os.path.basename(video), "| Respaldo:", os.path.basename(resp))
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
    main()
