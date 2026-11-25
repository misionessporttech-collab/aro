"""Corta 5 minutos (12:00 a 17:00) del archivo panorámico original de Veo para pruebas.
Deja Descargas\\muestra_panoramica_original.mp4."""
import glob, os, subprocess, sys
DL = os.path.join(os.path.expanduser("~"), "Downloads")
src = sorted(glob.glob(os.path.join(DL, "*panoramica_original*.mp4")), key=os.path.getsize)
if not src: sys.exit("No encuentro el archivo panorámico original en Descargas.")
out = os.path.join(DL, "muestra_panoramica_original.mp4")
base = ["ffmpeg", "-y", "-loglevel", "error", "-ss", "720", "-i", src[-1], "-t", "300", "-an"]
for enc in (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]):
    print("Cortando con", enc[1], "...")
    if subprocess.run(base + enc + [out]).returncode == 0:
        print("Listo:", out); break
else:
    sys.exit("No se pudo cortar el video.")
