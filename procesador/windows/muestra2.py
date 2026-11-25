"""Corta 5 minutos (12:00 a 17:00) del archivo panorámico original de Veo para pruebas,
en dos partes de 2:30 para que cada una pese menos de 400 MB.
Deja Descargas\\muestra_pano_a.mp4 y Descargas\\muestra_pano_b.mp4."""
import glob, os, subprocess, sys
DL = os.path.join(os.path.expanduser("~"), "Downloads")
src = [f for f in glob.glob(os.path.join(DL, "*panoramica_original*.mp4")) if "muestra" not in os.path.basename(f)]
if not src: sys.exit("No encuentro el archivo panorámico original en Descargas.")
src = max(src, key=os.path.getsize)
for name, ss in (("muestra_pano_a.mp4", 720), ("muestra_pano_b.mp4", 870)):
    out = os.path.join(DL, name)
    base = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(ss), "-i", src, "-t", "150", "-an"]
    for enc in (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "22"], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22"]):
        print("Cortando", name, "con", enc[1], "...")
        if subprocess.run(base + enc + [out]).returncode == 0:
            mb = os.path.getsize(out) / 2**20
            print(f"  listo: {mb:.0f} MB"); break
    else:
        sys.exit("No se pudo cortar el video.")
old = os.path.join(DL, "muestra_panoramica_original.mp4")
if os.path.exists(old): os.remove(old)
print("Listo. Avisale a Claude.")
