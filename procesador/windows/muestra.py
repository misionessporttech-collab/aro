"""Corta 5 minutos sincronizados del video de seguimiento y de la panorámica, para pruebas.
Deja los dos archivos en Descargas: muestra_seguimiento.mp4 y muestra_panoramica.mp4."""
import glob, json, os, subprocess, sys
HOME = os.path.expanduser("~")
DL = os.path.join(HOME, "Downloads")
DESDE, DUR = 720, 300  # segundos del video del partido (12:00 a 17:00)

def main():
    seg = sorted(glob.glob(os.path.join(DL, "partido-basquet-mitre*.mp4")))
    part = os.path.join(HOME, "Aro", "partidos", "mitre-suardi")
    pans = [f for f in glob.glob(os.path.join(part, "*.mp4")) if "con_nombres" not in f]
    if not seg: sys.exit("No encuentro el video de seguimiento (partido-basquet-mitre...mp4) en Descargas.")
    if not pans: sys.exit("No encuentro la grabación panorámica en Aro\\partidos\\mitre-suardi.")
    T0 = json.load(open(os.path.join(part, "fisico.json"))).get("T0", 2.7)
    jobs = [(seg[0], DESDE, os.path.join(DL, "muestra_seguimiento.mp4")),
            (max(pans, key=os.path.getsize), DESDE - T0, os.path.join(DL, "muestra_panoramica.mp4"))]
    for src, ss, out in jobs:
        print("Cortando", os.path.basename(src), "->", os.path.basename(out))
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.3f}", "-i", src, "-t", str(DUR), "-an",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", out])
        if r.returncode: sys.exit("Falló el corte de " + os.path.basename(src))
    print("Listo. Avisale a Claude.")

if __name__ == "__main__":
    main()
