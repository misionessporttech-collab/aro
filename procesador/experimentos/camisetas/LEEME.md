# Prueba: identificar jugadores leyendo camisetas en el video de seguimiento

Muestra: 5 min (12:00–17:00) del partido Mitre–Suardi, seguimiento 1080p + panorámica 720p.

1. `detf.py`: jugadores en el video de seguimiento (2 por segundo) con colores de camiseta.
2. `run_fitC.py` / `fitcam2.py`: ubica la cámara de seguimiento en la cancha. Modelo cilíndrico con el mismo
   centro que la panorámica, ajustado al borde del piso de madera y a las dos zonas pintadas.
   Resultado: 72 % de los cuadros ubicados; jugadores proyectados a 0,8 m (mediana) de la panorámica.
3. `ocr2.py`: lector genérico (RapidOCR) sobre el torso de los jugadores de amarillo: 222 lecturas,
   casi todas 35, 10 y 32 (no lee bien 3, 4, 5, 14).
4. `assoc.py`: cada lectura se pasa a la cancha y se asigna al recorrido de la panorámica más cercano.
   85 lecturas asociadas, 23 recorridos con votos, 77 % de coincidencia entre votos del mismo recorrido.

Próximo: afinar la ubicación con las posiciones de los jugadores y entrenar un lector de números propio.

## Segunda vuelta (25/11)

**Lector de números propio.** 960 recortes de la muestra etiquetados a mano (Mitre y Suardi; `etiquetas/`).
Se ajustó PARSeq (Apache 2.0) con esos recortes, incluyendo ejemplos ilegibles como texto vacío (`ft_parseq2.py`).
Prueba sobre el último minuto (no usado para entrenar), con umbral de confianza 0,85:
- legibles: 41 de 64 bien, 4 mal; ilegibles: 4 lecturas falsas de 131.
- antes, con el lector genérico: 16 de 76 bien.

**Unir el video de seguimiento con la panorámica (720p): no alcanza.** Aun con el ajuste de cámara refinado,
se emparejan 2–3 jugadores por cuadro de ~9 visibles, con 85–91 % de coincidencia de equipo. El modelo de la cámara
de seguimiento no es lo bastante preciso para atar cada número a su recorrido.

**Cambio de enfoque: leer los números en la panorámica 4K de Veo.** En `Clip 2.mp4` (exportado de Veo en 4042x2160)
los jugadores miden 156–287 px de alto (más que en el seguimiento) y los números de espalda se leen (`read4k.py`).
Así no hace falta unir dos videos: el número se lee sobre el mismo recorrido que da las métricas.
Falta: conseguir la panorámica 4K del partido completo y sumar recortes 4K al entrenamiento.
