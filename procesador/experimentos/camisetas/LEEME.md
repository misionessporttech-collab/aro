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
