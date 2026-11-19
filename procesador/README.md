# Procesador de seguimiento (prototipo)

Calcula posiciones, distancia, velocidad y velocidad máxima de cada jugador a partir de la vista panorámica de Veo, y genera un video con etiquetas y minimapa.

Orden: `detect.py` (detección de personas) → `features.py` (color de cada jugador, con la cámara fija como fondo) → `track.py` (seguimiento en metros, equipos, sincronización con las jugadas etiquetadas, identidad y métricas) → `render.py` (video con etiquetas).

`calib.py` tiene el modelo de cámara cilíndrica calibrado para la cancha de Mitre (esquinas, línea central y borde del piso).
