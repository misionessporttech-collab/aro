# Aro · Procesador de rendimiento físico

Calcula la distancia, la velocidad media y máxima, la alta intensidad y los sprints de cada jugador a partir de la **vista panorámica de Veo** (grabada con OBS), y genera el video panorámico con el nombre de cada jugador.

**Forma más simple:** abrir el cuaderno en Google Colab con GPU:
https://colab.research.google.com/github/misionessporttech-collab/aro/blob/main/procesador/Aro_procesador_fisico.ipynb

## Pasos
1. `pipeline.py`: recorte automático de bandas negras, calibración automática de la cancha (parte de `calib_mitre.json`), detección de personas a 10 cuadros/s y color de cada una (restando el fondo, porque la cámara está fija). Guarda el avance y retoma si se corta.
2. `procesar.py`: separa equipos y árbitros por color, sigue a 5 jugadores por equipo en metros, se sincroniza solo con las jugadas etiquetadas en Aro (por tramos de 5 minutos, por si la grabación tuvo cortes), identifica a cada jugador cruzando su recorrido con esas jugadas (cortando en cada cruce entre compañeros) y calcula las métricas. Salida: `fisico.json` (se importa en Aro) y `etiquetas.json`.
3. `render.py` (opcional): video con etiquetas y minimapa.

## Grabar la panorámica
OBS Studio, ventana del navegador con el reproductor de Veo en vista panorámica a pantalla completa, máxima resolución del monitor, 30 fps constantes, sin mover el mouse ni el zoom durante la grabación.

## Precisión
Posición: unos 15 cm en la cancha de Mitre. Distancia: ±5-10 %. La velocidad máxima conviene tomarla como referencia. La identidad mejora con más jugadas etiquetadas; los tramos dudosos se corrigen en Aro (Rendimiento físico → Revisar identidades).

## En la PC (Windows con placa NVIDIA)

1. Doble clic en `windows/instalar.bat` (una sola vez): instala todo en `%USERPROFILE%\Aro`.
2. Por cada partido, una carpeta en `Aro\partidos\` con la grabación panorámica y el respaldo `.json` exportado de Aro.
3. `Aro\procesar.bat`: detección (la parte larga, con la placa de video) y análisis. Se actualiza solo desde GitHub.
   Si la detección ya está hecha, la reutiliza y el análisis tarda menos de un minuto.
4. `Aro\generar_video.bat` (opcional): arma `panoramica_con_nombres.mp4` para subir a YouTube y verla en Aro
   con el botón "Panorámica con nombres".

`fisico.json` incluye `identidad`: qué parte del tiempo tiene número asignado y una prueba interna de acierto
(se esconden jugadas, se identifica con el resto y se mide). Aro avisa cuando el acierto es bajo.
