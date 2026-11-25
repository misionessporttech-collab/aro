import cv2, json, numpy as np
from ultralytics import YOLO
m = YOLO('/home/claude/track/yolo11s.pt')
cap = cv2.VideoCapture('/mnt/user-data/uploads/Downloads/muestra_seguimiento.mp4'); fps = cap.get(5)
out = []; i = 0
while True:
    ok = cap.grab()
    if not ok: break
    if i % 15 == 0:
        ok, fr = cap.retrieve()
        r = m.predict(fr, imgsz=1280, classes=[0], conf=0.3, verbose=False)[0]
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        ds = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2)); h = y2 - y1
            tor = hsv[y1 + int(.15 * h): y1 + int(.5 * h), x1 + (x2 - x1) // 5: x2 - (x2 - x1) // 5]
            if tor.size == 0: continue
            H, S, V = tor[..., 0], tor[..., 1], tor[..., 2]
            gold = float(((H > 15) & (H < 35) & (S > 90) & (V > 110)).mean())
            white = float(((S < 50) & (V > 170)).mean())
            red = float((((H < 8) | (H > 170)) & (S > 110) & (V > 80)).mean())
            dark = float((V < 60).mean())
            ds.append([x1, y1, x2, y2, round(float(c), 2), round(gold, 2), round(white, 2), round(red, 2), round(dark, 2)])
        out.append({'t': round(i / fps, 3), 'd': ds})
    i += 1
json.dump(out, open('follow_dets.json', 'w')); print(len(out))
