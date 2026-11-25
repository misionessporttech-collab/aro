import cv2, json, re, numpy as np
from rapidocr_onnxruntime import RapidOCR
ROSTER = {"1","2","3","4","5","7","10","14","31","32","35"}
ocr = RapidOCR()
D = json.load(open('follow_dets.json'))
cap = cv2.VideoCapture('/mnt/user-data/uploads/Downloads/muestra_seguimiento.mp4')
reads = []
for k, fr_d in enumerate(D):
    cand = [d for d in fr_d['d'] if d[5] > 0.12 and d[3] - d[1] > 70]
    if not cand: continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, k * 15); fr = cap.read()[1]
    for d in cand:
        x1, y1, x2, y2 = map(int, d[:4]); h = y2 - y1
        crop = fr[y1 + int(.1 * h): y1 + int(.55 * h), max(0, x1 - 4): x2 + 4]
        if crop.size == 0: continue
        big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        r, _ = ocr(big)
        for _, txt, c in (r or []):
            s = re.sub(r'\D', '', txt)
            if s in ROSTER: reads.append({'k': k, 't': fr_d['t'], 'box': d[:4], 'j': s, 'c': round(float(c), 2)})
json.dump(reads, open('reads.json', 'w')); print('lecturas', len(reads))
from collections import Counter; print(Counter(r['j'] for r in reads))
