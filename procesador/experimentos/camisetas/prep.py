# posiciones de la panorámica para la ventana de la muestra
import sys, pickle, json, numpy as np
sys.path.insert(0,'/home/claude/aro-repo/procesador')
import procesar as P
sys.modules['__main__'].Track = P.Track
C = pickle.load(open('/home/claude/full/trabajo/seguimiento.pkl','rb'))
knots = C['knots']
pos = {}
for t in C['tracks']:
    for f, xy in t.pts.items():
        if 7100 <= f <= 10300: pos.setdefault(f, []).append((float(xy[0]), float(xy[1]), t.team, t.id))
json.dump({'knots': knots, 'pos': pos}, open('pano_pos.json','w'))
print(len(pos), knots[:3])
