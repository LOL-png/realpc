"""Compare saved, identically rendered camera overlays across SAM cadences."""
import argparse
from pathlib import Path
import cv2
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--task',required=True)
p.add_argument('--frame',type=int,required=True);p.add_argument('--out',required=True)
a=p.parse_args();root=Path(a.root);tiles=[]
for n in [10,1,2,3]:
    path=root/f'n{n}'/'audit'/a.task/'episode_0000'/f'{a.task}_episode_0000_{a.frame:04d}.jpg'
    im=cv2.imread(str(path))
    if im is None:raise FileNotFoundError(path)
    if not tiles:
        raw=im[:,:320].copy();tiles.append(('RGB reference',raw))
    tiles.append((f'SAM every {n}',im[:,320:].copy()))
panels=[]
for title,im in tiles:
    top=np.full((30,im.shape[1],3),35,np.uint8)
    cv2.putText(top,title,(8,22),0,.6,(255,255,255),1,cv2.LINE_AA)
    panels.append(np.vstack([top,im]))
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
cv2.imwrite(str(out),np.hstack(panels));print(out)
