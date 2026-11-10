"""Generate frozen, zero-lookahead current-frame CoTracker packets."""
import argparse,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.cloth_cotracker import CausalClothCoTracker
from realpc.cloth_causal import CausalClothTracker,CAMS
ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--out',required=True);ap.add_argument('--stop',type=int,default=453);a=ap.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cot=CausalClothCoTracker();cleaner=CausalClothTracker();start=time.time()
for t in range(a.stop):
    f=dict(np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz'));views=cleaner.views(f);packet={}
    for c in CAMS:
        p=cot.step(c,t,f[c+'_rgb'],views[c]['M']);packet.update({c+'_cot_'+k:v for k,v in p.items()})
    packet['max_observed_tick']=t;packet['zero_lookahead']=True;np.savez_compressed(out/f'{t:04d}.tmp.npz',**packet);(out/f'{t:04d}.tmp.npz').replace(out/f'{t:04d}.npz')
    if t%10==0:print(t,round(time.time()-start,1),{c:int(packet[c+'_cot_visible'].sum()) for c in CAMS},flush=True)
