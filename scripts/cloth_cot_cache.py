"""Generate frozen, zero-lookahead current-frame CoTracker packets."""
import argparse,sys,time,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.cloth_cotracker import CausalClothCoTracker
from realpc.cloth_causal import CausalClothTracker,CAMS
ap=argparse.ArgumentParser();ap.add_argument('--config');ap.add_argument('--cameras',nargs='+',choices=CAMS,default=list(CAMS));ap.add_argument('--inputs',required=True);ap.add_argument('--out',required=True);ap.add_argument('--stop',type=int,default=453);a=ap.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cot=CausalClothCoTracker();cleaner=CausalClothTracker.from_config(json.load(open(a.config))) if a.config else CausalClothTracker();start=time.time();times=[]
for t in range(a.stop):
    tick_start=time.perf_counter();f=dict(np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz'));views=cleaner.views(f);packet={}
    assert int(f.get('mask_max_observed_tick',t))==t
    for c in CAMS:
        if c in a.cameras:p=cot.step(c,t,f[c+'_rgb'],views[c]['M'])
        else:p=dict(uv=np.empty((0,2),np.float32),visible=np.empty(0,bool),seed_uv=np.empty((0,2),np.float32),birth=np.array(0))
        packet.update({c+'_cot_'+k:v for k,v in p.items()})
    packet['max_observed_tick']=t;packet['zero_lookahead']=True;np.savez_compressed(out/f'{t:04d}.tmp.npz',**packet);(out/f'{t:04d}.tmp.npz').replace(out/f'{t:04d}.npz')
    times.append(time.perf_counter()-tick_start)
    if t%10==0:print(t,round(time.time()-start,1),{c:int(packet[c+'_cot_visible'].sum()) for c in CAMS},flush=True)

(out/'summary.json').write_text(json.dumps(dict(frames=a.stop,cameras=a.cameras,seconds=time.time()-start,per_frame_seconds=times,causal=True,inputs=a.inputs,config=a.config),indent=2))
