"""Benchmark causal depth/SAM scheduling from raw current packets; old masks are evaluation only."""
import argparse,sys,json,time
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.cloth_sparse_masks import SparseClothMasks
p=argparse.ArgumentParser();p.add_argument('--fast-neighbors',action='store_true');p.add_argument('--sam-backend',choices=['video','image'],default='video');p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--interval',type=int,default=10);p.add_argument('--stop',type=int,default=453);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
import hashlib
(out/'provenance.json').write_text(json.dumps(dict(arguments=vars(a),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'realpc/cloth_sparse_masks.py',ROOT/'realpc/cloth_inputs.py']}),indent=2))
m=SparseClothMasks('/home/howardji/realpc/.model_cache/sam2/sam2.1_hiera_small.pt',a.interval,sam_backend=a.sam_backend,fast_neighbors=a.fast_neighbors);rows=[];start=time.perf_counter()
for t in range(a.stop):
    f=dict(np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz'));reference={c:f.pop(c+'_mask') for c in m.cameras}
    masks,info=m.step(t,f);scores={}
    for c in m.cameras:
        M=masks[c];R=reference[c];scores[c]=float((M&R).sum()/max((M|R).sum(),1));f[c+'_mask']=M
    # Wrist masks from the source cache are NOT used in this two-view benchmark.
    for c in ('left_wrist','right_wrist'):f[c+'_mask']=np.zeros_like(f[c+'_depth'],bool)
    f['mask_max_observed_tick']=t;f['mask_sam_called']=info['sam_called'];np.savez_compressed(out/'frames'/f'{t:04d}.npz',**f)
    rows.append(dict(**info,reference_iou=scores))
    if t%25==0:
        print(t,round(time.perf_counter()-start,1),'calls',len(m.events),'IoU',scores,flush=True)
        (out/'progress.json').write_text(json.dumps(rows[-1]))
    if t in [0,150,175,200,240,350,390,452]:
        tiles=[]
        for c in m.cameras:
            im=f[c+'_rgb'].copy();cs,_=cv2.findContours(masks[c].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,(0,255,255),1);tiles.append(cv2.resize(im,(640,420)))
        cv2.imwrite(str(out/f'overlay_{t:04d}.jpg'),np.vstack(tiles))
summary=m.summary();summary.update(wall_seconds=time.perf_counter()-start,reference='every-frame causal SAM masks, not independent labels',median_mask_iou={c:float(np.median([r['reference_iou'][c] for r in rows])) for c in m.cameras})
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'per_frame.json').write_text(json.dumps(rows));print(json.dumps(summary),flush=True)
