"""Replay adapter for CausalClothTracker.step; reads one current input at a time."""
import argparse,json,sys,time,pickle
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.cloth_causal import CausalClothTracker,at_image
from realpc.cloth_inputs import project

def overlay(im,X,E,K,T,D):
    out=im.copy();uv,z=project(X,K,T);d,inside=at_image(D,uv)
    ok=inside&(z>.08)&((d<=.08)|(z<d+.018));edges=E[ok[E].all(1)]
    xy=np.clip(np.rint(uv),-3000,3000).astype(np.int32)
    cv2.polylines(out,[xy[e].reshape(-1,1,2) for e in edges],False,(30,230,60),1,cv2.LINE_AA)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',help='Trusted local state checkpoint; experiments only, rerun final config from t0');ap.add_argument('--inputs',required=True);ap.add_argument('--cot-inputs');ap.add_argument('--cot-weight',type=float,default=0.);ap.add_argument('--cot-anchor',choices=['front','nearest'],default='nearest');ap.add_argument('--motion-scale',type=float,default=1.);ap.add_argument('--robust-depth',action='store_true');ap.add_argument('--out',required=True);ap.add_argument('--stop',type=int,default=453);ap.add_argument('--wrists',action='store_true');ap.add_argument('--wrist-flow',action='store_true');ap.add_argument('--grasp-layers',action='store_true');ap.add_argument('--dense',action='store_true');ap.add_argument('--predict',action='store_true');ap.add_argument('--plane',action='store_true');ap.add_argument('--strain-limit',type=float,default=0.);ap.add_argument('--data',type=float,default=2.);ap.add_argument('--coverage',type=float,default=0.);ap.add_argument('--arap',type=float,default=2);ap.add_argument('--track',type=float,default=8);ap.add_argument('--pin',type=float,default=35);ap.add_argument('--cell',type=float,default=.010);ap.add_argument('--template',choices=['top','tsdf'],default='tsdf');ap.add_argument('--iters',type=int,default=5);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);tr=CausalClothTracker(cell=a.cell,arap=a.arap,track=a.track,pin=a.pin,wrists=a.wrists,dense=a.dense,data=a.data,coverage=a.coverage,predict=a.predict,plane=a.plane,strain_limit=a.strain_limit,wrist_flow=a.wrist_flow,template=a.template,grasp_layers=a.grasp_layers,cot_weight=a.cot_weight,cot_anchor=a.cot_anchor,motion_scale=a.motion_scale,robust_depth=a.robust_depth);metrics=[];XX=[];start=time.time()
    if a.resume:
        with open(a.resume,'rb') as f:tr,XX,metrics=pickle.load(f)
        for key in ['wrists','dense','coverage','predict','plane','strain_limit','wrist_flow','grasp_layers','cot_weight','cot_anchor','motion_scale','robust_depth']:
            setattr(tr,key,getattr(a,key))
        tr.wa=a.arap;tr.wd=a.data;tr.wt=a.track;tr.wp=a.pin
    first=len(XX)
    writer=cv2.VideoWriter(str(out/('both_cameras.mp4' if not first else f'both_cameras_from_{first:04d}.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,840))
    for t in range(first,a.stop):
        p=Path(a.inputs)/'frames'/f'{t:04d}.npz'
        # The input producer is causal and can run concurrently; bounded wait.
        waited=0
        while not p.exists():
            time.sleep(1);waited+=1
            if waited>90:raise RuntimeError('Input missing: '+str(p))
        with np.load(p) as f:frame={k:f[k] for k in f.files}
        if a.cot_inputs:
            cp=Path(a.cot_inputs)/f'{t:04d}.npz';waited=0
            while not cp.exists():
                time.sleep(1);waited+=1
                if waited>90:raise RuntimeError('CoTracker packet missing: '+str(cp))
            with np.load(cp) as z:frame.update({k:z[k] for k in z.files})
        X,m,v=tr.step(t,frame,iters=a.iters);XX.append(X.astype(np.float32));metrics.append(m)
        rows=[]
        for c in ['top','side']:
            im=frame[c+'_rgb'];ov=overlay(im,X,tr.cm.E,v[c]['K'],v[c]['T'],frame[c+'_depth'])
            cv2.putText(ov,f'{c} t={t} fixed material mesh',(12,26),0,.6,(0,0,255),2);rows.append(np.hstack([im,ov]))
        canvas=np.vstack(rows);writer.write(canvas)
        if t in [0,100,150,180,210,240,270,300,360,390,420,450] or t==a.stop-1:cv2.imwrite(str(out/f'overlay_{t:04d}.jpg'),canvas)
        if t in [149,319]:
            with open(out/f'state_{t:04d}.pkl','wb') as f:pickle.dump((tr,XX,metrics),f)
        if t%25==0:
            print(t,round(time.time()-start,1),'med/p90',round(m['mesh_to_surface_med_mm'],2),round(m['mesh_to_surface_p90_mm'],2),'stretch',np.round(m['edge_ratio_p5_p50_p95'],2),'pins',m['pins'],flush=True)
            json.dump(metrics,open(out/'metrics.json','w'));json.dump(tr.events,open(out/'grasp_events.json','w'))
    writer.release();np.savez_compressed(out/'mesh.npz',X=np.array(XX),V0=tr.cm.V0,F=tr.cm.F,sample_faces=tr.cm.sf,sample_bary=tr.cm.sb,ticks=np.arange(len(XX)),causal=True)
    json.dump(metrics,open(out/'metrics.json','w'));json.dump(tr.events,open(out/'grasp_events.json','w'));json.dump(vars(a),open(out/'config.json','w'),indent=2)
    print('DONE',out,round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
