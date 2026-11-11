"""Common RGB-D silhouette/depth audit of saved cloth trajectories."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.diffcalib import Renderer
from realpc.cloth_inputs import project

def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',action='append',required=True,help='label=mesh.npz');p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--ticks',type=int,nargs='+',help='Audit only these saved frame indices for short diagnostic gates');a=p.parse_args()
    models={};results={}
    for spec in a.fit:
        name,path=spec.split('=',1);z=np.load(path);models[name]=(z['X'],torch.as_tensor(z['F'],dtype=torch.int64,device='cuda'));results[name]=[]
    count=min(len(x[0]) for x in models.values());torch.set_num_threads(2)
    ticks=sorted(set(a.ticks)) if a.ticks is not None else list(range(count))
    if any(t<0 or t>=count for t in ticks):p.error('Audit ticks must exist in every trajectory')
    for t in ticks:
        with np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz') as z:
            robot=z['robot'];frames={c:{k:z[c+'_'+k] for k in ('depth','mask','K','T')} for c in ('top','side')}
        rows={name:dict(t=t) for name in models}
        for c,f in frames.items():
            D,M,K,T=f['depth'],f['mask'],f['K'],f['T'];h,w=D.shape;R=Renderer(K,h,w)
            uv,zz=project(robot,K,T);xy=np.rint(uv).astype(int);ok=(zz>.08)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            robot_mask=np.zeros((h,w),np.uint8);robot_mask[xy[ok,1],xy[ok,0]]=1;robot_mask=cv2.dilate(robot_mask,np.ones((5,5),np.uint8));valid=robot_mask==0;target=M&valid
            for name,(X,F) in models.items():
                Q=(X[t]-T[:3,3])@T[:3,:3]
                with torch.inference_mode():
                    mask,depth=R.raster(torch.as_tensor(Q[None],dtype=torch.float32,device='cuda'),F)
                pred=(mask[0].cpu().numpy()>.5)&valid;Z=depth[0].cpu().numpy();overlap=pred&target&(D>.08)
                error=np.abs(Z[overlap]-D[overlap])*1000
                rows[name][c]=dict(iou=float((pred&target).sum()/max((pred|target).sum(),1)),mask_recall=float((pred&target).sum()/max(target.sum(),1)),depth_med_mm=float(np.median(error)) if len(error) else None,depth_p90_mm=float(np.percentile(error,90)) if len(error) else None)
        for name in models:results[name].append(rows[name])
        if t%100==0:print('audit',t,flush=True)
    summaries={}
    for name,rows in results.items():
        summaries[name]={}
        for phase,lo,hi in [('before_lift',0,160),('first_fold',160,250),('between_folds',250,335),('second_fold',335,count)]:
            part=[r for r in rows if lo<=r['t']<hi]
            if part:summaries[name][phase]={c:{k:float(np.median([r[c][k] for r in part if r[c][k] is not None])) for k in ('iou','mask_recall','depth_med_mm','depth_p90_mm')} for c in ('top','side')}
    Path(a.out).write_text(json.dumps(dict(frames=count,audited_ticks=ticks,summary=summaries,per_frame=results,note='Identical current-frame masks/depth/calibration for every model. FK robot pixels excluded. Silhouette and depth consistency are not material-correspondence ground truth.'),indent=2));print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
