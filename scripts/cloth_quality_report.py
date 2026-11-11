"""Audit exported cloth trajectories; no fitting and no ground-truth claims."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def report(path,check_all=False):
    import ipctk
    path=Path(path);z=np.load(path/'mesh.npz');X=z['X'];F=z['F'];V=z['V0']
    E=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[0,2]]]),1),axis=0)
    rest=np.linalg.norm(V[E[:,0]]-V[E[:,1]],axis=1)
    mesh=ipctk.CollisionMesh(np.asarray(X[0],float),E.astype(np.int32),F.astype(np.int32))
    ticks=np.arange(len(X)) if check_all else np.unique(np.r_[0,np.arange(125,len(X),25),len(X)-1])
    audit=[]
    for t in ticks:
        ratio=np.linalg.norm(X[t,E[:,0]]-X[t,E[:,1]],axis=1)/rest
        contacts=ipctk.NormalCollisions();contacts.build(mesh,np.asarray(X[t],float),.003)
        gap=float(np.sqrt(contacts.compute_minimum_distance(mesh,np.asarray(X[t],float))))
        audit.append(dict(t=int(t),intersects=bool(ipctk.has_intersections(mesh,np.asarray(X[t],float))),minimum_self_gap_mm=gap*1000 if np.isfinite(gap) else None,edge_ratio_p1_p50_p99=np.percentile(ratio,[1,50,99]).tolist(),max_extension=float(ratio.max()-1),max_compression=float(1-ratio.min())))
    metrics=json.loads((path/'metrics.json').read_text());phases={}
    for name,start,stop in [('before_lift',0,160),('first_fold',160,250),('between_folds',250,335),('second_fold',335,len(X))]:
        rows=metrics[start:min(stop,len(X))]
        if not rows:continue
        summary={}
        for cam in ('top','side'):
            summary[cam]={k:float(np.nanmedian([r['cameras'].get(cam,{}).get(k,np.nan) for r in rows])) for k in ('surface_to_mesh_med_mm','surface_to_mesh_p90_mm','surface_coverage_10mm')}
        summary['max_grasp_residual_mm']=max(r['grasp_error_mm'] for r in rows)
        phases[name]=summary
    result=dict(frames=len(X),phases=phases,physical_audit=audit,note='Observed-surface/mesh consistency only; hidden material correspondence is not ground truth. IPC audit checks geometric crossings, not finite clearance or all inter-frame tunneling.')
    (path/'quality_report.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fit');p.add_argument('--all',action='store_true');a=p.parse_args();print(json.dumps(report(a.fit,a.all),indent=2))
