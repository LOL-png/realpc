"""Fit the static tray library from both current-t0 cameras across episodes."""
import sys,json,argparse
from pathlib import Path
import cv2,numpy as np,trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,read_depth_frames
from realpc.calib_rig import RigCalib
from realpc.objlib import depth_points,table_crop
from realpc.box_fixture import fit_box,surface_residual,refine_box_pose
ap=argparse.ArgumentParser();ap.add_argument('--reuse',action='store_true');ap.add_argument('--episodes',nargs='+',type=int,default=list(range(50)));ap.add_argument('--out',required=True);ap.add_argument('--assets',default='assets/objects_v3/sweep');a=ap.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);assets=Path(a.assets);cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');records=[];scans=[];clouds=[];cols=[]
for i in a.episodes:
    cache=out/f'episode_{i:04d}.npz'
    if a.reuse and cache.exists():
        z=np.load(cache);T,dims,P,C=z['T'],z['dims'],z['P'],z['C'];records.append({'episode':cache.stem,'dimensions_m':dims.tolist()});clouds.append(P);cols.append(C);scans.append({'episode':'sweep/'+cache.stem,'T_world_obj_t0':T.tolist(),'points':len(P),'kind':'current_frame_geometry'});continue
    ep=Episode(f'/home/howardji/data_ext/agentwm_probe/hf/sweep/episode_{i:04d}');region=json.load(open(Path(ep.path)/'table_region.json'));pp=[];cc=[]
    for cam in ['top','side']:
        cap=cv2.VideoCapture(ep.video(cam));_,im=cap.read();cap.release();D=read_depth_frames(str(Path(ep.path)/(cam+'_depth.mkv')),[0])[0].astype(np.float32)*float(ep.meta['depth_scale'][cam])
        P,C,_,_=depth_points(D,im,ep.intrinsics(cam),cal.T_world_cam(cam),stride=2);keep=table_crop(P,region,*ep.table_plane(),above=-.04);pp.append(P[keep]);cc.append(C[keep])
    try:
        T,dims,P,C,info=fit_box(np.concatenate(pp),np.concatenate(cc));info.update(episode=ep.name,T_world_box=T.tolist());records.append(info);clouds.append(P);cols.append(C);scans.append({'episode':'sweep/'+ep.name,'T_world_obj_t0':T.tolist(),'points':len(P),'kind':'current_frame_geometry'})
        np.savez_compressed(out/(ep.name+'.npz'),P=P,C=C,T=T,dims=dims)
        print(ep.name,np.round(dims*1000,1).tolist(),len(P),flush=True)
    except Exception as e:print('FAILED',ep.name,repr(e),flush=True);records.append({'episode':ep.name,'error':str(e)})
good=[r for r in records if 'dimensions_m' in r];dims=np.median([r['dimensions_m'] for r in good],axis=0);x,y,h=dims;th=.003
# Refine each current-observation pose against robust shared dimensions.
clouds=[];cols=[]
for sc in scans:
 epname=sc['episode'].split('/')[-1];z=np.load(out/(epname+'.npz'));old=z['T'];W=z['P']@old[:3,:3].T+old[:3,3];T,detail=refine_box_pose(old,W,dims,fit_dimensions=True);P=(W-T[:3,3])@T[:3,:3];C=z['C'];sc['T_world_obj_t0']=T.tolist();sc.update(detail);np.savez_compressed(out/(epname+'.npz'),P=P,C=C,T=T,dims=dims);clouds.append(P);cols.append(C)
dims=np.median([s["fitted_dimensions_m"] for s in scans],axis=0);x,y,h=dims
parts=[]
def slab(size,center):
    m=trimesh.creation.box(size);m.apply_translation(center);parts.append(m)
slab([x,y,th],[0,0,-th/2])
for sign in [-1,1]:
    slab([th,y,h],[sign*(x-th)/2,0,h/2]);slab([x,th,h],[0,sign*(y-th)/2,h/2])
mesh=trimesh.util.concatenate(parts);mesh.visual.vertex_colors=[163,126,77,255];assets.mkdir(parents=True,exist_ok=True);mesh.export(assets/'box.ply')
P=np.concatenate(clouds);C=np.concatenate(cols);_,ii=np.unique(np.floor(P/.003).astype(int),axis=0,return_index=True);P,C=P[ii],C[ii];np.savez_compressed(assets/'box_points.npz',P=P.astype(np.float32),C=C.astype(np.float32))
meta={'name':'box','task':'sweep','version':'v3 cardboard tray, multiepisode two-camera fit','passive_fixture':True,'dimensions_m':dims.tolist(),'wall_thickness_m':th,'scans':scans,'n_scans':len(scans),'frame':'origin at centre of interior floor; x long side; z floor normal','runtime_init':'current-frame cardboard component and rectangle/floor fit; no episode-specific pose lookup','unseen_geometry':'3mm wall/floor thickness is a prior; upper-edge notches not yet modelled','watertight':bool(mesh.is_watertight),'calibration':'configs/calib_yam_rig_v3.json'}
(assets/'box.json').write_text(json.dumps(meta,indent=2));(out/'fit_summary.json').write_text(json.dumps({'dims_m':dims.tolist(),'records':records,'surface_median_mm':float(np.median(surface_residual(P,dims))*1000)},indent=2))
