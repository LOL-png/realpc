"""Quick current-frame correction diagnostic; does not alter baseline tracks."""
import sys,argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,read_depth_frames
from realpc.calib_rig import RigCalib
from realpc.config import load_config,build_arms
from realpc.diffcalib import TorchArm
from realpc.v4 import load_library
from realpc.meshtrack import MeshTracker,CamView
from realpc.broom_refine import BroomRefiner
p=argparse.ArgumentParser();p.add_argument('--converted',required=True);p.add_argument('--ticks',nargs='+',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(2)
d=dict(np.load(a.converted));ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/'+str(d['task'])+'/'+str(d['episode']));cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints']);arms=build_arms(cfg,ep.calib);arms.T_base['right']=cal.T_world_rb();obs,_=load_library(ROOT/'assets/objects_v3/sweep',100);broom=next(o for o in obs if o.name=='broom');trk=MeshTracker([broom],{'broom':np.eye(4)},arm);ref=BroomRefiner(broom);bi=list(d['obj_names']).index('broom');ts=a.ticks;cams=['top','side','left_wrist','right_wrist'];caps={c:cv2.VideoCapture(ep.video(c)) for c in cams};ds={c:read_depth_frames(str(Path(ep.path)/(c+'_depth.mkv')),ts) for c in cams};rows=[];seg={}
for c in ['top','side']:
 z=np.load('/home/howardji/data_ext/astra_meshes/dataset_v2/seg/'+str(d['task'])+'/'+str(d['episode'])+'_'+c+'.npz');old=list(z['names']).index('broom')+1;labels=z['labels'];seg[c]=(np.where(labels==old,1,np.where(labels>=254,labels,np.where(labels>0,2,0))).astype(np.uint8),int(z['start']))
for t in ts:
 start=time.time();k=int(np.argmin(abs(d['ticks']-t)));T=d['obj_T'][bi,k];q={s:cal.q_at(ep,s,cal.robot_tick(ep,'top',t)) for s in ['left','right']};rob=trk.robot(q,{'left':np.eye(4),'right':cal.T_world_rb()});views=[];rz=[]
 for c in cams:
  caps[c].set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=caps[c].read();assert ok;D=ds[c][t].astype(float)*float(ep.meta['depth_scale'][c])
  if 'wrist' in c:
   s=c.split('_')[0];A=np.eye(4);A[:3,:3]=[[0,1,0],[1,0,0],[0,0,-1]];Twc=rob[s]['G']@A@np.asarray(ep.calib['X_'+s])
  else:Twc=cal.T_world_cam(c)
  v=CamView(c,D,ep.intrinsics(c),Twc,rgb=im,labels=seg[c][0][t-seg[c][1]] if c in seg else None);views.append(v);z=[v.zbuf(r['V'],arm.faces).cpu().numpy() for r in rob.values()];rz.append(np.minimum.reduce([np.where(x>0,x,1e9) for x in z]))
 Tn,info=ref.refine(trk,T,views,rz);d['obj_T'][bi,k]=Tn;rows.append({'tick':t,'seconds':time.time()-start,**info});print(rows[-1],flush=True)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'corrected.npz',**d);(out/'diagnostic.json').write_text(json.dumps(rows,indent=2))
