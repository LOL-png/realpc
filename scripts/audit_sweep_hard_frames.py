"""One difficult current frame per episode: paired old pose vs causal cold estimate.
This tests reacquisition, not an all-episode tracking success rate.
"""
import os,sys,json,time
from pathlib import Path
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,read_depth_frames
from realpc.v4 import StreamConverter,load_library
from realpc.calib_rig import RigCalib
from realpc.config import build_arms,load_config
from realpc.diffcalib import TorchArm,Renderer
os.environ.update(REALPC_SIL='1',REALPC_CAM_EQUAL='1',REALPC_SIL_METHOD='contour',REALPC_COLD_COMPONENTS='1',REALPC_BROOM_MATERIAL='1',REALPC_BROOM_HOLE='0',REALPC_BROOM_CLEAN_MASK='0',REALPC_WRIST_WEIGHT='1',REALPC_BROOM_REJECT_BG='0');torch.set_num_threads(2)
OUT=Path('/home/howardji/data_ext/astra_rigid/hard50');OUT.mkdir(parents=True,exist_ok=True);cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot'];arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints']);oldlibs={}
for key,path in [('v1',ROOT/'assets/objects/sweep'),('v3',Path('/home/howardji/realpc-astra-meshes/assets/objects_v3/sweep'))]:oldlibs[key]=[o for o in load_library(path,100)[0] if o.name=='broom'][0]
records=json.load(open(OUT/"metrics.json"))["records"] if len(sys.argv)>1 else []
for ei in ([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else range(50)):
 records=[x for x in records if x["episode"]!=f"episode_{ei:04d}"]
 start=time.time();ep=Episode(f'/home/howardji/data_ext/agentwm_probe/hf/sweep/episode_{ei:04d}');latest=ei in [0,10,20,30,45];version='v3' if latest else 'v1'
 if latest:
  report=json.load(open(f'/home/howardji/data_ext/astra_meshes/dataset_v2/qa/sweep/{ep.name}/metrics.json'));data=report['per_frame']['tracker']['broom/top'];source=f'/home/howardji/data_ext/realpc_out/v2/sweep/{ep.name}.npz';seg='/home/howardji/data_ext/astra_meshes/dataset_v2/seg'
 else:
  report=json.load(open(f'/home/howardji/data_ext/realpc_out/v1/sweep/{ep.name}_selfqa.json'));q=report['per_frame']['broom/top'];data=list(zip(q['ticks'],q['iou']));source=f'/home/howardji/data_ext/realpc_out/v1/sweep/{ep.name}.npz';seg='/home/howardji/data_ext/realpc_v4/seg'
 t=int(min(data,key=lambda x:x[1])[0]);frames={}
 try:
  for c in ['top','side','left_wrist','right_wrist']:
   D=read_depth_frames(str(Path(ep.path)/(c+'_depth.mkv')),[t])[t].astype(np.float32)*float(ep.meta['depth_scale'][c]);cap=cv2.VideoCapture(ep.video(c));cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();cap.release();assert ok;frames[c]=(D,im)
  conv=StreamConverter(ep,cal,arm,build_arms(cfg,ep.calib),str(ROOT/'assets/objects_v3/sweep'),start=t,cold=True,seg_dir=seg);out=conv.step(t,frames);obj=next(o for o in conv.objs if o.name=='broom');old=np.load(source);oi=list(old['obj_names']).index('broom');k=int(np.argmin(abs(old['ticks']-t)));before=old['obj_T'][oi,k];after=out['obj_T']['broom'];q=out['q'];robot=conv.trk.robot(q,conv.T_world_base);row={'episode':ep.name,'tick':t,'baseline':version,'initialization':conv.init_info['broom'],'seconds':time.time()-start,'cameras':{}};tiles=[]
  for c in ['top','side']:
   L,s0=conv.seg[c];lab=L[t-s0];lid=[o.name for o in conv.objs].index('broom')+1;H,W=(180,320) if c=='top' else (240,320);lab=cv2.resize(lab,(W,H),interpolation=0);gt=lab==lid;valid=~np.isin(lab,[254,255]);K=ep.intrinsics(c).copy();K[:2]*=320/frames[c][1].shape[1];ren=Renderer(K,H,W);Tc=torch.tensor(np.linalg.inv(cal.T_world_cam(c)),device='cuda',dtype=torch.float32)
   rz=[]
   for rob in robot.values():
    _,z=ren.raster((rob['V']@Tc[:3,:3].T+Tc[:3,3])[None],arm.faces);z=z[0].cpu().numpy();rz.append(np.where(z>0,z,np.inf))
   rz=np.minimum.reduce(rz);raw=cv2.resize(frames[c][1],(W,H));ims=[raw];scores=[];depth=cv2.resize(frames[c][0],(W,H),interpolation=0)
   for mesh,T in [(oldlibs[version],before),(obj,after)]:
    T=torch.tensor(T,device='cuda',dtype=torch.float32);V=mesh.verts_t@T[:3,:3].T+T[:3,3];_,z=ren.raster((V@Tc[:3,:3].T+Tc[:3,3])[None],mesh.faces_t);z=z[0].cpu().numpy();pred=(z>0)&(z<rz)&valid;scores.append(float((pred&gt).sum()/max(1,(pred|gt).sum())));im=raw.copy();m=pred&((depth<.1)|(z<depth+.015));cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,(255,100,0),1);ims.append(im)
   row['cameras'][c]={'before_iou':scores[0],'after_iou':scores[1],'reference_pixels':int(gt.sum())};tile=np.concatenate(ims,1);cv2.putText(tile,f'{ep.name} t{t} {c}: RGB / {version} / current cold pose',(5,17),0,.45,(255,255,255),1);tiles.append(cv2.copyMakeBorder(tile,0,240-H,0,0,0))
  cv2.imwrite(str(OUT/(ep.name+'.jpg')),np.concatenate(tiles));np.savez_compressed(OUT/(ep.name+'.npz'),tick=t,obj_names=np.array(list(out['obj_T'])),obj_T=np.stack(list(out['obj_T'].values())),q_left=q['left'],q_right=q['right']);records.append(row);print(ep.name,t,{k:(round(v['before_iou'],3),round(v['after_iou'],3)) for k,v in row['cameras'].items()},round(time.time()-start,1),flush=True)
  del conv,old
 except Exception as e:
  records.append({'episode':ep.name,'tick':t,'error':repr(e)});print('FAILED',ep.name,repr(e),flush=True)
 (OUT/'metrics.json').write_text(json.dumps({'scope':'one worst previously audited top-view frame per episode; current-frame cold reacquisition; NOT full-episode replay; v3 baseline for five pilot episodes, legacy v1 for the rest; references are causal SAM masks, not hand ground truth','records':records},indent=2))
