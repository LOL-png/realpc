"""Compare explicit current-packet live API against a converted hang prefix."""
import argparse,os,sys,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.io import Episode,DepthStream
from realpc.calib_rig import RigCalib
from realpc.config import load_config,build_arms
from realpc.diffcalib import TorchArm
from realpc.v4 import StreamConverter
from realpc.depth_pipeline import DepthMeshPipeline
p=argparse.ArgumentParser();p.add_argument('--converted',required=True);p.add_argument('--out',required=True);a=p.parse_args()
os.environ['REALPC_DEPTH_MESH']='1'
d=np.load(a.converted);ep=Episode('/home/howardji/data_ext/agentwm_probe/hf/hang/episode_0000')
cal=RigCalib.load(ROOT/'configs/calib_yam_rig_v3.json');cfg=load_config(ROOT/'configs/yam_bimanual.json');r=cfg['robot']
arm=TorchArm(str(Path(cfg['_dir'])/r['urdf']),r['arm_joints'],r['finger_joints'])
conv=StreamConverter(ep,cal,arm,build_arms(cfg,ep.calib),str(ROOT/'assets/objects_v3/hang'),seg_dir=None)
pipeline=DepthMeshPipeline(conv,'/home/howardji/realpc/.model_cache/sam2/sam2.1_hiera_small.pt')
ds={c:DepthStream(ep.path+'/'+c+'_depth.mkv',0) for c in ('top','side')}
caps={c:cv2.VideoCapture(ep.video(c)) for c in ds}
def forbidden(*args,**kwargs):raise AssertionError('Live step tried to read recording joints/video instead of supplied packet')
cal.q_at=forbidden;ep.video=forbidden
errors=[]
for t in range(8):
    frames={}
    for c in ds:
        _,D=ds[c].next();ok,im=caps[c].read();assert ok
        frames[c]=(D.astype(np.float32)*float(ep.meta['depth_scale'][c]),im)
    out=pipeline.step(t,frames,q={s:d['q_'+s][t].astype(float) for s in ('left','right')})
    errors.append(max(float(np.max(abs(out['obj_T'][str(n)]-d['obj_T'][i,t]))) for i,n in enumerate(d['obj_names'])))
    assert np.allclose(out['robot_T'],d['robot_T'][t],atol=2e-6)
assert max(errors)<2e-5,errors
for x in ds.values():x.close()
for x in caps.values():x.release()
report={'frames':8,'max_pose_matrix_difference':max(errors),'recording_joint_and_video_access_blocked':True,'sam_events':pipeline.summary()['sam_events']}
Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(report)
