"""Current-frame calibrated inputs for cloth tracking and live inference.
No future camera pose, image, segmentation or joint sample is accessed.
"""
from pathlib import Path
import numpy as np,cv2
from .config import load_config,build_arms
from .calib_rig import RigCalib

class ClothRig:
 def __init__(self,ep,root):
  self.ep=ep;self.cal=RigCalib.load(Path(root)/'configs/calib_yam_rig_v3.json');self.arms=build_arms(load_config(Path(root)/'configs/yam_bimanual.json'),ep.calib)
  # Recorder EEF and URDF gripper use a fixed axis convention.
  A=np.eye(4);A[:3,:3]=[[0,1,0],[1,0,0],[0,0,-1]]
  self.handeye={a:A@np.asarray(ep.calib['X_'+a]) for a in ['left','right']}
  self.arms.T_base['right']=self.cal.T_world_rb();self.gripper={a:self.arms.bodies.index((a,'gripper')) for a in ['left','right']}
  self.tipids={a:np.flatnonzero(np.isin(self.arms.body,[i for i,b in enumerate(self.arms.bodies) if b[0]==a and b[1] in ['tip_left','tip_right']])) for a in ['left','right']}
 def joints(self,t):return {a:self.cal.q_at(self.ep,a,t-3) for a in ['left','right']}
 def poses(self,q):
  LP=self.arms.link_poses(q);G={a:LP[i] for a,i in self.gripper.items()};cams={c:self.cal.T_world_cam(c) for c in ['top','side']}
  cams.update({a+'_wrist':G[a]@self.handeye[a] for a in ['left','right']});return LP,G,cams

def unproject(D,K,T,stride=2,mask=None):
 yy,xx=np.mgrid[0:D.shape[0]:stride,0:D.shape[1]:stride];z=D[yy,xx];ok=(z>.08)&(z<2.5)
 if mask is not None:ok &= mask[yy,xx]
 yy,xx,z=yy[ok],xx[ok],z[ok];C=np.c_[(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z];return C@T[:3,:3].T+T[:3,3],np.c_[xx,yy]

def project(P,K,T):
 Q=(P-T[:3,3])@T[:3,:3];return Q[:,:2]/np.maximum(Q[:,2:],.001)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]],Q[:,2]

class OnlineClothMasks:
 """SAM2 forward memory fed exactly one new frame at a time; reusable live API."""
 def __init__(self,checkpoint):
  import torch
  from sam2.build_sam import build_sam2_video_predictor
  self.torch=torch;self.model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',checkpoint,device='cuda');self.states={};self.frames={};self.temp=[]
 def step(self,cam,image,seed=None):
  import tempfile
  torch=self.torch;model=self.model
  # Retain RGB history for SAM's lazy image reader, never preload future frames.
  if cam not in self.states:
   from sam2.sam2_image_predictor import SAM2ImagePredictor
   assert seed is not None
   tmp=tempfile.TemporaryDirectory(prefix='realpc_cloth_');self.temp.append(tmp);cv2.imwrite(tmp.name+'/000000.jpg',image)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    state=model.init_state(tmp.name,offload_video_to_cpu=True,offload_state_to_cpu=True)
    pred=SAM2ImagePredictor(model);pred.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));yy,xx=np.nonzero(seed);assert len(xx)>20
    box=np.array([max(0,xx.min()-12),max(0,yy.min()-12),min(image.shape[1]-1,xx.max()+12),min(image.shape[0]-1,yy.max()+12)],np.float32)
    dist=cv2.distanceTransform(seed.astype(np.uint8),cv2.DIST_L2,5);points=[];cand=np.c_[xx,yy]
    for _ in range(5):
     score=dist[yy,xx] if not points else np.minimum(dist[yy,xx],8)*np.min(np.linalg.norm(cand[:,None]-np.array(points)[None],axis=2),axis=1)
     points.append(cand[score.argmax()])
    mm,_,_=pred.predict(point_coords=np.array(points,np.float32),point_labels=np.ones(len(points),np.int32),box=box,multimask_output=False);seed=mm[0]
    model.add_new_mask(state,frame_idx=0,obj_id=1,mask=seed)
   self.states[cam]=state;self.frames[cam]=[]
   owner=self
   class Frames:
    def __getitem__(self,i):
     im=owner.frames[cam][i];im=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(model.image_size,model.image_size));x=torch.from_numpy(im).permute(2,0,1).float()/255
     return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    def __len__(self):return len(owner.frames[cam])
   state['images']=Frames();state['video_height'],state['video_width']=image.shape[:2]
  state=self.states[cam];self.frames[cam].append(image.copy());t=len(self.frames[cam])-1;state['num_frames']=t+1
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   result=list(model.propagate_in_video(state,start_frame_idx=t,max_frame_num_to_track=0,reverse=False));mask=result[-1][2][0,0].float().cpu().numpy()>0
  return mask
