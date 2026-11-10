"""Causal material-landmark constraint for the broom's green bristle envelope.
Current RGB/depth only. Fixed mesh color supplies canonical landmarks; FK supplies
an occlusion prediction and grasp prior. Wrist observations are optional.
"""
import copy,os
import cv2,numpy as np,trimesh,torch
from scipy.spatial import cKDTree
from .meshtrack import LibObject
from .joint_pose import refine_contours

class BroomRefiner:
    def __init__(self,obj):
        C=np.asarray(obj.mesh.visual.vertex_colors)[:,:3];h=cv2.cvtColor(C[None],cv2.COLOR_RGB2HSV)[0]
        green=(h[:,0]>32)&(h[:,0]<95)&(h[:,1]>65)&(h[:,2]>40)
        mesh=obj.mesh.submesh([np.flatnonzero(green[obj.mesh.faces].mean(1)>.66)],append=True,repair=False)
        self.head=LibObject('broom_bristles',mesh,n_samples=800)
        self.obj=obj
        self.stats={}
    def observe(self,T,views,robot_z):
        data=[];zs=[];clouds=[];weights=[];support={}
        W=self.head.icp_pts@T[:3,:3].T+T[:3,3]
        for v,rz in zip(views,robot_z):
            if v.rgb is None:continue
            H,S,V=cv2.split(cv2.cvtColor(v.rgb,cv2.COLOR_BGR2HSV));m=(H>32)&(H<95)&(S>65)&(V>35)
            # Prediction gates identity, not the measured pose. Allows 8cm correction.
            P=W@v.Tcw[:3,:3].T+v.Tcw[:3,3];P=P[P[:,2]>.08]
            if not len(P):continue
            uv=P[:,:2]/P[:,2:]*[v.K[0,0],v.K[1,1]]+[v.K[0,2],v.K[1,2]]
            margin=min(160,max(25, v.K[0,0]*.08/np.median(P[:,2])))
            lo=np.floor(uv.min(0)-margin).astype(int);hi=np.ceil(uv.max(0)+margin).astype(int)
            gate=np.zeros(m.shape,np.uint8);cv2.rectangle(gate,tuple(lo),tuple(hi),1,-1);m &= gate.astype(bool)
            # The largest connected green component in the predicted region.
            n,lab,stats,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
            if n<2:continue
            j=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);m=lab==j
            if m.sum()<80:continue
            vv=copy.copy(v);vv.measurement_weight=float(os.environ.get("REALPC_WRIST_WEIGHT","1")) if "wrist" in v.name else 1.;vv.labels=m.astype(np.uint8);Q=vv.masked_points(1)
            raw_count=len(Q)
            if len(Q):Q=Q[cKDTree(W).query(Q)[0]<.08]
            if os.environ.get("REALPC_BROOM_REJECT_BG","0")=="1" and raw_count>=40 and len(Q)<.25*raw_count:continue
            if 'wrist' in v.name:
                # Wrist hand-eye error is larger; require depth support near the fixed-camera pose.
                if len(Q)<60 or np.median(cKDTree(W).query(Q)[0])>.020:continue
            if len(Q)>=40:
                clouds.append(Q);weights.append(np.full(len(Q),(.3 if 'wrist' in v.name else 1.)/len(Q)))
            data.append(vv);zs.append(rz);support[v.name]={'green_px':int(m.sum()),'depth_points':len(Q)}
        return data,zs,clouds,weights,support
    def refine(self,tracker,T,views,robot_z,alternatives=()):
        vs,rz,clouds,weights,support=self.observe(T,views,robot_z)
        if not vs:return T,{'support':support,'accepted':False}
        candidates=[np.asarray(A) for A in alternatives]
        res=(np.nan,0)
        if clouds:
            M=np.concatenate(clouds);w=np.concatenate(weights);w*=len(w)/sum(w)
            Ti,res=tracker.icp_masked(self.head,T,M,w=w,iters=6)
            angle=np.degrees(np.arccos(np.clip((np.trace(Ti[:3,:3]@T[:3,:3].T)-1)/2,-1,1)))
            if np.linalg.norm(Ti[:3,3]-T[:3,3])<.06 and angle<20:candidates.append(Ti)
        from .broom_landmarks import hole_landmarks
        landmarks,hole_records=hole_landmarks(self.obj,T,views,robot_z)
        use_landmarks=landmarks if os.environ.get("REALPC_BROOM_HOLE","0")=="1" else ()
        refined,before,after=refine_contours(self.head,T,vs,rz,1,alternatives=candidates,rounds=2,landmarks=use_landmarks)
        accepted=bool(refined is not None and after>before+.003)
        # Green bristles constrain the head but cannot determine the handle's
        # roll reliably. Preserve the complete-object evidence in fixed cameras.
        full=[]
        if accepted:
            lid=list(tracker.objs).index("broom")+1
            for v,zrob in zip(views,robot_z):
                if v.labels is None or "wrist" in v.name:continue
                lab=cv2.resize(v.labels,(v.R.W,v.R.H),interpolation=0);gt=lab==lid
                if gt.sum()<60:continue
                valid=~(((lab>0)&~gt)|((zrob<1e8)&~gt))
                scores=[]
                for H in (T,refined):
                    V=self.obj.verts_t@torch.tensor(H[:3,:3].T,device='cuda',dtype=torch.float32)+torch.tensor(H[:3,3],device='cuda',dtype=torch.float32)
                    pred=(v.zbuf(V,self.obj.faces_t).cpu().numpy()>0)&valid;g=gt&valid;scores.append(float((pred&g).sum()/max(1,(pred|g).sum())))
                full.append({'camera':v.name,'before':scores[0],'after':scores[1]})
            if any(x['after']<x['before']-.025 for x in full):accepted=False
        result=refined if accepted else T
        return result,{'support':support,'accepted':accepted,'head_iou_before':before,'head_iou_after':after,'depth_mm':float(res[0]*1000),'depth_n':res[1],'full_object_guard':full,'hole_landmarks':hole_records}
