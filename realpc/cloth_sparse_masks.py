"""Causal sparse-SAM cloth masks, with depth/FK surface tracking between calls.

step consumes one current calibrated packet. Cached SAM masks in that packet
are deliberately ignored. This API is shared by streaming and conversion.
"""
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .cloth_inputs import unproject, project
from .cloth_seed import depth_seed

class SparseClothMasks:
    def __init__(self,checkpoint,interval=10,cameras=('top','side'),sam_backend='video'):
        self.checkpoint=checkpoint;self.interval=interval;self.cameras=cameras;self.sam_backend=sam_backend;self.video=None
        self.predictor=None;self.cloud=None;self.plane=None;self.previous=None
        self.tick=-1;self.last_sam=-1000;self.events=[];self.timings=[];self.previous_area={};self.previous_images={};self.previous_masks={}

    def _sam(self,image,seed,cam=None):
        if self.sam_backend=='video':
            from .cloth_inputs import OnlineClothMasks
            if self.video is None:self.video=OnlineClothMasks(self.checkpoint)
            return self.video.step(cam,image,seed)
        import torch
        if self.predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',self.checkpoint,device='cuda'))
        yy,xx=np.nonzero(seed)
        if len(xx)<20:raise ValueError('No current depth-supported cloth prompt')
        dist=cv2.distanceTransform(seed.astype(np.uint8),cv2.DIST_L2,5);cand=np.c_[xx,yy];points=[]
        for _ in range(5):
            score=dist[yy,xx] if not points else np.minimum(dist[yy,xx],8)*np.min(np.linalg.norm(cand[:,None]-np.array(points)[None],axis=2),axis=1)
            points.append(cand[score.argmax()])
        h,w=image.shape[:2];box=np.array([max(0,xx.min()-10),max(0,yy.min()-10),min(w-1,xx.max()+10),min(h-1,yy.max()+10)],np.float32)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            self.predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            mm,_,_=self.predictor.predict(point_coords=np.array(points,np.float32),point_labels=np.ones(5,np.int32),box=box,multimask_output=False)
        return mm[0].astype(bool)

    def step(self,t,frame,predicted_vertices=None):
        assert t==self.tick+1,'Supply current packets in causal order'
        start=time.perf_counter();self.tick=t;robot=cKDTree(frame['robot'])
        if t==0:
            initial,self.plane=depth_seed(frame)
            self.cloud,_=unproject(frame['top_depth'],frame['top_K'],frame['top_T'],2,initial)
        predicted=self.cloud.copy() if predicted_vertices is None else np.asarray(predicted_vertices).copy()
        if self.previous is not None and predicted_vertices is None:
            delta=np.zeros_like(predicted);mass=np.zeros(len(predicted))
            for arm in ('left','right'):
                if frame['q_'+arm][-1]>=.25 or self.previous['q_'+arm][-1]>=.25:continue
                G0=self.previous['G_'+arm];G1=frame['G_'+arm];T=G1@np.linalg.inv(G0)
                d=cKDTree(self.previous['tips_'+arm]).query(predicted)[0];w=np.exp(-(d/.10)**2)
                delta+=w[:,None]*(predicted@T[:3,:3].T+T[:3,3]-predicted);mass+=w
            predicted+=delta/np.maximum(mass,1)[:,None]
        tree=cKDTree(predicted);masks={};diagnostics={};bad=False
        for cam in self.cameras:
            D,K,T,rgb=[frame[cam+'_'+k] for k in ('depth','K','T','rgb')]
            P,uv=unproject(D,K,T,2);height=P[:,2]-np.c_[P[:,:2],np.ones(len(P))]@self.plane
            color=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)[uv[:,1],uv[:,0]]
            rd=robot.query(P,workers=1)[0];distance=tree.query(P,workers=1)[0]
            workspace=(P[:,0]>.05)&(P[:,0]<1.15)&(P[:,1]>-.85)&(P[:,1]<.35)&(height>-.003)&(height<.8)
            # A narrow gate includes cloth resting at table height; the wider
            # gate admits an elevated fold moving away from its previous surface.
            good=workspace&(rd>.012)&(color[:,1]<110)&(color[:,2]>45)&((distance<.014)|((height>.006)&(distance<.070)))
            M=np.zeros(D.shape,np.uint8);M[uv[good,1],uv[good,0]]=1
            M=cv2.morphologyEx(M,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            if cam in self.previous_images:
                # Track the last observed silhouette every frame without SAM.
                # Depth alone cannot distinguish a flat shirt from its table.
                gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
                flow=cv2.calcOpticalFlowFarneback(gray,self.previous_images[cam],None,.5,3,21,3,5,1.2,0)
                yy,xx=np.mgrid[:len(M),:M.shape[1]].astype(np.float32)
                warped=cv2.remap(self.previous_masks[cam].astype(np.float32),xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR)>.5
                nearby=cv2.dilate(warped.astype(np.uint8),np.ones((11,11),np.uint8))>0
                elevated=np.zeros(D.shape,np.uint8);elevated[uv[height>.008,1],uv[height>.008,0]]=1
                elevated=cv2.morphologyEx(elevated,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))>0
                M &= (warped|(nearby&elevated)).astype(np.uint8)
            nc,L,stats,_=cv2.connectedComponentsWithStats(M)
            if nc>1:
                keep=np.flatnonzero(stats[:,4]>max(50,stats[1:,4].max()*.03));keep=keep[keep!=0];M=np.isin(L,keep).astype(np.uint8)
            area=int(M.sum());ratio=area/max(self.previous_area.get(cam,area),1)
            failed=area<100 or ratio<.60 or ratio>1.65
            bad |= failed;diagnostics[cam]=dict(area=area,area_ratio=float(ratio),failed=bool(failed))
            masks[cam]=M.astype(bool)
        # Current cross-camera geometry is an independent reacquisition cue.
        # A stale top silhouette on the table must not keep prompting SAM there.
        elevated_clouds={}
        for cam in self.cameras:
            P,_=unproject(frame[cam+'_depth'],frame[cam+'_K'],frame[cam+'_T'],2,masks[cam])
            h=P[:,2]-np.c_[P[:,:2],np.ones(len(P))]@self.plane
            elevated_clouds[cam]=P[(h>.008)&(robot.query(P,workers=1)[0]>.012)]
        cross_seeds={}
        for cam in self.cameras:
            sources=[P for other,P in elevated_clouds.items() if other!=cam and len(P)]
            D=frame[cam+'_depth'];seed=np.zeros(D.shape,np.uint8)
            if sources:
                uv,z=project(np.concatenate(sources),frame[cam+'_K'],frame[cam+'_T']);xy=np.rint(uv).astype(int)
                ok=(z>.08)&(xy[:,0]>=0)&(xy[:,0]<D.shape[1])&(xy[:,1]>=0)&(xy[:,1]<len(D))
                xy=xy[ok];z=z[ok];dd=D[xy[:,1],xy[:,0]];keep=(dd>.08)&(abs(dd-z)<.020)
                xy=xy[keep];seed[xy[:,1],xy[:,0]]=1
                seed=cv2.morphologyEx(seed,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
                seed=cv2.dilate(seed,np.ones((3,3),np.uint8))
            cross_seeds[cam]=seed.astype(bool)
            recall=float((seed.astype(bool)&masks[cam]).sum()/max(seed.sum(),1))
            mismatch=seed.sum()>150 and recall<.70
            diagnostics[cam].update(cross_support=int(seed.sum()),cross_recall=recall,cross_failed=bool(mismatch))
            bad |= mismatch
        due=t==0 or (self.interval>0 and t-self.last_sam>=self.interval) or (bad and t-self.last_sam>=3)
        calls=0
        if due:
            for cam in self.cameras:
                seed=initial if t==0 and cam=='top' else (cross_seeds[cam] if diagnostics[cam]['cross_failed'] else masks[cam]|cross_seeds[cam])
                if seed.sum()<20:continue
                masks[cam]=self._sam(frame[cam+'_rgb'],seed,cam);calls+=1
            if calls:
                self.last_sam=t;self.events.append(dict(tick=t,reason='init' if t==0 else ('failure' if bad else 'periodic'),camera_calls=calls,diagnostics=diagnostics))
        points=[]
        for cam in self.cameras:
            P,uv=unproject(frame[cam+'_depth'],frame[cam+'_K'],frame[cam+'_T'],2,masks[cam]);P=P[robot.query(P,workers=1)[0]>.012]
            if len(P):points.append(P)
            self.previous_area[cam]=int(masks[cam].sum())
            self.previous_images[cam]=cv2.cvtColor(frame[cam+'_rgb'],cv2.COLOR_BGR2GRAY);self.previous_masks[cam]=masks[cam].copy()
        if points:
            P=np.concatenate(points);self.cloud=P[::max(1,len(P)//6000)]
        self.previous={k:frame[k].copy() for arm in ('left','right') for k in ('q_'+arm,'G_'+arm,'tips_'+arm)}
        elapsed=time.perf_counter()-start;self.timings.append(elapsed)
        return masks,dict(tick=t,sam_called=bool(calls),camera_calls=calls,seconds=elapsed,diagnostics=diagnostics,max_observed_tick=t)

    def summary(self):
        n=len(self.timings);calls=sum(e['camera_calls'] for e in self.events)
        return dict(interval=self.interval,sam_backend=self.sam_backend,frames=n,sam_frames=len(self.events),sam_frame_rate=len(self.events)/max(n,1),sam_camera_calls=calls,seconds=sum(self.timings),fps=n/max(sum(self.timings),1e-9),events=self.events,causal=True)
