"""Exactly one current RGB frame enters SAM2 per step, shared by replay/live.
Seeds come from current RGB-D + the fixed mesh/robot library. No manual labels.
"""
import tempfile,os
import numpy as np,cv2,torch

class Frames:
    def __init__(self,ims,size):self.ims,self.size=ims,size
    def __len__(self):return len(self.ims)
    def __getitem__(self,i):
        im=cv2.resize(cv2.cvtColor(self.ims[i],cv2.COLOR_BGR2RGB),(self.size,self.size));x=torch.from_numpy(im).permute(2,0,1).float()/255
        return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

class ObjectMaskStream:
    def __init__(self,model,image,seeds,zbufs,ids,depth,cold=True,image_refine=True,debug_dir=None,reference_colors=None):
        self.model=model;self.images=[image.copy()];self.t=-1;self.temp=tempfile.TemporaryDirectory(prefix='realpc_masks_')
        cv2.imwrite(self.temp.name+'/000000.jpg',image);h,w=image.shape[:2]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=model.init_state(self.temp.name,offload_video_to_cpu=True,offload_state_to_cpu=True)
            state['images']=Frames(self.images,model.image_size);state['num_frames']=1;state['video_height'],state['video_width']=h,w
            if image_refine:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                image_predictor=SAM2ImagePredictor(model)
                image_predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            for oid, m in seeds.items():
                if image_refine and oid < 254:
                    # The renderer provides identity/location, not the final RGB silhouette.
                    # For a heavily occluded cold object, the visible seed may be only
                    # its neck. Use the amodal projected extent for the SAM box so the
                    # visible handle is not cropped away; positives still need depth.
                    box_mask = (zbufs[ids.index(oid)] > 0) if cold else m
                    yy, xx = np.nonzero(box_mask)
                    pad = max(8, int(.12 * max(xx.max()-xx.min(), yy.max()-yy.min())))
                    box = np.array([max(0,xx.min()-pad), max(0,yy.min()-pad),
                                    min(w-1,xx.max()+pad), min(h-1,yy.max()+pad)], np.float32)
                    point_mask = m.copy()
                    if cold and oid < 254:
                        observed=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
                        predicted_depth=zbufs[ids.index(oid)]
                        explained=m&(observed>.1)&(predicted_depth>0)&(np.abs(observed-predicted_depth)<.025)
                        if explained.sum()>30:
                            point_mask=explained
                    if oid < 254:
                        from realpc import objlib
                        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).reshape(-1,3).astype(float)/255
                        ref = reference_colors[int(oid)] if reference_colors is not None else rgb[point_mask.ravel()]
                        probability = objlib.colour_model(ref,rgb[::40])(rgb).reshape(h,w)
                        support = point_mask & (probability > max(.40,float(probability[point_mask].max())*.70))
                        if support.sum()>30:
                            point_mask=support
                    dist = cv2.distanceTransform(point_mask.astype(np.uint8), cv2.DIST_L2, 5)
                    y0,x0 = np.unravel_index(dist.argmax(),dist.shape)
                    points = [(x0,y0)]
                    py,px=np.nonzero(point_mask)
                    candidates = np.c_[px,py];clear = dist[py,px]
                    for _ in range(2):
                        dd = np.min(np.linalg.norm(candidates[:,None,:]-np.array(points)[None],axis=2),axis=1)
                        j = np.argmax(dd * np.minimum(clear/3.,1.))
                        points.append(tuple(candidates[j]))
                    pos_count=len(points)
                    outside=~cv2.dilate(m.astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool)
                    roi=np.zeros((h,w),bool);x1,y1,x2,y2=box.astype(int);roi[y1:y2+1,x1:x2+1]=True
                    outside &= roi
                    by,bx=np.nonzero(outside)
                    if len(bx):
                        bg=np.c_[bx,by]
                        for target in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                            points.append(tuple(bg[np.argmin(np.linalg.norm(bg-np.array(target),axis=1))]))
                    point_labels=np.r_[np.ones(pos_count,np.int32),np.zeros(len(points)-pos_count,np.int32)]
                    predicted, _, _ = image_predictor.predict(point_coords=np.asarray(points,np.float32),
                        point_labels=point_labels,box=box,multimask_output=False)
                    refined = predicted[0].astype(bool)
                    if cold:
                        # A prompted held object often includes its gripper. Known FK
                        # robot depth removes that lobe before the seed-quality guard.
                        robot_depth=np.minimum.reduce([np.where(zbufs[j]>0,zbufs[j],np.inf) for j,k in enumerate(ids) if k>=254])
                        observed=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
                        refined &= ~((robot_depth<np.inf)&(observed>.1)&(robot_depth<=observed+.012))
                    debug=image.copy()
                    cv2.rectangle(debug,tuple(box[:2].astype(int)),tuple(box[2:].astype(int)),(255,0,0),2)
                    for ip,pt in enumerate(points):cv2.circle(debug,tuple(map(int,pt)),4,(0,255,0) if ip<pos_count else (255,0,255),-1)
                    debug[refined]=(.35*np.array([0,0,255])+.65*debug[refined]).astype(np.uint8)
                    if debug_dir is not None:cv2.imwrite(os.path.join(debug_dir,f"seed{oid}.jpg"),debug)
                    # A clear, depth-supported cold pose deserves the same local
                    # uncertainty guard as a warm pose. Occluded starts retain the
                    # wider amodal mask needed to discover a hidden handle.
                    amodal=zbufs[ids.index(oid)]>0;predicted=zbufs[ids.index(oid)]
                    observed=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
                    rz=np.minimum.reduce([np.where(zbufs[j]>0,zbufs[j],np.inf) for j,k in enumerate(ids) if k>=254])
                    near_robot=cv2.dilate(((rz<np.inf)&(rz<predicted+.020)&amodal).astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
                    unoccluded=(amodal&~near_robot).sum()/max(1,amodal.sum())
                    agreement=(amodal&(observed>.1)&(abs(observed-predicted)<.025)).sum()/max(1,amodal.sum())
                    local_guard=not cold or (unoccluded>.98 and agreement>.65)
                    if local_guard:
                        refined &= cv2.dilate(m.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
                    coverage = (refined & m).sum() / max(1,m.sum())
                    # Cold pose hypotheses are uncertain: demanding 72% overlap with
                    # their rendered silhouette rejects correct image boundaries.
                    threshold = .25 if cold else .72
                    area_prior = box_mask.sum() if cold else m.sum()
                    if coverage < threshold or refined.sum() > 2.0*area_prior:
                        # A boundary seed can hit background or only one material of a
                        # multi-material object. Preserve the conservative original seed.
                        refined = m
                        print(f"seed {oid}: image refinement rejected (coverage={coverage:.2f})",flush=True)
                    else:
                        ncc, cc, stats, _ = cv2.connectedComponentsWithStats(refined.astype(np.uint8))
                        keep = [j for j in range(1,ncc) if ((cc==j)&m).sum()>=20]
                        refined = np.isin(cc,keep)
                    model.add_new_mask(state, frame_idx=0, obj_id=int(oid), mask=refined)
                else:
                    model.add_new_mask(state, frame_idx=0, obj_id=int(oid), mask=m)
        self.state=state
    def step(self,image):
        self.t+=1
        if self.t>0:self.images.append(image.copy())
        self.state['num_frames']=len(self.images)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=list(self.model.propagate_in_video(self.state,start_frame_idx=self.t,max_frame_num_to_track=0,reverse=False))
            _,oids,logits=result[-1];x=logits[:,0].float().cpu().numpy();lab=np.asarray(oids,np.uint8)[x.argmax(0)];lab[x.max(0)<=0]=0
        return lab

def render_seeds(conv,c,shape,q):
    """Automatic identities/seeds from this frame's initialized meshes and FK."""
    from .diffcalib import Renderer
    h,w=shape;K=conv.ep.intrinsics(c).copy();K[:2]*=w/(1280 if c=='top' else 640);R=Renderer(K,h,w);Tc=torch.tensor(np.linalg.inv(conv.cal.T_world_cam(c)),device='cuda',dtype=torch.float32)
    zbufs=[];ids=[]
    for i,o in enumerate(conv.objs):
        if not conv.trk.state[o.name].valid:continue
        T=torch.tensor(conv.trk.state[o.name].T,device='cuda',dtype=torch.float32);V=o.verts_t@T[:3,:3].T+T[:3,3];_,z=R.raster((V@Tc[:3,:3].T+Tc[:3,3])[None],o.faces_t);zbufs.append(z[0].cpu().numpy());ids.append(i+1)
    rob=conv.trk.robot(q,conv.T_world_base)
    for side,oid in [('left',254),('right',255)]:
        V=rob[side]['V'];_,z=R.raster((V@Tc[:3,:3].T+Tc[:3,3])[None],conv.tarm.faces);zbufs.append(z[0].cpu().numpy());ids.append(oid)
    Z=np.stack([np.where(z>0,z,np.inf) for z in zbufs]);front=Z.argmin(0);valid=np.isfinite(Z).any(0);seeds={}
    for k,oid in enumerate(ids):
        mask=cv2.erode((valid&(front==k)).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        if mask.sum()>=40:seeds[oid]=mask
    return seeds,zbufs,ids

class OnlineObjectSegmenter:
    """Feed one current packet; returns current top/side labels for StreamConverter."""
    def __init__(self,conv,model):self.conv,self.model,self.memory=conv,model,{}
    def step(self,t,frames,q=None):
        conv=self.conv;first_packet=not self.memory
        if q is None:q={a:conv.cal.q_at(conv.ep,a,conv.cal.robot_tick(conv.ep,'top',t)) for a in ['left','right']}
        if conv.trk is None:conv._init(t,frames,q=q)
        out={}
        for c in ['top','side']:
            if c not in frames:continue
            D,im=frames[c];image=cv2.resize(im,(640,int(round(im.shape[0]*640/im.shape[1]))))
            if c not in self.memory:
                seeds,zbufs,ids=render_seeds(conv,c,image.shape[:2],q);colors={i+1:conv.trk.lib_points[o.name][1] for i,o in enumerate(conv.objs)}
                self.memory[c]=ObjectMaskStream(self.model,image,seeds,zbufs,ids,D,cold=conv.cold,image_refine=True,reference_colors=colors)
            out[c]=self.memory[c].step(image)
        if first_packet:
            # Cached replay initializes from these same current masks. The live
            # geometry-only pose was a seed, not an emitted tracker state.
            conv.trk=None
        return out
