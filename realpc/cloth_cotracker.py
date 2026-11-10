"""Zero-lookahead CoTracker3 wrapper: only one newly observed RGB per step.

The stock online helper emits retrospective estimates with 8-frame latency.
Here partial windows contain only observed frames (the model repeats CURRENT
image for padding), and online memory is committed only at a complete window.
Already emitted tracks are never revised. Different cameras share weights.
"""
import copy,sys
import cv2,numpy as np,torch

class CausalClothCoTracker:
    def __init__(self,root='/home/howardji/realpc/.model_cache',reseed=128):
        sys.path.insert(0,root+'/hub/facebookresearch_co-tracker_main')
        from cotracker.predictor import CoTrackerOnlinePredictor
        self.model=CoTrackerOnlinePredictor(checkpoint=root+'/hub/checkpoints/scaled_online.pth').cuda().eval()
        self.states={};self.reseed=reseed

    @staticmethod
    def seeds(gray,mask):
        corners=cv2.goodFeaturesToTrack(gray,180,.01,8,mask=mask.astype(np.uint8)*255)
        a=corners[:,0] if corners is not None else np.empty((0,2),np.float32)
        yy,xx=np.mgrid[0:mask.shape[0]:16,0:mask.shape[1]:16];b=np.c_[xx[mask[yy,xx]],yy[mask[yy,xx]]].astype(np.float32)
        if len(b)>100:b=b[np.linspace(0,len(b)-1,100).astype(int)]
        if len(a) and len(b):b=b[np.min(np.linalg.norm(b[:,None]-a[None],axis=2),axis=1)>5]
        return np.concatenate([a,b])[:256]

    @torch.inference_mode()
    def step(self,cam,t,rgb,mask):
        model=self.model;core=model.model
        if cam not in self.states or t-self.states[cam]['birth']>=self.reseed:
            uv=self.seeds(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),mask);q=torch.as_tensor(np.c_[np.zeros(len(uv)),uv],dtype=torch.float32,device='cuda')[None]
            image=torch.from_numpy(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)).permute(2,0,1)[None,None].float().cuda()
            model(image,is_first_step=True,queries=q,add_support_grid=True)
            self.states[cam]=dict(birth=t,images=[],start=0,seed=uv,queries=model.queries.clone(),N=model.N,
                memory={k:copy.deepcopy(v) for k,v in vars(core).items() if k.startswith('online_')})
        s=self.states[cam];s['images'].append(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB));relative=t-s['birth']
        assert relative==len(s['images'])-1
        model.queries=s['queries'];model.N=s['N']
        for k,val in s['memory'].items():setattr(core,k,copy.deepcopy(val))
        chunk=torch.from_numpy(np.stack(s['images'][s['start']:])).permute(0,3,1,2)[None].float().cuda()
        assert chunk.shape[1]<=16
        with torch.autocast('cuda',dtype=torch.bfloat16):tracks,vis=model(chunk,add_support_grid=True)
        current=tracks[0,relative].float().cpu().numpy();visible=vis[0,relative].cpu().numpy()
        if chunk.shape[1]==16:
            s['memory']={k:copy.deepcopy(v) for k,v in vars(core).items() if k.startswith('online_')};s['start']+=model.step
        return dict(uv=current,visible=visible,seed_uv=s['seed'],birth=np.array(s['birth']))
