"""Shared causal depth/mesh extraction with event-only SAM initialization/recovery."""
import os,time
import numpy as np
from .meshtrack import CamView

class DepthMeshPipeline:
    def __init__(self,converter,sam_checkpoint):
        os.environ['REALPC_DEPTH_MESH']='1'
        self.converter=converter;converter.seg={}
        self.checkpoint=sam_checkpoint;self.model=None;self.events=[]
        self.bad={};self.pending=[];self.last_recovery={};self.timings=[];self.last_health={}

    def _sam(self,t,frames,q,initial):
        start=time.perf_counter();c=self.converter
        if self.model is None:
            from sam2.build_sam import build_sam2_video_predictor
            self.model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',self.checkpoint,device='cuda')
        if not initial:
            views=[CamView(k,D,c.ep.intrinsics(k),c.cal.T_world_cam(k),rgb=im) for k,(D,im) in frames.items() if k in ('top','side')]
            rob=c.trk.robot(q,c.T_world_base)
            for n in self.pending:
                st=c.trk.state[n]
                if st.held is not None:
                    st.T=rob[st.held]['G']@st.T_grip_obj
                else:
                    others=[o.samples@c.trk.state[k].T[:3,:3].T+c.trk.state[k].T[:3,3] for k,o in c.trk.objs.items() if k!=n and c.trk.state[k].valid]
                    T=c.trk.relocalize(c.trk.objs[n],views,rob,others)
                    if T is not None:st.T=T;st.valid=True;st.carrier=None;st.T_car_obj=None
                self.last_recovery[n]=t;self.bad[n]=0
        from .online_masks import OnlineObjectSegmenter
        sampler=OnlineObjectSegmenter(c,self.model,reset_tracker_on_seed=initial)
        labels=sampler.step(t,frames,q=q)
        for m in sampler.memory.values():self.model.reset_state(m.state)
        sampler.memory.clear()
        self.events.append({'tick':int(t),'reason':'initialize' if initial else 'reacquire','objects':[o.name for o in c.objs] if initial else list(self.pending),'trigger':{} if initial else {n:self.last_health.get(n,{}) for n in self.pending},'camera_calls':len(labels),'seconds':time.perf_counter()-start})
        return labels

    def step(self,t,frames,q=None):
        start=time.perf_counter();c=self.converter
        if q is None:q={a:c.cal.q_at(c.ep,a,c.cal.robot_tick(c.ep,'top',t)) for a in ('left','right')}
        initial=c.trk is None
        labels=self._sam(t,frames,q,initial) if initial or self.pending else None
        out=c.step(t,frames,labels=labels,q=q)
        self.pending=[]
        for n,d in getattr(c.trk,'depth_diagnostics',{}).items():
            st=c.trk.state[n]
            if getattr(c.trk.objs[n],'passive_fixture',False):continue
            bad=d['visible']>=80 and (d['free_fraction']>.40 or d['support']<30)
            residual,count=st.last_res
            self.last_health[n]={**d,'residual_mm':float(residual*1000) if np.isfinite(residual) else None,'residual_count':int(count),'mode':st.mode}
            bad |= d['visible']>=80 and count>=40 and np.isfinite(residual) and residual>.010
            self.bad[n]=self.bad.get(n,0)+1 if bad else 0
            if self.bad[n]>=3 and t-self.last_recovery.get(n,c.start)>=30:self.pending.append(n)
        self.timings.append(time.perf_counter()-start)
        return out

    def summary(self):
        x=np.asarray(self.timings)
        return {'frames':len(x),'step_seconds':float(x.sum()),'step_fps':float(len(x)/x.sum()) if len(x) else 0,'steady_median_ms':float(np.median(x[1:])*1000) if len(x)>1 else None,'sam_events':self.events,'sam_camera_calls':sum(e['camera_calls'] for e in self.events)}
