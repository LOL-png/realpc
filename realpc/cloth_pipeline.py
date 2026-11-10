"""One causal RGB-D/FK -> sparse masks -> tracks -> material-mesh pipeline."""
import time
import numpy as np
from .cloth_causal import CausalClothTracker,CAMS
from .cloth_sparse_masks import SparseClothMasks
from .cloth_cotracker import CausalClothCoTracker

class CausalClothPipeline:
    def __init__(self,config):
        self.config=dict(config);self.tracker=CausalClothTracker.from_config(config)
        self.masker=SparseClothMasks(config.get('sam_checkpoint','/home/howardji/realpc/.model_cache/sam2/sam2.1_hiera_small.pt'),interval=config.get('mask_interval',10),sam_backend=config.get('mask_backend','video'),fast_neighbors=config.get('fast_neighbors',False))
        self.cot=CausalClothCoTracker() if config.get('cot_weight',0) else None
        self.timings=[];self.last_packet=None

    def step(self,t,raw_packet):
        start=time.perf_counter()
        # Never reuse material tracks seeded from an uncounted mask producer.
        packet={k:v for k,v in raw_packet.items() if '_cot_' not in k}
        masks,mask_stats=self.masker.step(t,packet)
        for c in CAMS:packet[c+'_mask']=masks.get(c,np.zeros_like(packet[c+'_depth'],bool))
        if self.cot is not None:
            clean=self.tracker.views(packet)
            for c in CAMS:
                if c in self.masker.cameras:track=self.cot.step(c,t,packet[c+'_rgb'],clean[c]['M'])
                else:track=dict(uv=np.empty((0,2),np.float32),visible=np.empty(0,bool),seed_uv=np.empty((0,2),np.float32),birth=np.array(0))
                packet.update({c+'_cot_'+k:v for k,v in track.items()})
            packet['max_observed_tick']=t;packet['zero_lookahead']=True
        result=self.tracker.step(t,packet,iters=self.config.get('iters',5))
        seconds=time.perf_counter()-start;self.timings.append(seconds)
        result[1]['mask_pipeline']=mask_stats;result[1]['pipeline_seconds']=seconds
        self.last_packet=packet
        return result

    def summary(self):
        seconds=sum(self.timings)
        return dict(frames=len(self.timings),seconds=seconds,fps=len(self.timings)/max(seconds,1e-9),masks=self.masker.summary(),cameras=list(self.masker.cameras),causal=True,cached_cotracker_used=False)
