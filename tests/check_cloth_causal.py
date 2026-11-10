"""Real-input API parity + fixed material grasp regression. No future frames."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.cloth_causal import CausalClothTracker,front_face_anchors

def main():
    # Overlapping layers: the camera sees z=1, even if a depth-biased query
    # would be closer in 3D to the back layer at z=1.01.
    tri=np.array([[-1.,-1.,1.],[1.,-1.,1.],[0.,1.,1.]])
    back=tri.copy();back[:,2]+=.01;vertices=np.vstack([back,tri])
    ii,depth=front_face_anchors(vertices,np.array([[0,1,2],[3,4,5]]),np.array([[0.,0.]]),np.eye(3),np.eye(4))
    assert ii[0]>=3 and abs(depth[0]-1)<1e-10
    p=Path('/home/howardji/data_ext/astra_cloth/cloth01/inputs/frames')
    config=dict(dense=True,coverage=2.,arap=4.,track=4.)
    live=CausalClothTracker(**config);replay=CausalClothTracker(**config)
    for t in range(3):
        # A live producer supplies exactly this single image/depth/FK packet.
        with np.load(p/f'{t:04d}.npz') as z:packet={k:z[k] for k in z.files}
        A,_,_=live.step(t,packet)
        B,_,_=replay.step(t,{k:v.copy() for k,v in packet.items()})
        assert np.array_equal(A,B),'live/replay diverged'
    tr=live;X=tr.cm.X.copy();packet={k:v.copy() for k,v in packet.items()}
    for arm in ('left','right'):packet['q_'+arm][-1]=1.
    tr.update_pins(packet,X)
    packet['q_left'][-1]=0.;packet['tips_left']=X[[0]];packet['G_left']=np.eye(4)
    idx,target=tr.update_pins(packet,X);assert len(idx)>0
    frozen=tr.pins['left'][0].copy();local=tr.pins['left'][1].copy()
    packet['tips_left']=X[[-1]];packet['G_left'][0,3]=.02
    tr.update_pins(packet,X)
    assert np.array_equal(frozen,tr.pins['left'][0]),'reselected a new material layer'
    assert np.array_equal(local,tr.pins['left'][1]),'changed grasp coordinates'
    packet['q_left'][-1]=1.;tr.update_pins(packet,X);assert 'left' not in tr.pins
    try:tr.step(1,packet)
    except AssertionError:pass
    else:raise AssertionError('accepted out-of-order frame')
    print('PASS front-layer ray binding, current-packet live/replay parity, fixed material grasp, release, frame order')
if __name__=='__main__':main()
