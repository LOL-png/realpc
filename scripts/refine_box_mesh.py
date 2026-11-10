"""Fit four slightly slanted cardboard walls from the aligned multiepisode scans."""
import sys,json
from pathlib import Path
import numpy as np,trimesh
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));a=ROOT/'assets/objects_v3/sweep';meta=json.load(open(a/'box.json'));z=np.load(a/'box_points.npz');P=z['P'];dims=np.array(meta['dimensions_m']);walls=[]
for axis in [0,1]:
 for sign in [-1,1]:
  near=(np.abs(P[:,axis]-sign*dims[axis]/2)<.018)&(P[:,2]>.012)&(P[:,2]<dims[2]-.006)
  Q=P[near];X=np.c_[Q[:,1-axis],Q[:,2],np.ones(len(Q))];fit=least_squares(lambda v:X@v-Q[:,axis],[0,0,sign*dims[axis]/2],loss='soft_l1',f_scale=.002,bounds=([-.1,-.4,sign*dims[axis]/2-.015],[.1,.4,sign*dims[axis]/2+.015]));walls.append({'axis':axis,'sign':sign,'coeff':fit.x.tolist(),'median_mm':float(np.median(abs(X@fit.x-Q[:,axis]))*1000)})
def corner(sx,sy,height):
 wx=next(w for w in walls if w['axis']==0 and w['sign']==sx)['coeff'];wy=next(w for w in walls if w['axis']==1 and w['sign']==sy)['coeff'];xy=np.linalg.solve([[1,-wx[0]],[-wy[0],1]],[wx[1]*height+wx[2],wy[1]*height+wy[2]]);return np.r_[xy,height]
# Closed thin prisms. Top-edge profile is fitted only where repeatedly observed.
parts=[];height=dims[2];th=.003
floor=trimesh.creation.box([dims[0],dims[1],th]);floor.apply_translation([0,0,-th/2]);parts.append(floor)
for w in walls:
 axis,sign=w['axis'],w['sign'];other=1-axis;bottom=[corner(sign,s,0) if axis==0 else corner(s,sign,0) for s in [-1,1]];top=[corner(sign,s,height) if axis==0 else corner(s,sign,height) for s in [-1,1]]
 v=np.array([bottom[0],bottom[1],top[1],top[0]]);normal=np.zeros(3);normal[axis]=sign*th/2;V=np.r_[v-normal,v+normal];f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]);parts.append(trimesh.Trimesh(V,f,process=False))
m=trimesh.util.concatenate(parts);m.visual.vertex_colors=[163,126,77,255];m.export(a/'box.ply');meta['wall_planes']=walls;meta['shape']='floor + four measured slanted wall planes, 3mm thickness prior';meta['watertight']=bool(m.is_watertight);(a/'box.json').write_text(json.dumps(meta,indent=2));print(walls)
