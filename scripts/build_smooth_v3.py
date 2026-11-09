#!/usr/bin/env python
"""Reproducible smooth priors, dimensioned in the measured v1 object frames.
Unseen backs, wall thickness and peg/handle completion are priors, not scans.
"""
import sys,json,shutil
from pathlib import Path
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter,distance_transform_edt
from scipy.interpolate import splprep,splev
from skimage.measure import marching_cubes
import cv2
ROOT=Path(__file__).resolve().parents[1]

def color(m,c):
 m.visual.vertex_colors=np.tile(np.r_[c,255],(len(m.vertices),1));return m

def cylinder(a,b,r,c,n=48):
 a,b=np.array(a),np.array(b);m=trimesh.creation.cylinder(radius=r,segment=np.array([a,b]),sections=n);return color(m,c)

def lathe(profile,n=96):
 # closed cross-section in (radius,z), including interior wall for the mug
 p=np.array(profile);theta=np.arange(n)*2*np.pi/n
 V=np.stack([p[:,0,None]*np.cos(theta),p[:,0,None]*np.sin(theta),np.broadcast_to(p[:,1,None],(len(p),n))],-1).reshape(-1,3)
 F=[]
 for j in range(len(p)):
  for i in range(n):
   a=j*n+i;b=j*n+(i+1)%n;c=((j+1)%len(p))*n+i;d=((j+1)%len(p))*n+(i+1)%n;F += [[a,c,b],[b,c,d]]
 m=trimesh.Trimesh(V,F,process=True);m.fix_normals();return m

def tube(path,r1,r2,c,n=16):
 path=np.asarray(path);vs=[]
 for i,p in enumerate(path):
  t=path[min(i+1,len(path)-1)]-path[max(i-1,0)];t/=np.linalg.norm(t)
  v=np.cross(t,[0,1,0]);v/=np.linalg.norm(v);w=np.cross(t,v)
  th=np.arange(n)*2*np.pi/n;vs.extend(p+ r1*np.cos(th)[:,None]*v+r2*np.sin(th)[:,None]*w)
 fs=[]
 for j in range(len(path)-1):
  for i in range(n):
   a=j*n+i;b=j*n+(i+1)%n;c1=a+n;d=b+n;fs += [[a,b,c1],[b,d,c1]]
 # flat end caps, buried inside the cup
 for i in range(1,n-1):fs += [[0,i+1,i],[(len(path)-1)*n,(len(path)-1)*n+i,(len(path)-1)*n+i+1]]
 m=trimesh.Trimesh(vs,fs,process=True);m.fix_normals();return color(m,c)

def field_mesh(F,origin,step,c):
 V,Fc,_,_=marching_cubes(F,1e-7,spacing=(step,)*3,allow_degenerate=False);m=trimesh.Trimesh(V+origin,Fc,process=True);m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces());m.remove_unreferenced_vertices();m.fix_normals()
 # topology preserving smoothing, sub-mm; retain actual thin-wall/hole structure
 trimesh.smoothing.filter_taubin(m,iterations=4)
 if len(m.faces)>16000:m=m.simplify_quadric_decimation(face_count=16000)
 return color(m,c)

def footprint(outline,holes=(),step=.0015):
 outline=np.array(outline);lo=outline.min(0)-.012;hi=outline.max(0)+.012;xy=np.mgrid[lo[0]:hi[0]:step,lo[1]:hi[1]:step]
 mask=np.zeros(xy.shape[1:],np.uint8)
 # interpolate a closed smooth contour, rather than preserving scan triangulation
 tck,_=splprep(outline.T,s=0,per=True);q=np.array(splev(np.linspace(0,1,400),tck)).T
 cv2.fillPoly(mask,[np.round((q-lo)/step).astype(np.int32)[:,::-1]],1)
 for center,radii,angle in holes:
  d=xy-np.array(center)[:,None,None];ca,sa=np.cos(angle),np.sin(angle);u=ca*d[0]+sa*d[1];v=-sa*d[0]+ca*d[1];mask[(u/radii[0])**2+(v/radii[1])**2<1]=0
 sd=(distance_transform_edt(~mask.astype(bool))-distance_transform_edt(mask))*step
 return xy,sd,lo

def extrude(outline,center,half,c,holes=(),step=.0015):
 xy,sd,lo=footprint(outline,holes,step);z=np.arange(-.03,.045,step)
 mid=center(xy);h=half(xy) if callable(half) else half
 # rounded plate edges without closing holes
 dz=np.abs(z[None,None,:]-mid[:,:,None])-np.asarray(h)
 F=np.maximum(sd[:,:,None],dz)
 F=gaussian_filter(F,.6)
 return field_mesh(F,np.r_[lo,z[0]],step,c)

def build(name):
 if name=='mug':
  # Lip rounded in profile; hollow interior with a 2.5mm wall and 4mm base.
  p=[(.0001,-.037),(.034,-.037),(.039,-.034),(.0425,.033),(.043,.037),(.042,.039),(.0405,.039),(.040,.037),(.038,-.030),(.034,-.033),(.0001,-.033)]
  cup=color(lathe(p),[205,83,19]);th=np.linspace(-np.pi/2,np.pi/2,55)
  path=np.c_[.040+.027*np.cos(th),np.zeros(len(th)),.029*np.sin(th)]
  h=tube(path,.005,.007,[205,83,19]);h.apply_transform(trimesh.transformations.rotation_matrix(-.50,[0,0,1]));m=trimesh.util.concatenate([cup,h])
  z=np.array([-.86,.02,.51]);z/=np.linalg.norm(z);x=np.array([0.,0.,1]);x-=z*x.dot(z);x/=np.linalg.norm(x);y=np.cross(z,x)
  R=np.stack([x,y,z],1);m.vertices=np.asarray(m.vertices)@R.T+np.array([.003,.002,-.006]);return m
 if name=='stand':
  c=[171,134,83];base=lathe([(.0001,-.091),(.074,-.091),(.079,-.088),(.080,-.084),(.078,-.080),(.071,-.077),(.0001,-.077)])
  color(base,c)
  parts=[base,cylinder([.003,0,-.080],[-.005,0,.171],.009,c)]
  parts += [cylinder([-.082,-.003,.050],[.088,.007,.063],.0046,c),cylinder([-.009,-.080,.150],[-.005,.034,.132],.0046,c)]
  return trimesh.util.concatenate(parts)
 if name=='broom':
  white=[(-.164,.051),(-.157,.063),(-.128,.054),(-.091,.040),(-.061,.030),(-.036,.022),(-.012,.035),(.020,.061),(.030,.066),(.032,.037),(.017,-.020),(.003,-.075),(-.011,-.081),(-.035,-.054),(-.044,-.013),(-.061,.006),(-.101,.022),(-.140,.037),(-.164,.051)]
  handle=extrude(white,lambda q:.004+.030*q[0],.007,[229,226,214],holes=[((-.137,.047),(.014,.004),-.34)])
  br=[(.027,.062),(.110,.045),(.101,.016),(.082,-.046),(.067,-.094),(.041,-.101),(.003,-.080),(.010,-.037),(.027,.062)]
  green=extrude(br,lambda q:.006+q[0]*0,.006,[35,130,60])
  # Sub-mm parallel bundle ridges retain a smooth envelope but expose bristle direction.
  v=np.array(green.vertices);phase=2*np.pi*(v[:,1]-.18*v[:,0])/.0025
  v[:,2]+=.00045*np.cos(phase)*np.tanh((v[:,2]-.006)/.002);green.vertices=v
  shade=.85+.15*np.cos(phase);green.visual.vertex_colors=np.c_[np.array([45,155,70])[None]*shade[:,None],np.full(len(v),255)].astype(np.uint8)
  return trimesh.util.concatenate([handle,green])
 if name=='dustpan':
  step=.0015;x,y,z=np.mgrid[-.17:.12:step,-.14:.14:step,-.017:.043:step]
  # Back corners rounded; front lip flat and open, thin floor and rising side/back rims.
  qx=np.abs(x-.027)-.073;qy=np.abs(y)-.111
  d=np.sqrt(np.maximum(qx,0)**2+np.maximum(qy,0)**2)+np.minimum(np.maximum(qx,qy),0)-.010
  floor=np.maximum(d,np.abs(z+.005)-.0018)
  rim=np.maximum.reduce([d,-d-.0045,z-(.018*np.clip((.106-x)/.075,0,1)), -.006-z,x-.104])
  # smoothly tapered integral handle, including hanging hole
  handle=np.maximum.reduce([np.abs(y)-(.012+.010*np.exp(-((x+.055)/.025)**2)), -x-.160,x+.042,np.abs(z-(.014-.04*x))-.004])
  hole=((x+.143)/.010)**2+(y/.0045)**2-1
  handle=np.maximum(handle,-hole*.004)
  F=gaussian_filter(np.minimum(np.minimum(floor,rim),handle),.7)
  return field_mesh(F,[-.17,-.14,-.017],step,[39,43,42])
 if name=='debris':
  # Four observed stacked blocks represented separately; hidden interfaces inferred.
  parts=[]
  for i in range(4):
   m=trimesh.creation.box([.041,.045,.008]);m.apply_translation([-.0005,.0005,-.010+i*.0085]);color(m,[191+i*4,170+i*4,127+i*4]);parts.append(m)
  return trimesh.util.concatenate(parts)

if __name__=='__main__':
 out=ROOT/'assets/objects_v3'
 for task,names in [('sweep',['broom','debris','dustpan']),('hang',['mug','stand'])]:
  (out/task).mkdir(parents=True,exist_ok=True)
  for n in names:
   m=build(n);m.export(out/task/(n+'.ply'))
   src=ROOT/'assets/objects'/task;meta=json.load(open(src/(n+'.json')))
   meta.update(version='v3 smooth prior, iteration 01',shape='smooth parametric/outline model',unseen_geometry='Prior-completed: backs, wall thickness, hidden peg and handle sections. Not measured surfaces.',fit_source='v1 measured cloud dimensions + initial two-camera RGB; canonical v1 frame retained',watertight=bool(m.is_watertight),volume_m3=float(m.volume),extents_m=m.extents.tolist(),n_vertices=len(m.vertices),n_faces=len(m.faces))
   json.dump(meta,open(out/task/(n+'.json'),'w'),indent=2)
   shutil.copy2(src/(n+'_points.npz'),out/task/(n+'_points.npz'))
   print(task,n,np.round(m.extents*100,1),len(m.faces),m.is_watertight,flush=True)
