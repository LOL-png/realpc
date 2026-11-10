"""Offline-buildable swept strap handle; runtime consumes the resulting fixed mesh."""
import json
import numpy as np,trimesh
from scipy.interpolate import CubicHermiteSpline, CubicSpline

def cup_frame(meta):
    # Earlier metadata retained only the LAST global fit. Recover the complete
    # affine mapping from the original ordered lathe vertices to the frozen cup.
    from pathlib import Path
    from scripts.build_smooth_v3 import lathe
    path=Path(__file__).resolve().parents[1]/'assets/objects_v3/hang/mug.ply'
    body,_=split_mug(trimesh.load(path,process=False))
    profile=[(.0001,-.037),(.034,-.037),(.039,-.034),(.0425,.033),(.043,.037),(.042,.039),(.0405,.039),(.040,.037),(.038,-.030),(.034,-.033),(.0001,-.033)]
    source=lathe([(r*.96,z*1.25) for r,z in profile]).vertices
    assert len(source)==len(body.vertices)
    Q=np.c_[source,np.ones(len(source))];A=np.linalg.lstsq(Q,body.vertices,rcond=None)[0]
    if np.linalg.norm(Q@A-body.vertices,axis=1).max()>1e-6:raise ValueError('Cup no longer matches the original affine lathe; do not refit body silently')
    return A[:3].T,A[3]

def split_mug(mesh):
    parts=mesh.split(only_watertight=False);assert len(parts)==2
    return tuple(sorted(parts,key=lambda m:-m.area))

DEFAULT=np.array([[.040,.036],[.050,.040],[.064,.034],[.070,.022],[.070,.004],[.068,-.013],[.059,-.025],[.047,-.029],[.037,-.030]])

def profile_basis(samples=65,profile="straight"):
    if profile=="ear":
        return CubicSpline(np.linspace(0,1,9),np.eye(9),axis=0,bc_type="natural")(np.linspace(0,1,samples))
    if profile!="straight":raise ValueError(profile)
    u=np.linspace(0,1,9);eye=np.eye(9);derivative=np.gradient(eye,u,axis=0)
    # The outer leg is a straight segment, with C1 rounded transitions.
    derivative[3:6]=(eye[5]-eye[3])/(u[5]-u[3])
    return CubicHermiteSpline(u,eye,derivative,axis=0)(np.linspace(0,1,samples))

def sweep_arrays(control=DEFAULT,angle=-.13,half_thickness=.0035,half_width=.0065,samples=65,section=24,profile="straight",taper_ratio=1.):
    # General interpolating spline, including a long, almost straight outer leg.
    basis=profile_basis(samples,profile)
    path=basis@np.asarray(control);tangent=np.gradient(path,axis=0);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True);normal=np.c_[-tangent[:,1],tangent[:,0]]
    # Superellipse = rounded rectangle, not a circular/elliptical tube.
    th=np.arange(section)*2*np.pi/section;cs=np.sign(np.cos(th))*abs(np.cos(th))**.5;sn=np.sign(np.sin(th))*abs(np.sin(th))**.5
    taper=np.minimum(1,np.minimum(np.arange(samples)+1,np.arange(samples,0,-1))/3)
    taper*=1-(1-taper_ratio)*np.linspace(0,1,samples)**1.2
    xz=path[:,None,:]+normal[:,None,:]*cs[None,:,None]*half_thickness*taper[:,None,None]
    V=np.stack([xz[:,:,0],np.broadcast_to(half_width*sn,(samples,section))*taper[:,None],xz[:,:,1]],-1).reshape(-1,3)
    c,s=np.cos(angle),np.sin(angle);V=V@np.array([[c,s,0],[-s,c,0],[0,0,1]])
    F=[]
    for j in range(samples-1):
        for i in range(section):
            a=j*section+i;b=j*section+(i+1)%section;F.extend([[a,b,a+section],[b,b+section,a+section]])
    for i in range(1,section-1):F.extend([[0,i+1,i],[(samples-1)*section,(samples-1)*section+i,(samples-1)*section+i+1]])
    return V,np.array(F)

def build_handle(meta,control=DEFAULT,angle=-.13,half_thickness=.0035,half_width=.0065,profile="straight",taper_ratio=1.):
    # Union the swept volume implicitly: explicit normal-offset rings overlap
    # at tight inner corners even when their silhouette looks reasonable.
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    path=profile_basis(2049,profile)@np.asarray(control);step=.00055
    lo=np.array([path[:,0].min()-half_thickness-.001,-half_width-.001,path[:,1].min()-half_thickness-.001])
    hi=np.array([path[:,0].max()+half_thickness+.001,half_width+.001,path[:,1].max()+half_thickness+.001])
    axes=[np.arange(a,b+step,step) for a,b in zip(lo,hi)]
    x,z=np.meshgrid(axes[0],axes[2],indexing='ij');dist,nearest=cKDTree(path).query(np.c_[x.ravel(),z.ravel()]);dist=dist.reshape(x.shape)
    taper=(1-(1-taper_ratio)*(nearest/(len(path)-1))**1.2).reshape(x.shape)
    field=(dist[:,None,:]/(half_thickness*taper[:,None,:]))**4+(np.abs(axes[1])[None,:,None]/(half_width*taper[:,None,:]))**4-1
    # Bury the ends in the existing wall, without protruding into the cup lumen.
    radius=.03744+(.0408-.03744)*(axes[2]+.0425)/.08375
    radial=np.sqrt(axes[0][:,None,None]**2+axes[1][None,:,None]**2)
    field=np.maximum(field,(radius[None,None,:]-.0012-radial)/half_thickness)
    V,F,_,_=marching_cubes(field,0,spacing=(step,)*3,allow_degenerate=False);V+=lo
    c,s=np.cos(angle),np.sin(angle);V=V@np.array([[c,s,0],[-s,c,0],[0,0,1]])
    B,o=cup_frame(meta);m=trimesh.Trimesh(V@B.T+o,F,process=True);m.fix_normals()
    if len(m.faces)>5000:m=m.simplify_quadric_decimation(face_count=5000)
    m=max(m.split(only_watertight=False),key=lambda part:part.area)
    m.visual.vertex_colors=np.tile([205,83,19,255],(len(m.vertices),1));return m
