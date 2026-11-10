"""Clean one-sheet or sewn two-panel shirt, fitted from the current packet only.
No scan triangles, future poses, or per-episode fold animation are used.
"""
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import least_squares
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter


def outline():
    # Clockwise: hem, side seam, underarm, cuff, shoulder, neck, mirrored.
    return np.array([[-.38,-.48],[.38,-.48],[.37,.00],[.49,-.06],[.62,.16],
                     [.34,.38],[.13,.45],[.10,.37],[0,.34],[-.10,.37],[-.13,.45],
                     [-.34,.38],[-.62,.16],[-.49,-.06],[-.37,.00]])


def make_panels(poly,cell=.012,thickness=.002,layers=2):
    if layers not in (1,2):raise ValueError('shirt layers must be 1 or 2')
    boundary=[];seams=[]
    # Neck/cuffs/hem are garment openings; all other boundary spans sewn.
    openings={0,3,6,7,8,9,12}
    for i,(a,b) in enumerate(zip(poly,np.roll(poly,-1,axis=0))):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/cell)))
        for j in range(n):boundary.append(a+(b-a)*j/n);seams.append(i not in openings)
    boundary=np.array(boundary);lo=poly.min(0);hi=poly.max(0)
    xx,yy=np.meshgrid(np.arange(lo[0]+cell/2,hi[0],cell),np.arange(lo[1]+cell/2,hi[1],cell))
    grid=np.c_[xx.ravel(),yy.ravel()];grid=grid[Path(poly).contains_points(grid)]
    grid=grid[cKDTree(boundary).query(grid)[0]>cell*.3]
    import triangle
    segments=np.c_[np.arange(len(boundary)),np.roll(np.arange(len(boundary)),-1)]
    result=triangle.triangulate({'vertices':np.vstack([boundary,grid]),'segments':segments},f'pq25a{cell*cell*.45}Y')
    uv=result['vertices'];F=result['triangles']
    cross=np.cross(uv[F[:,1]]-uv[F[:,0]],uv[F[:,2]]-uv[F[:,0]])
    F[cross<0]=F[cross<0,::-1]
    if layers==1:
        n=len(uv)
        return np.c_[uv,np.zeros(n)],F,n,np.arange(n)
    n=len(uv);V=np.vstack([np.c_[uv,np.full(n,thickness/2)],np.c_[uv,np.full(n,-thickness/2)]])
    # Sew by sharing material vertices, not 2-mm side-wall springs. Tiny wall
    # edges impose artificial thickness-strain constraints on a textile seam.
    sewn=set()
    for i,yes in enumerate(seams):
        if yes:sewn.update([i,(i+1)%len(boundary)])
    sewn=np.array(sorted(sewn),int);keep=np.ones(2*n,bool);keep[n+sewn]=False
    remap=np.cumsum(keep)-1;remap[n+sewn]=remap[sewn]
    V[sewn,2]=0
    faces=remap[np.vstack([F,F[:,::-1]+n])];V=V[keep]
    pair=np.arange(len(V))
    for i in range(n):pair[remap[i]]=remap[i+n];pair[remap[i+n]]=remap[i]
    return V,faces,n,pair



def fit_current(views,cell=.012,flat_rest=False,table_plane=None,back_on_table=False,layers=2,flexible=False):
    """Optimize a low-dimensional flat pattern, then current-frame drape heights.
    Two cameras contribute geometry; top footprint rejects distant side clutter.
    """
    top=views['top']['P'];side=views['side']['P'];tree=cKDTree(top[:,:2]);side=side[tree.query(side[:,:2])[0]<.025]
    P=np.vstack([top,side]);xy=top[:,:2];center=np.median(xy,0)
    # Occupancy outline in metric XY; robust against sparse depth holes.
    cellmask=.005;origin=xy.min(0)-.02;idx=np.round((xy-origin)/cellmask).astype(int)
    import cv2
    mask=np.zeros(tuple(idx.max(0)+5),np.uint8);mask[idx[:,0],idx[:,1]]=1
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=max(contours,key=cv2.contourArea)[:,0,::-1]*cellmask+origin
    target=cKDTree(contour);base=outline()
    def transform(p,uv):
        angle,sx,sy,cx,cy=p;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        return uv*np.array([sx,sy])@R.T+[cx,cy]
    line=np.vstack([a+(b-a)*np.linspace(0,1,20)[:,None] for a,b in zip(base,np.roll(base,-1,axis=0))])
    def fun(p):
        S=transform(p,line);return np.r_[target.query(S)[0],cKDTree(S).query(contour[::3])[0]]
    candidates=[]
    for angle in [np.pi/2,-np.pi/2,0,np.pi]:
        p0=[angle,.46,.34,*center]
        fit=least_squares(fun,p0,bounds=([angle-.7,.30,.24,center[0]-.15,center[1]-.15],[angle+.7,.65,.65,center[0]+.15,center[1]+.15]),max_nfev=90,loss='soft_l1',f_scale=.01)
        candidates.append(fit)
    fit=min(candidates,key=lambda f:np.mean(fun(f.x)**2));poly=transform(fit.x,base)
    original=poly.copy()
    def outline_res(q):
        q=q.reshape(-1,2);samples=np.vstack([a+(b-a)*np.linspace(0,1,12)[:,None] for a,b in zip(q,np.roll(q,-1,axis=0))])
        # Preserve garment structure while allowing asymmetric initial placement.
        return np.r_[target.query(samples)[0],cKDTree(samples).query(contour[::3])[0],(.12 if flexible else .6)*(q-original).ravel(),(.08 if flexible else .3)*((q-np.roll(q,1,axis=0))-(original-np.roll(original,1,axis=0))).ravel()]
    bounds=np.full_like(poly,.10 if flexible else .03);bounds[6:11]=.04 if flexible else .015
    q=least_squares(outline_res,poly.ravel(),bounds=((poly-bounds).ravel(),(poly+bounds).ravel()),max_nfev=200 if flexible else 35,loss='soft_l1',f_scale=.01)
    poly=q.x.reshape(-1,2)
    if flat_rest:
        # Estimate scale from the complete sleeve span and garment length, not
        # from the compressed torso silhouette. Keep a regular flat cut pattern.
        from scipy.interpolate import RBFInterpolator
        angle=fit.x[0];R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        local=(xy-center)@R;extent=np.percentile(local,99,axis=0)-np.percentile(local,1,axis=0)
        sx,sy=extent/np.ptp(base,axis=0)
        rest_poly=base*np.array([sx,sy])@R.T+center
        V,F,n,pair=make_panels(rest_poly,cell,layers=layers)
        warp=RBFInterpolator(rest_poly,poly,kernel='thin_plate_spline',smoothing=1e-7)
        fit_xy=warp(V[:n,:2])
    else:
        V,F,n,pair=make_panels(poly,cell,layers=layers);fit_xy=V[:n,:2]
    outline_alpha=1.
    if flexible:
        # A wide outline fit can invert small triangles under a TPS warp.
        # Start from a valid planar map; subsequent 3D IPC fitting may fold it.
        front=F[:len(F)//2] if layers==2 else F
        source=V[:n,:2].copy()
        a=source[front];orientation=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
        while outline_alpha>1e-4:
            candidate=source+outline_alpha*(fit_xy-source);a=candidate[front]
            if np.all(np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])*orientation>orientation**2*.02):break
            outline_alpha*=.8
        fit_xy=source+outline_alpha*(fit_xy-source)
    # Smooth, bounded height regression; retain the clean pattern as rest metric.
    distances,js=cKDTree(P[:,:2]).query(fit_xy,k=16);w=np.exp(-(distances/.025)**2)
    heights=(w*P[js,2]).sum(1)/np.maximum(w.sum(1),1e-12)
    floor=float(np.percentile(top[:,2],2)-.003)
    support=floor if table_plane is None else np.c_[fit_xy,np.ones(len(fit_xy))]@table_plane
    heights=np.maximum(heights,support+.003)
    X=V.copy();X[:n,:2]=fit_xy;X[:n,2]=heights
    if layers==2:
        for i in range(n):
            if pair[i]==i:X[i,2]=heights[i]-.001
            else:X[pair[i]]=np.r_[fit_xy[i],heights[i]-.002]
    if back_on_table and layers==2:
        # The hidden panel is not another observed copy of the front wrinkles.
        # Start it supported by the table, smoothly meeting shared side seams.
        sewn=np.flatnonzero(pair[:n]==np.arange(n))
        distance,nearest=cKDTree(fit_xy[sewn]).query(fit_xy)
        support=np.broadcast_to(support,(n,))
        boundary_lift=np.maximum(X[sewn[nearest],2]-support[sewn[nearest]]-.001,0.)
        back_height=np.minimum(support+.001+boundary_lift*np.exp(-distance/.02),heights-.002)
        free=np.flatnonzero(pair[:n]!=np.arange(n))
        X[pair[free],2]=back_height[free]
    V[:,2]+=floor+.002
    return V,F,X,{'outline_warp_alpha':outline_alpha,'flexible_outline':flexible,'layers':layers,'flat_rest':flat_rest,'back_on_table':back_on_table and layers==2,'fit_parameters':fit.x.tolist(),'outline_rms_mm':float(np.sqrt(np.mean(fun(fit.x)**2))*1000),'panel_vertices':n,'panel_pair':pair.tolist(),'fit_outline':poly.tolist(),'initial_landmark_ids':cKDTree(X[:n,:2]).query(poly[[0,1,3,4,6,10,12,13]])[1].tolist(),'seams':'shared vertices' if layers==2 else 'none: one sheet','floor_z':floor,'current_frame_only':True,'openings':['neck','left cuff','right cuff','hem'] if layers==2 else ['continuous shirt-outline boundary']}
