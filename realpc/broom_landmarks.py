"""High-confidence hanging-hole detection from the current RGB image.
Canonical geometry comes from the precomputed mesh library, not runtime labels.
Returns no measurement under occlusion, tiny projection, or ambiguous contours.
"""
import cv2,numpy as np

def hole_landmarks(obj,T,views,robot_z):
    meta=getattr(obj,'metadata',{});M=np.asarray(meta.get('mesh_fit_transform',np.eye(4)));scale=np.asarray(meta.get('mesh_fit_scale',[1,1,1]))
    theta=np.linspace(0,2*np.pi,48,endpoint=False);a=-.34;R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);xy=np.c_[.014*np.cos(theta),.004*np.sin(theta)]@R.T+[-.137,.047]
    raw=np.c_[xy,.004+.030*xy[:,0]+.007];ring=(raw*scale)@M[:3,:3].T+M[:3,3];local=ring.mean(0);W=ring@T[:3,:3].T+T[:3,3];result=[];records=[]
    for v,rz in zip(views,robot_z):
        if v.rgb is None:continue
        C=W@v.Tcw[:3,:3].T+v.Tcw[:3,3]
        if C[:,2].min()<.08:continue
        uv=C[:,:2]/C[:,2:]*[v.K[0,0],v.K[1,1]]+[v.K[0,2],v.K[1,2]];center=uv.mean(0);H,Wim=v.D.shape
        if not(15<center[0]<Wim-15 and 15<center[1]<H-15):continue
        rx,ry=(center*v.s).astype(int)
        if rz[ry,rx]<C[:,2].mean()-.01:continue
        ellipse=cv2.fitEllipse(uv.astype(np.float32));expected=np.sort(ellipse[1]);area=np.prod(expected)
        if expected[0]<2.5 or expected[1]<7:continue
        rad=int(max(20,min(100,expected[1]*2.5)));x0=max(0,int(center[0])-rad);y0=max(0,int(center[1])-rad);roi=v.rgb[y0:min(H,y0+2*rad),x0:min(Wim,x0+2*rad)];gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY);edges=cv2.Canny(gray,15,45);contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE);candidates=[]
        for contour in contours:
            if len(contour)<12:continue
            e=cv2.fitEllipse(contour);size=np.sort(e[1]);ratio=size/expected;target=np.array(e[0])+[x0,y0];err=np.linalg.norm(target-center)
            if np.any(ratio<.65) or np.any(ratio>1.55) or err>min(50,expected[1]*1.5):continue
            mask=np.zeros(gray.shape,np.uint8);cv2.ellipse(mask,e,1,-1);inner=cv2.erode(mask,np.ones((3,3),np.uint8)).astype(bool);outer=cv2.dilate(mask,np.ones((5,5),np.uint8)).astype(bool)&~mask.astype(bool)
            if inner.sum()<8 or outer.sum()<15:continue
            contrast=float(np.mean(gray[outer])-np.mean(gray[inner]))
            if contrast<12 or np.mean(gray[outer])<100:continue
            iy,ix=np.nonzero(outer);iy+=y0;ix+=x0;measured=v.D[iy,ix];rz_sample=rz[np.clip((iy*v.s).astype(int),0,rz.shape[0]-1),np.clip((ix*v.s).astype(int),0,rz.shape[1]-1)]
            if ((rz_sample<1e8)&((measured<.1)|(rz_sample<measured+.008))).mean()>.35:continue
            dz=measured[measured>.1]
            if len(dz)>10 and abs(np.median(dz)-C[:,2].mean())>.06:continue
            candidates.append((err/25+np.linalg.norm(np.log(ratio)),target,contrast))
        if not candidates:continue
        candidates.sort(key=lambda x:x[0]);best=candidates[0]
        if any(np.linalg.norm(x[1]-best[1])>5 and x[0]<best[0]+.3 for x in candidates[1:]):continue
        result.append((local,best[1],v,6. if "wrist" in v.name else 3.));records.append({'camera':v.name,'uv':best[1].tolist(),'prediction_error_px':float(np.linalg.norm(best[1]-center)),'contrast':best[2]})
    return result,records
