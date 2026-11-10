"""Current depth associated to predicted meshes; no learned per-frame segmentation."""
import cv2
import numpy as np


def associate_depth(tracker, views, robot_z, object_z):
    names=list(tracker.objs)
    diagnostics={n:{'visible':0,'free':0,'support':0} for n in names}
    for vi,v in enumerate(views):
        # SAM masks are accepted only on explicit initialization/recovery frames.
        if v.labels is not None:continue
        h,w=v.R.H,v.R.W
        D=cv2.resize(v.D,(w,h),interpolation=cv2.INTER_NEAREST)
        K=v.K.copy();K[:2]*=v.s
        yy,xx=np.mgrid[:h,:w]
        C=np.stack([(xx-K[0,2])*D/K[0,0],(yy-K[1,2])*D/K[1,1],D],-1)
        P=C@v.Twc[:3,:3].T+v.Twc[:3,3]
        valid=(D>.1)&np.isfinite(D)
        if tracker.plane is not None:
            n,d=tracker.plane;valid &= (P@np.asarray(n)+d)>.002
        rz=robot_z[vi]
        robot=(rz<1e8)&((D<.1)|(rz<=D+.012))
        valid &= ~robot
        rgb=cv2.cvtColor(cv2.resize(v.rgb,(w,h)),cv2.COLOR_BGR2RGB).reshape(-1,3)/255. if v.rgb is not None else None
        best=np.full((h,w),np.inf);labels=np.zeros((h,w),np.uint8)
        for oi,name in enumerate(names,1):
            st=tracker.state[name]
            if not st.valid:continue
            z=object_z[name][vi];pred=z>0
            other=np.full((h,w),np.inf)
            for n,zs in object_z.items():
                if n!=name:other=np.minimum(other,np.where(zs[vi]>0,zs[vi],np.inf))
            visible=pred&~robot&(D>.1)&(other>=z-.008)
            # Missing depth and observed foreground occlusion are not failures.
            observable=visible&(D>=z-.015)
            diagnostics[name]['visible']+=int(observable.sum())
            diagnostics[name]['free']+=int((observable&(D>z+.025)).sum())
            region=cv2.dilate(pred.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)
            idx=np.flatnonzero(region&valid)
            if not len(idx):continue
            Q=P.reshape(-1,3)[idx];L=(Q-st.T[:3,3])@st.T[:3,:3]
            dist,_=tracker.objs[name].tree.query(L)
            # Permit small prediction errors; a distant surface never attracts ICP.
            usable=dist<.025
            color_cost=np.zeros(len(idx))
            if rgb is not None and tracker.lib_points is not None:
                from .objlib import colour_model
                if not hasattr(tracker,"depth_color"):tracker.depth_color={}
                if name not in tracker.depth_color:
                    tracker.depth_color[name]=colour_model(tracker.lib_points[name][1],tracker.bg_colors)
                # Illumination / newly exposed faces can score zero despite valid
                # geometry. Colour breaks ambiguous surface ownership; it must
                # not erase depth support for a shadowed, moving object.
                color_cost=.003*(1-tracker.depth_color[name](rgb[idx]))
            if not usable.any():continue
            idx=idx[usable];score=(dist+color_cost)[usable]
            win=score<best.ravel()[idx]
            best.ravel()[idx[win]]=score[win];labels.ravel()[idx[win]]=oi
        labels[robot]=254
        # Unknown depth is not background evidence for contour fitting.
        labels[~((D>.1)&np.isfinite(D))]=254
        v.labels=labels
        v.depth_generated=True
        # Keep the exact returns that passed robot/table/surface rejection.
        # Upsampling the labels onto other depth pixels can reintroduce foreground
        # robot edges and table returns which were never tested above.
        v.associated_points={oi:P[labels==oi] for oi in range(1,len(names)+1)}
        for oi,name in enumerate(names,1):diagnostics[name]['support']+=int((labels==oi).sum())
    for d in diagnostics.values():d['free_fraction']=d['free']/max(1,d['visible'])
    tracker.depth_diagnostics=diagnostics
