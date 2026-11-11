"""Current-observation whole-sheet translation; exactly preserves rest lengths
and self-separation along its continuous path. No trajectory lookup or future.
"""
import numpy as np
from scipy.spatial import cKDTree

def translate_sheet(X,views,table_plane=None,pin_ids=(),pin_targets=(),max_step=.010):
    clouds=[views[c]['P'][::max(1,len(views[c]['P'])//1000)] for c in ('top','side')]
    def scores(Y):
        tree=cKDTree(Y)
        return np.array([np.mean(np.minimum(tree.query(p)[0],.08)) for p in clouds])
    before=scores(X)
    if max(before)<.010:return X,dict(applied=False,before_mm=(before*1000).tolist())
    tree=cKDTree(X);deltas=[]
    for p in clouds:
        d,j=tree.query(p);ok=d<.12
        if ok.sum()<30:return X,dict(applied=False,reason='insufficient current support')
        deltas.append(np.median(p[ok]-X[j[ok]],axis=0))
    delta=np.mean(deltas,axis=0)
    ids=np.asarray(pin_ids,int);targets=np.asarray(pin_targets).reshape(-1,3)
    if len(ids):delta=.5*delta+.5*np.mean(targets-X[ids],axis=0)
    delta*=min(1,max_step/max(np.linalg.norm(delta),1e-12))
    if table_plane is not None:
        normal=np.r_[-table_plane[:2],1.];height=(X+delta)@normal-table_plane[2]
        delta+=max(0.,.0002-height.min())*normal/(normal@normal)
    Y=X+delta;after=scores(Y)
    okay=np.all(after<=before+.0003) and np.mean(after)<np.mean(before)-.0001
    if len(ids):okay &= np.mean(np.sum((Y[ids]-targets)**2,axis=1))<=np.mean(np.sum((X[ids]-targets)**2,axis=1))
    return (Y if okay else X),dict(applied=bool(okay),before_mm=(before*1000).tolist(),after_mm=(after*1000).tolist(),translation_mm=(delta*1000).tolist())
