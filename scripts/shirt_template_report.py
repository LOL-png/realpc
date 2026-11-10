"""Render clean topology and report the first-frame two-camera fit."""
import argparse,json,sys
from pathlib import Path
import numpy as np,trimesh
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--out',required=True);a=p.parse_args();fit=Path(a.fit);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
m=trimesh.load(fit/'clean_template.ply',process=False);V=np.array(m.vertices);F=np.array(m.faces);n=json.load(open(fit/'template_info.json'))['panel_vertices']
fig=plt.figure(figsize=(12,5))
for j,explode in enumerate([False,True]):
 ax=fig.add_subplot(1,2,j+1,projection='3d');X=V.copy()
 if explode:X[:n,2]+=.045;X[n:,2]-=.045
 color=np.where((F<n).all(1)[:,None],np.array([.35,.65,.9,1]),np.array([.95,.55,.35,1]))
 collection=Poly3DCollection(X[F],facecolors=color,edgecolors=(.1,.1,.1,.25),linewidths=.2);ax.add_collection3d(collection)
 mid=V.mean(0);r=np.ptp(V[:,:2],axis=0).max()/2+.03;ax.set_xlim(mid[0]-r,mid[0]+r);ax.set_ylim(mid[1]-r,mid[1]+r);ax.set_zlim(mid[2]-.12,mid[2]+.12);ax.set_box_aspect((1,1,.4));ax.view_init(65,0)
 ax.set_title('Clean sewn template' if not explode else 'Front / back separated for topology inspection');ax.set_xlabel('world x (m)');ax.set_ylabel('world y (m)')
fig.tight_layout();fig.savefig(out/'clean_template.png',dpi=140);plt.close(fig)
u,c=np.unique(m.edges_sorted,axis=0,return_counts=True);edges=u[c==1];from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
idx=np.unique(edges);lookup=np.full(len(V),-1,int);lookup[idx]=np.arange(len(idx));e=lookup[edges];A=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(idx),len(idx)))
nb,_=connected_components(A,directed=False)
report={'vertices':len(V),'faces':len(F),'nonmanifold_edges':int((c>2).sum()),'surface_components':len(m.split(only_watertight=False)),'boundary_loops':int(nb),'extent_m':np.ptp(V,axis=0).tolist(),'note':'Four open boundaries expected: neck, two cuffs, hem. Exploded view is illustrative only.'}
json.dump(report,open(out/'topology.json','w'),indent=2);print(report)
