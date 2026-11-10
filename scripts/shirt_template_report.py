"""Render clean topology and report the first-frame two-camera fit."""
import argparse,json,sys
from pathlib import Path
import numpy as np,trimesh
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--out',required=True);a=p.parse_args();fit=Path(a.fit);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
m=trimesh.load(fit/'clean_template.ply',process=False);V=np.array(m.vertices);F=np.array(m.faces);info=json.load(open(fit/'template_info.json'));n=info['panel_vertices']
# Show the cut pattern in garment coordinates, with a readable orientation.
angle=info['fit_parameters'][0];R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
U=(V[:,:2]-V[:,:2].mean(0))@R
fig,axes=plt.subplots(1,2,figsize=(10,6));from matplotlib.collections import PolyCollection
for j,ax in enumerate(axes):
 faces=F[:len(F)//2] if j==0 else F[len(F)//2:]
 ax.add_collection(PolyCollection(U[faces],facecolors=['#76bde7' if j==0 else '#eea36b'],edgecolors=(.1,.1,.1,.25),linewidths=.25))
 ax.autoscale();ax.set_aspect('equal');ax.set_title('Front panel' if j==0 else 'Back panel');ax.set_xlabel('garment width (m)');ax.set_ylabel('garment length (m)');ax.grid(alpha=.2)
fig.suptitle('Clean two-layer T-shirt: sewn shoulders/sides; open neck, cuffs and hem')
fig.tight_layout();fig.savefig(out/'clean_template.png',dpi=150);plt.close(fig)
u,c=np.unique(m.edges_sorted,axis=0,return_counts=True);edges=u[c==1];from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
idx=np.unique(edges);lookup=np.full(len(V),-1,int);lookup[idx]=np.arange(len(idx));e=lookup[edges];A=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(idx),len(idx)))
nb,_=connected_components(A,directed=False)
report={'vertices':len(V),'faces':len(F),'nonmanifold_edges':int((c>2).sum()),'surface_components':len(m.split(only_watertight=False)),'boundary_loops':int(nb),'extent_m':np.ptp(V,axis=0).tolist(),'flat_rest_metric':info.get('flat_rest',False),'note':'Four open boundaries: neck, two cuffs, hem. Panels shown separately for inspection; the actual mesh shares sewn boundary vertices. Size is inferred from t0, not independently measured.'}
json.dump(report,open(out/'topology.json','w'),indent=2);print(report)

import shutil
if (fit/'overlay_0000.jpg').exists():shutil.copy2(fit/'overlay_0000.jpg',out/'first_frame_fit.jpg')
