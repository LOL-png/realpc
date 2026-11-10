"""Common 4K-sample, full-episode v1/v2 cloth audit. No fitting occurs here."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.cloth_causal import CausalClothTracker,at_image
from realpc.cloth_inputs import project

def score(S,v):
    result={};tree=cKDTree(S)
    for c in ('top','side'):
        z=v[c];d=tree.query(z['P'])[0];uv,depth=project(S,z['K'],z['T']);dd,ok=at_image(z['D'],uv)
        ok &= (dd>.08)&(depth<dd+.012);dz=np.abs(dd[ok]-depth[ok])
        result[c]={'observed_to_mesh_med_mm':float(np.median(d)*1000),'observed_to_mesh_p90_mm':float(np.percentile(d,90)*1000),'coverage_10mm':float(np.mean(d<.01)),'visible_depth_med_mm':float(np.median(dz)*1000) if len(dz) else None,'visible_depth_p90_mm':float(np.percentile(dz,90)*1000) if len(dz) else None,'visible_fraction':float(ok.mean())}
    P=np.concatenate([v[c]['P'] for c in ('top','side')]);d=cKDTree(P).query(S)[0]
    result['all_mesh']={'med_mm':float(np.median(d)*1000),'p90_mm':float(np.percentile(d,90)*1000),'support_10mm':float(np.mean(d<.01))};return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--mesh',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    old=np.load('/home/howardji/data_ext/realpc_out/v1/greycloth/episode_0000.npz')['cloth_points'];new=np.load(a.mesh);X=new['X'];F=new['F'];sf=new['sample_faces'][:4000];sb=new['sample_bary'][:4000];tr=CausalClothTracker();rows=[]
    for t in range(len(X)):
        f=dict(np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz'));v=tr.views(f);S=np.einsum('nk,nkd->nd',sb,X[t][F[sf]])
        rows.append(dict(t=t,v1=score(old[:,t],v),v2=score(S,v)))
        if t%100==0:print(t,flush=True)
    summary={}
    for name,lo,hi in [('pre_fold',0,150),('first_fold',150,270),('second_fold',270,453),('whole',0,453)]:
        part=rows[lo:hi]
        if not part:continue
        summary[name]={}
        for model in ('v1','v2'):
            summary[name][model]={g:{k:float(np.median([r[model][g][k] for r in part if r[model][g][k] is not None])) for k in part[0][model][g]} for g in part[0][model]}
    json.dump({'notes':['Same current masks/depth and 4000 samples for both models. Fit consistency, not material ground truth.','v1 contains old irregular-tick resampling; used exactly as the published v1 dataset.','All-mesh distance includes hidden material; visible-depth metrics exclude points behind observed front layer.','No observed-to-mesh distances are censored.'], 'summary':summary,'frames':rows},open(a.out,'w'),indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
