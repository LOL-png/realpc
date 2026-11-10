"""Render a saved causal trajectory; visualization never feeds the tracker."""
import argparse,json,sys,subprocess
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.diffcalib import Renderer
from realpc.cloth_inputs import project
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes

def render(frame,c,X,F,layers=2):
    im=frame[c+'_rgb'].copy();h,w=im.shape[:2];K,T=frame[c+'_K'],frame[c+'_T'];D=frame[c+'_depth']
    Q=(X-T[:3,3])@T[:3,:3];R=Renderer(K,h,w)
    with torch.inference_mode():
        v=torch.as_tensor(Q,dtype=torch.float32,device='cuda');f=torch.as_tensor(F,dtype=torch.int64,device='cuda')
        mesh=Meshes(verts=[R._ndc(v[None])[0]],faces=[f])
        pix,z,bary,_=rasterize_meshes(mesh,image_size=(h,w),faces_per_pixel=1,cull_backfaces=False,perspective_correct=True,max_faces_per_bin=100000)
        ids=pix[0,:,:,0].cpu().numpy();depth=z[0,:,:,0].cpu().numpy();bc=bary[0,:,:,0].cpu().numpy()
    # Occlude only by measured robot pixels. Do not hide a misplaced garment
    # merely because it lies behind observed cloth (that is a fit error).
    uv,zz=project(frame['robot'],K,T);xy=np.rint(uv).astype(int);good=(zz>.08)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    robot=np.zeros((h,w),np.uint8);robot[xy[good,1],xy[good,0]]=1;robot=cv2.dilate(robot,np.ones((5,5),np.uint8))
    visible=(ids>=0)&~((robot>0)&(D>.08)&(depth>D-.005))
    tri=Q[F];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1),1e-10)[:,None]
    shade=.65+.35*np.abs(normal[:,2]);front=(normal[:,2]<0) if layers==1 else (np.arange(len(F))<len(F)//2);colors=np.where(front[:,None],np.array([230.,170.,30.]),np.array([40.,145.,245.]))*shade[:,None]
    tint=colors[np.maximum(ids,0)];im[visible]=(.55*im[visible]+.45*tint[visible]).astype(np.uint8)
    edges=visible&(bc.min(2)<.025);im[edges]=(.6*im[edges]+.4*np.array([35,55,65])).astype(np.uint8)
    return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',required=True);p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--fps',type=float,default=15);a=p.parse_args()
    fit=Path(a.fit);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);z=np.load(fit/'mesh.npz');XX,F=z['X'],z['F']
    layers=json.load(open(fit/'template_info.json')).get('layers',2) if (fit/'template_info.json').exists() else 2
    video=out/'both_cameras.mp4';ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x840','-r',str(a.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    picks=np.unique(np.minimum([0,150,175,210,240,300,350,390,452],len(XX)-1));sheets=[]
    for t,X in enumerate(XX):
        with np.load(Path(a.inputs)/'frames'/f'{t:04d}.npz') as f:frame={k:f[k] for k in f.files}
        rows=[]
        for c in ('top','side'):
            im=frame[c+'_rgb'];ov=render(frame,c,X,F,layers=layers);cv2.rectangle(ov,(0,0),(640,29),(35,35,35),-1)
            cv2.putText(ov,f'{c} | frame {t} | '+('single sheet: two surface sides' if layers==1 else 'blue: initial front, orange: back'),(8,20),0,.44,(255,255,255),1,cv2.LINE_AA)
            rows.append(np.hstack([im,ov]))
        canvas=np.ascontiguousarray(np.vstack(rows));ff.stdin.write(canvas.tobytes())
        if t in picks:
            cv2.imwrite(str(out/f'frame_{t:04d}.jpg'),canvas);sheets.append(cv2.resize(canvas,(640,420)))
        if t%100==0:print('render',t,flush=True)
    ff.stdin.close();error=ff.stderr.read().decode();code=ff.wait()
    if code:raise RuntimeError(error)
    while len(sheets)%3:sheets.append(np.zeros_like(sheets[0]))
    cv2.imwrite(str(out/'contact_sheet.jpg'),np.vstack([np.hstack(sheets[i:i+3]) for i in range(0,len(sheets),3)]))
    print(video)

if __name__=='__main__':main()
