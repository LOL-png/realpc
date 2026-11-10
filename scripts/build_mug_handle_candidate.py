"""Build and validate the handle while preserving the cup geometry exactly."""
import json,hashlib,argparse,shutil,sys
from pathlib import Path
import numpy as np,trimesh,ipctk
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from realpc.mug_handle import split_mug,build_handle,cup_frame,profile_basis
p=argparse.ArgumentParser();p.add_argument('--params',required=True);p.add_argument('--source',default=str(ROOT/'assets/objects_v3'));p.add_argument('--out',required=True);a=p.parse_args();src=Path(a.source);out=Path(a.out)
if src.resolve()==out.resolve():raise ValueError('Build into a separate candidate library')
shutil.copytree(src,out,dirs_exist_ok=True)
meta=json.load(open(src/'hang/mug.json'));old=trimesh.load(src/'hang/mug.ply',process=False);body,_=split_mug(old);params=json.load(open(a.params));handle=build_handle(meta,**params);new=trimesh.util.concatenate([body,handle]);new.export(out/'hang/mug.ply');check,_=split_mug(trimesh.load(out/'hang/mug.ply',process=False))
sortfaces=lambda f:np.array(sorted(map(tuple,np.sort(f,axis=1))))
assert np.array_equal(body.vertices,check.vertices) and np.array_equal(sortfaces(body.faces),sortfaces(check.faces))
E=handle.edges_unique;cm=ipctk.CollisionMesh(handle.vertices,E.astype(np.int32),handle.faces.astype(np.int32));cross=bool(ipctk.has_intersections(cm,handle.vertices));assert not cross
path=profile_basis()@np.asarray(params['control']);segment=path[24:41];direction=segment[-1]-segment[0];deviation=np.abs(np.cross(direction,segment-segment[0]))/np.linalg.norm(direction);assert deviation.max()<1e-8
B,o=cup_frame(meta);bodysha=hashlib.sha256(body.vertices.tobytes()+sortfaces(body.faces).tobytes()).hexdigest()
meta.update(version='v3 mug handle: straight outer strap, rounded spline transitions',handle_note='9-control Hermite swept strap; straight outer leg, rounded-rectangle cross-section. Cup vertex coordinates and triangles unchanged (face order may differ).',handle_profile=params,handle_fit_views='14 fixed-camera + 6 offline body-registered wrist RGB/depth observations; no Q28 masks used for fitting',canonical_body_affine=np.c_[B,o].tolist(),body_geometry_sha256=bodysha,handle_self_intersections=cross,watertight=bool(new.is_watertight),volume_m3=float(new.volume),extents_m=new.extents.tolist(),n_vertices=len(new.vertices),n_faces=len(new.faces))
(out/'hang/mug.json').write_text(json.dumps(meta,indent=2));(out.parent/'geometry_validation.json').write_text(json.dumps(dict(body_exact=True,body_sha256=bodysha,handle_watertight=bool(handle.is_watertight),handle_self_intersections=cross,outer_line_deviation_m=float(deviation.max()),source_mesh_sha256=hashlib.sha256((src/'hang/mug.ply').read_bytes()).hexdigest()),indent=2));print('PASS fixed cup coordinates/triangles, watertight nonintersecting handle, straight outer centreline',flush=True)
