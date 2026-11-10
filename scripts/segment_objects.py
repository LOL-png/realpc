#!/usr/bin/env python
"""Object masks for the tracker: SAM2 video propagation, FORWARD ONLY from the start tick (each frame's mask uses only
that frame and earlier memory: causal), seeded at the start tick with our own renders of the library meshes at their
initial poses and of the two FK arms (the arms are extra SAM objects so a held object separates from its gripper).
No hand labels. Output per camera: labels uint8 [T,h,w] at 640 px width (0 none, 1..P objects, 254/255 arms).

  LD_PRELOAD=... python scripts/segment_objects.py --episode <ep> --out_dir ~/data_ext/realpc_v4/seg [--start S --stop E]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.calib_rig import RigCalib  # noqa: E402
from realpc.config import build_arms, load_config  # noqa: E402
from realpc.diffcalib import Renderer, TorchArm  # noqa: E402
from realpc.io import Episode, read_depth_frames  # noqa: E402
from realpc.v4 import StreamConverter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--cameras",nargs="+",default=["top","side"],choices=["top","side"])
ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=None)
ap.add_argument("--calib", default=os.path.join(ROOT, "configs", "calib_yam_rig_v3.json"))
ap.add_argument("--assets", default=os.path.join(ROOT, "assets", "objects"))
ap.add_argument("--sam_ckpt", default=os.path.join(ROOT, ".model_cache", "sam2", "sam2.1_hiera_small.pt"))
ap.add_argument("--cold", action="store_true", default=True)
ap.add_argument("--image-refine-seeds", action="store_true", help="infer initial silhouettes from RGB using mesh-derived boxes/points, instead of copying mesh masks")
a = ap.parse_args()
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

ep = Episode(a.episode); cal = RigCalib.load(a.calib)
cfg = load_config(os.path.join(ROOT, "configs", "yam_bimanual.json")); r = cfg["robot"]
tarm = TorchArm(os.path.join(cfg["_dir"], r["urdf"]), r["arm_joints"], r["finger_joints"])
lib = os.path.join(a.assets, ep.task)
if not os.path.isdir(lib):
    print("no library for", ep.task); sys.exit(0)
conv = StreamConverter(ep, cal, tarm, build_arms(cfg, ep.calib), lib, start=a.start, cold=a.cold)
T_total = len(ep.low["t_cap"]); stop = min(a.stop or T_total, T_total)
frames0 = {}
for c in ("top", "side"):
    D = read_depth_frames(os.path.join(ep.path, f"{c}_depth.mkv"), [a.start])[a.start].astype(np.float32) * float(ep.meta["depth_scale"][c])
    cap = cv2.VideoCapture(ep.video(c)); cap.set(cv2.CAP_PROP_POS_FRAMES, a.start); _, im = cap.read()
    frames0[c] = (D, im)
conv._init(a.start, frames0)
init = {o.name: conv.trk.state[o.name].T for o in conv.objs}
os.makedirs(os.path.join(a.out_dir, ep.task), exist_ok=True)
model = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", a.sam_ckpt, device="cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class Frames:
    """lazy SAM input frames (the video is decoded once into memory at 640 px width)."""

    def __init__(self, ims, size):
        self.ims, self.size = ims, size

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, i):
        x = cv2.resize(cv2.cvtColor(self.ims[i], cv2.COLOR_BGR2RGB), (self.size, self.size))
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
        return (x - MEAN) / STD


for c in a.cameras:
    t0 = time.time()
    cap = cv2.VideoCapture(ep.video(c)); cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    ims = []
    for _ in range(a.start, stop):
        ok, im = cap.read()
        if not ok:
            break
        H0, W0 = im.shape[:2]
        ims.append(cv2.resize(im, (640, int(round(H0 * 640 / W0)))))
    h, w = ims[0].shape[:2]
    from realpc.online_masks import render_seeds
    q={side:cal.q_at(ep,side,cal.robot_tick(ep,c,a.start)) for side in ['left','right']}
    seeds,zbufs,ids=render_seeds(conv,c,(h,w),q)
    labels = np.zeros((len(ims), h, w), np.uint8)
    from realpc.online_masks import ObjectMaskStream
    memory=ObjectMaskStream(model,ims[0],seeds,zbufs,ids,frames0[c][0],cold=a.cold,image_refine=a.image_refine_seeds,debug_dir=a.out_dir,reference_colors={i+1:np.load(os.path.join(lib,o.name+"_points.npz"))["C"] for i,o in enumerate(conv.objs)})
    for fi,im in enumerate(ims):labels[fi]=memory.step(im)
    model.reset_state(memory.state)
    np.savez_compressed(os.path.join(a.out_dir, ep.task, f"{ep.name}_{c}.npz"), labels=labels, start=a.start,
                        names=np.array([o.name for o in conv.objs]), seeds=np.array(sorted(seeds)), image_refined_seed=a.image_refine_seeds)
    print(f"{ep.task}/{ep.name} {c}: {len(ims)} frames, seeds {sorted(seeds)}, {time.time() - t0:.0f} s", flush=True)
