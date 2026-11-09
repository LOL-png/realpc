# realpc: time-consistent 4D point clouds from bimanual RGB-D recordings

## v4 (current, 11-09): calibrated robot + object mesh library + causal 6-DoF mesh pose tracking

One causal code path (`realpc/v4.py` `StreamConverter.step(t, frames)`) serves training-data conversion and live
inference; it can start at any tick (cold start: objects are registered into the first cloud it sees).

| part | how |
|---|---|
| calibration | `configs/calib_yam_rig_v3.json` (`realpc/calib_rig.py` RigCalib): rigid camera rig, one base correction per arm, joint zero offsets, constant camera lag (+3 ticks). Fitted by `scripts/calib_v3.py --rig --const_lag 3 --offsets` (track bundle adjustment through the FK + depth + pytorch3d soft silhouettes, `realpc/handeye_ba.py`, `realpc/diffcalib.py`) |
| robot | FK of the recorded joints (+offsets): 9600 fixed link-surface samples |
| objects | library meshes `assets/objects/<task>/<name>.ply` (`scripts/build_object_lib.py`, `scripts/refine_object_lib.py`): one 6-DoF pose per tick, 1500 fixed samples each |
| object masks | `scripts/segment_objects.py`: SAM2 video propagation forward only, seeded with renders of the library + FK arms at the first tick |
| pose tracking | `realpc/meshtrack.py`: held = FK gripper x grasp offset; carried objects attach to their carrier; mask-gated point-to-plane ICP on top+side depth; lost objects re-localised from their mask |
| static scene | table-region points of the first tick (robot, objects, cloth removed) |
| cloth | greycloth: cloth v1 tracks mapped into the v3 frame, valid before the fold only |

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6      # open3d with the conda env's old libstdc++
python scripts/segment_objects.py --episode <ep> --out_dir <seg>                 # causal object masks
python scripts/convert_v4.py --episodes <ep> ... --out_dir <out> --seg_dir <seg> [--start S --cold]
python scripts/verify_v4.py --converted <out>/<task>/<ep>.npz --episode <ep> --out v.mp4
python scripts/eval_alignment.py --converted <npz> --episode <ep> --masks <SAM2 dir> --out a.json   # acceptance
python scripts/make_manifest.py --out_dir <out>
python -c "from realpc.dataset import load; d = load('<npz>'); print(d['points'].shape)"   # [T, N, 3]
```

Requirements added for v4: pytorch3d 0.7.8 (prebuilt wheel), open3d 0.20, SAM2 (weights in `.model_cache/sam2/`),
fast_simplification, rtree, scikit-image.

---


`realpc` turns a recorded robot episode into tracked points: every point keeps its identity over time. The inputs are
RGB-D videos of the top and side cameras, the recorder's fused cloud, and joint angles.

The same causal code serves two uses:
- **Training data:** batch conversion of recorded episodes.
- **Inference:** a streaming call per frame (past and current data only).

It can start at any frame, with only a short past window, as a world model sees it.

## Pipeline (v3, causal)

| part | how | correspondence |
|---|---|---|
| robot | URDF FK of `q_left` / `q_right`: surface samples of every link mesh, fingers included (I2RT YAM, `robot_models/yam_v1`) | exact |
| robot hardware missing from the URDF (wrist cameras, cables) | segments hugging the FK surface are fixed to their nearest link | exact |
| table / static scene | table plane + table colour; colour/geometry segments that never move | exact (static) |
| rigid objects | see below | one SE(3) per object per step |
| cloth | see below (`scripts/cloth_mesh_track.py`) | fixed barycentric samples |

**Rigid objects.**
1. Depth: the top camera's depth is refined per frame with the colour image (`realpc/depth_refine.py`, from the
   user's harness).
2. Tracks: CoTracker3 online tracks on the top RGB, re-seeded every 32 steps with queries on objects.
3. Lifting: tracks are lifted to 3D with the refined depth. Tracks that fall onto the FK arm are rejected.
4. Pose: RANSAC (3-point sets, 5 mm) + Huber-weighted Kabsch of the object's tracks between the instance start and
   now (`realpc/se3.py`, the same math as the SE(3)-patch encoder).
5. Grasps: while a gripper holds the object, it follows `G(t) Y`, with `Y` taken at grasp time.
6. Free objects move only on contact (robot or a moving neighbour), after 3 consecutive agreeing steps.

**Cloth** (`scripts/cloth_mesh_track.py`).
1. Fused surface per frame: refined top depth and side depth, with robot and table pixels removed, cloth colour only,
   fused in a TSDF and meshed with marching cubes.
2. Template: the first frame's surface, decimated to about 1.5K vertices.
3. Tracking: ARAP + point-to-plane, plus fingertip pins.
4. Points: 4K fixed barycentric samples on the template, so identity persists.

**Camera sync.** The per-tick timestamps are identical for both cameras. Content analysis shows the side stream
drifting against the top stream within an episode, so `realpc/sync.py` maps top ticks to side frames by DTW of the
two motion-energy signals.

Outputs (`realpc.pipeline.Tracks`, `.npz`):
- `pos [N,T,3]` (NaN before birth and after death);
- `group [N]`: 0 gripper, 1 robot link, 2 static, 3 rigid object, 4 cloth;
- `body [N]`;
- `birth`;
- `held [N,T]`;
- `part_T [P,T,4,4]`: object poses;
- `obj_res [P,T,2]`: per-object residual vs the measured cloud;
- `times`, `ticks`.

**Offline refine** (`realpc.pipeline.offline_refine`, not causal) back-fills points first seen later along their
object's pose history and smooths poses over ±3 steps.

## Run

```bash
pip install -r requirements.txt       # + ffmpeg; CoTracker3 weights download once into .model_cache/ (gitignored)
python scripts/convert_v3.py --episode <task>/episode_0000 --out tracks.npz [--start S --stop E]
python scripts/cloth_mesh_track.py --episode greycloth/episode_0000 --out cloth.npz [--start S --stop E]
python scripts/verify_video.py --episode <dir> --tracks tracks.npz --out verify.mp4 --sync_cache .cache
python scripts/robot_overlay.py --episode <dir> --out overlay.png --json overlay.json      # FK mesh vs images
python scripts/sync_cameras.py --episode <dir>                                             # per-camera lag
PLWM_SE3_SRC=<dir with se3patch.py, se3latent.py> python scripts/evaluate.py --episode <dir> --tracks tracks.npz \
    --out metrics.json --cold_dir <cold-start windows> --task <task> [--fb_ckpt ...] [--rc_ckpt ...]
```

Older stages, kept for comparison:
- `scripts/convert.py`: object tracker with ICP on the fused cloud;
- `realpc/calib.py`, `scripts/handeye_tracks.py`: depth / track hand-eye;
- `scripts/calibrate_cameras.py`: silhouette-edge camera calibration. It was evaluated and not adopted; see
  `configs/calib_yam_rig_20261108.json`.

Streaming: `realpc.tracker.Tracker.step(frame)` is the per-frame call. `realpc.v3.V3Converter` drives it from
CoTracker instances; one predictor is kept per alive instance, and outputs arrive with the online model's 8-tick
latency.

## Known limits

- **Cloth.** No true correspondences. The v1 mesh tracker holds through simple lifts (2-3 mm to the fused surface) but
  loses the shirt during the fold. It has no 2D-track term yet, and cloth near the gripper is cut out with the robot.
- **Calibration.** The recorder calibration is used. FK silhouettes sit about 5-9 px from image edges in the side
  camera. The top camera is too close to the arms for depth, and its edge metric saturates.
- **Sync.** The side camera's content offset varies within an episode (up to about 1-2 s on greycloth). The DTW map
  is estimated from motion energy and is not validated frame by frame.
- **Segmentation** is colour/geometry region growing. Small dark segments near the arms can still move spuriously;
  objects smaller than 250 points move only while held.
- **Deformable switch.** `allow_deform` is a per-task switch, for scenes that contain cloth.

## Credits

Depth refinement (`realpc/depth_refine.py`, verbatim), the grasp-offset idea (X E(t) Y), hidden-surface memory and
±3-step fusion are from the user's reconstruction harness. Robot model: I2RT Robotics (MIT). Point tracking:
CoTracker3 (Meta, downloaded at runtime).
