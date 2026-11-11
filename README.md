# realpc: causal scene extraction

The current entry point is `ScenePipeline.step(t, frames, q)` in `realpc/scene_pipeline.py`. Live inference and training-data conversion call this same function. It produces fixed-identity robot, rigid-object, static-scene and cloth points from current RGB-D and calibrated joints.

The shared policy is **SAM every processed frame for every object**. This gave the highest equal-object mean silhouette overlap in full-episode comparisons of intervals 1, 2, 3 and 10. It does not win every object or depth metric. See [validation and limitations](docs/VALIDATION.md) for the complete accuracy/speed comparison and unresolved cloth deformation errors. The dataset conversion is staged after the release checks and videos; earlier partial outputs are preserved.

## What runs

1. One scene segmentation stream refreshes all objects using one shared SAM2 model and one forward memory per camera. The selected schedule refreshes every frame. With a larger configured interval, current depth, mesh predictions and optical flow carry masks between refreshes; skipped frames never enter SAM.
2. Rigid objects use depth/mesh fitting and gripper FK when held. The cloth uses a clean shirt template, causal point tracks and a deformable mesh. Robot geometry comes from FK.
3. Fixed mesh/link samples produce the point cloud. The recording adapter saves the same output that a live consumer receives.

`configs/scene_pipeline_v2.json` is the shared config for the SAM interval, recovery thresholds and tracker settings. There is no task-specific SAM schedule. Calibration and object libraries are offline setup; runtime uses only current/past frames. See [how extraction works](docs/HOW_EXTRACTION_WORKS.md).

## Offline setup

- Camera/arm calibration: `configs/calib_yam_rig_v3.json`, fixed three-frame joint lag and joint zero offsets.
- Robot: `configs/yam_bimanual.json` and the YAM URDF/link meshes.
- Rigid library: `assets/objects_v3/<task>/`, including the sweep tray, four-peg stand and asymmetric ear-shaped mug handle.
- Cloth: fixed single-sheet rest asset under `assets/cloth/tshirt_candidate132/`, fitted offline using multi-view evidence across the episode. Current frame0 registration fits its pose/wrinkles; topology, rest metric and material sample IDs then persist.
- Models: SAM2 small and online CoTracker under `.model_cache/`. SAM uses the checkpoint in the shared config; CoTracker defaults to the repo cache (`REALPC_MODEL_CACHE` can override its root). Its checkout is `hub/facebookresearch_co-tracker_main/` and checkpoint is `hub/checkpoints/scaled_online.pth`. Model weights and recordings are not committed.

Install `requirements.txt` and `requirements-cloth.txt`, plus the matching PyTorch/PyTorch3D, Open3D and SAM2 packages. `ffmpeg` must be on PATH. The tested local environment is `/home/howardji/miniconda3/envs/myenv`; its Open3D build needs the system `libstdc++` preload below.

## Convert recordings

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python scripts/convert_scene.py \
  --episodes /path/to/sweep/episode_0000 /path/to/hang/episode_0000 /path/to/greycloth/episode_0000 \
  --out /path/to/dataset_v2 --config configs/scene_pipeline_v2.json --save-masks
```

`--stop 12` runs a prefix check. `--skip-existing` reuses only files with matching config, implementation/assets hash and frame count. Outputs are written atomically, then loaded through `realpc.dataset.load`. Each episode has a QA summary, config, source hashes and file checksum. The root has `schema.json` and `manifest.json`.

```python
from realpc.dataset import load
d = load('/path/to/dataset_v2/sweep/episode_0000.npz')
print(d['points'].shape, d['valid'].shape)  # [T,N,3], [T,N]
```

## Live inference

Export `meta.json`, `calib.json` and `table_region.json` from the calibrated offline setup into a setup directory. The live constructor needs no recording, fused frames or joint-history file:

```python
from realpc.live import SceneSetup, build_live_pipeline
setup = SceneSetup('/path/to/setup', task='greycloth')
pipeline = build_live_pipeline(setup, 'configs/scene_pipeline_v2.json', cloth=True)
# frames: {camera: (depth_in_metres, BGR_image)}; q: calibrated, time-aligned left/right joints.
out = pipeline.step(t, frames, q)
points = pipeline.points(out)
```

Supply consecutive ticks starting at zero and current top/side RGB-D. Wrist RGB-D is optional and uses current FK camera poses. Joint alignment and zero-offset correction happen in the input adapter using the fixed past lag; the live step never reads a recording or future joints. `SceneSetup` exposes only static calibration metadata. The live constructor rejects recording-specific lag curves.

## Validation and limits

- `tests/check_scene_live_parity.py` compares live packets with converted points and masks while blocking recording access inside the live pipeline.
- All cloth geometry remains **approximate**, including initialization. `cloth_valid` is the shared current-frame support/strain/crossing/contact gate, not a frame-number cutoff or a hidden-material certificate. See [quality flags](docs/CLOTH_QUALITY_FLAGS.md). The selected tracker bounds local strain and uses collision-filtered fitting; brief held-contact lag triggers a bounded extra solve. Visible fold alignment remains approximate.
- Dark/occluded rigid surfaces, fast motion and imperfect wrist calibration remain difficult. SAM recovery cannot correct a wrong mesh or calibration.
- Playback frame rate is not extraction throughput. Final timings must include mask propagation, tracking and initialization; report the actual fraction of frames using SAM.

## One GPU job for the complete release

See [cluster staging, submission and recovery](docs/CLUSTER.md). One sequential
Slurm allocation runs environment checks, tests, seven validation episodes,
live/conversion parity, all 52 conversions, per-episode videos and quality flags,
manifest/checksum verification, and the zero-shot learned/analytic encoder audit.
Completed stages resume from fingerprinted markers; no nested jobs are submitted.

```bash
sbatch scripts/cluster/run.sbatch
```

The staged run must have `READY` before submission/release. The provided launcher
uses one Ampere-or-newer GPU on `sc-loprio`, account `viscam`, 96 GB RAM, and a
48-hour limit. Exclude currently bad nodes at submission as described in the guide.
Orion inputs, environment and outputs share a 50 GB ceiling; a 47 GB guard leaves
headroom. Finished outputs can be copied to external storage without recomputation.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_*.py
python tests/check_scene_live_parity.py --converted /path/to/dataset_v2/sweep/episode_0000.npz --frames 32 --out parity.json
```

Set `REALPC_EPISODES_ROOT` for parity and QA, `REALPC_MASKS_ROOT` for optional
independent QA masks, and `PLWM_SE3_SRC` for the external SE(3) encoder source.
The cluster driver sets these consistently. Do not mistake the round-trip error
of extracted points for alignment accuracy against camera observations.

## Repository scope

`realpc/` contains the shared runtime and its geometry helpers; `scripts/` has
conversion, QA, rendering and encoder entry points; `configs/` and `assets/`
contain the selected setup; `tests/` checks geometry, flags and live parity.
The historical experiment launchers and competing conversion scripts are preserved
on `archive/experiments-20261111` and tag `realpc-before-cleanup-20261111`, and have
been removed from this branch. Compatibility helpers required by the current
runtime/tests remain. Weights, recordings, output datasets and credentials are
external to the repository.

The earlier 18-test and parity results are recorded in [preflight](docs/PREFLIGHT_1111.md).
The new cluster run records its own tests and full-dataset results; pending results
are not certified by the code release tag. Cloth is always approximate, with
per-frame support, strain, intersection and contact flags retained in the dataset.
