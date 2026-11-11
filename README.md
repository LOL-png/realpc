# realpc: causal scene extraction

The current entry point is `ScenePipeline.step(t, frames, q)` in `realpc/scene_pipeline.py`. Live inference and training-data conversion call this same function. It produces fixed-identity robot, rigid-object, static-scene and cloth points from current RGB-D and calibrated joints.

The shared policy is **SAM every processed frame for every object**. This gave the highest equal-object mean silhouette overlap in full-episode comparisons of intervals 1, 2, 3 and 10. It does not win every object or depth metric. See [validation and limitations](docs/VALIDATION.md) for the complete accuracy/speed comparison and unresolved cloth deformation errors. The 52-episode conversion remains cancelled; completed outputs are preserved.

## What runs

1. One scene segmentation stream refreshes all objects using one shared SAM2 model and one forward memory per camera. The selected schedule refreshes every frame. With a larger configured interval, current depth, mesh predictions and optical flow carry masks between refreshes; skipped frames never enter SAM.
2. Rigid objects use depth/mesh fitting and gripper FK when held. The cloth uses a clean shirt template, causal point tracks and a deformable mesh. Robot geometry comes from FK.
3. Fixed mesh/link samples produce the point cloud. The recording adapter saves the same output that a live consumer receives.

`configs/scene_pipeline_v2.json` is the shared config for the SAM interval, recovery thresholds and tracker settings. There is no task-specific SAM schedule. Calibration and object libraries are offline setup; runtime uses only current/past frames. See [how extraction works](docs/HOW_EXTRACTION_WORKS.md).

## Offline setup

- Camera/arm calibration: `configs/calib_yam_rig_v3.json`, fixed three-frame joint lag and joint zero offsets.
- Robot: `configs/yam_bimanual.json` and the YAM URDF/link meshes.
- Rigid library: `assets/objects_v3/<task>/`, including the sweep tray, four-peg stand and asymmetric ear-shaped mug handle.
- Cloth: clean single-sheet shirt outline fitted to frame 0; topology and material sample IDs persist afterward.
- Models: SAM2 small and online CoTracker under `.model_cache/` (the local shared config points to the installed cache). Model weights and recordings are not committed.

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

Construct a `StreamConverter` with the same calibration, robot and object library, then:

```python
from realpc.scene_pipeline import ScenePipeline, load_scene_config
pipeline = ScenePipeline(converter, load_scene_config('configs/scene_pipeline_v2.json'), cloth=has_cloth)
# frames: {camera: (depth_in_metres, BGR_image)}; q: calibrated, time-aligned left/right joints.
out = pipeline.step(t, frames, q)
points = pipeline.points(out)
```

Supply consecutive ticks starting at zero and current top/side RGB-D. Wrist RGB-D is optional and uses current FK camera poses. Joint alignment happens in the input adapter using the fixed past lag; the live step never reads a recording or future joints.

## Validation and limits

- `tests/check_scene_live_parity.py` compares live packets with converted points and masks while blocking recording access inside the live pipeline.
- Cloth folds remain **approximate**: every-frame SAM reduces the second-fold tracking loss, but visible-surface alignment does not establish correct hidden material correspondence. Excessive stretch and residual grasp error remain. Fold frames are marked separately and excluded from the conservative `cloth_valid` gate.
- Dark/occluded rigid surfaces, fast motion and imperfect wrist calibration remain difficult. SAM recovery cannot correct a wrong mesh or calibration.
- Playback frame rate is not extraction throughput. Final timings must include mask propagation, tracking and initialization; report the actual fraction of frames using SAM.

Older conversion/segmentation scripts are retained for reproducible ablations, not as alternate production paths. Their former documentation is in [legacy pipelines](docs/LEGACY_PIPELINES.md).
