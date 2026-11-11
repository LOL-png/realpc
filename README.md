# realpc

Causal RGB-D scene extraction for calibrated robot workspaces. **Exactly two
pipelines** share one tracking implementation and the same current assets:

| Pipeline | SAM policy | Intended use | Config |
|---|---|---|---|
| **accurate** | Every processed frame | Dataset production and difficult occlusion/contact sequences | [pipeline_accurate.json](configs/pipeline_accurate.json) |
| **fast** | Initialization and tracking-failure events only | Lower-cost rigid tracking when occasional alignment regressions are acceptable | [pipeline_fast.json](configs/pipeline_fast.json) |

Both fit rigid meshes to current depth, generate robot points from FK, and track
cloth with the current deformable shirt template. Between SAM events, fast uses
depth/mesh association for rigid objects and depth/flow propagation for cloth.
It is not a second segmentation model. Both modes call the same
`ScenePipeline.step(t, frames, q)` in live inference and recording conversion;
only current/past observations enter the step.

## Install and assets

From a checkout, install the tested Python dependencies:

```bash
pip install -r requirements.txt -r requirements-cloth.txt -r requirements-dev.txt
```

Also install matching CUDA PyTorch/PyTorch3D, Open3D, SAM2, and put `ffmpeg` on
PATH. Tested package versions are recorded in the requirements files. For an
Open3D `GLIBCXX` error, use a compatible system/Conda C++ runtime; machine-specific
library paths are not baked into the commands.

Place model dependencies under `.model_cache/`, or set `REALPC_MODEL_CACHE`:

```
.model_cache/
  sam2/sam2.1_hiera_small.pt
  hub/facebookresearch_co-tracker_main/   # CoTracker source checkout
  hub/checkpoints/scaled_online.pth
```

`REALPC_SAM_CHECKPOINT` optionally overrides only the SAM checkpoint. Both modes
resolve asset paths relative to their config, not the shell's working directory.

Only the selected assets are on main:

- `assets/objects_v3/`: sweep box/tray, broom, dustpan, debris; hang mug with the
  asymmetric **ear-shaped handle**, and the **four-peg stand**.
- `configs/calib_yam_rig_v3.json`: current camera/arm calibration and fixed past
  joint alignment; `configs/yam_bimanual.json` plus `robot_models/yam_v1/`: current robot.
- `assets/cloth/tshirt_candidate132/`: current offline rest asset, used by the
  bounded tracker with adaptive contact refinement. The directory's candidate
  number identifies the asset, not an older runtime pipeline.

Older assets/calibrations/configs are preserved on
[`archive/pre-two-pipelines-20261111`](https://github.com/LOL-png/realpc/tree/archive/pre-two-pipelines-20261111).
They are not loaded by either preset. Model weights and recordings are not committed.

## Accurate pipeline

Use this for accuracy-first conversion, occlusion, hanging contact and folding.
It is the previous selected `scene_pipeline_v2` configuration, now named
`pipeline_accurate.json`, with SAM on every frame and depth fitting afterward.

```bash
python -m realpc accurate convert --episodes data/sweep/episode_0000 \
  --out out/accurate --save-masks
python -m realpc accurate live --setup setup/rig --task sweep \
  --source camera_adapter:packets --out out/live_accurate
```

The fixed-tracker SAM-rate benchmark measured **0.866 fps sweep, 1.251 fps hang**,
and equal-object/camera mean IoU **0.8231** at interval 1. The bounded cloth
reference measured **0.0676 fps in concurrent runs**, not an isolated speed test.
Later contact and initialization corrections have separate validation, so these
are benchmark-snapshot measurements, not a new timing of this release.

## Fast pipeline

Use this when reduced SAM computation matters more than the best alignment.
`interval: 0` disables periodic refresh; SAM still runs at initialization and
persistent/urgent fit failures, subject to the same configured cooldowns. Rigid
tracking uses depth, mesh predictions and gripper FK between events. Cloth uses
the same bounded solver, so fewer SAM calls do **not** guarantee faster cloth fitting.

```bash
python -m realpc fast convert --episodes data/hang/episode_0000 \
  --out out/fast --save-masks
python -m realpc fast live --setup setup/rig --task hang \
  --source camera_adapter:packets --out out/live_fast
```

The **historical event-only rigid fallback** measured **2.69 fps sweep / 5.46 fps
hang**, versus 1.06 / 1.80 fps for its contemporaneous SAM baseline. It called SAM
on 11.7% / 3.2% of frames. Mug median IoU fell from 0.851/0.843 to **0.768/0.833**
(top/side); broom changed from 0.783/0.818 to **0.803/0.809**. See the
[historical event-only table and scope](docs/benchmarks/event_only_1109/README.md).
Those runs predate later asset/tracker changes. **Current fast with the newest
assets has not been benchmarked end to end; event-only cloth is unmeasured.**

The shared SAM-rate benchmark did not test event-only mode. Its interval-10 result
is context, not a measurement of fast:

| SAM interval | Mean IoU | Sweep fps | Hang fps | Cloth fps (concurrent) |
|---|---:|---:|---:|---:|
| 1 | 0.8231 | 0.866 | 1.251 | 0.0676 |
| 2 | 0.8072 | 1.369 | 1.968 | 0.0652 |
| 3 | 0.8124 | 1.689 | 2.490 | 0.0779 |
| 10 | 0.8032 | 2.490 | 3.977 | 0.0587 |

These are recorded benchmark ablations, not additional supported pipelines.
[Per-object results and methodology](docs/benchmarks/bounded_cadence_1110/README.md)
distinguish independent rigid masks, cloth proxy masks and the box proxy excluded
from the primary score. One episode per task does not establish generalization.
Playback FPS is unrelated to extraction throughput.

## Live input and output contract

The live CLI imports your `camera_adapter:packets` zero-argument factory. It must
yield dictionaries with consecutive `tick` values starting at zero, `frames`
mapping camera names to `(depth_metres, BGR_image)`, and calibrated, time-aligned
left/right joint vectors in `q`. Top/side RGB-D are required; wrists are optional.
The hardware adapter owns acquisition; the tracker never opens future frames.
`--setup` contains only `meta.json`, `calib.json`, and `table_region.json`.

```python
from realpc.live import SceneSetup, build_live_pipeline
from realpc.__main__ import preset_config

pipeline = build_live_pipeline(SceneSetup('setup/rig', task='greycloth'),
                               preset_config('accurate'), cloth=True)
row = pipeline.step(t, current_frames, calibrated_joints)
points = pipeline.points(row)
```

The live CLI writes one point NPZ per tick, with finite-point validity, SAM-call
status and cloth quality flags. Use the Python API to consume those same outputs
in memory. Recording conversion writes compact fixed-identity samples/poses plus
QA, provenance, config and checksums; its format is deliberately different from
the simple per-frame live export:

```python
from realpc.dataset import load
d = load('out/accurate/sweep/episode_0000.npz')
print(d['points'].shape, d['valid'].shape)  # [T,N,3], [T,N]
```

`convert --stop 12` checks a prefix; `--skip-existing` only reuses matching
config/code/assets and complete frame counts. `--save-masks` enables shared-mask QA.
All cloth is **approximate**. `cloth_valid` checks current surface support,
strain, crossings and held-contact consistency; it does not certify hidden layers
or material correspondence. See [quality flags](docs/CLOTH_QUALITY_FLAGS.md) and
[validation](docs/VALIDATION.md).

## Tests, QA and cluster processing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_*.py
export REALPC_EPISODES_ROOT=/path/to/recordings
python tests/check_scene_live_parity.py \
  --converted out/accurate/sweep/episode_0000.npz --frames 32 --out parity.json
python scripts/qa_scene_dataset.py --dataset out/accurate --out out/qa --example-videos
```

`REALPC_MASKS_ROOT` supplies optional independent QA masks; cloth QA additionally
uses `--cloth-reference-inputs`. These references never enter conversion.
`PLWM_SE3_SRC` locates the external SE(3) AE source for round-trip evaluation.
[Cluster instructions](docs/CLUSTER.md) cover staged environments, choosing a
preset, one-job conversion/QA/encoder evaluation, checksums and optional delivery.
Never update a running job's frozen checkout when updating this repository.

The public entry points are the four commands above. `scripts/` contains their
conversion, audit and batch helpers; `realpc/` contains shared tracking and geometry
code. Historical experiments live on archive branches, not as alternate presets.
