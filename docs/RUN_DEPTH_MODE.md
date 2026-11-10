# Run the shared scene pipeline

Depth tracking is now part of the one production pipeline. SAM refresh, rigid depth/mesh fitting, deformable cloth tracking and robot FK use `configs/scene_pipeline_v2.json` and `ScenePipeline.step` for both live use and conversion.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python scripts/convert_scene.py --episodes /path/to/task/episode_0000 \
  --out /path/to/dataset_v2 --config configs/scene_pipeline_v2.json --save-masks
```

Use `--stop 12` for a quick prefix and `--skip-existing` for an exact-config/code resume. Do not precompute SAM masks or pass cached cloth tracks to this command. The same current packets generate the scene masks and point tracks inside the shared pipeline.

For a live/replay check:

```bash
python tests/check_scene_live_parity.py \
  --converted /path/to/dataset_v2/task/episode_0000.npz --out parity.json
```

The test blocks video/joint recording access inside the live step, compares decoded points and saved masks, and records the actual SAM frame count. Legacy `convert_v4.py --depth-mesh` and cadence scripts remain research ablations only.
