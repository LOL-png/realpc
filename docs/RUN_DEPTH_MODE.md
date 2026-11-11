# Run the shared scene pipeline

Depth tracking is now part of the one production pipeline. SAM refresh, rigid depth/mesh fitting, deformable cloth tracking and robot FK use `configs/scene_pipeline_v2.json` and `ScenePipeline.step` for both live use and conversion.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python scripts/convert_scene.py --episodes /path/to/task/episode_0000 \
  --out /path/to/dataset_v2 --config configs/scene_pipeline_v2.json --save-masks
```

Use `--stop 32` for a quick prefix and `--skip-existing` for an exact-config/code resume. Do not precompute SAM masks or pass cached cloth tracks to this command. The same current packets generate the scene masks and point tracks inside the shared pipeline.

For a live/replay check:

```bash
python tests/check_scene_live_parity.py \
  --converted /path/to/dataset_v2/task/episode_0000.npz --frames 32 --out parity.json
```

The test blocks video/joint recording access inside the live step, compares decoded points and saved masks, and records the actual SAM frame count. Legacy `convert_v4.py --depth-mesh` and cadence scripts are retained on `archive/experiments-20261111` only.

For the complete release, use the single GPU job in [CLUSTER.md](CLUSTER.md). The generic batch wrapper below is available on a sufficiently provisioned processing host; it is not used on the local monitoring PC:

```bash
python scripts/convert_scene_batch.py --episodes-root /path/to/recordings \
  --out /path/to/dataset_v2 --workers 2
python scripts/qa_scene_dataset.py --dataset /path/to/dataset_v2 \
  --out /path/to/qa --example-videos --cloth-reference-inputs /path/to/qa_packets
```

The batch wrapper resumes completed files only when code/assets/config match, then verifies the complete collection and writes `SHA256SUMS`. Cloth reference packets are used only by the optional QA renderer/audit, never conversion. `verify_scene_dataset.py` can repeat the loader and checksum checks independently.
