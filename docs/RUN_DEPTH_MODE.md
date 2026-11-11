# Depth tracking presets

Both supported pipelines fit current depth with the same modern meshes:

```bash
python -m realpc accurate convert --episodes data/hang/episode_0000 --out out/accurate --save-masks
python -m realpc fast convert --episodes data/hang/episode_0000 --out out/fast --save-masks
```

Accurate uses SAM every frame. Fast uses SAM only at initialization and detected
tracking-failure events; it does not mean SAM is completely disabled. There is no
third legacy depth-mode entry point. See the [README](../README.md) for live
commands, calibration/model setup, benchmark provenance and quality trade-offs.
