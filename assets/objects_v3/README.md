# Q29 smooth object library and causal joint-camera tracker

Units: metres. Object coordinates retain the original v1 reference frames, so pose matrices have the same interpretation. Calibration is fixed at `configs/calib_yam_rig_v3.json` (including three-tick lag).

Models: rounded thin dustpan tray with integral hollow handle; curved hand-broom body with hanging hole and a bristle envelope; four stacked wooden blocks; hollow lathed mug with continuous swept handle; peg stand with rounded base and four individually fitted angled pegs at two heights.

These are parametric/outline reconstructions fitted to observed clouds and initial RGB-D silhouettes from both cameras. Hidden backs, underside, wall thicknesses, the mug interior thickness, upper pole completion, peg caps and individual bristle structure are priors, not measured scans. Bristles use an envelope with sub-mm bundle ridges. Separate intersecting watertight components represent assemblies; not all meshes are a single boolean-unioned manufacturing solid.

`*_points.npz` retains the measured colored reference cloud for appearance registration; it is not a surface resampling of the new mesh. JSON `scans` are inherited initial-pose registrations, not claims that the smooth model was fitted independently on every listed episode.

Worktree/branch: `/home/howardji/realpc-astra-meshes`, `astra-meshes`. Iteration log: `/home/howardji/point_tracker/plans/ASTRA_Q29_LOG_1109.md`. Frozen per-iteration assets, metrics and overlays: `/home/howardji/data_ext/astra_meshes/`. Full episode-0 sweep/hang audits improve on the original library/tracker; this is not a completed 50-episode production conversion. Consult the report and log for exact versions, comparisons, and remaining failures.

## Validated invocation

From the `astra-meshes` worktree, use the `myenv` Python environment and:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export REALPC_SIL=1 REALPC_CAM_EQUAL=1 REALPC_SIL_METHOD=contour
export REALPC_COLD_COMPONENTS=1
```

`segment_objects.py --image-refine-seeds --assets assets/objects_v3` produces causal object masks. Give the same mask directory to `convert_v4.py --assets assets/objects_v3 --seg_dir ...`. For arbitrary-frame starts, give BOTH scripts `--cold --start T` and produce fresh masks beginning at T. The local SAM checkpoint is `/home/howardji/realpc/.model_cache/sam2/sam2.1_hiera_small.pt`; pass it explicitly when running from this worktree. Do not substitute Q28 evaluation masks for tracker inputs.

The live API is `StreamConverter.step(t, frames, labels=current_labels, q=calibrated_time_aligned_joints)`. It uses the same initializer and tracker as replay. Masks use library order +1, arms254/255. Supplied joints must include v3 offsets and the fixed three-tick image lag. Causal segmentation must retain only current/past memory. The implementation has been tested locally, but end-to-end real-time camera streaming performance has not been benchmarked.

Cold localization optionally uses `sweep/grasp_priors.json`: fixed hypotheses extracted from other demonstrations, explicitly excluding evaluation episodes0/15/30/45. Current two-camera evidence selects/refines a hypothesis; the evaluated episode's future is never used. Total cold-start occlusion remains unobservable without a trusted grasp/object prior. `obj_valid` means a finite pose was defined, not independently certified accuracy.

Final `.ply` files are authoritative. `build_smooth_v3.py` builds initial shapes; subsequent `quick_fit_v3.py` fits and the mug-handle repair are logged per iteration. Running the initial builder alone does not reproduce the final fitted library.

Review artifacts: `/home/howardji/point_tracker/reports/ASTRA_Q29_1109/`. Videos show raw RGB beside rendered outlines with robot/object and current-depth scene occlusion. Scoring retains the original robot/object-only visibility convention; scene-depth occlusion affects display only. Independent Q28 masks are SAM-derived estimates, not manual ground truth. Thin bristles, hidden surfaces, rapid releases, and strong occlusion remain approximate.

Stand revision v3.1 (Q30): four observed peg ends, two per height, verified across both fixed cameras and wrist views. Replaces two straight through-bars with four independent arms; restores the previously shortened upper +y arm. Static mesh fitting uses top/side depth throughout hang; online pose estimation remains causal. Evidence and new validation: `/home/howardji/point_tracker/plans/ASTRA_Q30_CLOTH_LOG_1109.md`. Prior Q29 stand metrics refer to the earlier mesh.
