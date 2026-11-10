# Q29 smooth object library and causal joint-camera tracker

Units: metres. Object coordinates retain the original v1 reference frames, so pose matrices have the same interpretation. Calibration is fixed at `configs/calib_yam_rig_v3.json` (including three-tick lag).

Models: open cardboard sweep box with measured wall planes; rounded thin dustpan tray with integral hollow handle; curved hand-broom body with hanging hole and a bristle envelope; four stacked wooden blocks; hollow lathed mug with continuous swept handle; peg stand with rounded base and four individually fitted angled pegs at two heights.

These are parametric/outline reconstructions fitted to observed clouds and initial RGB-D silhouettes from both cameras. Hidden backs, underside, wall thicknesses, the mug interior thickness, upper pole completion, peg caps and individual bristle structure are priors, not measured scans. Bristles use an envelope with sub-mm bundle ridges. Separate intersecting watertight components represent assemblies; not all meshes are a single boolean-unioned manufacturing solid.

`*_points.npz` retains the measured colored reference cloud for appearance registration; it is not a surface resampling of the new mesh. JSON `scans` preserve offline registrations. The box was fitted using current-t0 clouds from all 50 sweep episodes; other entries retain the earlier inherited scan records. Production initialization does not read episode-indexed poses from these records.

Worktree/branch: `/home/howardji/realpc-astra-meshes`, `astra-meshes`. Iteration log: `/home/howardji/point_tracker/plans/ASTRA_Q29_LOG_1109.md`. Frozen per-iteration assets, metrics and overlays: `/home/howardji/data_ext/astra_meshes/`. Full episode-0 sweep/hang audits improve on the original library/tracker; this is not a completed 50-episode production conversion. Consult the report and log for exact versions, comparisons, and remaining failures.

## Validated invocation

From the `astra-meshes` worktree, use the `myenv` Python environment and:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export REALPC_SIL=1 REALPC_CAM_EQUAL=1 REALPC_SIL_METHOD=contour
export REALPC_COLD_COMPONENTS=1 REALPC_BROOM_MATERIAL=1
export REALPC_BROOM_CLEAN_MASK=0 REALPC_BROOM_HOLE=0
export REALPC_WRIST_WEIGHT=1 REALPC_BROOM_REJECT_BG=0
```

`segment_objects.py --image-refine-seeds --assets assets/objects_v3` produces causal object masks. Give the same mask directory to `convert_v4.py --assets assets/objects_v3 --seg_dir ...`. For arbitrary-frame starts, give BOTH scripts `--cold --start T` and produce fresh masks beginning at T. The local SAM checkpoint is `/home/howardji/realpc/.model_cache/sam2/sam2.1_hiera_small.pt`; pass it explicitly when running from this worktree. Do not substitute Q28 evaluation masks for tracker inputs.

The live API is `StreamConverter.step(t, frames, labels=current_labels, q=calibrated_time_aligned_joints)`. It uses the same initializer and tracker as replay. Masks use library order +1, arms254/255. Supplied joints must include v3 offsets and the fixed three-tick image lag. Causal segmentation must retain only current/past memory. The implementation has been tested locally, but end-to-end real-time camera streaming performance has not been benchmarked.

Cold localization optionally uses `sweep/grasp_priors.json`: fixed hypotheses extracted from other demonstrations, explicitly excluding evaluation episodes0/15/30/45. Current two-camera evidence selects/refines a hypothesis; the evaluated episode's future is never used. Total cold-start occlusion remains unobservable without a trusted grasp/object prior. `obj_valid` means a finite pose was defined, not independently certified accuracy.

Final `.ply` files are authoritative. `build_smooth_v3.py` builds initial shapes; subsequent `quick_fit_v3.py` fits and the mug-handle repair are logged per iteration. Running the initial builder alone does not reproduce the final fitted library.

Review artifacts: `/home/howardji/point_tracker/reports/ASTRA_Q29_1109/`. Videos show raw RGB beside rendered outlines with robot/object and current-depth scene occlusion. Scoring retains the original robot/object-only visibility convention; scene-depth occlusion affects display only. Independent Q28 masks are SAM-derived estimates, not manual ground truth. Thin bristles, hidden surfaces, rapid releases, and strong occlusion remain approximate.

Stand revision v3.1 (Q30): four observed peg ends, two per height, verified across both fixed cameras and wrist views. Replaces two straight through-bars with four independent arms; restores the previously shortened upper +y arm. Static mesh fitting uses top/side depth throughout hang; online pose estimation remains causal. Evidence and new validation: `/home/howardji/point_tracker/plans/ASTRA_Q30_CLOTH_LOG_1109.md`. Prior Q29 stand metrics refer to the earlier mesh.

## Sweep box and shared streaming update

The box mesh is approximately 26.3 × 17.9 × 7.9 cm. Four wall planes and the floor come from 50 two-camera scans; 3 mm thickness is a prior, and upper-edge notches are simplified. It is a passive fixture, not assumed immobile: current observations can update sliding/yaw during acquisition or likely contact, but it never attaches to a gripper. Away from contact it retains its pose to avoid chasing foreground occlusion. Both contour and depth refinement preserve its acquired table-plane height/normal; this resting-tray model supports sliding/yaw, not lifting/tipping. An unlocatable cold-start box remains invalid until reacquired.

`configs/rigid_tracking_selected_1109.json` records the selected defaults. The broom uses current multiview bristle evidence, supported wrist observations, FK prediction and a full-object acceptance guard. More aggressive mask cleanup and a separate hole constraint remain disabled experiments because no reliable joint-camera gain was established.

`realpc.online_masks.OnlineObjectSegmenter` is the shared current-frame segmentation API. `convert_v4.py --online-seg` uses it directly. The cache-producing script uses the same `ObjectMaskStream`; no future image is exposed to the model. Six-frame cache/live conversion produced exactly identical poses and attachments after matching initialization. Six- versus twelve-frame segmentation produced identical first-six masks. Live masks must use current library IDs; stored caches are remapped by their object names when loading, so adding the box does not silently relabel the broom.

`StreamConverter.points(out)` and `point_valid(out)` return the live point array and its validity mask. Unknown bodies retain fixed point slots with invalid/NaN coordinates. `realpc.dataset.load` handles the same representation. Validity is not a certificate of geometric accuracy; inspect observation support and QA.

The paused dataset_v2 files are preserved. Previously finished sweep files lack the new box and need refreshed conversion when that task resumes. Full-episode and sampled/cold-start results are separate in `reports/ASTRA_Q29_1109/SWEEP_BOX_BROOM_REVIEW.md`; do not interpret the hard-frame audit as an all-episode closed-loop score.

Forward SAM memory retention keeps the initial conditioning frame(s) and at least32 recent outputs, expanded from the installed model memory/pointer spans. Old unselected mask tensors and raw images are discarded. This applies only to the current-frame streaming API, not retrospective interactive edits. A96-frame mask check crossing the retention boundary matched the untrimmed run exactly (0/22,118,400 differing pixels).
