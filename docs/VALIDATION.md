# Shared scene pipeline validation — 2026-11-10

The selected shared SAM interval is **1 processed frame**, for every object and both live inference and conversion. It has the highest equal-object mean silhouette overlap over the tested full episodes. N3 is better for cloth alone, and N10 is better for broom top-view overlap. Speed received no selection reward. See [all rate tables and references](benchmarks/sam_cadence_1110/README.md) and [decision and regressions](benchmarks/sam_cadence_1110/DECISION.md).

| SAM interval | Mean object/camera IoU | Hang fps | Sweep fps | Cloth fps |
|---|---:|---:|---:|---:|
| 1 (selected) | .8143 | 1.251 | .866 | .564 |
| 2 | .7811 | 1.968 | 1.369 | .483 |
| 3 | .8054 | 2.490 | 1.689 | .595 |
| 10 | .7834 | 3.977 | 2.490 | .534 |

One full validation episode per task: hang 499, sweep 849, cloth 453 frames. These measurements do not establish dataset-wide performance. The box has no independent mask and is excluded from the macro score; raw-depth and fixed-reference consistency remain reported. The selected run invokes SAM on all frames. FPS is complete conversion walltime on the local RTX4070TiSUPER, not video playback speed; no real-time 30 Hz claim is made.

## Causality and shared implementation

One config: configs/scene_pipeline_v2.json. ScenePipeline.step(t, frames, q) uses supplied current RGB-D and calibrated joints plus previous state. There is one segmentation stream and one forward SAM memory per camera. Rigid meshes, cloth deformation and robot FK consume this stream. Conversion saves the same outputs as live inference. Calibration and asset construction are offline; evaluation masks never enter runtime.

The four cadence configurations differ only in interval; implementation/assets hashes and reference-visible evaluation cohorts match. Robot transforms, joints and static points are identical across rates. All converted files passed realpc.dataset.load. Actual live/replay verification results are recorded with this benchmark.

## Remaining errors

N1 has a release-time broom error around frames 587–599 and a dustpan pose error at 750. Some depth metrics regress despite improved average silhouette overlap. The full tables and contact sheets retain these cases; a single cadence does not dominate every metric.

The cloth second fold improves substantially versus N10: median visible depth 19.09/29.10→7.02/8.01mm, maximum grasp error 276.7→33.9mm. N3 is better on visible fold alignment but not the overall object score. None of these runs is a physically valid cloth reconstruction: N1 maximum local extension is 168.5%, and tracker intersection flags do not verify hidden layers or finite-thickness collisions. Fold frames remain approximate and excluded from the conservative cloth_valid gate. That saved gate is a dataset QA annotation, not a decision supplied to live tracking.

Artifacts: /home/howardji/point_tracker/reports/SAM_ACCURACY_1110/. Historical N10 measurements are retained in [the previous validation](VALIDATION_N10_1110.md). The bulk 52-episode conversion and encoder audit remain cancelled;27 finished episodes and logs are preserved unchanged.

## Verification of the selected setting

Four pipeline tests pass with episode data supplied (no skips). Actual 12-frame live/replay checks pass for hang, sweep and cloth with recorded video/joint reads blocked inside the live step. Saved masks match exactly and maximum point difference is 5.96e-8m. Each prefix uses 12 SAM refreshes. All 12 benchmark outputs passed load verification. See [verification records](benchmarks/sam_cadence_1110/verification.json).
