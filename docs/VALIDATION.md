# Unified pipeline validation — 2026-11-10

The selected shared configuration is `configs/scene_pipeline_v2.json`: SAM every processed frame, rigid depth/mesh tracking, FK robot points, and a bounded single-sheet cloth tracker. Live inference and conversion call the same `ScenePipeline.step`.

## SAM cadence

The completed fixed-tracker N1/N2/N3/N10 benchmark selects N1 by accuracy, with equal-object/camera mean IoU .8231/.8072/.8124/.8032. N1 also wins using only independent rigid references or including the box proxy. The [per-object tables, SAM call fractions, speed and caveats](benchmarks/bounded_cadence_1110/README.md) retain all rates. The subsequent contact refinement is a separate N1 validation.

## Shirt alignment and physics

| Phase | Previous N1 depth top/side mm | Selected depth top/side mm | Previous IoU | Selected IoU |
|---|---:|---:|---:|---:|
| before lift | 1.93/2.18 | 6.26/3.93 | 0.866/0.893 | 0.837/0.890 |
| first fold | 8.04/10.72 | 4.59/4.14 | 0.773/0.847 | 0.835/0.863 |
| second fold | 7.02/8.01 | 5.89/4.10 | 0.715/0.634 | 0.832/0.807 |

Depth entries are medians of per-frame visible-overlap medians against common masks/depth. Initial/pre-lift depth and some upper tails regress. Silhouette/depth consistency does not measure hidden material identity. Held-out long LK-track errors during the first fold are 9.8/12.1 px versus 6.8/10.1 px previously (top/side), so the surface-fit gains do not establish improved material tracking. Feature populations differ; [full track diagnostics](benchmarks/bounded_cloth_1110/tracking_errors.json) report counts and all phases.

All 453 exported meshes pass the independent triangle-crossing check. Maximum extension/compression is 1.90%/1.80%; contact-centre residual is below 1e-6 mm in this trajectory. A bounded extra solve on a lagging held contact uses nine additional iterations in total, fixing the earlier 36.6 mm exception. Relative to the unrefined bounded tracker, second-fold IoU changes .835/.819 to .832/.807 and median depth changes 5.97/4.44 to 5.89/4.10 mm. This is a contact-reliability improvement with a small overlap cost, not a win on every metric. The surrounding soft grasp patch can still deviate; it is reported separately from the constrained centres.

The shirt is a single-sheet surface approximation. Its fixed rest asset uses all-frame/two-view offline evidence; frame 0 registration and subsequent tracking use only current/past packets. Sleeves, wrinkles, occluded material and fold-edge placement remain imperfect. All cloth is marked approximate. The [quality policy](CLOTH_QUALITY_FLAGS.md) uses current support, physical checks and a 5 mm held-contact gate, never a frame-number cutoff.

## Scope of verification

Candidate136 ran fresh through the entire unified pipeline at each cadence. The adaptive candidate138 was tested as a causal continuation from the verified frame 347 checkpoint; its unchanged prefix is reused. The restored unchanged next step differed from the original by only 5.9 micrometres. No future pose or frame is used. At N1, the segmentation stream refreshes every frame and after_tracking returns without cloth-fit feedback, so these current packets remain applicable to this tracker-only change. This is not described as a new full frame-zero run. The final dataset conversion starts each episode fresh at 0 and will be audited again.

Fresh 32-frame candidate138 conversion and live replay pass, including online tracker memory commits. Live construction uses only meta.json, calib.json and table_region.json; recording/video/joint reads are blocked inside step. Masks and quality flags match exactly; maximum point difference is 5.96e-8 m. Hang and sweep also pass 32-frame checks. One initial hang repeat differed by 0.18 mm at frame 2, while fresh conversion and subsequent repeat matched; that unexplained numerical variation remains recorded. These checks do not prove bitwise reproducibility for every native path.

## Speed and remaining rigid errors

The isolated rigid N1 benchmark measures 0.866 fps sweep and 1.251 fps hang. The fresh bounded N1 cloth reference takes 6,699 s / 0.0676 fps while cadence jobs overlap; the adaptive continuation is not an isolated full-pipeline speed measurement. SAM runs on 100% of frames, with one scene refresh per camera. Causality does not imply real-time throughput.

Known rigid weaknesses include a release-time broom error near 587–599 and dustpan pose error at750 in the sampled sweep episode. The box lacks independent masks. Full-dataset QA must report missing poses, raw-depth agreement and stream consistency separately from independent mask scores.

## Artifacts

- [Selected shirt contact sheet](benchmarks/bounded_cloth_1110/contact_sheet.jpg) and [machine-readable summary](benchmarks/bounded_cloth_1110/summary.json).
- Full shirt video, per-frame metrics and plots: `/home/howardji/point_tracker/reports/ASTRA_Q30_1110/shirt138/`.
- Sweep/hang videos: `/home/howardji/point_tracker/reports/SAM_ACCURACY_1110/sweep_n1/` and `hang_n1/`.
- Earlier evidence: [historical pre-bounded validation](VALIDATION_PRE_BOUNDED_1110.md).

The release is local only. The 52-episode conversion follows the finalized code, parity checks and delivery of these videos; existing partial data are preserved.
