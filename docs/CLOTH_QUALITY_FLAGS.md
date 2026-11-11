# Cloth quality flags

`ScenePipeline.step` computes the cloth flags from the current frame. Conversion
stores those exact flags; it no longer labels frames by a recording-specific
frame150 cutoff. Live/replay checks compare the saved flags as well as geometry.

`cloth_valid` requires finite samples, an explicit negative intersection check,
the configured strain bound when enabled, and current top/side surface support.
When `segmentation.cloth_fit_max_hard_grasp_mm` is configured, held contact centres
must also remain within that current-frame error limit (5 mm in candidate 138).
This rejects collision-limited contact lag without relaxing the mesh constraints;
released frames do not require a held-contact check. Historical configurations
without the field retain their original gate.
The support thresholds reuse `segmentation.cloth_fit_min_coverage` and
`segmentation.cloth_fit_max_median_mm`. Detailed checks are in
`cloth_metrics.quality`. In a mode without a strain bound,
`strain_bound_enforced` is false: validity is not a physical certificate.

`cloth_approximate` is always true. The garment is an estimated template, including
at initialization; hidden layers and material correspondence are not measured
ground truth. The legacy `cloth_fold_approximate` field carries the same conservative
flag for compatibility. `cloth_geometry_observed_only` is false. Per-point
`cloth_observed` remains a separate current-depth visibility/support flag.

Alignment audits must include all frames, including invalid ones, and report
coverage as well as depth error. Encoder reports must state the valid fraction
and distinguish approximate cloth from rigid-object groups.

For an experimental conversion, `--cloth-preview-every 25` saves current input
packets and mesh/metric prefixes under `TASK/EPISODE_preview/`. These are diagnostic
outputs, never inputs to the live step. This option is off for bulk conversion.
