# Historical event-only depth benchmark

These measurements are from the earlier rigid tracker (timed commit `8846263`; reporting commit `c94720f`), not the current two-preset release. They predate later handle/initialization/cloth changes. Do not treat these numbers as a benchmark of current `fast`, or compare their medians directly with the cadence benchmark means. No cloth event-only result was measured.

| Episode | Frames | SAM fps | Depth fps | Speedup | SAM event frames in depth mode |
|---|---:|---:|---:|---:|---:|
| sweep | 849 | 1.06 | 2.69 | 2.53× | 99/849 (11.7%) |
| hang | 499 | 1.80 | 5.46 | 3.03× | 16/499 (3.2%) |

| Object | Scored frames SAM → depth (top/side) | SAM pass % top/side | Depth pass % top/side | Median IoU SAM → depth (top/side) |
|---|---:|---:|---:|---|
| sweep/box* | 836/849 → 833/849 | 84.9/97.2 | 84.2/99.6 | 0.772/0.954 → 0.783/0.955 |
| sweep/broom | 849/841 → 847/840 | 91.0/97.7 | 96.5/98.0 | 0.783/0.818 → 0.803/0.809 |
| sweep/debris | 709/692 → 710/694 | 89.8/95.4 | 85.8/93.1 | 0.836/0.870 → 0.842/0.855 |
| sweep/dustpan | 849/849 → 849/849 | 93.2/97.6 | 93.4/99.8 | 0.886/0.884 → 0.875/0.878 |
| hang/mug | 499/499 → 499/499 | 94.0/100.0 | 92.2/99.8 | 0.851/0.843 → 0.768/0.833 |
| hang/stand | 499/499 → 499/499 | 99.2/100.0 | 96.0/100.0 | 0.851/0.890 → 0.838/0.872 |

IoU >= 0.5 on assessable views. Every timestamp is tested; a view is scored when reference OR rendered foreground has at least 60 pixels. Counts can differ because a spurious visible prediction is scored as a failure even if the other tracker correctly predicts an occluded object. No unmatched predictions are dropped. Robot/other-object occlusion is excluded. Independent Q28 masks except box, which uses the same fresh SAM reference for both runs. These are mask agreements, not ground-truth 6-DoF pose errors. Box (*) has no independent Q28 mask. Timing uses serialized runs on an RTX 4070 Ti SUPER with OMP/MKL=2; includes initialization, input and compact conversion, excludes process startup and optional QA-mask export. Videos use recording time, not extraction speed.
