# Accuracy-first shared SAM cadence

Selected interval: **1 processed frame**, for all scene objects and both live/conversion. Speed is not part of the selection score.

The fixed bounded cloth baseline (candidate 136) was freshly converted for all 453 frames at each interval. Rigid code/assets are unchanged from the completed full hang (499 frames) and sweep (849 frames) benchmark, whose results are reused. All four cloth configurations differ only in interval and have matching source/assets hashes.

| Object (mean top / side IoU) | N1 | N2 | N3 | N10 |
|---|---:|---:|---:|---:|
| mug | 0.798 / 0.867 | 0.792 / 0.852 | 0.786 / 0.845 | 0.774 / 0.823 |
| stand | 0.821 / 0.885 | 0.816 / 0.876 | 0.813 / 0.873 | 0.788 / 0.865 |
| broom | 0.734 / 0.778 | 0.741 / 0.785 | 0.734 / 0.775 | 0.764 / 0.779 |
| dustpan | 0.824 / 0.851 | 0.823 / 0.856 | 0.820 / 0.843 | 0.819 / 0.842 |
| debris | 0.780 / 0.842 | 0.636 / 0.831 | 0.757 / 0.808 | 0.765 / 0.793 |
| box (proxy) | 0.622 / 0.907 | 0.634 / 0.921 | 0.624 / 0.921 | 0.619 / 0.920 |
| cloth (proxy) | 0.846 / 0.850 | 0.840 / 0.837 | 0.846 / 0.847 | 0.818 / 0.809 |

| Interval | Equal object/camera mean IoU | Cloth SAM frames / 453 | Hang fps | Sweep fps | Cloth fps* |
|---|---:|---:|---:|---:|---:|
| 1 | 0.8231 | 453 | 1.251 | 0.866 | 0.0676 |
| 2 | 0.8072 | 227 | 1.968 | 1.369 | 0.0652 |
| 3 | 0.8124 | 151 | 2.490 | 1.689 | 0.0779 |
| 10 | 0.8032 | 46 | 3.977 | 2.490 | 0.0587 |

*Cloth timings are complete conversions running concurrently, not isolated speed comparisons. Rigid timings are isolated. The bounded solver dominates cloth cost; lower SAM frequency is not monotonically faster. Playback FPS is unrelated.

The score gives equal weight to five rigid objects plus cloth, and equal weight to cameras. Rigid reference masks are Q28 evaluation-only masks; they never enter extraction. Cloth uses the same fixed current-mask/depth reference for every rate. Box has no independent masks and uses a fixed proxy; it is reported but excluded from the primary score. N1 also wins using only the five independent rigid objects, or including the box proxy (see sensitivity.json).

Each bounded cloth run has zero exported-mesh triangle crossings and at most 1.90% extension. Good silhouette overlap does not establish material identity or hidden-layer correctness. Some broom/top and depth metrics favor other rates; one rate does not dominate every object. The subsequent adaptive-contact change is a separate N1 cloth validation, not a new four-rate benchmark.

Full per-frame audits and videos: reports/ASTRA_Q30_1110/unified136/ and reports/SAM_ACCURACY_1110/.
