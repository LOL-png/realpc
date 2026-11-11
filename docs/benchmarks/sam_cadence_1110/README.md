# Unified SAM accuracy comparison

Ranking excludes box consistency (no independent box mask) and requires separate depth/visual review. Cloth masks are a fixed proxy. One full validation episode per task; not a held-out generalization claim. More frequent SAM does not validate hidden cloth or physical deformation.

Primary IoU uses reference-visible pixels (>=60 at audit resolution) with the SAME cohort at every cadence. Rigid reference: Q28; cloth: fixed current-frame masks/depth. Box scores are consistency only and excluded from the macro ranking.

| Object (mean IoU top / side) | Every1 | Every2 | Every3 | Every10 |
|---|---|---|---|---|
| mug | 0.798 / 0.867 | 0.792 / 0.852 | 0.786 / 0.845 | 0.774 / 0.823 |
| stand | 0.821 / 0.885 | 0.816 / 0.876 | 0.813 / 0.873 | 0.788 / 0.865 |
| broom | 0.734 / 0.778 | 0.741 / 0.785 | 0.734 / 0.775 | 0.764 / 0.779 |
| dustpan | 0.824 / 0.851 | 0.823 / 0.856 | 0.820 / 0.843 | 0.819 / 0.842 |
| debris | 0.780 / 0.842 | 0.636 / 0.831 | 0.757 / 0.808 | 0.765 / 0.793 |
| box | 0.622 / 0.907 | 0.634 / 0.921 | 0.624 / 0.921 | 0.619 / 0.920 |
| cloth | 0.783 / 0.807 | 0.663 / 0.702 | 0.794 / 0.816 | 0.684 / 0.704 |

| Object (depth median mm top / side) | Every1 | Every2 | Every3 | Every10 |
|---|---|---|---|---|
| mug | 5.815 / 7.857 | 5.522 / 6.866 | 5.788 / 7.059 | 5.917 / 6.143 |
| stand | 3.596 / 5.444 | 3.539 / 5.207 | 3.512 / 5.035 | 3.624 / 4.864 |
| broom | 8.200 / 10.698 | 7.783 / 9.873 | 7.895 / 9.873 | 7.075 / 9.455 |
| dustpan | 14.864 / 11.807 | 13.183 / 11.244 | 12.899 / 11.812 | 13.305 / 11.234 |
| debris | 2.697 / 9.600 | 2.768 / 9.499 | 2.415 / 9.338 | 2.660 / 9.258 |
| box | 4.936 / 6.752 | 5.520 / 6.733 | 4.669 / 6.810 | 5.041 / 7.433 |
| cloth | 5.855 / 7.976 | 4.901 / 7.520 | 4.520 / 6.927 | 4.497 / 7.221 |

| End-to-end fps | Every1 | Every2 | Every3 | Every10 |
|---|---|---|---|---|
| hang | 1.251 | 1.968 | 2.490 | 3.977 |
| sweep | 0.866 | 1.369 | 1.689 | 2.490 |
| greycloth | 0.564 | 0.483 | 0.595 | 0.534 |

Cloth fold phase medians (top / side; fixed reference, hidden layers unmeasured):

| Interval | First-fold IoU | Second-fold IoU | First-fold depth mm | Second-fold depth mm |
|---|---|---|---|---|
| 1 | 0.773 / 0.847 | 0.715 / 0.634 | 8.037 / 10.717 | 7.018 / 8.008 |
| 2 | 0.766 / 0.844 | 0.218 / 0.233 | 7.785 / 9.474 | 16.186 / 27.893 |
| 3 | 0.804 / 0.856 | 0.732 / 0.677 | 6.448 / 7.563 | 6.742 / 7.221 |
| 10 | 0.762 / 0.859 | 0.261 / 0.244 | 7.804 / 7.708 | 19.093 / 29.096 |

Actual SAM fractions, per-view IoU>=0.5 pass rates, depth, and step-only fps are in [CSV](per_object_alignment.csv) and [JSON](summary.json). Robot/table geometry does not depend on SAM cadence.
A SAM refresh also invokes rigid silhouette refinement in the existing shared pipeline; this comparison measures the complete refresh policy.
N10 cloth reuses the completed full control after exact source/config/asset hashes and output checksum verification; the other measured conversions are fresh.

Accuracy ranking (completed rates): [1, 3, 10, 2]. Selected shared interval: **1**. See [decision and visual review](DECISION.md) for trade-offs and verification.
