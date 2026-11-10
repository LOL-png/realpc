# Mug handle correction, 11-10

Replaced the small circular arc with a general swept strap: straight outer leg, rounded Hermite transitions and rounded-rectangle cross-section. Original cup vertex coordinates and triangles are exactly preserved; only face ordering may differ. Four-peg stand and calibration are unchanged.

| Object/view | Alignment pass before → after | Median IoU before → after |
|---|---:|---:|
| mug/top | 92.2% → 91.8% | 0.768 → 0.829 |
| stand/top | 96.0% → 95.4% | 0.838 → 0.837 |
| mug/side | 99.8% → 100.0% | 0.833 → 0.812 |
| stand/side | 100.0% → 100.0% | 0.872 → 0.858 |

Accepted iter11 baseline; same499 timestamps, causal depth mode, independent Q28 mask IoU≥.5. This is a shape correction with mixed tracking changes, not a universal accuracy gain. Full converted replay loads as499×21207×3 points, all valid points finite.


Full evidence: /home/howardji/point_tracker/reports/ASTRA_Q29_1109/mug_handle_review/README.md

Rebuild with scripts/build_mug_handle_candidate.py --params <handle_params.json> --source assets/objects_v3 --out <separate candidate library>. No runtime algorithm change.
