# Shirt asset and constrained tracking experiments

Work is isolated on `astra-cloth`. These experiments are **not a validated
replacement** for the shared pipeline. Bulk conversion, QA and encoder work were
cancelled at the user's request; 27 completed episodes and all logs remain.

The current-packet API remains `CausalClothTracker.step`; `ScenePipeline` and the
recording adapter use that same implementation and configuration factory. Offline
asset builders may inspect the entire calibration recording. Runtime loads only
fixed geometry/landmarks and uses current/past RGB-D, masks, tracks and FK.
No frame-indexed pose or deformation is stored in an asset.

## Offline candidates

- `scripts/shirt_asset_build.py`: all-frame, two-view visible depth chords,
  15 outline controls and a regular flat cut pattern. Correspondences come from
  an imperfect earlier track; this is one alternating fit, not a global optimum.
- `scripts/shirt_asset_tracks.py`: persistent CoTracker/depth distances instead
  of projected mesh distances; track births still need a material association.
- `scripts/shirt_asset_metric.py`: all-frame intrinsic edge calibration on the
  clean shirt topology with a non-flat canonical prior. This is **not** a verified
  manufacturing pattern or a reconstruction of hidden layers.

Each builder writes `asset.npz` and a `build.json` with per-frame evidence and
limitations. Shape selection on this recording is not held-out generalization.

## Runtime experiments

`shirt_asset` fixes the rest metric and topology. It requires unit material and
length scale settings; current-frame registration changes pose, not rest lengths.
The `flat_rest` compatibility flag keeps that metric fixed even for a non-flat
canonical asset (the asset metadata records its actual geometry).

`max_stretch=.02` with an asset bounds accepted edge lengths to ±2%.
`endpoint_projection` uses the same colored metric projection in Python and an
optional C++ acceleration. The native implementation is compiled locally with
`g++` into a source-hashed cache; absent compiler support falls back to NumPy.
The tested native and Python outputs were identical for the benchmark mesh.

IPC checks every accepted endpoint and continuous motion segments. Positive
`contact_clearance` uses a stricter accepted-state clearance than its CCD guard
to avoid equality/roundoff stalls. These checks do not prove correct hidden
material correspondence. Interpolated vertex paths are not exactly isometric.

`global_slide` is a bounded, current-two-view translation recovery that preserves
all edge lengths/self-distances and respects the table and held-point objective.
`held_iters` increases iterations based only on current grasp state.
`held_step` bounds each proposed solver displacement while held.

`constraint_qp` activates an experimental joint sparse quadratic step at the
first current-frame grasp and retains it thereafter. The initial unheld prefix
uses the original registration solver. The QP combines linearized material,
table and self-contact constraints with bounded incremental grasp targets;
exact edge/collision checks still gate accepted results. It needs the optional
`requirements-cloth-qp.txt` dependency. Its API follows the
[official OSQP Python documentation](https://osqp.org/docs/interfaces/python.html).
The one-frame diagnostic improved pickup progress; full-fold validation remains
required. Solver failure explicitly retains the previous mesh.

`contact_projection`, `constraint_restore`, and `compression_soft` are rejected
or unproven diagnostics, default off. In particular, `compression_soft` does not
satisfy the strict two-sided metric bound and must not be presented as an
inextensible cloth result.

The replay driver saves checkpoints every25frames and finishes/saves its current
frame on SIGINT/SIGTERM. Resuming cannot replace the asset or topology.

## Validation status

`tests/check_shirt_asset.py ASSET` checks causal live/replay parity, immutable
rest/sample IDs, ±2% accepted-state edge bounds, no endpoint crossings, and packet
ordering on a three-frame prefix. This is a software/constraint test, **not** a
full-fold accuracy test. Full failures and candidate comparisons are recorded in
`~/point_tracker/plans/ASTRA_Q30_CLOTH_LOG_1109.md`; no candidate is promoted yet.

## Numerical pickup regression
The initial QP could repeatedly solve but accept no movement because solver
residuals exceeded the stricter geometric acceptance tolerance. Inner lower-edge
and table margins fix that recorded stall; below-table effective contact goals
are projected to the table, while raw FK residuals remain reported.
`tests/check_cloth_qp_regression.py` checks deterministic current-packet replay,
actual progress, immutable material coordinates, strain bounds and crossings
on a trusted held-state checkpoint. This is not a whole-fold quality claim.

Full strict control120 satisfies ±2% edge bounds and has zero endpoint crossings
on453frames but worsens first-fold camera alignment. Candidate117's nonflat rest
metric is incompatible with the first-grasp separation at196 under its selected
material contacts (distance/geodesic1.123 while both arms remain closed). It is
rejected. Flat112's corresponding bound is0.977 and does not prove infeasibility.
Do not use released-grasp frames for this test.

`grasp_init_iters` raises the fitting budget only on a current-frame acquisition
event. A128-iteration first-acquisition diagnostic resolves the hem-target offset
to numerical precision while retaining3.72mm mesh residual and the same physical
checks; it takes19.4s. This avoids treating an initial wrong spatial match as a
new rest metric. Trial126 tests both acquisitions and the first fold; no success
is inferred from the one-frame result.

## Exact upper-length subproblem
`constraint_conic` together with `constraint_qp` uses Clarabel second-order cones
to impose each Euclidean edge upper bound inside the quadratic subproblem.
Lower length bounds remain linearized inner constraints; exact geometry and
continuous collision checks still gate accepted motion. This removes the large
post-solve shortening observed with linear upper bounds. Optional dependency
and source: `requirements-cloth-qp.txt`, `realpc/cloth_conic.py`,
[Clarabel Python API](https://clarabel.org/stable/python/getting_started_py/).
Limit solver threads to1 and set `OPENBLAS_NUM_THREADS=1` for these small meshes.
The175lift test with12steps of5mm takes6.27s, reaches the hard contact to numerical
precision (8.79mm maximum across all soft grasp-region points), and retains
edge ratios[.98200004,1.019000001]. Two-repeat regression passes exactly.
Trial127 starts before acquisition with this solver; it is not yet promoted.

## End-of-loop assessment
Trial127 is rejected after187frames: contact centres track, but the selected
material correspondence pulls the wrong surface and visible alignment worsens.
The completed453-frame120comparison is also a visible-fit failure despite
bounded strain and zero exported crossings. No candidate is promoted. The
current report contains full per-frame camera metrics and video; hidden layers
are not evaluated. Remaining work is joint rest-pattern/material registration
against multi-view evidence, not a claim that solver/contact accuracy alone
fixes the shirt. Cancelled dataset/QA/encoder work stays cancelled.
`cloth_grasp_feasibility.py` reproduces the necessary geodesic/contact bound,
checks only frames before release, and separates hard centres from soft regions.
