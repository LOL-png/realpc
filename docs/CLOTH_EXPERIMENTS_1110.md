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
