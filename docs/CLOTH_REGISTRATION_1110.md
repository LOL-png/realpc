# Flexible cloth initialization and sparse SAM experiments

Frame0 registration is separate from tracking: `init_registration_iters` enables a current-frame fit with relaxed ARAP/strain/bending, a wider safe outline warp, top-contour coverage and joint top/side depth. CoTracker material offsets are bound after registration. Original tracking weights, rest metric and sample identities are retained; initial fitting does not create velocity.

The improved two-layer initialization101 reaches exact-frame0 coverage94.3/92.2%, depth1.96/2.33mm (top/side), versus85.2/92.0% and3.84/6.51mm for52b. Worst local strain remains large. First-lift tracking still fails in100/102/103; initialization is not a full-fold solution.

`feasible_predict` collision-filters coherent FK/material prediction before Newton fitting. `contact_clearance` optionally enforces finite separation in both IPC barrier and CCD. Initialization blends toward the clean template only as needed to establish that clearance, without resetting rest lengths. These options are experimental; no master/dataset promotion.

`SparseClothMasks.step(t, current_packet)` ignores cached masks. It tracks current depth using its past surface/FK prediction and optical-flow silhouette transport. Current cross-camera elevated depth supplies a failure detector. SAM receives only initialization, periodic refreshes or failure frames; video memory sees only those selected images. `interval=0` is events only. Two-view benchmark uses top+side; wrist masks are disabled rather than silently reusing per-frame SAM.

`cloth_sparse_mask_bench.py` records actual SAM frames and camera requests, mask-only compute speed, wall time including cache IO, and agreement with the every-frame SAM reference. This is not mesh alignment or independent ground truth. Stateless-image pilots drifted in the second fold and are rejected. The video-memory benchmark is ongoing; no interval winner is claimed yet.

Tests: clean topology, IPC crossing/finite clearance, fixed rest and sample IDs, exact live/conversion registration prefix, cached-mask independence and causal packet order.

Evidence/log: `/home/howardji/point_tracker/plans/ASTRA_Q30_CLOTH_LOG_1109.md`; frame0 report: `/home/howardji/point_tracker/reports/ASTRA_Q30_1110/frame0_registration/README.md`.
