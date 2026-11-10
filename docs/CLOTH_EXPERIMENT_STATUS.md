# Clean shirt tracker: experimental, not accepted

Work remains isolated on `astra-cloth`. Master, dataset conversion, and encoder evaluation are paused.

The clean garment has flat front/back panels sewn along sides/shoulders, with four open boundaries (neck, two cuffs, hem). Frame 0 supplies calibration-fixed depth/masks to fit size and initial shape. Rest geometry and barycentric sample identities are then frozen. `CausalClothTracker.step` consumes one current packet, retained history, FK, and zero-lookahead CoTracker observations. Conversion uses the same `from_config` factory. No future cloth pose, fold animation, or hand mask enters runtime.

Recent corrections: actual tilted support plane instead of constant height; table-feasible pinch targets preserving captured layer spacing; feature bindings established after initial fitting; sewn-seam-safe bending; native dynamics starts from the fitted mesh; correctly scaled extension barrier. Optional hidden-back-panel and visible-grasp priors are still being compared.

## Validation and limitations

- `tests/check_shirt.py`: topology, contact CCD, extension filter, tilted support, and 2D-ray regressions.
- `tests/check_flat_shirt.py`: standard joint solver preserves rest/samples and exactly matches live/conversion on the tested three-frame prefix.
- `tests/check_flat_shirt.py --dynamics`: currently FAILS exact replay equality (up to 4.79 mm at frame 0 in the tested configuration). Keep this test as a guard; do not claim the native backend is validated.
- Native MuJoCo IPC is experimental: some configurations exhaust its contact arena; exact state resume is not established. Resumed runs are diagnostics. Accepted output must be rerun uninterrupted from frame 0.
- The complete clean-template runs so far have unacceptable fold alignment or strain. Zero detected crossings alone does not establish correct material tracking. No production cloth selection or completion claim has been made.

Primary log: `/home/howardji/point_tracker/plans/ASTRA_Q30_CLOTH_LOG_1109.md`.
Matching complete diagnostic assets: `/home/howardji/point_tracker/reports/ASTRA_Q30_1110/flat_template_review/` (shirt52b; explicitly not accepted).
Latest short iterations/checkpoints: `/home/howardji/data_ext/astra_cloth/shirt67` through `shirt75`.

Further diagnostic iterations76–84 test intrinsic grasp-span inconsistency and automatic garment-corner registration. Registration at closure can be incompatible with the retained layered pose; neither solver has yet produced an accepted fold. A confirmed initialization bug is fixed: t0 registration displacement no longer becomes t1 velocity. The corresponding standard live/conversion prefix regression passes. Optional `init_landmarks` now constrains only the two automatically fitted hem corners at frame0, leaving uncertain neck/cuff matches free.

Trial84 completes the first lift with corner registration (at175: grasp centers1.28mm, worst edge distortion2.89%) but surface alignment still fails (top17.16/side8.38mm median). Trial86 is a fresh uninterrupted453-frame audit with explicit initialization gain1 and tracking gain5. It is still a candidate, not a production choice; its two folds must be inspected.
