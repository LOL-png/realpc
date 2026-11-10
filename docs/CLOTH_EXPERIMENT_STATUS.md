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
