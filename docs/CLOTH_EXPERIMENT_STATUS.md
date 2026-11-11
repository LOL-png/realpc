# Selected cloth method — 2026-11-10

The default `configs/scene_pipeline_v2.json` selects candidate138: fixed offline single-sheet asset, frame0 registration, same causal per-frame tracker, metric/collision constraints, and bounded extra current-frame fitting when a held contact lags. SAM refreshes the shared scene stream every frame. The optional CoTracker grasp fallback remains off; its results were mixed.

See [the current validation](VALIDATION.md) for before/after camera metrics, physical/contact limits, checkpoint-continuation scope, fresh live/conversion parity, throughput and remaining errors. All cloth is approximate, including hidden layers. The next complete dataset conversion starts every episode from frame0 and audits the resulting trajectory.

[Historical experiments](CLOTH_HISTORY_1110.md) are retained for provenance. The full iteration log is `/home/howardji/point_tracker/plans/ASTRA_Q30_CLOTH_LOG_1109.md`.
