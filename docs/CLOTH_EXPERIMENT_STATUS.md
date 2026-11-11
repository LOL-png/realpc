# Current cloth validation — 2026-11-10

The complete bounded candidate is `configs/experimental/scene_cloth136.json`. It has **not yet replaced the default config or master**. A fresh unified run completed all 453 frames, generating current SAM masks and zero-lookahead CoTracker observations inside the same `ScenePipeline.step` used live.

The fixed offline asset is `assets/cloth/tshirt_candidate132/asset.npz`: one sheet with the shirt outline, 1,556 vertices and 2,975 faces. Its dimensions were fitted using all-frame/two-view offline evidence; a current-frame registration initializes its pose and wrinkles. Runtime never reads an offline pose sequence. Rest geometry and material sample identities remain fixed.

| Matched phase | Previous N1 depth, top/side mm | Candidate depth, top/side mm | Previous IoU | Candidate IoU |
|---|---:|---:|---:|---:|
| Before lift | 1.93 / 2.18 | 6.26 / 3.93 | .866 / .893 | .837 / .890 |
| First fold | 8.04 / 10.72 | 4.59 / 4.14 | .773 / .847 | .835 / .863 |
| Second fold | 7.02 / 8.01 | 5.97 / 4.44 | .715 / .634 | .835 / .819 |

Depth entries are medians of per-frame visible-overlap medians, against common masks/depth. Coverage and P90 tails are reported separately; some top-view tails regress. These are visible-surface consistency measurements, not hidden material ground truth.

An independent audit of all exported meshes found zero triangle crossings, maximum extension 1.90% and compression 1.81%. All 4,000 exported material tracks exactly match fixed barycentric mesh samples over all frames. The assumed grasp centres briefly lag FK by 36.6 mm and 12.9 mm at frames 348–349; sampled near-zero contact errors elsewhere must not be presented as a full-episode bound. All cloth remains approximate.

Actual live/replay checks use a constructor with only three static setup JSON files. Thirty-two-frame checks pass for cloth, hang and sweep with recording access blocked during the step, exact masks and maximum point difference 5.96e-8 m. One earlier hang repeat differed by 0.18 mm at frame 2; the fresh conversion and subsequent repeat matched, and the original failure remains in the log. This does not establish bitwise determinism for every native numerical path.

N1 conversion took 6,699 s (0.068 fps) while other validation jobs ran. This bounded cloth solver is much slower than the historical unconstrained version. The N1/N2/N3/N10 cadence recheck is finishing. Two small diagnostic continuations compare persistent-track grasp selection and extra current-frame fitting when a held contact lags; neither has been promoted.

Full video, per-frame CSV, physical audit, tracking diagnostics and live checks:
`/home/howardji/point_tracker/reports/ASTRA_Q30_1110/unified136/`.

Earlier two-layer/single-sheet experiments remain in [the historical log](CLOTH_HISTORY_1110.md). Dataset conversion and encoder evaluation wait for the selected configuration, documented local release and videos.
