# Cold-start correction found during dataset preflight

The first five full sweep conversions exposed reversed dustpan handles in episodes 20 and 40. A defined pose and silhouette IoU above 0.5 did not establish correct orientation; the both-camera visual review caught the error. The remaining dataset conversion was held, and all original outputs were retained.

When current masks provide enough depth, cold registration now uses samples of the actual mesh rather than the older partial scan used to construct its colour model. That scan can differ from the final, refitted mesh. The colour-only fallback remains available when masked depth is insufficient. No episode-index pose, future frame or hand-labelled runtime landmark is introduced.

First-frame diagnostic, top/side:

| Episode | Original dustpan IoU | Mesh-source IoU | Original median depth (mm) | Mesh-source median depth (mm) |
|---|---:|---:|---:|---:|
| sweep20 | 0.715 / 0.699 | 0.905 / 0.936 | 13.2 / 14.0 | 10.3 / 9.2 |
| sweep40 | 0.717 / 0.756 | 0.922 / 0.945 | 12.2 / 11.2 | 12.1 / 9.0 |

The handle orientation is visibly corrected in both views. Expanding the ICP shortlist to retain distinct orientations did not fix the problem alone and was reverted; the mesh-source change alone reproduces the correction. First-frame checks on sweep0/10/30 and hang show no large silhouette regression. These are diagnostics, not full-episode validation or independent accuracy labels.

All 18 repository tests pass. Five full sweep reruns and actual live/conversion parity checks are pending. Master and its original release tag remain unchanged until those checks complete. Evidence is under `point_tracker/reports/REALPC_UNIFIED_1110/preflight_fix/`; corrected full-run evidence will be under `meshinit_preflight5/`.
