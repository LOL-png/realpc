# Cold-start correction found during dataset preflight

The first five full sweep conversions exposed reversed dustpan handles in episodes 20 and 40. A defined pose and silhouette IoU above 0.5 did not establish correct orientation; the both-camera visual review caught the error. The remaining dataset conversion was held, and all original outputs were retained.

When current masks provide enough depth, cold registration now uses samples of the actual mesh rather than the older partial scan used to construct its colour model. That scan can differ from the final, refitted mesh. The colour-only fallback remains available when masked depth is insufficient. No episode-index pose, future frame or hand-labelled runtime landmark is introduced.

First-frame diagnostic, top/side:

| Episode | Original dustpan IoU | Mesh-source IoU | Original median depth (mm) | Mesh-source median depth (mm) |
|---|---:|---:|---:|---:|
| sweep20 | 0.715 / 0.699 | 0.905 / 0.936 | 13.2 / 14.0 | 10.3 / 9.2 |
| sweep40 | 0.717 / 0.756 | 0.922 / 0.945 | 12.2 / 11.2 | 12.1 / 9.0 |

The handle orientation is visibly corrected in both views. Expanding the ICP shortlist to retain distinct orientations did not fix the problem alone and was reverted; the mesh-source change alone reproduces the correction. First-frame checks on sweep0/10/30 and hang show no large silhouette regression. These are diagnostics, not full-episode validation or independent accuracy labels.

All 18 repository tests pass. Actual 32-frame live/conversion checks pass for sweep, hang and cloth: exact masks and maximum point difference 5.96e-8 m. The cloth check uses an identical asset at a different absolute worktree path; its asset hash matches the canonical copy.

Corrected full sweep20 and sweep30 are complete and their both-camera sheets have been reviewed. The dustpan20 handle stays correctly oriented through release. Against the same saved masks, sampled median dustpan IoU changes from 0.699/0.786 to 0.913/0.889, with median per-frame depth errors 15.71/18.64 to 9.76/7.34 mm. Sweep30 remains broadly comparable. Broom occlusion errors and uncertain debris placement inside the box remain; this is not a claim of perfect all-object alignment.

The remaining corrected episodes are pending after two system-wide RAM exhaustion events. The second occurred with only two conversion jobs; the kernel snapshot contains 1,321 AWS processes belonging to a separate workload. Finished outputs and failure logs are preserved. Recovery must keep conversion concurrency bounded and audits serial, and control the external memory pressure before another long run.

Master and its original release tag remain unchanged until validation completes. Evidence is under `point_tracker/reports/REALPC_UNIFIED_1110/preflight_fix/`, `meshinit_preflight5/`, `meshinit_common_masks/` and `recovery_1111/`. The partial paired-frame comparison explicitly distinguishes newly visible depth pixels from the common comparison cohort. The remaining dataset conversion and encoder test have not started.
