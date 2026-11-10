# Shared scene pipeline validation — 2026-11-10

## Configuration and causality

One config: `configs/scene_pipeline_v2.json`. SAM interval **10 processed frames**, plus a centrally scheduled recovery after repeated tracking failures (ordinary cooldown8, urgent cooldown3). All objects use the same scene labels and refresh event. A refresh counts once per tick even though both camera memories update.

`ScenePipeline.step(t, frames, q)` uses only the supplied current RGB-D/joints and previous state. Recorded joints are aligned by the fixed calibrated past lag. There is no backward smoothing or recorded-mask input. Offline calibration, library construction and evaluation masks are separate from runtime.

Live/replay checks on12-frame prefixes of sweep, hang and greycloth produced identical masks and maximum point difference5.96e-8m. The check disables recording video/joint access inside the live step. The latest recovery configuration also passed sweep replay. Native registration threads and seeds are fixed to avoid symmetric-object initialization changes between runs.

Four pipeline tests pass. Depth numerical/association, sparse-cloth mask and shirt constraint regression checks also pass. These checks establish implementation consistency, not physical tracking accuracy.

## Why N=10

Earlier full-episode cadence comparisons used the same camera-memory and mask-carry components before their integration into the single scene stream. They inform the choice; they are not claimed to be a full factorial benchmark of the final pipeline.

| Refresh | Cloth SAM frames /453 | Cloth mask IoU top/side | Hang SAM frames /499 | Hang mug pass top/side |
|---|---:|---:|---:|---:|
| N5 + events |92|.926/.942|102|92.6%/100%|
| N10 + events |49|.904/.920|53|91.6%/99.8%|
| N30 + events |21|.875/.883|27|91.0%/99.6%|
| Events only |6|.832/.849|15|91.0%/99.8%|

Cloth agreement is measured against every-frame causal SAM reference masks, not independent labels. Hang pass is visible silhouette IoU>=0.5 against Q28 evaluation masks. N10 reduces calls substantially relative to N5 while retaining more cloth coverage than N30/events. The shared N applies to every task; actual calls include recovery.

## Whole-episode final-pipeline results

Complete serial conversions:

| Task | Frames | SAM frames | SAM rate | Conversion seconds | Conversion fps |
|---|---:|---:|---:|---:|---:|
| hang |499|53|10.62%|123.9|4.03|
| sweep |849|106|12.49%|342.4|2.48|
| greycloth |453|47|10.38%|848.2|0.53|

Times include initialization, packet reads, tracking, saving and load verification on the local RTX4070TiSUPER workstation. Bulk concurrent-job timings will be reported separately; playback fps is unrelated.

## Limits and QA interpretation

Per-object tables separate independent Q28 masks (available only on selected episodes) from agreement with the pipeline's own propagated masks. High self-consistency cannot establish that an occluded mesh has the right pose. Contact sheets and both-camera videos remain necessary.

The cloth fold is approximate. The preceding single-sheet run had no exported endpoint intersection flags, but worst local extension exceeded200% and second-fold grasp lag reached roughly227mm. Do not interpret no-crossing flags as no stretch, finite-thickness collision safety, or correct hidden material correspondence. Dataset fold frames remain marked approximate and excluded from `cloth_valid` from tick150 onward. The final shared run has 0/453 exported endpoint crossing flags, 168.5% maximum local extension, and 277 mm maximum second-fold grasp lag. This is not a successful physical fold reconstruction.


The saved `cloth_valid` cutoff is a conservative dataset QA annotation for this recording, not a mask or decision fed into live tracking. Both adapters extract the same points. A reliable general runtime cloth-confidence gate is not yet established.

The implementation is causal, but the measured throughput is below the recorded camera frame rate. No claim of real-time30Hz extraction is made. Live callers must preserve timestamp/joint alignment; silently skipping packets changes the tested temporal behavior.

## Cloth comparison to the earlier two-layer52b run

Both use the same evaluation depth/masks. Frame0 coverage top/side changes85.2%/92.0% to94.9%/91.6%; depth3.84/6.51mm to1.77/2.21mm. First-fold median depth improves18.65/16.76mm to7.80/7.71mm. The second fold regresses: silhouette IoU is about.26/.24 and depth19.1/29.1mm. This release improves initialization and the first fold; it does not solve the second fold.

## Full independent rigid audit, episode0000

Pass = visible silhouette IoU>=0.5 against Q28 masks; no evaluation labels enter extraction.

| Object | Top pass | Side pass | Median IoU top/side |
|---|---:|---:|---:|
| Broom |97.5%|96.8%|.801/.813|
| Debris |90.6%|90.7%|.843/.857|
| Dustpan |94.8%|99.9%|.876/.881|
| Mug |92.8%|99.6%|.862/.831|
| Stand |98.2%|100%|.833/.885|

The tray has no independent Q28 label. Its dataset-wide table uses stream consistency and visual QA instead. Independent masks exist only on selected episodes; do not extrapolate these percentages to all50 sweep episodes.

Artifacts: `/home/howardji/point_tracker/reports/ASTRA_UNIFIED_1110/validation/` contains full both-camera videos for sweep/hang/greycloth, cloth clean-template/first-fit images, per-frame measurements and contact sheets. Dataset-wide outputs and the encoder audit will be added after the full conversion.
