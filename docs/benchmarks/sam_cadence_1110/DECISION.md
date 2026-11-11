# Decision: one shared SAM interval of 1

Select **every processed frame** for every object, for both live inference and conversion. The predeclared accuracy criterion is equal-object, equal-camera mean silhouette IoU, without a speed term. Scores: N1 **0.8143**, N3 **0.8054**, N10 **0.7834**, N2 **0.7811**. Box consistency is excluded because there is no independent box mask. Aggregate IoU>=0.5 pass rates are 96.76%, 96.83%, 93.19%, 92.09% respectively: N1 and N3 are effectively tied on this thresholded diagnostic. This is a measured trade-off, not a universally best rate.

N1 improves mug, stand and debris overlap. **N3 is better for cloth alone**, with full-episode IoU .794/.816 versus N1 .783/.807 and better fold depth. N10 retains the best broom top-view overlap. Frequent SAM also enables more silhouette refinement in the existing tracker; the experiment measures this whole refresh policy, not an isolated segmentation change.

Visual review confirms a broom pose regression around release (frames 587–599) and a dustpan pose error at frame 750 with N1. N2 loses the debris pose late in the sweep. Mug improvement is visible around frame 387. These failures are retained in the contact sheets. N1 also worsens mug-side median depth from 6.14 to 7.86 mm and broom depth from 7.07/9.45 to 8.20/10.70 mm. No object-specific cadence exception or tracker fix is hidden in this comparison.

For the cloth, N1 second-fold median depth improves from N10 **19.09/29.10 to 7.02/8.01 mm**, and maximum grasp error from 276.7 to 33.9 mm. N3 is slightly better on visible depth (6.74/7.22 mm). **The physical shirt is still not solved:** maximum exported local extension is 168.5% for N1 (242.3% for N3). Zero tracker crossing flags do not establish collision safety or correct hidden layers. The new all-frame shirt-asset experiments have not been promoted by this rate test.

## Evidence and reproducibility

All rates use full causal episodes: hang0 499 frames, sweep0 849, greycloth0 453. One validation episode per task; this is not a claim about all 52 episodes or unseen generalization. Only the SAM interval changes. Runtime/assets hashes and reference-visible cohorts match across all rates. Robot transforms, joints and static points are bit-identical across rates. Each conversion verifies its output with realpc.dataset.load.

Rigid accuracy uses fixed Q28 masks for evaluation only. Cloth uses the same cached mask/depth proxy for every rate; this is not independent material ground truth. Box overlap is agreement with a fixed N10 mask cache, supplemented by raw-depth checks and visual review. Recovery thresholds are unchanged. N1/N2/N3 SAM counts equal ceil(frames/N); N10 includes extra recovery refreshes.

Conversions ran serially on the local RTX4070TiSUPER. FPS includes initialization, reading, tracking, saving and load verification; it excludes the offline audit and video rendering. N10 cloth reuses the completed full control after exact source/config/asset and output-checksum verification; the other conversions are fresh. Throughput and alignment tables are in README.md; per-view pass rates, actual SAM fractions and depth are in per_object_alignment.csv and summary.json. Physical audit: cloth_physics.json. Full per-frame measurements: /home/howardji/data_ext/astra_accuracy_1110/nN/audit/.

Reproduce using scripts/benchmark_scene_cadence.py, then scripts/summarize_scene_cadence.py. Review before setting --selected; the summary verifies that all configurations differ only in interval. scripts/cadence_cloth_physics.py and scripts/plot_cadence_cloth.py reproduce the cloth diagnostics.

## Artifacts

Report directory: /home/howardji/point_tracker/reports/SAM_ACCURACY_1110/. Contact sheets compare all rates at identical frames: sweep_0587_all_rates.jpg, sweep_0599_all_rates.jpg, sweep_0637_all_rates.jpg, sweep_0750_all_rates.jpg, hang_0387_all_rates.jpg. cloth_per_frame.png plots unsmoothed camera metrics. cloth_n1/both_cameras.mp4 is the selected-policy cloth video; cloth_n3 and cloth_n10 provide comparisons.

The cancelled 52-episode dataset conversion stays cancelled. The 27 finished outputs and logs are unchanged; they are not relabelled as N1 outputs. No encoder run, remote upload or push is part of this work.

## Verification

Four pipeline tests pass (no skips). Actual 12-frame live/replay checks pass for all three tasks with recording reads blocked inside the live step: masks exactly match, maximum point difference 5.96e-8m, and 12 SAM refreshes each. All 12 converted files passed load verification. Selected shared config exactly matches the measured N1 config. Details: verification.json.
