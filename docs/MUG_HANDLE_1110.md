# Ear-shaped mug handle, 11-10

Current assets use an asymmetric natural-cubic spline with a rounded upper bulge and tapered lower lobe. The rounded-rectangle section tapers to84.7% toward the bottom. Cup geometry and four-peg stand are unchanged. Serialized handle is watertight and intersection-free.

Offline fitting uses14 top/side and6 body-registered wrist RGB-D observations. Runtime uses the same causal depth tracker, SAM only for initialization/recovery. No new calibration or inference logic.

Matched499-frame pass rate previousD→ear: mug top91.8→91.0%, side100→99.8%; stand top95.4→93.6%, side100→100%. Mug side median IoU improves.812→.848. Handle-only side IoU improves.514→.641, but wrist depth is mixed. Shape correction is not an overall tracking gain.

Replay105.5s (~4.7fps including setup),30 SAM camera calls. Loader and saved-mesh geometry checks pass. No dataset_v2 overwrite.

[Full report, four views, overlays and video](/home/howardji/point_tracker/reports/ASTRA_Q29_1109/mug_ear_review/README.md).

Rebuild: `scripts/build_mug_handle_candidate.py --params <handle_params.json> --source assets/objects_v3 --out <separate candidate library>`; the profile field preserves support for prior straight-strap candidates.
