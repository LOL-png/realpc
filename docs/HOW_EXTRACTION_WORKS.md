# How extraction works

Choose `accurate` (SAM each frame) or `fast` (SAM at initialization/failure).
Both use the current objects_v3 library, calibration v3, robot model and shirt
rest template. These meshes and calibration are prepared offline; their reuse is
analogous to reusing a robot URDF, not access to future camera frames at runtime.

Each step receives current top/side RGB-D, optional wrist RGB-D and calibrated
joints. One shared scene segmentation stream supplies all object masks. Robot
geometry comes from FK. Rigid meshes are fit to visible depth using previous-pose
and held-gripper predictions. Cloth uses the current deformable template, causal
CoTracker tracks, depth/flow and bounded physical/contact constraints. Between SAM
events, fast associates current depth with predicted meshes instead of encoding
another SAM frame. It propagates cloth masks without a separate SAM schedule.

Live inference and dataset conversion call the identical `ScenePipeline.step`.
The live adapter acquires current packets; conversion reads packets sequentially
from recordings with the calibrated fixed past joint lag. Neither tracker reads
future frames or evaluation masks. Fixed material/link samples give persistent
point identities; conversion stores compact samples/poses, while the live API
returns points and quality flags directly.

The four public commands, input packet contract, measured benchmark rates and
limitations are in the [README](../README.md). The [quality policy](CLOTH_QUALITY_FLAGS.md)
marks every shirt estimate approximate and records per-frame validity separately.
Hidden surfaces and material identity remain uncertain even when visible depth
and silhouettes agree. Exact current-fast speed/accuracy is not yet measured.
