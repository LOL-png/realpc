# How the points are extracted

The robot supplies two RGB-D views and joint angles. Before processing, we prepare the camera calibration, robot model and a mesh library for the rigid objects. These are reusable setup assets, not labels supplied during tracking. The shirt uses a clean rest template prepared offline, then registered to the first observed frame. Offline asset fitting can use multiple recorded views; runtime deformation cannot use future frames.

Every frame goes through one segmentation stream. The selected accuracy-first setting runs SAM on every frame for the whole scene. There is one SAM memory per camera, shared across the scene's objects; the cloth tracker does not run a separate SAM job. Larger intervals remain available: depth, optical flow and predicted geometry carry masks between scheduled or failure-triggered refreshes.

The rigid trackers fit their meshes to the current depth. Held objects also use gripper motion. The cloth tracker deforms the shirt template using both views, point tracks and grasp constraints. FK places the robot links. Fixed samples on these surfaces become the output point cloud, preserving sample identity over time.

Live inference calls `ScenePipeline.step` once with the current images and calibrated joints. Training conversion reads the next recorded packet and calls exactly the same function. It additionally saves poses, points, masks and QA metadata. It does not smooth backward, look ahead, or consume evaluation masks. Calibration and asset building can use offline recordings; that work is not repeated inside the causal step.

The shared SAM interval is 1 frame. The completed rigid benchmark measured 0.87 fps for sweep and 1.25 fps for hang. The fresh bounded cloth reference measured 0.068 fps in a concurrent validation run; the historical 0.56 fps cloth result belongs to the older, excessively stretching tracker. These are extraction rates, not playback rates or real-time performance. See `VALIDATION.md` for the cadence comparison and `CLOTH_EXPERIMENT_STATUS.md` for the current shirt audit. Hidden material and fine wrinkles remain uncertain even when visible masks and depth agree.
