# How the points are extracted

The robot supplies two RGB-D views and joint angles. Before processing, we prepare the camera calibration, robot model and a mesh library for the rigid objects. These are reusable setup assets, not labels supplied during tracking. The shirt starts from a clean template fitted to the first observed frame.

Every frame goes through one segmentation stream. Most frames carry the masks forward with depth, optical flow and predicted object geometry. SAM refreshes the whole scene at the shared interval, or when tracking checks fail. There is one SAM memory per camera, shared across the scene's objects; the cloth tracker does not run a separate SAM job.

The rigid trackers fit their meshes to the current depth. Held objects also use gripper motion. The cloth tracker deforms the shirt template using both views, point tracks and grasp constraints. FK places the robot links. Fixed samples on these surfaces become the output point cloud, preserving sample identity over time.

Live inference calls `ScenePipeline.step` once with the current images and calibrated joints. Training conversion reads the next recorded packet and calls exactly the same function. It additionally saves poses, points, masks and QA metadata. It does not smooth backward, look ahead, or consume evaluation masks. Calibration and asset building can use offline recordings; that work is not repeated inside the causal step.

The candidate shared SAM interval is10 frames plus failures. Whole-episode measurements are being verified in `VALIDATION.md`. Cloth folds are still approximate, particularly hidden layers and the second grasp; the dataset records that limitation rather than treating them as verified material tracks.
