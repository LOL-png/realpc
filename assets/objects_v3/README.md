# Selected rigid mesh library

Both public presets load this library. Sweep contains broom, dustpan, debris and
the static box/tray. Hang contains a mug with an asymmetric ear-shaped handle and
a stand with four separate pegs at two heights. Current camera calibration is
`configs/calib_yam_rig_v3.json`; robot geometry is `robot_models/yam_v1/`.

Each object has a PLY mesh, JSON metadata and construction/appearance points.
The latest cold initializer registers actual mesh samples when current masked
depth is sufficient; construction colors remain a fallback. Static and held
objects remain subject to visible-depth checks. Offline asset fitting may use
multiple views/frames, but online poses use only current/past inputs.

The shared runtime produces its own segmentation. Evaluation masks must never be
supplied as tracker inputs. Thin bristles, hidden surfaces, fast releases and
occlusion remain imperfect; a defined pose is not proof of alignment. See the
repository README and validation reports for measured scope. Historical assets
are preserved on archive branches and are not loaded by either public preset.
