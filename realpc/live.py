"""Construct the shared scene step from offline setup, without a recording."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class SceneSetup:
    """Only reusable setup metadata; no recorded frames, joints or poses.

    The directory contains meta.json (camera intrinsics/depth scales),
    calib.json (robot/wrist/table setup), and table_region.json. These can be
    exported from the calibrated recording setup; no video/low_dim/fused data
    is needed or read here.
    """
    def __init__(self, directory, task, name='live'):
        self.path = str(Path(directory).resolve())
        self.task, self.name = task, name
        self.meta = json.loads((Path(self.path) / 'meta.json').read_text())
        self.calib = json.loads((Path(self.path) / 'calib.json').read_text())
        # The depth/mesh initializer consumes this offline workspace region.
        json.loads((Path(self.path) / 'table_region.json').read_text())

    def intrinsics(self, camera='top'):
        return np.asarray(self.meta['intrinsics'][camera], float)

    def table_plane(self):
        return np.asarray(self.calib['table_n'], float), float(self.calib['table_d'])

    def T_world_base(self):
        return {'left': np.eye(4), 'right': np.asarray(self.calib['T_lb_rb'], float)}


def build_live_pipeline(setup, config, *, cloth=False, library=None,
                        calibration=None, robot_config=None):
    """Return ScenePipeline; callers supply current RGB-D and aligned joints."""
    from .calib_rig import RigCalib
    from .config import load_config, build_arms
    from .diffcalib import TorchArm
    from .scene_pipeline import ScenePipeline, load_scene_config
    from .v4 import StreamConverter

    cfg = load_config(robot_config or ROOT / 'configs/yam_bimanual.json')
    robot = cfg['robot']
    tarm = TorchArm(str(Path(cfg['_dir']) / robot['urdf']),
                   robot['arm_joints'], robot['finger_joints'])
    cal = RigCalib.load(calibration or ROOT / 'configs/calib_yam_rig_v3.json')
    if cal.lags:
        raise ValueError('Live setup requires a fixed calibrated lag, not recording-specific lag curves')
    converter = StreamConverter(setup, cal, tarm, build_arms(cfg, setup.calib),
                               str(library or ROOT / 'assets/objects_v3' / setup.task), seg_dir=None)
    settings = load_scene_config(config) if not isinstance(config, dict) else config
    return ScenePipeline(converter, settings, cloth=cloth)
