"""Shared live/conversion RGB-D + calibrated joints -> scene points step."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .scene_masks import SceneMaskStream


def load_scene_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    if config.get('version') != 1:
        raise ValueError('Unsupported scene pipeline config')
    # Both live callers and conversion load the same offline assets, regardless
    # of their working directory. Relative paths belong to the config file.
    for section, key in (('cloth', 'shirt_asset'), ('segmentation', 'checkpoint')):
        value = config.get(section, {}).get(key)
        if value:
            asset = Path(os.path.expandvars(value)).expanduser()
            config[section][key] = str(asset if asset.is_absolute() else (path.parent / asset).resolve())
    cache = os.environ.get('REALPC_MODEL_CACHE')
    if cache and config.get('segmentation', {}).get('checkpoint'):
        config['segmentation']['checkpoint'] = str(Path(cache).expanduser().resolve() / 'sam2/sam2.1_hiera_small.pt')
    checkpoint = os.environ.get('REALPC_SAM_CHECKPOINT')
    if checkpoint:
        config.setdefault('segmentation', {})['checkpoint'] = str(Path(checkpoint).expanduser().resolve())
    return config


class ScenePipeline:
    def __init__(self, converter, config, cloth=False):
        # Stabilize symmetric-object cold hypotheses across episode order and
        # live/replay processes. Import native registration before setting its
        # OpenMP pool; otherwise a later library load escapes the limit.
        import torch
        from threadpoolctl import threadpool_limits
        from .objlib import o3d
        o3d()
        self.thread_limits = threadpool_limits(limits=config.get('cpu_threads',1))
        torch.set_num_threads(config.get('cpu_threads',1))
        np.random.seed(config.get('seed',0))
        torch.manual_seed(config.get('seed',0))
        self.config = copy.deepcopy(config)
        self.config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.converter = converter
        converter.seg = {}  # No recording-mask cache is consulted by this API.
        os.environ['REALPC_DEPTH_MESH'] = '1'
        for key, value in config.get('rigid_environment', {}).items():
            os.environ[key] = str(value)
        self.tick = -1
        self.timings = []
        self.cloth = None
        self.cot = None
        self.cloth_id = len(converter.objs) + 1 if cloth else None
        if cloth:
            from .cloth_causal import CausalClothTracker
            from .cloth_cotracker import CausalClothCoTracker
            self.cloth = CausalClothTracker.from_config(config['cloth'])
            if config['cloth'].get('cot_weight', 0):
                self.cot = CausalClothCoTracker()
        self.masks = SceneMaskStream(converter, config['segmentation'], self.cloth_id)
        self.last_packet = None

    def packet(self, frames, q):
        """Pure current-packet adapter: no camera/joint recording reads."""
        c = self.converter
        poses = c.arms.link_poses(q)
        robot = c.arms.points(q, poses=poses)
        packet = dict(robot=robot)
        cameras = {cam: c.cal.T_world_cam(cam) for cam in ('top', 'side')}
        A = np.eye(4)
        A[:3, :3] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        for arm in ('left', 'right'):
            G = poses[c.arms.bodies.index((arm, 'gripper'))]
            tipids = np.flatnonzero(np.isin(c.arms.body, [i for i, b in enumerate(c.arms.bodies)
                if b[0] == arm and b[1] in ('tip_left', 'tip_right')]))
            packet.update({'q_' + arm: np.asarray(q[arm]), 'G_' + arm: G,
                           'tips_' + arm: robot[tipids]})
            cameras[arm + '_wrist'] = G @ A @ np.asarray(c.ep.calib['X_' + arm])
        for cam in ('top', 'side', 'left_wrist', 'right_wrist'):
            if cam in frames:
                depth, image = frames[cam]
                scale = 640 / image.shape[1]
                size = (640, round(image.shape[0] * scale))
                K = c.ep.intrinsics(cam).copy()
                K[:2] *= scale
                packet[cam + '_rgb'] = cv2.resize(image, size)
                packet[cam + '_depth'] = cv2.resize(depth, size, interpolation=0)
                packet[cam + '_K'] = K
            else:
                # Optional wrist cameras contribute no observations when absent.
                packet[cam + '_rgb'] = np.zeros((480, 640, 3), np.uint8)
                packet[cam + '_depth'] = np.zeros((480, 640), np.float32)
                packet[cam + '_K'] = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
            packet[cam + '_T'] = cameras[cam]
            packet[cam + '_mask'] = np.zeros_like(packet[cam + '_depth'], bool)
        return packet

    def step(self, t, frames, q):
        """Exactly one current frame; q is supplied from calibrated past joints.

        Live callers and the recording adapter call this identical function.
        No optional fallback can read recorded joint arrays or future images.
        """
        if t != self.tick + 1:
            raise ValueError('Supply consecutive frames starting at zero')
        if not all(cam in frames for cam in ('top', 'side')):
            raise ValueError('The configured pipeline needs current top and side RGB-D')
        start = time.perf_counter()
        packet = self.packet(frames, q)
        initial_labels = self.masks.initialize(t, frames, q, packet) if t == 0 else None

        def provide(tick, views, robot, robot_z, object_z):
            self.masks.step(tick, frames, q, packet, views, robot, robot_z, object_z)

        out = self.converter.step(t, frames, q=q, labels=initial_labels, mask_provider=provide)
        if self.cloth is not None:
            from .cloth_causal import CAMS, at_image
            from .cloth_inputs import project
            for cam, labels in self.masks.labels.items():
                packet[cam + '_mask'] = labels == self.cloth_id
            if self.cot is not None:
                clean = self.cloth.views(packet)
                for cam in CAMS:
                    if cam in self.masks.labels:
                        track = self.cot.step(cam, t, packet[cam + '_rgb'], clean[cam]['M'])
                    else:
                        track = dict(uv=np.empty((0, 2), np.float32), visible=np.empty(0, bool),
                                     seed_uv=np.empty((0, 2), np.float32), birth=np.array(0))
                    packet.update({cam + '_cot_' + k: v for k, v in track.items()})
                packet['max_observed_tick'] = t
                packet['zero_lookahead'] = True
            X, metric, views = self.cloth.step(t, packet, iters=self.config['cloth']['iters'])
            samples = self.cloth.cm.samples()[:self.config.get('cloth_samples', 4000)]
            observed = np.zeros(len(samples), bool)
            for cam in ('top', 'side'):
                uv, z = project(samples, views[cam]['K'], views[cam]['T'])
                depth, inside = at_image(views[cam]['D'], uv)
                observed |= inside & (depth > .08) & (abs(z - depth) < .012)
            out.update(cloth_points=samples.astype(np.float32), cloth_observed=observed,
                       cloth_metrics=metric, cloth_vertices=X.copy(), cloth_faces=self.cloth.cm.F)
            from .cloth_quality import frame_quality
            quality = frame_quality(samples, metric, self.config)
            metric['quality'] = quality
            out.update(cloth_valid=quality['valid'], cloth_approximate=quality['approximate'],
                       cloth_quality=quality)
            if t == 0:
                from scipy.spatial import cKDTree
                P, C = self.converter.static
                keep = cKDTree(samples).query(P)[0] > .015
                self.converter.static = P[keep], C[keep]
        self.masks.after_tracking(out.get('cloth_metrics'))
        out['scene_labels'] = self.masks.labels
        out['sam_called'] = self.masks.last_was_sam
        out['max_observed_tick'] = t
        self.last_packet = packet
        self.tick = t
        self.timings.append(time.perf_counter() - start)
        return out

    def points(self, out):
        points = self.converter.points(out)
        if 'cloth_points' in out:
            points = np.concatenate([points, out['cloth_points']])
        return points

    def summary(self):
        seconds = sum(self.timings)
        return dict(config_sha256=self.config_hash, frames=len(self.timings), seconds=seconds,
                    fps=len(self.timings) / max(seconds, 1e-9), segmentation=self.masks.summary(),
                    causal=True, shared_live_conversion_step=True,
                    cloth_fold_approximate=self.cloth is not None)
