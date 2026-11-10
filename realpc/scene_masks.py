"""One causal segmentation stream, one SAM memory and one refresh schedule.

Mesh prediction transports rigid identities; depth/flow transports cloth masks.
All trackers consume the resulting instance-label images, including skipped-SAM
frames. Neither carrier owns a SAM model or chooses its own refresh interval.
"""
import time
import cv2
import numpy as np

from .online_masks import ObjectMaskStream, OnlineObjectSegmenter, render_seeds
from .depth_masks import associate_depth


class SceneMaskStream:
    def __init__(self, converter, config, cloth_id=None):
        self.converter = converter
        self.config = dict(config)
        self.interval = int(config['interval'])
        if self.interval < 0:
            raise ValueError('interval must be nonnegative')
        self.cloth_id = cloth_id
        self.carry = None
        if cloth_id is not None:
            from .cloth_sparse_masks import SparseClothMasks
            # prepare/finish only: this carrier never calls its SAM backend.
            self.carry = SparseClothMasks('', interval=0, fast_neighbors=True,health=config['cloth_health'])
        self.model = None
        self.memory = {}
        self.labels = {}
        self.events = []
        self.rows = []
        self.tick = -1
        self.last_sam = -100000
        self.pending = []
        self.bad = {}
        self.health = {}
        self.last_was_sam = False
        self.cloth_fit_bad = 0
        self.cloth_failure_latched = False
        self.cloth_recovery_pending = False

    def _refresh(self, t, frames, q, proposal, reason):
        start = time.perf_counter()
        c = self.converter
        if self.model is None:
            from sam2.build_sam import build_sam2_video_predictor
            self.model = build_sam2_video_predictor(
                'configs/sam2.1/sam2.1_hiera_s.yaml', self.config['checkpoint'], device='cuda')
        labels = {}
        for cam in ('top', 'side'):
            if cam not in frames:
                continue
            depth, image = frames[cam]
            image = cv2.resize(image, (640, round(image.shape[0] * 640 / image.shape[1])))
            if cam not in self.memory:
                seeds, zbufs, ids = render_seeds(c, cam, image.shape[:2], q)
                colors = {i + 1: c.trk.lib_points[o.name][1] for i, o in enumerate(c.objs)}
                for oid in range(1,len(c.objs)+1):
                    seeds.setdefault(oid,np.zeros(image.shape[:2],bool))
                cloth_ids = ()
                if proposal is not None:
                    seed = proposal['initial'] if cam == 'top' else (
                        proposal['cross_seeds'][cam] if proposal['diagnostics'][cam]['cross_failed']
                        else proposal['masks'][cam] | proposal['cross_seeds'][cam])
                    seeds[self.cloth_id] = seed
                    zbufs.append(np.where(seed, cv2.resize(depth, (seed.shape[1], seed.shape[0]), interpolation=0), 0))
                    ids.append(self.cloth_id)
                    cloth_ids = (self.cloth_id,)
                    if not c.objs:
                        seeds = {self.cloth_id: seed}
                self.memory[cam] = ObjectMaskStream(
                    self.model, image, seeds, zbufs, ids, depth,
                    cold=c.cold, reference_colors=colors, cloth_ids=cloth_ids)
            labels[cam] = self.memory[cam].step(image)
        reprompts = []
        if t and c.objs:
            # Reuse the current-frame geometric recovery helper on the SAME
            # model and camera states. It does not create a second stream.
            helper = OnlineObjectSegmenter(c, self.model, reset_tracker_on_seed=False)
            helper.memory = self.memory
            reprompts = helper.reacquire_empty(frames, q, labels)
        self.last_sam = t
        self.events.append(dict(tick=t, reason=reason, camera_calls=len(labels),
                                objects=[o.name for o in c.objs] + (['cloth'] if self.cloth_id else []),
                                failures=list(self.pending) + (['cloth_fit'] if self.cloth_recovery_pending else []), reprompts=reprompts,
                                seconds=time.perf_counter() - start))
        self.pending = []
        self.bad = {}
        self.cloth_recovery_pending = False
        return labels

    def initialize(self, t, frames, q, packet):
        if self.tick != -1 or t != 0:
            raise ValueError('Initialize from the first current packet only')
        c = self.converter
        if c.trk is None:
            c._init(t, frames, q=q)
        proposal = self.carry.prepare(t, packet) if self.carry else None
        self.labels = self._refresh(t, frames, q, proposal, 'initialize')
        if self.carry:
            masks = {cam: self.labels[cam] == self.cloth_id for cam in self.carry.cameras}
            self.carry.finish(t, packet, proposal, masks)
        self.tick = t
        self.last_was_sam = True
        self.rows.append(dict(tick=t, sam=True, max_observed_tick=t))
        # Geometric cold poses only seeded masks. Register emitted rigid poses
        # from those current masks, matching the validated depth pipeline.
        if c.objs:
            c.trk = None
        return self.labels

    def step(self, t, frames, q, packet, views, robot, robot_z, object_z):
        if t == 0 and self.tick == 0:
            for v in views:
                v.labels = self.labels[v.name]
            return
        if t != self.tick + 1:
            raise ValueError('Scene masks require consecutive current packets')
        c = self.converter
        associate_depth(c.trk, views, robot_z, object_z, self.config['rigid_association'])
        proposal = self.carry.prepare(t, packet) if self.carry else None
        cloth_failed = bool(proposal and proposal['bad'])
        failure = bool(self.pending) or cloth_failed or self.cloth_recovery_pending
        periodic = self.interval > 0 and t - self.last_sam >= self.interval
        urgent=any(d.get('visible',0)>=self.config['min_visible_pixels']
                   and d.get('support',0)<self.config.get('urgent_support_pixels',5)
                   and d.get('free_fraction',0)>self.config.get('urgent_free_fraction',.9)
                   for name,d in self.health.items() if name in self.pending)
        if proposal:
            urgent |= any(d['area']<self.config['cloth_health']['min_area']
                          or d['area_ratio']<self.config.get('urgent_area_ratio',.3)
                          or (d['cross_support']>self.config['cloth_health']['min_cross_support']
                              and d['cross_recall']<self.config.get('urgent_cross_recall',.25))
                          for d in proposal['diagnostics'].values())
        cooldown=self.config.get('urgent_cooldown',3) if urgent else self.config['failure_cooldown']
        refresh = periodic or (failure and t - self.last_sam >= cooldown)
        if refresh:
            for name in self.pending:
                st = c.trk.state[name]
                st.lost = max(st.lost, 2)
                if st.held is not None:
                    st.T = robot[st.held]['G'] @ st.T_grip_obj
            labels = self._refresh(t, frames, q, proposal, 'failure' if failure else 'periodic')
            for v in views:
                v.labels = labels[v.name]
                v.depth_generated = False
                v.associated_points = {}
                v.depth_appearance = {}
            c.trk.cfg['sil_refine'] = True
        else:
            labels = {}
            for v in views:
                h, w = packet[v.name + '_depth'].shape
                lab = cv2.resize(v.labels, (w, h), interpolation=0)
                if proposal:
                    mask = proposal['masks'][v.name]
                    lab[mask & (lab == 0)] = self.cloth_id
                    # Rigid consumers and cloth consumers see the same labels.
                    v.labels = cv2.resize(lab, (v.R.W, v.R.H), interpolation=0)
                labels[v.name] = lab
        if self.carry:
            masks = {cam: labels[cam] == self.cloth_id for cam in self.carry.cameras}
            self.carry.finish(t, packet, proposal, masks)
        self.labels = {cam: lab.copy() for cam, lab in labels.items()}
        self.tick = t
        self.last_was_sam = refresh
        self.rows.append(dict(tick=t, sam=refresh, max_observed_tick=t,
                              cloth_health=proposal['diagnostics'] if proposal else {}))

    def after_tracking(self, cloth_metrics=None):
        c = self.converter
        if self.last_was_sam:
            return
        if cloth_metrics is not None:
            cameras=list(cloth_metrics.get('cameras',{}).values())
            bad=bool(cameras) and all(
                v.get('surface_coverage_10mm',1)<self.config.get('cloth_fit_min_coverage',.5)
                and v.get('surface_to_mesh_med_mm',0)>self.config.get('cloth_fit_max_median_mm',25)
                for v in cameras)
            self.cloth_fit_bad=self.cloth_fit_bad+1 if bad else 0
            if not bad:self.cloth_failure_latched=False
            if self.cloth_fit_bad>=self.config['failure_frames'] and not self.cloth_failure_latched:
                # Retry once per failure episode, not every few frames forever
                # when the deformable solver cannot fit otherwise-good masks.
                self.cloth_recovery_pending=True
                self.cloth_failure_latched=True
        for name, d in getattr(c.trk, 'depth_diagnostics', {}).items():
            st = c.trk.state[name]
            if getattr(c.trk.objs[name], 'passive_fixture', False):
                continue
            visible = d['visible'] >= self.config['min_visible_pixels']
            residual, count = st.last_res
            bad = visible and (d['free_fraction'] > self.config['max_free_fraction']
                               or d['support'] < self.config['min_support_pixels'])
            bad |= any(x < self.config['min_view_iou'] for x in d.get('view_iou', {}).values())
            bad |= visible and count >= 40 and np.isfinite(residual) and residual > self.config['max_residual_m']
            self.health[name] = dict(d, residual_mm=float(residual * 1000) if np.isfinite(residual) else None)
            self.bad[name] = self.bad.get(name, 0) + 1 if bad else 0
            if self.bad[name] >= self.config['failure_frames'] and name not in self.pending:
                self.pending.append(name)

    def summary(self):
        return dict(frames=len(self.rows), interval=self.interval, sam_frames=len(self.events),
                    sam_frame_rate=len(self.events) / max(len(self.rows), 1),
                    sam_camera_calls=sum(e['camera_calls'] for e in self.events),
                    events=self.events, causal=True, one_shared_stream=True)
