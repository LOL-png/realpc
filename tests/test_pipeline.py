"""Tests. The FK test needs no data; the episode tests need REALPC_TEST_EPISODE=<episode dir> (skipped otherwise).

  python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from realpc.config import build_arms, load_config, table_plane, tracker_config  # noqa: E402
from realpc.robot import URDFRobot  # noqa: E402

CFG = os.path.join(ROOT, "configs", "yam_bimanual.json")
EP = os.environ.get("REALPC_TEST_EPISODE")


def test_fk_home_and_chain():
    r = URDFRobot(os.path.join(ROOT, "robot_models", "yam_v1", "yam.urdf"))
    T = r.fk({})
    assert set(T) >= {"base", "link1", "link5", "gripper", "tip_left", "tip_right"}
    # rotating joint1 (axis = base z up to the URDF's rounded 1.5708) keeps the gripper's height and distance from the base z axis
    T2 = r.fk({"joint1": 0.7})
    g1, g2 = T["gripper"][:3, 3], T2["gripper"][:3, 3]
    assert abs(g1[2] - g2[2]) < 1e-4
    assert abs(np.linalg.norm(g1[:2]) - np.linalg.norm(g2[:2])) < 1e-4


@pytest.mark.skipif(not EP, reason="REALPC_TEST_EPISODE not set")
def test_fk_matches_recorded_eef():
    from realpc.io import Episode
    ep = Episode(EP)
    arms = build_arms(load_config(CFG), ep.calib)
    low = ep.low
    for tick in np.linspace(0, len(low["t_wall"]) - 1, 20).astype(int):
        q = {"left": low["joint_pos_left"][tick], "right": low["joint_pos_right"][tick]}
        m = arms.robot.fk(arms.qmap(q["left"]))
        assert np.linalg.norm(m["gripper"][:3, 3] - low["eef_pos_left"][tick]) < 1e-4


@pytest.mark.skipif(not EP, reason="REALPC_TEST_EPISODE not set")
def test_batch_equals_streaming():
    from realpc.io import Episode
    from realpc.pipeline import convert
    from realpc.tracker import Tracker
    ep = Episode(EP)
    cfg = load_config(CFG)
    arms = build_arms(cfg, ep.calib)
    tr = convert(ep, arms, 5, 15, cfg=tracker_config(cfg), verbose=False)
    trk = Tracker(tracker_config(cfg), arms, *table_plane(cfg, ep.calib))
    for k, i in enumerate(range(5, 15)):
        out = trk.step(ep.frame(i))
        ref = tr.pos[: len(out.pos), k]
        ok = np.isfinite(out.pos).all(1)
        assert np.allclose(out.pos[ok], ref[ok], atol=1e-5)


def test_ransac_kabsch_recovers_pose_with_outliers():
    from realpc.se3 import apply, ransac_kabsch
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 3)) * 0.05
    from scipy.spatial.transform import Rotation as Rot
    T = np.eye(4); T[:3, :3] = Rot.from_rotvec([0.2, -0.1, 0.3]).as_matrix(); T[:3, 3] = [0.1, -0.05, 0.02]
    B = apply(T, A) + rng.normal(size=A.shape) * 0.001
    B[:40] += rng.normal(size=(40, 3)) * 0.05                      # 20 % outliers
    Te, inl, _ = ransac_kabsch(A, B, thr=0.005)
    assert np.abs(Te - T).max() < 5e-3 and inl[40:].mean() > 0.9
