"""Loader of v4 converted episodes (scripts/convert_v4.py): expands per-body poses + local samples into points.

  d = load("~/data_ext/realpc_out/v1/sweep/episode_0000.npz")
  d["points"] [T,N,3] float32 world (left arm base), fixed identity; d["group"] [N] (0 robot_left, 1 robot_right,
  2 static, 3 library object, 4 cloth); d["obj_id"] [N] (-1 none); d["is_finger"] [N]; d["valid"] [T,N]; d["color"] [N,3].
"""
from __future__ import annotations

import numpy as np

GROUPS = {0: "robot_left", 1: "robot_right", 2: "static", 3: "object", 4: "cloth"}


def load(path, with_points=True):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    T = len(d["ticks"])
    parts, group, obj_id, finger, color, valid = [], [], [], [], [], []
    # robot
    RT = d["robot_T"]; L = d["robot_local"]; B = d["robot_body"].astype(int)
    names = [str(x) for x in d["robot_bodies"]]
    if with_points:
        RB = RT[:, B]
        parts.append(np.einsum("tnij,nj->tni", RB[..., :3, :3], L) + RB[..., :3, 3])
    arm = np.array([0 if names[b].startswith("left/") else 1 for b in B])
    group.append(arm); obj_id.append(np.full(len(L), -1)); color.append(np.full((len(L), 3), 90, np.uint8))
    finger.append(np.array([names[b].split("/")[1] in ("gripper", "tip_left", "tip_right") for b in B]))
    valid.append(np.ones((T, len(L)), bool))
    # library objects
    if "obj_local" in d:
        OL = d["obj_local"]; OI = d["obj_index"].astype(int); OT = d["obj_T"]
        if with_points:
            parts.append(np.einsum("tnij,nj->tni", OT[OI].transpose(1, 0, 2, 3)[:, :, :3, :3], OL) + OT[OI].transpose(1, 0, 2, 3)[:, :, :3, 3])
        group.append(np.full(len(OL), 3)); obj_id.append(OI); color.append(d["obj_color"]); finger.append(np.zeros(len(OL), bool))
        valid.append(d["obj_valid"][OI].T)
    # static
    S = d["static_points"]
    if with_points:
        parts.append(np.broadcast_to(S[None], (T,) + S.shape))
    group.append(np.full(len(S), 2)); obj_id.append(np.full(len(S), -1)); color.append(d["static_color"]); finger.append(np.zeros(len(S), bool))
    valid.append(np.ones((T, len(S)), bool))
    # cloth
    if "cloth_points" in d:
        C = d["cloth_points"]
        if with_points:
            parts.append(C.transpose(1, 0, 2))
        group.append(np.full(len(C), 4)); obj_id.append(np.full(len(C), -1)); color.append(np.full((len(C), 3), 150, np.uint8))
        finger.append(np.zeros(len(C), bool)); valid.append(np.repeat(d["cloth_valid"][:, None], len(C), 1))
    out = dict(d)
    if with_points:
        out["points"] = np.concatenate(parts, 1).astype(np.float32)
    out["group"] = np.concatenate(group).astype(np.int8); out["obj_id"] = np.concatenate(obj_id).astype(np.int16)
    out["is_finger"] = np.concatenate(finger); out["color"] = np.concatenate(color).astype(np.uint8)
    out["valid"] = np.concatenate(valid, 1)
    return out
