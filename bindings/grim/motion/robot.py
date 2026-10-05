"""The robot a motion kernel is built for.

A :class:`MotionRobot` is everything a per-robot build bakes in: the kinematic tables (one
row per URDF joint, fixed joints included, so every link has a frame), the joint orders the
caller uses, and the URDF itself for the optional cricket trace.

Conventions (shared with pyroffi, so a pyroffi robot passes its own tables unchanged):

* joints are in URDF document order; ``topo_inv`` lists them parent-first;
* a twist is ``[v, w]`` (linear first): ``[0, axis]`` revolute/continuous, ``[axis, 0]``
  prismatic, zero for fixed;
* ``parent_tf`` is the joint origin as ``[qw, qx, qy, qz, x, y, z]``;
* actuated joints are the non-fixed, non-mimic ones, in document order; a mimic joint has
  ``act_idx == -1`` and ``mimic_act_idx`` naming the actuated joint it follows;
* ``q`` (seeds, outputs) is ordered like ``actuated_names``.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class MotionRobot:
    urdf_xml: str
    joint_names: tuple[str, ...]        # every joint (frame), document order
    actuated_names: tuple[str, ...]     # q order
    link_names: tuple[str, ...]
    link_parent_joint: np.ndarray       # (n_links,) joint posing each link, -1 for the root
    twists: np.ndarray                  # (n_joints, 6)
    parent_tf: np.ndarray               # (n_joints, 7)
    parent_idx: np.ndarray              # (n_joints,)
    act_idx: np.ndarray                 # (n_joints,)
    mimic_mul: np.ndarray               # (n_joints,)
    mimic_off: np.ndarray               # (n_joints,)
    mimic_act_idx: np.ndarray           # (n_joints,)
    topo_inv: np.ndarray                # (n_joints,) parent-first processing order
    lower: np.ndarray                   # (n_act,)
    upper: np.ndarray                   # (n_act,)

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def n_act(self) -> int:
        return len(self.actuated_names)

    def tables(self) -> tuple[np.ndarray, ...]:
        """The 8 kinematic tables, in the order the CUDA FK walks them."""
        return (self.twists, self.parent_tf, self.parent_idx, self.act_idx,
                self.mimic_mul, self.mimic_off, self.mimic_act_idx, self.topo_inv)

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)

    def link_joint(self, link: str) -> int:
        """Index of the joint whose frame is ``link``'s frame."""
        j = int(self.link_parent_joint[self.link_names.index(link)])
        if j < 0:
            raise ValueError(f"link {link!r} is the root; it has no joint frame")
        return j

    def chain(self, joint: int) -> list[int]:
        """Joint ``joint`` and its ancestors."""
        out = []
        while joint != -1:
            out.append(joint)
            joint = int(self.parent_idx[joint])
        return out

    @staticmethod
    def from_urdf(urdf: str) -> "MotionRobot":
        """Parse a URDF (path or XML text)."""
        xml = urdf if urdf.lstrip().startswith("<") else open(urdf).read()
        return _parse(xml)


def _floats(s: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.array([float(v) for v in s.split()]) if s else np.array(default, float)


def _quat_wxyz_from_rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _parse(xml: str) -> MotionRobot:
    # Vendor extension tags (<gazebo:...>) often use unbound prefixes, which the stdlib
    # parser rejects; they never carry kinematics, so drop the prefixes before parsing.
    root = ET.fromstring(re.sub(r"<(/?)[A-Za-z_][\w.-]*:", r"<\1", xml))
    joints = root.findall("joint")
    links = [l.get("name") for l in root.findall("link")]
    names = [j.get("name") for j in joints]
    jtype = {j.get("name"): j.get("type") for j in joints}
    mimic = {j.get("name"): j.find("mimic") for j in joints}
    actuated = [n for n in names if jtype[n] != "fixed" and mimic[n] is None]
    child_joint = {j.find("child").get("link"): i for i, j in enumerate(joints)}

    twists, parent_tf, parent_idx, act_idx = [], [], [], []
    mul, off, mimic_act, lower, upper = [], [], [], [], []
    for j in joints:
        n = j.get("name")
        axis = _floats(j.find("axis").get("xyz") if j.find("axis") is not None else None, (1, 0, 0))
        t = jtype[n]
        twists.append(np.r_[np.zeros(3), axis] if t in ("revolute", "continuous")
                      else np.r_[axis, np.zeros(3)] if t == "prismatic" else np.zeros(6))
        o = j.find("origin")
        xyz = _floats(o.get("xyz") if o is not None else None, (0, 0, 0))
        rpy = _floats(o.get("rpy") if o is not None else None, (0, 0, 0))
        parent_tf.append(np.r_[_quat_wxyz_from_rpy(*rpy), xyz])
        parent_idx.append(child_joint.get(j.find("parent").get("link"), -1))
        m = mimic[n]
        if m is not None:
            src = m.get("joint")
            act_idx.append(-1)
            mimic_act.append(actuated.index(src) if src in actuated else -1)
            mul.append(float(m.get("multiplier", 1.0)))
            off.append(float(m.get("offset", 0.0)))
        else:
            act_idx.append(actuated.index(n) if n in actuated else -1)
            mimic_act.append(-1)
            mul.append(1.0)
            off.append(0.0)
        if n in actuated:
            lim = j.find("limit")
            lo, hi = (lim.get("lower"), lim.get("upper")) if lim is not None else (None, None)
            if lo is None or hi is None:
                if t != "continuous":
                    raise ValueError(f"joint {n!r} ({t}) has no limits")
                lo, hi = -math.pi, math.pi
            lower.append(float(lo))
            upper.append(float(hi))

    # Parent-first order, honouring mimic sources (as pyroffi's topological sort does).
    order, done_links, done = [], set(), set()
    child_links = {j.find("child").get("link") for j in joints}
    roots = {j.find("parent").get("link") for j in joints} - child_links
    pending = list(range(len(joints)))
    while pending:
        for k, i in enumerate(pending):
            j = joints[i]
            pl = j.find("parent").get("link")
            m = mimic[names[i]]
            if (pl in roots or pl in done_links) and (m is None or m.get("joint") in done):
                order.append(i)
                done_links.add(j.find("child").get("link"))
                done.add(names[i])
                pending.pop(k)
                break
        else:
            raise ValueError(f"URDF joints do not form a tree: {[names[i] for i in pending]}")

    return MotionRobot(
        urdf_xml=xml, joint_names=tuple(names), actuated_names=tuple(actuated),
        link_names=tuple(links),
        link_parent_joint=np.array([child_joint.get(l, -1) for l in links], np.int32),
        twists=np.array(twists, np.float64), parent_tf=np.array(parent_tf, np.float64),
        parent_idx=np.array(parent_idx, np.int32), act_idx=np.array(act_idx, np.int32),
        mimic_mul=np.array(mul, np.float64), mimic_off=np.array(off, np.float64),
        mimic_act_idx=np.array(mimic_act, np.int32), topo_inv=np.array(order, np.int32),
        lower=np.array(lower, np.float64), upper=np.array(upper, np.float64))
