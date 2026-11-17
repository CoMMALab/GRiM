"""TrajOpt (tesseract-robotics/trajopt 0.35.0) through the ``tesseract-robotics`` 0.35.0 wheel, on
the shared problem (mbm.py): the same spherized Panda (URDF variant without the base sphere),
the shared self pairs as the SRDF's allowed-collision complement, MBM's boxes and cylinders as
native Tesseract geometry, and the straight-line seed of ``mbm.T`` states (Tesseract's
``generateInterpolatedProgram`` with exactly T - 1 steps). The planner is called as in
tesseract_python's tests/tesseract_motion_planning/test_trajopt_planner.py, with TrajOpt's
default move and composite profiles.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mbm  # noqa: E402

NS = "TrajOptMotionPlannerTask"


def robot_files(inflate: float = 0.0) -> tuple[Path, Path]:
    """The shared robot for libraries that load a URDF: no base sphere, no visuals; and its SRDF
    (group "manipulator", every link pair outside mbm.self_pairs allowed to collide). ``inflate``
    grows every sphere's radius: a margin m on sphere-obstacle distance is a radius + m with
    margin 0 (2 m for a self pair, slightly conservative)."""
    out = mbm.common.inputs_path("trajopt", f"panda_inflate{inflate:g}").with_suffix(".urdf")
    srdf = out.with_suffix(".srdf")
    root = ET.parse(mbm.URDF).getroot()
    for link in root.iter("link"):
        for v in list(link.findall("visual")):
            link.remove(v)
        if link.get("name") == mbm.BASE:
            for c in list(link.findall("collision")):
                link.remove(c)
        for sph in link.iter("sphere"):
            sph.set("radius", repr(float(sph.get("radius")) + inflate))
    root.set("xmlns:tesseract", "https://github.com/tesseract-robotics/tesseract")
    root.set("tesseract:make_convex", "false")   # required by Tesseract 0.35; the model has no meshes
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out)
    links = [l.get("name") for l in root.iter("link")]
    keep = {tuple(sorted(p)) for p in mbm.self_pairs()}
    pairs = "".join(f'  <disable_collisions link1="{a}" link2="{b}" reason="Never"/>\n'
                    for i, a in enumerate(links) for b in links[i + 1:] if tuple(sorted((a, b))) not in keep)
    srdf.write_text(f'<?xml version="1.0"?>\n<robot name="{root.get("name")}">\n'
                    f'  <group name="manipulator"><chain base_link="{mbm.BASE}" tip_link="panda_hand"/></group>\n'
                    # Tesseract's own default collision managers (Bullet), from the pinned support files
                    '  <contact_managers_plugin_config filename="package://tesseract/support/urdf/contact_manager_plugins.yaml"/>\n'
                    f"{pairs}</robot>\n")
    return out, srdf


def _pose(R: np.ndarray, t: np.ndarray):
    from tesseract_robotics.tesseract_common import Isometry3d
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return Isometry3d(T)


def environment(p: dict, urdf: Path, srdf: Path):
    """A Tesseract environment of the robot and problem ``p``'s obstacles."""
    from tesseract_robotics.tesseract_common import FilesystemPath, GeneralResourceLocator
    from tesseract_robotics.tesseract_environment import AddLinkCommand, Environment
    from tesseract_robotics.tesseract_geometry import Box, Cylinder
    from tesseract_robotics.tesseract_scene_graph import Collision, Joint, JointType_FIXED, Link
    env = Environment()
    if not env.init(FilesystemPath(str(urdf)), FilesystemPath(str(srdf)), GeneralResourceLocator()):
        raise RuntimeError("Tesseract could not load the robot")
    obstacles = [(f"box{i}", Box(*(2 * b[12:15])), b[3:12].reshape(3, 3).T, b[:3]) for i, b in enumerate(p["boxes"])]
    for i, c in enumerate(p["cylinders"]):
        z = c[3:6] / np.linalg.norm(c[3:6])
        x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
        x /= np.linalg.norm(x)
        obstacles.append((f"cyl{i}", Cylinder(c[6], 2 * c[7]), np.stack([x, np.cross(z, x), z], 1), c[:3]))
    for name, geom, R, t in obstacles:
        link = Link(name)
        col = Collision()
        col.geometry = geom
        link.collision.push_back(col)
        joint = Joint(name + "_joint")
        joint.parent_link_name = mbm.BASE
        joint.child_link_name = name
        joint.type = JointType_FIXED
        joint.parent_to_joint_origin_transform = _pose(R, t)
        if not env.applyCommand(AddLinkCommand(link, joint)):
            raise RuntimeError(f"Tesseract could not add {name}")
    return env


def program(env, p: dict):
    """The straight-line seed from p["start"] to p["goal"]: exactly mbm.T states."""
    from tesseract_robotics.tesseract_command_language import (
        CompositeInstruction, InstructionPoly_wrap_MoveInstruction, JointWaypoint, MoveInstruction,
        MoveInstructionType_FREESPACE, WaypointPoly_wrap_JointWaypoint)
    from tesseract_robotics.tesseract_common import ManipulatorInfo
    from tesseract_robotics.tesseract_motion_planners_simple import generateInterpolatedProgram
    manip = ManipulatorInfo()
    manip.tcp_frame, manip.manipulator, manip.working_frame = "panda_hand", "manipulator", mbm.BASE
    prog = CompositeInstruction("DEFAULT")
    prog.setManipulatorInfo(manip)
    for q in (p["start"], p["goal"]):
        wp = JointWaypoint(mbm.JOINTS, np.asarray(q, np.float64))
        prog.append(InstructionPoly_wrap_MoveInstruction(
            MoveInstruction(WaypointPoly_wrap_JointWaypoint(wp), MoveInstructionType_FREESPACE, "DEFAULT")))
    # segment lengths too large to bind: min_steps alone sets the count
    return generateInterpolatedProgram(prog, env, 1e6, 1e6, 1e6, mbm.T - 1)


def solver():
    from tesseract_robotics.tesseract_common import ProfileDictionary
    from tesseract_robotics.tesseract_motion_planners_trajopt import (
        TrajOptDefaultCompositeProfile, TrajOptDefaultMoveProfile, TrajOptMotionPlanner)
    profiles = ProfileDictionary()
    profiles.addProfile(NS, "DEFAULT", TrajOptDefaultMoveProfile())
    profiles.addProfile(NS, "DEFAULT", TrajOptDefaultCompositeProfile())
    return TrajOptMotionPlanner(NS), profiles


def solve(planner, profiles, env, prog) -> tuple[np.ndarray, bool]:
    from tesseract_robotics.tesseract_command_language import (
        InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly)
    from tesseract_robotics.tesseract_motion_planners import PlannerRequest
    req = PlannerRequest()
    req.instructions, req.env, req.profiles = prog, env, profiles
    res = planner.solve(req)
    qs = []
    for instr in res.results.flatten():   # as the example reads them: keep each object alive
        move = InstructionPoly_as_MoveInstructionPoly(instr)
        wp = move.getWaypoint()
        if not wp.isStateWaypoint():
            raise RuntimeError(f"TrajOpt returned a non-state waypoint (successful={res.successful})")
        qs.append(np.array(WaypointPoly_as_StateWaypointPoly(wp).getPosition(), float).flatten())
    return np.array(qs, float), bool(res.successful)
