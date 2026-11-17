"""The MotionBenchMaker (MBM) problems as one trajectory-optimization problem for every method.

* **Problems:** MBM's 8 Panda scenes x 100 problems (robometrics' ``mb_set.yaml``, the set
  cuRobo's motion-planning benchmark loads), from ``external/robometrics`` at the pinned commit.
  Obstacles are cuboids and solid cylinders, in GRiM's layouts: boxes ``[centre, x, y, z axes,
  half-extents]`` (15), cylinders ``[centre, unit axis, radius, half-height]`` (8).
* **Robot:** the spherized Panda of the other benchmarks (``robots/panda.urdf``) for every method,
  without its fixed base sphere (58 spheres; it rests on MBM's pedestal box in every scene), and
  one self-collision link-pair list (``self_pairs``), sampled the way MoveIt's
  setup assistant does: adjacent links, pairs in collision in almost every sample, and pairs
  never in collision are left out.
* **Endpoints:** MBM's start, and the first of its goal IK solutions that is collision-free under
  that model. MBM's solutions are collision-free for its mesh model; problems where none is (or
  whose start is not) under the sphere model are dropped, and the count is recorded.
* **Seed:** the straight line from start to goal, ``T`` waypoints.
* **Margin:** every method optimizes to the same safety margin ``MARGIN`` (1 cm), each through its
  own setting (TrajOpt, whose Python bindings do not expose its margin: spheres inflated by it).
* **Grading (float64, this model):** success = every waypoint and ``SUBSTEPS`` linear
  interpolations of each segment free of world and self collision (signed distance >= 0, no
  margin), inside the URDF limits, endpoints within ``ENDPOINT_TOL``; ``free`` drops the endpoint
  test (for solvers whose endpoints are soft constraints, e.g. pyroki's augmented Lagrangian).
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

MBM = HERE.parent / "external/robometrics/robometrics/content/dataset/mb_set.yaml"
# Tuning set, disjoint from MBM: the first TUNE_PER_SCENE feasible problems of each MπNets
# category (same Panda, same box/cylinder obstacle schema). Methods' parameters are chosen on it
# and frozen before they see MBM. DATASET selects which set load() returns.
MPINETS = MBM.with_name("mpinets_set.yaml")
TUNE_PER_SCENE = 5
DATASET = "mbm"   # or "mpinets"
URDF = common.FK_URDF["panda"]
T = 32
SUBSTEPS = 10
ENDPOINT_TOL = 1e-3
LIMIT_TOL = 1e-4
PAIR_SAMPLES = 20000
# The safety margin every method optimizes to (grading stays at 0 clearance, densely): methods
# that constrain only their waypoints or samples to 0 skim obstacles there and cut them between
# (TrajOpt, Drake), and larger margins make MBM's grasp goals infeasible (pyroki's 5 cm example).
MARGIN = 0.01
JOINTS = [f"panda_joint{i}" for i in range(1, 8)]   # the URDF's actuated order (checked in load)


def _rot(wxyz) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def world(obstacles: dict) -> dict:
    """MBM obstacles -> {"boxes": (Mb, 15), "cylinders": (My, 8)} in GRiM's layouts, plus the raw
    poses for libraries that take a pose and dimensions."""
    boxes, cyls = [], []
    for o in obstacles.get("cuboid", {}).values():
        p, R = np.asarray(o["pose"][:3]), _rot(o["pose"][3:])
        boxes.append(np.r_[p, R.T.reshape(-1), 0.5 * np.asarray(o["dims"])])
    for o in obstacles.get("cylinder", {}).values():
        p, R = np.asarray(o["pose"][:3]), _rot(o["pose"][3:])
        cyls.append(np.r_[p, R[:, 2], o["radius"], 0.5 * o["height"]])
    return {"boxes": np.array(boxes, float).reshape(-1, 15), "cylinders": np.array(cyls, float).reshape(-1, 8)}


# ---------------------------------------------------------------- robot model

BASE = "panda_link0"  # fixed, and resting on MBM's pedestal box in every scene


@lru_cache(maxsize=None)
def robot_spheres():
    """(link per sphere, centre in link frame (S, 3), radius (S,)), the URDF's sphere order,
    without the fixed base's sphere: it touches MBM's pedestal in every problem, a contact no
    trajectory can change (MoveIt disables base-table pairs for the same reason)."""
    links, centres, radii = common.spheres(URDF)
    keep = [i for i, l in enumerate(links) if l != BASE]
    return tuple(links[i] for i in keep), centres[keep], radii[keep]


def sphere_centres(q: np.ndarray) -> np.ndarray:
    """(B, S, 3) world centres of ``robot_spheres``, float64 (Pinocchio)."""
    links = common.spheres(URDF)[0]
    return common.reference_spheres(URDF, np.atleast_2d(q))[:, [i for i, l in enumerate(links) if l != BASE]]


def _link_order() -> list[str]:
    return list(dict.fromkeys(robot_spheres()[0]))


def _adjacent() -> set[tuple[str, str]]:
    """Link pairs joined by a joint, fixed joints collapsed (MoveIt's 'adjacent')."""
    import xml.etree.ElementTree as ET
    root = ET.parse(URDF).getroot()
    parent = {j.find("child").get("link"): (j.find("parent").get("link"), j.get("type"))
              for j in root.iter("joint")}
    has = set(_link_order())

    def up(link):  # nearest ancestor with spheres
        while link in parent:
            link = parent[link][0]
            if link in has:
                return link
        return None
    pairs = set()
    for link in has:
        a = up(link)
        if a is not None:
            pairs.add(tuple(sorted((link, a))))
    return pairs


def _link_pair_min(c: np.ndarray, links, radii, a: str, b: str) -> np.ndarray:
    ia = [i for i, l in enumerate(links) if l == a]
    ib = [i for i, l in enumerate(links) if l == b]
    d = np.linalg.norm(c[:, ia, None] - c[:, None, ib], axis=-1) - radii[ia][:, None] - radii[ib][None]
    return d.reshape(len(c), -1).min(axis=1)


def self_pairs(path: Path | None = None) -> list[tuple[str, str]]:
    """The shared self-collision link pairs (cached in results/inputs)."""
    path = path or common.inputs_path("trajopt", "panda_self_pairs").with_suffix(".json")
    if path.exists():
        return [tuple(p) for p in json.loads(path.read_text())]
    links, _, radii = robot_spheres()
    order, adj = _link_order(), _adjacent()
    q = common.configs(URDF, PAIR_SAMPLES, seed=11).astype(np.float64)
    c = sphere_centres(q)
    keep = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if tuple(sorted((a, b))) in adj:
                continue
            frac = float((_link_pair_min(c, links, radii, a, b) < 0).mean())
            if 0 < frac < 0.95:  # never: cannot collide; almost always: overlapping by design
                keep.append((a, b))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keep))
    return keep


# ---------------------------------------------------------------- distances (float64)

def box_dist(c: np.ndarray, r: np.ndarray, box: np.ndarray) -> np.ndarray:
    """(..., 3) centres, (...) radii -> signed distance to a box (GRiM layout)."""
    local = (c - box[:3]) @ box[3:12].reshape(3, 3).T
    q = np.abs(local) - box[12:15]
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - r


def cylinder_dist(c: np.ndarray, r: np.ndarray, cyl: np.ndarray) -> np.ndarray:
    """Signed distance to a solid cylinder (GRiM layout): the clamped closest point outside,
    minus the depth to the nearest face inside."""
    v = c - cyl[:3]
    a = v @ cyl[3:6]
    radial = v - a[..., None] * cyl[3:6]
    rho = np.linalg.norm(radial, axis=-1)
    R, h = cyl[6], cyl[7]
    inside = (rho <= R) & (np.abs(a) <= h)
    u = radial / np.where(rho > 0, rho, 1)[..., None]
    closest = cyl[:3] + np.clip(a, -h, h)[..., None] * cyl[3:6] + np.minimum(rho, R)[..., None] * u
    out = np.linalg.norm(c - closest, axis=-1)
    return np.where(inside, -np.minimum(R - rho, h - np.abs(a)), out) - r


def clearance(q: np.ndarray, w: dict, pairs) -> tuple[np.ndarray, np.ndarray]:
    """(B, n) configurations -> (world clearance (B,), self clearance (B,)), float64."""
    links, _, radii = robot_spheres()
    c = sphere_centres(q)
    d = np.full(len(c), np.inf)
    for box in w["boxes"]:
        d = np.minimum(d, box_dist(c, radii, box).min(axis=1))
    for cyl in w["cylinders"]:
        d = np.minimum(d, cylinder_dist(c, radii, cyl).min(axis=1))
    s = np.full(len(c), np.inf)
    for a, b in pairs:
        s = np.minimum(s, _link_pair_min(c, links, radii, a, b))
    return d, s


# ---------------------------------------------------------------- problems

def seed(start: np.ndarray, goal: np.ndarray, t: int = T) -> np.ndarray:
    return np.linspace(start, goal, t)


def problems_path():
    """The cache of the DATASET problems load() returns (read by the TrajOpt harness too)."""
    return common.inputs_path("trajopt", f"panda_{DATASET}").with_suffix(".json")


def load() -> list[dict]:
    """The DATASET problems with sphere-model-feasible endpoints (cached in results/inputs): all of
    MBM, or the MπNets tuning set."""
    path = problems_path()
    if path.exists():
        return [{**p, **{k: np.asarray(p[k]) for k in ("start", "goal", "boxes", "cylinders")}}
                for p in json.loads(path.read_text())]
    import yaml
    data = yaml.safe_load((MBM if DATASET == "mbm" else MPINETS).read_text())
    pairs = self_pairs()
    names = common.joint_names(URDF)
    if names != JOINTS:
        raise ValueError(f"unexpected joint order {names}")
    out, dropped = [], {}
    for scene, problems in data.items():
        kept = 0
        for i, p in enumerate(problems):
            if DATASET == "mpinets" and kept == TUNE_PER_SCENE:
                break
            w = world(p["obstacles"])
            start = np.asarray(p["start"], float)
            cand = np.asarray(p["goal_ik"], float)
            d, s = clearance(np.vstack([start, cand]), w, pairs)
            ok = (d >= 0) & (s >= 0)
            if not ok[0] or not ok[1:].any():
                dropped[scene] = dropped.get(scene, 0) + 1
                continue
            goal = cand[int(np.argmax(ok[1:]))]
            kept += 1
            out.append({"scene": scene, "index": i, "start": start, "goal": goal, **w,
                        "obstacles": p["obstacles"]})   # MBM's own dicts, for libraries that read them
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in p.items()}
                                for p in out]))
    path.with_name(path.stem + "_dropped.json").write_text(json.dumps(dropped))
    return out


def grade(traj: np.ndarray, p: dict, pairs) -> dict:
    """Float64 verdict and path metrics of one trajectory (T', n) for problem ``p``."""
    traj = np.asarray(traj, float)
    m = common.model(URDF)
    lo, hi = m.lowerPositionLimit, m.upperPositionLimit
    s = np.linspace(0, 1, SUBSTEPS + 1)[:-1]
    dense = np.concatenate([(traj[:-1, None] + s[None, :, None] * np.diff(traj, axis=0)[:, None])
                            .reshape(-1, traj.shape[1]), traj[-1:]])
    d, sc = clearance(dense, p, pairs)
    in_limits = bool(np.all((traj >= lo - LIMIT_TOL) & (traj <= hi + LIMIT_TOL)))
    ends = max(np.abs(traj[0] - p["start"]).max(), np.abs(traj[-1] - p["goal"]).max())
    vel = np.diff(traj, axis=0)
    free = bool(d.min() >= 0 and sc.min() >= 0 and in_limits)
    return {"success": free and bool(ends <= ENDPOINT_TOL), "free": free,
            "world_clear": float(d.min()), "self_clear": float(sc.min()), "in_limits": in_limits,
            "endpoint_err": float(ends), "path_length": float(np.linalg.norm(vel, axis=1).sum()),
            "smoothness": float((np.diff(vel, axis=0) ** 2).sum()) if len(traj) > 2 else 0.0}
