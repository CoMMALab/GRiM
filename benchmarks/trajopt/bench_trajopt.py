"""Kinematic trajectory optimization on MotionBenchMaker (see mbm.py for the problem).

Every method solves each MBM problem on its own (one problem per call, per-problem latency),
from the same straight-line seed of ``mbm.T`` waypoints, with its own default solver settings
(``--tuned``: those chosen on the MπNets tuning set), and is graded by ``mbm.grade`` on the
shared sphere model. Published runs repeat every method over seeds 0-4 (``--seed``).

* ``grim_chomp`` / ``grim_stomp`` / ``grim_sco``  GRiM's mantis-traced trajectory optimizers
  (``grim.motion.trajopt``), float, through the JAX FFI
* ``pyroki``   pyroki's jaxls trajopt (examples/pyroki_snippets/_trajopt.py, its IK step removed)
* ``curobo``   cuRobo v2 as designed: ``MotionPlanner.plan_cspace`` (its own seeds, RRT fallback,
  retries), configured as its own benchmark; no iteration budget
* ``trajopt``  TrajOpt (tesseract-robotics/trajopt 0.35.0, built from external/; trajopt_harness.cpp)
* ``drake``    Drake ``KinematicTrajectoryOptimization`` (1.51.1 wheel)

    python trajopt/bench_trajopt.py prepare
    python trajopt/bench_trajopt.py <method> [--iters N] [--seed S] [--tuned] [--limit N] [scene ...]
    python trajopt/bench_trajopt.py score <method> <same flags>   # grade a run from an env without Pinocchio

Each method runs in its own env (RUNNING.md); results go to
results/trajopt_<method>[_params][_i<N>][_scenes][_n<L>]_s<S>.json (tables.py traj_section).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import mbm  # noqa: E402

WARMUP = 2  # problems solved untimed first (the build, first-call costs)
# Iteration budget (``--iters N``; None: each method's default). One iteration is the solver's own
# step: CHOMP / STOMP / pyroki (LM) / cuRobo (L-BFGS) / Drake (SNOPT major) iterations, SCO's inner
# L-BFGS steps (outer x inner = N, at most 30 per linearization), TrajOpt's SQP iterations (set in
# its C++ harness; the Python bindings can't). Drake's limit applies to each of its two solves.
# cuRobo's L-BFGS runs in CUDA-graph blocks of 25, so budgets are multiples of 25.
ITERS: int | None = None
LIMIT: int | None = None
# Parameter overrides (``--params '{"w_collision_max": 200}'``), for tuning on the MπNets set:
# GRiM: the trajopt function's keywords; pyroki: pyroki_trajopt.solve's weights; cuRobo:
# TrajectoryOptimizerCfg.create's keywords; TrajOpt: "profile" (default | example) and
# "coeff" (the collision constraint coefficient); Drake: drake_trajopt's module constants.
PARAMS: dict = {}
# Repeat seed (``--seed S``, 0-4 in the published runs): GRiM STOMP's rng_seed and cuRobo's
# random_seed. TrajOpt, Drake, CHOMP and SCO are deterministic and pyroki has no seed; their
# repeats measure timing (and pyroki's run-to-run variation). Files carry ``_s<S>``.
SEED = 0
# Each method's tuned parameters (``--tuned``), chosen on the MπNets tuning set (``--tune``, 60
# problems) at 100 iterations from a 2-4 point grid per method, by success rate, then median time,
# then frozen before any MBM run. Methods absent here kept their defaults (best on the grid).
# The grids and results: RUNNING.md, "Tuning".
TUNED: dict = {
    "grim_stomp": {"w_collision_max": 1000},
    "trajopt": {"profile": "example"},
    "drake": {"COLLISION_SAMPLES": 50},
}
BUDGETS = (25, 50, 100, 200)
# pyroki's trajopt bounds joint velocity at its waypoint dt. The example's 0.02 s (50 steps of a
# short move) caps a joint at ~1.3 rad over our 31 steps, which makes 771 of the 800 MBM problems
# infeasible; 0.1 s is 1.5x what the worst straight-line seed needs (0.067 s), so the bound never
# binds on the seed. The problem is kinematic: dt only scales the velocity bound and acceleration cost.
PYROKI_DT = 0.1


# ---------------------------------------------------------------- GRiM

def _grim_model():
    from grim.motion import MotionRobot
    from grim.motion._build import TrajCollision
    m = MotionRobot.from_urdf(str(mbm.URDF))
    links, centres, radii = mbm.robot_spheres()
    rows = [m.link_joint(l) for l in links]
    R, S = m.n_joints + 1, max(rows.count(r) for r in set(rows))
    off, rad, fill = np.zeros((R, S, 3)), -np.ones((R, S)), [0] * R
    for r, c, rr in zip(rows, centres, radii):
        off[r, fill[r]], rad[r, fill[r]] = c, rr
        fill[r] += 1
    row = {l: m.link_joint(l) for l in links}
    pairs = mbm.self_pairs()
    pi = np.array([row[a] for a, _ in pairs], np.int32)
    pj = np.array([row[b] for _, b in pairs], np.int32)
    if list(m.actuated_names) != mbm.JOINTS:
        raise ValueError("GRiM joint order differs from the URDF's")
    return m, TrajCollision(off, rad, pi, pj)


def run_grim(kernel: str):
    def run(problems):
        import jax
        from grim.motion import trajopt
        from grim.motion.ik import World
        m, coll = _grim_model()
        fn = {"chomp": trajopt.chomp_trajopt, "stomp": trajopt.stomp_trajopt, "sco": trajopt.sco_trajopt}[kernel]
        budget = {} if ITERS is None else (
            {"n_inner_iters": min(30, ITERS), "n_outer_iters": max(1, ITERS // min(30, ITERS))} if kernel == "sco"
            else {"n_iters": ITERS})
        if kernel == "stomp":   # the only stochastic GRiM kernel
            budget["rng_seed"] = SEED
        out = []
        for i, p in enumerate(problems):
            w = World(boxes=p["boxes"], cylinders=p["cylinders"])
            x0 = mbm.seed(p["start"], p["goal"])[None].astype(np.float32)
            call = lambda: fn(m, x0, collision=coll, world=w, start=p["start"], goal=p["goal"],
                              collision_margin=mbm.MARGIN, **budget, **PARAMS)
            if i < WARMUP:   # the one build (the world is a runtime argument) and first-call costs
                jax.block_until_ready(call())
            t0 = time.perf_counter()
            trajs, _ = jax.block_until_ready(call())
            out.append((np.asarray(trajs[0], float), 1e3 * (time.perf_counter() - t0)))
        return out
    return run


# ---------------------------------------------------------------- pyroki

def run_pyroki(problems):
    import jax
    import yourdfpy
    import pyroki as pk
    import pyroki_trajopt as pt
    urdf = yourdfpy.URDF.load(str(mbm.URDF), load_meshes=False)
    robot = pk.Robot.from_urdf(urdf)
    if list(robot.joints.actuated_names) != mbm.JOINTS:
        raise ValueError("pyroki joint order differs from the URDF's")
    links, centres, radii = mbm.robot_spheres()
    dec = {}
    for l, c, r in zip(links, centres, radii):
        dec.setdefault(l, {"centers": [], "radii": []})
        dec[l]["centers"].append(c.tolist())
        dec[l]["radii"].append(float(r))
    names = list(robot.links.names)
    keep = {tuple(sorted(x)) for x in mbm.self_pairs()}
    ignore = tuple((a, b) for i, a in enumerate(names) for b in names[i + 1:] if tuple(sorted((a, b))) not in keep)
    coll = pk.collision.RobotCollision.from_sphere_decomposition(dec, urdf, user_ignore_pairs=ignore,
                                                                  ignore_immediate_adjacents=False)
    g2l = np.asarray(coll._geom_to_link_idx)   # sphere mode: active pairs are sphere indices
    active = {tuple(sorted((coll.link_names[g2l[i]], coll.link_names[g2l[j]])))
              for i, j in zip(coll.active_idx_i, coll.active_idx_j)}
    if active != keep:
        raise ValueError(f"pyroki's self pairs differ from the shared list: {active ^ keep}")
    # The whole solve under jax.jit: one compile per obstacle shape (each MBM scene), untimed;
    # unjitted, jaxls recompiles every call (~20 s of compile around a ~3 s solve).
    solve = jax.jit(lambda a, b, w: pt.solve(robot, coll, w, a, b, mbm.T, PYROKI_DT, host=False, max_iters=ITERS, **PARAMS))
    out = []
    for i, p in enumerate(problems):
        world = pt.world_geoms(p)
        call = lambda: jax.block_until_ready(solve(p["start"], p["goal"], world))
        if i == 0 or problems[i - 1]["scene"] != p["scene"]:
            call()   # this scene's compile
        t0 = time.perf_counter()
        traj = call()
        out.append((np.asarray(traj, float), 1e3 * (time.perf_counter() - t0)))
    return out


# ---------------------------------------------------------------- cuRobo

def _curobo_robot() -> dict:
    """cuRobo's own franka.yml (its tuned joint-space settings for this robot), on our URDF with
    the shared spheres and self pairs: no base sphere, no attached object, no buffers."""
    import xml.etree.ElementTree as ET
    from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
    cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    k = cfg["robot_cfg"]["kinematics"]
    links, centres, radii = mbm.robot_spheres()
    spheres = {}
    for l, c, r in zip(links, centres, radii):
        spheres.setdefault(l, []).append({"center": c.tolist(), "radius": float(r)})
    keep = {tuple(sorted(x)) for x in mbm.self_pairs()}
    all_links = [l.get("name") for l in ET.parse(mbm.URDF).getroot().iter("link")]
    joints = mbm.JOINTS
    k.update(urdf_path=str(mbm.URDF), asset_root_path="", base_link="panda_link0",
             collision_link_names=list(spheres), collision_spheres=spheres, collision_sphere_buffer=0.0,
             self_collision_buffer={l: 0.0 for l in spheres},
             self_collision_ignore={a: [b for b in all_links if b != a and tuple(sorted((a, b))) not in keep]
                                    for a in spheres},
             mesh_link_names=[], lock_joints={}, extra_links={}, extra_collision_spheres={},
             tool_frames=["panda_hand"])
    cs = k["cspace"]
    idx = [cs["joint_names"].index(j) for j in joints]
    for key in ("cspace_distance_weight", "null_space_weight", "default_joint_position"):
        cs[key] = [cs[key][i] for i in idx]
    cs["joint_names"] = joints
    return cfg


def run_curobo(problems):
    """cuRobo v2 as designed: ``MotionPlanner.plan_cspace`` (its joint-goal planner: trajopt seeds,
    RRT graph-planner fallback, up to 20 attempts, finetuning, time optimization), configured as its
    benchmark/motion_plan_benchmark.py configures the planner (32 IK / 4 trajopt seeds, activation
    0.0025 m, ``--mesh`` scenes since MBM has cylinders). The shared problem stays the same: our URDF
    limits (that benchmark widens them by 0.2 rad) and sphere model (_curobo_robot). No iteration
    budget applies: the planner runs its own; ``SEED`` is its random_seed."""
    import torch
    from curobo._src.geom.types import SceneCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.util_file import get_scene_configs_path, join_path, load_yaml
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    n_mesh = max(len(p["obstacles"].get("cuboid", {})) + len(p["obstacles"].get("cylinder", {})) for p in problems)
    scene = SceneCfg.create(load_yaml(join_path(get_scene_configs_path(), "collision_table.yml"))).get_mesh_world()
    robot = _curobo_robot()
    robot["robot_cfg"]["load_dynamics"] = False
    mg = MotionPlanner(MotionPlannerCfg.create(
        robot=robot, scene_model=scene, ik_optimizer_configs=["ik/lbfgs_ik.yml"],
        ik_transition_model="ik/transition_ik.yml", metrics_rollout="metrics_base.yml",
        trajopt_optimizer_configs=["trajopt/lbfgs_bspline_trajopt.yml"],
        trajopt_transition_model="trajopt/transition_bspline_trajopt.yml", num_ik_seeds=32,
        num_trajopt_seeds=4, collision_cache={"mesh": n_mesh}, optimizer_collision_activation_distance=0.0025,
        graph_planner_algorithm="rrt", random_seed=SEED, **PARAMS))
    if list(mg.joint_names) != mbm.JOINTS:
        raise ValueError(f"cuRobo joint order {mg.joint_names} differs from the URDF's")
    mg.warmup(enable_graph=True)
    dev = lambda x: torch.as_tensor(np.asarray(x, np.float32), device="cuda")
    out = []
    for i, p in enumerate(problems):
        # cuRobo caches meshes by obstacle name, and MBM reuses names (cube0, ...): unique names
        # per problem, or every problem collides against the first one's geometry.
        tag = f"{p['scene']}_{p['index']}_"
        mg.clear_scene_cache()
        mg.update_world(SceneCfg.create({kind: {tag + n: o for n, o in objs.items()}
                                         for kind, objs in p["obstacles"].items()}).get_mesh_world(merge_meshes=False))
        start, goal = JointState.from_position(dev(p["start"][None])), JointState.from_position(dev(p["goal"][None]))
        mg.reset_seed()   # as its benchmark, before every plan
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = mg.plan_cspace(goal, start)
        torch.cuda.synchronize()
        ms = 1e3 * (time.perf_counter() - t0)
        if r is None or r.interpolated_trajectory is None:   # no plan at all: graded as the seed (a failure)
            out.append((mbm.seed(p["start"], p["goal"]), ms))
            continue
        last = int(r.interpolated_last_tstep.reshape(-1)[0])
        traj = r.interpolated_trajectory.position.reshape(-1, len(mbm.JOINTS))[:last + 1].cpu().numpy()
        out.append((traj.astype(float), ms))
    return out


# ---------------------------------------------------------------- TrajOpt (Tesseract)

HARNESS = HERE.parent / "external/tesseract_ws/build_harness/trajopt_harness"   # RUNNING.md: building it


def run_trajopt(problems):
    """TrajOpt through trajopt_harness.cpp (the stack built from external/ at 0.35.0): the
    planner the wheel runs, plus the SQP iteration limit (the budget) and the margin, set natively.
    PARAMS "profile": "example" uses the collision configs of tesseract_planning's
    pick_and_place_example (swept LVS_DISCRETE checks; see the harness) instead of the defaults
    (per-waypoint DISCRETE)."""
    import json
    import os
    import subprocess
    import tempfile
    import tesseract_trajopt as tt
    urdf, srdf = tt.robot_files()   # the true spheres: the margin is TrajOpt's own setting here
    ws = HERE.parent / "external/tesseract_ws"
    profile = PARAMS.get("profile", "default")
    env = dict(os.environ, TRAJOPT_PROFILE=profile, TRAJOPT_COEFF=str(PARAMS.get("coeff", "")), TESSERACT_RESOURCE_PATH=str(HERE.parent / "external/tesseract"), TRAJOPT_LOG_THRESH="FATAL",
               LD_LIBRARY_PATH=f"{ws / 'install/lib'}:{Path.home() / 'miniconda3/envs/tesseract-src/lib'}")
    every = mbm.load()
    pos = {(p["scene"], p["index"]): i for i, p in enumerate(every)}
    idx = [pos[(p["scene"], p["index"])] for p in problems]
    out = []
    with tempfile.TemporaryDirectory() as d:
        k = 0
        while k < len(idx):   # contiguous runs of the cached problem list, one harness call each
            j = k
            while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
                j += 1
            res = Path(d) / f"{k}.jsonl"
            subprocess.run([str(HARNESS), str(urdf), str(srdf), str(mbm.problems_path()),
                            str(idx[k]), str(idx[j] + 1), str(mbm.T), str(ITERS or 0), str(mbm.MARGIN), str(res)],
                           check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            out += [(np.array(r["traj"], float), r["ms"]) for r in map(json.loads, res.read_text().splitlines())]
            k = j + 1
    return out


# ---------------------------------------------------------------- Drake

def run_drake(problems):
    import drake_trajopt as dt
    import tesseract_trajopt as tt
    for k, v in PARAMS.items():   # module constants, e.g. CONTROL_POINTS
        setattr(dt, k, v)
    urdf, _ = tt.robot_files()
    out = []
    for i, p in enumerate(problems):
        diagram, plant = dt.build(p, urdf)   # plant and program outside the timed solve
        trajopt, prog, coll = dt.problem(diagram, plant, p)
        if i < WARMUP:
            dt.solve(trajopt, prog, coll, ITERS)
            trajopt, prog, coll = dt.problem(diagram, plant, p)
        t0 = time.perf_counter()
        traj, _ = dt.solve(trajopt, prog, coll, ITERS)
        out.append((traj, 1e3 * (time.perf_counter() - t0)))
    return out


RUNNERS = {"grim_chomp": run_grim("chomp"), "grim_stomp": run_grim("stomp"), "grim_sco": run_grim("sco"),
           "pyroki": run_pyroki, "curobo": run_curobo, "trajopt": run_trajopt, "drake": run_drake}


def _tag(method, scenes):
    params = "".join(f"_{k}{v}" for k, v in sorted(PARAMS.items()))
    return (f"trajopt_{method}" + ("_tune" if mbm.DATASET != "mbm" else "") + params + (f"_i{ITERS}" if ITERS is not None else "") + ("_" + "_".join(scenes) if scenes else "")
            + (f"_n{LIMIT}" if LIMIT is not None else "") + f"_s{SEED}")


def main():
    global ITERS, LIMIT
    args = sys.argv[1:]
    global PARAMS, SEED
    if "--seed" in args:
        k = args.index("--seed")
        SEED = int(args[k + 1])
        args = args[:k] + args[k + 2:]
    if "--params" in args:
        k = args.index("--params")
        PARAMS = json.loads(args[k + 1])
        args = args[:k] + args[k + 2:]
    if "--tuned" in args:   # the frozen TUNED parameters of the method
        args.remove("--tuned")
        PARAMS = TUNED.get(args[1] if args[0] == "score" else args[0], {})
    if "--tune" in args:   # the MπNets tuning set (mbm.DATASET) instead of MBM
        args.remove("--tune")
        mbm.DATASET = "mpinets"
    if "--iters" in args:
        k = args.index("--iters")
        ITERS = int(args[k + 1])
        args = args[:k] + args[k + 2:]
    limit = None
    if "--limit" in args:   # the first N problems of each scene (quick checks)
        k = args.index("--limit")
        limit = LIMIT = int(args[k + 1])
        args = args[:k] + args[k + 2:]
    method, scenes = args[0], args[1:]
    if method == "prepare":
        print(f"{len(mbm.load())} problems, {len(mbm.self_pairs())} self pairs")
        return
    if method == "score":   # grade a run saved by an env without Pinocchio
        method, scenes = scenes[0], scenes[1:]
        tag = _tag(method, scenes)
        saved = np.load(common.RESULTS / f"{tag}_q.npz")
        problems = [p for p in mbm.load() if (not scenes or p["scene"] in scenes) and (limit is None or p["index"] < limit)]
        results = [(saved[f"q{i}"], float(saved["ms"][i])) for i in range(len(problems))]
    else:
        problems = [p for p in mbm.load() if (not scenes or p["scene"] in scenes) and (limit is None or p["index"] < limit)]
        results = RUNNERS[method](problems)
        tag = _tag(method, scenes)   # raw trajectories: graded later where Pinocchio is missing, and for viewing
        np.savez(common.RESULTS / f"{tag}_q.npz", ms=np.array([m for _, m in results]),
                 **{f"q{i}": q for i, (q, _) in enumerate(results)})
        try:
            import pinocchio  # noqa: F401
        except ImportError:   # e.g. the curobo env: grade in grim
            argv = " ".join(f"'{a}'" if " " in a else a for a in sys.argv[1:])
            print(f"saved {tag}_q.npz; grade with: python trajopt/bench_trajopt.py score {argv}")
            return
    pairs = mbm.self_pairs()
    rows = []
    for p, (traj, ms) in zip(problems, results):
        g = mbm.grade(traj, p, pairs)
        rows.append({"method": method, "budget": ITERS, "iters": ITERS, "scene": p["scene"],
                     "index": p["index"], "ms": ms, "params": PARAMS, "seed": SEED, **g})
    for scene in dict.fromkeys(r["scene"] for r in rows):
        rs = [r for r in rows if r["scene"] == scene]
        print(f"{method:11s} {scene:24s} n={len(rs):3d}  success {np.mean([r['success'] for r in rs]):.3f}"
              f"  median {np.median([r['ms'] for r in rs]):8.2f} ms", flush=True)
    common.save(_tag(method, scenes), rows)


if __name__ == "__main__":
    main()
