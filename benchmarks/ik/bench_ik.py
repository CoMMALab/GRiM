"""IK: P pose targets for one end-effector, S seeds each, best seed per problem.

Same targets (FK of random in-limit configurations), seeds, weights (position 50, orientation
10) and iteration cap (60) for every method:

* ``grim_<ls|dls|sqp|hjcd|mppi>`` GRiM's mantis-traced IK kernels (grim.motion.ik, float)
* ``mantis`` mantis's traced LM step (``ik_lm_step``: FK, Jacobian, residual, damped 6x6
              Cholesky solve and joint-limit projection in one traced function) inside a CUDA
              accept/reject loop with the same settings (ik/mantis_ik.py)
* ``mantis_ls_{f32,f64,mixed}``  GRiM's ``ls_ik`` formulation and loop, its step and
              merit traced by mantis, all-float / all-double / GRiM's float-FK, double-solve split (ik/mantis_ik_ls.py)
* ``mantis_sqp_{f32,f64,mixed}``  GRiM's ``sqp_ik`` formulation and loop, traced by mantis
              (ik/mantis_ik_sqp.py)
* ``pyroki``  pyroki's IK-Beam as shipped in its benchmark (``pose_cost_analytic_jac`` +
              ``limit_cost``, jaxls dense-Cholesky LM; its own 64->4 seeds, weights and 16 steps)
* ``frax``    frax 0.2.0 GlobalIKSolver as its parallel_global_ik.py example (diff-IK SQP, 32 starts)
* ``curobo``  cuRobo v2 IKSolver as configured by cuRobo's benchmark (LM-seeded L-BFGS); curobo env

Success is judged in float64 by Pinocchio on the returned configuration: position < 1 mm,
orientation < 0.01 rad, inside joint limits.

    python ik/bench_ik.py <prepare|grim_ls|grim_dls|grim_sqp|grim_hjcd|grim_mppi|grim_<kernel>_cuda|mantis|mantis_ls_f32|mantis_ls_f64|mantis_ls_mixed|pyroki|curobo|frax> [robot ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "bridge"))
import common  # noqa: E402

BATCHES = (1, 128, 1024)
SEEDS = 8
MAX_ITER = 60
POS_W, ORI_W = 50.0, 10.0
EE = {"panda": "panda_grasptarget", "baxter": "right_gripper", "g1": "right_hand_palm_link"}
POS_TOL, ORI_TOL = 1e-3, 1e-2


def prepare(robot: str) -> None:
    urdf = common.MODEL_URDF[robot]
    P = max(BATCHES)
    goal_q = common.configs(urdf, P, seed=1)
    T = common.link_poses_reference(urdf, goal_q, EE[robot])
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    target = np.concatenate([xyzw[:, [3, 0, 1, 2]], T[:, :3, 3]], 1).astype(np.float32)
    lo, hi = common.limits(urdf)
    seeds = common.configs(urdf, P * SEEDS, seed=2).reshape(P, SEEDS, -1)
    path = common.inputs_path("ik", robot)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, joint_names=np.array(common.joint_names(urdf)), target=target, seeds=seeds,
             lower=lo, upper=hi)


def score(robot: str, q: np.ndarray, target: np.ndarray) -> dict:
    """Best seed per problem, by Pinocchio's float64 pose error, inside the URDF's own limits
    (common.limits clips to [-pi, pi] for sampling, which would fail e.g. Panda joint 6 in
    (pi, 3.8223])."""
    m = common.model(common.MODEL_URDF[robot])
    lo, hi = m.lowerPositionLimit, m.upperPositionLimit
    from scipy.spatial.transform import Rotation
    P, S, n = q.shape
    T = common.link_poses_reference(common.MODEL_URDF[robot], q.reshape(P * S, n), EE[robot])
    T = T.reshape(P, S, 4, 4)
    pos = np.linalg.norm(T[..., :3, 3] - target[:, None, 4:], axis=-1)
    R_goal = Rotation.from_quat(target[:, [1, 2, 3, 0]]).as_matrix()
    R_err = np.einsum("pji,psjk->psik", R_goal, T[..., :3, :3])
    ori = np.arccos(np.clip((np.trace(R_err, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    inside = np.all((q >= lo - 1e-4) & (q <= hi + 1e-4), axis=-1)
    ok = (pos < POS_TOL) & (ori < ORI_TOL) & inside
    best = np.argmin(np.where(inside, pos + 0.1 * ori, np.inf), axis=1)
    i = np.arange(P)
    return {"success": float(ok.any(axis=1).mean()), "pos_mm": 1e3 * float(np.median(pos[i, best])),
            "ori_rad": float(np.median(ori[i, best]))}


def _grim_solver(robot, seeds, target, kernel):
    """GRiM's mantis-traced IK kernels (grim.motion.ik, float), our seeds and weights. hjcd is
    pyroffi's composition: coarse coordinate descent (k_max 20) then LM with stall kicks.
    Returns ``solve(s, g, tier)``, its inputs and the output permutation."""
    import jax.numpy as jnp
    from grim.motion import MotionRobot, ik
    m = MotionRobot.from_urdf(str(common.MODEL_URDF[robot]))
    names = list(common.load_inputs("ik", robot)["joint_names"])
    perm = [names.index(n) for n in m.actuated_names]
    inv = np.argsort(perm)
    ee = (m.link_joint(EE[robot]),)
    W = dict(pos_weight=POS_W, ori_weight=ORI_W, eps_pos=1e-4, eps_ori=1e-3)
    C = dict(collision_weight=1e4, collision_margin=0.02)  # inert (no collision), but required attrs
    LM = dict(W, max_iter=MAX_ITER, lambda_init=5e-3)
    settings = {"ls": dict(LM, **C), "dls": LM, "sqp": dict(LM, **C, n_inner_iters=10),
                "mppi": dict(W, **C, n_particles=16, n_mppi_iters=4, n_lbfgs_iters=MAX_ITER - 4,
                             m_lbfgs=5, sigma=0.1, mppi_temperature=0.05)}
    s, g = jnp.asarray(seeds[..., perm]), jnp.asarray(target[:, None])
    if kernel == "hjcd":
        tc, tl = ik.build("hjcd_coarse", m, ee), ik.build("hjcd_lm", m, ee)
        noise = jnp.asarray(0.5 * np.sin(1e3 * seeds[..., perm][:, :, None, :]
                                         + 17.0 * np.arange(MAX_ITER)[None, None, :, None]))
        def solve(s, g, tier):
            q0 = ik.run(tc, s, g, tier=tier, k_max=20, **C)[0]
            return ik.run(tl, q0, g, noise=noise, tier=tier, max_iter=MAX_ITER, lambda_init=5e-3,
                          eps_pos=1e-4, eps_ori=1e-3, stall_patience=6, limit_prior_weight=1e-4,
                          kick_scale=0.02, early_stop=False, **C)[0]
    else:
        t = ik.build(kernel, m, ee)
        def solve(s, g, tier):
            return ik.run(t, s, g, tier=tier, rng_seed=7, **settings[kernel])[0]
    return solve, s, g, inv, ik.TIERS


def run_grim(robot, seeds, target, kernel="ls"):
    """GRiM's IK through its JAX FFI surface (``_grim_solver``), fastest tier per cell."""
    import jax
    solve, s, g, inv, tiers = _grim_solver(robot, seeds, target, kernel)
    best = None
    for tier in tiers:  # the fastest tier per cell, as GRiM recommends choosing by batch
        f = jax.jit(lambda s, g, tier=tier: solve(s, g, tier))
        out = jax.block_until_ready(f(s, g))
        t0 = time.perf_counter()
        jax.block_until_ready(f(s, g))
        if best is not None and 1e3 * (time.perf_counter() - t0) > 3 * best[0]:
            continue  # clearly the slower tier: not worth K x REPS calls
        ms = common.amortized_ms(lambda: f(s, g), jax.block_until_ready)
        if best is None or ms < best[0]:
            best = (ms, np.asarray(out)[..., inv], tier)
    return best


def run_grim_cuda(robot, seeds, target, kernel="ls"):
    """The same GRiM kernels and calls as ``run_grim``, replayed through the C bridge
    (``bridge/grim_bridge.py``): no Python or JAX in the timed loop. Each tier's bridge output must
    equal the JAX output; fastest tier per cell."""
    import grim_bridge
    solve, s, g, inv, tiers = _grim_solver(robot, seeds, target, kernel)
    best = None
    for tier in tiers:
        with grim_bridge.record() as calls:
            ref = np.asarray(solve(s, g, tier))
        prog = grim_bridge.Program(calls)
        try:
            prog.run()
            out = prog.results[-1][0].get()
            grim_bridge.check(out, ref, f"{kernel} {robot} {tier}")
            ms = prog.time_ms(common.K, common.REPS, common.WARMUP)
        finally:
            prog.free()
        if best is None or ms < best[0]:
            best = (ms, out[..., inv], tier)
    return best


def run_pyroki(robot, seeds, target):
    """pyroki's IK-Beam, verbatim from external/pyroki/benchmark/ik_benchmark.py (PyrokiIkBeamHelper
    .solve_ik): 64 quasi-random (Roberts) seeds over the limits, 6 LM steps each, the best 4 run
    10 more, one answer per problem. Its own seeds, weights (pos 10, ori 5, limit_cost 50) and
    16-step budget; the shared seeds and settings above are not used. Reported times do vary with JAX precision set to high."""
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki as pk
    import yourdfpy
    from jax import lax
    pk_robot = pk.Robot.from_urdf(yourdfpy.URDF.load(str(common.MODEL_URDF[robot]), load_meshes=False))
    names = list(common.load_inputs("ik", robot)["joint_names"])
    inv = np.argsort([names.index(n) for n in pk_robot.joints.actuated_names])
    link = jnp.array(pk_robot.links.names.index(EE[robot]))
    n = pk_robot.joints.num_actuated_joints
    num_seeds_init, num_seeds_final, total_steps, init_steps = 64, 4, 16, 6

    def newton_raphson(f, iters):
        x, _ = lax.scan(lambda x, _: (x - f(x) / jax.grad(f)(x), None), 1.0, length=iters)
        return x

    root = newton_raphson(lambda x: x ** (n + 1) - x - 1, 10_000)
    basis = 1 - (1 / root ** (1 + jnp.arange(n)))
    initial_qs = pk_robot.joints.lower_limits + jnp.modf(
        jnp.arange(num_seeds_init)[:, None] * basis[None, :])[0] * (
        pk_robot.joints.upper_limits - pk_robot.joints.lower_limits)

    def solve_ik(goal):
        def solve_one(initial_q, lambda_initial, max_iters):
            var = pk_robot.joint_var_cls(0)
            factors = [pk.costs.pose_cost_analytic_jac(
                           pk_robot, var, jaxlie.SE3.from_rotation_and_translation(
                               jaxlie.SO3(goal[:4]), goal[4:]), link, pos_weight=10.0, ori_weight=5.0),
                       pk.costs.limit_cost(pk_robot, var, weight=50.0)]
            sol, summary = jaxls.LeastSquaresProblem(factors, [var]).analyze().solve(
                initial_vals=jaxls.VarValues.make([var.with_value(initial_q)]), verbose=False,
                linear_solver="dense_cholesky",
                termination=jaxls.TerminationConfig(max_iterations=max_iters, early_termination=False),
                trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                return_summary=True)
            return sol[var], summary

        vsolve = jax.vmap(solve_one, in_axes=(0, 0, None))
        sols, summary = vsolve(initial_qs, jnp.full(num_seeds_init, 10.0), init_steps)
        best = jnp.argsort(summary.cost_history[jnp.arange(num_seeds_init), -1])[:num_seeds_final]
        sols, summary = vsolve(sols[best], summary.lambda_history[jnp.arange(num_seeds_init), -1][best],
                               total_steps - init_steps)
        return sols[jnp.argmin(summary.cost_history[jnp.arange(num_seeds_final), summary.iterations])]

    # pyroki's precision, not run_all.sh's "highest" (kept for frax): IK-Beam is 4.7x slower
    # under "highest" on JAX 0.6, and its benchmark runs at the default.
    with jax.default_matmul_precision("default"):
        f = jax.jit(jax.vmap(solve_ik))
        g = jnp.asarray(target)
        out = f(g)
        ms = common.amortized_ms(lambda: f(g), jax.block_until_ready)
    return ms, np.asarray(out)[:, None][..., inv], "ik-beam 64->4"


def run_curobo(robot, seeds, target):
    """cuRobo v2's IKSolver, configured by cuRobo's own benchmark (external/curobo/benchmark/
    ik_benchmark.py ``create_ik_config``, collision_free=False as in its "IK" column): LM-seeded
    L-BFGS, its own seeds, CUDA graphs, exit_early, TF32 on. Our URDF and end-effector only; the
    shared seeds and settings above are not used. Run in the curobo env."""
    import tempfile

    import torch
    import yaml
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.types.tool_pose import GoalToolPose
    from curobo_robot import kinematics_dict
    from ik_benchmark import create_ik_config
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # as cuRobo's run_benchmark
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        yaml.safe_dump({"robot_cfg": {"kinematics": kinematics_dict(common.MODEL_URDF[robot], [EE[robot]])}}, f)
    P = len(target)
    solver = IKSolver(create_ik_config(f.name, None, P, use_cuda_graph=True, collision_free=False))
    data = common.load_inputs("ik", robot)
    names = list(data["joint_names"])
    idx = [names.index(n) for n in solver.joint_names]  # cuRobo drops joints that can't move the EE
    t = torch.as_tensor(target, device="cuda")
    goal = GoalToolPose(tool_frames=[EE[robot]], position=t[:, 4:].reshape(P, 1, 1, 1, 3).contiguous(),
                        quaternion=t[:, :4].reshape(P, 1, 1, 1, 4).contiguous())

    def solve():
        solver.reset_seed()
        return solver.solve_pose(goal_tool_poses=goal, seed_config=None)

    solver.config.exit_early = False  # cuRobo's warm-up
    for _ in range(3):
        solve()
    solver.config.exit_early = True
    q = np.broadcast_to((data["lower"] + data["upper"]) / 2, (P, 1, len(names))).copy()
    q[..., idx] = solve().js_solution.position.reshape(P, 1, -1).cpu().numpy()
    ms = common.amortized_ms(solve, lambda _: torch.cuda.synchronize())
    return ms, q, "lbfgs+lm-seed"


def frax_ee(urdf, ee: str):
    """(movable joint carrying ``ee``, the fixed transform from that joint's frame to ``ee``):
    frax tracks joint frames plus an offset (FrameTask), not URDF links."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    joints = {j.find("child").get("link"): j for j in ET.parse(urdf).getroot().findall("joint")}
    T, link = np.eye(4), ee
    while joints[link].get("type") == "fixed":
        o = joints[link].find("origin")
        X = np.eye(4)
        if o is not None:
            X[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
            X[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        T, link = X @ T, joints[link].find("parent").get("link")
    return joints[link].get("name"), T


def run_frax(robot, seeds, target):
    """frax 0.2.0's GlobalIKSolver as frax's examples/ik/parallel_global_ik.py sets it up: diff-IK
    SQP (ElastiQP) with a pose task (gains 10), a home-pose JointTask (weight 0.01) and
    JointLimits; 32 starts (home + 31 uniform), 200 iterations, tol 1e-4, lowest cost wins. Our
    URDF and end-effector (FrameTask on the joint carrying it); home is the limits' midpoint
    (frax's Panda default_q is not defined for our robots). vmapped over problems; device and
    float64 follow JAX_PLATFORMS / JAX_ENABLE_X64. The shared seeds are not used."""
    import jax
    import jax.numpy as jnp
    from frax.core.robot import Robot
    from frax.ik import DiffIKSolver, FrameTask, GlobalIKSolver, JointLimits, JointTask
    urdf = common.MODEL_URDF[robot]
    r = Robot(str(urdf))
    n = r.num_actuated_joints
    joint, offset = frax_ee(urdf, EE[robot])
    solver = DiffIKSolver(r, [FrameTask(r, [r.joint_name_to_index[joint]], gain_pos=10.0, gain_rot=10.0,
                                        offsets=offset),
                              JointTask(r, np.arange(n), "q_home", gain=1.0, weight=0.01)],
                          [JointLimits(r)])
    global_ik = GlobalIKSolver(solver)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    lo, hi = np.asarray(r.actuated_joint_lower_limits), np.asarray(r.actuated_joint_upper_limits)
    home = jnp.asarray((lo + hi) / 2, dtype)
    P = len(target)
    from scipy.spatial.transform import Rotation
    refs = {"frame_pos": jnp.asarray(target[:, None, 4:], dtype),
            "frame_rmat": jnp.asarray(Rotation.from_quat(target[:, [1, 2, 3, 0]]).as_matrix()[:, None], dtype),
            "q_home": jnp.broadcast_to(home, (P, n))}
    keys = jax.random.split(jax.random.key(0), P)
    f = jax.jit(jax.vmap(lambda ref, key: global_ik.solve(home, ref, key).q))
    out = f(refs, keys)
    ms = common.amortized_ms(lambda: f(refs, keys), jax.block_until_ready)
    names = list(common.load_inputs("ik", robot)["joint_names"])
    inv = np.argsort([names.index(j) for j in r.joint_names])
    return ms, np.asarray(out, np.float64)[:, None][..., inv], f"global-ik 32 ({jax.default_backend()})"


def run_mantis(robot, seeds, target):
    import mantis_ik
    return mantis_ik.run(robot, seeds, target, EE[robot], MAX_ITER, POS_W, ORI_W)


def run_mantis_sqp(precision):
    def run(robot, seeds, target):
        import mantis_ik_sqp
        return mantis_ik_sqp.run(robot, seeds, target, EE[robot], MAX_ITER, POS_W, ORI_W, precision)
    return run


def run_mantis_ls(precision):
    def run(robot, seeds, target):
        import mantis_ik_ls
        return mantis_ik_ls.run(robot, seeds, target, EE[robot], MAX_ITER, POS_W, ORI_W, precision)
    return run


RUNNERS = {**{f"grim_{k}": (lambda r, s, t, k=k: run_grim(r, s, t, k))
              for k in ("ls", "dls", "sqp", "hjcd", "mppi")},
           **{f"grim_{k}_cuda": (lambda r, s, t, k=k: run_grim_cuda(r, s, t, k))
              for k in ("ls", "dls", "sqp", "hjcd", "mppi")}, "mantis": run_mantis,
           "mantis_ls_f32": run_mantis_ls("f32"), "mantis_ls_f64": run_mantis_ls("f64"),
           "mantis_ls_mixed": run_mantis_ls("mixed"),
           **{f"mantis_sqp_{p}": run_mantis_sqp(p) for p in ("f32", "f64", "mixed")},
           "pyroki": run_pyroki, "curobo": run_curobo, "frax": run_frax}


def main():
    method, robots = sys.argv[1], sys.argv[2:] or list(common.ROBOTS)
    if method == "prepare":
        for robot in robots:
            prepare(robot)
        return
    if method == "score":  # grade solutions saved by a run in an env without Pinocchio (curobo)
        method, robots = robots[0], robots[1:] or list(common.ROBOTS)
        name = f"ik_{method}_{'_'.join(robots)}"
        rows, qs = json.loads((common.RESULTS / f"{name}.json").read_text()), np.load(common.RESULTS / f"{name}_q.npz")
        for row in rows:
            data = common.load_inputs("ik", row["robot"])
            P = row["batch"]
            row.update(score(row["robot"], qs[f"{row['robot']}_{P}"], data["target"][:P]))
            report(row)
        common.save(name, rows)
        return
    try:
        import pinocchio  # noqa: F401
        graded = True
    except ImportError:
        graded = False
    rows, qs = [], {}
    for robot in robots:
        data = common.load_inputs("ik", robot)
        for P in BATCHES:
            seeds, target = data["seeds"][:P], data["target"][:P]
            ms, q, tier = RUNNERS[method](robot, seeds, target)
            S = {"pyroki": 64, "curobo": 2, "frax": 32}.get(method, SEEDS)  # seeds itself (curobo: 2 L-BFGS after 32 LM)
            row = {"method": method, "robot": robot, "batch": P, "seeds": S, "ms": ms, "tier": tier}
            if graded:
                row.update(score(robot, q, target))
                report(row)
            else:
                qs[f"{robot}_{P}"] = q
                print(f"{method:8s} {robot:7s} P={P:5d}x{S} {ms:9.3f} ms ({tier})  ungraded", flush=True)
            rows.append(row)
    name = f"ik_{method}_{'_'.join(robots)}"
    common.save(name, rows)
    if not graded:
        np.savez(common.RESULTS / f"{name}_q.npz", **qs)
        print(f"no Pinocchio here: grade with `bench_ik.py score {method} {' '.join(robots)}` in the grim env")


def report(row):
    print(f"{row['method']:8s} {row['robot']:7s} P={row['batch']:5d}x{row['seeds']} {row['ms']:9.3f} ms "
          f"({row['tier']})  success {row['success']:.3f}  pos {row['pos_mm']:.2e} mm  ori {row['ori_rad']:.2e} rad",
          flush=True)


if __name__ == "__main__":
    main()
