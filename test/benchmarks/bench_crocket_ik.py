"""Every IK kernel (kernels/ik_crocket) in float and in double (``backend="crocket-f32"`` /
``"crocket-f64"``) at every tier. The hand-written kernels these replaced were measured the same
way; their rows are in results/tiers_*.json (backend "grim").

Same inputs, settings and launcher for every backend. Per (robot, scenario, kernel, batch):

* time: amortized ms per call (K back-to-back jitted calls, one sync), median of REPS runs, K
  chosen so a run lasts about MIN_RUN_S; every tier of every crocket build (thread / warp / block,
  one seed per thread / warp / 64-thread block). A (backend, tier) whose extrapolated call at the
  next batch exceeds MAX_CALL_S is recorded as skipped;
* success (graded on the first SCORE_MAX problems): the fraction of problems whose best seed (lowest reported error) reaches the target
  within 1 mm / 0.01 rad by the float64 oracle, inside the joint limits, and (collision
  scenario) collision-free by the oracle's distances.

Scenarios: ``pose`` (one end-effector, no collision), ``multi`` (the robot's multi-end-effector
set), ``collision`` (one end-effector, test/motion/ik_cases.collision_case: spheres on every moving
frame, self pairs, a world of every obstacle kind). Canonical IK is timed on solved
configurations; analytic IK on fr3 only (the family it solves).

    python test/benchmarks/bench_crocket_ik.py <robot> [--out results/crocket_ik_<robot>.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from motion.conftest import ee_sets, load_robot  # noqa: E402
from motion.ik_cases import collision_case, reachable_targets, seeds  # noqa: E402

from grim.motion import ik  # noqa: E402
from grim.motion.reference import collision as C  # noqa: E402
from grim.motion.reference import kinematics as K  # noqa: E402

BACKENDS = ("crocket-f32", "crocket-f64")
BATCHES = (1, 128, 1024, 4096, 16384)
SEEDS = 8
REPS = 5
MIN_RUN_S = 0.25
MAX_CALL_S = 20.0      # skip a (backend, tier) whose extrapolated call at the next batch exceeds this
MIN_EXTRAP_B = 1024    # extrapolate only from batches this large: below it the GPU is far from saturated
SCORE_MAX = 1024       # problems graded by the oracle (the first ones of the batch)
TIERS = ("thread", "warp", "block")
POS_TOL, ORI_TOL = 1e-3, 1e-2
W = dict(pos_weight=50.0, ori_weight=10.0, lambda_init=5e-3, eps_pos=1e-4, eps_ori=1e-3)
COLL = dict(collision_weight=1e4, collision_margin=0.02)
KERNELS = ("ls", "dls", "sqp", "hjcd_coarse", "hjcd_lm", "mppi")


def settings(kernel: str, s: np.ndarray) -> dict:
    if kernel == "ls":
        return dict(max_iter=60, **W, **COLL)
    if kernel == "dls":
        return dict(max_iter=100, **W)
    if kernel == "sqp":
        return dict(max_iter=60, n_inner_iters=10, **W, **COLL)
    if kernel == "hjcd_coarse":
        return dict(k_max=20, **COLL)
    if kernel == "hjcd_lm":
        noise = np.random.default_rng(5).normal(size=(*s.shape[:2], 60, s.shape[2])).astype(np.float32)
        return dict(noise=noise, max_iter=60, stall_patience=6, early_stop=0, lambda_init=5e-3,
                    limit_prior_weight=1e-4, kick_scale=0.02, eps_pos=1e-4, eps_ori=1e-3, **COLL)
    if kernel == "mppi":
        return dict(rng_seed=7, n_particles=16, n_mppi_iters=4, n_lbfgs_iters=30, m_lbfgs=5, sigma=0.1,
                    mppi_temperature=0.05, pos_weight=50.0, ori_weight=10.0, eps_pos=1e-4,
                    eps_ori=1e-3, **COLL)
    raise ValueError(kernel)


def amortized_ms(call) -> float:
    jax.block_until_ready(call())
    t = time.perf_counter()
    jax.block_until_ready(call())
    one = time.perf_counter() - t
    k = max(1, min(200, int(MIN_RUN_S / max(one, 1e-5))))
    runs = []
    for _ in range(REPS if one < 2.0 else 2):
        t = time.perf_counter()
        for _ in range(k):
            out = call()
        jax.block_until_ready(out)
        runs.append((time.perf_counter() - t) / k)
    return 1e3 * float(np.median(runs))


def collision_free(robot, q, coll, world) -> bool:
    T = K.frame_poses(robot, q)
    sc = coll.self_collision
    cs = [K.quat_rotate(T[sc.link_joint[l]][:4], sc.sph[a, :3]) + T[sc.link_joint[l]][4:]
          for l in range(len(sc.link_start) - 1) for a in range(sc.link_start[l], sc.link_start[l + 1])]
    for li, lj in zip(sc.pair_i, sc.pair_j):
        for a in range(sc.link_start[li], sc.link_start[li + 1]):
            for b in range(sc.link_start[lj], sc.link_start[lj + 1]):
                if np.linalg.norm(cs[a] - cs[b]) - sc.sph[a, 3] - sc.sph[b, 3] < 0:
                    return False
    prims = [(C.sphere_sphere, world.spheres), (C.sphere_capsule, world.capsules),
             (C.sphere_box, world.boxes), (C.sphere_halfspace, world.halfspaces)]
    for sph, j in zip(coll.robot_spheres, coll.robot_sphere_joint):
        c = K.quat_rotate(T[j][:4], sph[:3]) + T[j][4:]
        if any(f(c, sph[3], o) < 0 for f, arr in prims for o in arr):
            return False
    return True


def score(robot, q, e, targets, ees, coll=None, world=None) -> dict:
    q, e, targets = q[:SCORE_MAX], e[:SCORE_MAX], targets[:SCORE_MAX]
    best = q[np.arange(len(q)), e.argmin(1)]
    pos, ori, ok = [], [], []
    for p in range(len(best)):
        T = K.frame_poses(robot, best[p])
        r = np.stack([K.pose_residual(T[x], targets[p, k]) for k, x in enumerate(ees)])
        pe, oe = np.linalg.norm(r[:, :3], axis=1).max(), np.linalg.norm(r[:, 3:], axis=1).max()
        inside = np.all(best[p] >= robot.lower - 1e-5) and np.all(best[p] <= robot.upper + 1e-5)
        good = pe < POS_TOL and oe < ORI_TOL and inside
        if good and coll is not None:
            good = collision_free(robot, best[p], coll, world)
        pos.append(pe)
        ori.append(oe)
        ok.append(good)
    return {"success": float(np.mean(ok)), "pos_mm": 1e3 * float(np.median(pos)),
            "ori_rad": float(np.median(ori))}


def bench_solvers(robot, name, rows, scenario, which, skip=frozenset(), only=None):
    ees = ee_sets(robot)[which]
    coll, world = collision_case(robot) if scenario == "collision" else (None, ik.World())
    P = max(BATCHES)
    targets, _ = reachable_targets(robot, ees, P, seed=11)
    s_all = seeds(robot, P, SEEDS, seed=12)
    for kernel in KERNELS:
        if only is not None and kernel not in only:
            continue
        if kernel == "dls" and scenario == "collision":
            continue   # no collision terms
        for backend in BACKENDS:
            t0 = time.perf_counter()
            try:
                T = ik.build(kernel, robot, ees, collision=coll, world_counts=world.counts(), backend=backend)
            except Exception as exc:  # noqa: BLE001 - record and continue
                rows.append({"robot": name, "scenario": scenario, "kernel": kernel, "backend": backend,
                             "error": f"build: {exc}"[:400]})
                print(f"  {kernel} {backend}: BUILD FAILED {exc}"[:300], flush=True)
                continue
            build_s = time.perf_counter() - t0
            for tier in TIERS:
                if f"{scenario}:{kernel}:{backend}:{tier}" in skip:
                    rows.append({"robot": name, "scenario": scenario, "kernel": kernel, "backend": backend,
                                 "tier": tier, "error": "skipped by --skip"})
                    print(f"  {scenario:9s} {kernel:11s} {backend:12s} {tier:6s} skipped (--skip)", flush=True)
                    continue
                last = None   # (batch, ms) of this tier's previous batch
                for B in BATCHES:
                    base_row = {"robot": name, "n_q": robot.n_act, "n_ee": len(ees), "scenario": scenario,
                                "kernel": kernel, "backend": backend, "tier": tier, "batch": B, "seeds": SEEDS,
                                "build_s": build_s}
                    if last is not None and last[0] >= MIN_EXTRAP_B and last[1] * B / last[0] / 1e3 > MAX_CALL_S:
                        rows.append({**base_row, "skipped": f"est {last[1] * B / last[0] / 1e3:.0f} s/call"})
                        print(f"  {scenario:9s} {kernel:11s} {backend:12s} {tier:6s} B={B:5d}   skipped (est)", flush=True)
                        continue
                    s, t = s_all[:B], targets[:B]
                    kw = settings(kernel, s)
                    extra = {"tier": tier, "block_threads": 64}
                    f = jax.jit(lambda s, t, extra=extra: ik.run(T, s, t, world=world, **kw, **extra)[:2])
                    q, e = jax.block_until_ready(f(s, t))
                    ms = amortized_ms(lambda: f(s, t))
                    last = (B, ms)
                    row = {**base_row, "ms": ms, **score(robot, np.asarray(q), np.asarray(e), t, ees, coll, world)}
                    rows.append(row)
                    print(f"  {scenario:9s} {kernel:11s} {backend:12s} {tier:6s} B={B:5d} {ms:10.3f} ms "
                          f"success {row['success']:.3f}", flush=True)


def bench_canonical(robot, name, rows):
    ees = ee_sets(robot)[0]
    for scenario in ("pose", "collision"):
        coll, world = collision_case(robot) if scenario == "collision" else (None, ik.World())
        P = 16384
        targets, q = reachable_targets(robot, ees, P, seed=13)
        q_ref = np.clip(q + np.random.default_rng(3).normal(scale=0.3, size=q.shape), robot.lower, robot.upper)
        for backend in BACKENDS:
            T = ik.build_canonical(robot, ees, collision=coll, world_counts=world.counts(), backend=backend)
            for tier in TIERS:
                for B in (1, 1024, 16384):
                    kwt = {"tier": tier}
                    f = jax.jit(lambda a, b, c, kwt=kwt: ik.run_canonical(T, a, b, c, world=world, max_iters=200, **kwt))
                    args = (q[:B], q_ref[:B], targets[:B])
                    qs, _ = jax.block_until_ready(f(*args))
                    ms = amortized_ms(lambda: f(*args))
                    qs = np.asarray(qs)
                    n = min(B, 256)
                    res = [np.linalg.norm(np.concatenate([K.pose_residual(K.frame_poses(robot, qs[p])[x], targets[p, k])
                                                          for k, x in enumerate(ees)])) for p in range(n)]
                    rows.append({"robot": name, "n_q": robot.n_act, "n_ee": len(ees), "scenario": scenario,
                                 "kernel": "canonical", "backend": backend, "tier": tier, "batch": B, "seeds": 1,
                                 "ms": ms, "success": float(np.mean(np.array(res) < 1e-3)),
                                 "residual_max": float(np.max(res))})
                    print(f"  {scenario:9s} canonical   {backend:12s} {tier:6s} B={B:5d} {ms:10.3f} ms  max|r| {max(res):.1e}", flush=True)


def bench_analytic(rows):
    from grim.motion import analytic_ik as A
    from motion.test_analytic_ik import _matrix
    robot = load_robot("fr3")
    ee = robot.joint_index("fr3_joint7")
    g = A.arm_geometry(robot, ee)
    qs = np.random.default_rng(0).uniform(robot.lower, robot.upper, size=(max(BATCHES), robot.n_act))
    poses = np.stack([K.frame_poses(robot, q)[ee] for q in qs])
    targets = np.stack([_matrix(p) for p in poses]).astype(np.float32)
    q7 = np.linspace(g.lower[6], g.upper[6], 64).astype(np.float32)
    T0 = K.frame_poses(robot, np.zeros(robot.n_act))
    chain = [j for j in reversed(robot.chain(ee)) if robot.act_idx[j] != -1]
    sph = np.array([[*T0[j, 4:] + [0.05, 0, 0], 0.06] for j in chain], np.float32)
    pairs = np.array([[0, 4], [0, 5], [1, 6], [2, 6]], np.int32)
    world = ik.World(spheres=np.array([[0.4, 0.2, 0.4, 0.1]]), halfspaces=np.array([[0, 0, 1, 0, 0, -0.05]]))
    for scenario, kw in (("pose", {}), ("collision", dict(spheres_home=sph, sphere_joint=np.arange(7, dtype=np.int32),
                                                          self_pairs=pairs, world=world))):
        for backend in BACKENDS:
            target = A.build(robot, ee, backend)
            for tier in TIERS:
                for B in BATCHES:
                    tkw = {"tier": tier}
                    f = jax.jit(lambda t, tkw=tkw: A.run(target, t, q7, **kw, **tkw))
                    out = jax.block_until_ready(f(targets[:B]))
                    ms = amortized_ms(lambda: f(targets[:B]))
                    found = np.asarray(out[2])
                    rows.append({"robot": "fr3", "n_q": 7, "n_ee": 1, "scenario": scenario, "kernel": "analytic",
                                 "backend": backend, "tier": tier, "batch": B, "seeds": 64, "ms": ms,
                                 "success": float(found.mean())})
                    print(f"  {scenario:9s} analytic    {backend:12s} {tier:6s} B={B:5d} {ms:10.3f} ms  found {found.mean():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("robot")
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*", help="subset of: solvers canonical analytic")
    ap.add_argument("--kernels", nargs="*", help="scenario:kernel cells to run (default all)")
    ap.add_argument("--scenarios", nargs="*", default=("pose", "multi", "collision"))
    ap.add_argument("--skip", nargs="*", default=(), help="scenario:kernel:backend:tier cells to record as skipped")
    a = ap.parse_args()
    robot = load_robot(a.robot)
    rows: list = []
    out = Path(a.out or HERE / "results" / f"crocket_ik_{a.robot}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = set(a.only or ("solvers", "canonical", "analytic"))

    def save():
        out.write_text(json.dumps(rows, indent=1))

    if "solvers" in parts:
        for scenario, which in (("pose", 0), ("multi", 1), ("collision", 0)):
            if scenario not in a.scenarios or (a.kernels and not any(c.startswith(scenario + ":") for c in a.kernels)) or which >= len(ee_sets(robot)):
                continue
            print(f"{a.robot} {scenario}", flush=True)
            bench_solvers(robot, a.robot, rows, scenario, which, set(a.skip),
                          a.kernels and {c.split(':')[1] for c in a.kernels if c.split(':')[0] == scenario})
            save()
    if "canonical" in parts:
        bench_canonical(robot, a.robot, rows)
        save()
    if "analytic" in parts and a.robot == "fr3":
        bench_analytic(rows)
        save()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
