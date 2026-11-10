"""SCO, CHOMP and STOMP: the stock kernels (kernels/trajopt, backend "grim", their own
block shape) against the crocket kernels (kernels/trajopt_crocket) in float and double at every
tier (one trajectory, or STOMP sample, per thread / warp / block).

Same inputs, settings and launcher for every backend. Per (robot, scenario, method, batch):

* time: ms per jitted call, median of REPS runs of K back-to-back calls (K so a run lasts about
  MIN_RUN_S), after a warm-up call with the same configuration;
* success (graded on the first SCORE_MAX trajectories): the fraction that is collision-free at
  every timestep by the float64 oracle (every self pair and world obstacle at distance >= 0)
  and inside the joint limits within 1e-3 rad;
* cost: the float64 oracle's cost of the returned trajectories (mean over the graded ones), and
  the largest relative gap between the kernel's reported cost and the oracle's.

Scenarios: ``self`` (sphere_model's self pairs, no world) and ``world`` (plus obstacle_world's
sphere and floor). Problems: noisy straight lines between two in-limit configurations.

    python test/benchmarks/bench_crocket_trajopt.py <robot> [--out results/trajopt_<robot>.json]
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
from motion.conftest import load_robot  # noqa: E402
from motion.traj_cases import obstacle_world, sphere_model, straight_lines  # noqa: E402

from grim.motion import trajopt  # noqa: E402
from grim.motion.ik import World  # noqa: E402
from grim.motion.reference import trajopt as ref  # noqa: E402

T = 32
BATCHES = (64, 1024)
REPS, MIN_RUN_S, MAX_CALL_S = 3, 0.3, 20.0
SCORE_MAX = 64
W = dict(w_smooth=1.0, w_acc=0.5, w_jerk=0.1, w_limits=1.0, w_collision_max=100.0,
         collision_margin=0.01)
METHODS = {"sco": (trajopt.sco_trajopt, ref.sco_cost), "chomp": (trajopt.chomp_trajopt, ref.stomp_cost),
           "stomp": (trajopt.stomp_trajopt, ref.stomp_cost)}
STOCK_TIER = {"sco": "block", "chomp": "warp", "stomp": "block"}   # the stock kernels' shapes
CONFIGS = [("grim", None)] + [(b, t) for b in ("crocket-f32", "crocket-f64")
                              for t in ("thread", "warp", "block")]


def timed(fn, x) -> float:
    fn(x).block_until_ready()
    t0 = time.perf_counter()
    fn(x).block_until_ready()
    one = time.perf_counter() - t0
    k = max(1, int(MIN_RUN_S / max(one, 1e-6)))
    runs = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(k):
            out = fn(x)
        out.block_until_ready()
        runs.append((time.perf_counter() - t0) / k)
    return 1e3 * float(np.median(runs))


def grade(robot, trajs, costs, tc, world, cost_fn):
    ok, oracle, gap = [], [], 0.0
    for b in range(min(len(trajs), SCORE_MAX)):
        t = trajs[b].astype(float)
        free = all(ref.hinge_collision_cost(robot, q, tc, world, 0.0) == 0.0 for q in t)
        viol = np.maximum(t - robot.upper, 0) + np.maximum(robot.lower - t, 0)
        ok.append(free and viol.max() <= 1e-3)
        o = cost_fn(robot, t, tc, world, lower=robot.lower, upper=robot.upper, **W)
        oracle.append(o)
        gap = max(gap, abs(float(costs[b]) - o) / max(abs(o), 1e-9))
    return float(np.mean(ok)), float(np.mean(oracle)), gap


def bench(name: str) -> list[dict]:
    robot = load_robot(name)
    tc = sphere_model(robot)
    rows, skip = [], set()
    for B in BATCHES:
        x, s, g = straight_lines(robot, B, T)
        for scenario in ("self", "world"):
            world = World() if scenario == "self" else obstacle_world(robot, s, g)
            for method, (fn, cost_fn) in METHODS.items():
                for backend, tier in CONFIGS:
                    tier = tier or STOCK_TIER[method]
                    row = dict(robot=name, n_q=robot.n_act, scenario=scenario, method=method,
                               batch=B, backend=backend, tier=tier)
                    if (scenario, method, backend, tier) in skip:
                        rows.append({**row, "skipped": True})
                        continue
                    kw = {} if backend == "grim" else dict(backend=backend, tier=tier)
                    run = jax.jit(lambda x, fn=fn, kw=kw: fn(robot, x, collision=tc, world=world,
                                                            start=s, goal=g, **W, **kw))
                    try:
                        ms = timed(lambda x: run(x)[1], x)
                        trajs, costs = (np.asarray(a) for a in run(x))
                    except Exception as exc:   # noqa: BLE001 - a failed build is a result
                        rows.append({**row, "error": str(exc)[:300]})
                        print(f"{name} {scenario} {method} B={B} {backend} {tier}: {exc}", flush=True)
                        continue
                    succ, cost, gap = grade(robot, trajs, costs, tc, world, cost_fn)
                    rows.append({**row, "ms": ms, "success": succ, "oracle_cost": cost, "cost_gap": gap})
                    print(f"{name:7s} {scenario:5s} {method:5s} B={B:5d} {backend:11s} {tier:6s} "
                          f"{ms:10.2f} ms  s {succ:.2f}  cost {cost:9.4f}  gap {gap:.1e}", flush=True)
                    if ms / 1e3 * (BATCHES[-1] / B) > MAX_CALL_S:
                        skip.add((scenario, method, backend, tier))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("robot")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = bench(args.robot)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
