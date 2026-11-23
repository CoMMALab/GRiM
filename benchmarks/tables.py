"""results/*.json -> the tables between the markers in README.md (robots as columns)."""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

import common

FK = {"prrtc": "pRRTC `fk<Robot>`", "curobo": "cuRobo", "pyroki": "pyroki",
      "grim": "GRiD/GRiM (baked tables)", "mantis": "mantis (traced)",
      "grim_cuda": "GRiD/GRiM (baked tables), C bridge", "mantis_cuda": "mantis (traced), C bridge"}
# Global IK: full solvers from random seeds, compared head to head.
IK = {"grim_ls": "GRiM LS IK", "grim_dls": "GRiM DLS IK", "grim_sqp": "GRiM SQP IK",
      "grim_hjcd": "GRiM HJCD IK (coarse CD -> LM)", "grim_mppi": "GRiM MPPI IK (MPPI -> L-BFGS)",
      **{f"grim_{k}_cuda": f"GRiM {k.upper()} IK, C bridge" for k in ("ls", "dls", "sqp", "hjcd", "mppi")},
      "pyroki": "pyroki IK-Beam (analytic Jacobian; own 64->4 seeds)",
      "curobo": "cuRobo v2 IKSolver (LM-seeded L-BFGS; own seeds)"}
# Supplementary: mantis's traced IK steps in minimal benchmark loops (profile, not a comparison).
IK_STEP = {"mantis": "mantis `ik_lm_step` (6x6 LM)",
           "mantis_ls_f32": "mantis `ik_ls_step`, float", "mantis_ls_mixed": "mantis `ik_ls_step`, float system / double solve",
           "mantis_ls_f64": "mantis `ik_ls_step`, double",
           "mantis_sqp_f32": "mantis SQP step, float", "mantis_sqp_mixed": "mantis SQP step, float FK / double QP",
           "mantis_sqp_f64": "mantis SQP step, double"}
DYN = {"mantis_cuda": "mantis (C bridge)", "mantis": "mantis (JAX)", "grid_cuda": "GRiD (C++ bridge)", "grid_jax": "GRiD (JAX FFI)", "frax": "frax",
       "pinocchio": "Pinocchio codegen (CPU)", "rbdl": "RBDL (CPU, float64)"}
OPS = {"id": "ID (RNEA)", "crba": "M (CRBA)", "fd": "FD", "id_du": "∂ID/∂(q, q̇)"}


def load(prefix: str) -> list[dict]:
    rows = []
    for p in sorted(common.RESULTS.glob(f"{prefix}_*.json")):
        rows += json.loads(p.read_text())
    return rows


def table(head: str, rows: list[tuple[str, list[str]]]) -> str:
    out = [f"| {head} | " + " | ".join(common.ROBOTS) + " |",
           "|---|" + "---|" * len(common.ROBOTS)]
    out += [f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows]
    return "\n".join(out)


def cell(rows, **key):
    hit = [r for r in rows if all(r[k] == v for k, v in key.items())]
    return hit[0] if hit else None


def fk_section() -> str:
    rows = load("fk")
    parts = ["### FK: world centres of every collision sphere\n",
             "Spheres per robot: panda 59, baxter 75, g1 68. Time per call in ms (lower is better)."]
    for B in sorted({r["batch"] for r in rows}):
        parts.append(f"\n**B = {B}**\n")
        parts.append(table("method", [(label, [f"{c['ms']:.3f}" if (c := cell(rows, method=m, robot=r, batch=B)) else "—"
                                              for r in common.ROBOTS]) for m, label in FK.items()]))
    parts.append("\n**Max sphere-centre error vs Pinocchio float64 (m), over all batches**\n")
    parts.append(table("method", [(label, [f"{max(x['max_err'] for x in rows if x['method'] == m and x['robot'] == r):.1e}"
                                          if any(x['method'] == m and x['robot'] == r for x in rows) else "—"
                                          for r in common.ROBOTS]) for m, label in FK.items()]))
    return "\n".join(parts)


def ik_tables(rows, methods: dict) -> list[str]:
    parts = []
    for P in sorted({r["batch"] for r in rows if r["method"] in methods}):
        parts.append(f"\n**{P} problem{'s' if P > 1 else ''}**\n")
        parts.append(table("method", [(label, [
            (f"{c['ms']:.2f} ({100 * c['success']:.1f}%, {c['tier']})" if "success" in c else f"{c['ms']:.2f} (ungraded)")
            if (c := cell(rows, method=m, robot=r, batch=P)) else "—"
            for r in common.ROBOTS]) for m, label in methods.items()]))
    return parts


def ik_section() -> str:
    rows = load("ik")
    S = max(r["seeds"] for r in rows if r["method"] == "grim_ls")   # the shared seed count of this run
    return "\n".join([
        "### Global IK: one end-effector pose from random seeds\n",
        f"GRiM: {S} seeds, 60 iterations max, fastest tier; pyroki and cuRobo as their own benchmarks "
        "configure them. Time per call in ms, then success rate (position < 1 mm, "
        "orientation < 0.01 rad, within the URDF limits; Pinocchio float64).",
        *ik_tables(rows, IK),
        "\n### Supplementary: mantis IK steps\n",
        f"mantis's traced IK steps in the benchmark's own minimal CUDA loops (one seed per thread, {S} seeds, "
        "60 iterations), across precisions. A profile of the trace, not a comparison with the solvers above.",
        *ik_tables(rows, IK_STEP)])


def dyn_section() -> str:
    rows = load("dynamics")

    def fmt(c, key):
        if c is None:
            return "—"
        if c.get(key) is None:
            return {"unavailable": "n/a", "failed": "fail", "oom": "OOM"}.get(c.get("status"), "—")
        return f"{c[key]:.3f}" + ("†" if c.get("status") == "accuracy_warning" else "")

    def grid(key, B):
        return table("op · method", [(f"{OPS[op]} · {label}", [
            fmt(cell(rows, method=m, robot=r, batch=B, op=op), key) for r in common.ROBOTS])
            for op in OPS for m, label in DYN.items()])

    parts = ["### Dynamics: GRiD's release protocol\n",
             "Time per call in ms, median over 3 processes of 30 samples each (GRiD's `collect.py` "
             "defaults), every cell checked against an independent float64 Pinocchio oracle under GRiD's "
             "release-table policy (`fp32-fd-warnings`): †: a float32 FD cell past the strict entrywise "
             "gate but within relative L2 1e-3, kept with a warning as GRiD keeps it. GPU "
             "methods are timed device-resident (inputs and outputs on the GPU); the CPU libraries "
             "take host arrays, so theirs is the host call, as in GRiD's Figure 2b. B = 1 and 16384 "
             "are additions to GRiD's batches. n/a: the library has no such operation; OOM: the "
             "call's workspace does not fit in the GPU's 24 GB. mantis, GRiD (JAX FFI) and frax are "
             "called through JAX, so their resident time includes JAX dispatch (about 0.1 ms, which "
             "dominates below B = 1024); GRiD's C++ bridge times its kernel launch directly."]
    for B in sorted({r["batch"] for r in rows}):
        parts += [f"\n**B = {B}**\n", grid("ms", B)]
    parts.append("\n#### Host to host (host arrays in and out, including transfers)\n")
    for B in sorted({r["batch"] for r in rows}):
        parts += [f"\n**B = {B}**\n", grid("host_ms", B)]
    return "\n".join(parts)


TRAJ = {"grim_chomp": "GRiM CHOMP (64 starts)", "grim_stomp": "GRiM STOMP (64 starts)", "grim_sco": "GRiM SCO (64 starts)",
        "pyroki": "pyroki (jaxls, augmented Lagrangian, 64 starts)", "curobo": "cuRobo v2 MotionPlanner (64 starts, no RRT) ‡",
        "trajopt": "TrajOpt (Tesseract, CPU)", "drake": "Drake KinematicTrajectoryOptimization (SNOPT, CPU)"}


REF_BUDGET = 100   # paired tests, per-scene table and path quality are at this budget
BOOT = 2000


def _boot(rows, stat, rng):
    """95% CI of stat(rows) by a cluster bootstrap over problems (a problem's seeds stay together)."""
    by = {}
    for r in rows:
        by.setdefault((r["scene"], r["index"]), []).append(r)
    groups = list(by.values())
    vals = [stat([r for g in rng.integers(0, len(groups), len(groups)) for r in groups[g]]) for _ in range(BOOT)]
    return np.percentile(vals, [2.5, 97.5])


def traj_section(suffix: str = "") -> str:
    """MBM trajopt: every method at each iteration budget (cuRobo: its planner, one point), seed 0.
    Files trajopt_<method>[_params][_i<N>]<suffix>_s<seed>.json; suffix "" is the full MBM run (not
    the --tune set, scene subsets or --limit runs), e.g. "_table_pick_panda_n1" for a smoke run."""
    from scipy.stats import binomtest
    pat = re.compile(r"^trajopt_(?!.*_tune)(?!.*_trace)(.+?)(_i\d+)?" + re.escape(suffix) + r"_s0\.json$")   # seed 0 only
    files = [p for p in sorted(common.RESULTS.glob("trajopt_*_s*.json")) if pat.match(p.name)
             and (suffix or not re.search(r"_n\d+_s\d+\.json$|_panda", p.name))]
    rows = [r for p in files for r in json.loads(p.read_text())]
    if not rows:
        return ""
    rng = np.random.default_rng(0)
    var = lambda r: (r["method"], bool(r.get("params")))
    variants = [(m, t) for m in TRAJ for t in (False, True) if any(var(r) == (m, t) for r in rows)]
    label = lambda m, t: TRAJ[m] + (" (tuned: " + ", ".join(f"{k}={v}" for k, v in next(
        r["params"] for r in rows if var(r) == (m, t)).items()) + ")" if t else "")
    budgets = sorted({r["budget"] for r in rows if r["budget"] is not None})
    pick = lambda v, b: [r for r in rows if var(r) == v and (r["budget"] == b or r["budget"] is None)]
    succ = lambda rs: 100 * np.mean([r["success"] for r in rs])
    med = lambda rs: float(np.median([r["ms"] for r in rs]))
    seeds = len({r["seed"] for r in rows})

    def cell(rs):
        if not rs:
            return "—"
        (a, b), (c, d) = _boot(rs, succ, rng), _boot(rs, med, rng)
        return f"{succ(rs):.0f} [{a:.0f}, {b:.0f}] / {med(rs):.0f} [{c:.0f}, {d:.0f}] ms"

    out = ["### Kinematic trajectory optimization: MotionBenchMaker (Panda)\n",
           f"{len({(r['scene'], r['index']) for r in rows})} MBM problems" + (f" x {seeds} seeds" if seeds > 1 else "") + ", one problem per call, "
           "32 waypoints; GRiM, pyroki and cuRobo from the same 64 starts per problem (the straight line and 63 "
           "sine-bump perturbations of it, solved as one batch; pyroki keeps its best 4 after a few iterations, "
           "its paper's IK beam), TrajOpt and Drake (CPU) from the straight line alone; every method at the same safety margin (1 cm) except "
           "cuRobo (‡); graded in float64 on the shared sphere model, collision-free including dense "
           "interpolation, endpoints within 1e-3 rad (benchmarks/trajopt/mbm.py). A budget of N iterations is N "
           "of the solver's own steps (CHOMP / STOMP / pyroki LM / TrajOpt SQP / Drake SNOPT major iterations, "
           "per solve; SCO inner L-BFGS steps); iterations cost differently per method, so compare along time "
           "(the figure). Each method runs with its library's defaults; tuned rows use the parameters chosen "
           "on a disjoint MπNets tuning set (RUNNING.md, \"Trajopt tuning\"). Cells: success % / median time per "
           "problem, with 95% intervals from a bootstrap over problems (seeds kept together).\n",
           "‡ cuRobo runs as designed: its MotionPlanner.plan_cspace (IK-free joint goal, the shared 64 starts as its "
           "trajopt seeds, one attempt, no RRT fallback, time optimization) with its own benchmark's settings, including its "
           "2.5 mm activation distance; it has no iteration budget, so it is one point (every column).\n",
           "![success vs time](results/trajopt_success_vs_time" + suffix + ".png)\n",
           "![success within a time budget](results/trajopt_success_cdf" + suffix + ".png)\n",
           "| method | " + " | ".join(f"{b} iterations" for b in budgets) + " |", "|---|" + "---|" * len(budgets)]
    for v in variants:
        out.append(f"| {label(*v)} | " + " | ".join(cell(pick(v, b)) for b in budgets) + " |")

    ref = REF_BUDGET if REF_BUDGET in budgets else budgets[-1]
    at = {v: {(r["scene"], r["index"], r["seed"]): r for r in pick(v, ref)} for v in variants}
    best = max(variants, key=lambda v: succ(list(at[v].values())))
    tests = []
    for v in variants:
        if v == best:
            continue
        keys = at[v].keys() & at[best].keys()
        b_ = sum(at[best][k]["success"] and not at[v][k]["success"] for k in keys)
        c_ = sum(at[v][k]["success"] and not at[best][k]["success"] for k in keys)
        p = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0   # exact McNemar
        tests.append([v, b_, c_, p])
    adj = 0.0
    for k, t in enumerate(sorted(tests, key=lambda t: t[3])):   # Holm step-down, kept monotone
        adj = max(adj, min(1.0, t[3] * (len(tests) - k)))
        t[3] = adj
    out += [f"\n**Paired comparison at {ref} iterations** against the most successful method, {label(*best)}: "
            "exact McNemar test on (problem, seed) pairs, Holm-corrected. b: best solved and this method failed; "
            "c: the reverse.\n", "| method | success % | b | c | p (Holm) |", "|---|---|---|---|---|"]
    for v, b_, c_, p in sorted(tests, key=lambda t: -succ(list(at[t[0]].values()))):
        out.append(f"| {label(*v)} | {succ(list(at[v].values())):.1f} | {b_} | {c_} | {p:.2g} |")

    # One variant per method (tuned where tuning changed it), so a weak default doesn't empty the set.
    reps = [(m, (m, True) in variants) for m in dict.fromkeys(m for m, _ in variants)]
    common_keys = set.intersection(*[{k for k, r in at[v].items() if r["success"]} for v in reps])
    out += [f"\n**Path quality at {ref} iterations**, on the {len(common_keys)} (problem, seed) pairs every method "
            "solved (one variant per method: tuned where tuning changed it; each method minimizes its own "
            "objective, so this compares outcomes, not optimizers): median joint-space path length (rad) and "
            "smoothness (mbm.grade).\n",
            "| method | path length | smoothness |", "|---|---|---|"]
    for v in reps:
        q = [at[v][k] for k in common_keys]
        out.append(f"| {label(*v)} | " + (f"{np.median([r['path_length'] for r in q]):.3f} | "
                                         f"{np.median([r['smoothness'] for r in q]):.4f}" if q else "— | —") + " |")

    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    out += [f"\n**Success % per scene at {ref} iterations**\n",
            "| method | " + " | ".join(s.replace("_panda", "") for s in scenes) + " |", "|---|" + "---|" * len(scenes)]
    for v in variants:
        out.append(f"| {label(*v)} | " + " | ".join(f"{succ([r for r in at[v].values() if r['scene'] == sc]):.0f}"
                                                    if any(r["scene"] == sc for r in at[v].values()) else "—"
                                                    for sc in scenes) + " |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for v in variants:
        pts = [(med(pick(v, b)), succ(pick(v, b)), _boot(pick(v, b), succ, rng)) for b in budgets if pick(v, b)]
        if v[0] == "curobo":
            pts = pts[:1]
        x, y, ci = zip(*pts)
        ax.errorbar(x, y, yerr=np.abs(np.array(ci).T - np.array(y)), marker="o", capsize=2,
                    ls="none" if v[0] == "curobo" else "--" if v[1] else "-", label=label(*v))
    ax.set_xscale("log")
    ax.set_xlabel("median time per problem (ms, log)")
    ax.set_ylabel("success %")
    ax.set_title("MBM Panda: success vs time (points: 25 / 50 / 100 / 200 iterations)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(common.RESULTS / f"trajopt_success_vs_time{suffix}.png", dpi=150)

    # Success within time t, per problem, from the budgets already run: given t, a method runs the
    # largest budget that finished within t on that problem, and that run's success counts (no
    # oracle choice of budget). cuRobo, one run to completion, is its own time to solve.
    grid_t = np.logspace(0, np.log10(max(r["ms"] for r in rows)) + 0.05, 400)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for v in variants:
        runs = {}
        for b_ in [x for x in budgets if pick(v, x)] if v[0] != "curobo" else budgets[:1]:
            for r in pick(v, b_):
                runs.setdefault((r["scene"], r["index"], r["seed"]), []).append((r["ms"], r["success"]))
        frac = np.zeros_like(grid_t)
        for rs in runs.values():
            ms, ok = np.array(rs, float).T
            last = np.where(ms[None] <= grid_t[:, None], np.arange(len(ms)), -1).max(1)   # largest budget done by t
            frac += np.where(last >= 0, ok[np.maximum(last, 0)], 0)
        ax.step(grid_t, 100 * frac / len(runs), where="post", ls="--" if v[1] else "-", label=label(*v))
    ax.set_xscale("log")
    ax.set_xlabel("time budget per problem (ms, log)")
    ax.set_ylabel("solved within budget (%)")
    ax.set_title("MBM Panda: solved within a time budget (25 / 50 / 100 / 200 iteration runs)", fontsize=10)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(common.RESULTS / f"trajopt_success_cdf{suffix}.png", dpi=150)
    return "\n".join(out)


def trace_section(suffix: str = "") -> str:
    """MBM trajopt, trace mode (bench_trajopt.py --trace): every method once to 250 iterations, a
    checkpoint every 10 (its answer if stopped there, at the wall time reached). Files
    trajopt_<method>[_params]_trace250<suffix>_s0.json. Writes the success and path-length curves
    over time and the Pareto plot of the 250-iteration results."""
    pat = re.compile(r"^trajopt_(?!.*_tune)(.+?)_trace\d+" + re.escape(suffix) + r"_s0\.json$")
    rows = [r for p in sorted(common.RESULTS.glob("trajopt_*_trace*_s0.json")) if pat.match(p.name)
            for r in json.loads(p.read_text())]
    if not rows:
        return ""
    var = lambda r: (r["method"], bool(r.get("params")))
    variants = [(m, t) for m in TRAJ for t in (False, True) if any(var(r) == (m, t) for r in rows)]
    label = lambda m, t: TRAJ[m] + (" (tuned)" if t else "")
    color = {v: f"C{list(TRAJ).index(v[0])}" for v in variants}   # one color per method in every figure
    runs = {v: {} for v in variants}   # per problem: its checkpoints in time order
    for r in rows:
        runs[var(r)].setdefault((r["scene"], r["index"]), []).append(r)
    for v in variants:
        for cps in runs[v].values():
            cps.sort(key=lambda r: (r["ms"], r["final"]))
    n_prob = len({(r["scene"], r["index"]) for r in rows})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid_t = np.logspace(np.log10(min(r["ms"] for r in rows)) - 0.05, np.log10(max(r["ms"] for r in rows)) + 0.05, 500)

    def at_time(cps):   # (G,) index of the latest checkpoint reached by each time, -1 before the first
        ms = np.array([r["ms"] for r in cps])
        return np.where(ms[None] <= grid_t[:, None], np.arange(len(ms)), -1).max(1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5))
    for v in variants:
        ok = np.zeros(len(grid_t))
        length = [[] for _ in grid_t]
        for cps in runs[v].values():
            idx = at_time(cps)
            for g, i in enumerate(idx):
                if i >= 0 and cps[i]["success"]:
                    ok[g] += 1
                    length[g].append(cps[i]["path_length"])
        ls = "--" if v[1] else "-"
        a1.step(grid_t, 100 * ok / len(runs[v]), where="post", ls=ls, color=color[v], label=label(*v))
        a2.step(grid_t, [np.median(x) if x else np.nan for x in length], where="post", ls=ls, color=color[v], label=label(*v))
    for a, y in ((a1, "solved by time t (%)"), (a2, "median path length of the solved (rad)")):
        a.set_xscale("log")
        a.set_xlabel("time per problem (ms, log)")
        a.set_ylabel(y)
        a.grid(alpha=0.3)
    a1.set_title("Success over time (a checkpoint every 10 iterations)", fontsize=10)
    a2.set_title("Path quality over time", fontsize=10)
    a2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(common.RESULTS / f"trajopt_trace_over_time{suffix}.png", dpi=150)

    final = {v: [cps[-1] for cps in runs[v].values()] for v in variants}
    stat = {}
    out = ["### Trajectory optimization at 250 iterations (trace run)\n",
           f"{n_prob} MBM problems; each method's result after 250 of its own iterations. Path length and "
           "smoothness: medians over the problems that method solved; time: median per problem.\n",
           f"![over time](results/trajopt_trace_over_time{suffix}.png)\n",
           f"![Pareto](results/trajopt_trace_pareto{suffix}.png)\n",
           "| method | success % | path length (rad) | smoothness | median time (ms) |", "|---|---|---|---|---|"]
    for v in variants:
        rs, sol = final[v], [r for r in final[v] if r["success"]]
        stat[v] = (float(np.median([r["ms"] for r in rs])), 100 * len(sol) / len(rs),
                   float(np.median([r["path_length"] for r in sol])) if sol else np.nan)
        out.append(f"| {label(*v)} | {stat[v][1]:.1f} | {stat[v][2]:.3f} | "
                   + (f"{np.median([r['smoothness'] for r in sol]):.4f}" if sol else "—") + f" | {stat[v][0]:.1f} |")

    # Pareto: faster and more successful (left panel), faster and shorter (right panel)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, k, better in ((a1, 1, max), (a2, 2, min)):
        pts = sorted((stat[v][0], stat[v][k], v) for v in variants)
        front, best = [], None
        for t, y, v in pts:   # by increasing time: on the front if it beats everything faster
            if best is None or better(y, best) != best:
                front.append((t, y))
                best = y
        for t, y, v in pts:
            a.plot(t, y, "o", ms=8, color=color[v], mec="white", mew=1.5, label=label(*v))
        a.step(*zip(*front), where="post", color="0.5", ls=":", lw=1.5)
        a.set_xscale("log")
        a.set_xlabel("median time per problem (ms, log)")
        a.grid(alpha=0.3)
    a1.set_ylabel("success %")
    a2.set_ylabel("median path length of the solved (rad)")
    a1.set_title("Pareto: time vs success (dotted: frontier)", fontsize=10)
    a2.set_title("Pareto: time vs path length", fontsize=10)
    a2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(common.RESULTS / f"trajopt_trace_pareto{suffix}.png", dpi=150)
    return "\n".join(out)


def main():
    if sys.argv[1:2] == ["trace"]:   # only the trace section; README untouched
        print(trace_section(sys.argv[2] if len(sys.argv) > 2 else ""))
        return
    if sys.argv[1:2] == ["traj"]:   # only the trajopt section, of the runs with this suffix; README untouched
        print(traj_section(sys.argv[2] if len(sys.argv) > 2 else ""))
        return
    fast = bool(os.environ.get("BENCH_FAST"))   # run_all.sh --fast: print only, README untouched
    body = "\n\n".join(x for x in [fk_section(), ik_section(), dyn_section(), traj_section("_n1" if fast else ""),
                                       trace_section("_n1" if fast else "")] if x)
    if fast:
        print(body)
        return
    readme = common.HERE / "README.md"
    text = readme.read_text()
    new = re.sub(r"(<!-- tables -->\n).*(<!-- /tables -->)", lambda m: m[1] + body + "\n" + m[2],
                 text, flags=re.S)
    readme.write_text(new)
    print(body)


if __name__ == "__main__":
    main()
