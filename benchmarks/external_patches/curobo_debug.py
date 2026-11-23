"""Per-iteration capture for cuRobo v2's trajopt (needs curobo_debug_best.patch applied).

cuRobo already has a per-iteration recorder (``store_debug``, components/debug_recorder.py); the
patch makes it also keep ``best_action`` / ``best_cost``, what the optimizer returns if stopped
(``return_best_action``). Recording runs in Python inside the optimizer's step, so the planner must
be built with ``use_cuda_graph=False`` for a recorded pass (a captured graph would replay without
recording). Each recorded iteration is also timestamped (perf_counter after a GPU sync), the
clock bench_trajopt.py --trace reports.

    mg = MotionPlanner(MotionPlannerCfg.create(..., use_cuda_graph=False))
    enable(mg)
    mg.plan_cspace(...)
    rec = take(mg)    # every optimize() call of the plan, in order (trajopt, then finetune):
                      # {"best": [(B, H, n) knots per iteration], "best_cost": [(B,)], "action": [...],
                      #  "time": [perf_counter after each iteration]}
    q = positions(mg, rec["best"][i])   # (B, T', n) joint positions of those knots (its rollout)
"""

from __future__ import annotations


def _optimizers(mg):
    return mg.trajopt_solver.optimizer.optimizers


def enable(mg) -> None:
    """Attach a recorder to every trajopt optimizer; it keeps all optimize() calls until take()."""
    import time
    import torch
    from curobo._src.optim.components.debug_recorder import DebugRecorder
    for opt in _optimizers(mg):
        rec = DebugRecorder()
        if not hasattr(rec, "best_actions"):
            raise RuntimeError("apply external_patches/curobo_debug_best.patch first")
        rec.clear = lambda: None   # the core clears on every reset; keep the whole plan
        rec.times = []   # perf_counter after each iteration, the GPU synchronized (adds a sync per iteration)
        record = rec.record
        def timed(*a, rec=rec, record=record, **k):
            record(*a, **k)
            torch.cuda.synchronize()
            rec.times.append(time.perf_counter())
        rec.record = timed
        opt._core._debug = rec


def take(mg) -> dict:
    """The recorded iterations of every optimizer since the last take(), and reset."""
    out = {"best": [], "best_cost": [], "action": [], "time": []}
    for opt in _optimizers(mg):
        r = opt._core._debug
        out["best"] += r.best_actions
        out["best_cost"] += r.best_costs
        out["action"] += r.actions
        out["time"] += r.times
        r.times = []
        r.best_actions, r.best_costs, r.actions, r.costs = [], [], [], []
        r.line_search_indices, r.line_search_statuses, r.curvature_update_valid = [], [], []
    return out


def positions(mg, knots):
    """Joint positions (B, T', n) of trajopt knots (B, H, n), through the solver's metrics rollout."""
    m = mg.trajopt_solver.metrics_rollout.compute_metrics_from_action(knots)
    return m.state.joint_state.position
