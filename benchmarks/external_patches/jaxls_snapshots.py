"""Debug snapshots for jaxls (pyroki's solver): the solution after every k-th outer LM step,
returned in the solve summary, for success-vs-time curves from one recorded run.

Writes a patched copy of the installed jaxls (upstream 50a58be; both the >= 3.12 and the py310
sources) to ``benchmarks/external/jaxls_snapshots/jaxls``; the installed copy stays untouched. Put
that directory first on ``PYTHONPATH`` for a recorded pass:

    python external_patches/jaxls_snapshots.py apply  [--site <dir holding the installed jaxls/>]
    python external_patches/jaxls_snapshots.py revert          # deletes the copy
    PYTHONPATH=benchmarks/external/jaxls_snapshots:$PYTHONPATH python ...

After ``apply``:

    with jaxls.record_snapshots(every=5):          # decided at trace time: jit inside the block
        vals, summary = problem.solve(..., return_summary=True)
    summary.snapshots    # VarValues stacked (S, ...), S = max_iterations // 5; slot s is the
                         # solution a run with max_iterations = 5 (s + 1) returns (jaxls counts
                         # LM tries); slots past an early stop repeat the final solution. None outside a record_snapshots block.

The snapshots ride in the while-loop carry (a device buffer, no host callback per step). They add
work: time solves with no block active, where the solver is unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import site
import sys
from pathlib import Path

FILES = ("_solvers.py", "_py310/_solvers.py", "__init__.py")

RECORDER = '''

_active_snapshot_every: int = 0


@contextlib.contextmanager
def record_snapshots(every: int):
    """Stack the solution after every `every`-th outer step into `SolveSummary.snapshots` of any
    `solve(..., return_summary=True)` traced in this block (external_patches/jaxls_snapshots.py)."""
    global _active_snapshot_every
    prev, _active_snapshot_every = _active_snapshot_every, int(every)
    try:
        yield
    finally:
        _active_snapshot_every = prev

'''

LOOP_OLD = '''            state = jax.lax.while_loop(
                cond_fun=should_continue,
                body_fun=lambda state: self.lm_outer_step(problem, state),
                init_val=state,
            )
        else:
            state = jax.lax.fori_loop(
                0,
                self.termination.max_iterations,
                body_fun=lambda _, state: self.lm_outer_step(problem, state),
                init_val=state,
            )
'''

LOOP_NEW = '''            loop = lambda body, init: jax.lax.while_loop(
                cond_fun=lambda c: should_continue(c[0]), body_fun=body, init_val=init
            )
        else:
            loop = lambda body, init: jax.lax.fori_loop(
                0, self.termination.max_iterations, lambda _, c: body(c), init
            )
        # Snapshots (record_snapshots): the solution after every k-th outer step, in the carry.
        every = _active_snapshot_every
        n_snap = self.termination.max_iterations // every if every > 0 else 0
        hist = (
            jax.tree.map(lambda v: jnp.zeros((n_snap,) + v.shape, v.dtype), state.solution.vals)
            if n_snap
            else None
        )

        def body(carry):
            # summary.iterations counts LM tries and can step by several; a run with budget m stops
            # after the first outer step reaching m, so that step's solution fills every slot whose
            # budget (s + 1) * every it crossed.
            st, h = carry
            prev = st.summary.iterations
            st = self.lm_outer_step(problem, st)
            if n_snap:
                s = jnp.arange(n_snap)
                put = (s >= prev // every) & (s < st.summary.iterations // every)
                h = jax.tree.map(
                    lambda a, v: jnp.where(put.reshape((-1,) + (1,) * v.ndim), v[None], a),
                    h,
                    st.solution.vals,
                )
            return st, h

        state, hist = loop(body, (state, hist))
        if n_snap:
            done = state.summary.iterations // every
            hist = jax.tree.map(
                lambda a, v: jnp.where(
                    (jnp.arange(n_snap) >= done).reshape((-1,) + (1,) * v.ndim), v[None], a
                ),
                hist,
                state.solution.vals,
            )
            state = jdc.replace(state, summary=jdc.replace(state.summary, snapshots=hist))
'''


def edits(name: str) -> list[tuple[str, str]]:
    if name == "__init__.py":
        return [("    from ._solvers import record_iteration_times as record_iteration_times\n",
                 "    from ._solvers import record_iteration_times as record_iteration_times\n"
                 "    from ._solvers import record_snapshots as record_snapshots\n"),
                ("    from ._py310._solvers import record_iteration_times as record_iteration_times\n",
                 "    from ._py310._solvers import record_iteration_times as record_iteration_times\n"
                 "    from ._py310._solvers import record_snapshots as record_snapshots\n")]
    hist = "    lambda_history: jax.Array\n" if name == "_solvers.py" else "    lambda_history: Any\n"
    anchor = ("_active_iteration_time_recorder: list[float] | None = None\n" if name == "_solvers.py"
              else "_active_iteration_time_recorder: Any = None\n")
    return [(hist, hist + "    snapshots: object = None\n"
             "    \"\"\"record_snapshots: the solution after every k-th outer step, stacked.\"\"\"\n"),
            (anchor, anchor + RECORDER),
            (LOOP_OLD, LOOP_NEW)]


OUT = Path(__file__).resolve().parent.parent / "external/jaxls_snapshots"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("apply", "revert"))
    ap.add_argument("--site", default=None)
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    if a.action == "revert":
        print(f"jaxls snapshots: removed {OUT}")
        return
    src_root = Path(a.site or next(p for p in site.getsitepackages() if (Path(p) / "jaxls").is_dir())) / "jaxls"
    root = OUT / "jaxls"
    shutil.copytree(src_root, root, ignore=shutil.ignore_patterns("__pycache__"))
    for f in FILES:
        path = root / f
        src = path.read_text()
        for old, new in edits(f):
            if src.count(old) != 1:
                shutil.rmtree(OUT)
                sys.exit(f"{src_root / f}: anchor not found once (jaxls is not upstream 50a58be?): {old[:60]!r}")
            src = src.replace(old, new)
        path.write_text(src)
    print(f"jaxls snapshots: {root} (put {OUT} first on PYTHONPATH)")


if __name__ == "__main__":
    main()
