"""Print the saved results (results/*.json) as compact pivot tables: method x batch, one table per
benchmark, robot (and op for dynamics). ms per call; the fastest in each column is underlined (* when not a terminal).

    python report.py                  # everything
    python report.py ik dyn g1        # only some benchmarks and/or robots (fk ik dyn traj)

Reads results only; runs nothing and needs no GPU (any env with numpy, e.g. `grim`).
"""

from __future__ import annotations

import re
import sys

import common
from tables import DYN, FK, IK, IK_STEP, OPS, load

BENCH = {"fk": ("FK", "fk", FK), "ik": ("Global IK", "ik", IK),
         "ik_step": ("mantis IK steps (supplementary)", "ik", IK_STEP), "dyn": ("Dynamics", "dynamics", DYN)}
UNDERLINE = sys.stdout.isatty()  # piped or redirected: mark the best with * instead


def vis(s: str) -> int:
    """Printed width, without ANSI escapes."""
    return len(re.sub(r"\033\[\d+m", "", s))


STATUS = {"oom": "OOM", "unavailable": "n/a", "failed": "fail"}


def fmt(r: dict | None, best: bool) -> str:
    if r is None:
        return "—"
    if r.get("ms") is None:
        return STATUS.get(r.get("status"), "—")
    s = f"{r['ms']:.3f}"
    if best:
        s = f"\033[4m{s}\033[24m" if UNDERLINE else s + "*"
    if "success" in r:
        s += f" {100 * r['success']:.0f}%"
    if r.get("status") == "accuracy_warning":
        s += "†"
    return s


def pivot(title: str, rows: list[dict], labels: dict) -> None:
    rows = [r for r in rows if r["method"] in labels]
    if not rows:
        return
    batches = sorted({r["batch"] for r in rows})
    by = {(r["method"], r["batch"]): r for r in rows}
    best = {B: min((r["ms"] for r in rows if r["batch"] == B and r.get("ms") is not None), default=None)
            for B in batches}
    methods = [m for m in labels if any(r["method"] == m for r in rows)]
    grid = [[labels[m]] + [fmt(by.get((m, B)), by.get((m, B), {}).get("ms") == best[B]) for B in batches]
            for m in methods]
    head = ["method \\ batch"] + [str(B) for B in batches]
    w = [max(map(vis, col)) for col in zip(head, *grid)]
    print(f"\n{title}")
    for line in [head, ["-" * n for n in w], *grid]:
        print("  ".join(x.ljust(w[0]) if i == 0 else x.rjust(w[i] + len(x) - vis(x)) for i, x in enumerate(line)))


def main():
    args = sys.argv[1:]
    robots = [a for a in args if a in common.ROBOTS] or list(common.ROBOTS)
    wanted = [a for a in args if a not in common.ROBOTS] or ["fk", "ik", "dyn", "traj"]
    if "traj" in wanted:   # Panda only, per MBM scene: the README table as it is
        from tables import traj_section
        print("\n" + traj_section())
    if "ik" in wanted:
        wanted.append("ik_step")
    for key in BENCH:
        if key not in wanted:
            continue
        name, prefix, labels = BENCH[key]
        rows = load(prefix)
        print(f"\n=== {name} ===")
        for robot in robots:
            rs = [r for r in rows if r["robot"] == robot]
            if key == "dyn":
                for op, op_label in OPS.items():
                    pivot(f"{robot} · {op_label}", [r for r in rs if r["op"] == op], labels)
            else:
                pivot(robot, rs, labels)
    print("\nms per call; fastest in column underlined (* when piped); IK: % = success rate; † float32 FD past the strict gate "
          "(GRiD's fp32-fd-warnings); OOM / n/a / fail: no time. Details: README.md, RUNNING.md.")


if __name__ == "__main__":
    main()
