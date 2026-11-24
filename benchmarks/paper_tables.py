"""The paper's tables (LaTeX) and figures, regenerated from benchmarks/results.

    python benchmarks/paper_tables.py                 # -> benchmarks/results/paper/*.tex, *.pdf
    python benchmarks/paper_tables.py --paper <dir>   # also into the paper: <dir>/tables/*.tex, <dir>/images/*.pdf

| output                     | paper                                   | source                                         |
|----------------------------|-----------------------------------------|------------------------------------------------|
| ik.tex                     | IK at B = 1, 128, 1024                  | results/ik_<method>_panda_baxter_g1.json       |
| dynamics.tex               | RNEA / CRBA / ABA                       | results/dynamics_<method>_panda_baxter_g1.json |
| did.tex                    | d tau / d(q, qd)                        | the same (id_du), with mantis_cuda_{fwd,rev}_ad |
| diffsim.tex                | differentiable simulator, Mantis vs JAX | results/diffsim_jax_vs_mantis.json (*)         |
| fk.tex                     | FK sphere centres                       | results/fk_<method>_panda_baxter_g1.json       |
| trajopt_success_cdf.pdf    | trajopt success within a time budget    | results/trajopt_*_trace250_s0.json (tables.py) |

(*) a copy of unconventional-differentiable-simulator/benchmarks/jax_vs_mantis.json (its
jax_vs_mantis.py: 50-step float64 rollouts, environment steps per second, host to host).

Each .tex is the whole table environment, the file the paper \\input's (tables/<name>.tex).
Bold marks the lowest latency (highest throughput) the way each caption says.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OUT = common.RESULTS / "paper"
ROBOTS = ("panda", "baxter", "g1")


def load(prefix, method):
    """{(robot, batch, op): row} of results/<prefix>_<method>_panda_baxter_g1.json, or {} if absent."""
    f = common.RESULTS / f"{prefix}_{method}_panda_baxter_g1.json"
    return {(r["robot"], r["batch"], r.get("op")): r for r in json.loads(f.read_text())} if f.exists() else {}


def bold(s, on):
    return f"\\textbf{{{s}}}" if on else s


def ik():
    methods = [("grim_ls", "GRiM LS"), ("grim_dls", "GRiM DLS"), ("grim_sqp", "GRiM SQP"), ("grim_hjcd", "GRiM HJCD"),
               ("grim_mppi", "GRiM MPPI"), ("pyroki", "PyRoki IK-Beam"), ("curobo", "cuRobo v2")]
    d = {m: load("ik", m) for m, _ in methods}
    batches = (1, 128, 1024)
    best = {(r, b): min(d[m][(r, b, None)]["ms"] for m, _ in methods) for r in ROBOTS for b in batches}
    rows = []
    for m, label in methods:
        for k, b in enumerate(batches):
            cells = [bold(f"{d[m][(r, b, None)]['ms']:.2f}", d[m][(r, b, None)]["ms"] == best[r, b])
                     + f" ({100 * d[m][(r, b, None)]['success']:.1f})" for r in ROBOTS]
            rows.append(("    \\multirow{3}{*}{" + label + "}" if k == 0 else "    ") + f" & {b} & " + " & ".join(cells) + " \\\\")
        rows.append("    \\midrule")
    rows[-1] = "    \\bottomrule"
    return "\n".join([
        "\\begin{table}[t]", "  \\centering",
        "  \\caption{IK at batch sizes $B=1,128,1024$: time per call in ms (success rate in \\%). Bold marks the lowest latency across methods for each robot and batch size.}",
        "  \\label{tab:ik}", "  \\footnotesize", "  \\setlength{\\tabcolsep}{2pt}", "  \\begin{tabular}{lrccc}", "    \\toprule",
        "    Method & $B$ & Panda & Baxter & G1 \\\\", "    \\midrule", *rows, "  \\end{tabular}", "\\end{table}"])


def _dyn_rows(ops, methods, d, cols):
    rows = []
    for op, name in ops:
        best = {c: min(d[m][(c[1], c[0], op)]["ms"] for m, _ in methods
                       if (c[1], c[0], op) in d[m] and d[m][(c[1], c[0], op)]["status"] not in ("oom", "failed"))
                for c in cols}
        rows.append(f"    \\multirow{{{len(methods)}}}{{*}}{{{name}}}")
        for m, label in methods:
            cells = []
            for c in cols:
                r = d[m].get((c[1], c[0], op))
                if r is None or r["status"] in ("failed", "unavailable"):
                    cells.append("---")
                elif r["status"] == "oom":
                    cells.append("OOM")
                else:
                    cells.append(bold(f"{r['ms']:.3f}", r["ms"] == best[c]) + ("$^\\dagger$" if r["status"] == "accuracy_warning" else ""))
            rows.append(f"      & {label} & " + " & ".join(cells) + " \\\\")
        rows.append("    \\midrule")
    rows[-1] = "    \\bottomrule"
    return rows


DYN_HEAD = ["    & & \\multicolumn{3}{c}{$B = 1$} & \\multicolumn{3}{c}{$B = 1024$} & \\multicolumn{3}{c}{$B = 16384$} \\\\",
            "    \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
            "    & Method & Panda & Baxter & G1 & Panda & Baxter & G1 & Panda & Baxter & G1 \\\\", "    \\midrule"]
DYN_COLS = [(b, r) for b in (1, 1024, 16384) for r in ROBOTS]


def dynamics():
    methods = [("mantis_cuda", "GRiM (Mantis)"), ("grid_cuda", "GRiD"), ("frax", "FRaX"), ("pinocchio", "Pinocchio (CPU)")]
    d = {m: load("dynamics", m) for m, _ in methods}
    rows = _dyn_rows([("id", "RNEA"), ("crba", "CRBA"), ("fd", "ABA")], methods, d, DYN_COLS)
    return "\n".join([
        "\\begin{table*}[t]", "  \\centering",
        "  \\caption{Rigid-body dynamics: time per call (ms) at batch sizes $B$. Lower is better; bold marks the lowest latency within each algorithm and robot/batch combination. $\\dagger$: float32 result within relative $L_2$ error $10^{-3}$ but outside the strict entrywise tolerance, kept as in GRiD's release tables. OOM: out of GPU memory.}",
        "  \\label{tab:dynamics}", "  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}", "  \\begin{tabular}{llccccccccc}", "    \\toprule",
        *DYN_HEAD, *rows, "  \\end{tabular}", "\\end{table*}"])


def did():
    methods = [("mantis_cuda", "GRiM (analytic)"), ("mantis_cuda_fwd_ad", "GRiM (Forward AD)"),
               ("mantis_cuda_rev_ad", "GRiM (Reverse AD)"), ("grid_cuda", "GRiD"), ("frax", "FRaX"),
               ("pinocchio", "Pinocchio (CPU)")]
    d = {m: load("dynamics", m) for m, _ in methods}
    missing = [m for m, _ in methods if not d[m]]
    if missing:
        print(f"did.tex: no results for {missing} (dynamics/bench_dynamics.py <method>); their rows are ---")
    rows = _dyn_rows([("id_du", "$\\partial$RNEA")], methods, d, DYN_COLS)
    return "\n".join([
        "\\begin{table*}[t]", "  \\centering",
        "  \\caption{Inverse-dynamics derivatives $\\partial\\tau/\\partial(q,\\dot q)$: time per call (ms). GRiM (analytic) traces Mantis's analytical RNEA derivatives; GRiM (Forward AD) and (Reverse AD) compile Mantis's ahead-of-time forward- and reverse-mode differentiation of the RNEA trace; FRaX is \\texttt{jax.jacfwd} of its RNEA. Bold marks the lowest latency per robot and batch size. OOM: out of GPU memory.}",
        "  \\label{tab:did}", "  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}", "  \\begin{tabular}{llccccccccc}", "    \\toprule",
        *DYN_HEAD, *rows, "  \\end{tabular}", "\\end{table*}"])


def _rate(x):   # environment steps per second, 3 significant figures
    for div, suf in ((1e6, "\\,M"), (1e3, "\\,k")):
        if x >= div:
            v = x / div
            return (f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}") + suf
    return f"{x:.0f}"


def diffsim():
    rows = []
    for r in json.loads((common.RESULTS / "diffsim_jax_vs_mantis.json").read_text()):
        mantis = max(v for k, v in r.items() if k.startswith("mantis-device fwd"))   # fastest layout per batch
        rows.append(f"    {r['B']} & {_rate(r['jax fwd'])} & \\textbf{{{_rate(mantis)}}} & {mantis / r['jax fwd']:.1f}$\\times$ \\\\")
    return "\n".join([
        "\\begin{table}[t]", "  \\centering",
        "  \\caption{Differentiable simulator forward rollouts (50 steps, float64): environment steps per second, end to end. Higher is better.}",
        "  \\label{tab:diffsim}", "  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}", "  \\begin{tabular}{rrrr}", "    \\toprule",
        "    $B$ & JAX (XLA) & Mantis & Speedup \\\\", "    \\midrule", *rows, "    \\bottomrule", "  \\end{tabular}", "\\end{table}"])


def fk():
    methods = [("prrtc", "pRRTC"), ("curobo", "cuRobo"), ("pyroki", "PyRoki"), ("grim_cuda", "GRiD tables"),
               ("mantis_cuda", "GRiM (Mantis)")]
    d = {m: load("fk", m) for m, _ in methods}
    cols = [(b, r) for b in (1, 16384) for r in ROBOTS]
    best = {c: min(d[m][(c[1], c[0], None)]["ms"] for m, _ in methods) for c in cols}
    rows = [f"    {label} & " + " & ".join(bold(f"{d[m][(c[1], c[0], None)]['ms']:.3f}", d[m][(c[1], c[0], None)]["ms"] == best[c])
                                         for c in cols) + " \\\\" for m, label in methods]
    return "\n".join([
        "\\begin{table}[h]", "  \\centering",
        "  \\caption{FK sphere centers: time per call (ms). Bold marks the lowest latency per robot and batch size.}",
        "  \\label{tab:app:fk}", "  \\footnotesize", "  \\setlength{\\tabcolsep}{3pt}", "  \\begin{tabular}{lcccccc}", "    \\toprule",
        "    & \\multicolumn{3}{c}{$B = 1$} & \\multicolumn{3}{c}{$B = 16384$} \\\\", "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "    Method & Panda & Baxter & G1 & Panda & Baxter & G1 \\\\", "    \\midrule", *rows, "    \\bottomrule",
        "  \\end{tabular}", "\\end{table}"])


def main():
    paper = Path(sys.argv[sys.argv.index("--paper") + 1]) if "--paper" in sys.argv else None
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("ik", ik), ("dynamics", dynamics), ("did", did), ("diffsim", diffsim), ("fk", fk)]:
        (OUT / f"{name}.tex").write_text(fn() + "\n")
        print(f"wrote {OUT / name}.tex")
    import tables   # the trajopt figures: trace_section writes trajopt_trace_success.{png,pdf}
    tables.trace_section()
    shutil.copy(common.RESULTS / "trajopt_trace_success.pdf", OUT / "trajopt_success_cdf.pdf")
    print(f"wrote {OUT / 'trajopt_success_cdf.pdf'}")
    if paper:   # did.tex too, once the paper inputs it
        for name in ("ik", "dynamics", "did", "diffsim", "fk"):
            if name != "did" or (paper / "tables/did.tex").exists():
                shutil.copy(OUT / f"{name}.tex", paper / "tables" / f"{name}.tex")
        shutil.copy(OUT / "trajopt_success_cdf.pdf", paper / "images" / "trajopt_success_cdf.pdf")
        print(f"copied into {paper}/tables and {paper}/images")


if __name__ == "__main__":
    main()
