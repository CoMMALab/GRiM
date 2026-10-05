"""GRiM: GPU motion generation (``grim.motion``) built on GRiD.

GRiD (``external/GRiD``, a submodule) supplies the rigid-body dynamics codegen
(``grid_codegen``), its runtime wrapper (``grid_rbd``) and the URDFParser /
RBDReference / GLASS peers; GRiM adds the per-robot IK, trajectory optimization,
collision and least-squares kernels in ``grim.motion``. Importing ``grim`` puts the
GRiD checkout on ``sys.path`` unless ``grid_codegen`` is already importable, so a
source checkout works without installing GRiD. ``GRIM_GRID_PATH`` overrides the
checkout location.
"""

import importlib.util
import os
import sys
from pathlib import Path

GRID_DIR = Path(os.environ.get("GRIM_GRID_PATH", Path(__file__).resolve().parents[2] / "external" / "GRiD"))

if importlib.util.find_spec("grid_codegen") is None and (GRID_DIR / "grid_codegen").is_dir():
    for _d in (GRID_DIR / "external", GRID_DIR / "bindings", GRID_DIR):
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
