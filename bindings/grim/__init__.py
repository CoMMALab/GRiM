"""GRiM: GPU motion generation (``grim.motion``) built on GRiD.

GRiD (``external/GRiD``, a submodule) supplies the rigid-body dynamics codegen
(``grid_codegen``), its runtime wrapper (``grid_rbd``) and the URDFParser /
RBDReference / GLASS peers; GRiM adds the per-robot IK, trajectory optimization,
collision and least-squares kernels in ``grim.motion``. Importing ``grim`` puts the
GRiD checkout on ``sys.path`` unless ``grid_codegen`` and its peers are already importable, so a
source checkout works without installing GRiD. ``GRIM_GRID_PATH`` overrides the
checkout location. crocket (``external/crocket``), the trace compiler behind every IK kernel,
is put on ``sys.path`` the same way unless it is installed (``GRIM_CROCKET_PATH`` overrides).
"""

import importlib.util
import os
import sys
from pathlib import Path

GRID_DIR = Path(os.environ.get("GRIM_GRID_PATH", Path(__file__).resolve().parents[2] / "external" / "GRiD"))

if any(importlib.util.find_spec(m) is None for m in ("grid_codegen", "URDFParser", "RBDReference")) and (GRID_DIR / "grid_codegen").is_dir():
    for _d in (GRID_DIR / "external", GRID_DIR / "bindings", GRID_DIR):
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))

CROCKET_DIR = Path(os.environ.get("GRIM_CROCKET_PATH", Path(__file__).resolve().parents[2] / "external" / "crocket"))

if importlib.util.find_spec("crocket") is None and (CROCKET_DIR / "python" / "crocket").is_dir():
    sys.path.insert(0, str(CROCKET_DIR / "python"))
