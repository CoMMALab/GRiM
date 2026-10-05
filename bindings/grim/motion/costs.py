"""Least-squares and contact solvers compiled per problem structure.

These kernels are robot-agnostic: a *problem compiler* (pyroffi.costs is one, emitting CUDA
from jaxprs) writes the problem's residuals as device code, and GRiM compiles them into its
solvers. The header contracts:

``cost_gen.cuh`` (dense, :func:`dense_solver`), in ``namespace grim::costs_gen``:
    ``n_x``, ``n_r``, ``row_kind[n_r]`` (0 cost, 1 equality == 0, 2 inequality <= 0),
    ``has_constraints``, ``scratch_size`` and
    ``residual(x, p, r, scratch, rank, size)``,
    ``residual_jacobian(x, p, r, J, scratch, rank, size)`` (J row-major ``(n_r, n_x)``),
    written in GLASS form: lanes run straight-line code redundantly and split loops
    ``for (i = rank; i < n; i += size)`` each followed by ``GRIM_COST_SYNC()``.
    Optional variants, enabled by the header defining the macro: ``GRIM_COST_JVP``
    (``residual_jvp``, column-parallel tiers), ``GRIM_COST_NORMAL_EQUATIONS``
    (``normal_equations``, symbolic H/g), ``GRIM_COST_SPARSE_NE`` (``g_ptr``/``g_rows``,
    ``h_ptr``/``h_rows``/``h_a``/``h_c``/``h_nnz``: the structural H pattern).

``cost_gen.cuh`` (banded, :func:`banded_solver`): cost GROUPS of identical instances over
    local variables -- ``stage_res(g, xl, pl, r)``, ``stage_jvp(g, xl, pl, t, r, jt)``,
    the index tables ``vtab``/``ptab`` and the group layout (``n_groups``, ``g_nloc``,
    ``g_nploc``, ``g_nr``, ``g_vtab0``, ``g_ptab0``, ``g_row0``, ``g_j0``,
    ``g_inst_prefix``, ``g_col_prefix``, ``n_inst_total``, ``n_col_total``, ``max_nloc``,
    ``max_nploc``, ``max_nr``, ``j_size``), ``band``, ``fixed[n_x]``, ``n_x``, ``n_r``,
    ``row_kind``, ``has_constraints``.

``pointwise_gen.cuh`` (:func:`pointwise`), in ``namespace grim::pointwise_gen``:
    ``n_in``, ``n_out`` and ``pointwise(x, o, ...)``, one thread per row.

Every solver's FFI target takes ``x0 (B, n_x)``, ``params (B | 1, max(n_p, 1))`` and returns
``x``, ``cost``, ``viol`` and a byte workspace; settings are attributes (see the kernels).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

from . import _build

TIERS = {"thread": 0, "warp": 1, "block": 2}


def _precision_flags(real: str, io: str) -> tuple[str, ...]:
    return (f"-DGRIM_COST_REAL={real}", f"-DGRIM_COST_IO={io}",
            *(("-DGRIM_COST_IO_F64",) if io == "double" else ()))


@lru_cache(maxsize=None)
def dense_solver(header: str, tier: str, real: str = "float", io: str = "float") -> str:
    """LM / augmented Lagrangian at one tier (``kernels/costs/cost_lm.cu``) -> FFI target."""
    so = _build.compile_kernel("costs/cost_lm", {"cost_gen.cuh": header},
                               (f"-DGRIM_COST_TIER={TIERS[tier]}", *_precision_flags(real, io)))
    return _build.register(so, ("CostLmFfi",))[0]


@lru_cache(maxsize=None)
def banded_solver(header: str, real: str = "double", io: str = "double",
                  profile: bool = False) -> tuple[str, int]:
    """LM / augmented Lagrangian with a banded Cholesky, one block per problem
    (``kernels/costs/cost_lm_banded.cu``) -> (FFI target, workspace bytes per problem)."""
    so = _build.compile_kernel("costs/cost_lm_banded", {"cost_gen.cuh": header},
                               (*_precision_flags(real, io),
                                *(("-DGRIM_COST_PROFILE",) if profile else ())))
    lib = ctypes.CDLL(str(so))
    lib.grim_cost_lm_banded_workspace_bytes.restype = ctypes.c_size_t
    return _build.register(so, ("CostLmFfi",))[0], int(lib.grim_cost_lm_banded_workspace_bytes())


@lru_cache(maxsize=None)
def pointwise(header: str) -> str:
    """A generated function applied row-wise (``kernels/costs/pointwise.cu``) -> FFI target."""
    so = _build.compile_kernel("costs/pointwise", {"pointwise_gen.cuh": header})
    return _build.register(so, ("PointwiseFfi",))[0]


C3_SYMBOLS = ("C3ProjectFfi", "C3BoxFfi", "C3RiccatiFfi", "C3PcgFfi", "C3AdmmFfi", "C3PgsFfi",
              "C3MeritFfi")


@lru_cache(maxsize=None)
def c3(nx: int, nu: int, nl: int, nt: int) -> tuple[dict[str, str], ctypes.CDLL]:
    """The float64 C3+ inner-solve kernels for one problem shape (``kernels/trajopt/c3.cu``)
    -> ({symbol: FFI target}, the library, whose ``c3_workspace_doubles(B, N, backend)``
    sizes the ADMM workspace)."""
    so = _build.compile_kernel("trajopt/c3", {}, (f"-DC3_NX={nx}", f"-DC3_NU={nu}",
                                                   f"-DC3_NL={nl}", f"-DC3_NT={nt}"))
    lib = ctypes.CDLL(str(so))
    lib.c3_workspace_doubles.restype = ctypes.c_size_t
    lib.c3_workspace_doubles.argtypes = [ctypes.c_int] * 3
    return dict(zip(C3_SYMBOLS, _build.register(so, C3_SYMBOLS))), lib
