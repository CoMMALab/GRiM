"""Contact (C3+) numerics compiled per problem shape, for contact-aware trajectory optimization.

The generic least-squares solvers that used to live here moved to the standalone ``cudals``
package.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

from . import _build

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
