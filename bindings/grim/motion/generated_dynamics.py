"""Any-batch JAX FFI handlers for a robot's GRiM-generated dynamics (``kernels/dynamics/
generated_dynamics.cu``).

The caller supplies the generated ``grim.cuh`` (``grim_codegen.GRiMCodeGenerator`` output, so
the joint order, signs and floating-base layout are whatever its parsed model says) and gets
one FFI target per op. Unlike :mod:`grim.jax`'s handles, these take any batch size: the
spill workspace comes from XLA's scratch allocator per call, and nothing is preallocated.

Every target takes float32 ``(B, NUM_POS)`` buffers (velocity-like inputs zero-padded past
``NUM_VEL``) and a ``gravity`` attribute (except ``minv``); outputs per op:

==========  =================================  ==========================================
op          inputs                             output
==========  =================================  ==========================================
id          q, qd, qdd                         (B, NUM_POS), the first NUM_VEL live
fd          q, qd, u                           (B, NUM_POS), the first NUM_VEL live
minv        q                                  (B, n, n) [col, row], upper triangle
crba        q                                  (B, n, n) [col, row]
id_grad     q, qd, qdd                         (B, 2n, n) [col, row]: [dtau/dq | dtau/dqd]
fd_grad     q, qd, u                           (B, 2n, n) [col, row]: [dqdd/dq | dqdd/dqd]
idsva_so    q, qd, qdd                         (B, 4 n^3) second-order ID tensors
==========  =================================  ==========================================

with ``n = NUM_VEL``.
"""

from __future__ import annotations

from pathlib import Path

from . import _build

SYMBOLS = {"id": "DynIdFfi", "fd": "DynFdFfi", "minv": "DynMinvFfi", "crba": "DynCrbaFfi",
           "id_grad": "DynIdGradFfi", "fd_grad": "DynFdGradFfi", "idsva_so": "DynIdsvaSoFfi"}


def build(header: str, *, floating_base: bool = False,
          runtime_inertia: bool = False) -> tuple[dict[str, str], Path]:
    """Compile (or load) the handlers around ``header`` -> ``({op: FFI target}, library)``.

    ``floating_base`` must match the header's model. With ``runtime_inertia`` (a header
    generated with ``runtime_inertia=True``) the library also exports
    ``DynInertiaParamsSize()`` and ``DynSetInertiaParams(const float*)``.
    """
    flags = (*(("-DGRIM_GEN_DYN_FLOATING_BASE",) if floating_base else ()),
             *(("-DGRIM_GEN_DYN_RUNTIME_INERTIA",) if runtime_inertia else ()))
    so = _build.compile_kernel("dynamics/generated_dynamics", {"grim.cuh": header}, flags)
    return dict(zip(SYMBOLS, _build.register(so, tuple(SYMBOLS.values())))), so
