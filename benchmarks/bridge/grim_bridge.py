"""GRiM's kernels through a C bridge: the same FFI implementations the JAX rows call, launched from C
with no Python or JAX in the timed path.

1. **Record.** Importing this module wraps ``jax.ffi.ffi_call`` and GRiM's ``_build.compile_kernel`` /
   ``_build.register``. Run the JAX method once, eagerly, inside ``record()``: every FFI call is
   kept with its target, operands, attributes and result shapes, and every target with the
   kernel source, generated headers and nvcc flags GRiM built it from.
2. **Compile.** For each GRiM ``.so`` a bridge library that ``#include``s the same kernel file,
   unchanged, against the same headers and flags, and exports each handler's implementation
   through ``grim_invoke.cuh`` (GRiM's own launch code; only the XLA wrapper is skipped).
3. **Replay.** ``Program`` uploads the recorded inputs once, wires an operand that equals an
   earlier call's result to that result's buffer (e.g. HJCD coarse -> LM), appends any ported
   JAX step (``runtime.cu``), and times the sequence with ``common.amortized_ms``'s loop in C.
   ``check`` compares the bridge's output with the JAX output from the recording.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE / ".build"

import jax  # noqa: E402
from grim.motion import _build  # noqa: E402

SOURCES: dict[str, tuple] = {}   # .so path -> (kernel, generated headers, flags)
TARGETS: dict[str, tuple] = {}   # FFI target name -> (.so path, handler symbol)
_calls: list | None = None

_compile, _register, _ffi_call = _build.compile_kernel, _build.register, jax.ffi.ffi_call


def _spy_compile(kernel, generated, flags=()):
    so = _compile(kernel, generated, flags)
    SOURCES[str(so)] = (kernel, dict(generated), tuple(flags))
    return so


def _spy_register(so, symbols):
    names = _register(so, symbols)
    for name, symbol in zip(names, symbols):
        TARGETS[name] = (str(so), symbol)
    return names


def _spy_ffi_call(name, result_shape, *a, **kw):
    call = _ffi_call(name, result_shape, *a, **kw)

    def run(*operands, **attrs):
        out = call(*operands, **attrs)
        if _calls is not None:
            shapes = result_shape if isinstance(result_shape, (tuple, list)) else (result_shape,)
            outs = out if isinstance(out, (tuple, list)) else (out,)
            _calls.append({"name": name, "operands": [np.asarray(x) for x in operands], "attrs": attrs,
                           "shapes": [(tuple(s.shape), np.dtype(s.dtype)) for s in shapes],
                           "outputs": [np.asarray(x) for x in outs]})
        return out
    return run


_build.compile_kernel, _build.register, jax.ffi.ffi_call = _spy_compile, _spy_register, _spy_ffi_call


@contextmanager
def record():
    """Collect the FFI calls made inside (run the JAX code eagerly, not under ``jax.jit``)."""
    global _calls
    _calls = calls = []
    try:
        yield calls
    finally:
        _calls = None


# ---------------------------------------------------------------- compile

def _handler(source: str, symbol: str) -> tuple[str, list[str]]:
    """Implementation name and attribute names, in Bind order, of FFI handler ``symbol``."""
    m = re.search(rf"XLA_FFI_DEFINE_HANDLER_SYMBOL\(\s*{symbol}\s*,\s*(\w+)\s*,(.*?)\);", source, re.S)
    if not m:
        raise ValueError(f"no XLA_FFI_DEFINE_HANDLER_SYMBOL for {symbol}")
    return m[1], re.findall(r'\.Attr<[^>]*>\("(\w+)"\)', m[2])


def _nvcc(flags, includes, src: str, tag: str) -> Path:
    import jaxlib
    xla = Path(jaxlib.__file__).parent / "include"
    cmd = ["nvcc", "-O3", "-std=c++17", _build.gpu_arch(), "--shared", "--compiler-options", "-fPIC",
           "-cudart", "shared", "-w", *flags]
    key = hashlib.sha1("\x00".join([src, " ".join(cmd), *map(str, includes), str(xla),
                                    (HERE / "grim_invoke.cuh").read_text()]).encode()).hexdigest()[:16]
    lib = BUILD / f"{tag}-{key}.so"
    if not lib.exists():
        BUILD.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            cu = Path(d) / f"{tag}_bridge.cu"   # not the kernel's own name: it #includes that
            cu.write_text(src)
            subprocess.check_call([*cmd, *(f"-I{i}" for i in includes), f"-I{HERE}", f"-I{xla}",
                                   "-o", str(lib), str(cu)])
    return lib


_LIBS: dict[str, tuple] = {}


def _library(so: str):
    """The bridge library for GRiM's ``so``: (ctypes lib, {symbol: (impl, attr names)})."""
    if so not in _LIBS:
        kernel, generated, flags = SOURCES[so]
        source = (_build.KERNELS_DIR / f"{kernel}.cu").read_text()
        symbols = sorted({s for t, (o, s) in TARGETS.items() if o == so})
        handlers = {s: _handler(source, s) for s in symbols}
        src = (f'#include "{kernel}.cu"\n#include "grim_invoke.cuh"\n'
               + "".join(f'extern "C" int gb_{s}(cudaStream_t st, XLA_FFI_Buffer* b, const double* a) '
                         f"{{ return gb::invoke({impl}, st, b, a); }}\n" for s, (impl, _) in handlers.items()))
        with tempfile.TemporaryDirectory() as gen:
            for name, text in generated.items():
                (Path(gen) / name).write_text(text)
            lib = _nvcc(flags, [gen, _build.KERNELS_DIR, _build.GLASS_DIR], src, Path(kernel).name)
        _LIBS[so] = (ctypes.CDLL(str(lib)), handlers)
    return _LIBS[so]


def _runtime():
    if "runtime" not in _LIBS:
        src = (HERE / "runtime.cu").read_text()
        L = ctypes.CDLL(str(_nvcc([], [], src, "runtime")))
        L.gb_malloc.restype = ctypes.c_void_p
        L.gb_malloc.argtypes = [ctypes.c_size_t]
        L.gb_free.argtypes = [ctypes.c_void_p]
        L.gb_h2d.argtypes = L.gb_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        L.gb_run.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.gb_time.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_double)]
        _LIBS["runtime"] = L
    return _LIBS["runtime"]


# ---------------------------------------------------------------- replay

class XlaBuffer(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t), ("extension_start", ctypes.c_void_p),
                ("dtype", ctypes.c_int), ("data", ctypes.c_void_p), ("rank", ctypes.c_int64),
                ("dims", ctypes.POINTER(ctypes.c_int64))]


class StepC(ctypes.Structure):
    _fields_ = [("fn", ctypes.c_void_p), ("bufs", ctypes.POINTER(XlaBuffer)),
                ("attrs", ctypes.POINTER(ctypes.c_double))]


XLA_DTYPE = {np.dtype(np.float32): 11, np.dtype(np.int32): 4, np.dtype(np.bool_): 1}


class Array:
    """A device buffer of the bridge."""

    def __init__(self, shape, dtype, host=None):
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        self.ptr = _runtime().gb_malloc(self.nbytes)
        if not self.ptr:
            raise MemoryError(f"cudaMalloc of {self.nbytes} bytes failed")
        if host is not None:
            h = np.ascontiguousarray(host, self.dtype)
            _runtime().gb_h2d(self.ptr, h.ctypes.data, self.nbytes)

    def get(self) -> np.ndarray:
        h = np.empty(self.shape, self.dtype)
        if _runtime().gb_d2h(h.ctypes.data, self.ptr, self.nbytes):
            raise RuntimeError("device to host copy failed")
        return h

    def free(self):
        if self.ptr:
            _runtime().gb_free(self.ptr)
            self.ptr = None


class Program:
    """The recorded ``calls`` as a C program on device buffers; ``extra`` appends ported JAX steps:
    (runtime function name, [buffers: a host array (uploaded), ("call", i, j) for result j of call i, or ("out", k)
    for the k-th extra step's result], attrs, result (shape, dtype))."""

    def __init__(self, calls: list[dict], extra=()):
        rt = _runtime()
        self.arrays, self._keep, steps = [], [], []
        results = []   # per call: its result Arrays
        for i, c in enumerate(calls):
            so, symbol = TARGETS[c["name"]]
            lib, handlers = _library(so)
            _, attr_names = handlers[symbol]
            ins = [self._operand(x, calls[:i], results) for x in c["operands"]]
            outs = [self._new(s, d) for s, d in c["shapes"]]
            results.append(outs)
            fn = ctypes.cast(getattr(lib, f"gb_{symbol}"), ctypes.c_void_p).value
            # Buffers (Args then Rets) and attributes are separate streams, each in Bind order.
            steps.append(self._step(fn, ins + outs, [float(c["attrs"][n]) for n in attr_names]))
        self.extra_out = []
        for name, bufs, attrs, (shape, dtype) in extra:
            ins = [self._new(b.shape, b.dtype, b) if isinstance(b, np.ndarray)
                   else results[b[1]][b[2]] if b[0] == "call" else self.extra_out[b[1]] for b in bufs]
            out = self._new(shape, dtype)
            self.extra_out.append(out)
            fn = ctypes.cast(getattr(rt, name), ctypes.c_void_p).value
            steps.append(self._step(fn, ins + [out], list(attrs)))
        self.results = results
        self.steps = (StepC * len(steps))(*steps)

    def _new(self, shape, dtype, host=None):
        a = Array(shape, dtype, host)
        self.arrays.append(a)
        return a

    def _operand(self, x, earlier, results):
        for i, c in enumerate(earlier):
            for j, y in enumerate(c["outputs"]):
                if y.shape == x.shape and y.dtype == x.dtype and np.array_equal(y, x):
                    return results[i][j]
        return self._new(x.shape, x.dtype, x)

    def _step(self, fn, arrays, attrs):
        bufs = (XlaBuffer * max(len(arrays), 1))()
        for b, a in zip(bufs, arrays):
            dims = (ctypes.c_int64 * max(len(a.shape), 1))(*a.shape)
            self._keep.append(dims)
            b.struct_size, b.dtype, b.data, b.rank = ctypes.sizeof(XlaBuffer), XLA_DTYPE[a.dtype], a.ptr, len(a.shape)
            b.dims = ctypes.cast(dims, ctypes.POINTER(ctypes.c_int64))
        at = (ctypes.c_double * max(len(attrs), 1))(*attrs)
        self._keep += [bufs, at]
        return StepC(fn, bufs, at)

    def run(self):
        if rc := _runtime().gb_run(self.steps, len(self.steps)):
            raise RuntimeError(f"bridge program failed (rc {rc})")

    def time_ms(self, k: int, reps: int, warmup: int) -> float:
        ms = (ctypes.c_double * reps)()
        if rc := _runtime().gb_time(self.steps, len(self.steps), k, reps, warmup, ms):
            raise RuntimeError(f"bridge timing failed (rc {rc})")
        return float(np.median(list(ms)))

    def free(self):
        for a in self.arrays:
            a.free()


def check(out: np.ndarray, ref: np.ndarray, what: str, rtol=1e-5, atol=1e-5):
    """The bridge must reproduce the JAX row's output: same kernels, same inputs."""
    out, ref = np.asarray(out), np.asarray(ref)
    if out.shape != ref.shape or not np.allclose(out, ref, rtol=rtol, atol=atol):
        diff = np.abs(out.astype(np.float64) - ref.astype(np.float64)).max() if out.shape == ref.shape else None
        raise AssertionError(f"{what}: C bridge output differs from the JAX FFI output "
                             f"(shapes {out.shape} vs {ref.shape}, max diff {diff})")
