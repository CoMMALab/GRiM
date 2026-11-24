"""Run GRiD's release benchmark worker (``test/benchmarks/release/worker.py`` in the GRiD clone)
unchanged on our robots and inputs.

GRiD's worker owns the whole per-cell protocol: independent float64 oracle check, warmups plus a
sustained warm period, timed samples, post-timing and repeatability checks. Its adapters time
GRiD (C++ bridge and JAX), Pinocchio (CppADCodeGen bridge, persistent thread pool) and frax. The
worker builds its robot from ``fixtures.Fixture``, which only knows GRiD's manifest robots
(iiwa14, go2, g1); this module swaps in a ``Fixture`` with the same attributes, built by the same
RBDReference calls, for our URDF and our shared inputs (``results/inputs/dynamics_<robot>.npz``,
permuted to the project's joint order). Forward dynamics takes our ``tau`` as its torque.

    python dynamics/grid_protocol.py --robot panda --backend pinocchio --operation inverse_dynamics \\
        --batches 1 16 1024 --iterations 30 --warmups 5 --cpu-threads 16 --output out.json

Run in the ``grid-bench`` env (GRiD installed from external/grid; see RUNNING.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

GRID = HERE.parent / "external" / "grid"
sys.path[:0] = [str(GRID), str(GRID / "external")]

from test.benchmarks.release import fixtures, protocol, worker  # noqa: E402
from test.benchmarks.release.grid_adapter import stats  # noqa: E402

EE_JOINT = {"panda": "panda_joint7", "baxter": "right_w2", "g1": "right_wrist_yaw_joint"}


class Fixture:
    """``fixtures.Fixture`` for one of our robots: same attributes, same RBDReference builders."""

    def __init__(self, robot, count):
        from RBDReference.equivalents import build_adapter
        from RBDReference.equivalents.reference_backend import build_project_adapter
        from RBDReference.tests.model_sources import ResolvedRobotModel, RobotSpec
        urdf = str(common.MODEL_URDF[robot])
        self.base = "fixed"
        self.spec = RobotSpec(robot_id=robot, tier="mantis", embodiment="arm", source_kind="vendored",
                              description_name=robot, base_modes=["fixed"], preferred_variant="",
                              notes="mantis benchmark URDF", source_candidates=[])
        self.resolved = ResolvedRobotModel(robot_id=robot, source_kind="vendored", description_name=robot,
                                           urdf_path=urdf, package_path=None, repository_path=None,
                                           repository_url=None, revision=None, notes="")
        self.project = build_project_adapter(self.spec, self.resolved, base_mode=self.base)
        self.oracle = build_adapter(self.spec, self.resolved, base_mode=self.base, backend="pinocchio")
        self.nq, self.nv = self.project.nq, self.project.nv
        self.urdf = urdf
        self.target = EE_JOINT[robot]
        data = common.load_inputs("dynamics", robot)
        names = list(data["joint_names"])
        perm = [names.index(n) for n in self.project.joint_names]
        if len(data["q"]) < count:
            raise ValueError(f"{count} samples requested; the shared inputs hold {len(data['q'])}")
        take = lambda k: np.ascontiguousarray(data[k][:count][:, perm], dtype=np.float32)
        self.q, self.v, self.a, self.u = take("q"), take("qd"), take("qdd"), take("tau")
        self.metadata = {"urdf": urdf, "urdf_sha256": protocol.digest(urdf), "nq": self.nq, "nv": self.nv,
                         "target": self.target, "base": self.base,
                         "sampling": "mantis benchmarks/results/inputs (common.configs, seed 3); fd torque = tau",
                         "gravity": -9.81, "dtype": "float32", "oracle_dtype": "float64",
                         "joint_names": self.project.joint_names,
                         "convention": "project order, Pinocchio tangent convention"}

    args = fixtures.Fixture.args
    expected = fixtures.Fixture.expected


MANTIS_OPS = {"inverse_dynamics": "id", "crba": "crba", "forward_dynamics": "fd",
               "inverse_dynamics_gradient": "id_du"}


class MantisAdapter:
    """mantis's traced dynamics (GRiM's ``MantisDynamics``, float) behind GRiD's adapter
    interface. Each batch runs the tier (thread / warp / block) with the best run mean, chosen
    the way GRiD's Pinocchio adapter chooses its thread count: every variant timed with the
    worker's own ``timed`` and recorded. fd is ABA."""

    def __init__(self, backend, operation, fixture, directory, warmups=5, iterations=30):
        import jax
        from grim.motion import MotionRobot
        from grim.motion.dynamics import MantisDynamics
        jax.config.update("jax_default_matmul_precision", "highest")
        self.op, self.f, self.ck = operation, fixture, MANTIS_OPS[operation]
        robot = MotionRobot.from_urdf(fixture.urdf)
        self.dyn = MantisDynamics(robot)  # GRiM's default op set: ck_dynamics.cu needs them all
        names = list(fixture.project.joint_names)
        self.perm = np.array([names.index(n) for n in robot.actuated_names])  # mantis k <- project perm[k]
        self.inv = np.argsort(self.perm)
        self.tune = (warmups, iterations)
        self.metadata = {"backend": "mantis", "dtype": "float32", "method": f"mantis trace ({self.ck})",
                         "source_urdf_sha256": protocol.digest(fixture.urdf),
                         "tier_policy": "best run mean over thread / warp / block (protocol.timed); every variant recorded"}

    def prepare(self, batch):
        import jax
        args = [a[:, self.perm] for a in self.f.args(self.op, batch)]
        if self.ck == "crba":
            args = args[:1]
        variants, best = {}, None
        for tier in ("thread", "warp", "block"):
            fn = jax.jit(lambda *x, tier=tier: self.dyn(self.ck, *x, tier=tier))
            dev = tuple(jax.device_put(a) for a in args)
            jax.block_until_ready(dev)
            try:
                t = protocol.timed(lambda: fn(*dev), jax.block_until_ready, *self.tune)
            except jax.errors.JaxRuntimeError as e:  # GRiM refuses a tier the robot doesn't fit
                variants[tier] = {"unavailable": str(e).split("[")[0][:200]}
                continue
            variants[tier] = {k: t[k] for k in ("mean_us", "median_us", "min_us", "max_us")}
            if best is None or t["mean_us"] < best[0]:
                best = (t["mean_us"], tier, fn, dev)
        _, tier, fn, dev = best
        self.metadata.update(tier=tier, tier_variants=variants)
        self.resident = lambda: fn(*dev)
        self.host = lambda: jax.device_get(fn(*(jax.device_put(a.copy()) for a in args)))
        self.sync = jax.block_until_ready

    def normalize(self, out):
        a = np.asarray(out)
        inv, n = self.inv, self.f.nv
        if self.ck == "id_du":
            return a[:, inv][:, :, np.r_[inv, inv + n]]
        if self.ck == "crba":
            return a[:, inv][:, :, inv]
        return a[:, inv]


CK_OP_INDEX = {"id": 0, "crba": 1, "fd": 2, "id_du": 4}  # ck_dynamics.cu's Op enum
CK_TIERS = ("thread", "warp", "block")
# Backend -> MantisDynamics' id_du: the analytical RNEA derivatives, or mantis's AD of the RNEA trace.
ID_DU_MODES = {"mantis_cuda": "analytic", "mantis_cuda_fwd_ad": "forward", "mantis_cuda_rev_ad": "reverse"}


def _mantis_bridge(robot, id_du="analytic"):
    """GRiM's ``MantisDynamics`` for ``robot`` plus ``mantis_bridge.cu`` built against the same
    generated header and nvcc flags (captured from GRiM's ``compile_kernel``)."""
    import ctypes
    import hashlib
    import subprocess
    import tempfile
    from grim.motion import _build
    from grim.motion.dynamics import MantisDynamics
    seen, compile_kernel = {}, _build.compile_kernel

    def spy(kernel, generated, flags=()):
        seen.update(generated=generated, flags=tuple(flags))
        return compile_kernel(kernel, generated, flags)
    _build.compile_kernel = spy
    try:
        dyn = MantisDynamics(robot, id_du=id_du)
    finally:
        _build.compile_kernel = compile_kernel
    import jaxlib
    src = HERE / "mantis_bridge.cu"
    flags = ["-O3", "-std=c++17", _build.gpu_arch(), "--shared", "--compiler-options", "-fPIC", *seen["flags"]]
    key = hashlib.sha1("\x00".join([src.read_text(), (_build.KERNELS_DIR / "dynamics/ck_dynamics.cu").read_text(),
                                     *seen["generated"].values(), " ".join(flags)]).encode()).hexdigest()[:16]
    lib = HERE / ".build" / f"mantis_bridge-{key}.so"
    if not lib.exists():
        lib.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as gen:
            for name, text in seen["generated"].items():
                (Path(gen) / name).write_text(text)
            subprocess.check_call(["nvcc", *flags, f"-I{gen}", f"-I{_build.KERNELS_DIR}", f"-I{_build.GLASS_DIR}",
                                   f"-I{Path(jaxlib.__file__).parent / 'include'}", "-o", str(lib), str(src)])
    L = ctypes.CDLL(str(lib))
    fp, dp, ip = (ctypes.POINTER(t) for t in (ctypes.c_float, ctypes.c_double, ctypes.c_int))
    L.ck_create.argtypes = [ctypes.c_int, ctypes.c_int, fp, ctypes.c_int, ip]
    L.ck_create.restype = ctypes.c_void_p
    L.ck_close.argtypes = [ctypes.c_void_p]
    L.ck_host.argtypes = L.ck_resident.argtypes = [ctypes.c_void_p, fp]
    L.ck_time.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_int, dp, dp]
    return dyn, L, lib


class MantisCudaAdapter:
    """mantis's traced dynamics (GRiM's ``ck_dynamics.cu`` kernel, unchanged) through a C bridge
    (``mantis_bridge.cu``) timed like GRiD's own kernel bridge: host-clock launch + sync loops in
    C, no Python or JAX in the timed path. Same tier policy as ``MantisAdapter``, chosen on the
    bridge's resident mean; every variant recorded."""

    def __init__(self, backend, operation, fixture, directory, warmups=5, iterations=30):
        from grim.motion import MotionRobot
        self.op, self.f, self.ck = operation, fixture, MANTIS_OPS[operation]
        robot = MotionRobot.from_urdf(str(fixture.urdf))
        self.dyn, self.lib, lib = _mantis_bridge(robot, ID_DU_MODES[backend])
        names = list(fixture.project.joint_names)
        self.perm = np.array([names.index(self.dyn.actuated[i]) for i in self.dyn._to_gen])  # kernel order
        self.inv = np.argsort(self.perm)
        self.tune, self.ctx = (warmups, iterations), None
        self.metadata = {"backend": backend, "dtype": "float32", "method": f"mantis trace ({self.ck}, id_du {ID_DU_MODES[backend]}), C bridge",
                         "source_urdf_sha256": protocol.digest(fixture.urdf), "library_sha256": protocol.digest(lib),
                         "timing": "host clock around launch + cudaDeviceSynchronize in C (GRiD kernel_bridge's loop)",
                         "tier_policy": "best resident mean over thread / warp / block; every variant recorded"}

    def _time(self, ctx, warm_seconds=0.0):
        import ctypes
        dp = ctypes.POINTER(ctypes.c_double)
        w, c = np.empty(self.tune[1]), np.empty(self.tune[1])
        rc = self.lib.ck_time(ctx, warm_seconds, *self.tune, w.ctypes.data_as(dp), c.ctypes.data_as(dp))
        if rc:
            raise RuntimeError(f"mantis bridge timing failed (rc {rc})")
        return stats(w), stats(c)

    def prepare(self, batch):
        import ctypes
        self.close()
        args = self.f.args(self.op, batch)[:1 if self.ck == "crba" else 3]
        packed = np.ascontiguousarray(np.concatenate([a[:, self.perm] for a in args], axis=1), np.float32)
        variants, best = {}, None
        for tier, name in enumerate(CK_TIERS):
            rc = ctypes.c_int(0)
            ctx = self.lib.ck_create(CK_OP_INDEX[self.ck], tier, packed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     batch, ctypes.byref(rc))
            if not ctx:
                if rc.value != -10:
                    raise RuntimeError(f"mantis bridge failed (rc {rc.value})")
                variants[name] = {"unavailable": "GRiM refuses this tier: one configuration exceeds shared memory"}
                continue
            _, c = self._time(ctx)
            variants[name] = {k: c[k] for k in ("mean_us", "median_us", "min_us", "max_us")}
            if best is None or c["mean_us"] < best[0]:
                if best:
                    self.lib.ck_close(best[2])
                best = (c["mean_us"], name, ctx)
            else:
                self.lib.ck_close(ctx)
        _, tier, self.ctx = best
        self.batch, self.n_out = batch, {"id": 1, "fd": 1, "crba": self.f.nv, "id_du": 2 * self.f.nv}[self.ck] * self.f.nv
        self.metadata.update(tier=tier, tier_variants=variants)
        self.resident = lambda: self._call(self.lib.ck_resident)
        self.host = lambda: self._call(self.lib.ck_host)
        self.sync = lambda out: None

    def _call(self, fn):
        import ctypes
        out = np.empty((self.n_out, self.batch), np.float32)
        if rc := fn(self.ctx, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))):
            raise RuntimeError(f"mantis bridge call failed (rc {rc})")
        return out

    def time_host(self, warmups, iterations, warm_seconds=0.0):
        self.tune = (warmups, iterations)
        self._timed = self._time(self.ctx, warm_seconds)
        return self._timed[0], {}

    def extra_timings(self, warmups, iterations, warm_seconds=0.0):
        # The worker times ``resident`` through Python first; this replaces it with the C loop.
        return {"resident": self._timed[1]}

    def normalize(self, out):
        a, inv, n = np.asarray(out).T, self.inv, self.f.nv
        if self.ck in ("id", "fd"):
            return a[:, inv]
        if self.ck == "crba":
            return a.reshape(-1, n, n).swapaxes(1, 2)[:, inv][:, :, inv]
        D = a.reshape(-1, 2, n, n).swapaxes(2, 3)[:, :, inv][:, :, :, inv]
        return np.concatenate([D[:, 0], D[:, 1]], axis=-1)

    def close(self):
        if self.ctx:
            self.lib.ck_close(self.ctx)
            self.ctx = None


RBDL_OPS = {"inverse_dynamics": 0, "forward_dynamics": 4, "crba": 8}  # GRiD's operation indices
RBDL_PREFIX = Path(__import__("os").environ.get("RBDL_PREFIX", Path.home() / "miniconda3/envs/dyn-bench"))


class RbdlAdapter:
    """RBDL v3.4.0 (``external/rbdl``, built into ``RBDL_PREFIX``) through ``rbdl_bridge.cpp``:
    GRiD's ``PinAdapter`` protocol (persistent pool, thread-count candidates, best run mean,
    every variant recorded), float64. No inverse dynamics derivatives in RBDL."""

    def __init__(self, backend, operation, fixture, directory, cpu_threads=1):
        import ctypes
        import subprocess
        import xml.etree.ElementTree as ET
        if operation not in RBDL_OPS:
            raise NotImplementedError(f"RBDL has no {operation}")
        self.op, self.f, self.ceiling, self.active = operation, fixture, cpu_threads, 1
        release = GRID / "test/benchmarks/release"
        src = HERE / "rbdl_bridge.cpp"
        build = HERE / ".build"
        build.mkdir(exist_ok=True)
        lib = build / f"rbdl_bridge-{protocol.digest(src)[:12]}-{protocol.digest(release / 'release_pool.h')[:12]}.so"
        if not lib.exists():
            subprocess.check_call(["g++", "-std=c++17", "-O2", "-DNDEBUG", "-fPIC", "-shared", "-pthread",
                                   str(src), "-o", str(lib), f"-I{release}", f"-I{RBDL_PREFIX}/include",
                                   f"-I{RBDL_PREFIX}/include/eigen3", f"-L{RBDL_PREFIX}/lib",
                                   f"-Wl,-rpath,{RBDL_PREFIX}/lib", "-lrbdl", "-lrbdl_urdfreader"])
        self.lib = L = ctypes.CDLL(str(lib))
        L.rbdl_release_error.restype = ctypes.c_char_p
        L.rbdl_release_pool_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.rbdl_release_pool_create.restype = ctypes.c_void_p
        L.rbdl_release_pool_close.argtypes = [ctypes.c_void_p]
        L.rbdl_release_dof.argtypes = [ctypes.c_void_p]
        L.rbdl_release_q_link.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.rbdl_release_q_link.restype = ctypes.c_char_p
        fp = ctypes.POINTER(ctypes.c_float)
        L.rbdl_release_pool_eval.argtypes = [ctypes.c_void_p, fp, fp, fp, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        self.pool = L.rbdl_release_pool_create(fixture.urdf.encode(), RBDL_OPS[operation], cpu_threads)
        if not self.pool:
            raise RuntimeError(L.rbdl_release_error().decode())
        n = L.rbdl_release_dof(self.pool)
        if n != fixture.nv:
            raise ValueError(f"RBDL has {n} DOF, the fixture {fixture.nv}")
        joint_of = {j.find("child").get("link"): j.get("name")
                    for j in ET.parse(fixture.urdf).getroot().findall("joint")}
        names = list(fixture.project.joint_names)
        self.perm = np.array([names.index(joint_of[L.rbdl_release_q_link(self.pool, i).decode()]) for i in range(n)])
        self.inv = np.argsort(self.perm)
        self.metadata = {"backend": "rbdl", "cpu_thread_ceiling": cpu_threads,
                         "hardware_threads": __import__("os").cpu_count(),
                         "thread_policy": "GRiD PinAdapter's: best run mean over powers of two plus {B//16, ceiling}, "
                                          "capped by batch and ceiling; persistent C++ pool (GRiD release_pool.h), "
                                          "slice 0 on the caller; independent models; every variant recorded",
                         "method": {0: "InverseDynamics (RNEA)", 4: "ForwardDynamics (ABA)",
                                    8: "CompositeRigidBodyAlgorithm"}[RBDL_OPS[operation]],
                         "dtype": "float64", "input_storage_dtype": "float32",
                         "input_mapping": "fp32 samples widened to fp64 per sample inside the call",
                         "output_storage_dtype": "float64", "library_sha256": protocol.digest(lib),
                         "rbdl_prefix": str(RBDL_PREFIX), "allocation_policy": "persistent model/scratch; materialized host output per call"}

    candidates = __import__("test.benchmarks.release.pin_adapter", fromlist=["PinAdapter"]).PinAdapter.candidates
    time_host = __import__("test.benchmarks.release.pin_adapter", fromlist=["PinAdapter"]).PinAdapter.time_host

    def prepare(self, batch):
        import ctypes
        f, n = self.f, self.f.nv
        args = f.args(self.op, batch)
        args = args * 3 if len(args) == 1 else args  # crba reads q only; the bridge ignores v, t
        q, v, t = (np.ascontiguousarray(a[:, self.perm], np.float32) for a in args)
        size = n * n if self.op == "crba" else n
        fp = ctypes.POINTER(ctypes.c_float)
        ptrs = tuple(a.ctypes.data_as(fp) for a in (q, v, t))
        self._keep = (q, v, t)

        def call():
            out = np.empty((batch, size), np.float64)
            if self.lib.rbdl_release_pool_eval(self.pool, *ptrs, batch,
                                               out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.active):
                raise RuntimeError(self.lib.rbdl_release_error().decode())
            return out.reshape(batch, n, n) if self.op == "crba" else out
        self.active = 1
        self.metadata["active_cpu_threads"] = 1
        self.host, self.resident = call, None
        self.sync, self.download = (lambda x: None), (lambda x: x)
        self.batch = batch

    def normalize(self, out):
        inv = self.inv
        return out[:, inv][:, :, inv] if self.op == "crba" else out[:, inv]

    def close(self):
        if self.pool:
            self.lib.rbdl_release_pool_close(self.pool)
            self.pool = None


def _route_mantis():
    from test.benchmarks.release import competitors
    make = competitors.make_adapter

    def make_adapter(backend, operation, fixture, directory, cpu_threads=1):
        if backend == "mantis":
            return MantisAdapter(backend, operation, fixture, directory)
        if backend in ID_DU_MODES:
            return MantisCudaAdapter(backend, operation, fixture, directory)
        if backend == "rbdl":
            return RbdlAdapter(backend, operation, fixture, directory, cpu_threads=cpu_threads)
        return make(backend, operation, fixture, directory, cpu_threads=cpu_threads)
    competitors.make_adapter = make_adapter


def _frax_v020_compat():
    """GRiD's frax adapter predates frax 6787b79 (2026-10-03), which renamed ``Robot``'s
    ``add_floating_base: bool`` to ``floating_base: str | None``. Our robots are fixed base, where
    ``add_floating_base=False`` is ``floating_base=None``; anything else is refused."""
    try:
        import frax.core.robot as fr
    except ImportError:
        return
    base = fr.Robot

    class Robot(base):
        def __init__(self, urdf_filename, *args, add_floating_base=False, **kw):
            if add_floating_base:
                raise ValueError("mantis's dynamics benchmark is fixed base only")
            super().__init__(urdf_filename, *args, **kw)
    fr.Robot = Robot


def main():
    fixtures.Fixture = Fixture
    protocol.ROBOTS.update({r: "fixed" for r in common.ROBOTS})
    _frax_v020_compat()
    _route_mantis()
    worker.main()


if __name__ == "__main__":
    main()
