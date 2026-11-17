"""The IK kernels (kernels/ik_mantis, ``backend="mantis-f32"`` / ``"mantis-f64"``) against
the float64 oracle, where test_ik_solvers.py, test_ls_ik.py and test_dls_ik.py do not already
cover them:

* one LS / SQP step WITH collision against a float64 transcription of the Gauss-Newton collision
  terms (analytic gradients: closed-form normals times GRiD's point Jacobians), at the kernel's float32
  inputs, each seed held to the tolerance plus twice the reference's own sensitivity to those
  inputs (see ``_reference``);
* canonical IK's KKT certificate and analytic IK's FK certificate.

Every build is a test subject: these run per robot and per end-effector set.
"""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import collision as C
from grim.motion.reference import kinematics as K

from .conftest import assert_close_scaled, ee_sets, load_robot, requires_gpu
from .ik_cases import collision_case, reachable_targets, seeds

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence]

BACKENDS = ["mantis-f32", "mantis-f64"]
SETTINGS = dict(pos_weight=50.0, ori_weight=10.0, lambda_init=5e-3, eps_pos=1e-4, eps_ori=1e-3)
COLL = dict(collision_weight=1e4, collision_margin=0.02)


def _distances(robot, q, coll, world):
    """Every collision constraint's distance, in the kernels' order: self pairs (min over their
    sphere pairs), then each robot sphere against each obstacle kind."""
    T = K.frame_poses(robot, q)
    sc = coll.self_collision
    cs = [K.quat_rotate(T[sc.link_joint[l]][:4], sc.sph[a, :3]) + T[sc.link_joint[l]][4:]
          for l in range(len(sc.link_start) - 1) for a in range(sc.link_start[l], sc.link_start[l + 1])]
    out = [min(np.linalg.norm(cs[a] - cs[b]) - sc.sph[a, 3] - sc.sph[b, 3]
               for a in range(sc.link_start[li], sc.link_start[li + 1])
               for b in range(sc.link_start[lj], sc.link_start[lj + 1]))
           for li, lj in zip(sc.pair_i, sc.pair_j)]
    prims = [(C.sphere_sphere, world.spheres), (C.sphere_capsule, world.capsules),
             (C.sphere_box, world.boxes), (C.sphere_halfspace, world.halfspaces),
             (C.sphere_cylinder, world.cylinders)]
    for sph, j in zip(coll.robot_spheres, coll.robot_sphere_joint):
        c = K.quat_rotate(T[j][:4], sph[:3]) + T[j][4:]
        out += [f(c, sph[3], o) for f, arr in prims if arr is not None for o in arr]
    return np.array(out)


_KINDS = ("spheres", "capsules", "boxes", "halfspaces", "cylinders")


def _reference(robot, seed, target, ees, coll, world, rng):
    """The float64 step at the kernel's inputs, and its sensitivity to them -> (q1, cond, sens).

    The kernel reads seeds, targets and obstacles as float32 (its FFI buffers), so the reference
    starts from those values. ``sens`` is how far the reference step moves when all of them are
    jittered by float32 unit roundoff (worst of 3 draws): a collision-weighted, trust-capped step
    can amplify a 1e-8 input change ~1e3x on some seeds (baxter, g1), which a fixed tolerance
    cannot hold. Measured, the mantis-f64 kernel stays within 1.6 sens on every seed."""
    from grim.motion.ik import World
    r32 = lambda a: None if a is None else np.asarray(a, np.float32).astype(np.float64)  # noqa: E731
    def jit(a):
        return None if a is None else a * (1 + 2.0 ** -24 * rng.choice([-1, 1], np.shape(a)))
    q0, t0 = r32(seed), r32(target)
    w0 = World(**{k: r32(getattr(world, k)) for k in _KINDS})
    q1, cond = _collision_step(robot, q0, t0, ees, coll, w0)
    sens = max(np.abs(_collision_step(robot, jit(q0), jit(t0), ees, coll,
                                      World(**{k: jit(getattr(w0, k)) for k in _KINDS}))[0] - q1).max()
               for _ in range(3))
    return q1, cond, sens


_GRID: dict = {}


def _grid(robot):
    """GRiD's RBDReference on the robot's URDF (origins as written, as mantis parses them), and
    each actuated joint's column in its q: the analytic Jacobian side of the reference."""
    if id(robot) not in _GRID:
        import contextlib
        import io
        import sys
        import tempfile
        import warnings
        import mantis
        sys.path.insert(0, str(mantis.robot.GRID_DIR / "external"))
        from RBDReference.RBDReference import RBDReference
        from URDFParser import URDFParser
        with tempfile.NamedTemporaryFile("w", suffix=".urdf") as f, contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings(), mantis.robot.exact_origins():
            warnings.simplefilter("ignore")
            f.write(robot.urdf_xml)
            f.flush()
            rbd = RBDReference(URDFParser().parse(f.name))
        cols = [int(np.atleast_1d(rbd.robot.get_joint_index_q(rbd.robot.get_joint_by_name(n).get_id()))[0])
                for n in robot.actuated_names]
        _GRID[id(robot)] = (rbd, np.array(cols))
    return _GRID[id(robot)]


def _point_jacobians(robot, q, frames, points):
    """d point / d q (3 x n_act) of world points fixed in the named frames, from GRiD's spatial
    (WORLD) frame Jacobian: v_point = v_o + w x p."""
    rbd, cols = _grid(robot)
    qg = np.zeros(rbd.robot.get_num_pos())
    qg[cols] = q
    out, cache = [], {}
    for fr, pt in zip(frames, points):
        if fr not in cache:
            J = rbd.frame_jacobian(qg, fr, "WORLD")
            cache[fr] = (J[:3][:, cols], J[3:][:, cols])
        Jv, Jw = cache[fr]
        out.append(Jv + np.cross(Jw.T, pt).T)
    return out


def _collision_gradients(robot, q, coll, world, active):
    """Analytic d distance / d q of the constraints ``active`` (indices into ``_distances``):
    the closed-form unit normal (reference/collision.py's NORMALS; the self pair's argmin sphere
    pair) times the sphere centres' point Jacobians from GRiD. No finite differences."""
    T = K.frame_poses(robot, q)
    sc = coll.self_collision
    names = robot.joint_names
    sl = [(l, a) for l in range(len(sc.link_start) - 1) for a in range(sc.link_start[l], sc.link_start[l + 1])]
    cs = {a: K.quat_rotate(T[sc.link_joint[l]][:4], sc.sph[a, :3]) + T[sc.link_joint[l]][4:] for l, a in sl}
    link_of = {a: l for l, a in sl}
    prims = [(k, o) for k in ("spheres", "capsules", "boxes", "halfspaces", "cylinders")
             if getattr(world, k) is not None for o in getattr(world, k)]
    n_self = len(sc.pair_i)
    grads = {}
    for c in active:
        if c < n_self:
            li, lj = sc.pair_i[c], sc.pair_j[c]
            a, b = min(((a, b) for a in range(sc.link_start[li], sc.link_start[li + 1])
                        for b in range(sc.link_start[lj], sc.link_start[lj + 1])),
                       key=lambda ab: np.linalg.norm(cs[ab[0]] - cs[ab[1]]) - sc.sph[ab[0], 3] - sc.sph[ab[1], 3])
            u = cs[a] - cs[b]
            if np.linalg.norm(u) < 1e-9:
                continue   # coincident centres: skipped, as the kernels
            u /= np.linalg.norm(u)
            Ja, Jb = _point_jacobians(robot, q, [names[sc.link_joint[link_of[a]]], names[sc.link_joint[link_of[b]]]],
                                      [cs[a], cs[b]])
            grads[c] = u @ (Ja - Jb)
        else:
            i, m = divmod(c - n_self, len(prims))
            sph, j = coll.robot_spheres[i], coll.robot_sphere_joint[i]
            ctr = K.quat_rotate(T[j][:4], sph[:3]) + T[j][4:]
            kind, o = prims[m]
            u = C.NORMALS[kind](ctr, sph[3], np.asarray(o, np.float64))
            if u is None:
                continue
            grads[c] = u @ _point_jacobians(robot, q, [names[j]], [ctr])[0]
    return grads


def _collision_step(robot, q, target, ees, coll, world, lam=5e-3, wp=50.0, wo=10.0, wc=1e4, m=0.02):
    """One float64 LS-IK step with the collision Gauss-Newton terms (no SQP rows: with none
    active they coincide) -> (next q, condition number of the step's system)."""
    from grim.motion.reference.ik import stacked_residual_jacobian, trust_radius
    cols = list(range(robot.n_act))
    Wr = np.tile([wp] * 3 + [wo] * 3, len(ees))

    def merit(x):
        r, _ = stacked_residual_jacobian(robot, x, target, ees, cols)
        d = _distances(robot, x, coll, world)
        return float(np.sum((r * Wr) ** 2) + wc * np.sum(np.where(d < m, (d - m) ** 2, 0)))

    r, J = stacked_residual_jacobian(robot, q, target, ees, cols)
    fw, J = r * Wr, J * Wr[:, None]
    A, b = J.T @ J, -J.T @ fw
    d0 = _distances(robot, q, coll, world)
    for c, g in _collision_gradients(robot, q, coll, world, np.flatnonzero(d0 < m)).items():
        A += wc * np.outer(g, g)
        b += wc * (m - d0[c]) * g
    sc = np.sqrt(np.diag(A) + 1e-6 * np.diag(A).max()) + 1e-8   # as the kernels' jacobi_scale
    A, b = A / np.outer(sc, sc) + lam * np.eye(robot.n_act), b / sc
    cond = np.linalg.cond(A)
    delta = np.linalg.lstsq(A, b, rcond=None)[0] / sc
    R = trust_radius(r)
    if np.linalg.norm(delta) > R:
        delta *= R / (np.linalg.norm(delta) + 1e-18)
    trials = [np.clip(q + a * delta, robot.lower, robot.upper) for a in (1, 0.5, 0.25, 0.1, 0.025)]
    errs = [merit(x) for x in trials]
    k = int(np.argmin(errs))
    return (trials[k] if errs[k] < merit(q) * (1 - 1e-4) else q), cond


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,which", [("fr3", 0), ("iiwa14", 1), ("baxter", 1), ("g1", 1)])
def test_collision_step_matches_reference(name, which, backend):
    """baxter and g1 have joints that move collision spheres but no end-effector: scaling the
    assembled system (collision terms included) keeps the step well conditioned on every seed,
    though some seeds amplify input roundoff (``_reference``'s sensitivity bound)."""
    from grim.motion.ik import ls_ik
    robot = load_robot(name)
    ees = ee_sets(robot)[which]
    coll, world = collision_case(robot)
    targets, _ = reachable_targets(robot, ees, 4)
    s = seeds(robot, 4, 4)
    q = np.asarray(ls_ik(robot, s, targets, ees, world=world, collision=coll, max_iter=1,
                         backend=backend, **SETTINGS, **COLL)[0])
    tol = 2e-4 if backend == "mantis-f32" else 1e-5
    rng = np.random.default_rng(0)
    for p in range(s.shape[0]):
        for k in range(s.shape[1]):
            q1, cond, sens = _reference(robot, s[p, k], targets[p], ees, coll, world, rng)
            assert cond < 1e7, f"{name} p={p} s={k}: cond {cond:.1e}"
            assert_close_scaled(q[p, k], q1, tol, f"{backend} {name} p={p} s={k} (sens {sens:.1e})",
                                extra=2 * sens)


@pytest.mark.parametrize("backend", BACKENDS)
def test_world_is_a_runtime_argument(backend, monkeypatch):
    """One collision-aware IK build serves worlds of any obstacle counts: the same FFI target takes
    one step on an empty world, the test world and a smaller one, each matching the reference."""
    from grim.motion import _build
    from grim.motion.ik import World, ls_ik
    robot = load_robot("fr3")
    ees = ee_sets(robot)[0]
    coll, world = collision_case(robot)
    targets, _ = reachable_targets(robot, ees, 2)
    s = seeds(robot, 2, 4)
    names, register = [], _build.register

    def spy(so, symbols):
        out = register(so, symbols)
        names.extend(out)
        return out
    monkeypatch.setattr(_build, "register", spy)
    tol = 2e-4 if backend == "mantis-f32" else 1e-5
    rng = np.random.default_rng(0)
    for w in (World(), world, World(boxes=world.boxes, cylinders=world.cylinders)):
        q = np.asarray(ls_ik(robot, s, targets, ees, world=w, collision=coll, max_iter=1,
                             backend=backend, **SETTINGS, **COLL)[0])
        for p in range(s.shape[0]):
            for k in range(s.shape[1]):
                q1, _, sens = _reference(robot, s[p, k], targets[p], ees, coll, w, rng)
                assert_close_scaled(q[p, k], q1, tol, f"{backend} counts {w.counts()} p={p} s={k}",
                                    extra=2 * sens)
    assert len(set(names)) == 1, f"rebuilt per world: {set(names)}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["iiwa14", "fr3"])
def test_canonical_ik_kkt(name, backend):
    from grim.motion.ik import build_canonical, run_canonical
    robot = load_robot(name)
    ees = ee_sets(robot)[0]
    targets, q = reachable_targets(robot, ees, 16)
    q_ref = np.clip(q + np.random.default_rng(3).normal(scale=0.3, size=q.shape), robot.lower, robot.upper)
    t = build_canonical(robot, ees, backend=backend)
    q_star, _ = (np.asarray(x) for x in run_canonical(t, q, q_ref, targets, max_iters=2000))
    for p in range(len(q)):
        T = K.frame_poses(robot, q_star[p])
        r = np.concatenate([K.pose_residual(T[e], targets[p, k]) for k, e in enumerate(ees)])
        assert np.linalg.norm(r) < 1e-3, f"{backend} {name} p={p}: |r|={np.linalg.norm(r):.2e}"
        assert np.linalg.norm(q_star[p] - q_ref[p]) <= np.linalg.norm(q[p] - q_ref[p]) + 1e-5
        J = np.concatenate([K.pose_jacobian(robot, q_star[p], e) for e in ees])
        d = q_star[p] - q_ref[p]
        N = np.eye(robot.n_act) - np.linalg.pinv(J) @ J
        assert np.linalg.norm(N @ d) < 1e-2 * max(np.linalg.norm(d), 1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_analytic_ik_fr3(backend):
    from grim.motion.analytic_ik import analytic_ik, arm_geometry
    from .test_analytic_ik import _matrix
    robot = load_robot("fr3")
    ee = robot.joint_index("fr3_joint7")
    g = arm_geometry(robot, ee)
    qs = np.random.default_rng(0).uniform(robot.lower, robot.upper, size=(64, robot.n_act))
    poses = np.stack([K.frame_poses(robot, q)[ee] for q in qs])
    targets = np.stack([_matrix(p) for p in poses])
    q7 = np.linspace(g.lower[6], g.upper[6], 64)
    q, err, found, _ = (np.asarray(x) for x in analytic_ik(robot, targets, ee, q7_samples=q7, backend=backend))
    _, _, found_grim, _ = (np.asarray(x) for x in analytic_ik(robot, targets, ee, q7_samples=q7))
    assert np.array_equal(found, found_grim), "a different set of targets was found"
    for b in np.nonzero(found)[0]:
        full = np.zeros(robot.n_act)
        full[g.act] = q[b]
        r = K.pose_residual(K.frame_poses(robot, full)[ee], poses[b])
        assert np.linalg.norm(r[:3]) < 1e-3 and np.linalg.norm(r[3:]) < 1e-3, (b, r)
        assert np.all(q[b] >= g.lower - 1e-5) and np.all(q[b] <= g.upper + 1e-5)
    again = [np.asarray(x) for x in analytic_ik(robot, targets, ee, q7_samples=q7, backend=backend)]
    assert all(np.array_equal(a, b) for a, b in zip(again, (q, err, found)))
