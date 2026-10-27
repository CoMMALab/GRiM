"""The crocket IK builds (``backend="crocket-f32"`` / ``"crocket-f64"``, kernels/ik_crocket)
against the float64 oracle, where test_ik_solvers.py and test_ls_ik.py do not already cover them:

* DLS iterations against the transcription in grim.motion.reference.ik;
* one LS / SQP step WITH collision against a float64 transcription of the Gauss-Newton collision
  terms (gradients by central differences of the oracle's own distances), on the seeds whose
  system is well conditioned (see ``_collision_step``);
* canonical IK's KKT certificate and analytic IK's FK certificate, as for the hand-written kernels.

Every build is a test subject: these run per robot and per end-effector set.
"""

from __future__ import annotations

import numpy as np
import pytest

from grim.motion.reference import collision as C
from grim.motion.reference import ik as ref
from grim.motion.reference import kinematics as K

from .conftest import assert_close_scaled, crocket_available, ee_sets, load_robot, requires_gpu
from .ik_cases import collision_case, reachable_targets, seeds

pytestmark = [requires_gpu, pytest.mark.cuda_equivalence,
              pytest.mark.skipif(not crocket_available(), reason="crocket not installed")]

BACKENDS = ["crocket-f32", "crocket-f64"]
SETTINGS = dict(pos_weight=50.0, ori_weight=10.0, lambda_init=5e-3, eps_pos=1e-4, eps_ori=1e-3)
COLL = dict(collision_weight=1e4, collision_margin=0.02)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,which", [("fr3", 0), ("iiwa14", 0), ("baxter", 1), ("g1", 1)])
@pytest.mark.parametrize("iters", [0, 1, 2])
def test_dls_iterations_match_reference(name, which, iters, backend):
    from grim.motion.ik import dls_ik
    from grim.motion.kinematics import solved_columns
    robot = load_robot(name)
    ees = ee_sets(robot)[which]
    targets, _ = reachable_targets(robot, ees, 3)
    s = seeds(robot, 3, 4)
    q, e = (np.asarray(x) for x in dls_ik(robot, s, targets, ees, max_iter=iters, backend=backend,
                                          **SETTINGS))
    cols = solved_columns(robot, ees, chain_only=True)
    tol = 1e-4 if backend == "crocket-f32" else 1e-6
    for p in range(s.shape[0]):
        for k in range(s.shape[1]):
            q_ref, e_ref = ref.dls_ik(robot, s[p, k], targets[p], ees, cols, lower=robot.lower,
                                      upper=robot.upper, fixed_mask=np.zeros(robot.n_act),
                                      max_iter=iters, **SETTINGS)
            assert_close_scaled(q[p, k], q_ref, tol, f"{backend} {name} p={p} s={k} q")
            assert_close_scaled(e[p, k], e_ref, 10 * tol, f"{backend} {name} p={p} s={k} err")


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
             (C.sphere_box, world.boxes), (C.sphere_halfspace, world.halfspaces)]
    for sph, j in zip(coll.robot_spheres, coll.robot_sphere_joint):
        c = K.quat_rotate(T[j][:4], sph[:3]) + T[j][4:]
        out += [f(c, sph[3], o) for f, arr in prims for o in arr]
    return np.array(out)


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
    sc = np.sqrt((J ** 2).sum(0)) + 1e-8
    Js = J / sc
    A, b = Js.T @ Js + lam * np.eye(robot.n_act), -Js.T @ fw
    d0 = _distances(robot, q, coll, world)
    for c in np.flatnonzero(d0 < m):
        g = np.zeros(robot.n_act)
        for a in range(robot.n_act):
            e = np.zeros(robot.n_act)
            e[a] = 1e-6
            g[a] = (_distances(robot, q + e, coll, world)[c] - _distances(robot, q - e, coll, world)[c]) / 2e-6
        g /= sc
        A += wc * np.outer(g, g)
        b += wc * (m - d0[c]) * g
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
@pytest.mark.parametrize("name,which", [("fr3", 0), ("iiwa14", 1)])
def test_collision_step_matches_reference(name, which, backend):
    """A joint that moves collision spheres but no end-effector has column scale 1e-8, which
    makes the step's system singular (its collision rows are scaled by 1e8): any
    implementation's step is arbitrary there, the hand-written kernel's included. Those seeds are
    excluded by the reference's condition number; the robots here have none."""
    from grim.motion.ik import ls_ik
    robot = load_robot(name)
    ees = ee_sets(robot)[which]
    coll, world = collision_case(robot)
    targets, _ = reachable_targets(robot, ees, 4)
    s = seeds(robot, 4, 4)
    q = np.asarray(ls_ik(robot, s, targets, ees, world=world, collision=coll, max_iter=1,
                         backend=backend, **SETTINGS, **COLL)[0])
    tol = 2e-4 if backend == "crocket-f32" else 1e-5
    for p in range(s.shape[0]):
        for k in range(s.shape[1]):
            q1, cond = _collision_step(robot, s[p, k].astype(np.float64), targets[p], ees, coll, world)
            if cond < 1e7:
                assert_close_scaled(q[p, k], q1, tol, f"{backend} {name} p={p} s={k}")


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
