"""crocket dynamics (``CrocketDynamics``: float and double, thread / warp / block) vs Pinocchio,
float64: every op, including the composite ones (minv, fd_crba, fd_du)."""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import assert_close_scaled, config_samples, crocket_available, load_robot, requires_gpu

pin = pytest.importorskip("pinocchio")
pytestmark = [requires_gpu, pytest.mark.cuda_equivalence,
              pytest.mark.skipif(not crocket_available(), reason="crocket not installed")]

def _pin(robot):
    # Mimic joints follow their source in both models. A continuous joint is a revolute one
    # without limits; Pinocchio would give it nq = 2 (cos, sin), so it is read as revolute.
    xml = robot.urdf_xml.replace('type="continuous"', 'type="revolute"')
    model = pin.buildModelFromXML(xml, mimic=bool((robot.mimic_act_idx != -1).any()))
    idx_q = [model.idx_qs[model.getJointId(n)] for n in robot.actuated_names]
    idx_v = [model.idx_vs[model.getJointId(n)] for n in robot.actuated_names]
    return model, model.createData(), np.array(idx_q), np.array(idx_v)


def _states(robot, n=12, seed=0):
    rng = np.random.default_rng(seed)
    q = config_samples(robot, n_random=n)[-n:]
    scales = np.r_[np.zeros(1), np.ones(n // 3), 10 * np.ones(n - 1 - n // 3)]
    qd = rng.uniform(-1, 1, q.shape) * scales[:, None]
    qdd = rng.uniform(-1, 1, q.shape) * 5 * scales[:, None]
    return q, qd, qdd


RTOL = {"crocket-f32": 2e-5, "crocket-f64": 1e-6}   # f64 computes in double, returns float32


def _ref(model, data, iq, iv, op, q, qd, x):
    qp, vp, ap = np.zeros(model.nq), np.zeros(model.nv), np.zeros(model.nv)
    qp[iq], vp[iv], ap[iv] = q, qd, x
    ix = np.ix_(iv, iv)
    if op == "id":
        return pin.rnea(model, data, qp, vp, ap)[iv]
    M = pin.crba(model, data, qp)
    M = (np.triu(M) + np.triu(M, 1).T)[ix]
    if op == "crba":
        return M
    if op == "minv":
        return np.linalg.inv(M)
    if op == "fd_crba":   # M^-1 (tau - b): also defined with mimic joints, unlike ABA
        return np.linalg.solve(M, x - pin.rnea(model, data, qp, vp, 0 * ap)[iv])
    if op == "fd":
        return pin.aba(model, data, qp, vp, ap)[iv]
    if op == "id_du":
        dq, dv, _ = pin.computeRNEADerivatives(model, data, qp, vp, ap)
        return np.hstack([dq[ix], dv[ix]])
    pin.computeABADerivatives(model, data, qp, vp, ap)
    return np.hstack([data.ddq_dq[ix], data.ddq_dv[ix]])


@pytest.mark.parametrize("backend", ["crocket-f32", "crocket-f64"])
@pytest.mark.parametrize("name", ["iiwa14", "rizon4", "fr3", "gen3"])   # fr3: mimic fingers; gen3: continuous
def test_crocket_dynamics_match_pinocchio(name, backend):
    from grim.motion.dynamics import CrocketDynamics
    robot = load_robot(name)
    model, data, iq, iv = _pin(robot)
    q, qd, x = _states(robot)
    q32, qd32, x32 = (a.astype(np.float32).astype(float) for a in (q, qd, x))
    dyn = CrocketDynamics(robot, backend)
    for op in dyn.ops:
        want = np.array([_ref(model, data, iq, iv, op, q32[b], qd32[b], x32[b]) for b in range(len(q))])
        args = (q,) if op in ("crba", "minv") else (q, qd, x)
        for tier in ("thread", "warp", "block"):
            got = np.asarray(dyn(op, *args, tier=tier))
            # fd_du = -M^-1 dtau: float32 conditioning of M amplifies the input rounding.
            rtol = RTOL[backend] * (10 if op == "fd_du" else 1)
            assert_close_scaled(got, want, rtol, f"{name} {backend} {tier} {op}")
