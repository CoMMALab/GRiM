"""Traced dynamics (cricket) at the thread and warp-split tiers vs Pinocchio, float64.

Principle 1 samples (zero, conservative, high velocity / acceleration) through
grim.motion's config sampler; every op at both tiers; fd via ABA and via CRBA + Cholesky.
Needs cricket's Python extension (skipped otherwise)."""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import assert_close_scaled, config_samples, cricket_available, load_robot, requires_gpu

pin = pytest.importorskip("pinocchio")
pytestmark = [requires_gpu, pytest.mark.cuda_equivalence,
              pytest.mark.skipif(not cricket_available(), reason="cricket not installed")]


def _pin(robot):
    # Mimic joints follow their source in both models (cricket traces with mimic=true).
    model = pin.buildModelFromXML(robot.urdf_xml, mimic=bool((robot.mimic_act_idx != -1).any()))
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


@pytest.mark.parametrize("tier", ["thread", "block"])
@pytest.mark.parametrize("name", ["iiwa14", "rizon4", "fr3"])
def test_traced_dynamics_match_pinocchio(name, tier):
    from grim.motion.dynamics import TracedDynamics
    robot = load_robot(name)
    model, data, iq, iv = _pin(robot)
    q, qd, qdd = _states(robot)
    dyn = TracedDynamics(robot)
    mimic = "id_du" not in dyn.ops            # fr3: fingers mimic -> id, crba, fd_crba only
    tau = np.asarray(dyn.inverse_dynamics(q, qd, qdd, tier))
    M = np.asarray(dyn.mass_matrix(q, tier))
    D = None if mimic else np.asarray(dyn.inverse_dynamics_gradient(q, qd, qdd, tier))
    for b in range(len(q)):
        qp, vp, ap = (np.zeros(model.nq), np.zeros(model.nv), np.zeros(model.nv))
        qp[iq], vp[iv], ap[iv] = q[b], qd[b], qdd[b]
        t_ref = pin.rnea(model, data, qp, vp, ap)[iv]
        assert_close_scaled(tau[b], t_ref, 2e-5, f"{name} {tier} id b={b}")
        M_ref = pin.crba(model, data, qp)
        M_ref = np.triu(M_ref) + np.triu(M_ref, 1).T
        assert_close_scaled(M[b], M_ref[np.ix_(iv, iv)], 2e-5, f"{name} {tier} crba b={b}")
        if D is not None:
            dq, dv, _ = pin.computeRNEADerivatives(model, data, qp, vp, ap)
            D_ref = np.hstack([dq[np.ix_(iv, iv)], dv[np.ix_(iv, iv)]])
            assert_close_scaled(D[b], D_ref, 5e-5, f"{name} {tier} id_du b={b}")
    for via_crba in ((True,) if mimic else (False, True)):
        qdd_out = np.asarray(dyn.forward_dynamics(q, qd, tau, tier, via_crba=via_crba))
        assert_close_scaled(qdd_out, qdd, 1e-3, f"{name} {tier} fd crba={via_crba}")


def test_continuous_joints_are_refused():
    from grim.motion.dynamics import TracedDynamics
    with pytest.raises(ValueError, match="continuous"):
        TracedDynamics(load_robot("gen3"))
