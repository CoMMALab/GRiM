"""Fused sphere FK: numpy/Pinocchio surfaces and launch invariances."""
from pathlib import Path
import xml.etree.ElementTree as ET

import jax
import numpy as np
import pinocchio as pin
import pytest

from grim.motion import MotionRobot
from grim.motion.kinematics import sphere_kinematics
from grim.motion.reference.kinematics import sphere_centres
from .conftest import config_samples

ROBOTS = Path(__file__).resolve().parents[2] / "benchmarks/robots"


@pytest.mark.parametrize("name", ["panda", "baxter_spherized", "g1"])
def test_spheres(name):
    path = ROBOTS / f"{name}.urdf"
    robot = MotionRobot.from_urdf(str(path))
    joints, centres, links = [], [], []
    for link in ET.parse(path).getroot().findall("link"):
        for col in link.findall("collision"):
            if col.find("geometry/sphere") is not None:
                origin = col.find("origin")
                centres.append([float(x) for x in (
                    origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
                links.append(link.get("name"))
                joints.append(int(robot.link_parent_joint[robot.link_names.index(links[-1])]))
    joints, centres = np.asarray(joints), np.asarray(centres)
    q = config_samples(robot, n_random=7).astype(np.float32)
    ref = np.array([sphere_centres(robot, x, joints, centres) for x in q])
    model = pin.buildModelFromUrdf(str(path))
    data = model.createData()
    perm = [robot.actuated_names.index(n) for n in list(model.names)[1:]]
    expected = []
    for x in q:
        pin.framesForwardKinematics(model, data, x[perm].astype(float))
        expected.append([data.oMf[model.getFrameId(l, pin.BODY)].act(c)
                         for l, c in zip(links, centres)])
    np.testing.assert_allclose(ref, expected, atol=1e-10, rtol=1e-10)
    for traced in (False, True):
        baseline = None
        for threads in (1, 32, 47, 128, 256):
            fn = jax.jit(lambda x: sphere_kinematics(
                robot, x, joints, centres, traced=traced, threads=threads))
            actual = np.asarray(fn(q))
            np.testing.assert_allclose(actual, ref, atol=2e-6, rtol=2e-6)
            np.testing.assert_array_equal(actual, fn(q))
            np.testing.assert_array_equal(actual[:1], fn(q[:1]))
            np.testing.assert_array_equal(actual[::-1], fn(q[::-1]))
            if baseline is not None:
                np.testing.assert_array_equal(actual, baseline)
            baseline = actual
        # Root-only, duplicates, sparse selected frames, and empty outputs.
        js = np.array([-1, joints[-1], joints[-1]])
        cs = np.array([[.1, .2, .3], [.2, 0, .1], [.2, 0, .1]])
        out = sphere_kinematics(robot, q, js, cs, traced=traced)
        oracle = [sphere_centres(robot, x, js, cs) for x in q]
        np.testing.assert_allclose(out, oracle, atol=2e-6, rtol=2e-6)
        assert sphere_kinematics(robot, q, np.array([], int), np.empty((0, 3)),
                                 traced=traced).shape == (len(q), 0, 3)
