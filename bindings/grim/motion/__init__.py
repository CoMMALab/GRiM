"""GRiM motion generation: IK and trajectory-optimization kernels built per robot.

Every kernel is compiled for one :class:`MotionRobot` (and one collision setup) on first
use and cached on disk; there is no runtime robot description. See
``docs/source/user_guide/concepts/motion.rst`` for the design.
"""

from .robot import MotionRobot

__all__ = ["MotionRobot"]
