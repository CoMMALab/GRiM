Collision code generation
=========================

GRiM generates CUDA routines for self- and environment-collision checks from
URDF collision geometry. This is collision checking, not a contact-dynamics
simulator. The collision interface is generated CUDA C++, not a promise that
every collision operation is exposed through the NumPy, JAX, or PyTorch handles.

Generate collision routines
---------------------------

After the :doc:`installation <../getting_started/installation>`, run from the
repository root with a URDF whose collision assets resolve on your machine:

.. code-block:: shell

   grim-generate path/to/robot.urdf --collision --collision-res 0.10,0.05

This enables collision code generation and a coarse-to-fine covering-sphere
representation. The generated ``grim_collision::config_free`` device routine
checks a configuration against the configured self-collision pairs and obstacle
environment. It is called from your CUDA code, with device model/environment
data and the required scratch storage.

``--collision-native`` selects a representation using native sphere/capsule
rows where supported; remaining geometry uses covering spheres. It is a
different approximation and must be labeled separately in a comparison.

The repository example ``examples/codegen/generate_collision.py`` shows the
programmatic two-tier flow. See ``grim_codegen/cli.py`` for the current CLI and
``grim_codegen/collision/grim_collision_geometry.cuh`` for geometry types.
The integration notes in ``grim_codegen/collision/HANDOFF.md`` describe scratch
layouts and device calls; older module-name examples there predate the current
``grim-generate`` entry point.

Geometry coverage and correctness
---------------------------------

* Sphere spacing controls geometry resolution and work. Coarse-to-fine checks
  are intended to match the fine-only verdict for the same representation.
* Covering geometry is an approximation of the original robot. Agreement with
  a fine-tier sphere check is not proof of exact mesh collision detection.
* Mesh resolution failures can produce warnings and partial coverage. Resolve
  every intended collision asset and inspect generated coverage before relying
  on a verdict. Self-collision pair exclusions must also match your application.
* Correctness tests live under ``test/cuda_equivalents/test_cuda_collision_*``;
  they cover geometry, pairs, tiering, costs, and native representations.

The :doc:`release measurements <../../release_measurements>` do not include
collision timings. When benchmarking collisions, report latency alongside
geometry coverage and fine-tier verdict agreement.
