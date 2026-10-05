#!/usr/bin/env python3
"""Collision codegen: emit GRiM's two-tier `config_free` collision check for iiwa14.

GRiM can spherize a robot's URDF collision geometry into a broad/fine two-tier
covering-sphere hierarchy and emit a `grim_collision::` `config_free(q)` device
routine that returns a collision-free verdict against a sphere/capsule/cuboid/plane
environment. The broad tier rejects/masks quickly; the fine tier confirms — the
verdict is bit-identical to a fine-only check (see
test/cuda_equivalents/test_cuda_collision_two_tier.py).

Requires: pip install robot_descriptions  (dev dependency)

Run:
    python examples/codegen/generate_collision.py --output /tmp/grim_iiwa14_collision.cuh
"""
from __future__ import annotations

import argparse
from pathlib import Path

from robot_descriptions import iiwa14_description

from URDFParser import URDFParser
from grim_codegen import GRiMCodeGenerator
from grim_codegen.algorithms._collision import multi_tier_collision_spec_from_urdf

URDF_PATH = iiwa14_description.URDF_PATH


def main():
    ap = argparse.ArgumentParser(description="Generate iiwa14 GRiM code with two-tier collision.")
    ap.add_argument("--output", default="grim.cuh", help="Path for the generated CUDA header.")
    ap.add_argument("--resolutions", default="0.1,0.05",
                    help="Comma-separated covering-sphere resolutions, coarse->fine (two-tier).")
    args = ap.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolutions = [float(x) for x in args.resolutions.split(",") if x.strip()]

    print(f"Parsing URDF: {URDF_PATH}")
    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base=False)

    print(f"Spherizing collision geometry at resolutions {resolutions} (coarse -> fine)...")
    collision_spec = multi_tier_collision_spec_from_urdf(robot, URDF_PATH, resolutions)

    print("Generating GRiM CUDA code with the two-tier config_free collision routine...")
    codegen = GRiMCodeGenerator(robot, FILE_NAMESPACE="grid")
    codegen.gen_all_code(output_path=str(output_path), collision_spec=collision_spec)

    print(f"Done: {output_path} written (contains grim_collision:: SDF primitives + config_free).")
    print("Validate on-GPU with: pytest test/cuda_equivalents/test_cuda_collision_two_tier.py")


if __name__ == "__main__":
    main()
