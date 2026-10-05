#!/usr/bin/env python3
"""Compile and run the printGRiM CUDA executable to display generated kernel outputs.

Usage:
    python examples/codegen/print_grim.py PATH_TO_URDF [-n NAMESPACE] [-d] [-f]
    python examples/codegen/print_grim.py           # if grim.cuh already exists
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from URDFParser import URDFParser
from grim_codegen import GRiMCodeGenerator
from grim_codegen.cli import parseInputs, validateRobot


def detect_cuda_arch() -> str:
    env_arch = os.environ.get("GRIM_CUDA_ARCH")
    if env_arch:
        return env_arch.replace("sm_", "").replace("compute_", "").replace(".", "")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                compute_cap = line.strip()
                if compute_cap:
                    return compute_cap.replace(".", "")

    return "86"


def main():
    inputs = parseInputs(NO_ARG_OPTION=True)
    arch = detect_cuda_arch()

    with tempfile.TemporaryDirectory(prefix="grim_print_") as tmpdir:
        build_dir = Path(tmpdir)
        build_header = build_dir / "grim.cuh"

        if inputs is not None:
            # parseInputs returns the argparse Namespace (2026-09-08; the old
            # tuple unpack here had drifted a field short and crashed).
            parser = URDFParser()
            robot = parser.parse(inputs.urdf_path, floating_base=inputs.floating_base)

            validateRobot(robot, NO_ARG_OPTION=True)

            print("-----------------")
            print("Generating GRiM.cuh")
            print("-----------------")
            codegen = GRiMCodeGenerator(robot, inputs.debug, True, FILE_NAMESPACE=inputs.namespace)
            include_homogenous_transforms = not inputs.floating_base
            codegen.gen_all_code(
                include_homogenous_transforms=include_homogenous_transforms,
                fixed_target_name=inputs.fixed_target_names,
                output_path=str(build_header),
            )
            print(f"New code generated in temporary build directory: {build_header}")
        else:
            grim_header = Path("grim.cuh").resolve()
            if not grim_header.exists():
                print("grim.cuh does not exist. Generate it first or pass a URDF path.")
                sys.exit(1)
            shutil.copyfile(grim_header, build_header)

        shutil.copyfile(Path(__file__).resolve().parent / "printGRiM.cu", build_dir / "printGRiM.cu")

        print("-----------------")
        print("Compiling printGRiM")
        print("-----------------")
        result = subprocess.run(
            [
                "nvcc",
                "-std=c++11",
                "-o",
                "printGRiM.exe",
                "printGRiM.cu",
                "-gencode",
                f"arch=compute_{arch},code=sm_{arch}",
            ],
            cwd=build_dir,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            print("Compilation failed.")
            sys.exit(result.returncode)

        print("-----------------")
        print("Running printGRiM")
        print("-----------------")
        result = subprocess.run(
            ["./printGRiM.exe"],
            cwd=build_dir,
            capture_output=True,
            text=True,
        )
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            print("Runtime failed.")
            sys.exit(result.returncode)

        print(result.stdout)


if __name__ == "__main__":
    main()
