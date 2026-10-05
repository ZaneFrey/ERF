#!/usr/bin/env python3
"""Validate the persistent state written by the ClassicAD smoke test."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def state_lines(path: Path, tag: str, width: int) -> list[list[float]]:
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != [tag]:
        raise AssertionError(f"invalid {tag} file: {path}")

    turbine_count = int(lines[1][0])
    rows = [[float(value) for value in line] for line in lines[2:]]
    if len(rows) != turbine_count or any(len(row) != width for row in rows):
        raise AssertionError(f"invalid turbine state dimensions in {path}")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise AssertionError(f"non-finite turbine state in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()

    checkpoints = sorted(args.run_directory.glob("chk[0-9]*"))
    if not checkpoints:
        raise AssertionError("ClassicAD smoke test did not write a checkpoint")
    checkpoint = checkpoints[-1]

    memory = state_lines(
        checkpoint / "ClassicADMemoryState", "ClassicADMemoryState_v1", 2
    )
    if any(row[0] != 1.0 for row in memory):
        raise AssertionError("filtered disk-velocity state was not initialized")

    yaw_path = checkpoint / "ClassicADYawState"
    lines = [line.split() for line in yaw_path.read_text().splitlines() if line.strip()]
    if len(lines) < 4 or lines[0] != ["ClassicADYawState_v1"]:
        raise AssertionError(f"invalid ClassicADYawState file: {yaw_path}")
    turbine_count = int(lines[1][0])
    if len(lines[2]) != 2:
        raise AssertionError("invalid ClassicAD yaw scheduler state")
    scheduler = [float(value) for value in lines[2]]
    yaw = [[float(value) for value in line] for line in lines[3:]]
    if len(yaw) != turbine_count or any(len(row) != 8 for row in yaw):
        raise AssertionError("invalid ClassicAD yaw turbine state dimensions")
    values = scheduler + [value for row in yaw for value in row]
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("non-finite ClassicAD yaw state")
    if any(row[7] != 1.0 for row in yaw):
        raise AssertionError("yaw sensor filter was not initialized")
    if not any(abs(row[0]) > 0.0 for row in yaw):
        raise AssertionError("dynamic yaw did not respond to angled inflow")

    print(f"PASS: validated ClassicAD state in {checkpoint}")


if __name__ == "__main__":
    main()
