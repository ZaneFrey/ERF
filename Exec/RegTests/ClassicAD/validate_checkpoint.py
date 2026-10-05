#!/usr/bin/env python3
"""Validate the persistent state written by the ClassicAD smoke test."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


DIAMETER = 60.0
CPPRIME = 0.8
TSR = 8.0
START_TIME = 0.05
RAMP_TIME = 0.10


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    scale = max(1.0, abs(expected))
    if not math.isfinite(actual) or abs(actual - expected) > tolerance * scale:
        raise AssertionError(f"{label}: got {actual:.17g}, expected {expected:.17g}")


def diagnostic_table(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2 or lines[0][0] != "#":
        raise AssertionError(f"invalid ClassicAD diagnostic table: {path}")
    header = lines[0][1:]
    rows = [[float(value) for value in line] for line in lines[1:]]
    if any(len(row) != len(header) for row in rows):
        raise AssertionError(f"inconsistent diagnostic row width in {path}")
    return header, rows


def validate_rotation(run_directory: Path) -> None:
    header, rows = diagnostic_table(
        run_directory / "power_output_ClassicAD.txt"
    )
    index = {name: column for column, name in enumerate(header)}
    required = (
        "time",
        "Ud_f_0",
        "rho_d_0",
        "P_rotor_0",
        "Omega_0",
        "Q_target_0",
        "Q_applied_0",
        "Q_LES_0",
    )
    missing = [name for name in required if name not in index]
    if missing:
        raise AssertionError(
            f"missing ClassicAD diagnostic columns: {', '.join(missing)}"
        )

    rotor_radius = 0.5 * DIAMETER
    rotor_area = math.pi * rotor_radius * rotor_radius
    observed_loaded_torque = False
    previous_time = -math.inf
    for row_number, row in enumerate(rows, start=1):
        if not all(math.isfinite(value) for value in row):
            raise AssertionError(f"non-finite diagnostic value in row {row_number}")

        time = row[index["time"]]
        if time <= previous_time:
            raise AssertionError("ClassicAD diagnostic times are not increasing")
        previous_time = time

        velocity = row[index["Ud_f_0"]]
        density = row[index["rho_d_0"]]
        expected_power = (
            0.5 * density * rotor_area * CPPRIME * velocity**3
        )
        expected_omega = TSR * velocity / rotor_radius
        expected_torque = (
            0.5
            * density
            * rotor_area
            * CPPRIME
            * velocity**2
            * rotor_radius
            / TSR
        )
        load_factor = (
            1.0 - math.exp(-max(time - START_TIME, 0.0) / RAMP_TIME)
            if RAMP_TIME > 0.0
            else 1.0
        )
        expected_applied_torque = load_factor * expected_torque

        close(
            row[index["P_rotor_0"]],
            expected_power,
            2.0e-12,
            f"row {row_number} rotor power",
        )
        close(
            row[index["Omega_0"]],
            expected_omega,
            2.0e-12,
            f"row {row_number} angular velocity",
        )
        close(
            row[index["Q_target_0"]],
            expected_torque,
            2.0e-12,
            f"row {row_number} target torque",
        )
        close(
            row[index["Q_applied_0"]],
            expected_applied_torque,
            2.0e-12,
            f"row {row_number} applied torque",
        )
        close(
            row[index["P_rotor_0"]],
            row[index["Q_target_0"]] * row[index["Omega_0"]],
            2.0e-12,
            f"row {row_number} power-torque relation",
        )

        if abs(expected_applied_torque) > 1.0:
            observed_loaded_torque = True
            deposited_torque = row[index["Q_LES_0"]]
            if deposited_torque * expected_applied_torque >= 0.0:
                raise AssertionError(
                    f"row {row_number} deposited torque has the wrong sign"
                )
            close(
                abs(deposited_torque),
                abs(expected_applied_torque),
                2.0e-2,
                f"row {row_number} deposited torque magnitude",
            )

    if not observed_loaded_torque:
        raise AssertionError("wake-rotation regression never applied nonzero torque")


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

    validate_rotation(args.run_directory)

    print(f"PASS: validated ClassicAD rotation and state in {checkpoint}")


if __name__ == "__main__":
    main()
