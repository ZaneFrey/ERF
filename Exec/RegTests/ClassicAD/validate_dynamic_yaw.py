#!/usr/bin/env python3
"""Validate ClassicAD V1.1 yaw diagnostics against the discrete equations."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path


def wrap_pi(value: float) -> float:
    while value <= -math.pi:
        value += 2.0 * math.pi
    while value > math.pi:
        value -= 2.0 * math.pi
    return value


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    scale = max(1.0, abs(expected))
    if not math.isfinite(actual) or abs(actual - expected) > tolerance * scale:
        raise AssertionError(f"{label}: got {actual:.17g}, expected {expected:.17g}")


def read_table(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines or not lines[0].startswith("#"):
        raise AssertionError(f"missing diagnostics header in {path}")
    header = lines[0][1:].split()
    rows = [[float(value) for value in line.split()] for line in lines[1:]]
    if not rows:
        raise AssertionError(f"no diagnostics rows in {path}")
    if any(len(row) != len(header) for row in rows):
        raise AssertionError(f"inconsistent diagnostics width in {path}")
    return header, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--initial-reference-deg", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=2.0e-10)
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--require-filtering", action="store_true")
    args = parser.parse_args()

    header, rows = read_table(args.run_directory / "yaw_angles_ClassicAD.txt")
    index = {name: i for i, name in enumerate(header)}
    required = (
        "time", "psi_ref_deg_0", "psi_rotor_deg_0", "phi_cmd_deg_0",
        "Us_raw_0", "Vs_raw_0", "Us_f_0", "Vs_f_0", "yaw_offset_deg_0",
    )
    missing = [name for name in required if name not in index]
    if missing:
        raise AssertionError(f"missing columns: {', '.join(missing)}")

    alpha = 1.0 if args.tau == 0.0 else 1.0 - math.exp(-args.dt / args.tau)
    psi = math.radians(args.initial_reference_deg)
    uf = vf = 0.0
    initialized = False
    previous_time = None
    raw_angles: list[float] = []
    filtered_angles: list[float] = []
    for row_number, row in enumerate(rows, start=1):
        if not all(math.isfinite(value) for value in row):
            raise AssertionError(f"non-finite value in yaw row {row_number}")
        if previous_time is not None and row[index["time"]] <= previous_time:
            raise AssertionError("yaw diagnostic times are not strictly increasing")
        previous_time = row[index["time"]]
        uraw = row[index["Us_raw_0"]]
        vraw = row[index["Vs_raw_0"]]
        if not initialized or args.tau == 0.0:
            uf, vf = uraw, vraw
            initialized = True
        else:
            uf += alpha * (uraw - uf)
            vf += alpha * (vraw - vf)
        phi = math.atan2(vf, uf)
        raw_angles.append(math.atan2(vraw, uraw))
        filtered_angles.append(phi)
        psi = wrap_pi(psi + alpha * wrap_pi(phi - psi))
        offset = math.radians(row[index["yaw_offset_deg_0"]])
        rotor = wrap_pi(psi + offset)

        close(row[index["Us_f_0"]], uf, args.tolerance, f"row {row_number} filtered U")
        close(row[index["Vs_f_0"]], vf, args.tolerance, f"row {row_number} filtered V")
        close(math.radians(row[index["phi_cmd_deg_0"]]), phi,
              args.tolerance, f"row {row_number} phi")
        close(wrap_pi(math.radians(row[index["psi_ref_deg_0"]]) - psi), 0.0,
              args.tolerance, f"row {row_number} reference yaw")
        close(wrap_pi(math.radians(row[index["psi_rotor_deg_0"]]) - rotor), 0.0,
              args.tolerance, f"row {row_number} rotor yaw")

    _, power_rows = read_table(args.run_directory / "power_output_ClassicAD.txt")
    if not all(math.isfinite(value) for row in power_rows for value in row):
        raise AssertionError("non-finite ClassicAD power diagnostic")

    if args.require_checkpoint:
        states = list(args.run_directory.glob("chk*/ClassicADYawState"))
        if not states:
            raise AssertionError("checkpoint does not contain ClassicADYawState")
        for state in states:
            if state.read_text().splitlines()[0] != "ClassicADYawState_v1":
                raise AssertionError(f"bad yaw checkpoint version in {state}")

    if args.require_filtering:
        if len(raw_angles) < 3:
            raise AssertionError("at least three yaw rows are required for filtering statistics")
        raw_delta = [wrap_pi(b-a) for a, b in zip(raw_angles, raw_angles[1:])]
        filtered_delta = [wrap_pi(b-a) for a, b in zip(filtered_angles, filtered_angles[1:])]
        raw_variance = statistics.pvariance(raw_delta)
        filtered_variance = statistics.pvariance(filtered_delta)
        if raw_variance <= 0.0 or filtered_variance >= raw_variance:
            raise AssertionError(
                f"filter did not reduce angle-increment variance: raw={raw_variance}, "
                f"filtered={filtered_variance}"
            )
        print(f"filtered/raw angle-increment variance = {filtered_variance/raw_variance:.6g}")

    print(f"PASS: {len(rows)} yaw rows satisfy the ClassicAD V1.1 discrete equations")


if __name__ == "__main__":
    main()
