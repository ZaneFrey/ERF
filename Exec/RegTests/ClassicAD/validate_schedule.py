#!/usr/bin/env python3
"""Validate scheduled ClassicAD forcing across a checkpoint restart."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


DT = 0.05
START_TIME = 0.05
RAMP_TIME = 0.10
DERATE_TIME = 0.10
EVENTS = ((0.10, 0.0), (0.20, 1.0), (0.30, 0.5))
TIME_TOLERANCE = 1.0e-12


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    scale = max(1.0, abs(expected))
    if not math.isfinite(actual) or abs(actual - expected) > tolerance * scale:
        raise AssertionError(
            f"{label}: got {actual:.17g}, expected {expected:.17g}"
        )


def advance_segment(
    response: float, command: float, duration: float
) -> tuple[float, float]:
    if duration <= 0.0:
        return response, 0.0
    decay = math.exp(-duration / DERATE_TIME)
    integral = (
        command * duration
        + (response - command) * DERATE_TIME * (1.0 - decay)
    )
    return command + (response - command) * decay, integral


def advance_control(
    response: float, time: float
) -> tuple[float, float, float]:
    command = 1.0
    for event_time, fraction in EVENTS:
        if event_time <= time + TIME_TOLERANCE:
            command = fraction
        else:
            break

    segment_start = time
    integral = 0.0
    for event_time, fraction in EVENTS:
        if event_time <= time + TIME_TOLERANCE:
            continue
        if event_time > time + DT + TIME_TOLERANCE:
            break
        event_time = min(event_time, time + DT)
        response, contribution = advance_segment(
            response, command, event_time - segment_start
        )
        integral += contribution
        segment_start = event_time
        command = fraction

    response, contribution = advance_segment(
        response, command, time + DT - segment_start
    )
    integral += contribution
    return response, integral / DT, command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()

    table = args.run_directory / "power_output_ClassicAD.txt"
    lines = [line.split() for line in table.read_text().splitlines() if line.strip()]
    header = lines[0][1:]
    rows = [[float(value) for value in line] for line in lines[1:]]
    index = {name: column for column, name in enumerate(header)}
    required = (
        "time",
        "Ud_f_0",
        "forcing_factor_0",
        "forcing_enabled_0",
        "T_target_0",
        "T_applied_0",
        "Q_target_0",
        "Q_applied_0",
        "Q_LES_0",
    )
    missing = [name for name in required if name not in index]
    if missing:
        raise AssertionError(f"missing scheduled diagnostics: {', '.join(missing)}")

    response = 1.0
    saw_disabled_statistics = False
    saw_reenabled = False
    previous_time = -math.inf
    for row_number, row in enumerate(rows, start=1):
        time = row[index["time"]]
        if time < previous_time:
            raise AssertionError("scheduled diagnostic times decrease across restart")
        previous_time = time

        response, response_average, command = advance_control(response, time)
        startup_factor = 1.0 - math.exp(
            -max(time - START_TIME, 0.0) / RAMP_TIME
        )
        expected_factor = startup_factor * response_average
        close(
            row[index["forcing_factor_0"]],
            expected_factor,
            3.0e-12,
            f"row {row_number} forcing factor",
        )
        expected_enabled = 1.0 if command > 0.0 else 0.0
        if row[index["forcing_enabled_0"]] != expected_enabled:
            raise AssertionError(f"row {row_number} enabled flag is incorrect")

        target = row[index["T_target_0"]]
        applied = row[index["T_applied_0"]]
        close(applied, expected_factor * target, 3.0e-12,
              f"row {row_number} applied thrust")
        torque_target = row[index["Q_target_0"]]
        torque_applied = row[index["Q_applied_0"]]
        close(torque_applied, expected_factor * torque_target, 3.0e-12,
              f"row {row_number} applied torque")

        if expected_enabled == 0.0:
            if abs(row[index["Ud_f_0"]]) <= 1.0 or abs(target) <= 1.0:
                raise AssertionError("disabled turbine stopped reporting target statistics")
            saw_disabled_statistics = True
        if time >= 0.20 and expected_enabled == 1.0:
            saw_reenabled = True

        if abs(torque_applied) > 1.0:
            les_torque = row[index["Q_LES_0"]]
            if les_torque * torque_applied >= 0.0:
                raise AssertionError("scheduled deposited torque has the wrong sign")
            close(abs(les_torque), abs(torque_applied), 2.0e-2,
                  f"row {row_number} deposited torque")

    if not saw_disabled_statistics or not saw_reenabled:
        raise AssertionError("scheduled run did not cover disable and re-enable states")

    checkpoint = args.run_directory / "chk00004" / "ClassicADMemoryState"
    state = [line.split() for line in checkpoint.read_text().splitlines() if line.strip()]
    if state[0] != ["ClassicADMemoryState_v2"] or len(state[2]) != 4:
        raise AssertionError("scheduled checkpoint does not contain v2 control state")
    close(float(state[2][3]), math.exp(-1.0), 3.0e-12,
          "checkpoint forcing response")

    print("PASS: validated scheduled ClassicAD forcing and restart continuity")


if __name__ == "__main__":
    main()
