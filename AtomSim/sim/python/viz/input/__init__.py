"""Input devices for teleop. See base.py for the core types and protocol."""

from __future__ import annotations

from .base import InputDevice, TeleopInput
from .composite import CompositeInput
from .gamepad import GamepadInput, detect_gamepad
from .keyboard import KeyboardInput


def diff_drive_wheels_from_input(
    inp: TeleopInput,
    max_wheel_speed: float,
    turn_rate_k: float = 1.0,
) -> tuple[float, float]:
    """Tank-style mix: (forward, turn) → (v_left, v_right).

    `turn_rate_k` < 1 trades turn-aggressiveness for forward velocity when the
    user pushes both at once (matches what you want for a manipulator-equipped
    robot — turning shouldn't fully zero out forward motion)."""
    v_left = max_wheel_speed * (inp.forward - inp.turn * turn_rate_k)
    v_right = max_wheel_speed * (inp.forward + inp.turn * turn_rate_k)
    return v_left, v_right


__all__ = [
    "CompositeInput",
    "GamepadInput",
    "InputDevice",
    "KeyboardInput",
    "TeleopInput",
    "detect_gamepad",
    "diff_drive_wheels_from_input",
]
