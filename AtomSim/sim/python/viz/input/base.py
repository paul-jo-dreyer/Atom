"""Input abstraction: a normalized teleop signal independent of source.

Every InputDevice (keyboard, gamepad, ...) emits a `TeleopInput` per poll,
in the same `forward ∈ [-1, 1]`, `turn ∈ [-1, 1]` space. Mapping to wheel
velocities lives in `viz.input.diff_drive_wheels_from_input` so analog and
discrete devices feed the same downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pygame


@dataclass
class TeleopInput:
    """Normalized teleop signal at one frame."""
    forward: float = 0.0      # [-1, 1] — +1 full forward, -1 full reverse
    turn: float = 0.0         # [-1, 1] — +1 full CCW (left), -1 full CW (right)
    reset: bool = False       # one-shot: re-init episode
    quit: bool = False        # one-shot: stop the loop


class InputDevice(Protocol):
    """Polled-each-frame input source.

    `events` is the list returned by `pygame.event.get()` for the frame, used
    to detect edge-triggered actions (key presses, button-down). Devices
    typically also call `pygame.key.get_pressed()` or `joystick.get_axis()`
    for held-state inputs — both styles are valid.
    """

    def poll(self, events: list[pygame.event.Event]) -> TeleopInput: ...
