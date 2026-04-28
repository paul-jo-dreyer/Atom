"""Gamepad input via pygame.joystick — twin-stick layout.

Default axis mapping (SDL2 game-controller convention, modern Xbox / PS / etc.):
    axis 0 — left stick X    (left = -1, right = +1)
    axis 1 — left stick Y    (up   = -1, down  = +1)
    axis 2 — right stick X   (left = -1, right = +1)
    axis 3 — right stick Y   (up   = -1, down  = +1)

We map:
    forward = -axis[1]       (push LEFT stick  UP    = +forward)
    turn    = -axis[2]       (push RIGHT stick LEFT  = +turn = CCW)

A small radial-rescaled deadzone removes idle drift. Buttons:
    button 0   — reset (often "A" on Xbox / "Cross" on PS)
    button 6   — quit  (often "Back/Select")

Linux evdev sometimes shuffles axes (e.g. xpad puts triggers on 2 and 5 with
the right stick on 3 and 4). If yours is different, override `axis_*` /
`*_button` at construction.
"""

from __future__ import annotations

import pygame

from .base import InputDevice, TeleopInput


def _deadzone(v: float, dz: float) -> float:
    """Apply a deadzone with rescale: |v|<dz → 0, else linearly map to [0,1]."""
    if abs(v) < dz:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (abs(v) - dz) / (1.0 - dz)


class GamepadInput:
    def __init__(
        self,
        joy_index: int = 0,
        deadzone: float = 0.10,
        axis_forward: int = 1,        # left stick Y
        axis_turn: int = 2,           # right stick X
        invert_forward: bool = True,    # pygame Y is up=-1; we want up=+1
        invert_turn: bool = True,       # pygame X is right=+1; we want left=+1 (CCW)
        reset_button: int = 0,
        quit_button: int = 6,
    ) -> None:
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        if joy_index >= pygame.joystick.get_count():
            raise RuntimeError(
                f"No joystick at index {joy_index} (count={pygame.joystick.get_count()})"
            )
        self._js = pygame.joystick.Joystick(joy_index)
        self._js.init()
        self._dz = deadzone
        self._a_fwd = axis_forward
        self._a_turn = axis_turn
        self._inv_fwd = invert_forward
        self._inv_turn = invert_turn
        self._reset_btn = reset_button
        self._quit_btn = quit_button

    @property
    def name(self) -> str:
        return self._js.get_name()

    def poll(self, events: list[pygame.event.Event]) -> TeleopInput:
        fwd_raw = self._js.get_axis(self._a_fwd)
        turn_raw = self._js.get_axis(self._a_turn)
        fwd = _deadzone(-fwd_raw if self._inv_fwd else fwd_raw, self._dz)
        turn = _deadzone(-turn_raw if self._inv_turn else turn_raw, self._dz)

        reset = False
        do_quit = False
        for ev in events:
            if ev.type == pygame.JOYBUTTONDOWN and ev.instance_id == self._js.get_instance_id():
                if ev.button == self._reset_btn:
                    reset = True
                elif ev.button == self._quit_btn:
                    do_quit = True

        return TeleopInput(forward=fwd, turn=turn, reset=reset, quit=do_quit)


def detect_gamepad(**kwargs: object) -> GamepadInput | None:
    """Return a GamepadInput for joystick 0 if any is connected, else None."""
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        return None
    return GamepadInput(joy_index=0, **kwargs)  # type: ignore[arg-type]


# Type-check Protocol conformance.
def _conformance_check() -> InputDevice:
    return GamepadInput.__new__(GamepadInput)  # avoid construction (no joy in tests)
