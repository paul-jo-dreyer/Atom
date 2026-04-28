"""Composite input: combine multiple devices into one signal.

forward/turn are summed and clipped to [-1, 1]. reset/quit are OR'd. Lets
you keep keyboard active alongside a gamepad without device-switching."""

from __future__ import annotations

import pygame

from .base import InputDevice, TeleopInput


class CompositeInput:
    def __init__(self, *devices: InputDevice) -> None:
        if not devices:
            raise ValueError("CompositeInput requires at least one device")
        self._devices = devices

    def poll(self, events: list[pygame.event.Event]) -> TeleopInput:
        fwd = 0.0
        turn = 0.0
        reset = False
        do_quit = False
        for d in self._devices:
            inp = d.poll(events)
            fwd += inp.forward
            turn += inp.turn
            reset = reset or inp.reset
            do_quit = do_quit or inp.quit
        return TeleopInput(
            forward=max(-1.0, min(1.0, fwd)),
            turn=max(-1.0, min(1.0, turn)),
            reset=reset,
            quit=do_quit,
        )
