"""Keyboard input: WASD/arrows for drive, R for reset, ESC/Q for quit."""

from __future__ import annotations

import pygame

from .base import InputDevice, TeleopInput


class KeyboardInput:
    """Discrete keyboard teleop. forward/turn ∈ {-1, 0, +1}."""

    def poll(self, events: list[pygame.event.Event]) -> TeleopInput:
        keys = pygame.key.get_pressed()
        forward = float(
            (keys[pygame.K_w] or keys[pygame.K_UP])
            - (keys[pygame.K_s] or keys[pygame.K_DOWN])
        )
        turn = float(
            (keys[pygame.K_a] or keys[pygame.K_LEFT])
            - (keys[pygame.K_d] or keys[pygame.K_RIGHT])
        )

        reset = False
        do_quit = False
        for ev in events:
            if ev.type == pygame.QUIT:
                do_quit = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    do_quit = True
                elif ev.key == pygame.K_r:
                    reset = True

        return TeleopInput(forward=forward, turn=turn, reset=reset, quit=do_quit)


# Confirm Protocol conformance at type-check time.
_: InputDevice = KeyboardInput()
