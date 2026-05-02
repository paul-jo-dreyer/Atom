"""Tests for the chess-style sliding-window win-rate tracker."""

from __future__ import annotations

import pytest

from AtomGym.training.win_rate_tracker import Outcome, WinRateTracker


# ---------------------------------------------------------------------------
# Construction & invariants
# ---------------------------------------------------------------------------


def test_init_empty() -> None:
    t = WinRateTracker(window_size=10)
    assert len(t) == 0
    assert t.window_size == 10
    assert t.is_full is False


def test_window_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        WinRateTracker(window_size=0)
    with pytest.raises(ValueError):
        WinRateTracker(window_size=-1)


# ---------------------------------------------------------------------------
# Recording & is_full
# ---------------------------------------------------------------------------


def test_record_grows_until_full() -> None:
    t = WinRateTracker(window_size=3)
    assert t.is_full is False
    t.record(Outcome.WIN)
    assert len(t) == 1 and not t.is_full
    t.record(Outcome.LOSS)
    assert len(t) == 2 and not t.is_full
    t.record(Outcome.DRAW)
    assert len(t) == 3 and t.is_full


def test_record_evicts_oldest_when_full() -> None:
    """Once full, each new record drops the oldest. Length stays at K."""
    t = WinRateTracker(window_size=3)
    t.record(Outcome.WIN)
    t.record(Outcome.WIN)
    t.record(Outcome.WIN)
    # 3 wins, rate=1.0
    assert t.win_rate == 1.0
    # Push two losses; the two oldest wins should be evicted.
    t.record(Outcome.LOSS)
    t.record(Outcome.LOSS)
    assert len(t) == 3
    # Now: 1 win, 2 losses → rate = 1/3
    assert t.win_rate == pytest.approx(1.0 / 3.0)


def test_record_rejects_non_outcome() -> None:
    t = WinRateTracker()
    with pytest.raises(TypeError):
        t.record(1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        t.record("win")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------


def test_win_rate_undefined_until_full() -> None:
    t = WinRateTracker(window_size=5)
    with pytest.raises(RuntimeError, match="window"):
        _ = t.win_rate
    for _ in range(4):
        t.record(Outcome.WIN)
    with pytest.raises(RuntimeError, match="window"):
        _ = t.win_rate
    t.record(Outcome.WIN)  # now full
    assert t.win_rate == 1.0


def test_chess_style_scoring() -> None:
    """W=1.0, L=0.0, T=0.5 — mixed window."""
    t = WinRateTracker(window_size=4)
    t.record(Outcome.WIN)   # 1.0
    t.record(Outcome.WIN)   # 1.0
    t.record(Outcome.DRAW)  # 0.5
    t.record(Outcome.LOSS)  # 0.0
    # (1 + 1 + 0.5 + 0) / 4 = 0.625
    assert t.win_rate == pytest.approx(0.625)


def test_all_draws_gives_half() -> None:
    t = WinRateTracker(window_size=4)
    for _ in range(4):
        t.record(Outcome.DRAW)
    assert t.win_rate == 0.5


def test_all_losses_gives_zero() -> None:
    t = WinRateTracker(window_size=3)
    for _ in range(3):
        t.record(Outcome.LOSS)
    assert t.win_rate == 0.0


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_empties_window() -> None:
    t = WinRateTracker(window_size=3)
    for _ in range(3):
        t.record(Outcome.WIN)
    assert t.is_full
    t.reset()
    assert len(t) == 0
    assert t.is_full is False
    with pytest.raises(RuntimeError):
        _ = t.win_rate


def test_reset_then_record_starts_fresh() -> None:
    t = WinRateTracker(window_size=2)
    t.record(Outcome.WIN)
    t.record(Outcome.WIN)
    t.reset()
    t.record(Outcome.LOSS)
    t.record(Outcome.LOSS)
    assert t.win_rate == 0.0  # no carry-over from pre-reset wins
