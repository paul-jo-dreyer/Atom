"""Tests for the snapshot pool data structure (training/snapshot_pool.py).

Pool tests use plain dicts of numpy arrays as fake state_dicts — the
pool doesn't care what's inside, only that deepcopy works correctly."""

from __future__ import annotations

import numpy as np
import pytest

from AtomGym.training.snapshot_pool import Snapshot, SnapshotPool


# ---------------------------------------------------------------------------
# Construction & invariants
# ---------------------------------------------------------------------------


def test_pool_init_empty() -> None:
    p = SnapshotPool(capacity=5)
    assert len(p) == 0
    assert p.capacity == 5


def test_pool_default_capacity() -> None:
    p = SnapshotPool()
    assert p.capacity == 20


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SnapshotPool(capacity=0)
    with pytest.raises(ValueError):
        SnapshotPool(capacity=-3)


# ---------------------------------------------------------------------------
# Add / eviction
# ---------------------------------------------------------------------------


def test_add_increments_length() -> None:
    p = SnapshotPool(capacity=5)
    p.add({"w": np.array([1.0])}, iteration=100)
    assert len(p) == 1
    p.add({"w": np.array([2.0])}, iteration=200)
    assert len(p) == 2


def test_add_returns_stored_snapshot() -> None:
    p = SnapshotPool(capacity=5)
    snap = p.add({"w": np.array([1.0])}, iteration=100)
    assert isinstance(snap, Snapshot)
    assert snap.iteration == 100


def test_add_evicts_oldest_at_capacity() -> None:
    p = SnapshotPool(capacity=3)
    for i in range(5):
        p.add({"w": np.array([float(i)])}, iteration=i * 100)
    assert len(p) == 3
    iterations = [s.iteration for s in p]
    # First two (iter=0, 100) should have been evicted.
    assert iterations == [200, 300, 400]


def test_default_elo_and_zero_counters() -> None:
    p = SnapshotPool()
    snap = p.add({}, iteration=100)
    assert snap.elo == 1000.0
    assert snap.episodes_played == 0
    assert snap.wins == 0


def test_explicit_elo() -> None:
    p = SnapshotPool()
    snap = p.add({}, iteration=100, elo=1234.5)
    assert snap.elo == 1234.5


def test_add_deepcopies_state_dict() -> None:
    """Mutating the original after add() must not affect the stored snapshot."""
    p = SnapshotPool(capacity=5)
    sd = {"w": np.array([1.0, 2.0, 3.0])}
    p.add(sd, iteration=100)
    sd["w"][0] = 999.0
    sd["new_key"] = "added"
    stored = p.latest().state_dict
    assert stored["w"][0] == 1.0
    assert "new_key" not in stored


# ---------------------------------------------------------------------------
# latest() / iter()
# ---------------------------------------------------------------------------


def test_latest_returns_most_recent() -> None:
    p = SnapshotPool(capacity=5)
    p.add({}, iteration=10)
    p.add({}, iteration=20)
    p.add({}, iteration=30)
    assert p.latest().iteration == 30


def test_latest_on_empty_raises() -> None:
    p = SnapshotPool()
    with pytest.raises(IndexError, match="empty"):
        p.latest()


def test_iter_yields_oldest_first() -> None:
    p = SnapshotPool(capacity=5)
    iters = [10, 20, 30, 40]
    for i in iters:
        p.add({}, iteration=i)
    yielded = [s.iteration for s in p]
    assert yielded == iters


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------


def test_sample_returns_a_pool_entry() -> None:
    p = SnapshotPool(capacity=5)
    for i in range(3):
        p.add({}, iteration=i)
    rng = np.random.default_rng(0)
    s = p.sample(rng)
    assert s.iteration in (0, 1, 2)


def test_sample_on_empty_raises() -> None:
    p = SnapshotPool()
    with pytest.raises(IndexError, match="empty"):
        p.sample(np.random.default_rng(0))


def test_sample_is_approximately_uniform() -> None:
    """Over many draws each entry should be picked roughly equally often."""
    p = SnapshotPool(capacity=5)
    for i in range(5):
        p.add({}, iteration=i)
    rng = np.random.default_rng(0)
    counts = [0] * 5
    n_draws = 5000
    for _ in range(n_draws):
        s = p.sample(rng)
        counts[s.iteration] += 1
    expected = n_draws / 5
    # Loose bound — within 20% of expected for any single bucket.
    for c in counts:
        assert abs(c - expected) < 0.2 * expected, f"counts={counts}"


def test_sample_reproducible_with_seeded_rng() -> None:
    """Same rng seed → same sample sequence."""
    p = SnapshotPool(capacity=5)
    for i in range(5):
        p.add({}, iteration=i)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    seq_a = [p.sample(rng_a).iteration for _ in range(20)]
    seq_b = [p.sample(rng_b).iteration for _ in range(20)]
    assert seq_a == seq_b


# ---------------------------------------------------------------------------
# record_outcome()
# ---------------------------------------------------------------------------


def test_record_outcome_increments_episodes_and_wins() -> None:
    p = SnapshotPool()
    p.add({}, iteration=100)
    p.add({}, iteration=200)

    # Learner LOST against snapshot 100 → snapshot wins+=1, episodes+=1.
    found = p.record_outcome(iteration=100, learner_won=False)
    assert found is True
    snap = next(s for s in p if s.iteration == 100)
    assert snap.episodes_played == 1
    assert snap.wins == 1

    # Learner WON against snapshot 100 → episodes+=1, wins unchanged.
    p.record_outcome(iteration=100, learner_won=True)
    snap = next(s for s in p if s.iteration == 100)
    assert snap.episodes_played == 2
    assert snap.wins == 1


def test_record_outcome_does_not_affect_other_entries() -> None:
    p = SnapshotPool()
    p.add({}, iteration=100)
    p.add({}, iteration=200)
    p.record_outcome(iteration=100, learner_won=False)
    snap_200 = next(s for s in p if s.iteration == 200)
    assert snap_200.episodes_played == 0
    assert snap_200.wins == 0


def test_record_outcome_for_evicted_returns_false() -> None:
    p = SnapshotPool(capacity=2)
    p.add({}, iteration=10)
    p.add({}, iteration=20)
    p.add({}, iteration=30)  # evicts iter=10
    found = p.record_outcome(iteration=10, learner_won=False)
    assert found is False


# ---------------------------------------------------------------------------
# Snapshot is immutable from outside
# ---------------------------------------------------------------------------


def test_snapshot_is_namedtuple() -> None:
    """NamedTuple should not allow field assignment."""
    snap = Snapshot(state_dict={}, iteration=1, elo=1000.0, episodes_played=0, wins=0)
    with pytest.raises(AttributeError):
        snap.iteration = 999  # type: ignore[misc]


def test_handle_returned_to_caller_is_stale_after_record() -> None:
    """`record_outcome` rewrites the pool's INTERNAL entry. A handle the
    caller obtained earlier still points to the old (immutable) record —
    documented in the module docstring. This test pins that behaviour so
    it doesn't change silently."""
    p = SnapshotPool()
    p.add({}, iteration=100)
    handle = p.latest()
    p.record_outcome(iteration=100, learner_won=False)
    # External handle is unchanged (NamedTuples are immutable).
    assert handle.episodes_played == 0
    assert handle.wins == 0
    # But the pool's internal entry has been updated.
    assert p.latest().episodes_played == 1
    assert p.latest().wins == 1
