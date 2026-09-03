from __future__ import annotations

from zws_nullity_lab.cycle import find_cycle
from zws_nullity_lab.initial import balanced_random, paired_impulse
from zws_nullity_lab.margolus import run_trajectory
from zws_nullity_lab.measures import stress_finite_windows
from zws_nullity_lab.rules import ReversibleRule


def test_identity_cycle_has_period_one_and_zero_transient() -> None:
    result = find_cycle(paired_impulse(4, 4), ReversibleRule.identity(), max_ticks=10)
    assert result.found
    assert result.transient == 0
    assert result.period == 1


def test_window_stress_returns_all_sequences() -> None:
    initial = balanced_random(1, 16, 16, seed=21)
    trajectory = run_trajectory(initial, ReversibleRule.reactive_rotation(12), 40)
    report = stress_finite_windows(trajectory, points=4)
    assert {row.sequence for row in report.measurements} == {
        "balanced",
        "time-first",
        "space-first",
        "width-first",
        "height-first",
        "shifted-balanced",
    }
    assert {row.candidate for row in report.convergence} == {
        "active_density",
        "conditional_entropy",
        "compression_ratio",
    }
