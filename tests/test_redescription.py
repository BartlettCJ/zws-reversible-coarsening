from __future__ import annotations

from zws_nullity_lab.initial import balanced_random
from zws_nullity_lab.redescription import audit_state_relabellings
from zws_nullity_lab.rules import ReversibleRule


def test_declared_redescriptions_are_exact_conjugacies() -> None:
    state = balanced_random(1, 16, 16, seed=17)
    audit = audit_state_relabellings(state, ReversibleRule.reactive_rotation(16), steps=32)
    assert audit.passed
    assert all(case.exact_trace_equivariance for case in audit.cases)
    assert all(case.max_state_entropy_error < 2e-7 for case in audit.cases)
    assert all(case.max_block_entropy_error < 2e-7 for case in audit.cases)
    assert all(case.max_spatial_mi_error < 1e-7 for case in audit.cases)
    assert all(case.active_density_error == 0.0 for case in audit.cases)
    assert all(case.conditional_entropy_error < 1e-12 for case in audit.cases)
    assert all(case.compression_ratio_error >= 0.0 for case in audit.cases)
