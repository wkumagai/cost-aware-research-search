"""
Unit tests for Thompson Sampling arm selection and update.

These test the pure functions: select_arm, update_arm, apply_diversity_mask.
All tests are deterministic (using seeded RNG) and require no external services.
"""

import random
import pytest


# ---------------------------------------------------------------------------
# These imports will fail until the Coder implements the functions in src/loop.
# That is intentional (RED phase of TDD).
# ---------------------------------------------------------------------------
from src.loop import (
    select_arm,
    update_arm,
    apply_diversity_mask,
    RESEARCH_DIRECTIONS,
)


class TestApplyDiversityMask:
    """Tests for apply_diversity_mask: prevents the same arm from being
    selected more than max_consecutive times in a row."""

    def test_empty_history_allows_all(self):
        mask = apply_diversity_mask(history=[], n_arms=5, max_consecutive=2)
        assert mask == [True, True, True, True, True]

    def test_single_entry_allows_all(self):
        mask = apply_diversity_mask(history=[0], n_arms=5, max_consecutive=2)
        assert mask == [True, True, True, True, True]

    def test_two_consecutive_same_blocks_that_arm(self):
        mask = apply_diversity_mask(history=[2, 2], n_arms=5, max_consecutive=2)
        assert mask[2] is False
        # Other arms remain allowed
        assert mask[0] is True
        assert mask[1] is True
        assert mask[3] is True
        assert mask[4] is True

    def test_two_different_allows_all(self):
        mask = apply_diversity_mask(history=[1, 3], n_arms=5, max_consecutive=2)
        assert all(mask)

    def test_longer_history_only_checks_tail(self):
        # History: [0, 0, 1, 1] -- last 2 are arm 1
        mask = apply_diversity_mask(history=[0, 0, 1, 1], n_arms=5, max_consecutive=2)
        assert mask[1] is False
        assert mask[0] is True  # arm 0 was consecutive earlier, but not at the tail

    def test_three_consecutive_with_max3(self):
        mask = apply_diversity_mask(history=[4, 4, 4], n_arms=5, max_consecutive=3)
        assert mask[4] is False

    def test_two_consecutive_with_max3_allows(self):
        mask = apply_diversity_mask(history=[4, 4], n_arms=5, max_consecutive=3)
        assert all(mask)

    def test_all_blocked_fallback(self):
        """Edge case: if n_arms=1 and it would be blocked, unblock all."""
        mask = apply_diversity_mask(history=[0, 0], n_arms=1, max_consecutive=2)
        assert mask == [True]


class TestUpdateArm:
    """Tests for update_arm: converts a 1-10 score to a Beta distribution update."""

    def test_high_score_increases_alpha(self):
        states = [{"alpha": 1.0, "beta": 1.0}]
        update_arm(states, arm_idx=0, score=10)
        assert states[0]["alpha"] > 1.0
        assert states[0]["beta"] > 1.0  # beta also increases slightly (1-reward small)
        # For score=10: reward = (10-2)/8 = 1.0, so alpha += 1.0, beta += 0.0
        assert states[0]["alpha"] == pytest.approx(2.0)
        assert states[0]["beta"] == pytest.approx(1.0)

    def test_low_score_increases_beta(self):
        states = [{"alpha": 1.0, "beta": 1.0}]
        update_arm(states, arm_idx=0, score=2)
        # reward = (2-2)/8 = 0.0, so alpha += 0.0, beta += 1.0
        assert states[0]["alpha"] == pytest.approx(1.0)
        assert states[0]["beta"] == pytest.approx(2.0)

    def test_mid_score_updates_both(self):
        states = [{"alpha": 1.0, "beta": 1.0}]
        update_arm(states, arm_idx=0, score=6)
        # reward = (6-2)/8 = 0.5
        assert states[0]["alpha"] == pytest.approx(1.5)
        assert states[0]["beta"] == pytest.approx(1.5)

    def test_score_below_2_clamps_to_zero_reward(self):
        states = [{"alpha": 1.0, "beta": 1.0}]
        update_arm(states, arm_idx=0, score=1)
        # reward = max(0, (1-2)/8) = 0.0
        assert states[0]["alpha"] == pytest.approx(1.0)
        assert states[0]["beta"] == pytest.approx(2.0)

    def test_mutates_correct_arm(self):
        states = [
            {"alpha": 1.0, "beta": 1.0},
            {"alpha": 1.0, "beta": 1.0},
            {"alpha": 1.0, "beta": 1.0},
        ]
        update_arm(states, arm_idx=1, score=8)
        # Only arm 1 should change
        assert states[0] == {"alpha": 1.0, "beta": 1.0}
        assert states[1]["alpha"] > 1.0
        assert states[2] == {"alpha": 1.0, "beta": 1.0}


class TestSelectArm:
    """Tests for select_arm: Thompson Sampling with diversity constraint."""

    def test_returns_valid_index(self):
        states = [{"alpha": 1.0, "beta": 1.0} for _ in range(5)]
        rng = random.Random(42)
        idx = select_arm(states, history=[], rng=rng)
        assert 0 <= idx < 5

    def test_favors_high_alpha_arm(self):
        """An arm with much higher alpha should be selected most of the time."""
        states = [
            {"alpha": 1.0, "beta": 10.0},   # very bad
            {"alpha": 1.0, "beta": 10.0},   # very bad
            {"alpha": 50.0, "beta": 1.0},   # very good
            {"alpha": 1.0, "beta": 10.0},   # very bad
            {"alpha": 1.0, "beta": 10.0},   # very bad
        ]
        counts = [0] * 5
        for seed in range(100):
            rng = random.Random(seed)
            idx = select_arm(states, history=[], rng=rng)
            counts[idx] += 1
        # Arm 2 should be selected the vast majority of the time
        assert counts[2] > 80

    def test_diversity_blocks_consecutive_arm(self):
        """If an arm was selected 2x in a row, it should not be selected again
        even if it has the best stats."""
        states = [
            {"alpha": 1.0, "beta": 10.0},
            {"alpha": 1.0, "beta": 10.0},
            {"alpha": 100.0, "beta": 1.0},  # dominant arm
            {"alpha": 1.0, "beta": 10.0},
            {"alpha": 1.0, "beta": 10.0},
        ]
        # Arm 2 was selected last 2 times -- should be blocked
        history = [2, 2]
        for seed in range(20):
            rng = random.Random(seed)
            idx = select_arm(states, history=history, rng=rng)
            assert idx != 2, f"Arm 2 should be blocked but was selected with seed={seed}"

    def test_returns_int(self):
        states = [{"alpha": 1.0, "beta": 1.0} for _ in range(3)]
        rng = random.Random(0)
        idx = select_arm(states, history=[], rng=rng)
        assert isinstance(idx, int)


class TestResearchDirections:
    """Basic sanity checks on the RESEARCH_DIRECTIONS constant."""

    def test_has_at_least_3_directions(self):
        assert len(RESEARCH_DIRECTIONS) >= 3

    def test_each_direction_has_name_and_description(self):
        for d in RESEARCH_DIRECTIONS:
            assert "name" in d
            assert "description" in d
            assert isinstance(d["name"], str)
            assert len(d["name"]) > 0
            assert isinstance(d["description"], str)
            assert len(d["description"]) > 0
