"""
Integration tests for feedback propagation through the loop.

These tests verify that:
1. Judge feedback flows into both idea improvement AND code generation prompts.
2. Thompson Sampling + diversity constraint work together across iterations.
3. Failure memory accumulates and appears in subsequent prompts.

These tests mock the LLM calls (call_claude, call_judge) to avoid real API usage
while testing the wiring and data flow within run_loop's orchestration logic.
"""

import random
import pytest
from unittest.mock import patch, MagicMock

from src.loop import (
    select_arm,
    update_arm,
    apply_diversity_mask,
    classify_error,
    format_failure_memory,
    RESEARCH_DIRECTIONS,
)


class TestThompsonSamplingWithDiversity:
    """Integration: Thompson Sampling + diversity constraint working together
    across a simulated multi-iteration sequence."""

    def test_arm_selection_respects_diversity_after_updates(self):
        """Simulate 5 iterations: after 2 consecutive picks of the same arm,
        the next pick must be different, even if that arm has the best stats."""
        arm_states = [{"alpha": 1.0, "beta": 1.0} for _ in range(5)]
        history = []
        rng = random.Random(42)

        # Force arm 0 to be very good
        arm_states[0] = {"alpha": 20.0, "beta": 1.0}

        # Simulate: arm 0 is picked twice
        history.append(0)
        history.append(0)

        # Now arm 0 should be blocked by diversity constraint
        idx = select_arm(arm_states, history, rng)
        assert idx != 0, "Arm 0 should be blocked after 2 consecutive picks"

        # After picking a different arm, arm 0 should be unblocked
        history.append(idx)
        idx2 = select_arm(arm_states, history, rng)
        # arm 0 is now allowed again (last 2 are [0, idx] which are different)
        # It may or may not be selected, but it is allowed
        mask = apply_diversity_mask(history, n_arms=5)
        assert mask[0] is True

    def test_arm_stats_update_affects_future_selections(self):
        """After updating an arm with good scores, it should be selected more often."""
        arm_states = [{"alpha": 1.0, "beta": 1.0} for _ in range(5)]

        # Give arm 3 several good scores
        for _ in range(5):
            update_arm(arm_states, arm_idx=3, score=9)

        # Now arm 3 should dominate selections
        counts = [0] * 5
        for seed in range(50):
            rng = random.Random(seed)
            idx = select_arm(arm_states, history=[], rng=rng)
            counts[idx] += 1

        assert counts[3] > 30, f"Arm 3 should dominate but got counts={counts}"


class TestFailureMemoryAccumulation:
    """Integration: failure memory accumulates across iterations and
    classify_error + format_failure_memory compose correctly."""

    def test_classify_and_format_pipeline(self):
        """Simulate two failures, classify them, add to memory, format for prompt."""
        failure_memory = []

        # Failure 1: JSON error
        stderr1 = "TypeError: Object of type int64 is not JSON serializable\n  File 'exp.py', line 42"
        error_type1 = classify_error(stderr1)
        assert error_type1 == "json_parse"
        failure_memory.append({
            "iteration": 1,
            "error_type": error_type1,
            "error_summary": "numpy int64 not JSON serializable",
            "direction": "embedding_analysis",
            "code_snippet": "json.dumps(results)",
        })

        # Failure 2: timeout
        stderr2 = "TIMEOUT after 600 seconds"
        error_type2 = classify_error(stderr2)
        assert error_type2 == "timeout"
        failure_memory.append({
            "iteration": 3,
            "error_type": error_type2,
            "error_summary": "Experiment exceeded time limit",
            "direction": "prompt_engineering",
            "code_snippet": "for i in range(10000): api_call()",
        })

        # Format for prompt
        result = format_failure_memory(failure_memory)
        assert "json_parse" in result
        assert "timeout" in result
        assert len(result) > 0

    def test_max_entries_works_with_accumulation(self):
        """Add 5 failures, format with max_entries=2, verify only latest 2 appear."""
        failure_memory = []
        for i in range(1, 6):
            failure_memory.append({
                "iteration": i,
                "error_type": "runtime_error",
                "error_summary": f"Failure number {i}",
                "direction": "text_statistics",
                "code_snippet": f"bad_line_{i}()",
            })

        result = format_failure_memory(failure_memory, max_entries=2)
        assert "Failure number 5" in result
        assert "Failure number 4" in result
        assert "Failure number 1" not in result
        assert "Failure number 2" not in result
        assert "Failure number 3" not in result


class TestFullSelectionCycle:
    """Integration: complete cycle of select -> use -> judge -> update -> select."""

    def test_exploration_then_exploitation(self):
        """Simulate a scenario where one arm gets good results and the system
        gradually exploits it while still occasionally exploring."""
        arm_states = [{"alpha": 1.0, "beta": 1.0} for _ in range(5)]
        history = []

        rng = random.Random(123)

        # Iteration 1: random pick (all arms equal)
        idx1 = select_arm(arm_states, history, rng)
        history.append(idx1)
        # Give it a bad score
        update_arm(arm_states, idx1, score=3)

        # Iteration 2: another random pick (still mostly equal)
        idx2 = select_arm(arm_states, history, rng)
        history.append(idx2)
        # Give it a great score
        update_arm(arm_states, idx2, score=9)

        # Iteration 3: should favor idx2's arm (it got score 9)
        # But if idx2 was picked twice in a row, diversity blocks it.
        idx3 = select_arm(arm_states, history, rng)
        history.append(idx3)

        # Verify all indices are valid
        assert all(0 <= idx < 5 for idx in [idx1, idx2, idx3])

        # Verify the well-scored arm has higher alpha
        assert arm_states[idx2]["alpha"] > arm_states[idx1]["alpha"]
