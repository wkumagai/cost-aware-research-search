"""
Unit tests for Stage 0 feasibility check, including the new sample size enforcement.

Tests the stage0_check function which validates idea specs before code generation.
"""

import pytest

from src.loop import stage0_check, MIN_SAMPLES


# A valid idea spec that should pass all checks
VALID_SPEC = """
hypothesis: "Adding structured output format constraints improves consistency"
intervention_family: prompt_engineering
target_component: "GPT-4o output formatting"
implementation_scope:
  lines_estimate: 120
  dependencies: [openai, numpy]
  compute: cpu
  time_estimate_min: 5
proxy_evaluation:
  metric: consistency_score
  baseline: unstructured_output
  success_threshold: ">10% improvement"
full_evaluation:
  dataset: synthetic_prompts
  n_samples: 100
  n_conditions: 3
expected_failure_modes:
  - API rate limiting
  - Inconsistent model outputs
rollback_plan: "Reduce sample size to minimum viable"
"""

# Spec with too few samples
SMALL_SAMPLES_SPEC = """
hypothesis: "Test hypothesis"
intervention_family: prompt_engineering
target_component: "some component"
implementation_scope:
  lines_estimate: 100
  dependencies: [openai]
  compute: cpu
  time_estimate_min: 5
proxy_evaluation:
  metric: accuracy
  baseline: random
  success_threshold: ">5%"
full_evaluation:
  dataset: test
  n_samples: 20
  n_conditions: 2
expected_failure_modes:
  - something
rollback_plan: "simplify"
"""


class TestStage0Check:
    """Tests for stage0_check with sample size enforcement."""

    def test_valid_spec_passes(self):
        result = stage0_check(VALID_SPEC)
        assert result["passed"] is True
        assert result["valid_yaml"] is True
        assert result["has_hypothesis"] is True
        assert result["has_metric"] is True
        assert result["cpu_only"] is True
        assert result["time_ok"] is True

    def test_insufficient_samples_fails(self):
        result = stage0_check(SMALL_SAMPLES_SPEC)
        assert result["sufficient_samples"] is False
        assert result["passed"] is False

    def test_valid_spec_has_sufficient_samples(self):
        result = stage0_check(VALID_SPEC)
        assert result["sufficient_samples"] is True

    def test_missing_n_samples_fails(self):
        spec = """
hypothesis: "Test"
intervention_family: test
target_component: "test"
implementation_scope:
  lines_estimate: 100
  dependencies: []
  compute: cpu
  time_estimate_min: 3
proxy_evaluation:
  metric: acc
  baseline: random
  success_threshold: ">0"
full_evaluation:
  dataset: test
expected_failure_modes: []
rollback_plan: "none"
"""
        result = stage0_check(spec)
        assert result["sufficient_samples"] is False

    def test_invalid_yaml_fails(self):
        result = stage0_check("this: is: not: valid: yaml: {{{}}")
        assert result["valid_yaml"] is False
        assert result["passed"] is False

    def test_missing_hypothesis_fails(self):
        spec = """
intervention_family: test
target_component: "test"
implementation_scope:
  compute: cpu
  time_estimate_min: 3
proxy_evaluation:
  metric: acc
full_evaluation:
  n_samples: 100
"""
        result = stage0_check(spec)
        assert result["has_hypothesis"] is False
        assert result["passed"] is False

    def test_gpu_compute_fails(self):
        spec = """
hypothesis: "Test"
intervention_family: test
target_component: "test"
implementation_scope:
  compute: gpu
  time_estimate_min: 3
proxy_evaluation:
  metric: acc
full_evaluation:
  n_samples: 100
"""
        result = stage0_check(spec)
        assert result["cpu_only"] is False

    def test_time_over_15_fails(self):
        spec = """
hypothesis: "Test"
intervention_family: test
target_component: "test"
implementation_scope:
  compute: cpu
  time_estimate_min: 20
proxy_evaluation:
  metric: acc
full_evaluation:
  n_samples: 100
"""
        result = stage0_check(spec)
        assert result["time_ok"] is False


class TestMinSamplesConstant:
    """Verify MIN_SAMPLES is defined and has a reasonable value."""

    def test_min_samples_exists(self):
        assert MIN_SAMPLES is not None

    def test_min_samples_is_at_least_50(self):
        assert MIN_SAMPLES >= 50
