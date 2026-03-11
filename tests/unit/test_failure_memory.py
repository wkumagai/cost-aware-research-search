"""
Unit tests for failure memory: classify_error and format_failure_memory.

These test the pure functions that classify execution errors and format them
for prompt injection. No external services required.
"""

import pytest

from src.loop import classify_error, format_failure_memory


class TestClassifyError:
    """Tests for classify_error: keyword-based stderr classification."""

    def test_timeout_error(self):
        stderr = "TIMEOUT after 600 seconds"
        assert classify_error(stderr) == "timeout"

    def test_json_serialization_error(self):
        stderr = "TypeError: Object of type bool_ is not JSON serializable"
        assert classify_error(stderr) == "json_parse"

    def test_json_with_serialize_keyword(self):
        stderr = "json.decoder.JSONDecodeError: Expecting value"
        assert classify_error(stderr) == "json_parse"

    def test_import_error(self):
        stderr = "ModuleNotFoundError: No module named 'transformers'"
        assert classify_error(stderr) == "import_error"

    def test_import_keyword(self):
        stderr = "ImportError: cannot import name 'foo' from 'bar'"
        assert classify_error(stderr) == "import_error"

    def test_memory_error(self):
        stderr = "MemoryError: Unable to allocate 4.00 GiB"
        assert classify_error(stderr) == "memory_error"

    def test_generic_runtime_error(self):
        stderr = "ValueError: invalid literal for int() with base 10: 'abc'"
        assert classify_error(stderr) == "runtime_error"

    def test_empty_stderr(self):
        assert classify_error("") == "runtime_error"

    def test_multiline_stderr_with_timeout_keyword(self):
        stderr = "Some other output\nwarning: blah\nTIMEOUT after 300 seconds"
        assert classify_error(stderr) == "timeout"


class TestFormatFailureMemory:
    """Tests for format_failure_memory: formats failure list for prompt injection."""

    def test_empty_memory_returns_empty_string(self):
        result = format_failure_memory([])
        assert result == ""

    def test_single_failure(self):
        memory = [
            {
                "iteration": 1,
                "error_type": "json_parse",
                "error_summary": "numpy bool not serializable",
                "direction": "embedding_analysis",
                "code_snippet": "json.dumps(results)",
            }
        ]
        result = format_failure_memory(memory)
        assert "json_parse" in result
        assert "numpy bool not serializable" in result
        assert "Iter 1" in result or "iteration 1" in result.lower() or "1" in result

    def test_max_entries_limits_output(self):
        memory = [
            {
                "iteration": i,
                "error_type": "runtime_error",
                "error_summary": f"Error {i}",
                "direction": "prompt_engineering",
                "code_snippet": f"line {i}",
            }
            for i in range(1, 6)
        ]
        result = format_failure_memory(memory, max_entries=3)
        # Should contain the 3 most recent (iterations 3, 4, 5)
        # Should NOT contain iterations 1 and 2
        assert "Error 5" in result
        assert "Error 4" in result
        assert "Error 3" in result
        # Older entries should be excluded
        lines_with_error1 = [line for line in result.split("\n") if "Error 1" in line]
        assert len(lines_with_error1) == 0

    def test_returns_string(self):
        memory = [
            {
                "iteration": 1,
                "error_type": "timeout",
                "error_summary": "Took too long",
                "direction": "model_comparison",
                "code_snippet": "time.sleep(999)",
            }
        ]
        result = format_failure_memory(memory)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_default_max_entries_is_3(self):
        """Verify the default max_entries parameter is 3."""
        memory = [
            {
                "iteration": i,
                "error_type": "runtime_error",
                "error_summary": f"Error {i}",
                "direction": "prompt_engineering",
                "code_snippet": f"line {i}",
            }
            for i in range(1, 10)
        ]
        result = format_failure_memory(memory)
        # Should show at most 3 entries. Count bullet points or entry markers.
        # The exact format is up to the Coder, but there should be at most 3 entries.
        # We check that early failures are not included.
        assert "Error 1" not in result
        assert "Error 9" in result
