"""Tests for the R1-style reward system (utils/cql_rewards.py).

Tests:
  - Format reward scoring (0 / 0.5 / 1.0)
  - Structure reward (Jaccard of pipeline functions)
  - Field reward (F1 of entities)
  - CQL extraction from <think> responses
  - Combined reward in [0, 1]
  - GRPO-relevant: group variance is non-zero
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.cql_rewards import (
    compute_combined_reward,
    compute_field_reward,
    compute_format_reward,
    compute_structure_reward,
    extract_cql_from_response,
)


class TestFormatReward:
    """Format reward: 0.0 / 0.5 / 1.0 based on <think> tag presence."""

    def test_no_tags(self):
        assert compute_format_reward("just bare CQL") == 0.0

    def test_open_tag_only(self):
        assert compute_format_reward("<think>reasoning but no close") == 0.5

    def test_close_tag_only(self):
        assert compute_format_reward("some text</think>more") == 0.5

    def test_both_tags(self):
        assert compute_format_reward("<think>reasoning</think>\nCQL") == 1.0

    def test_empty_tags(self):
        assert compute_format_reward("<think></think>") == 1.0

    def test_empty_string(self):
        assert compute_format_reward("") == 0.0


class TestExtractCQL:
    """CQL extraction: separate thinking from answer."""

    def test_with_think_tags(self):
        cql, thinking = extract_cql_from_response("<think>my reasoning</think>\nSELECT 1")
        assert cql == "SELECT 1"
        assert thinking == "my reasoning"

    def test_no_tags(self):
        cql, thinking = extract_cql_from_response("bare CQL query")
        assert cql == "bare CQL query"
        assert thinking is None

    def test_empty_after_tags(self):
        cql, thinking = extract_cql_from_response("<think>only thinking</think>")
        assert cql == ""
        assert thinking == "only thinking"

    def test_empty_string(self):
        cql, thinking = extract_cql_from_response("")
        assert cql == ""
        assert thinking is None

    def test_multiline_thinking(self):
        resp = "<think>\nline1\nline2\n</think>\n#event=X | count()"
        cql, thinking = extract_cql_from_response(resp)
        assert cql == "#event=X | count()"
        assert "line1" in thinking


class TestStructureReward:
    """Structure reward: Jaccard of pipeline function names."""

    def test_identical_pipeline(self):
        q = "#type=X | where y>1 | groupBy(z) | head(10)"
        assert compute_structure_reward(q, q) == 1.0

    def test_completely_different_functions(self):
        a = "#type=X | where y>1"
        b = "#type=X | groupBy(z) | stats(count())"
        score = compute_structure_reward(a, b)
        assert score == 0.0  # no shared pipeline functions

    def test_partial_overlap(self):
        a = "#type=X | where y>1 | groupBy(z)"
        b = "#type=X | where y>1 | stats(count())"
        score = compute_structure_reward(a, b)
        assert 0.0 < score < 1.0  # share 'where' but differ on last stage

    def test_empty_both(self):
        assert compute_structure_reward("", "") == 1.0

    def test_empty_vs_nonempty(self):
        assert compute_structure_reward("", "#type=X | count()") == 0.0


class TestFieldReward:
    """Field reward: F1 of semantic entities (tags, field names, strings)."""

    def test_identical_query(self):
        q = '#event_simpleName=ProcessRollup2 | where FileName="cmd.exe"'
        assert compute_field_reward(q, q) == 1.0

    def test_completely_different(self):
        a = '#event_simpleName=ProcessRollup2 | where FileName="cmd.exe"'
        b = '#event_simpleName=DnsRequest | where DomainName="evil.com"'
        score = compute_field_reward(a, b)
        assert score < 1.0  # share #event_simpleName token but different values

    def test_partial_overlap(self):
        a = '#event_simpleName=ProcessRollup2 | where FileName="cmd.exe" | groupBy(ComputerName)'
        b = '#event_simpleName=ProcessRollup2 | where FileName="powershell.exe" | groupBy(ComputerName)'
        score = compute_field_reward(a, b)
        assert 0.0 < score < 1.0  # share event type and ComputerName, differ on filename

    def test_empty_both(self):
        assert compute_field_reward("", "") == 1.0

    def test_empty_vs_nonempty(self):
        assert compute_field_reward("", "#type=X | count()") == 0.0

    def test_right_fields_wrong_functions(self):
        """Same fields but different operations should still score high on fields."""
        a = "#event_simpleName=ProcessRollup2 | where FileName=X | groupBy(ComputerName)"
        b = "#event_simpleName=ProcessRollup2 | where FileName=X | stats(count()) by ComputerName"
        score = compute_field_reward(a, b)
        assert score > 0.5  # same entities, just different functions


class TestCombinedReward:
    """Combined reward: weighted sum, clamped [0, 1]."""

    GOLDEN = "#event_simpleName=ProcessRollup2 | groupBy(ComputerName, function=count()) | sort(_count, order=desc) | head(10)"

    def test_perfect_with_think(self):
        resp = f"<think>reasoning</think>\n{self.GOLDEN}"
        r = compute_combined_reward(resp, self.GOLDEN)
        assert r["reward"] == 1.0
        assert r["format"] == 1.0
        assert r["structure"] == 1.0
        assert r["fields"] == 1.0
        assert r["has_thinking"] is True

    def test_perfect_no_think(self):
        r = compute_combined_reward(self.GOLDEN, self.GOLDEN)
        # format=0 (no tags), structure=1.0, fields=1.0
        expected = 0.1 * 0.0 + 0.3 * 1.0 + 0.6 * 1.0
        assert r["reward"] == pytest.approx(expected)
        assert r["format"] == 0.0
        assert r["structure"] == 1.0
        assert r["fields"] == 1.0

    def test_empty_response(self):
        r = compute_combined_reward("", self.GOLDEN)
        assert r["reward"] == 0.0

    def test_think_only_no_cql(self):
        r = compute_combined_reward("<think>only thinking</think>", self.GOLDEN)
        # format=1.0, structure=0 (no functions), fields=0 (no entities)
        assert r["reward"] == pytest.approx(0.1)
        assert r["format"] == 1.0
        assert r["structure"] == 0.0

    def test_reward_range(self):
        """All rewards in [0, 1] for diverse inputs."""
        queries = [
            "",
            "garbage",
            self.GOLDEN,
            f"<think>x</think>\n{self.GOLDEN}",
            "<think>open only",
            "close only</think>",
            "<think></think>",
        ]
        for q in queries:
            r = compute_combined_reward(q, self.GOLDEN)
            assert 0.0 <= r["reward"] <= 1.0, f"Out of range for: {q}"

    def test_custom_weights(self):
        resp = f"<think>x</think>\n{self.GOLDEN}"
        r = compute_combined_reward(resp, self.GOLDEN, {"format": 0.5, "structure": 0.25, "fields": 0.25, "execution": 0.0})
        assert r["reward"] == 1.0  # all components are 1.0

    def test_think_tags_always_improve_reward(self):
        """Adding think tags to a correct response should increase reward."""
        r_bare = compute_combined_reward(self.GOLDEN, self.GOLDEN)
        r_think = compute_combined_reward(f"<think>reasoning</think>\n{self.GOLDEN}", self.GOLDEN)
        assert r_think["reward"] > r_bare["reward"]

    def test_breakdown_keys(self):
        r = compute_combined_reward("test", self.GOLDEN)
        assert set(r.keys()) == {"reward", "format", "structure", "fields", "execution", "extracted_cql", "has_thinking"}

    def test_right_structure_wrong_fields(self):
        """Right functions but wrong field names should score between 0 and 1."""
        wrong_fields = "#event_simpleName=DnsRequest | groupBy(DomainName, function=count()) | sort(_count, order=desc) | head(10)"
        r = compute_combined_reward(wrong_fields, self.GOLDEN)
        assert r["structure"] == 1.0  # same pipeline structure
        assert r["fields"] < 1.0     # different entities
        assert 0.0 < r["reward"] < 1.0

    def test_wrong_structure_right_fields(self):
        """Right field names but wrong operations should score between 0 and 1."""
        wrong_ops = "#event_simpleName=ProcessRollup2 | where ComputerName=X | count()"
        r = compute_combined_reward(wrong_ops, self.GOLDEN)
        assert r["structure"] < 1.0   # different pipeline functions
        assert r["fields"] > 0.0      # shares some entities
