"""Tests for the R1-style reward system (utils/cql_rewards.py).

Tests:
  - Format reward scoring (0 / 0.5 / 1.0)
  - N-gram reward [0, 1] range
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
    compute_format_reward,
    compute_ngram_reward,
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


class TestNgramReward:
    """N-gram reward: bigram Dice coefficient."""

    def test_identical(self):
        q = "#type=X | count()"
        assert compute_ngram_reward(q, q) == 1.0

    def test_completely_different(self):
        assert compute_ngram_reward("alpha beta gamma", "x y z w") == 0.0

    def test_partial_overlap(self):
        score = compute_ngram_reward(
            "#type=ProcessRollup2 | count()",
            "#type=ProcessRollup2 | sum(field)",
        )
        assert 0.0 < score < 1.0

    def test_empty_both(self):
        # Two identical (empty) strings have perfect similarity
        assert compute_ngram_reward("", "") == 1.0

    def test_empty_one(self):
        assert compute_ngram_reward("", "#type=X | count()") == 0.0
        assert compute_ngram_reward("#type=X | count()", "") == 0.0


class TestCombinedReward:
    """Combined reward: weighted sum, clamped [0, 1]."""

    GOLDEN = "#event_simpleName=ProcessRollup2 | groupBy(ComputerName, function=count()) | sort(_count, order=desc) | head(10)"

    def test_perfect_with_think(self):
        resp = f"<think>reasoning</think>\n{self.GOLDEN}"
        r = compute_combined_reward(resp, self.GOLDEN)
        assert r["reward"] == 1.0
        assert r["format"] == 1.0
        assert r["ngram"] == 1.0
        assert r["has_thinking"] is True

    def test_perfect_no_think(self):
        r = compute_combined_reward(self.GOLDEN, self.GOLDEN)
        assert r["reward"] == pytest.approx(0.8)
        assert r["format"] == 0.0
        assert r["ngram"] == 1.0

    def test_empty_response(self):
        r = compute_combined_reward("", self.GOLDEN)
        assert r["reward"] == 0.0

    def test_think_only_no_cql(self):
        r = compute_combined_reward("<think>only thinking</think>", self.GOLDEN)
        assert r["reward"] == pytest.approx(0.2)
        assert r["format"] == 1.0
        assert r["ngram"] == 0.0

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
        r1 = compute_combined_reward(resp, self.GOLDEN, {"format": 0.5, "ngram": 0.5, "execution": 0.0})
        r2 = compute_combined_reward(resp, self.GOLDEN, {"format": 0.0, "ngram": 1.0, "execution": 0.0})
        assert r1["reward"] == 1.0  # 0.5*1.0 + 0.5*1.0
        assert r2["reward"] == 1.0  # 0.0*1.0 + 1.0*1.0

    def test_think_tags_always_improve_reward(self):
        """Adding think tags to a correct response should increase reward."""
        r_bare = compute_combined_reward(self.GOLDEN, self.GOLDEN)
        r_think = compute_combined_reward(f"<think>reasoning</think>\n{self.GOLDEN}", self.GOLDEN)
        assert r_think["reward"] > r_bare["reward"]

    def test_breakdown_keys(self):
        r = compute_combined_reward("test", self.GOLDEN)
        assert set(r.keys()) == {"reward", "format", "ngram", "execution", "extracted_cql", "has_thinking"}
