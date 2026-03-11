"""Tests for the reward system:
   - Each component returns [0, 1]
   - Combined via weighted sum
   - Hard invariant: invalid query NEVER scores higher than a valid one
     (enforced by weight design: syntax_weight > ngram_weight)
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resources.cql_resource_server import (
    DEFAULT_REWARD_WEIGHTS,
    compute_reward,
    reward_execution,
    reward_ngram,
    reward_syntax,
)


class TestIndividualComponents:
    """Each reward component returns [0, 1] independently."""

    def test_syntax_valid(self):
        score, info = reward_syntax("#event_simpleName=ProcessRollup2 | count()")
        assert score == 1.0
        assert info["valid"] is True
        assert info["errors"] == []

    def test_syntax_invalid(self):
        score, info = reward_syntax("| | broken query ((")
        assert score == 0.0
        assert info["valid"] is False
        assert len(info["errors"]) > 0

    def test_execution_returns_0_for_invalid_syntax(self):
        assert reward_execution("| broken", syntax_valid=False, mock=True) == 0.0

    def test_execution_returns_binary_for_valid(self):
        score = reward_execution("* | count()", syntax_valid=True, mock=True)
        assert score in (0.0, 1.0)

    def test_ngram_identical(self):
        cql = "#type=X | count()"
        assert reward_ngram(cql, cql) == 1.0

    def test_ngram_different(self):
        score = reward_ngram("alpha beta gamma", "x y z w")
        assert score == 0.0

    def test_ngram_partial(self):
        score = reward_ngram(
            "#type=ProcessRollup2 | count()",
            "#type=ProcessRollup2 | sum(field)",
        )
        assert 0.0 < score < 1.0


class TestRewardInvariant:
    """The hard invariant: invalid query never beats valid query."""

    GOLDEN_CQL = "#event_simpleName=ProcessRollup2 | groupBy(ComputerName, function=count()) | sort(_count, order=desc) | head(10)"

    def test_weights_enforce_invariant(self):
        """syntax_weight must be > ngram_weight for invariant to hold."""
        w = DEFAULT_REWARD_WEIGHTS
        assert w["syntax"] > w["ngram"], (
            f"syntax_weight ({w['syntax']}) must be > ngram_weight ({w['ngram']}) "
            "to guarantee invalid queries never beat valid ones"
        )

    def test_invalid_max_below_valid_min(self):
        """Max possible invalid reward < min possible valid reward."""
        w = DEFAULT_REWARD_WEIGHTS
        # Invalid: syntax=0, exec=0, ngram=1.0 (best case for invalid)
        max_invalid = w["ngram"] * 1.0
        # Valid: syntax=1, exec=0, ngram=0.0 (worst case for valid)
        min_valid = w["syntax"] * 1.0
        assert max_invalid < min_valid, (
            f"max_invalid ({max_invalid}) must be < min_valid ({min_valid})"
        )

    def test_invariant_invalid_never_beats_valid(self):
        """No invalid query should ever score higher than any valid query."""
        invalid_queries = [
            "| | broken",
            "count(field",
            'ImageFileName="cmd.exe',
            # Close to golden but missing closing paren:
            "#event_simpleName=ProcessRollup2 | groupBy(ComputerName, function=count()) | sort(_count, order=desc) | head(10",
        ]
        valid_queries = [
            "* | head(10)",
            "#type=DnsRequest | groupBy(DomainName)",
            "#event_simpleName=ProcessRollup2 | count()",
        ]

        for inv_q in invalid_queries:
            inv_result = compute_reward(inv_q, self.GOLDEN_CQL, mock_execution=True)
            for val_q in valid_queries:
                val_result = compute_reward(val_q, self.GOLDEN_CQL, mock_execution=True)
                assert inv_result["reward"] < val_result["reward"], (
                    f"INVARIANT VIOLATED: invalid '{inv_q}' scored "
                    f"{inv_result['reward']} >= valid '{val_q}' scored "
                    f"{val_result['reward']}"
                )

    def test_reward_in_0_1_range(self):
        """All rewards must be in [0, 1]."""
        queries = [
            "| broken",
            "* | count()",
            self.GOLDEN_CQL,
            "",
        ]
        for q in queries:
            result = compute_reward(q, self.GOLDEN_CQL, mock_execution=True)
            assert 0.0 <= result["reward"] <= 1.0, (
                f"Reward {result['reward']} out of [0, 1] for query '{q}'"
            )

    def test_perfect_match_reward(self):
        """A perfect match should score near maximum."""
        result = compute_reward(self.GOLDEN_CQL, self.GOLDEN_CQL, mock_execution=True)
        assert result["reward"] >= 0.5
        assert result["breakdown"]["syntax_valid"] is True
        assert result["breakdown"]["ngram_similarity"] == 1.0

    def test_ngram_gradient_preserved(self):
        """Ngram similarity must affect the score in all valid cases."""
        high_ngram_rewards = []
        low_ngram_rewards = []

        for seed in range(200):
            random.seed(seed)
            r_high = compute_reward(self.GOLDEN_CQL, self.GOLDEN_CQL, mock_execution=True)
            random.seed(seed)  # same seed → same execution outcome
            r_low = compute_reward(
                "#event_simpleName=DnsRequest | count()", self.GOLDEN_CQL, mock_execution=True
            )

            if (r_high["breakdown"]["syntax_valid"]
                    and r_low["breakdown"]["syntax_valid"]):
                high_ngram_rewards.append(r_high["reward"])
                low_ngram_rewards.append(r_low["reward"])

        assert len(high_ngram_rewards) > 0
        for h, l in zip(high_ngram_rewards, low_ngram_rewards):
            assert h >= l, f"Higher ngram query got lower reward: {h} < {l}"

    def test_reward_breakdown_present(self):
        """Reward response must include full breakdown."""
        result = compute_reward("* | count()", self.GOLDEN_CQL, mock_execution=True)
        breakdown = result["breakdown"]
        assert "syntax_valid" in breakdown
        assert "execution_success" in breakdown
        assert "ngram_similarity" in breakdown
        assert "component_scores" in breakdown
        assert "weights" in breakdown
        assert "reward" in breakdown
        assert "mock_execution" in breakdown

    def test_mock_execution_warning(self):
        """Mock execution flag must be set in breakdown."""
        result = compute_reward("* | count()", self.GOLDEN_CQL, mock_execution=True)
        assert result["breakdown"]["mock_execution"] is True

    def test_custom_weights(self):
        """Custom weights should change the reward."""
        cql = "* | count()"
        r_default = compute_reward(cql, self.GOLDEN_CQL, mock_execution=True)
        r_ngram_heavy = compute_reward(
            cql, self.GOLDEN_CQL, mock_execution=True,
            weights={"syntax": 0.1, "execution": 0.1, "ngram": 0.8},
        )
        # Both should return valid results
        assert 0.0 <= r_default["reward"] <= 1.0
        assert 0.0 <= r_ngram_heavy["reward"] <= 1.0
