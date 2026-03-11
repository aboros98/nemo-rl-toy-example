#!/usr/bin/env python3
"""Test reward logic locally on Mac — no GPU, no NeMo RL, no Ray needed.

Usage:
    python3 scripts/test_rewards_local.py
    python3 scripts/test_rewards_local.py --data data/train.jsonl --n 20
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cql_validator import validate
from utils.cql_tokenizer import bigram_similarity


def compute_reward(response: str, ground_truth: str) -> dict:
    """Same logic the CQL environment uses — extend this to test new rewards."""
    result = validate(response)
    sim = bigram_similarity(response, ground_truth)

    if not result.is_valid:
        reward = max(-0.5 + 0.1 * sim, -1.0)
        category = "INVALID"
    else:
        reward = 0.4 + 0.6 * sim
        category = "VALID"

    return {
        "reward": round(reward, 3),
        "category": category,
        "is_valid": result.is_valid,
        "errors": result.errors,
        "bigram_sim": round(sim, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "train.jsonl"))
    parser.add_argument("--n", type=int, default=10, help="Number of examples to test")
    args = parser.parse_args()

    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= args.n:
                break

    print(f"Testing reward on {len(examples)} examples from {args.data}\n")
    print(f"{'#':>3}  {'Reward':>7}  {'Valid':>5}  {'BiSim':>5}  Query (first 60 chars)")
    print("-" * 90)

    rewards = []
    for i, ex in enumerate(examples):
        gt = ex["cql_query"]
        # Test 1: reward for the ground truth itself (should be high)
        r = compute_reward(gt, gt)
        rewards.append(r["reward"])
        print(f"{i+1:3d}  {r['reward']:7.3f}  {str(r['is_valid']):>5}  {r['bigram_sim']:5.3f}  {gt[:60].replace(chr(10), ' | ')}")

    print("-" * 90)
    print(f"Mean reward (ground truth vs itself): {sum(rewards)/len(rewards):.3f}")
    print()

    # Test with deliberately broken queries
    print("=== Broken query tests ===")
    broken = [
        ("Empty string", "", examples[0]["cql_query"]),
        ("Just text", "show me all events", examples[0]["cql_query"]),
        ("Unclosed paren", "search(foo", examples[0]["cql_query"]),
        ("Unclosed string", 'where name="hello', examples[0]["cql_query"]),
        ("Almost correct", examples[0]["cql_query"].replace("|", "||"), examples[0]["cql_query"]),
    ]
    print(f"\n{'Test':<25}  {'Reward':>7}  {'Valid':>5}  {'BiSim':>5}  Errors")
    print("-" * 90)
    for name, query, gt in broken:
        r = compute_reward(query, gt)
        errs = "; ".join(r["errors"][:2]) if r["errors"] else "-"
        print(f"{name:<25}  {r['reward']:7.3f}  {str(r['is_valid']):>5}  {r['bigram_sim']:5.3f}  {errs[:40]}")

    # Invariant check
    print("\n=== Invariant check: invalid must always score < valid ===")
    valid_r = compute_reward(examples[0]["cql_query"], examples[0]["cql_query"])
    invalid_r = compute_reward("search(foo", examples[0]["cql_query"])
    ok = invalid_r["reward"] < valid_r["reward"]
    print(f"  Valid reward:   {valid_r['reward']}")
    print(f"  Invalid reward: {invalid_r['reward']}")
    print(f"  Invariant holds: {'✓ YES' if ok else '✗ NO — BUG!'}")


if __name__ == "__main__":
    main()
