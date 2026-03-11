#!/usr/bin/env python3
"""Test reward logic locally on Mac — no GPU, no NeMo RL, no Ray needed.

Uses the EXACT same reward functions as the real CQL environment.
Edit compute_combined_reward() in environments/cql_environment.py to change rewards —
this script imports from there, so local tests always match production.

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

from utils.cql_rewards import (
    compute_combined_reward,
    compute_format_reward,
    extract_cql_from_response,
)

DEFAULT_WEIGHTS = {"format": 0.2, "ngram": 0.8, "execution": 0.0}


def main():
    parser = argparse.ArgumentParser(description="Test CQL rewards locally")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "train.jsonl"))
    parser.add_argument("--n", type=int, default=10, help="Number of examples")
    parser.add_argument("--weights", type=str, default=None,
                        help='JSON weights, e.g. \'{"format":0.2,"ngram":0.8,"execution":0.0}\'')
    args = parser.parse_args()

    weights = json.loads(args.weights) if args.weights else DEFAULT_WEIGHTS
    print(f"Reward weights: format={weights['format']}, ngram={weights['ngram']}, execution={weights['execution']}\n")

    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= args.n:
                break

    # --- Test 1: ground truth (bare CQL, no think tags) ---
    print(f"=== Ground truth queries (no <think> tags) — {len(examples)} examples ===")
    print(f"{'#':>3}  {'Total':>6}  {'Fmt':>5}  {'Ngram':>5}  Query (first 60 chars)")
    print("-" * 90)
    totals = []
    for i, ex in enumerate(examples):
        gt = ex["cql_query"]
        r = compute_combined_reward(gt, gt, weights)
        totals.append(r["reward"])
        print(f"{i+1:3d}  {r['reward']:6.3f}  {r['format']:5.2f}  {r['ngram']:5.2f}  {gt[:60].replace(chr(10), ' | ')}")
    print(f"Mean: {sum(totals)/len(totals):.3f}  (format=0.0 expected since no think tags)\n")

    # --- Test 2: ground truth WITH think tags ---
    print("=== Same queries wrapped in <think>...</think> ===")
    print(f"{'#':>3}  {'Total':>6}  {'Fmt':>5}  {'Ngram':>5}  Think?")
    print("-" * 90)
    totals2 = []
    for i, ex in enumerate(examples[:5]):
        gt = ex["cql_query"]
        wrapped = f"<think>\nI need to write a query for {ex.get('nl_query', 'this task')[:40]}...\n</think>\n{gt}"
        r = compute_combined_reward(wrapped, gt, weights)
        totals2.append(r["reward"])
        print(f"{i+1:3d}  {r['reward']:6.3f}  {r['format']:5.2f}  {r['ngram']:5.2f}  {r['has_thinking']}")
    print(f"Mean: {sum(totals2)/len(totals2):.3f}  (format=1.0 expected with proper tags)\n")

    # --- Test 3: broken / edge cases ---
    gt0 = examples[0]["cql_query"]
    broken = [
        ("Empty string", ""),
        ("Just text (no CQL)", "show me all events"),
        ("Think but no CQL after", "<think>I'm thinking...</think>"),
        ("Think + wrong CQL", f"<think>reasoning</think>\nSELECT * FROM events"),
        ("Partial think (no close)", f"<think>reasoning\n{gt0}"),
        ("Perfect with think", f"<think>Let me write this query</think>\n{gt0}"),
        ("Perfect without think", gt0),
    ]
    print("=== Edge cases ===")
    print(f"{'Test':<28}  {'Total':>6}  {'Fmt':>5}  {'Ngram':>5}  {'Think':>5}  Extracted CQL (first 50)")
    print("-" * 110)
    for name, query in broken:
        r = compute_combined_reward(query, gt0, weights)
        cql_preview = r["extracted_cql"][:50].replace("\n", " ")
        print(f"{name:<28}  {r['reward']:6.3f}  {r['format']:5.2f}  {r['ngram']:5.2f}  {str(r['has_thinking']):>5}  {cql_preview}")

    # --- Test 4: format reward incentive check ---
    print("\n=== Incentive check: does <think> get higher reward? ===")
    r_bare = compute_combined_reward(gt0, gt0, weights)
    r_think = compute_combined_reward(f"<think>reasoning</think>\n{gt0}", gt0, weights)
    print(f"  Bare CQL:         reward={r_bare['reward']:.3f} (format={r_bare['format']:.1f}, ngram={r_bare['ngram']:.2f})")
    print(f"  With <think> tags: reward={r_think['reward']:.3f} (format={r_think['format']:.1f}, ngram={r_think['ngram']:.2f})")
    delta = r_think["reward"] - r_bare["reward"]
    print(f"  Delta: +{delta:.3f} {'✓ think tags rewarded' if delta > 0 else '✗ NO INCENTIVE — check weights!'}")


if __name__ == "__main__":
    main()
