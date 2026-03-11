#!/usr/bin/env python3
"""Test reward logic locally on Mac — no GPU, no NeMo RL, no Ray needed.

Uses the EXACT same reward functions as the real CQL environment.
Edit utils/cql_rewards.py to change rewards — this script imports from there,
so local tests always match production.

Usage:
    # Run on training data
    python3 scripts/test_rewards_local.py

    # Test a specific golden query
    python3 scripts/test_rewards_local.py --golden "#event_simpleName=ProcessRollup2 | where FileName=cmd.exe | groupBy(ComputerName)"

    # Custom weights
    python3 scripts/test_rewards_local.py --weights '{"format":0.1,"structure":0.4,"fields":0.5,"execution":0.0}'
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cql_rewards import (
    _extract_entities,
    compute_combined_reward,
    compute_field_reward,
    compute_format_reward,
    compute_structure_reward,
    extract_cql_from_response,
)
from utils.cql_tokenizer import extract_function_names

DEFAULT_WEIGHTS = {"format": 0.1, "structure": 0.3, "fields": 0.6, "execution": 0.0}


def explain(response: str, golden: str, weights: dict, label: str = ""):
    """Score a response and show full breakdown of what's being compared."""
    extracted, thinking = extract_cql_from_response(response)
    r = compute_combined_reward(response, golden, weights)

    gen_funcs = extract_function_names(extracted)
    ref_funcs = extract_function_names(golden)
    gen_ents = _extract_entities(extracted)
    ref_ents = _extract_entities(golden)
    shared_ents = gen_ents & ref_ents

    tag = f" [{label}]" if label else ""
    print(f"\n{'─' * 80}")
    print(f"  {label}" if label else "")
    print(f"  Response:      {_trunc(response, 70)}")
    print(f"  Extracted CQL: {_trunc(extracted, 70)}")
    if thinking is not None:
        print(f"  Thinking:      {_trunc(thinking, 70)}")
    print(f"{'─' * 80}")

    # FORMAT
    print(f"  FORMAT  = {r['format']:.1f}  × {weights.get('format',0):.1f} = {r['format']*weights.get('format',0):.3f}")
    print(f"    <think> present: {'<think>' in response}   </think> present: {'</think>' in response}")

    # STRUCTURE
    print(f"  STRUCTURE = {r['structure']:.2f}  × {weights.get('structure',0):.1f} = {r['structure']*weights.get('structure',0):.3f}")
    print(f"    Generated functions: {gen_funcs}")
    print(f"    Reference functions: {ref_funcs}")
    if gen_funcs and ref_funcs:
        shared_f = set(gen_funcs) & set(ref_funcs)
        only_gen = set(gen_funcs) - set(ref_funcs)
        only_ref = set(ref_funcs) - set(gen_funcs)
        if shared_f:
            print(f"    ✓ Shared: {sorted(shared_f)}")
        if only_gen:
            print(f"    ✗ Only in generated: {sorted(only_gen)}")
        if only_ref:
            print(f"    ✗ Missing from generated: {sorted(only_ref)}")

    # FIELDS
    print(f"  FIELDS  = {r['fields']:.2f}  × {weights.get('fields',0):.1f} = {r['fields']*weights.get('fields',0):.3f}")
    print(f"    Generated entities: {sorted(gen_ents)[:8]}{'...' if len(gen_ents)>8 else ''}")
    print(f"    Reference entities: {sorted(ref_ents)[:8]}{'...' if len(ref_ents)>8 else ''}")
    if gen_ents and ref_ents:
        only_gen_e = gen_ents - ref_ents
        only_ref_e = ref_ents - gen_ents
        if shared_ents:
            print(f"    ✓ Shared: {sorted(shared_ents)[:6]}{'...' if len(shared_ents)>6 else ''}")
        if only_gen_e:
            print(f"    ✗ Extra (hallucinated): {sorted(only_gen_e)[:5]}")
        if only_ref_e:
            print(f"    ✗ Missing: {sorted(only_ref_e)[:5]}")

    # TOTAL
    print(f"{'─' * 80}")
    print(f"  ► TOTAL = {r['reward']:.3f}")
    print(f"{'─' * 80}")
    return r


def _trunc(s: str, n: int) -> str:
    s = s.replace("\n", " | ")
    return s[:n] + "..." if len(s) > n else s


def run_golden_demo(golden: str, weights: dict):
    """Given a golden query, show how different model outputs score."""
    ref_funcs = extract_function_names(golden)
    ref_ents = _extract_entities(golden)

    print(f"\n{'═' * 80}")
    print(f"  GOLDEN QUERY ANALYSIS")
    print(f"{'═' * 80}")
    print(f"  Query:     {golden}")
    print(f"  Functions: {ref_funcs}")
    print(f"  Entities:  {sorted(ref_ents)}")
    print(f"  Weights:   format={weights['format']}, structure={weights['structure']}, "
          f"fields={weights['fields']}, execution={weights['execution']}")

    # Build illustrative responses
    parts = golden.replace("\n", " | ").split(" | ")
    parts = [p.strip() for p in parts if p.strip()]

    # 1. Perfect match with think tags
    explain(f"<think>I need to query for this data</think>\n{golden}",
            golden, weights, "PERFECT + THINK TAGS")

    # 2. Perfect match, no think tags
    explain(golden, golden, weights, "PERFECT, NO THINK TAGS")

    # 3. Right functions, wrong fields — keep structure, swap entities
    if len(parts) > 1:
        wrong_fields = parts[0].split("=")[0] + "=WrongEventType"
        for p in parts[1:]:
            wrong_fields += " | " + p.split("(")[0] + "(WrongField)"
        explain(f"<think>close</think>\n{wrong_fields}",
                golden, weights, "RIGHT OPERATIONS, WRONG DATA")

    # 4. Right fields, wrong functions — keep entities, swap operations
    if ref_ents:
        ent_list = sorted(ref_ents)
        wrong_funcs = parts[0] + " | completelyWrongFunc(" + ", ".join(ent_list[:2]) + ")"
        explain(f"<think>hmm</think>\n{wrong_funcs}",
                golden, weights, "RIGHT DATA, WRONG OPERATIONS")

    # 5. Partial query — only first half of pipeline
    if len(parts) > 2:
        partial = " | ".join(parts[:len(parts)//2 + 1])
        explain(f"<think>partial attempt</think>\n{partial}",
                golden, weights, "PARTIAL QUERY (first half)")

    # 6. Completely wrong
    explain("<think>I'll try SQL</think>\nSELECT * FROM events WHERE type='dns'",
            golden, weights, "COMPLETELY WRONG (SQL)")

    # 7. Think tags only, no CQL
    explain("<think>I'm not sure how to write this query...</think>",
            golden, weights, "THINKING ONLY, NO CQL")

    # 8. Empty
    explain("", golden, weights, "EMPTY RESPONSE")

    # Summary table
    print(f"\n{'═' * 80}")
    print(f"  SUMMARY — what each component rewards")
    print(f"{'═' * 80}")
    print(f"""
  STRUCTURE (weight {weights['structure']}):
    Answers: "Did the model use the right CQL operations?"
    Compares: pipeline function names (where, groupBy, stats, sort, head, ...)
    Metric: Jaccard similarity = |shared| / |union|
    Example: 'where | groupBy | head' vs 'where | stats | head' → Jaccard = 2/3

  FIELDS (weight {weights['fields']}):
    Answers: "Did the model reference the right data?"
    Compares: event types (#tags), field names, string literals — NOT function names
    Metric: F1 = 2 × precision × recall / (precision + recall)
    Example: both mention 'ProcessRollup2' and 'FileName' but one has extra 'UserName' → F1 < 1.0

  FORMAT (weight {weights['format']}):
    Answers: "Did the model show its reasoning?"
    Checks: <think>...</think> tag presence
    Values: 0.0 (no tags) / 0.5 (one tag) / 1.0 (both tags)

  EXECUTION (weight {weights['execution']}):
    Not implemented yet. Returns 0.0 always.
    Will check: does the CQL compile in LogScale Docker container?
""")


def run_data_demo(data_path: str, n: int, weights: dict):
    """Run on training data examples."""
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= n:
                break

    print(f"\n{'═' * 80}")
    print(f"  TRAINING DATA — {len(examples)} examples, ground truth scored against itself")
    print(f"{'═' * 80}")
    print(f"{'#':>3}  {'Total':>6}  {'Fmt':>5}  {'Struc':>5}  {'Field':>5}  Golden query (first 50)")
    print("-" * 90)
    for i, ex in enumerate(examples):
        gt = ex["cql_query"]
        r = compute_combined_reward(gt, gt, weights)
        print(f"{i+1:3d}  {r['reward']:6.3f}  {r['format']:5.2f}  {r['structure']:5.2f}  "
              f"{r['fields']:5.2f}  {_trunc(gt, 50)}")

    print(f"\nNote: ground truth vs itself → structure=1.0, fields=1.0, format=0.0 (no think tags)")
    print(f"Max without think tags: {weights['structure'] + weights['fields']:.1f}  |  Max with: 1.0")


def main():
    parser = argparse.ArgumentParser(description="Test CQL rewards locally")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "train.jsonl"))
    parser.add_argument("--n", type=int, default=5, help="Number of data examples")
    parser.add_argument("--golden", type=str, default=None,
                        help='Golden CQL query to test against, e.g. "#event=X | where y>1 | count()"')
    parser.add_argument("--weights", type=str, default=None,
                        help='JSON weights, e.g. \'{"format":0.1,"structure":0.3,"fields":0.6,"execution":0.0}\'')
    args = parser.parse_args()

    weights = json.loads(args.weights) if args.weights else DEFAULT_WEIGHTS
    print(f"Reward weights: format={weights['format']}, structure={weights['structure']}, "
          f"fields={weights['fields']}, execution={weights['execution']}")

    if args.golden:
        run_golden_demo(args.golden, weights)
    else:
        # Use first training example as default golden demo
        with open(args.data) as f:
            first = json.loads(f.readline())
        run_golden_demo(first["cql_query"], weights)
        run_data_demo(args.data, args.n, weights)


if __name__ == "__main__":
    main()
