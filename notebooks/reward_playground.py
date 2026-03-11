#!/usr/bin/env python3
"""
CQL Reward Playground — interactive exploration of how rewards work.

Run as a script:
    python notebooks/reward_playground.py

Or use in Jupyter / IPython:
    %run notebooks/reward_playground.py
    # then call any function interactively:
    score("your CQL here", reference="ref CQL")

No GPU, no NeMo RL, no Ray. Pure Python on Mac.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cql_rewards import (
    compute_combined_reward,
    compute_field_reward,
    compute_format_reward,
    compute_structure_reward,
    extract_cql_from_response,
    _extract_entities,
)
from utils.cql_tokenizer import (
    extract_function_names,
    tokenize as cql_tokenize,
    structural_similarity,
)

# ─── Load real training data ─────────────────────────────────────────────────

DATA_PATH = PROJECT_ROOT / "data" / "train.jsonl"
EXAMPLES = []
if DATA_PATH.exists():
    with open(DATA_PATH) as f:
        EXAMPLES = [json.loads(line) for line in f]
    print(f"✓ Loaded {len(EXAMPLES)} training examples from {DATA_PATH.name}")
else:
    print(f"⚠ No training data at {DATA_PATH} — run scripts/fetch_data.py first")

DEFAULT_WEIGHTS = {"format": 0.1, "structure": 0.3, "fields": 0.6, "execution": 0.0}


# ─── Helper functions ─────────────────────────────────────────────────────────

def score(response: str, reference: str = "", weights: dict = None) -> dict:
    """Score a single model response against a reference. Pretty-prints result."""
    w = weights or DEFAULT_WEIGHTS
    r = compute_combined_reward(response, reference, w)

    print(f"\n{'─' * 70}")
    print(f"  Response:      {_trunc(response, 80)}")
    print(f"  Reference:     {_trunc(reference, 80)}")
    print(f"  Extracted CQL: {_trunc(r['extracted_cql'], 80)}")
    print(f"  Has thinking:  {r['has_thinking']}")
    print(f"{'─' * 70}")
    print(f"  Format reward:    {r['format']:.2f}  × {w.get('format', 0):.1f}  = {r['format'] * w.get('format', 0):.3f}")
    print(f"  Structure reward: {r['structure']:.2f}  × {w.get('structure', 0):.1f}  = {r['structure'] * w.get('structure', 0):.3f}")
    print(f"  Fields reward:    {r['fields']:.2f}  × {w.get('fields', 0):.1f}  = {r['fields'] * w.get('fields', 0):.3f}")
    print(f"  Execution reward: {r['execution']:.2f}  × {w.get('execution', 0):.1f}  = {r['execution'] * w.get('execution', 0):.3f}")
    print(f"{'─' * 70}")
    print(f"  ► TOTAL REWARD: {r['reward']:.3f}")
    print(f"{'─' * 70}\n")
    return r


def show_example(idx: int = 0):
    """Show a training example and score it in different ways."""
    ex = EXAMPLES[idx]
    nl = ex["nl_query"]
    cql = ex["cql_query"]
    src = ex.get("source", "?")

    print(f"\n{'═' * 70}")
    print(f"  Example #{idx}  (source: {src})")
    print(f"  NL:  {_trunc(nl, 70)}")
    print(f"  CQL: {_trunc(cql, 70)}")
    print(f"{'═' * 70}")

    print("\n1) Model returns the exact reference (no think tags):")
    score(cql, cql)

    print("2) Model wraps it in <think> tags (perfect format + perfect CQL):")
    score(f"<think>\nI need to write a query for {_trunc(nl, 40)}\n</think>\n{cql}", cql)

    print("3) Model gets right functions but wrong fields:")
    funcs = extract_function_names(cql)
    if len(cql.split(" | ")) > 1:
        wrong = "#event_simpleName=DnsRequest | " + " | ".join(f"{f}(wrongField)" for f in funcs[1:] if f)
    else:
        wrong = "#event_simpleName=DnsRequest"
    score(f"<think>close but wrong data</think>\n{wrong}", cql)

    print("4) Model outputs garbage:")
    score("<think>I don't know</think>\nSELECT * FROM events", cql)

    print("5) Model outputs nothing after think:")
    score("<think>I'm stuck...</think>", cql)


def compare(responses: list[str], reference: str, weights: dict = None):
    """Score multiple responses against the same reference. Shows GRPO-style ranking."""
    w = weights or DEFAULT_WEIGHTS
    results = []
    for resp in responses:
        r = compute_combined_reward(resp, reference, w)
        results.append((r["reward"], r["format"], r["structure"], r["fields"], resp))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'═' * 90}")
    print(f"  GRPO Group — {len(responses)} rollouts for same prompt")
    print(f"  Reference: {_trunc(reference, 65)}")
    print(f"{'═' * 90}")
    print(f"  {'Rank':>4}  {'Reward':>7}  {'Fmt':>5}  {'Struc':>5}  {'Field':>5}  Response")
    print(f"  {'─' * 84}")
    rewards = []
    for i, (reward, fmt, struc, fld, resp) in enumerate(results):
        rewards.append(reward)
        prefix = "  ►" if i == 0 else "   "
        print(f"{prefix}{i+1:3d}   {reward:7.3f}  {fmt:5.1f}  {struc:5.2f}  {fld:5.2f}  {_trunc(resp.replace(chr(10), ' '), 40)}")

    mean_r = sum(rewards) / len(rewards) if rewards else 0
    print(f"  {'─' * 84}")
    print(f"  Mean={mean_r:.3f}  Spread={max(rewards) - min(rewards):.3f}  "
          f"Std={_std(rewards):.3f}")

    # Show GRPO advantages
    print(f"\n  GRPO advantages (reward - mean):")
    for i, (reward, _, _, _, resp) in enumerate(results):
        adv = reward - mean_r
        bar = "█" * int(abs(adv) * 40)
        sign = "+" if adv >= 0 else "-"
        print(f"    Rollout {i+1}: {sign}{abs(adv):.3f}  {'▓' if adv >= 0 else '░'}{bar}")
    print()


def show_tokens(cql: str):
    """Show how the CQL tokenizer sees a query."""
    tokens = cql_tokenize(cql)
    print(f"\n  Input: {cql}")
    print(f"  Tokens ({len(tokens)}):")
    for tok in tokens:
        print(f"    {tok}")
    print()


def explain_structure(a: str, b: str):
    """Show how structure reward compares two CQL queries."""
    funcs_a = extract_function_names(a)
    funcs_b = extract_function_names(b)
    sim = structural_similarity(a, b)

    print(f"\n{'─' * 60}")
    print(f"  A: {_trunc(a, 55)}")
    print(f"  B: {_trunc(b, 55)}")
    print(f"{'─' * 60}")
    print(f"  Functions A: {funcs_a}")
    print(f"  Functions B: {funcs_b}")
    print(f"  Jaccard similarity: {sim:.4f}")
    print(f"{'─' * 60}\n")


def explain_fields(a: str, b: str):
    """Show how field reward compares two CQL queries."""
    ents_a = _extract_entities(a)
    ents_b = _extract_entities(b)
    overlap = ents_a & ents_b
    only_a = ents_a - ents_b
    only_b = ents_b - ents_a
    sim = compute_field_reward(a, b)

    print(f"\n{'─' * 60}")
    print(f"  A: {_trunc(a, 55)}")
    print(f"  B: {_trunc(b, 55)}")
    print(f"{'─' * 60}")
    print(f"  Entities A ({len(ents_a)}): {sorted(ents_a)[:10]}")
    print(f"  Entities B ({len(ents_b)}): {sorted(ents_b)[:10]}")
    print(f"  Shared ({len(overlap)}): {sorted(overlap)[:10]}")
    if only_a:
        print(f"  Only in A: {sorted(only_a)[:5]}")
    if only_b:
        print(f"  Only in B: {sorted(only_b)[:5]}")
    print(f"  F1 score: {sim:.4f}")
    print(f"{'─' * 60}\n")


def sweep_weights(response: str, reference: str):
    """Show how reward changes as structure vs fields weight shifts."""
    print(f"\n{'═' * 65}")
    print(f"  Weight sweep: structure ← → fields (format=0.1 fixed)")
    print(f"  Response:  {_trunc(response, 50)}")
    print(f"  Reference: {_trunc(reference, 50)}")
    print(f"{'═' * 65}")
    print(f"  {'Struc W':>8}  {'Field W':>8}  {'Reward':>7}  {'Bar'}")
    print(f"  {'─' * 55}")
    for s_w in [i / 10 for i in range(10)]:
        f_w = 0.9 - s_w  # remaining after format=0.1
        w = {"format": 0.1, "structure": s_w, "fields": f_w, "execution": 0.0}
        r = compute_combined_reward(response, reference, w)
        bar = "█" * int(r["reward"] * 30)
        print(f"  {s_w:>7.1f}   {f_w:>7.1f}   {r['reward']:7.3f}  {bar}")
    print()


def grpo_sim(reference: str, n_rollouts: int = 8):
    """Simulate a GRPO group with synthetic rollouts of varying quality."""
    import random

    cql_parts = reference.split(" | ")
    rollouts = []

    # Perfect with think
    rollouts.append(f"<think>I'll construct the full query</think>\n{reference}")
    # Perfect without think
    rollouts.append(reference)
    # Right structure, wrong fields
    funcs = extract_function_names(reference)
    if funcs:
        wrong_fields = "#event_simpleName=DnsRequest | " + " | ".join(f"{f}(wrongField)" for f in funcs[1:] if f)
        rollouts.append(f"<think>close but wrong data</think>\n{wrong_fields}")
    # Partial (first N-1 stages)
    if len(cql_parts) > 1:
        partial = " | ".join(cql_parts[:-1])
        rollouts.append(f"<think>almost there</think>\n{partial}")
    # Wrong but valid-looking
    rollouts.append("<think>let me try</think>\n#event=Unknown | count()")
    # Only thinking
    rollouts.append("<think>I need to think about this more carefully</think>")
    # Garbage
    rollouts.append("SELECT * FROM events WHERE type = 'dns'")
    # Empty
    rollouts.append("")

    # Pad to n_rollouts with random mutations
    while len(rollouts) < n_rollouts:
        if cql_parts:
            shuffled = cql_parts.copy()
            random.shuffle(shuffled)
            mutated = " | ".join(shuffled[:max(1, len(shuffled) - 1)])
            rollouts.append(f"<think>trying a variation</think>\n{mutated}")

    rollouts = rollouts[:n_rollouts]
    compare(rollouts, reference)


def interactive():
    """Interactive REPL: type CQL, see rewards instantly."""
    if not EXAMPLES:
        print("No training data loaded. Run scripts/fetch_data.py first.")
        return

    ex = EXAMPLES[0]
    reference = ex["cql_query"]
    print(f"\n{'═' * 70}")
    print(f"  Interactive Reward Tester")
    print(f"  Reference CQL: {_trunc(reference, 55)}")
    print(f"  Type a model response and press Enter to see its reward.")
    print(f"  Commands: 'next' (new reference), 'quit', 'ref' (show reference)")
    print(f"{'═' * 70}\n")

    idx = 0
    while True:
        try:
            user_input = input("response> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "next":
            idx = (idx + 1) % len(EXAMPLES)
            ex = EXAMPLES[idx]
            reference = ex["cql_query"]
            print(f"\n  New reference (#{idx}): {_trunc(reference, 60)}\n")
            continue
        elif user_input.lower() == "ref":
            print(f"\n  NL:  {ex['nl_query']}")
            print(f"  CQL: {reference}\n")
            continue
        elif not user_input:
            continue

        score(user_input, reference)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _trunc(s: str, n: int) -> str:
    s = s.replace("\n", " ↵ ")
    return s[:n] + "..." if len(s) > n else s


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


# ─── Main: run all demos ─────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  CQL REWARD PLAYGROUND")
    print("  Understand how each component contributes to the final reward.")
    print("=" * 70)

    if not EXAMPLES:
        print("\n⚠ No training data. Run: python scripts/fetch_data.py\n")
        return

    # ── Demo 1: Score a single example in multiple ways ──
    print("\n\n" + "▸" * 35)
    print("  DEMO 1: How the same query scores differently")
    print("▸" * 35)
    show_example(0)

    # ── Demo 2: Tokenizer + entity extraction ──
    print("\n" + "▸" * 35)
    print("  DEMO 2: How the CQL tokenizer and entity extraction work")
    print("▸" * 35)
    ref = EXAMPLES[0]["cql_query"]
    show_tokens(ref)
    print("  Extracted entities (tags, fields, strings — NOT function names):")
    ents = _extract_entities(ref)
    print(f"    {sorted(ents)}")
    print(f"\n  Pipeline functions:")
    print(f"    {extract_function_names(ref)}")

    # ── Demo 3: Structure + field comparison step by step ──
    print("\n" + "▸" * 35)
    print("  DEMO 3: Structure and field rewards — step by step")
    print("▸" * 35)
    explain_structure(ref, ref)
    if len(ref.split(" | ")) > 2:
        partial = " | ".join(ref.split(" | ")[:2])
        explain_structure(partial, ref)
    explain_fields(ref, ref)
    explain_fields("#event_simpleName=DnsRequest | groupBy(DomainName)", ref)

    # ── Demo 4: Weight sweep ──
    print("\n" + "▸" * 35)
    print("  DEMO 4: How weights change the reward")
    print("▸" * 35)
    sweep_weights(f"<think>reasoning</think>\n{ref}", ref)
    sweep_weights(f"<think>reasoning</think>\nSELECT * FROM events", ref)

    # ── Demo 5: GRPO group simulation ──
    print("\n" + "▸" * 35)
    print("  DEMO 5: Simulated GRPO group (8 rollouts)")
    print("▸" * 35)
    grpo_sim(ref, 8)

    # ── Key insights ──
    print("\n" + "▸" * 35)
    print("  KEY INSIGHTS")
    print("▸" * 35)
    print("""
    1. Fields (0.6 weight) is the primary signal — did you reference the right data?
    2. Structure (0.3 weight) — did you use the right CQL operations?
    3. Format (0.1 weight) adds +0.10 for <think> tags — consistent bonus
    4. Structure rewards equivalent queries that use different syntax
    5. Fields rewards queries that reference the right event types and field names
    6. Empty responses score 0.0 — GRPO pushes away from them
    7. In a GRPO group, SPREAD matters: diverse rewards → strong gradient signal

    To play interactively:
        from notebooks.reward_playground import *
        score("<think>reasoning</think>\\n#event=X | count()", "#event=X | count()")
        explain_structure("where x>1 | count()", "where x>1 | groupBy(y)")
        explain_fields("#event=X | where y=1", "#event=X | where z=2")
        grpo_sim("#event=DnsRequest | groupBy(DomainName) | head(100)")
        interactive()  # type responses, see rewards live
    """)


if __name__ == "__main__":
    main()
