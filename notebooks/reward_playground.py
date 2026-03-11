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
    compute_format_reward,
    compute_ngram_reward,
    extract_cql_from_response,
)
from utils.cql_tokenizer import tokenize as cql_tokenize, bigram_similarity

# ─── Load real training data ─────────────────────────────────────────────────

DATA_PATH = PROJECT_ROOT / "data" / "train.jsonl"
EXAMPLES = []
if DATA_PATH.exists():
    with open(DATA_PATH) as f:
        EXAMPLES = [json.loads(line) for line in f]
    print(f"✓ Loaded {len(EXAMPLES)} training examples from {DATA_PATH.name}")
else:
    print(f"⚠ No training data at {DATA_PATH} — run scripts/fetch_data.py first")

DEFAULT_WEIGHTS = {"format": 0.2, "ngram": 0.8, "execution": 0.0}


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
    print(f"  N-gram reward:    {r['ngram']:.2f}  × {w.get('ngram', 0):.1f}  = {r['ngram'] * w.get('ngram', 0):.3f}")
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

    print("3) Model gets CQL partially right (missing tail):")
    partial = " | ".join(cql.split(" | ")[:2])  # keep first 2 pipe stages
    score(f"<think>partial attempt</think>\n{partial}", cql)

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
        results.append((r["reward"], r["format"], r["ngram"], resp))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'═' * 80}")
    print(f"  GRPO Group — {len(responses)} rollouts for same prompt")
    print(f"  Reference: {_trunc(reference, 65)}")
    print(f"{'═' * 80}")
    print(f"  {'Rank':>4}  {'Reward':>7}  {'Fmt':>5}  {'Ngram':>6}  Response")
    print(f"  {'─' * 74}")
    rewards = []
    for i, (reward, fmt, ngram, resp) in enumerate(results):
        rewards.append(reward)
        prefix = "  ►" if i == 0 else "   "
        print(f"{prefix}{i+1:3d}   {reward:7.3f}  {fmt:5.1f}  {ngram:6.3f}  {_trunc(resp.replace(chr(10), ' '), 45)}")

    mean_r = sum(rewards) / len(rewards) if rewards else 0
    print(f"  {'─' * 74}")
    print(f"  Mean={mean_r:.3f}  Spread={max(rewards) - min(rewards):.3f}  "
          f"Std={_std(rewards):.3f}")

    # Show GRPO advantages
    print(f"\n  GRPO advantages (reward - mean):")
    for i, (reward, _, _, resp) in enumerate(results):
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


def explain_ngram(a: str, b: str):
    """Show exactly how bigram similarity is computed between two CQL strings."""
    tokens_a = cql_tokenize(a)
    tokens_b = cql_tokenize(b)

    # Build bigrams from token values
    vals_a = [t[0] if isinstance(t, tuple) else str(t) for t in tokens_a]
    vals_b = [t[0] if isinstance(t, tuple) else str(t) for t in tokens_b]

    bigrams_a = set(zip(vals_a, vals_a[1:])) if len(vals_a) >= 2 else set()
    bigrams_b = set(zip(vals_b, vals_b[1:])) if len(vals_b) >= 2 else set()

    overlap = bigrams_a & bigrams_b
    only_a = bigrams_a - bigrams_b
    only_b = bigrams_b - bigrams_a

    sim = bigram_similarity(a, b)

    print(f"\n{'─' * 60}")
    print(f"  A: {_trunc(a, 55)}")
    print(f"  B: {_trunc(b, 55)}")
    print(f"{'─' * 60}")
    print(f"  Tokens A: {vals_a}")
    print(f"  Tokens B: {vals_b}")
    print(f"  Bigrams A: {len(bigrams_a)}  Bigrams B: {len(bigrams_b)}")
    print(f"  Overlap:   {len(overlap)}")
    if overlap:
        print(f"    ✓ shared: {list(overlap)[:8]}{'...' if len(overlap) > 8 else ''}")
    if only_a:
        print(f"    - only in A: {list(only_a)[:5]}{'...' if len(only_a) > 5 else ''}")
    if only_b:
        print(f"    - only in B: {list(only_b)[:5]}{'...' if len(only_b) > 5 else ''}")
    print(f"  Dice coefficient: 2×{len(overlap)} / ({len(bigrams_a)}+{len(bigrams_b)}) = {sim:.4f}")
    print(f"{'─' * 60}\n")


def sweep_weights(response: str, reference: str):
    """Show how reward changes as format vs ngram weight shifts."""
    print(f"\n{'═' * 60}")
    print(f"  Weight sweep: format ← → ngram")
    print(f"  Response:  {_trunc(response, 50)}")
    print(f"  Reference: {_trunc(reference, 50)}")
    print(f"{'═' * 60}")
    print(f"  {'Format W':>9}  {'Ngram W':>8}  {'Reward':>7}  {'Bar'}")
    print(f"  {'─' * 50}")
    for fmt_w in [i / 10 for i in range(11)]:
        ngram_w = 1.0 - fmt_w
        w = {"format": fmt_w, "ngram": ngram_w, "execution": 0.0}
        r = compute_combined_reward(response, reference, w)
        bar = "█" * int(r["reward"] * 30)
        print(f"  {fmt_w:>8.1f}   {ngram_w:>7.1f}   {r['reward']:7.3f}  {bar}")
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
    # Partial (first N-1 stages)
    if len(cql_parts) > 1:
        partial = " | ".join(cql_parts[:-1])
        rollouts.append(f"<think>almost there</think>\n{partial}")
    # Wrong but valid-looking
    rollouts.append("<think>let me try</think>\n#event=Unknown | count()")
    # Only thinking
    rollouts.append("<think>I need to think about this more carefully but I'm not sure what the right approach is</think>")
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

    # ── Demo 2: Tokenizer internals ──
    print("\n" + "▸" * 35)
    print("  DEMO 2: How the CQL tokenizer works")
    print("▸" * 35)
    ref = EXAMPLES[0]["cql_query"]
    show_tokens(ref)

    # ── Demo 3: N-gram similarity step by step ──
    print("\n" + "▸" * 35)
    print("  DEMO 3: N-gram similarity — step by step")
    print("▸" * 35)
    parts = ref.split(" | ")
    if len(parts) > 2:
        partial = " | ".join(parts[:2])
    else:
        partial = parts[0] if parts else ""
    explain_ngram(ref, ref)               # identical
    explain_ngram(partial, ref)            # partial match
    explain_ngram("SELECT * FROM x", ref)  # completely different

    # ── Demo 4: Weight sweep ──
    print("\n" + "▸" * 35)
    print("  DEMO 4: How weights change the reward")
    print("▸" * 35)
    # Good CQL with think tags
    sweep_weights(f"<think>reasoning</think>\n{ref}", ref)
    # Wrong CQL with think tags — format can't save you
    sweep_weights(f"<think>reasoning</think>\nSELECT * FROM events", ref)

    # ── Demo 5: GRPO group simulation ──
    print("\n" + "▸" * 35)
    print("  DEMO 5: Simulated GRPO group (8 rollouts)")
    print("▸" * 35)
    grpo_sim(ref, 8)

    # ── Demo 6: Key insights ──
    print("\n" + "▸" * 35)
    print("  KEY INSIGHTS")
    print("▸" * 35)
    print("""
    1. N-gram (0.8 weight) is the primary signal — accuracy matters most
    2. Format (0.2 weight) adds +0.20 for <think> tags — a small but consistent bonus
    3. After extraction, only the CQL after </think> is compared to the reference
    4. The tokenizer treats #tags, function names, and quoted strings as atomic units
    5. Empty responses score 0.0 — GRPO pushes away from them
    6. In a GRPO group, SPREAD matters: diverse rewards → strong gradient signal

    To play interactively:
        from notebooks.reward_playground import *
        score("<think>reasoning</think>\\n#event=X | count()", "#event=X | count()")
        show_example(5)
        grpo_sim("#event=DnsRequest | groupBy(DomainName) | head(100)")
        interactive()  # type responses, see rewards live
    """)


if __name__ == "__main__":
    main()
