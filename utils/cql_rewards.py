"""CQL reward components — pure Python, no GPU dependencies.

Three R1-style reward signals:
  1. N-gram similarity — bigram F1 vs reference (accuracy)
  2. Format compliance — <think>...</think> tags (R1 format reward)
  3. Execution — placeholder for Docker LogScale (correctness)

Importable anywhere: local testing, environment, notebooks.
"""

import re

from utils.cql_tokenizer import bigram_similarity


def extract_cql_from_response(response: str) -> tuple[str, str | None]:
    """Extract CQL answer and optional thinking from model response.

    Returns:
        (cql_text, thinking_text) — thinking is None if no tags found.
    """
    response = response.strip()

    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        after_think = response[match.end():].strip()
        return (after_think, thinking)  # may be empty string if no CQL after tags

    return (response, None)


def compute_format_reward(response: str) -> float:
    """Score format compliance: proper <think>...</think> tag usage.

    Returns float in [0, 1]:
      - 0.0: no think tags at all
      - 0.5: partial (open only, wrong order, or empty/trivial thinking)
      - 1.0: properly paired with substantive thinking (≥10 chars)
    """
    has_open = "<think>" in response
    has_close = "</think>" in response

    if has_open and has_close:
        open_pos = response.index("<think>")
        close_pos = response.index("</think>")
        if close_pos > open_pos:
            # Check thinking is substantive (prevent empty <think></think> hacking)
            thinking = response[open_pos + len("<think>"):close_pos].strip()
            return 1.0 if len(thinking) >= 10 else 0.5
        return 0.5
    elif has_open:
        return 0.5
    return 0.0


def compute_ngram_reward(generated_cql: str, reference_cql: str) -> float:
    """Bigram F1 similarity between generated and reference CQL. Returns [0, 1]."""
    return bigram_similarity(generated_cql, reference_cql)


def compute_execution_reward(cql: str) -> float:
    """Placeholder for Docker LogScale compilation check.

    TODO: Replace with HTTP call to LogScale sandbox container.
    Returns 0.0 always. Set execution weight to 0 until Docker is ready.
    """
    return 0.0


def compute_combined_reward(
    response: str,
    reference_cql: str,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute all reward components and weighted total.

    Args:
        response: Full model response (may contain <think> tags).
        reference_cql: Ground truth CQL query.
        weights: Dict with keys 'format', 'ngram', 'execution'.

    Returns:
        Dict with 'reward', 'format', 'ngram', 'execution',
        'extracted_cql', 'has_thinking'.
    """
    if weights is None:
        weights = {"format": 0.2, "ngram": 0.8, "execution": 0.0}

    extracted_cql, thinking = extract_cql_from_response(response)
    fmt = compute_format_reward(response)
    ngram = compute_ngram_reward(extracted_cql, reference_cql)
    execution = compute_execution_reward(extracted_cql)

    combined = (
        weights.get("format", 0.0) * fmt
        + weights.get("ngram", 0.0) * ngram
        + weights.get("execution", 0.0) * execution
    )
    combined = max(0.0, min(1.0, combined))

    return {
        "reward": combined,
        "format": fmt,
        "ngram": ngram,
        "execution": execution,
        "extracted_cql": extracted_cql,
        "has_thinking": thinking is not None,
    }
