"""CQL reward components — pure Python, no GPU dependencies.

Four R1-style reward signals:
  1. Format compliance — <think>...</think> tags (R1 format reward)
  2. Structure match   — Jaccard of pipeline function names (right operations)
  3. Field match       — F1 of entities: tags, field names, string literals (right data)
  4. Execution         — placeholder for Docker LogScale compilation (correctness)

Importable anywhere: local testing, environment, notebooks.
"""

import re

from utils.cql_tokenizer import (
    extract_function_names,
    structural_similarity,
    tokenize_typed,
)


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
        return (after_think, thinking)

    return (response, None)


def compute_format_reward(response: str) -> float:
    """Score format compliance: proper <think>...</think> tag usage.

    Returns float in [0, 1]:
      - 0.0: no think tags at all
      - 0.5: has one tag (<think> or </think>) but not both
      - 1.0: has both <think> and </think>
    """
    has_open = "<think>" in response
    has_close = "</think>" in response

    if has_open and has_close:
        return 1.0
    elif has_open or has_close:
        return 0.5
    return 0.0


def compute_structure_reward(generated_cql: str, reference_cql: str) -> float:
    """Jaccard similarity of pipeline function names.

    Measures: did the model use the right CQL operations (where, groupBy, stats, etc.)
    in the right combination? Order-insensitive — only checks function presence.

    Returns [0, 1].
    """
    return structural_similarity(generated_cql, reference_cql)


def _extract_entities(cql: str) -> set[str]:
    """Extract semantic entities from CQL: tags, field names, string literals.

    Excludes pipeline function names (those are captured by structure reward).
    Lowercased for case-insensitive matching.
    """
    typed_tokens = tokenize_typed(cql)
    func_names = set(extract_function_names(cql))

    entities = set()
    for token, ttype in typed_tokens:
        if ttype == "TAG":
            entities.add(token.lower())
        elif ttype == "STRING":
            entities.add(token.lower())
        elif ttype == "IDENTIFIER" and token not in func_names:
            entities.add(token.lower())
    return entities


def compute_field_reward(generated_cql: str, reference_cql: str) -> float:
    """Set F1 of semantic entities (tags, field names, string literals).

    Measures: did the model reference the right event types, field names,
    and filter values? Ignores function names (handled by structure reward)
    and operators/syntax (too noisy).

    Returns [0, 1].
    """
    gen_entities = _extract_entities(generated_cql)
    ref_entities = _extract_entities(reference_cql)

    if not gen_entities and not ref_entities:
        return 1.0
    if not gen_entities or not ref_entities:
        return 0.0

    intersection = gen_entities & ref_entities
    precision = len(intersection) / len(gen_entities)
    recall = len(intersection) / len(ref_entities)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


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
        weights: Dict with keys 'format', 'structure', 'fields', 'execution'.

    Returns:
        Dict with 'reward', 'format', 'structure', 'fields', 'execution',
        'extracted_cql', 'has_thinking'.
    """
    if weights is None:
        weights = {"format": 0.1, "structure": 0.3, "fields": 0.6, "execution": 0.0}

    extracted_cql, thinking = extract_cql_from_response(response)
    fmt = compute_format_reward(response)
    structure = compute_structure_reward(extracted_cql, reference_cql)
    fields = compute_field_reward(extracted_cql, reference_cql)
    execution = compute_execution_reward(extracted_cql)

    combined = (
        weights.get("format", 0.0) * fmt
        + weights.get("structure", 0.0) * structure
        + weights.get("fields", 0.0) * fields
        + weights.get("execution", 0.0) * execution
    )
    combined = max(0.0, min(1.0, combined))

    return {
        "reward": combined,
        "format": fmt,
        "structure": structure,
        "fields": fields,
        "execution": execution,
        "extracted_cql": extracted_cql,
        "has_thinking": thinking is not None,
    }
