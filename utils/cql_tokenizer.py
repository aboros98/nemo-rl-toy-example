"""CQL Tokenizer — semantic-level tokenizer for CrowdStrike LogScale Query Language.

Splits CQL at the semantic level, preserving function names, #tag values,
and quoted strings as atomic units. Provides bigram similarity and
structural similarity metrics.
"""

import re
from collections import Counter


# Regex patterns for CQL token types
_PATTERNS = [
    # Double-quoted strings (preserve as atomic)
    (r'"(?:[^"\\]|\\.)*"', "STRING"),
    # Single-quoted strings
    (r"'(?:[^'\\]|\\.)*'", "STRING"),
    # Regex literals /pattern/flags
    (r"/(?:[^/\\]|\\.)+/[gimsuy]*", "REGEX"),
    # Tag values: #tag or #"quoted tag"
    (r'#"(?:[^"\\]|\\.)*"', "TAG"),
    (r"#[a-zA-Z_][a-zA-Z0-9_]*", "TAG"),
    # Numbers (integers and floats, including scientific notation)
    (r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b", "NUMBER"),
    # Comparison operators (must come before logical ! to match != first)
    (r"[!<>=]=|[<>]", "OPERATOR"),
    # Logical operators must come BEFORE pipe to avoid || splitting into two |
    (r"&&|\|\||!(?!=)", "LOGICAL"),
    # Assignment operator
    (r":=", "OPERATOR"),
    # Pipe operator (after || is already matched)
    (r"\|", "PIPE"),
    # Wildcards
    (r"\*", "WILDCARD"),
    # Parentheses and brackets
    (r"[()]", "PAREN"),
    (r"[\[\]]", "BRACKET"),
    (r"[{}]", "BRACE"),
    # Commas
    (r",", "COMMA"),
    # Identifiers and function names (including dotted paths like field.subfield)
    (r"[a-zA-Z_][a-zA-Z0-9_.]*", "IDENTIFIER"),
    # Catch-all for other single characters
    (r"\S", "OTHER"),
]

_COMBINED_PATTERN = "|".join(f"(?P<T{i}>{pat})" for i, (pat, _) in enumerate(_PATTERNS))
_COMPILED = re.compile(_COMBINED_PATTERN)
_TOKEN_TYPE_MAP = {f"T{i}": ttype for i, (_, ttype) in enumerate(_PATTERNS)}

# Pattern to detect pipeline stage names: first identifier in a stage
# Captures both function-style (groupBy(...)) and keyword-style (where ..., sort ...)
_PIPELINE_STAGE_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_.]*)")


def tokenize(cql: str) -> list[str]:
    """Tokenize a CQL query into semantic tokens.

    Preserves:
    - Function names as atomic units
    - #tag values as atomic units
    - Quoted strings as atomic units
    - Operators, pipes, and structural characters

    Args:
        cql: A CQL query string.

    Returns:
        List of token strings.
    """
    tokens = []
    for match in _COMPILED.finditer(cql):
        token = match.group()
        tokens.append(token)
    return tokens


def tokenize_typed(cql: str) -> list[tuple[str, str]]:
    """Tokenize a CQL query, returning (token, type) pairs.

    Args:
        cql: A CQL query string.

    Returns:
        List of (token_text, token_type) tuples.
    """
    result = []
    for match in _COMPILED.finditer(cql):
        token = match.group()
        for group_name, group_val in match.groupdict().items():
            if group_val is not None:
                ttype = _TOKEN_TYPE_MAP[group_name]
                result.append((token, ttype))
                break
    return result


def extract_function_names(cql: str) -> list[str]:
    """Extract the ordered list of function/keyword names from a CQL pipeline.

    Identifies the first identifier in each pipeline stage, capturing both
    function-style calls (groupBy(...)) and keyword-style (where ..., sort ...).

    Args:
        cql: A CQL query string.

    Returns:
        Ordered list of pipeline stage names.
    """
    functions = []
    stages = cql.split("|")
    for stage in stages:
        stage = stage.strip()
        if not stage:
            continue
        m = _PIPELINE_STAGE_RE.match(stage)
        if m:
            functions.append(m.group(1))
    return functions


def _bigrams(tokens: list[str]) -> Counter:
    """Compute bigram counts from a token list."""
    return Counter(
        (tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)
    )


def bigram_similarity(cql_a: str, cql_b: str) -> float:
    """Compute bigram similarity (symmetric F1 of bigram overlap).

    This is the Dice coefficient of bigrams:
        2 * |intersection| / (|bigrams_a| + |bigrams_b|)

    Args:
        cql_a: First CQL query string.
        cql_b: Second CQL query string.

    Returns:
        Float in [0, 1]. 1.0 means identical bigram sets.
    """
    tokens_a = tokenize(cql_a)
    tokens_b = tokenize(cql_b)

    if len(tokens_a) < 2 and len(tokens_b) < 2:
        # Both have 0 or 1 token — compare directly
        return 1.0 if tokens_a == tokens_b else 0.0

    bg_a = _bigrams(tokens_a)
    bg_b = _bigrams(tokens_b)

    if not bg_a and not bg_b:
        return 1.0 if tokens_a == tokens_b else 0.0

    # Intersection count (min overlap per bigram)
    overlap = sum((bg_a & bg_b).values())
    total = sum(bg_a.values()) + sum(bg_b.values())

    if total == 0:
        return 0.0

    return 2.0 * overlap / total


def structural_similarity(cql_a: str, cql_b: str) -> float:
    """Compute structural similarity as Jaccard of ordered function name lists.

    Jaccard index = |intersection| / |union| of the multisets of function names.

    Args:
        cql_a: First CQL query string.
        cql_b: Second CQL query string.

    Returns:
        Float in [0, 1]. 1.0 means identical function pipeline structure.
    """
    funcs_a = Counter(extract_function_names(cql_a))
    funcs_b = Counter(extract_function_names(cql_b))

    if not funcs_a and not funcs_b:
        return 1.0

    intersection = sum((funcs_a & funcs_b).values())
    union = sum((funcs_a | funcs_b).values())

    if union == 0:
        return 0.0

    return intersection / union
