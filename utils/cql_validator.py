"""CQL Syntax Validator — validates CrowdStrike LogScale Query Language queries.

Checks:
- Balanced parentheses and brackets
- Valid pipe structure (no empty stages, no leading/trailing pipes)
- No unclosed string literals
- Only known CQL function names (whitelist)

Returns a result object with is_valid and a list of errors.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of CQL validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)


# Comprehensive whitelist of known CQL (LogScale) functions
KNOWN_FUNCTIONS: set[str] = {
    # Aggregation functions
    "avg", "count", "counterRate", "max", "min", "percentile",
    "range", "rate", "rdns", "stddev", "sum", "variance",
    # Transformation functions
    "bucket", "case", "cidr", "collect", "concat", "concatArray",
    "convert", "copyEvent", "crypto", "default", "drop", "dropEvent",
    "eval", "eventFieldCount", "eventSize", "extract",
    "fieldset", "findTimestamp", "format", "formatDuration",
    "formatTime", "geohash", "groupBy", "hash", "head",
    "if", "in", "ipLocation", "join", "kvParse", "length",
    "lookup", "lower", "lowercase", "match", "math",
    "now", "parseCEF", "parseCsv", "parseFixedWidth",
    "parseHexString", "parseJson", "parseLEEF", "parseTimestamp",
    "parseUrl", "parseXml", "peek", "regex", "rename",
    "replace", "round", "sample", "select", "selfJoin",
    "selfJoinFilter", "series", "session", "sort",
    "splitString", "start", "stats", "statusHealth", "stint",
    "strip", "stripAnsi", "substring", "table", "tail",
    "test", "time", "timeChart", "timeDelta", "tokenHash",
    "top", "transpose", "trim", "type", "unit", "uniq",
    "upper", "uppercase", "urlDecode", "urlEncode",
    "wildcard", "window", "worldMap", "writeJson",
    # Query functions
    "search", "filter", "where", "not",
    # Field manipulation
    "addField", "removeField", "setField",
    # String functions
    "contains", "startsWith", "endsWith", "indexOf",
    "lastIndexOf", "leftPad", "rightPad",
    # Array functions
    "array", "arrayLength", "flatten",
    # Math functions
    "abs", "ceil", "floor", "log", "log2", "log10",
    "pow", "sqrt", "mod",
    # Date functions
    "formatDate", "parseDate", "dateDiff",
    # Network functions
    "asn", "communityId",
    # Visualization
    "sankey", "worldmap",
    # CrowdStrike-specific
    "aid", "cid", "aidMasterSwitch",
}


def validate(cql: str) -> ValidationResult:
    """Validate a CQL query for syntactic correctness.

    Args:
        cql: A CQL query string.

    Returns:
        ValidationResult with is_valid flag and list of error messages.
    """
    errors = []

    if not cql or not cql.strip():
        return ValidationResult(is_valid=False, errors=["Empty query"])

    _check_balanced_delimiters(cql, errors)
    _check_string_literals(cql, errors)
    _check_pipe_structure(cql, errors)
    _check_function_names(cql, errors)

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def _check_balanced_delimiters(cql: str, errors: list[str]) -> None:
    """Check that parentheses, brackets, and braces are balanced."""
    stack = []
    matching = {")": "(", "]": "[", "}": "{"}
    openers = set("([{")
    closers = set(")]}")

    in_string = False
    string_char = None
    i = 0

    while i < len(cql):
        ch = cql[i]

        # Handle escape sequences inside strings
        if in_string and ch == "\\":
            i += 2
            continue

        # Handle string boundaries
        if ch in ('"', "'") and not in_string:
            in_string = True
            string_char = ch
            i += 1
            continue
        elif in_string and ch == string_char:
            in_string = False
            string_char = None
            i += 1
            continue

        if in_string:
            i += 1
            continue

        # Handle regex literals (skip them)
        if ch == "/" and not in_string:
            # Look ahead to see if this is a regex
            j = i + 1
            while j < len(cql) and cql[j] != "/" and cql[j] != "\n":
                if cql[j] == "\\":
                    j += 1
                j += 1
            if j < len(cql) and cql[j] == "/":
                i = j + 1
                continue

        if ch in openers:
            stack.append((ch, i))
        elif ch in closers:
            if not stack:
                errors.append(
                    f"Unmatched closing '{ch}' at position {i}"
                )
            elif stack[-1][0] != matching[ch]:
                errors.append(
                    f"Mismatched delimiter: expected closing for "
                    f"'{stack[-1][0]}' at position {stack[-1][1]}, "
                    f"got '{ch}' at position {i}"
                )
                stack.pop()
            else:
                stack.pop()

        i += 1

    for opener, pos in stack:
        closer = {"(": ")", "[": "]", "{": "}"}[opener]
        errors.append(f"Unclosed '{opener}' at position {pos}, expected '{closer}'")


def _check_string_literals(cql: str, errors: list[str]) -> None:
    """Check for unclosed string literals."""
    i = 0
    while i < len(cql):
        ch = cql[i]
        if ch in ('"', "'"):
            quote_char = ch
            start_pos = i
            i += 1
            found_close = False
            while i < len(cql):
                if cql[i] == "\\":
                    i += 2
                    continue
                if cql[i] == quote_char:
                    found_close = True
                    i += 1
                    break
                i += 1
            if not found_close:
                errors.append(
                    f"Unclosed string literal starting at position {start_pos}"
                )
        else:
            i += 1


def _check_pipe_structure(cql: str, errors: list[str]) -> None:
    """Check valid pipe structure: no empty stages, proper pipe usage."""
    # Remove strings to avoid false pipe detection inside quoted content
    cleaned = _remove_strings(cql)

    # Split by pipe, but not inside parentheses
    stages = _split_by_pipe(cleaned)

    if len(stages) == 0:
        return

    # Check for empty stages (consecutive pipes or leading/trailing pipe)
    stripped = cql.strip()
    if stripped.startswith("|"):
        errors.append("Query starts with a pipe operator")
    if stripped.endswith("|"):
        errors.append("Query ends with a pipe operator")

    for i, stage in enumerate(stages):
        if not stage.strip():
            if i > 0 and i < len(stages) - 1:
                errors.append(f"Empty pipeline stage at position {i}")


def _check_function_names(cql: str, errors: list[str]) -> None:
    """Check that function calls use known CQL function names."""
    # Remove strings to avoid matching inside quoted content
    cleaned = _remove_strings(cql)

    # Find function calls: identifier followed by (
    func_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

    for match in func_pattern.finditer(cleaned):
        func_name = match.group(1)
        if func_name not in KNOWN_FUNCTIONS:
            errors.append(f"Unknown function: '{func_name}'")


def _remove_strings(cql: str) -> str:
    """Remove string literals from CQL, replacing with placeholders."""
    result = []
    i = 0
    while i < len(cql):
        if cql[i] in ('"', "'"):
            quote = cql[i]
            i += 1
            while i < len(cql):
                if cql[i] == "\\":
                    i += 2
                    continue
                if cql[i] == quote:
                    i += 1
                    break
                i += 1
            result.append("_STR_")
        else:
            result.append(cql[i])
            i += 1
    return "".join(result)


def _split_by_pipe(cql: str) -> list[str]:
    """Split CQL by pipe operators, respecting parentheses nesting."""
    stages = []
    current = []
    depth = 0

    for ch in cql:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "|" and depth == 0:
            stages.append("".join(current))
            current = []
        else:
            current.append(ch)

    stages.append("".join(current))
    return stages
