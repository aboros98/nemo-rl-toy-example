"""Tests for CQL validator — at least 10 cases covering all validation checks."""

import pytest

from utils.cql_validator import validate, KNOWN_FUNCTIONS


class TestValidQueries:
    def test_simple_filter(self):
        result = validate('#event_simpleName=ProcessRollup2')
        assert result.is_valid
        assert result.errors == []

    def test_pipeline_query(self):
        result = validate(
            '#event_simpleName=ProcessRollup2 '
            '| groupBy(ComputerName, function=count()) '
            '| sort(_count, order=desc) '
            '| head(10)'
        )
        assert result.is_valid
        assert result.errors == []

    def test_query_with_strings(self):
        result = validate(
            '#event_simpleName=ProcessRollup2 '
            'ImageFileName="cmd.exe" '
            '| count()'
        )
        assert result.is_valid

    def test_query_with_eval(self):
        result = validate(
            '* | eval(duration=end-start) | avg(duration)'
        )
        assert result.is_valid

    def test_query_with_nested_parens(self):
        result = validate(
            '#type=ProcessRollup2 '
            '| groupBy(ImageFileName, function=count(aid)) '
            '| sort(_count, order=desc)'
        )
        assert result.is_valid


class TestInvalidQueries:
    def test_empty_query(self):
        result = validate("")
        assert not result.is_valid
        assert any("Empty query" in e for e in result.errors)

    def test_whitespace_only(self):
        result = validate("   ")
        assert not result.is_valid

    def test_unbalanced_parens(self):
        result = validate("count(field")
        assert not result.is_valid
        assert any("Unclosed" in e for e in result.errors)

    def test_unbalanced_brackets(self):
        result = validate("field[0")
        assert not result.is_valid
        assert any("Unclosed" in e for e in result.errors)

    def test_mismatched_delimiters(self):
        result = validate("count(field]")
        assert not result.is_valid

    def test_unclosed_string(self):
        result = validate('ImageFileName="cmd.exe')
        assert not result.is_valid
        assert any("Unclosed string" in e for e in result.errors)

    def test_leading_pipe(self):
        result = validate("| count()")
        assert not result.is_valid
        assert any("starts with a pipe" in e for e in result.errors)

    def test_trailing_pipe(self):
        result = validate("#type=ProcessRollup2 |")
        assert not result.is_valid
        assert any("ends with a pipe" in e for e in result.errors)

    def test_unknown_function(self):
        result = validate("#type=X | fakeFunction(field)")
        assert not result.is_valid
        assert any("Unknown function" in e for e in result.errors)
        assert any("fakeFunction" in e for e in result.errors)

    def test_multiple_errors(self):
        result = validate('| unknownFunc(field"unclosed')
        assert not result.is_valid
        assert len(result.errors) >= 2


class TestEdgeCases:
    def test_escaped_quotes_in_string(self):
        result = validate(r'search("hello \"world\"") | count()')
        assert result.is_valid

    def test_single_quoted_string(self):
        result = validate("search('hello') | count()")
        assert result.is_valid

    def test_known_functions_comprehensive(self):
        """Verify key functions are in the whitelist."""
        expected = [
            "groupBy", "count", "avg", "sum", "min", "max",
            "sort", "head", "tail", "table", "eval", "search",
            "filter", "where", "rename", "replace", "regex",
            "timeChart", "top", "stats", "join",
        ]
        for func in expected:
            assert func in KNOWN_FUNCTIONS, f"{func} not in whitelist"

    def test_pipe_inside_string_not_split(self):
        """Pipes inside strings should not be treated as pipeline operators."""
        result = validate('search("hello | world") | count()')
        assert result.is_valid
