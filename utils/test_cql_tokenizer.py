"""Tests for CQL tokenizer."""

import pytest

from utils.cql_tokenizer import (
    bigram_similarity,
    extract_function_names,
    structural_similarity,
    tokenize,
    tokenize_typed,
)


class TestTokenize:
    def test_simple_filter(self):
        tokens = tokenize('#type=ProcessRollup2 ImageFileName="cmd.exe"')
        assert "#type" in tokens
        assert "=" in tokens
        assert "ProcessRollup2" in tokens
        assert '"cmd.exe"' in tokens

    def test_pipe_operator(self):
        tokens = tokenize("* | count()")
        assert "|" in tokens
        assert "*" in tokens

    def test_quoted_string_preserved(self):
        tokens = tokenize('search("hello world")')
        assert '"hello world"' in tokens

    def test_tag_values(self):
        tokens = tokenize("#event_simpleName=ProcessRollup2")
        assert "#event_simpleName" in tokens

    def test_logical_operators(self):
        tokens = tokenize("foo && bar || baz")
        assert "&&" in tokens
        assert "||" in tokens

    def test_comparison_operators(self):
        tokens = tokenize("count > 10 status != 200")
        assert ">" in tokens
        assert "!=" in tokens

    def test_complex_query(self):
        cql = (
            '#event_simpleName=ProcessRollup2 '
            '| ImageFileName=/\\\\cmd\\.exe$/i '
            '| groupBy(ComputerName, function=count()) '
            '| sort(_count, order=desc) '
            '| head(10)'
        )
        tokens = tokenize(cql)
        assert "#event_simpleName" in tokens
        assert "|" in tokens
        # Should have multiple pipes
        assert tokens.count("|") == 4

    def test_empty_string(self):
        assert tokenize("") == []

    def test_assignment_operator(self):
        tokens = tokenize("newField := oldField + 1")
        assert ":=" in tokens


class TestTokenizeTyped:
    def test_types_correct(self):
        typed = tokenize_typed('#tag=value | count()')
        types = [t for _, t in typed]
        assert "TAG" in types
        assert "PIPE" in types

    def test_string_type(self):
        typed = tokenize_typed('"hello"')
        assert typed[0][1] == "STRING"

    def test_number_type(self):
        typed = tokenize_typed("42")
        assert typed[0][1] == "NUMBER"


class TestExtractFunctionNames:
    def test_simple_pipeline(self):
        funcs = extract_function_names(
            "#type=ProcessRollup2 | groupBy(field) | count() | sort(order=desc)"
        )
        # First stage is a filter (no function), then groupBy, count, sort
        assert "groupBy" in funcs
        assert "count" in funcs
        assert "sort" in funcs

    def test_no_functions(self):
        funcs = extract_function_names("#type=ProcessRollup2")
        # No pipe-separated function calls
        assert funcs == [] or all(f != "" for f in funcs)

    def test_head_tail(self):
        funcs = extract_function_names("* | head(10) | tail(5)")
        assert "head" in funcs
        assert "tail" in funcs


class TestBigramSimilarity:
    def test_identical_queries(self):
        cql = '#type=ProcessRollup2 | groupBy(field) | count()'
        assert bigram_similarity(cql, cql) == 1.0

    def test_completely_different(self):
        a = "alpha beta gamma"
        b = "x y z w"
        sim = bigram_similarity(a, b)
        assert sim == 0.0

    def test_partial_overlap(self):
        a = '#type=ProcessRollup2 | count()'
        b = '#type=ProcessRollup2 | sum(field)'
        sim = bigram_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        assert bigram_similarity("", "") == 1.0

    def test_single_token(self):
        assert bigram_similarity("hello", "hello") == 1.0
        assert bigram_similarity("hello", "world") == 0.0

    def test_symmetry(self):
        a = "#type=ProcessRollup2 | count()"
        b = "#type=DnsRequest | count()"
        assert bigram_similarity(a, b) == bigram_similarity(b, a)


class TestStructuralSimilarity:
    def test_identical_structure(self):
        a = "#type=X | groupBy(f1) | count() | sort(order=desc)"
        b = "#type=Y | groupBy(f2) | count() | sort(order=asc)"
        assert structural_similarity(a, b) == 1.0

    def test_different_structure(self):
        a = "* | groupBy(f1) | count()"
        b = "* | table(f1, f2, f3)"
        sim = structural_similarity(a, b)
        assert sim < 1.0

    def test_empty_queries(self):
        assert structural_similarity("", "") == 1.0

    def test_subset_structure(self):
        a = "* | groupBy(f) | count()"
        b = "* | groupBy(f) | count() | sort(order=desc)"
        sim = structural_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_symmetry(self):
        a = "* | groupBy(f) | count()"
        b = "* | count() | sort()"
        assert structural_similarity(a, b) == structural_similarity(b, a)
