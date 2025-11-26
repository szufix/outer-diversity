import pytest

from src.domain.fishburn import largest_fishburn_domain, is_fishburn_domain


def test_fishburn_domains_are_valid():
    """For several sizes m, generated Fishburn domains should satisfy the Fishburn rule."""
    for m in range(1, 8+1):
        dom = largest_fishburn_domain(m)
        assert isinstance(dom, list)
        # domain votes should be lists of length m
        for vote in dom:
            assert len(vote) == m
        assert is_fishburn_domain(dom)


def test_empty_domain_is_valid():
    assert is_fishburn_domain([]) is True


def test_invalid_domain_detected():
    # create a small domain that violates the rule for m=3: alternatives 0,1,2
    # For j=1 (odd) it must never be top; create a vote where 1 is top -> invalid
    bad_vote = [1, 0, 2]
    assert is_fishburn_domain([bad_vote]) is False

