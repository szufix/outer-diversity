import pytest

from src.domain.condorcet import largest_condorcet_domain, is_condorcet_domain


def test_condorcet_domains_are_valid():
    """For sizes m=2..8, the largest_condorcet_domain should be a Condorcet domain."""
    for m in range(2, 8+1):
        dom = largest_condorcet_domain(m)
        assert isinstance(dom, list)
        # domain votes should be lists of length m
        for vote in dom:
            assert len(vote) == m

        is_cd, witness = is_condorcet_domain(dom, return_witness=True)
        assert is_cd is True
        assert witness is None


def test_empty_domain_raises():
    with pytest.raises(ValueError):
        is_condorcet_domain([])


def test_invalid_domain_detected():
    # create a small domain that generates a Condorcet cycle on {0,1,2}
    dom = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    is_cd, witness = is_condorcet_domain(dom, return_witness=True)
    assert is_cd is False
    assert isinstance(witness, dict)
    assert 'triple' in witness and 'pattern_codes' in witness and 'orders' in witness
    assert len(witness['pattern_codes']) == 3

