from src.domain.validator import validate_domain
import numpy as np


@validate_domain
def extend_to_reversed(domain):
    """
    Extends a domain to include the reverse of each ranking, ensuring uniqueness.
    Accepts rankings as lists or tuples.
    """
    as_tuples = [tuple(r) for r in domain]  # ensure all are tuples
    extended = set(as_tuples) | {tuple(reversed(r)) for r in as_tuples}
    extended = np.unique(list(extended), axis=0)
    return extended



def _get_neighbors(ranking):
    """
    Get all neighbors of a ranking by swapping adjacent elements.
    Returns a set of tuples representing the neighbors.
    """
    neighbors = set()
    for i in range(len(ranking) - 1):
        new_ranking = list(ranking)
        new_ranking[i], new_ranking[i + 1] = new_ranking[i + 1], new_ranking[i]
        neighbors.add(tuple(new_ranking))
    return neighbors


@validate_domain
def extend_by_swaps(domain, num_swaps):
    if num_swaps <= 0:
        return domain

    extended = set([tuple(r) for r in domain])  # ensure all are tuples
    for ranking in domain:
        neighbors = _get_neighbors(ranking)
        for neighbor in neighbors:
            extended.add(neighbor)

    return extend_by_swaps(list(extended), num_swaps-1)


