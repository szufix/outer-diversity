from src.domain.validator import validate_domain
from src.domain.extenders import extend_by_swaps

from src.diversity.diversity_utils import (
    swap_distance_between_potes,
    vote_to_pote
)

def compute_popularity(domain_votes, lc_votes):
    D_potes = [vote_to_pote(v) for v in domain_votes]
    LC_potes = [vote_to_pote(list(v)) for v in lc_votes]
    popularity = [0 for _ in domain_votes]
    for lc_pote in LC_potes:
        min_dist = None
        min_idxs = []
        for i, d_pote in enumerate(D_potes):
            dist = swap_distance_between_potes(lc_pote, d_pote)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_idxs = [i]
            elif dist == min_dist:
                min_idxs.append(i)
        for idx in min_idxs:
            popularity[idx] += 1. / len(min_idxs)
    return popularity


@validate_domain
def ball_domain(num_candidates, radius):
    """
    Build a swap-ball around the identity ranking (0,1,...,num_candidates-1).
    Start from the center ranking and iteratively add next layers using
    adjacent-swap expansions (via `extend_by_swaps`) until `radius` layers
    have been added.

    Returns a sequence (list) of unique rankings (each a list or tuple).
    """
    if num_candidates <= 0:
        return []
    if radius < 0:
        raise ValueError("radius must be non-negative")

    center = list(range(num_candidates))
    # start from the center ranking
    domain = [center]
    # extend by `radius` layers (each call adds neighbors at swap distance 1)
    if radius == 0:
        return domain

    return extend_by_swaps(domain, radius)
