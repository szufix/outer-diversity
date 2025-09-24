import itertools
import math
import random

from src.diversity.diversity_utils import *
from src.diversity.single_peaked import (
    distance_vote_single_peaked_domain,
    distance_vote_single_peaked_on_circle_domain
)
from src.diversity.single_crossing import distance_vote_single_crossing_domain

from src.domain.domain import SC_Domain

def kth_permutation(elements, k):
    """Return the k-th permutation (0-indexed) of the given list using factorial number system."""
    elems = list(elements)
    n = len(elems)
    perm = []

    # Precompute factorials to avoid repeated calculations
    factorials = [1]
    for i in range(1, n):
        factorials.append(factorials[-1] * i)

    for i in range(n):
        if i == n - 1:
            perm.append(elems[0])
        else:
            fact = factorials[n - 1 - i]
            idx = k // fact
            k %= fact
            perm.append(elems.pop(idx))

    return tuple(perm)


def spread_permutations(num_elements: int, num_samples: int):
    """
    Deterministically sample 'num_samples' permutations uniformly
    spread across the lexicographic ordering of permutations.
    Optimized version that avoids repeated factorial calculations.
    """
    if num_samples <= 0:
        return []

    total = math.factorial(num_elements)
    if num_samples >= total:
        # Return all permutations if samples >= total
        return list(itertools.permutations(range(num_elements)))

    # Use numpy for faster arithmetic if available
    try:
        import numpy as np
        indices = np.linspace(0, total - 1, num_samples, dtype=int)
        return [kth_permutation(range(num_elements), int(idx)) for idx in indices]
    except ImportError:
        # Fallback to regular Python
        step = (total - 1) / (num_samples - 1) if num_samples > 1 else 0
        return [kth_permutation(range(num_elements), int(i * step)) for i in range(num_samples)]


def sample_impartial_culture(num_candidates: int, num_samples: int):
    return [tuple(random.sample(range(num_candidates), num_candidates)) for _ in range(num_samples)]


def outer_diversity_sampling(
        domain,
        num_samples: int = 100,
):
    """
    Compute outer diversity by sampling from Impartial Culture and measuring
    swap distances to the closest vote in the domain.

    Args:
        domain: List of votes (tuples) representing the domain
        num_samples: Number of random votes to sample from Impartial Culture

    Returns:
        Sum of distances from each sampled vote to its closest domain vote
    """
    if not domain:
        raise ValueError("Domain cannot be empty")

    # Determine number of candidates from the first vote in domain
    num_candidates = len(domain[0])

    # Validate that all domain votes have the same length
    if not all(len(vote) == num_candidates for vote in domain):
        raise ValueError("All votes in domain must have the same length")

    # Check if num_samples is larger than num_candidates factorial
    if num_candidates < 10 and num_samples >= math.factorial(num_candidates):
        # Use all permutations if samples >= total
        print("ALL", num_candidates)
        sampled_votes = list(itertools.permutations(range(num_candidates)))
    else:
        # sampled_votes = spread_permutations(num_candidates, num_samples)
        sampled_votes = sample_impartial_culture(num_candidates, num_samples)


    sampled_potes = votes_to_potes(sampled_votes)

    domain_potes = votes_to_potes(domain)

    distances = []
    for sampled_pote in sampled_potes:
        # Find minimum distance to any vote in the domain

        tmp_distances = []
        for domain_pote in domain_potes:
            distance = swap_distance_between_potes(sampled_pote, domain_pote)
            tmp_distances.append(distance)
        distances.append(int(min(tmp_distances)))

    # Simplified normalization to avoid large factorials
    total_distance = sum(distances) / len(distances) * 2 / math.comb(num_candidates, 2)
    total_distance = 1 - total_distance

    return total_distance, len(sampled_votes)


def outer_diversity_sampling_for_structered_domains(
        domain_name: str,
        domain,
        num_candidates: int,
        num_samples: int,
):

    # Check if num_samples is larger than num_candidates factorial
    if num_candidates < 10 and num_samples >= math.factorial(num_candidates):
        # Use all permutations if samples >= total
        print("ALL", num_candidates)
        sampled_votes = list(itertools.permutations(range(num_candidates)))
    else:
        # sampled_votes = spread_permutations(num_candidates, num_samples)
        sampled_votes = sample_impartial_culture(num_candidates, num_samples)

    distances = []
    if domain_name == 'sc':
        # Precompute domain object for single-crossing distance
        domain_obj = SC_Domain(domain)

    for vote in sampled_votes:

        if domain_name == 'sp':
            distance = distance_vote_single_peaked_domain(vote)
        elif domain_name == 'spoc':
            distance = distance_vote_single_peaked_on_circle_domain(vote)
        elif domain_name == 'sc':
            distance = distance_vote_single_crossing_domain(vote, domain, domain_obj)
        else:
            raise ValueError(f"Unknown domain name: {domain_name}")

        distances.append(distance)

    # Simplified normalization to avoid large factorials
    total_distance = sum(distances) / len(distances) * 2 / math.comb(num_candidates, 2)
    total_distance = 1 - total_distance

    return total_distance, len(sampled_votes)