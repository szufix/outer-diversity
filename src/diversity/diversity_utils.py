import math
import numpy as np
import itertools

from src.domain.extenders import (
    extend_by_swaps
)


def normalization(m):
    return math.factorial(m) // 2 * math.comb(m, 2)


def get_max_num_swaps(m):
    return int(m*(m-1)/2 + 1)


def compute_balls_increase(balls):
    """Helper function to compute balls increase from balls data."""
    balls_increase = [balls[i] - balls[i - 1] for i in range(1, len(balls))]
    return balls_increase


def compute_domain_balls(votes):
    balls = []
    balls.append(len(votes))

    max_num_swaps = get_max_num_swaps(len(votes[0]))

    for _ in range(1, max_num_swaps + 1):
        if len(balls) >= 2 and balls[-1] == balls[-2]:  # no increase in balls
            balls.append(balls[-1])
        else:
            votes = extend_by_swaps(votes, 1)
            balls.append(len(votes))

    return balls


def outer_diversity_from_balls_increase(balls_increase, num_candidates):
    total_diversity = 0

    for i in range(len(balls_increase)-1):
        total_diversity += (i+1) * balls_increase[i]

    total_diversity /= normalization(num_candidates)
    total_diversity = 1 - total_diversity

    return total_diversity


def swap_distance_between_potes(pote_1: list, pote_2: list) -> int:
    """
    Computes the swap distance between two potes (i.e. positional votes).

    Parameters
    ----------
        pote_1 : list
            First vector.
        pote_2 : list
            Second vector.
    Returns
    -------
        int
            Swap distance.
    """
    swap_distance = 0
    for i, j in itertools.combinations(pote_1, 2):
        if (pote_1[i] > pote_1[j] and pote_2[i] < pote_2[j]) or (
            pote_1[i] < pote_1[j] and pote_2[i] > pote_2[j]
        ):
            swap_distance += 1
    return swap_distance


def spearman_distance_between_votes(v1, v2):
    """
    Spearman footrule distance between two votes (permutations).
    Each vote is a sequence/tuple of candidate identifiers (same set and length).
    Returns the integer sum of absolute differences of candidate positions.
    """
    if len(v1) != len(v2):
        raise ValueError("Votes must have the same length")
    pos1 = {c: i for i, c in enumerate(v1)}
    pos2 = {c: i for i, c in enumerate(v2)}
    if set(pos1.keys()) != set(pos2.keys()):
        raise ValueError("Votes must contain the same candidates")
    return sum(abs(pos1[c] - pos2[c]) for c in pos1)


def votes_to_potes(votes):
    return np.array([[list(vote).index(i) for i, _ in enumerate(vote)]
              for vote in votes])


def vote_to_pote(vote):
    return np.array([list(vote).index(i) for i, _ in enumerate(vote)])