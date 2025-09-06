import math
import numpy as np

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


def swap_distance_between_potes(pote_1: list, pote_2: list, m : int) -> int:
    """ Return: Swap distance between two potes """
    swap_distance = 0
    for a in range(m):
        for b in range(m):
            if (pote_1[a] < pote_1[b] and pote_2[a] >= pote_2[b]):
                swap_distance += 0.5
            if (pote_1[a] <= pote_1[b] and pote_2[a] > pote_2[b]):
                swap_distance += 0.5
    return swap_distance


def votes_to_potes(votes):
    return np.array([[list(vote).index(i) for i, _ in enumerate(vote)]
              for vote in votes])


def vote_to_pote(vote):
    return np.array([list(vote).index(i) for i, _ in enumerate(vote)])