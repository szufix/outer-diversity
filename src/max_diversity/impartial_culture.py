import numpy as np

from src.diversity.growing_balls import outer_diversity_growing_balls
from src.max_diversity.utils import compute_total_cost
from src.diversity.sampling import (
    sample_impartial_culture,
    sample_diverse_votes,
    outer_diversity_sampling,
)


def diversity_for_ic(vote_graph, domain_size):
    n = len(vote_graph.nodes)

    random_facilities = np.random.choice(range(n), size=domain_size, replace=False).tolist()

    total_cost = compute_total_cost(vote_graph, random_facilities)

    return random_facilities, total_cost


def diversity_for_smpl_ic(num_candidates, domain_size, num_samples=None):

    votes = sample_impartial_culture(num_candidates, domain_size)

    total_distance, _ = outer_diversity_sampling(votes, num_samples)

    return votes, total_distance


def diversity_for_smpl_holy_ic(num_candidates, domain_size, threshold, num_samples=None):

    votes = sample_diverse_votes(num_candidates, domain_size, threshold)
    print(f"Generated {len(votes)} votes with threshold {threshold}")
    # total_distance, _ = outer_diversity_sampling(votes, num_samples)
    total_distance = outer_diversity_growing_balls(votes, num_candidates)

    return votes, total_distance