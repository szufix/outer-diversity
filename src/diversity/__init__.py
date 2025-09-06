import math

from src.diversity.growing_balls import outer_diversity_growing_balls
from src.diversity.sampling import outer_diversity_sampling


def _outer_diversity_brute_force(domain):
    pass



OUTER_DIVERSITY_METHODS = {
    "brute_force": _outer_diversity_brute_force,
    "growing_balls": outer_diversity_growing_balls,
    "sampling": outer_diversity_sampling
}


def compute_outer_diversity(domain, method, **kwargs):
    return OUTER_DIVERSITY_METHODS[method](domain, **kwargs)





