

def _outer_diversity_brute_force(votes):
    pass


def _outer_diversity_growing_balls(votes):
    pass


def _outer_diversity_sampling(votes):
    pass


OUTER_DIVERSITY_METHODS = {
    "brute_force": _outer_diversity_brute_force,
    "growing_balls": _outer_diversity_growing_balls,
    "sampling": _outer_diversity_sampling
}


def compute_outer_diversity(votes, method):
    return OUTER_DIVERSITY_METHODS[method](votes)





