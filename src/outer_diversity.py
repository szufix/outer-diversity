

def _outer_diversity_brute_force(election):
    pass


def _outer_diversity_growing_balls(election):
    pass


def _outer_diversity_sampling(election):
    pass


OUTER_DIVERSITY_METHODS = {
    "brute_force": _outer_diversity_brute_force,
    "growing_balls": _outer_diversity_growing_balls,
    "sampling": _outer_diversity_sampling
}


def compute_outer_diversity(election, method):
    return OUTER_DIVERSITY_METHODS[method](election)





