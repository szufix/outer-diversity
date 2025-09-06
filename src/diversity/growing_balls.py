from src.diversity.diversity_utils import (
    compute_domain_balls,
    compute_balls_increase,
    outer_diversity_from_balls_increase
)






def outer_diversity_growing_balls(domain, **kwargs):
    """ Computes the outer diversity of a given set of votes using the growing balls method. """
    balls = compute_domain_balls(domain)
    balls_increase = compute_balls_increase(balls)
    outer_diversity = outer_diversity_from_balls_increase(balls_increase, len(votes[0]))

    return outer_diversity







